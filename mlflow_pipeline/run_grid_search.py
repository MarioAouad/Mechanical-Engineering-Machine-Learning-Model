"""
run_grid_search.py — 4D Overnight Hyperparameter Grid Search
=============================================================
Tests every combination of:
  SPECS × SCALERS × LOSSES × SCORING = 2 × 2 × 3 × 2 = 24 runs

Usage:  cd mlflow_pipeline && python run_grid_search.py
View:   mlflow ui → http://localhost:5000
"""

import os, sys, json, glob, copy, gc
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import librosa, joblib, mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use('Agg')

# ===========================================================================
# THE 4D GRID
# ===========================================================================
SPECS   = ['Mel']
SCALERS = ['MinMax']
LOSSES  = ['MSE']
SCORING = ['Reconstruction']

# Fixed parameters (constant across all grid runs)
FIXED = {
    'N_FFT': 1024, 'HOP_LENGTH': 512, 'N_MELS': 128,
    'SAMPLE_RATE': 16000, 'VAL_SPLIT': 0.15, 'RANDOM_STATE': 42,
    'BATCH_SIZE': 512, 'LEARNING_RATE': 1e-3, 'WEIGHT_DECAY': 1e-5,
    'EPOCHS': 200, 'PATIENCE': 20, 'PATCH_WIDTH': 32, 'PATCH_STRIDE': 16,
    'BOTTLENECK_DIM': 16, 'DROPOUT_RATE': 0.15, 'NOISE_STD': 0.01,
    'MIXUP_ALPHA': 0.0, 'IF_CONTAMINATION': 0.05, 'GMM_COMPONENTS': 5,
}

# All paths LOCAL to mlflow_pipeline/
RAW_BASE   = os.path.join(".", "data", "raw")
PROC_BASE  = os.path.join(".", "data", "processed")
MODELS_DIR = os.path.join(".", "experiments", "models")
RUNS_DIR   = os.path.join(".", "experiments", "runs")

# ===========================================================================
# CNN AUTOENCODER
# ===========================================================================
class CNNAutoencoder(nn.Module):
    def __init__(self, input_height, patch_width, bottleneck_dim, dropout_rate):
        super().__init__()
        self.encoder = nn.Sequential(
            self._cb(1, 32, dropout_rate), self._cb(32, 64, dropout_rate),
            self._cb(64, 128, dropout_rate), self._cb(128, 128, dropout_rate))
        h, w = input_height // 16, patch_width // 16
        self.flat_size, self.h_enc, self.w_enc = 128 * h * w, h, w
        self.flatten = nn.Flatten()
        self.bottleneck_encode = nn.Linear(self.flat_size, bottleneck_dim)
        self.bottleneck_decode = nn.Linear(bottleneck_dim, self.flat_size)
        self.decoder = nn.Sequential(
            self._db(128, 128), self._db(128, 64), self._db(64, 32),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1), nn.Sigmoid())
    def _cb(self, ic, oc, dr):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, 1, 1), nn.BatchNorm2d(oc),
                             nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2), nn.Dropout2d(dr))
    def _db(self, ic, oc):
        return nn.Sequential(nn.ConvTranspose2d(ic, oc, 3, 2, 1, output_padding=1),
                             nn.BatchNorm2d(oc), nn.LeakyReLU(0.2))
    def forward(self, x):
        e = self.encoder(x)
        z = self.bottleneck_encode(self.flatten(e))
        d = self.bottleneck_decode(z).view(-1, 128, self.h_enc, self.w_enc)
        return self.decoder(d)
    def encode(self, x):
        return self.bottleneck_encode(self.flatten(self.encoder(x)))

# ===========================================================================
# HELPERS
# ===========================================================================
def extract_patches(data, width, stride):
    patches = []
    for spec in data:
        for s in range(0, spec.shape[1] - width + 1, stride):
            patches.append(spec[:, s:s + width])
    return np.array(patches)

def get_loss_fn(name):
    if name == 'Huber': return nn.SmoothL1Loss()
    elif name == 'L1':  return nn.L1Loss()
    else:               return nn.MSELoss()

def mixup_batch(batch, alpha=0.2):
    # If alpha is 0, skip the math and return the normal batch
    if alpha <= 0.0:
        return batch
  
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch.size(0)).to(batch.device)
    return lam * batch + (1 - lam) * batch[idx]

def compute_pauc(y_true, y_score, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    stop = np.searchsorted(fpr, max_fpr, side='right')
    if stop == 0: return 0.0
    fpr_t, tpr_t = fpr[:stop], tpr[:stop]
    if fpr_t[-1] < max_fpr:
        tpr_t = np.append(tpr_t, np.interp(max_fpr, fpr, tpr))
        fpr_t = np.append(fpr_t, max_fpr)
    return np.trapz(tpr_t, fpr_t) / max_fpr

# ===========================================================================
# PHASE 1: PREPROCESS
# ===========================================================================
def run_prep(spec_type, scaler_type):
    print(f"\n{'#'*70}\n  PREP: Spec={spec_type}, Scaler={scaler_type}\n{'#'*70}")
    machines = sorted([d for d in os.listdir(RAW_BASE)
                       if os.path.isdir(os.path.join(RAW_BASE, d))])
    for machine in machines:
        proc_dir = os.path.join(PROC_BASE, machine)
        os.makedirs(proc_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(RAW_BASE, machine, "train", "*.wav")))
        if not wav_files: continue

        specs = []
        for fp in wav_files:
            y, sr = librosa.load(fp, sr=FIXED['SAMPLE_RATE'])
            if spec_type == 'Linear':
                stft = np.abs(librosa.stft(y, n_fft=FIXED['N_FFT'], hop_length=FIXED['HOP_LENGTH']))
                specs.append(librosa.amplitude_to_db(stft, ref=np.max))
            else:
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=FIXED['N_FFT'],
                    hop_length=FIXED['HOP_LENGTH'], n_mels=FIXED['N_MELS'])
                specs.append(librosa.power_to_db(mel, ref=np.max))

        X_all = np.array(specs)
        X_train, X_val = train_test_split(
            X_all, test_size=FIXED['VAL_SPLIT'], random_state=FIXED['RANDOM_STATE'])
        N_tr, H, W = X_train.shape
        N_va = X_val.shape[0]
        scaler = StandardScaler() if scaler_type == 'Standard' else MinMaxScaler((0, 1))
        X_tr_s = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(N_tr, H, W)
        X_va_s = scaler.transform(X_val.reshape(-1, 1)).reshape(N_va, H, W)

        np.save(os.path.join(proc_dir, "X_train.npy"), X_tr_s)
        np.save(os.path.join(proc_dir, "X_val.npy"), X_va_s)
        joblib.dump(scaler, os.path.join(proc_dir, "scaler.save"))
        meta = {'SPECTROGRAM_TYPE': spec_type, 'SCALER_TYPE': scaler_type,
                'SPEC_HEIGHT': H, 'SPEC_WIDTH': W, **FIXED}
        with open(os.path.join(proc_dir, "prep_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  {machine}: {X_tr_s.shape}")

# ===========================================================================
# PHASE 2: TRAIN (with dimension crop fix)
# ===========================================================================
def run_train(loss_name, device):
    print(f"\n{'#'*70}\n  TRAIN: Loss={loss_name}\n{'#'*70}")
    machines = sorted([d for d in os.listdir(PROC_BASE)
                       if os.path.isdir(os.path.join(PROC_BASE, d))])
    PW, PS, BS = FIXED['PATCH_WIDTH'], FIXED['PATCH_STRIDE'], FIXED['BATCH_SIZE']
    criterion = get_loss_fn(loss_name)
    os.makedirs(MODELS_DIR, exist_ok=True)

    for machine in machines:
        tr_path = os.path.join(PROC_BASE, machine, "X_train.npy")
        if not os.path.exists(tr_path): continue
        X_tr_full = np.load(tr_path)
        X_va_full = np.load(os.path.join(PROC_BASE, machine, "X_val.npy"))

        X_tr_p = extract_patches(X_tr_full, PW, PS)
        X_va_p = extract_patches(X_va_full, PW, PS)
        X_tr_t = torch.tensor(X_tr_p, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_va_p, dtype=torch.float32).unsqueeze(1)

        # === DIMENSION CROP FIX ===
        H_patch = X_tr_t.shape[2]
        H_crop = (H_patch // 16) * 16
        if H_crop != H_patch:
            X_tr_t = X_tr_t[:, :, :H_crop, :]
            X_va_t = X_va_t[:, :, :H_crop, :]
            print(f"  {machine}: Cropped {H_patch} → {H_crop}")
        input_height = H_crop
        # ==========================

        train_ld = DataLoader(TensorDataset(X_tr_t, X_tr_t), batch_size=BS, shuffle=True)
        val_ld = DataLoader(TensorDataset(X_va_t, X_va_t), batch_size=BS, shuffle=False)
        model = CNNAutoencoder(
            input_height, PW, FIXED['BOTTLENECK_DIM'], FIXED['DROPOUT_RATE']).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FIXED['LEARNING_RATE'], weight_decay=FIXED['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        best_val, patience_ctr, best_state = float('inf'), 0, None
        for epoch in range(1, FIXED['EPOCHS'] + 1):
            model.train()
            t_loss = 0
            for batch, _ in train_ld:
                batch = batch.to(device)
                batch = mixup_batch(batch, FIXED['MIXUP_ALPHA'])
                noisy = (batch + FIXED['NOISE_STD'] * torch.randn_like(batch)).clamp(0, 1)
                optimizer.zero_grad()
                loss = criterion(model(noisy), batch)
                loss.backward(); optimizer.step()
                t_loss += loss.item()
            t_loss /= len(train_ld)

            model.eval()
            v_loss = 0
            with torch.no_grad():
                for batch, _ in val_ld:
                    v_loss += criterion(model(batch.to(device)), batch.to(device)).item()
            v_loss /= len(val_ld)
            scheduler.step(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if epoch % 1 == 0:
                print(f"    {machine} Ep {epoch:>3} | Tr: {t_loss:.6f} | Va: {v_loss:.6f}")
            if patience_ctr >= FIXED['PATIENCE']:
                print(f"    {machine} Early stop at epoch {epoch}"); break

        torch.save(best_state, os.path.join(MODELS_DIR, f"best_model_{machine}.pth"))
        print(f"  {machine}: best_val={best_val:.6f}")
        del model, optimizer, scheduler, train_ld, val_ld
        torch.cuda.empty_cache(); gc.collect()

# ===========================================================================
# PHASE 3: EVALUATE (with dimension crop fix)
# ===========================================================================
def run_eval(scoring_method, spec_type, device):
    print(f"\n{'#'*70}\n  EVAL: Scoring={scoring_method}\n{'#'*70}")
    machines = sorted([d for d in os.listdir(PROC_BASE)
                       if os.path.isdir(os.path.join(PROC_BASE, d))])
    PW, PS = FIXED['PATCH_WIDTH'], FIXED['PATCH_STRIDE']
    pixel_crit = nn.L1Loss(reduction='none')
    metrics = {}

    for machine in machines:
        model_path = os.path.join(MODELS_DIR, f"best_model_{machine}.pth")
        scaler_path = os.path.join(PROC_BASE, machine, "scaler.save")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path): continue

        meta_path = os.path.join(PROC_BASE, machine, "prep_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f: pm = json.load(f)
            input_height = pm['SPEC_HEIGHT']
        else:
            input_height = FIXED['N_MELS']

        # === DIMENSION CROP FIX ===
        model_height = (input_height // 16) * 16
        # ==========================

        state = torch.load(model_path, map_location=device)
        bn_dim = state['bottleneck_encode.weight'].shape[0]
        model = CNNAutoencoder(model_height, PW, bn_dim, FIXED['DROPOUT_RATE']).to(device)
        model.load_state_dict(state); model.eval()
        scaler = joblib.load(scaler_path)

        embedding_scorer = None
        if scoring_method in ('IsolationForest', 'GMM'):
            X_tr = np.load(os.path.join(PROC_BASE, machine, "X_train.npy"))
            X_tr = X_tr[:, :model_height, :]  # Crop fix
            tr_patches = extract_patches(X_tr, PW, PS)
            tr_t = torch.tensor(tr_patches, dtype=torch.float32).unsqueeze(1).to(device)
            embeds = []
            with torch.no_grad():
                for i in range(0, len(tr_t), 256):
                    embeds.append(model.encode(tr_t[i:i+256]).cpu().numpy())
            embeds = np.concatenate(embeds)
            if scoring_method == 'IsolationForest':
                embedding_scorer = IsolationForest(
                    contamination=FIXED['IF_CONTAMINATION'], random_state=42, n_jobs=-1)
            else:
                embedding_scorer = GaussianMixture(
                    n_components=FIXED['GMM_COMPONENTS'], random_state=42)
            embedding_scorer.fit(embeds)

        test_files = sorted(glob.glob(os.path.join(RAW_BASE, machine, "test", "*.wav")))
        if not test_files: continue
        labels, scores = [], []

        for fpath in test_files:
            is_anomaly = 1 if 'anomaly' in os.path.basename(fpath).lower() else 0
            try: y, sr = librosa.load(fpath, sr=FIXED['SAMPLE_RATE'])
            except: continue
            if spec_type == 'Linear':
                stft = np.abs(librosa.stft(y, n_fft=FIXED['N_FFT'], hop_length=FIXED['HOP_LENGTH']))
                spec_db = librosa.amplitude_to_db(stft, ref=np.max)
            else:
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=FIXED['N_FFT'],
                    hop_length=FIXED['HOP_LENGTH'], n_mels=FIXED['N_MELS'])
                spec_db = librosa.power_to_db(mel, ref=np.max)

            H, W = spec_db.shape
            scaled = scaler.transform(spec_db.reshape(-1, 1)).reshape(H, W)
            scaled = scaled[:model_height, :]  # CROP FIX

            fp_list = [scaled[:, s:s+PW] for s in range(0, scaled.shape[1] - PW + 1, PS)]
            if not fp_list: continue
            batch_t = torch.tensor(np.array(fp_list), dtype=torch.float32).unsqueeze(1).to(device)

            with torch.no_grad():
                if scoring_method == 'Reconstruction':
                    out = model(batch_t)
                    errs = pixel_crit(out, batch_t).mean(dim=(1,2,3)).cpu().numpy()
                    file_score = float(np.max(errs))
                else:
                    z = model.encode(batch_t).cpu().numpy()
                    if scoring_method == 'IsolationForest':
                        ps = -embedding_scorer.decision_function(z)
                    else:
                        ps = -embedding_scorer.score_samples(z)
                    file_score = float(np.max(ps))
            labels.append(is_anomaly); scores.append(file_score)

        if len(labels) > 0 and len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            pauc = compute_pauc(labels, scores)
            metrics[machine] = {'auc': auc, 'pauc': pauc}
            print(f"  {machine}: AUC={auc:.4f} pAUC={pauc:.4f}")
        del model; torch.cuda.empty_cache(); gc.collect()
    return metrics

# ===========================================================================
# MAIN — 4D GRID SEARCH LOOP
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    total = len(SPECS) * len(SCALERS) * len(LOSSES) * len(SCORING)
    run_num = 0
    mlflow.set_experiment("GridSearch_4D_AcousticAnomaly")

    for spec_type in SPECS:                    # Dim 1
        for scaler_type in SCALERS:            # Dim 2
            run_prep(spec_type, scaler_type)   # Re-prep when spec/scaler changes
            for loss_name in LOSSES:           # Dim 3
                run_train(loss_name, device)   # Re-train when loss changes
                for scoring_method in SCORING: # Dim 4
                    run_num += 1
                    combo = f"{spec_type}_{scaler_type}_{loss_name}_{scoring_method}"
                    print(f"\n{'='*70}")
                    print(f"  RUN {run_num}/{total}: {combo}")
                    print(f"{'='*70}")

                    eval_metrics = run_eval(scoring_method, spec_type, device)

                    with mlflow.start_run(run_name=combo):
                        mlflow.log_param("spec_type", spec_type)
                        mlflow.log_param("scaler", scaler_type)
                        mlflow.log_param("loss", loss_name)
                        mlflow.log_param("scoring", scoring_method)
                        for k, v in FIXED.items():
                            mlflow.log_param(k, v)
                        auc_vals, pauc_vals = [], []
                        for m, r in eval_metrics.items():
                            mlflow.log_metric(f"auc_{m}", r['auc'])
                            mlflow.log_metric(f"pauc_{m}", r['pauc'])
                            auc_vals.append(r['auc'])
                            pauc_vals.append(r['pauc'])
                        if auc_vals:
                            mlflow.log_metric("avg_auc", np.mean(auc_vals))
                            mlflow.log_metric("avg_pauc", np.mean(pauc_vals))

                    # Summary table
                    print(f"\n{'Machine':<12} | {'AUC':>8} | {'pAUC':>8}")
                    print("-" * 35)
                    for m, r in eval_metrics.items():
                        print(f"{m:<12} | {r['auc']:8.4f} | {r['pauc']:8.4f}")
                    if auc_vals:
                        print(f"{'AVERAGE':<12} | {np.mean(auc_vals):8.4f} | {np.mean(pauc_vals):8.4f}")

                    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'#'*70}")
    print(f"  GRID SEARCH COMPLETE — {total} combinations tested")
    print(f"  Run: mlflow ui → http://localhost:5000")
    print(f"{'#'*70}")

if __name__ == "__main__":
    main()
