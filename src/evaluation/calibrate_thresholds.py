"""
Compute Anomaly Thresholds — Development Data Only (DCASE 2024 Task 2)

This script computes per-machine anomaly thresholds using ONLY the
training/validation data (development set). It NEVER touches the test set,
following the strict experimental protocol outlined in the DCASE challenge.

Method:
  For each machine, we compute anomaly scores on the training set
  (all normal sounds) using the machine's best scoring strategy.
  The threshold is set at:
      threshold = percentile_95(normal_scores)
  This means ~5% of normal training sounds would be flagged, which
  is a standard contamination-aware calibration.

Output:
  configs/thresholds.json -- contains per-machine config with:
    - best scoring strategy
    - threshold value
    - pipeline (V1 or V2)
    - score statistics (mean, std, min, max, percentiles)
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, json, joblib
from sklearn.neighbors import NearestNeighbors

# ==========================================================================
# CONFIG
# ==========================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "weights")
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
V1_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed_v1")
V2_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed_v2")

PATCH_WIDTH  = 64
PATCH_STRIDE = 32
SAMPLE_RATE  = 16000
N_MELS       = 128
HOP_LENGTH   = 512
N_FFT        = 2048
BATCH_SIZE   = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================================================
# Best strategy per machine (from eval_best.py results)
# ==========================================================================
BEST_CONFIG = {
    "ToyCar":   {"pipeline": "V1", "strategy": "Recon_Max"},
    "ToyTrain": {"pipeline": "V2", "strategy": "Recon_Max"},
    "bearing":  {"pipeline": "V2", "strategy": "KNN_Mean"},
    "fan":      {"pipeline": "V1", "strategy": "Recon_Mean"},
    "gearbox":  {"pipeline": "V2", "strategy": "Recon_Mean"},
    "slider":   {"pipeline": "V2", "strategy": "Recon_Mean"},
    "valve":    {"pipeline": "V1", "strategy": "Neg_Recon"},
}

# ==========================================================================
# V1 MODEL: 5-layer CNN, Sigmoid output
# ==========================================================================
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._enc_block(1,   16, 5, 2, 2)
        self.enc2 = self._enc_block(16,  32, 5, 2, 2)
        self.enc3 = self._enc_block(32,  64, 3, 2, 1)
        self.enc4 = self._enc_block(64, 128, 3, 2, 1)
        self.enc5 = self._enc_block(128,128, 3, 2, 1)
        self.fc_enc = nn.Linear(1024, 64)
        self.fc_dec = nn.Linear(64, 1024)
        self.dec5 = self._dec_block(128,128, 3, 2, 1, 1)
        self.dec4 = self._dec_block(128, 64, 3, 2, 1, 1)
        self.dec3 = self._dec_block(64,  32, 3, 2, 1, 1)
        self.dec2 = self._dec_block(32,  16, 5, 2, 2, 1)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid())

    def _enc_block(self, cin, cout, k, s, p):
        return nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=p),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2))

    def _dec_block(self, cin, cout, k, s, p, op):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, return_embedding=False):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        e4 = self.enc4(e3); e5 = self.enc5(e4)
        flat = e5.view(e5.size(0), -1)
        encoded = self.fc_enc(flat)
        if return_embedding: return encoded
        d_in = self.fc_dec(encoded).view(-1, 128, 4, 2)
        return self.dec1(self.dec2(self.dec3(self.dec4(self.dec5(d_in)))))

# ==========================================================================
# V2 MODEL: 3-layer CNN, Linear output
# ==========================================================================
class CNNAutoencoderV2(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.flatten_size = 128 * 16 * 8
        self.fc_enc = nn.Linear(self.flatten_size, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, self.flatten_size)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.dec1 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, return_embedding=False):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        flat = e3.view(e3.size(0), -1)
        embedding = self.fc_enc(flat)
        if return_embedding: return embedding
        d_in = self.fc_dec(embedding).view(-1, 128, 16, 8)
        return self.dec1(self.dec2(self.dec3(d_in)))

# ==========================================================================
# HELPERS
# ==========================================================================
V1_MACHINES = {"ToyCar", "fan", "valve"}

def extract_patches(spectrogram, width=PATCH_WIDTH, stride=PATCH_STRIDE):
    _, W = spectrogram.shape
    patches = []
    for start in range(0, W - width + 1, stride):
        patches.append(spectrogram[:, start:start+width])
    if patches and (W - width) % stride != 0:
        patches.append(spectrogram[:, W-width:W])
    if not patches and W >= width:
        patches.append(spectrogram[:, :width])
    return np.array(patches)

def extract_all_features(model, patches):
    model.eval()
    x = torch.FloatTensor(patches).unsqueeze(1)
    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=False)
    all_mses, all_embs = [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = torch.mean((batch - recon)**2, dim=(1,2,3)).cpu().numpy()
            all_mses.append(mse)
            emb = model(batch, return_embedding=True).cpu().numpy()
            all_embs.append(emb)
    return np.concatenate(all_mses), np.concatenate(all_embs)

def compute_file_score(file_features, strategy):
    """Compute a single anomaly score for one file given a strategy name."""
    mses = file_features['mses']
    knn_dists = file_features.get('knn_dists', None)

    if strategy == 'Recon_Mean':
        return float(np.mean(mses))
    elif strategy == 'Recon_Max':
        return float(np.max(mses))
    elif strategy == 'Recon_P90':
        return float(np.percentile(mses, 90))
    elif strategy == 'KNN_Mean':
        return float(np.mean(knn_dists))
    elif strategy == 'KNN_Max':
        return float(np.max(knn_dists))
    elif strategy == 'Neg_Recon':
        return float(-np.mean(mses))
    elif strategy == 'Neg_Recon_Max':
        return float(-np.max(mses))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ==========================================================================
# MAIN: Compute thresholds on development data
# ==========================================================================
def main():
    machine_types = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
    thresholds = {}

    print(f"\n{'='*70}")
    print(f"COMPUTING THRESHOLDS (development data only)")
    print(f"{'='*70}")

    for machine in machine_types:
        config = BEST_CONFIG[machine]
        is_v1 = config['pipeline'] == 'V1'
        strategy = config['strategy']
        processed_dir = V1_PROCESSED if is_v1 else V2_PROCESSED

        print(f"\n{'='*50}")
        print(f"{machine} [{config['pipeline']}] — Strategy: {strategy}")
        print(f"{'='*50}")

        # Load metadata and model
        meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                          map_location=device, weights_only=False)
        n_frames = meta['n_frames']

        if is_v1:
            model = CNNAutoencoder().to(device)
            ref_value = 1.0
        else:
            bn_dim = meta.get('bottleneck_dim', 128)
            model = CNNAutoencoderV2(bottleneck_dim=bn_dim).to(device)
            ref_value = 'np.max'

        model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, machine, "best_model.pth"),
            map_location=device, weights_only=False))
        model.eval()

        scaler = joblib.load(os.path.join(processed_dir, machine, "scaler.save"))

        # Load training data and compute scores
        X_train = np.load(os.path.join(processed_dir, machine, "X_train.npy"))
        print(f"  Loaded training data: {X_train.shape}")

        # Extract patches from all training spectrograms
        all_file_features = []
        knn = None

        # If strategy needs KNN, fit it first
        if 'KNN' in strategy:
            print(f"  Fitting KNN on training embeddings...")
            train_patches = np.concatenate([extract_patches(s) for s in X_train])
            _, train_embs = extract_all_features(model, train_patches)
            knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
            knn.fit(train_embs)
            print(f"  KNN fitted on {len(train_embs)} patches")

        # Compute per-file scores on training data
        print(f"  Computing per-file scores on {len(X_train)} training files...")
        file_scores = []
        for i, spec in enumerate(X_train):
            patches = extract_patches(spec)
            if len(patches) == 0:
                continue
            mses, embs = extract_all_features(model, patches)
            features = {'mses': mses}
            if knn is not None:
                features['knn_dists'] = np.mean(knn.kneighbors(embs)[0], axis=1)
            score = compute_file_score(features, strategy)
            file_scores.append(score)

            if (i + 1) % 200 == 0 or (i + 1) == len(X_train):
                print(f"    Scored {i+1}/{len(X_train)} training files...")

        file_scores = np.array(file_scores)

        # Compute threshold statistics
        mean_score = float(np.mean(file_scores))
        std_score = float(np.std(file_scores))
        p90 = float(np.percentile(file_scores, 90))
        p95 = float(np.percentile(file_scores, 95))
        p99 = float(np.percentile(file_scores, 99))

        # Threshold: 95th percentile of normal training scores
        # This means ~5% false positive rate on known normal data
        threshold = p95

        thresholds[machine] = {
            "pipeline": config['pipeline'],
            "strategy": strategy,
            "threshold": threshold,
            "ref_value": str(ref_value),
            "n_frames": n_frames,
            "score_stats": {
                "mean": mean_score,
                "std": std_score,
                "min": float(np.min(file_scores)),
                "max": float(np.max(file_scores)),
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "n_files": len(file_scores),
            }
        }

        print(f"  Score distribution (normal training data):")
        print(f"    Mean ± Std  : {mean_score:.6f} ± {std_score:.6f}")
        print(f"    [Min, Max]  : [{np.min(file_scores):.6f}, {np.max(file_scores):.6f}]")
        print(f"    P90 / P95   : {p90:.6f} / {p95:.6f}")
        print(f"    *** THRESHOLD (P95): {threshold:.6f} ***")

    # Save thresholds
    configs_dir = os.path.join(PROJECT_ROOT, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    out_path = os.path.join(configs_dir, "thresholds.json")
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n{'='*70}")
    print(f"THRESHOLDS COMPUTED — SUMMARY")
    print(f"{'='*70}")
    print(f"{'Machine':<12} | {'Pipeline':>4} | {'Strategy':<12} | {'Threshold':>12} | {'Mean±Std':>18}")
    print("-" * 75)
    for machine in machine_types:
        t = thresholds[machine]
        s = t['score_stats']
        print(f"{machine:<12} | {t['pipeline']:>4} | {t['strategy']:<12} | {t['threshold']:12.6f} | "
              f"{s['mean']:.6f} ± {s['std']:.6f}")
    print("-" * 75)
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
