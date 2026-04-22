"""
Phase 2 & 3: Train autoencoders and evaluate anomaly detection.
Standalone script version for reliable execution.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
import copy
import time
import glob

# Set working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ACOUSTIC ANOMALY DETECTION — TRAINING & EVALUATION")
print("=" * 70)
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

# ======================================================================
# HYPERPARAMETERS
# ======================================================================
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5
NUM_EPOCHS     = 100
PATIENCE       = 15
BOTTLENECK_DIM = 32

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models")
RESULTS_DIR   = os.path.join("results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nHyperparameters:")
print(f"  Batch Size    : {BATCH_SIZE}")
print(f"  Learning Rate : {LEARNING_RATE}")
print(f"  Weight Decay  : {WEIGHT_DECAY}")
print(f"  Max Epochs    : {NUM_EPOCHS}")
print(f"  Early Stop    : {PATIENCE} epochs patience")
print(f"  Bottleneck    : {BOTTLENECK_DIM} dimensions")
print(f"  Device        : {device}")


# ======================================================================
# MODEL ARCHITECTURE
# ======================================================================
class AcousticAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(original_shape)


# Smoke test
print("\nSmoke test:")
for width in [313, 376]:
    input_dim = 128 * width
    m = AcousticAutoencoder(input_dim, BOTTLENECK_DIM).to(device)
    dummy = torch.randn(4, 128, width).to(device)
    with torch.no_grad():
        out = m(dummy)
    params = sum(p.numel() for p in m.parameters())
    assert dummy.shape == out.shape
    print(f"  Width {width}: {dummy.shape} -> {out.shape}  params={params:,}  PASS")
    del m


# ======================================================================
# TRAINING FUNCTION
# ======================================================================
def train_one_machine(machine_name, X_train, X_val, config):
    n_samples, n_mels, n_frames = X_train.shape
    input_dim = n_mels * n_frames

    model = AcousticAutoencoder(input_dim, config['bottleneck_dim']).to(config['device'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val)),
        batch_size=config['batch_size'], shuffle=False
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['num_epochs']):
        # TRAIN
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(config['device'])
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # VALIDATE
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(config['device'])
                output = model(batch)
                loss = criterion(output, batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            marker = 'Best'
        else:
            epochs_no_improve += 1
            marker = f'No improve {epochs_no_improve}/{config["patience"]}'

        if (epoch + 1) % 5 == 0 or epochs_no_improve == 0 or epochs_no_improve >= config['patience']:
            print(f"    Epoch {epoch+1:3d}/{config['num_epochs']} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                  f"LR: {current_lr:.1e} | {marker}")

        if epochs_no_improve >= config['patience']:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    return model, history


# ======================================================================
# PHASE 2: TRAIN ALL MACHINES
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 2: TRAINING AUTOENCODERS")
print("=" * 70)

machine_types = sorted([
    d for d in os.listdir(PROCESSED_DIR)
    if os.path.isdir(os.path.join(PROCESSED_DIR, d))
])
print(f"Found {len(machine_types)} machine types: {machine_types}")

config = {
    'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
    'num_epochs': NUM_EPOCHS, 'patience': PATIENCE,
    'bottleneck_dim': BOTTLENECK_DIM, 'device': device,
}

all_histories = {}
training_summary = []
total_start = time.time()

for machine in machine_types:
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {machine}")
    print(f"{'=' * 70}")

    X_train = np.load(os.path.join(PROCESSED_DIR, machine, "X_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DIR, machine, "X_val.npy"))
    print(f"  Data: X_train={X_train.shape}  X_val={X_val.shape}")

    t0 = time.time()
    model, history = train_one_machine(machine, X_train, X_val, config)
    elapsed = time.time() - t0

    best_val = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val) + 1

    save_dir = os.path.join(MODELS_DIR, machine)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    torch.save({
        'input_dim': model.input_dim, 'bottleneck_dim': BOTTLENECK_DIM,
        'n_mels': X_train.shape[1], 'n_frames': X_train.shape[2],
        'best_val_loss': best_val, 'best_epoch': best_epoch,
        'total_epochs': len(history['val_loss']),
    }, os.path.join(save_dir, "metadata.pth"))

    all_histories[machine] = history
    training_summary.append({
        'machine': machine, 'best_val_loss': best_val,
        'best_epoch': best_epoch, 'total_epochs': len(history['val_loss']),
        'time_sec': elapsed,
    })

    print(f"  Saved to {save_dir}")
    print(f"  Best val loss: {best_val:.6f} (epoch {best_epoch}) | Time: {elapsed:.1f}s")

total_elapsed = time.time() - total_start

# Training summary
print(f"\n{'=' * 70}")
print(f"TRAINING SUMMARY — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
print(f"{'=' * 70}")
print(f"{'Machine':<12s} | {'Best Val Loss':>14s} | {'Best Epoch':>10s} | {'Total Epochs':>12s} | {'Time':>6s}")
print("-" * 68)
for s in training_summary:
    print(f"{s['machine']:<12s} | {s['best_val_loss']:14.6f} | {s['best_epoch']:10d} | {s['total_epochs']:12d} | {s['time_sec']:6.1f}s")

# Training curves
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for idx, (machine, history) in enumerate(all_histories.items()):
    if idx >= len(axes): break
    ax = axes[idx]
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs_range, history['train_loss'], label='Train', linewidth=1.5)
    ax.plot(epochs_range, history['val_loss'], label='Validation', linewidth=1.5)
    best_ep = history['val_loss'].index(min(history['val_loss'])) + 1
    ax.axvline(x=best_ep, color='red', linestyle='--', alpha=0.5, label=f'Best (ep {best_ep})')
    ax.set_title(machine, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for idx in range(len(all_histories), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Training & Validation Loss Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs(MODELS_DIR, exist_ok=True)
plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved training curves to {os.path.join(MODELS_DIR, 'training_curves.png')}")
plt.close()


# ======================================================================
# PHASE 3: EVALUATION
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 3: ANOMALY DETECTION EVALUATION")
print("=" * 70)

import librosa
import joblib
from sklearn.metrics import roc_auc_score, roc_curve

SAMPLE_RATE = 16000
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
RAW_DIR     = os.path.join("data", "raw")

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_machine(machine_name):
    print(f"  Loading model and scaler...")

    meta = torch.load(os.path.join(MODELS_DIR, machine_name, "metadata.pth"),
                      map_location=device, weights_only=False)
    model = AcousticAutoencoder(meta['input_dim'], meta['bottleneck_dim']).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, machine_name, "best_model.pth"),
                   map_location=device, weights_only=True)
    )
    model.eval()

    n_mels   = meta['n_mels']
    n_frames = meta['n_frames']

    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine_name, "scaler.save"))

    test_dir  = os.path.join(RAW_DIR, machine_name, "test")
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    if len(wav_files) == 0:
        print(f"  No test files found!")
        return None

    print(f"  Processing {len(wav_files)} test files...")

    scores = []
    labels = []

    for fpath in wav_files:
        fname = os.path.basename(fpath)

        if "anomaly" in fname:
            label = 1
        elif "normal" in fname:
            label = 0
        else:
            continue

        try:
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Pad or truncate to match training width
            if log_mel.shape[1] > n_frames:
                log_mel = log_mel[:, :n_frames]
            elif log_mel.shape[1] < n_frames:
                pad_w = n_frames - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0,0),(0,pad_w)),
                                 mode='constant', constant_values=log_mel.min())

            H, W = log_mel.shape
            flat = log_mel.reshape(1, H * W)
            scaled = scaler.transform(flat)
            scaled = scaled.reshape(1, H, W)

            with torch.no_grad():
                x = torch.FloatTensor(scaled).to(device)
                recon = model(x)
                mse = torch.mean((recon - x) ** 2).item()

            scores.append(mse)
            labels.append(label)

        except Exception as e:
            print(f"  Error: {fname}: {e}")

    scores = np.array(scores)
    labels = np.array(labels)

    auc  = roc_auc_score(labels, scores)
    pauc = roc_auc_score(labels, scores, max_fpr=0.1)

    print(f"  AUC={auc:.4f}  pAUC={pauc:.4f}  "
          f"({int(np.sum(labels==0))} normal, {int(np.sum(labels==1))} anomaly)")

    return {
        'scores': scores, 'labels': labels,
        'auc': auc, 'pauc': pauc,
        'n_normal': int(np.sum(labels == 0)),
        'n_anomaly': int(np.sum(labels == 1)),
        'model': model,
    }


# Run evaluation
all_results = {}

for machine in machine_types:
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {machine}")
    print(f"{'=' * 60}")
    result = evaluate_machine(machine)
    if result is not None:
        all_results[machine] = result


# ======================================================================
# RESULTS SUMMARY
# ======================================================================
print(f"\n{'=' * 70}")
print(f"ANOMALY DETECTION RESULTS")
print(f"{'=' * 70}")
print(f"{'Machine':<12s} | {'AUC-ROC':>8s} | {'pAUC(10%)':>9s} | {'Normal':>6s} | {'Anomaly':>7s} | {'Verdict':>10s}")
print("-" * 70)

aucs = []
paucs = []

for machine in sorted(all_results.keys()):
    r = all_results[machine]
    auc = r['auc']
    pauc = r['pauc']
    aucs.append(auc)
    paucs.append(pauc)

    if auc >= 0.9: verdict = "Excellent"
    elif auc >= 0.8: verdict = "Good"
    elif auc >= 0.7: verdict = "Acceptable"
    elif auc >= 0.6: verdict = "Weak"
    else: verdict = "Poor"

    print(f"{machine:<12s} | {auc:8.4f} | {pauc:9.4f} | {r['n_normal']:6d} | {r['n_anomaly']:7d} | {verdict:>10s}")

print("-" * 70)
print(f"{'AVERAGE':<12s} | {np.mean(aucs):8.4f} | {np.mean(paucs):9.4f} |        |         |")
print(f"{'=' * 70}")


# ======================================================================
# PLOTS
# ======================================================================

# 1. Score distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for idx, (machine, result) in enumerate(sorted(all_results.items())):
    if idx >= len(axes): break
    ax = axes[idx]
    normal_scores  = result['scores'][result['labels'] == 0]
    anomaly_scores = result['scores'][result['labels'] == 1]
    ax.hist(normal_scores, bins=30, alpha=0.6, color='steelblue', label='Normal', density=True)
    ax.hist(anomaly_scores, bins=30, alpha=0.6, color='crimson', label='Anomaly', density=True)
    ax.set_title(f"{machine} (AUC={result['auc']:.3f})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Anomaly Score (MSE)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for idx in range(len(all_results), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Anomaly Score Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_distributions.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'score_distributions.png')}")
plt.close()

# 2. ROC curves
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for idx, (machine, result) in enumerate(sorted(all_results.items())):
    if idx >= len(axes): break
    ax = axes[idx]
    fpr, tpr, _ = roc_curve(result['labels'], result['scores'])
    ax.plot(fpr, tpr, color='steelblue', linewidth=2, label=f"AUC = {result['auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    mask = fpr <= 0.1
    if np.any(mask):
        ax.fill_between(fpr[mask], 0, tpr[mask], alpha=0.2, color='orange',
                        label=f'pAUC = {result["pauc"]:.3f}')
    ax.set_title(machine, fontsize=11, fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
for idx in range(len(all_results), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('ROC Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(RESULTS_DIR, 'roc_curves.png')}")
plt.close()

# 3. Sample reconstructions
machines_to_show = sorted(all_results.keys())[:4]
fig, axes = plt.subplots(len(machines_to_show), 4, figsize=(18, 4 * len(machines_to_show)))

for row, machine in enumerate(machines_to_show):
    result = all_results[machine]
    model_vis = result['model']
    model_vis.eval()

    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine, "scaler.save"))
    meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                      map_location=device, weights_only=False)
    n_mels_v, n_frames_v = meta['n_mels'], meta['n_frames']
    test_dir = os.path.join(RAW_DIR, machine, "test")

    normal_file = sorted(glob.glob(os.path.join(test_dir, "*normal*.wav")))[0]
    anomaly_file = sorted(glob.glob(os.path.join(test_dir, "*anomaly*.wav")))[0]

    for col_off, (fpath, label) in enumerate([(normal_file, 'Normal'), (anomaly_file, 'Anomaly')]):
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        if log_mel.shape[1] > n_frames_v:
            log_mel = log_mel[:, :n_frames_v]
        elif log_mel.shape[1] < n_frames_v:
            pad_w = n_frames_v - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0,0),(0,pad_w)), mode='constant', constant_values=log_mel.min())

        flat = log_mel.reshape(1, n_mels_v * n_frames_v)
        scaled = scaler.transform(flat).reshape(1, n_mels_v, n_frames_v)

        with torch.no_grad():
            x = torch.FloatTensor(scaled).to(device)
            recon = model_vis(x).cpu().numpy()[0]
            mse_val = float(np.mean((scaled[0] - recon) ** 2))

        axes[row, col_off * 2].imshow(scaled[0], aspect='auto', origin='lower', cmap='magma')
        axes[row, col_off * 2].set_title(f"{machine} - {label} (Original)", fontsize=9)
        axes[row, col_off * 2].set_ylabel('Mel Band')

        axes[row, col_off * 2 + 1].imshow(recon, aspect='auto', origin='lower', cmap='magma')
        axes[row, col_off * 2 + 1].set_title(f"Reconstruction (MSE={mse_val:.5f})", fontsize=9)

plt.suptitle('Original vs Reconstruction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_reconstructions.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(RESULTS_DIR, 'sample_reconstructions.png')}")
plt.close()

print("\n" + "=" * 70)
print("ALL DONE! Check the results/ and models/ directories for outputs.")
print("=" * 70)
