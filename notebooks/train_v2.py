"""
CNN Autoencoder V2 — Train + Evaluate (DCASE 2024 Task 2)

Key changes from V1:
  - 3-layer CNN (simpler, less overfitting risk)
  - Linear output (no Sigmoid — can reconstruct any z-scored value)
  - MSE loss (matches evaluation metric)
  - No SpecAugment (standard AE, not denoising)
  - No Dropout (simpler model doesn't need heavy regularization)
  - Z-score normalized scoring (neutralizes domain-dependent error baselines)
  - Saves to models_v2/ and results_v2/ (does NOT overwrite V1)
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, copy, time, glob, librosa, joblib
from sklearn.metrics import roc_auc_score, roc_curve

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================================
# CONFIG
# ==========================================================================
PATCH_WIDTH    = 64
PATCH_STRIDE   = 32
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
NUM_EPOCHS     = 150
PATIENCE       = 25
SAMPLE_RATE    = 16000
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048

PROCESSED_DIR = os.path.join("data", "processed_v2")
RAW_DIR       = os.path.join("data", "raw")
MODELS_DIR    = os.path.join("models_v2")
RESULTS_DIR   = os.path.join("results_v2")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | PyTorch: {torch.__version__}")

# ==========================================================================
# CNN AUTOENCODER V2
# ==========================================================================
class CNNAutoencoderV2(nn.Module):
    """
    3-layer Conv2D Autoencoder for (1, 128, 64) spectrogram patches.
    Linear output — can reconstruct any value (matches z-scored input).
    128-dim bottleneck for richer embeddings.
    """
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        # Encoder: 3 conv layers with stride-2 downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Bottleneck: 128*16*8 = 16384 → bottleneck_dim
        self.flatten_size = 128 * 16 * 8
        self.fc_enc = nn.Linear(self.flatten_size, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, self.flatten_size)

        # Decoder: 3 transposed conv layers
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Final layer: NO activation (linear output)
        self.dec1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, return_embedding=False):
        # Encode
        e1 = self.enc1(x)    # (B, 32, 64, 32)
        e2 = self.enc2(e1)   # (B, 64, 32, 16)
        e3 = self.enc3(e2)   # (B, 128, 16, 8)
        # Bottleneck
        flat = e3.view(e3.size(0), -1)
        embedding = self.fc_enc(flat)
        if return_embedding:
            return embedding
        # Decode
        dec_flat = self.fc_dec(embedding)
        d_in = dec_flat.view(-1, 128, 16, 8)
        d3 = self.dec3(d_in)  # (B, 64, 32, 16)
        d2 = self.dec2(d3)    # (B, 32, 64, 32)
        d1 = self.dec1(d2)    # (B, 1, 128, 64)
        return d1

# Smoke test
m = CNNAutoencoderV2().to(device)
dummy = torch.randn(2, 1, 128, PATCH_WIDTH).to(device)
with torch.no_grad():
    out = m(dummy)
params = sum(p.numel() for p in m.parameters())
print(f"Smoke test: {dummy.shape} -> {out.shape} | {params:,} params")
assert dummy.shape == out.shape, f"Shape mismatch! {dummy.shape} vs {out.shape}"
del m

# ==========================================================================
# PATCH EXTRACTION
# ==========================================================================
def extract_patches(spectrogram, width=PATCH_WIDTH, stride=PATCH_STRIDE):
    """Extract overlapping patches from a (128, W) spectrogram."""
    _, W = spectrogram.shape
    patches = []
    for start in range(0, W - width + 1, stride):
        patches.append(spectrogram[:, start:start+width])
    if patches and (W - width) % stride != 0:
        patches.append(spectrogram[:, W-width:W])
    if not patches and W >= width:
        patches.append(spectrogram[:, :width])
    return np.array(patches)

# ==========================================================================
# TRAINING FUNCTION
# ==========================================================================
def train_one_machine(machine, X_train_patches, X_val_patches):
    print(f"  Train patches: {X_train_patches.shape} | Val patches: {X_val_patches.shape}")

    X_tr = torch.FloatTensor(X_train_patches).unsqueeze(1)
    X_va = torch.FloatTensor(X_val_patches).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    model = CNNAutoencoderV2().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )

    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train': [], 'val': []}

    for epoch in range(NUM_EPOCHS):
        # Train — standard AE (NO SpecAugment)
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                val_losses.append(criterion(output, batch).item())

        avg_tr = np.mean(losses)
        avg_va = np.mean(val_losses)
        history['train'].append(avg_tr)
        history['val'].append(avg_va)

        scheduler.step(avg_va)
        lr = optimizer.param_groups[0]['lr']

        if avg_va < best_val:
            best_val = avg_va
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            mk = 'Best'
        else:
            no_improve += 1
            mk = f'{no_improve}/{PATIENCE}'

        if (epoch+1) % 10 == 0 or no_improve == 0 or no_improve >= PATIENCE:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | Tr: {avg_tr:.6f} | Va: {avg_va:.6f} | LR: {lr:.1e} | {mk}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, history

# ==========================================================================
# COMPUTE TRAINING RECONSTRUCTION STATS (for z-score scoring)
# ==========================================================================
def compute_train_stats(model, X_train_patches):
    """Compute mean and std of reconstruction MSE on training patches."""
    model.eval()
    X = torch.FloatTensor(X_train_patches).unsqueeze(1)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)

    all_errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            per_patch = torch.mean((batch - recon)**2, dim=(1,2,3)).cpu().numpy()
            all_errors.append(per_patch)

    all_errors = np.concatenate(all_errors)
    return float(np.mean(all_errors)), float(np.std(all_errors))

# ==========================================================================
# PHASE 1: TRAIN ALL MACHINES
# ==========================================================================
machine_types = sorted([d for d in os.listdir(PROCESSED_DIR)
                        if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
print(f"\nMachines: {machine_types}\n")

all_histories = {}
training_summary = []
total_start = time.time()

for machine in machine_types:
    print(f"{'='*60}\nTRAINING: {machine}\n{'='*60}")

    X_train = np.load(os.path.join(PROCESSED_DIR, machine, "X_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DIR, machine, "X_val.npy"))
    print(f"  Raw data: train={X_train.shape}, val={X_val.shape}")

    train_patches = np.concatenate([extract_patches(s) for s in X_train])
    val_patches   = np.concatenate([extract_patches(s) for s in X_val])

    t0 = time.time()
    model, hist = train_one_machine(machine, train_patches, val_patches)
    elapsed = time.time() - t0

    # Compute training reconstruction stats for z-score scoring
    print(f"  Computing training reconstruction stats...")
    train_mu, train_sigma = compute_train_stats(model, train_patches)
    print(f"  Train MSE: mu={train_mu:.6f}, sigma={train_sigma:.6f}")

    best_v = min(hist['val'])
    best_ep = hist['val'].index(best_v) + 1

    save_dir = os.path.join(MODELS_DIR, machine)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    torch.save({
        'best_val': best_v, 'best_epoch': best_ep,
        'total_epochs': len(hist['val']), 'n_frames': X_train.shape[2],
        'train_mse_mean': train_mu, 'train_mse_std': train_sigma
    }, os.path.join(save_dir, "metadata.pth"))

    all_histories[machine] = hist
    training_summary.append({
        'machine': machine, 'best_val': best_v, 'best_ep': best_ep,
        'epochs': len(hist['val']), 'time': elapsed,
        'n_patches': len(train_patches), 'train_mu': train_mu, 'train_sigma': train_sigma
    })
    print(f"  Best val: {best_v:.6f} (ep {best_ep}) | {elapsed:.0f}s | {len(train_patches)} patches\n")

total_time = time.time() - total_start
print(f"\n{'='*60}\nTRAINING DONE — {total_time:.0f}s ({total_time/60:.1f} min)\n{'='*60}")
print(f"{'Machine':<12} | {'Val Loss':>10} | {'Epoch':>5} | {'Patches':>8} | {'Time':>6} | {'mu':>10} | {'sigma':>10}")
print("-"*80)
for s in training_summary:
    print(f"{s['machine']:<12} | {s['best_val']:10.6f} | {s['best_ep']:5d} | {s['n_patches']:8d} | {s['time']:5.0f}s | {s['train_mu']:10.6f} | {s['train_sigma']:10.6f}")

# Training curves
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for idx, (mc, h) in enumerate(all_histories.items()):
    if idx >= 8: break
    ax = axes[idx]
    ax.plot(h['train'], label='Train', lw=1.5)
    ax.plot(h['val'], label='Val', lw=1.5)
    bp = h['val'].index(min(h['val'])) + 1
    ax.axvline(bp, color='red', ls='--', alpha=0.5, label=f'Best ep {bp}')
    ax.set_title(mc, fontweight='bold'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
for i in range(len(all_histories), 8): axes[i].set_visible(False)
plt.suptitle('CNN V2 Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved training curves.\n")

# ==========================================================================
# PHASE 2: EVALUATE WITH Z-SCORE SCORING
# ==========================================================================
print(f"{'='*60}\nPHASE 2: EVALUATION (Z-Score Scoring)\n{'='*60}")

def score_file(model, fpath, scaler, n_frames, train_mu, train_sigma):
    """Score a test file using z-score normalized reconstruction error."""
    y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    # ref=np.max to match training preprocessing
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad/truncate
    if log_mel.shape[1] > n_frames:
        log_mel = log_mel[:, :n_frames]
    elif log_mel.shape[1] < n_frames:
        pw = n_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pw)), constant_values=log_mel.min())

    # Apply same StandardScaler (per mel band)
    scaled = scaler.transform(log_mel.T).T

    # Extract patches
    patches = extract_patches(scaled)
    if len(patches) == 0:
        return 0.0

    # Compute per-patch MSE
    x = torch.FloatTensor(patches).unsqueeze(1).to(device)
    with torch.no_grad():
        recon = model(x)
        per_patch_mse = torch.mean((x - recon)**2, dim=(1,2,3)).cpu().numpy()

    # Z-score normalize against training distribution
    z_scores = (per_patch_mse - train_mu) / (train_sigma + 1e-10)

    # Anomaly score = mean z-score across all patches
    return float(np.mean(z_scores))

all_results = {}
for machine in machine_types:
    print(f"\n{'='*50}\nEVALUATING: {machine}\n{'='*50}")

    meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                      map_location=device, weights_only=False)
    model = CNNAutoencoderV2().to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, machine, "best_model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine, "scaler.save"))
    n_frames   = meta['n_frames']
    train_mu   = meta['train_mse_mean']
    train_sigma = meta['train_mse_std']
    print(f"  Train stats: mu={train_mu:.6f}, sigma={train_sigma:.6f}")

    test_dir = os.path.join(RAW_DIR, machine, "test")
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    print(f"  {len(wav_files)} test files")

    scores, labels = [], []
    for fp in wav_files:
        fn = os.path.basename(fp)
        if "anomaly" in fn: label = 1
        elif "normal" in fn: label = 0
        else: continue
        try:
            s = score_file(model, fp, scaler, n_frames, train_mu, train_sigma)
            scores.append(s); labels.append(label)
        except Exception as e:
            print(f"  Error: {fn}: {e}")

    scores = np.array(scores)
    labels = np.array(labels)
    auc  = roc_auc_score(labels, scores)
    pauc = roc_auc_score(labels, scores, max_fpr=0.1)
    nn_  = int(np.sum(labels==0))
    na_  = int(np.sum(labels==1))
    print(f"  AUC={auc:.4f}  pAUC={pauc:.4f}  ({nn_} normal, {na_} anomaly)")
    all_results[machine] = {'scores': scores, 'labels': labels,
                            'auc': auc, 'pauc': pauc, 'n': nn_, 'a': na_}

# ==========================================================================
# RESULTS
# ==========================================================================
print(f"\n{'='*70}")
print(f"CNN V2 ANOMALY DETECTION RESULTS")
print(f"{'='*70}")
print(f"{'Machine':<12} | {'AUC':>8} | {'pAUC':>8} | {'Normal':>6} | {'Anom':>5} | {'Verdict':>10}")
print("-"*65)
aucs, paucs = [], []
for mc in sorted(all_results):
    r = all_results[mc]
    aucs.append(r['auc']); paucs.append(r['pauc'])
    v = "Excellent" if r['auc']>=0.9 else "Good" if r['auc']>=0.8 else "OK" if r['auc']>=0.7 else "Weak" if r['auc']>=0.6 else "Poor"
    print(f"{mc:<12} | {r['auc']:8.4f} | {r['pauc']:8.4f} | {r['n']:6d} | {r['a']:5d} | {v:>10}")
print("-"*65)
print(f"{'AVERAGE':<12} | {np.mean(aucs):8.4f} | {np.mean(paucs):8.4f} |")
print(f"{'='*70}")

# Score distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 8)); axes = axes.flatten()
for i, (mc, r) in enumerate(sorted(all_results.items())):
    if i >= 8: break
    ax = axes[i]
    ax.hist(r['scores'][r['labels']==0], bins=30, alpha=0.6, color='steelblue', label='Normal', density=True)
    ax.hist(r['scores'][r['labels']==1], bins=30, alpha=0.6, color='crimson', label='Anomaly', density=True)
    ax.set_title(f"{mc} (AUC={r['auc']:.3f})", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
for i in range(len(all_results), 8): axes[i].set_visible(False)
plt.suptitle('CNN V2 Score Distributions (Z-Score Normalized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, axes = plt.subplots(2, 4, figsize=(20, 8)); axes = axes.flatten()
for i, (mc, r) in enumerate(sorted(all_results.items())):
    if i >= 8: break
    ax = axes[i]
    fpr, tpr, _ = roc_curve(r['labels'], r['scores'])
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC={r['auc']:.3f}")
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    mask = fpr <= 0.1
    if np.any(mask):
        ax.fill_between(fpr[mask], 0, tpr[mask], alpha=0.2, color='orange', label=f"pAUC={r['pauc']:.3f}")
    ax.set_title(mc, fontweight='bold'); ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1])
for i in range(len(all_results), 8): axes[i].set_visible(False)
plt.suptitle('CNN V2 ROC Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved all plots to {RESULTS_DIR}/")
print("DONE!")
