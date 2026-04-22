"""
CNN Autoencoder for Acoustic Anomaly Detection (DCASE 2024 Task 2)
Patch-based Conv2D approach for maximum detection accuracy.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, sys, copy, time, glob, librosa, joblib
from sklearn.metrics import roc_auc_score, roc_curve

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================================
# CONFIG
# ==========================================================================
PATCH_WIDTH    = 64    # frames per patch (~2 sec)
PATCH_STRIDE   = 32    # 50% overlap
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 150
PATIENCE       = 20
SAMPLE_RATE    = 16000
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048

PROCESSED_DIR = os.path.join("data", "processed")
RAW_DIR       = os.path.join("data", "raw")
MODELS_DIR    = os.path.join("models_cnn")
RESULTS_DIR   = os.path.join("results_cnn")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | PyTorch: {torch.__version__}")

# ==========================================================================
# CNN AUTOENCODER
# ==========================================================================
class CNNAutoencoder(nn.Module):
    """
    Conv2D Autoencoder for (1, 128, 64) spectrogram patches.
    Encoder: 5 conv layers with stride-2 downsampling.
    Bottleneck: (128, 4, 2) = 1024 values -> 8:1 compression.
    Decoder: 5 transposed conv layers.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._enc_block(1,   16, 5, 2, 2)
        self.enc2 = self._enc_block(16,  32, 5, 2, 2)
        self.enc3 = self._enc_block(32,  64, 3, 2, 1)
        self.enc4 = self._enc_block(64, 128, 3, 2, 1)
        self.enc5 = self._enc_block(128,128, 3, 2, 1)
        # Decoder
        self.dec5 = self._dec_block(128,128, 3, 2, 1, 1)
        self.dec4 = self._dec_block(128, 64, 3, 2, 1, 1)
        self.dec3 = self._dec_block(64,  32, 3, 2, 1, 1)
        self.dec2 = self._dec_block(32,  16, 5, 2, 2, 1)
        # Final layer: no BN, sigmoid output
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def _enc_block(self, cin, cout, k, s, p):
        return nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=p),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))

    def _dec_block(self, cin, cout, k, s, p, op):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)   # (16, 64, 32)
        e2 = self.enc2(e1)  # (32, 32, 16)
        e3 = self.enc3(e2)  # (64, 16, 8)
        e4 = self.enc4(e3)  # (128, 8, 4)
        e5 = self.enc5(e4)  # (128, 4, 2)
        # Decode
        d5 = self.dec5(e5)  # (128, 8, 4)
        d4 = self.dec4(d5)  # (64, 16, 8)
        d3 = self.dec3(d4)  # (32, 32, 16)
        d2 = self.dec2(d3)  # (16, 64, 32)
        d1 = self.dec1(d2)  # (1, 128, 64)
        return d1

# Smoke test
m = CNNAutoencoder().to(device)
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
    # Ensure we get the rightmost edge
    if patches and (W - width) % stride != 0:
        patches.append(spectrogram[:, W-width:W])
    if not patches and W >= width:
        patches.append(spectrogram[:, :width])
    return np.array(patches)  # (N, 128, 64)

def augment_patch(patch):
    """SpecAugment: random freq/time masking for regularization."""
    p = patch.clone()
    # Frequency mask (mask 1-8 mel bands)
    if torch.rand(1) < 0.5:
        f = torch.randint(1, 9, (1,)).item()
        f0 = torch.randint(0, 128 - f, (1,)).item()
        p[:, f0:f0+f, :] = 0
    # Time mask (mask 1-8 frames)
    if torch.rand(1) < 0.5:
        t = torch.randint(1, 9, (1,)).item()
        t0 = torch.randint(0, PATCH_WIDTH - t, (1,)).item()
        p[:, :, t0:t0+t] = 0
    return p

# ==========================================================================
# TRAINING FUNCTION
# ==========================================================================
def train_one_machine(machine, X_train_patches, X_val_patches):
    print(f"  Train patches: {X_train_patches.shape} | Val patches: {X_val_patches.shape}")

    # Add channel dim: (N, 128, 64) -> (N, 1, 128, 64)
    X_tr = torch.FloatTensor(X_train_patches).unsqueeze(1)
    X_va = torch.FloatTensor(X_val_patches).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

    model = CNNAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train': [], 'val': []}

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            # Apply SpecAugment
            aug_batch = torch.stack([augment_patch(b) for b in batch])
            aug_batch = aug_batch.to(device)
            output = model(aug_batch)
            loss = criterion(output, batch)  # Compare reconstruction to CLEAN input
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
        scheduler.step()
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
# PHASE 2: TRAIN
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

    # Extract patches
    train_patches = np.concatenate([extract_patches(s) for s in X_train])
    val_patches   = np.concatenate([extract_patches(s) for s in X_val])

    t0 = time.time()
    model, hist = train_one_machine(machine, train_patches, val_patches)
    elapsed = time.time() - t0

    best_v = min(hist['val'])
    best_ep = hist['val'].index(best_v) + 1

    save_dir = os.path.join(MODELS_DIR, machine)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    torch.save({'best_val': best_v, 'best_epoch': best_ep,
                'total_epochs': len(hist['val']), 'n_frames': X_train.shape[2]},
               os.path.join(save_dir, "metadata.pth"))

    all_histories[machine] = hist
    training_summary.append({'machine': machine, 'best_val': best_v,
                             'best_ep': best_ep, 'epochs': len(hist['val']),
                             'time': elapsed, 'n_patches': len(train_patches)})
    print(f"  Best val: {best_v:.6f} (ep {best_ep}) | {elapsed:.0f}s | {len(train_patches)} patches\n")

total_time = time.time() - total_start
print(f"\n{'='*60}\nTRAINING DONE — {total_time:.0f}s ({total_time/60:.1f} min)\n{'='*60}")
print(f"{'Machine':<12} | {'Val Loss':>10} | {'Epoch':>5} | {'Patches':>8} | {'Time':>6}")
print("-"*55)
for s in training_summary:
    print(f"{s['machine']:<12} | {s['best_val']:10.6f} | {s['best_ep']:5d} | {s['n_patches']:8d} | {s['time']:5.0f}s")

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
plt.suptitle('CNN Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved training curves.\n")

# ==========================================================================
# PHASE 3: EVALUATE
# ==========================================================================
print(f"{'='*60}\nPHASE 3: EVALUATION\n{'='*60}")

def score_file(model, fpath, scaler, n_frames):
    """Process one test wav file and return anomaly score."""
    y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad/truncate
    if log_mel.shape[1] > n_frames:
        log_mel = log_mel[:, :n_frames]
    elif log_mel.shape[1] < n_frames:
        pw = n_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pw)), constant_values=log_mel.min())

    # Scale
    H, W = log_mel.shape
    scaled = scaler.transform(log_mel.reshape(1, H*W)).reshape(H, W)

    # Extract patches
    patches = extract_patches(scaled)
    if len(patches) == 0:
        return 0.0

    # Compute per-patch MSE
    x = torch.FloatTensor(patches).unsqueeze(1).to(device)
    with torch.no_grad():
        recon = model(x)
        per_patch_mse = torch.mean((x - recon)**2, dim=(1,2,3)).cpu().numpy()

    # Score: mean of top 20% worst-reconstructed patches
    k = max(1, len(per_patch_mse) // 5)
    top_k = np.sort(per_patch_mse)[-k:]
    return float(np.mean(top_k))

all_results = {}
for machine in machine_types:
    print(f"\n{'='*50}\nEVALUATING: {machine}\n{'='*50}")

    meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                      map_location=device, weights_only=False)
    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, machine, "best_model.pth"),
                                     map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine, "scaler.save"))
    n_frames = meta['n_frames']

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
            s = score_file(model, fp, scaler, n_frames)
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
print(f"CNN ANOMALY DETECTION RESULTS")
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
plt.suptitle('CNN Score Distributions', fontsize=14, fontweight='bold')
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
plt.suptitle('CNN ROC Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved all plots to {RESULTS_DIR}/")
print("DONE!")
