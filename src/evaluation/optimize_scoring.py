"""
Optimize Evaluation V2 (DCASE 2024 Task 2)

This script does NOT train anything. It loads already-trained CNN V2 models
and tests multiple scoring strategies to find the best one per machine.

The model class is defined inline to avoid importing train_v2.py
(which would re-execute its training loop since it has no __main__ guard).
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, glob, librosa, joblib
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# ==========================================================================
# CONFIG
# ==========================================================================
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v2")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "weights")

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
# MODEL CLASS (copied from train_v2.py to avoid import side-effects)
# ==========================================================================
class CNNAutoencoderV2(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.flatten_size = 128 * 16 * 8
        self.fc_enc = nn.Linear(self.flatten_size, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, self.flatten_size)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.dec1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, return_embedding=False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        flat = e3.view(e3.size(0), -1)
        embedding = self.fc_enc(flat)
        if return_embedding:
            return embedding
        dec_flat = self.fc_dec(embedding)
        d_in = dec_flat.view(-1, 128, 16, 8)
        d3 = self.dec3(d_in)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1

# ==========================================================================
# HELPERS
# ==========================================================================
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
    """Returns both reconstruction MSEs and bottleneck embeddings."""
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

# ==========================================================================
# MAIN EVALUATION
# ==========================================================================
machine_types = sorted([d for d in os.listdir(PROCESSED_DIR)
                        if os.path.isdir(os.path.join(PROCESSED_DIR, d))])

print(f"\n{'='*70}")
print(f"OPTIMIZING SCORING STRATEGIES (NO TRAINING)")
print(f"{'='*70}")

final_results = []

for machine in machine_types:
    print(f"\n{'='*50}\n{machine}\n{'='*50}")

    # Load metadata first to get architecture params
    meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                      map_location=device, weights_only=False)
    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine, "scaler.save"))
    n_frames = meta['n_frames']
    bn_dim = meta.get('bottleneck_dim', 128)  # default 128 for original V2 models

    # Load model with correct bottleneck size (NO training)
    model = CNNAutoencoderV2(bottleneck_dim=bn_dim).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, machine, "best_model.pth"),
        map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded model (bottleneck={bn_dim})")

    # Extract training features for KNN fitting
    print(f"  Loading training embeddings for KNN...")
    X_train = np.load(os.path.join(PROCESSED_DIR, machine, "X_train.npy"))
    train_patches = np.concatenate([extract_patches(s) for s in X_train])
    train_mses, train_embs = extract_all_features(model, train_patches)

    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(train_embs)
    print(f"  KNN fitted on {len(train_embs)} training embeddings")

    # Process test files
    test_dir = os.path.join(RAW_DIR, machine, "test")
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    labels = []
    file_features = []

    for fp in wav_files:
        fn = os.path.basename(fp)
        if "anomaly" in fn: labels.append(1)
        elif "normal" in fn: labels.append(0)
        else: continue

        y, sr = librosa.load(fp, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        if log_mel.shape[1] > n_frames:
            log_mel = log_mel[:, :n_frames]
        elif log_mel.shape[1] < n_frames:
            pw = n_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0,0),(0,pw)), constant_values=log_mel.min())

        scaled = scaler.transform(log_mel.T).T
        patches = extract_patches(scaled)

        if len(patches) > 0:
            mses, embs = extract_all_features(model, patches)
            knn_dists = np.mean(knn.kneighbors(embs)[0], axis=1)
            file_features.append({'mses': mses, 'knn_dists': knn_dists})
        else:
            file_features.append({'mses': np.zeros(1), 'knn_dists': np.zeros(1)})

    labels = np.array(labels)

    # Test all scoring strategies
    strategies = {}

    # Reconstruction-based
    strategies['Recon_Mean'] = np.array([np.mean(f['mses']) for f in file_features])
    strategies['Recon_Max']  = np.array([np.max(f['mses']) for f in file_features])
    strategies['Recon_P90']  = np.array([np.percentile(f['mses'], 90) for f in file_features])

    # KNN-based
    strategies['KNN_Mean'] = np.array([np.mean(f['knn_dists']) for f in file_features])
    strategies['KNN_Max']  = np.array([np.max(f['knn_dists']) for f in file_features])

    # Hybrid: normalized recon + knn
    recon_scores = strategies['Recon_Mean']
    knn_scores   = strategies['KNN_Mean']
    r_z = (recon_scores - recon_scores.mean()) / (recon_scores.std() + 1e-10)
    k_z = (knn_scores - knn_scores.mean()) / (knn_scores.std() + 1e-10)
    strategies['Hybrid_Mean'] = r_z + k_z

    # Evaluate all strategies with both AUC and pAUC
    aucs = {}
    paucs = {}
    for name, scores in strategies.items():
        aucs[name] = roc_auc_score(labels, scores)
        paucs[name] = roc_auc_score(labels, scores, max_fpr=0.1)

    # Select best by combined (AUC + pAUC) / 2
    combined = {name: (aucs[name] + paucs[name]) / 2 for name in aucs}
    best_name = max(combined, key=combined.get)

    print(f"  {'Strategy':<15} | {'AUC':>8} | {'pAUC':>8} | {'Avg':>8}")
    print(f"  {'-'*48}")
    for name in sorted(combined, key=combined.get, reverse=True):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<15} | {aucs[name]:8.4f} | {paucs[name]:8.4f} | {combined[name]:8.4f}{marker}")

    final_results.append({
        'machine': machine,
        'best_strategy': best_name,
        'best_auc': aucs[best_name],
        'best_pauc': paucs[best_name],
        'all_aucs': aucs,
        'all_paucs': paucs
    })

# ==========================================================================
# SUMMARY
# ==========================================================================
print(f"\n{'='*70}")
print(f"OPTIMIZED RESULTS SUMMARY")
print(f"{'='*70}")
print(f"{'Machine':<12} | {'Best Strategy':<15} | {'AUC':>8} | {'pAUC':>8}")
print("-" * 55)
avg_auc, avg_pauc = [], []
for r in final_results:
    print(f"{r['machine']:<12} | {r['best_strategy']:<15} | {r['best_auc']:8.4f} | {r['best_pauc']:8.4f}")
    avg_auc.append(r['best_auc'])
    avg_pauc.append(r['best_pauc'])
print("-" * 55)
print(f"{'AVERAGE':<12} | {'(best each)':<15} | {np.mean(avg_auc):8.4f} | {np.mean(avg_pauc):8.4f}")
print(f"{'='*70}")
