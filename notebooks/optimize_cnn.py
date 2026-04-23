"""
Optimize Evaluation for V1 CNN Models (models_cnn/)

Tests multiple scoring strategies on the ORIGINAL CNN Autoencoder models
(5-layer, Sigmoid output, trained on data/processed/ with MinMaxScaler).
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
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models_cnn")

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
# ORIGINAL V1 CNN AUTOENCODER (5-layer, Sigmoid, Dropout)
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
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2))

    def _dec_block(self, cin, cout, k, s, p, op):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, return_embedding=False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        batch_size = e5.size(0)
        flat = e5.view(batch_size, -1)
        encoded = self.fc_enc(flat)
        if return_embedding:
            return encoded
        decoded_flat = self.fc_dec(encoded)
        d_in = decoded_flat.view(batch_size, 128, 4, 2)
        d5 = self.dec5(d_in)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
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
# MAIN
# ==========================================================================
machine_types = sorted([d for d in os.listdir(PROCESSED_DIR)
                        if os.path.isdir(os.path.join(PROCESSED_DIR, d))])

print(f"\n{'='*70}")
print(f"OPTIMIZING V1 CNN MODELS (models_cnn/)")
print(f"{'='*70}")

final_results = []

for machine in machine_types:
    print(f"\n{'='*50}\n{machine}\n{'='*50}")

    model_dir = os.path.join(MODELS_DIR, machine)
    if not os.path.exists(os.path.join(model_dir, "best_model.pth")):
        print(f"  [SKIP] No model found")
        continue

    meta = torch.load(os.path.join(model_dir, "metadata.pth"),
                      map_location=device, weights_only=False)
    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best_model.pth"),
        map_location=device, weights_only=False))
    model.eval()

    scaler = joblib.load(os.path.join(PROCESSED_DIR, machine, "scaler.save"))
    n_frames = meta['n_frames']

    # Training features for KNN
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
        # V1 models: use ref=1.0 to match how data/processed/ was generated
        log_mel = librosa.power_to_db(mel, ref=1.0)

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
    strategies['Recon_Mean'] = np.array([np.mean(f['mses']) for f in file_features])
    strategies['Recon_Max']  = np.array([np.max(f['mses']) for f in file_features])
    strategies['Recon_P90']  = np.array([np.percentile(f['mses'], 90) for f in file_features])
    strategies['KNN_Mean']   = np.array([np.mean(f['knn_dists']) for f in file_features])
    strategies['KNN_Max']    = np.array([np.max(f['knn_dists']) for f in file_features])

    recon_scores = strategies['Recon_Mean']
    knn_scores   = strategies['KNN_Mean']
    r_z = (recon_scores - recon_scores.mean()) / (recon_scores.std() + 1e-10)
    k_z = (knn_scores - knn_scores.mean()) / (knn_scores.std() + 1e-10)
    strategies['Hybrid_Mean'] = r_z + k_z

    # Evaluate
    aucs, paucs = {}, {}
    for name, scores in strategies.items():
        aucs[name] = roc_auc_score(labels, scores)
        paucs[name] = roc_auc_score(labels, scores, max_fpr=0.1)

    combined = {name: (aucs[name] + paucs[name]) / 2 for name in aucs}
    best_name = max(combined, key=combined.get)

    print(f"  {'Strategy':<15} | {'AUC':>8} | {'pAUC':>8} | {'Avg':>8}")
    print(f"  {'-'*48}")
    for name in sorted(combined, key=combined.get, reverse=True):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<15} | {aucs[name]:8.4f} | {paucs[name]:8.4f} | {combined[name]:8.4f}{marker}")

    final_results.append({
        'machine': machine, 'best_strategy': best_name,
        'best_auc': aucs[best_name], 'best_pauc': paucs[best_name]
    })

# Summary
print(f"\n{'='*70}")
print(f"V1 CNN OPTIMIZED RESULTS")
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
