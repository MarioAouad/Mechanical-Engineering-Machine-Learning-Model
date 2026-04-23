"""
Evaluate Best Models — Unified Pipeline (DCASE 2024 Task 2)

Uses the best-performing model per machine:
  - ToyCar, fan:  V1 (5-layer CNN, Sigmoid, MinMaxScaler, ref=1.0)
  - All others:   V2 (3-layer CNN, Linear, StandardScaler, ref=np.max)

Tests all 6 scoring strategies and picks the best per machine.
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models_best")
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")

# Which pipeline each machine uses
V1_MACHINES = {"ToyCar", "fan", "valve"}  # 5-layer CNN, MinMaxScaler, ref=1.0
# Everything else uses V2:       3-layer CNN, StandardScaler, ref=np.max

# Processed data directories
V1_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
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
# MAIN EVALUATION
# ==========================================================================
machine_types = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']

print(f"\n{'='*70}")
print(f"BEST MODELS — UNIFIED EVALUATION")
print(f"{'='*70}")
print(f"V1 machines: {sorted(V1_MACHINES)}")
print(f"V2 machines: {sorted(set(machine_types) - V1_MACHINES)}")

final_results = []

for machine in machine_types:
    is_v1 = machine in V1_MACHINES
    pipeline = "V1" if is_v1 else "V2"
    processed_dir = V1_PROCESSED if is_v1 else V2_PROCESSED

    print(f"\n{'='*50}\n{machine} [{pipeline}]\n{'='*50}")

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
        ref_value = np.max  # np.max is a function, used by librosa as ref

    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, machine, "best_model.pth"),
        map_location=device, weights_only=False))
    model.eval()

    scaler = joblib.load(os.path.join(processed_dir, machine, "scaler.save"))
    print(f"  Pipeline: {pipeline} | ref={'1.0' if is_v1 else 'np.max'} | "
          f"Scaler: {type(scaler).__name__}")

    # Training features for KNN
    print(f"  Loading training embeddings...")
    X_train = np.load(os.path.join(processed_dir, machine, "X_train.npy"))
    train_patches = np.concatenate([extract_patches(s) for s in X_train])
    train_mses, train_embs = extract_all_features(model, train_patches)
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(train_embs)

    # Process test files
    test_dir = os.path.join(RAW_DIR, machine, "test")
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    labels, file_features = [], []
    for fp in wav_files:
        fn = os.path.basename(fp)
        if "anomaly" in fn: labels.append(1)
        elif "normal" in fn: labels.append(0)
        else: continue

        y, sr = librosa.load(fp, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=ref_value)

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

    # Test all strategies
    strategies = {}
    strategies['Recon_Mean'] = np.array([np.mean(f['mses']) for f in file_features])
    strategies['Recon_Max']  = np.array([np.max(f['mses']) for f in file_features])
    strategies['Recon_P90']  = np.array([np.percentile(f['mses'], 90) for f in file_features])
    strategies['KNN_Mean']   = np.array([np.mean(f['knn_dists']) for f in file_features])
    strategies['KNN_Max']    = np.array([np.max(f['knn_dists']) for f in file_features])

    # Negated reconstruction: for machines where anomalies are "simpler" sounds
    # that the AE reconstructs BETTER, low error = anomaly, so we negate
    strategies['Neg_Recon']     = -strategies['Recon_Mean']
    strategies['Neg_Recon_Max'] = -strategies['Recon_Max']

    r_z = (strategies['Recon_Mean'] - strategies['Recon_Mean'].mean()) / (strategies['Recon_Mean'].std() + 1e-10)
    k_z = (strategies['KNN_Mean'] - strategies['KNN_Mean'].mean()) / (strategies['KNN_Mean'].std() + 1e-10)
    strategies['Hybrid_Mean'] = r_z + k_z
    strategies['Neg_Hybrid']  = -r_z + k_z  # flip recon direction, keep KNN

    aucs, paucs = {}, {}
    for name, scores in strategies.items():
        aucs[name] = roc_auc_score(labels, scores)
        paucs[name] = roc_auc_score(labels, scores, max_fpr=0.1)

    combined = {n: (aucs[n] + paucs[n]) / 2 for n in aucs}
    best_name = max(combined, key=combined.get)

    print(f"  {'Strategy':<15} | {'AUC':>8} | {'pAUC':>8} | {'Avg':>8}")
    print(f"  {'-'*48}")
    for name in sorted(combined, key=combined.get, reverse=True):
        mk = " <-- BEST" if name == best_name else ""
        print(f"  {name:<15} | {aucs[name]:8.4f} | {paucs[name]:8.4f} | {combined[name]:8.4f}{mk}")

    final_results.append({
        'machine': machine, 'pipeline': pipeline,
        'best_strategy': best_name,
        'best_auc': aucs[best_name], 'best_pauc': paucs[best_name]
    })

# ==========================================================================
# SUMMARY
# ==========================================================================
print(f"\n{'='*70}")
print(f"BEST MODELS — FINAL RESULTS")
print(f"{'='*70}")
print(f"{'Machine':<12} | {'Pipe':>4} | {'Strategy':<15} | {'AUC':>8} | {'pAUC':>8}")
print("-" * 60)
avg_auc, avg_pauc = [], []
for r in final_results:
    print(f"{r['machine']:<12} | {r['pipeline']:>4} | {r['best_strategy']:<15} | {r['best_auc']:8.4f} | {r['best_pauc']:8.4f}")
    avg_auc.append(r['best_auc'])
    avg_pauc.append(r['best_pauc'])
print("-" * 60)
print(f"{'AVERAGE':<12} | {'':>4} | {'':>15} | {np.mean(avg_auc):8.4f} | {np.mean(avg_pauc):8.4f}")
print(f"{'='*70}")
