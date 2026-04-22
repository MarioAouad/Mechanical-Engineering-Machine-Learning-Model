import torch
import torch.nn as nn
import numpy as np
import os, glob, librosa, joblib
from sklearn.metrics import roc_auc_score

# ==========================================================================
# CNN AUTOENCODER
# ==========================================================================
class CNNAutoencoder(nn.Module):
    """
    Conv2D Autoencoder for (1, 128, 64) spectrogram patches.
    Encoder: 5 conv layers with stride-2 downsampling.
    Bottleneck: Dense layer compressing to 64 dimensions.
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
        
        # Dense Bottleneck (128 * 4 * 2 = 1024 -> 64)
        self.fc_enc = nn.Linear(1024, 64)
        self.fc_dec = nn.Linear(64, 1024)

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
            nn.BatchNorm2d(cout), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1) # Spatial dropout to prevent pixel memorization
        )

    def _dec_block(self, cin, cout, k, s, p, op):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm2d(cout), 
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, return_embedding=False):
        # Encode
        e1 = self.enc1(x)   # (B, 16, 64, 32)
        e2 = self.enc2(e1)  # (B, 32, 32, 16)
        e3 = self.enc3(e2)  # (B, 64, 16, 8)
        e4 = self.enc4(e3)  # (B, 128, 8, 4)
        e5 = self.enc5(e4)  # (B, 128, 4, 2)
        
        # Flatten and bottleneck
        batch_size = e5.size(0)
        flat = e5.view(batch_size, -1)
        encoded = self.fc_enc(flat)
        
        if return_embedding:
            return encoded
            
        # Expand and reshape
        decoded_flat = self.fc_dec(encoded)
        d_in = decoded_flat.view(batch_size, 128, 4, 2)
        
        # Decode
        d5 = self.dec5(d_in)  # (B, 128, 8, 4)
        d4 = self.dec4(d5)  # (B, 64, 16, 8)
        d3 = self.dec3(d4)  # (B, 32, 32, 16)
        d2 = self.dec2(d3)  # (B, 16, 64, 32)
        d1 = self.dec1(d2)  # (B, 1, 128, 64)
        return d1

def extract_patches(spectrogram, width=64, stride=32):
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

# ==========================================================================
# CONFIGURATION
# ==========================================================================
SAMPLE_RATE    = 16000
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models_cnn")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================
# SCORING LOGIC (TWEAK THIS AS MUCH AS YOU WANT)
# ==========================================================================
def score_file(model, knn_model, fpath, scaler, n_frames):
    """Process one test wav file and return anomaly score using KNN distance."""
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

    # ---------------------------------------------------------
    # SCALER LOGIC
    # ---------------------------------------------------------
    # NEW SCALER (Because the scalers on disk expect 128 features)
    scaled = scaler.transform(log_mel.T).T 

    # Extract patches
    patches = extract_patches(scaled)
    if len(patches) == 0:
        return 0.0

    # ---------------------------------------------------------
    # ANOMALY SCORING LOGIC
    # ---------------------------------------------------------
    x = torch.FloatTensor(patches).unsqueeze(1).to(device)
    with torch.no_grad():
        embeddings = model(x, return_embedding=True).cpu().numpy()
        
    # KNN kneighbors returns (distances, indices)
    # The anomaly score is the mean distance to the 5 nearest neighbors
    distances, _ = knn_model.kneighbors(embeddings)
    anomaly_scores = np.mean(distances, axis=1)

    # Pinpoint scoring: take top 2 worst patches
    top_2 = np.sort(anomaly_scores)[-2:]
    return float(np.mean(top_2))

# ==========================================================================
# EVALUATION LOOP
# ==========================================================================
def main():
    machine_types = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
    
    print(f"Device: {device}")
    print("=" * 60)
    print("RAPID EVALUATION MODE (No Training)")
    print("=" * 60)

    overall_metrics = []

    for machine in machine_types:
        print(f"\nEVALUATING: {machine}")
        
        # Load Model & Metadata
        model_dir = os.path.join(MODELS_DIR, machine)
        if not os.path.exists(model_dir):
            print(f"  [SKIPPED] No trained model found at {model_dir}")
            continue
            
        meta = torch.load(os.path.join(model_dir, "metadata.pth"), map_location=device, weights_only=False)
        model = CNNAutoencoder().to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=False))
        model.eval()

        # Load Scaler
        scaler_path = os.path.join(PROJECT_ROOT, "data", "processed", machine, "scaler.save")
        scaler = joblib.load(scaler_path)
        
        # Load KNN
        knn_path = os.path.join(model_dir, "knn.save")
        if not os.path.exists(knn_path):
            print(f"  [SKIPPED] No KNN model found at {knn_path}")
            continue
        knn_model = joblib.load(knn_path)

        # Get Test Files
        test_dir = os.path.join(RAW_DIR, machine, "test")
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
        
        y_true, y_scores = [], []
        
        for fpath in test_files:
            fname = os.path.basename(fpath)
            is_anomaly = 1 if "anomaly" in fname else 0
            score = score_file(model, knn_model, fpath, scaler, meta['n_frames'])
            
            y_true.append(is_anomaly)
            y_scores.append(score)

        # Calculate AUC
        auc = roc_auc_score(y_true, y_scores)
        
        # Calculate pAUC (max_fpr = 0.1)
        # Using the standard formula to rescale the partial area
        min_tpr = 0.0
        max_fpr = 0.1
        pauc = roc_auc_score(y_true, y_scores, max_fpr=max_fpr)
        # Rescale pAUC to 0-1 range to match standard DCASE scoring output
        pauc_rescaled = 0.5 * (1 + (pauc - 0.5) / 0.5) 

        print(f"  AUC={auc:.4f}  pAUC={pauc_rescaled:.4f}  (Test files: {len(test_files)})")
        overall_metrics.append({'machine': machine, 'auc': auc, 'pauc': pauc_rescaled})

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Machine':<12} | {'AUC':>8} | {'pAUC':>8}")
    print("-" * 35)
    
    avg_auc = np.mean([m['auc'] for m in overall_metrics])
    avg_pauc = np.mean([m['pauc'] for m in overall_metrics])
    
    for m in overall_metrics:
        print(f"{m['machine']:<12} | {m['auc']:8.4f} | {m['pauc']:8.4f}")
    print("-" * 35)
    print(f"{'AVERAGE':<12} | {avg_auc:8.4f} | {avg_pauc:8.4f}")

if __name__ == "__main__":
    main()
