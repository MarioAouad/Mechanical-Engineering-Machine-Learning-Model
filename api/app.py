"""
FastAPI Server — Acoustic Anomaly Detection (DCASE 2024 Task 2)

Endpoints:
  POST /predict           — Upload a .wav file → get anomaly prediction
  GET  /machines          — List supported machine types
  GET  /health            — System health + drift alerts
  GET  /stats             — Detailed per-machine monitoring stats
  GET  /thresholds        — Current thresholds per machine

Usage:
  cd <project_root>
  conda activate ML
  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import json
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import librosa
import joblib
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

from sklearn.neighbors import NearestNeighbors

# Add project root to path so we can find our modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from api.monitor import DriftMonitor

# ==========================================================================
# CONFIG
# ==========================================================================
MODELS_DIR   = os.path.join(PROJECT_ROOT, "weights")
V1_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed_v1")
V2_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed_v2")

PATCH_WIDTH  = 64
PATCH_STRIDE = 32
SAMPLE_RATE  = 16000
N_MELS       = 128
HOP_LENGTH   = 512
N_FFT        = 2048
BATCH_SIZE   = 128

V1_MACHINES = {"ToyCar", "fan", "valve"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================
# MODEL ARCHITECTURES (identical to eval_best.py)
# ==========================================================================
class CNNAutoencoder(nn.Module):
    """V1: 5-layer CNN with Sigmoid output."""
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


class CNNAutoencoderV2(nn.Module):
    """V2: 3-layer CNN with Linear output."""
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


def extract_all_features(model, patches, device_):
    model.eval()
    x = torch.FloatTensor(patches).unsqueeze(1)
    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=False)
    all_mses, all_embs = [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device_)
            recon = model(batch)
            mse = torch.mean((batch - recon)**2, dim=(1,2,3)).cpu().numpy()
            all_mses.append(mse)
            emb = model(batch, return_embedding=True).cpu().numpy()
            all_embs.append(emb)
    return np.concatenate(all_mses), np.concatenate(all_embs)


def compute_file_score(mses, knn_dists, strategy):
    """Compute a single anomaly score for one file."""
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
# APPLICATION STARTUP
# ==========================================================================
app = FastAPI(
    title="Acoustic Anomaly Detection API",
    description="DCASE 2024 Task 2 — Unsupervised anomaly detection for machine sounds",
    version="1.0.0",
)

# Serve the web UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the web UI."""
    return RedirectResponse(url="/static/index.html")

# Monitoring
monitor = DriftMonitor(window_size=100)

# Global state loaded at startup
models = {}
scalers = {}
thresholds_config = {}
knn_models = {}
machine_metadata = {}


@app.on_event("startup")
def load_models():
    """Load all models, scalers, thresholds, and KNN models at startup."""
    global models, scalers, thresholds_config, knn_models, machine_metadata

    # Load thresholds
    thresholds_path = os.path.join(PROJECT_ROOT, "configs", "thresholds.json")
    if not os.path.exists(thresholds_path):
        print(f"WARNING: {thresholds_path} not found. Run compute_thresholds.py first.")
        return

    with open(thresholds_path) as f:
        thresholds_config = json.load(f)

    for machine, config in thresholds_config.items():
        is_v1 = machine in V1_MACHINES
        processed_dir = V1_PROCESSED if is_v1 else V2_PROCESSED

        print(f"Loading {machine} [{config['pipeline']}]...")

        # Load metadata
        meta = torch.load(os.path.join(MODELS_DIR, machine, "metadata.pth"),
                          map_location=device, weights_only=False)
        machine_metadata[machine] = meta

        # Load model
        if is_v1:
            model = CNNAutoencoder().to(device)
        else:
            bn_dim = meta.get('bottleneck_dim', 128)
            model = CNNAutoencoderV2(bottleneck_dim=bn_dim).to(device)

        model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, machine, "best_model.pth"),
            map_location=device, weights_only=False))
        model.eval()
        models[machine] = model

        # Load scaler (check processed dir first, then weights dir for Docker)
        scaler_path = os.path.join(processed_dir, machine, "scaler.save")
        scaler_path_alt = os.path.join(MODELS_DIR, machine, "scaler.save")
        if os.path.exists(scaler_path):
            scalers[machine] = joblib.load(scaler_path)
        elif os.path.exists(scaler_path_alt):
            scalers[machine] = joblib.load(scaler_path_alt)

        # Load KNN if needed for this machine's strategy
        if 'KNN' in config['strategy']:
            knn_path = os.path.join(MODELS_DIR, machine, "knn.save")
            if os.path.exists(knn_path):
                knn_models[machine] = joblib.load(knn_path)
                print(f"  KNN loaded from {knn_path}")
            else:
                # Fit KNN from training data
                train_path = os.path.join(processed_dir, machine, "X_train.npy")
                if os.path.exists(train_path):
                    print(f"  Fitting KNN from training data...")
                    X_train = np.load(train_path)
                    train_patches = np.concatenate([extract_patches(s) for s in X_train])
                    _, train_embs = extract_all_features(model, train_patches, device)
                    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
                    knn.fit(train_embs)
                    knn_models[machine] = knn
                    # Save for next startup
                    joblib.dump(knn, knn_path)
                    print(f"  KNN fitted and saved ({len(train_embs)} embeddings)")

        print(f"  [OK] {machine} ready (strategy={config['strategy']}, "
              f"threshold={config['threshold']:.6f})")

    print(f"\n{'='*50}")
    print(f"All {len(models)} models loaded. Server ready.")
    print(f"{'='*50}")


# ==========================================================================
# ENDPOINTS
# ==========================================================================
@app.get("/machines")
def list_machines():
    """List all supported machine types and their configurations."""
    return {
        "machines": {
            machine: {
                "pipeline": config["pipeline"],
                "strategy": config["strategy"],
                "threshold": config["threshold"],
            }
            for machine, config in thresholds_config.items()
        }
    }


@app.get("/thresholds")
def get_thresholds():
    """Return full threshold configuration per machine."""
    return thresholds_config


@app.get("/health")
def health_check():
    """System health with drift alerts."""
    return monitor.get_health()


@app.get("/stats")
def monitoring_stats():
    """Detailed per-machine monitoring statistics."""
    return monitor.get_stats()


@app.post("/predict")
async def predict(
    audio: UploadFile = File(..., description="A .wav audio file"),
    machine_type: str = Query(..., description="Machine type (e.g., 'fan', 'ToyCar')"),
):
    """
    Upload a .wav file and get an anomaly prediction.

    Returns:
      - anomaly_score: The raw anomaly score
      - threshold: The calibrated threshold for this machine
      - is_anomaly: True if score > threshold
      - confidence: How far the score is from the threshold (in std units)
      - strategy: Which scoring method was used
    """
    t_start = time.time()

    # Validate machine type
    if machine_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown machine type '{machine_type}'. "
                   f"Supported: {list(models.keys())}"
        )

    # Save uploaded file to temp
    suffix = os.path.splitext(audio.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and preprocess audio
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)

        is_v1 = machine_type in V1_MACHINES
        config = thresholds_config[machine_type]
        model = models[machine_type]
        scaler = scalers[machine_type]
        meta = machine_metadata[machine_type]
        n_frames = meta['n_frames']

        # Compute spectrogram with correct ref
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        if is_v1:
            log_mel = librosa.power_to_db(mel, ref=1.0)
        else:
            log_mel = librosa.power_to_db(mel, ref=np.max)

        # Frame alignment
        if log_mel.shape[1] > n_frames:
            log_mel = log_mel[:, :n_frames]
        elif log_mel.shape[1] < n_frames:
            pw = n_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0,0),(0,pw)), constant_values=log_mel.min())

        # Scale
        scaled = scaler.transform(log_mel.T).T
        spec_mean = float(scaled.mean())

        # Extract patches and compute features
        patches = extract_patches(scaled)
        if len(patches) == 0:
            raise HTTPException(status_code=400, detail="Audio too short to extract patches")

        mses, embs = extract_all_features(model, patches, device)

        # Compute KNN distances if needed
        knn_dists = None
        if 'KNN' in config['strategy'] and machine_type in knn_models:
            knn_dists = np.mean(knn_models[machine_type].kneighbors(embs)[0], axis=1)

        # Compute final score
        score = compute_file_score(mses, knn_dists, config['strategy'])
        threshold = config['threshold']
        is_anomaly = score > threshold

        # Confidence: how many std units the score is from the threshold
        score_stats = config['score_stats']
        std = score_stats['std']
        if std > 0:
            confidence = abs(score - threshold) / std
        else:
            confidence = 0.0

        latency = time.time() - t_start

        # Record for monitoring
        monitor.record(machine_type, score, is_anomaly, latency, spec_mean)

        return {
            "machine_type": machine_type,
            "anomaly_score": round(score, 8),
            "threshold": round(threshold, 8),
            "is_anomaly": is_anomaly,
            "decision": "ANOMALY" if is_anomaly else "NORMAL",
            "confidence": round(confidence, 4),
            "strategy": config['strategy'],
            "pipeline": config['pipeline'],
            "latency_ms": round(latency * 1000, 1),
        }

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
