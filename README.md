# Acoustic Anomaly Detection for Machine Condition Monitoring

> **DCASE 2024 Challenge - Task 2: First-Shot Unsupervised Anomalous Sound Detection**

An end-to-end machine learning pipeline that detects anomalous sounds in industrial machines (fans, valves, gearboxes, bearings, etc.) by learning what **normal** sounds look like and flagging anything that deviates. The system is trained entirely on normal machine recordings (unsupervised) and deployed as a production-ready REST API with real-time drift monitoring.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Architecture](#project-architecture)
3. [Detailed File Reference](#detailed-file-reference)
4. [Setup & Installation](#setup--installation)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
6. [Running the API Server](#running-the-api-server)
7. [Docker Deployment](#docker-deployment)
8. [Monitoring & Drift Detection](#monitoring--drift-detection)
9. [Best Results Per Machine](#best-results-per-machine)
10. [Key Design Decisions](#key-design-decisions)

---

## How It Works

### The Core Idea

Industrial machines produce consistent sound patterns when operating normally. A failing bearing creates a distinctive grinding noise. A clogged fan develops a wobble frequency. These acoustic "fingerprints" deviate from the normal baseline.

We exploit this by training a **Convolutional Autoencoder** — a neural network that compresses and reconstructs spectrogram images — exclusively on **normal** sounds. After training:

- **Normal sounds**: The model reconstructs them accurately (low reconstruction error)
- **Anomalous sounds**: The model fails to reconstruct unfamiliar patterns (high reconstruction error)

The reconstruction error becomes our **anomaly score**.

### The Pipeline (4 Phases)

```
Phase 1: PREPROCESS        Phase 2: TRAIN           Phase 3: EVALUATE         Phase 4: DEPLOY
.wav audio files     -->   Train CNN Autoencoder --> Score test files      --> FastAPI server
Log-Mel Spectrograms       on normal patches only   Find best strategy        with thresholds
MinMax/StandardScaler      Early stopping            per machine type          and monitoring
```

### What Makes This Challenging

1. **No labeled anomalies for training** — We only have "normal" sounds during training
2. **Domain shift** — The DCASE dataset has ~990 "source" files and only ~10 "target" files from a different recording environment
3. **7 different machine types** — Each machine has different acoustic characteristics and needs its own model, scaler, and scoring strategy
4. **Threshold calibration** — Must be done on development data only, never on test data (to avoid overfitting the evaluation)

---

## Project Architecture

```
Mechanical-Engineering-Machine-Learning-Model/
│
├── README.md                     # This file — complete project documentation
├── requirements.txt              # Python dependencies with explanations
├── .gitignore                    # Excludes large binary files from Git
│
├── configs/                      # Configuration files
│   └── thresholds.json           # Per-machine anomaly thresholds (computed on dev data only)
│
├── data/                         # All data (gitignored — regenerate locally)
│   ├── raw/                      # DCASE 2024 .wav files (download separately)
│   │   ├── ToyCar/
│   │   │   ├── train/            # ~1000 normal .wav files
│   │   │   └── test/             # ~200 .wav files (mix of normal + anomaly)
│   │   ├── ToyTrain/
│   │   ├── bearing/
│   │   ├── fan/
│   │   ├── gearbox/
│   │   ├── slider/
│   │   └── valve/
│   ├── processed_v1/             # Spectrograms scaled with MinMaxScaler + ref=1.0
│   │   └── <machine>/
│   │       ├── X_train.npy       # Training spectrograms (shape: [N, 128, T])
│   │       ├── X_val.npy         # Validation spectrograms (15% hold-out)
│   │       └── scaler.save       # Fitted MinMaxScaler (joblib serialized)
│   └── processed_v2/             # Spectrograms scaled with StandardScaler + ref=np.max
│       └── <machine>/
│           ├── X_train.npy
│           ├── X_val.npy
│           └── scaler.save       # Fitted StandardScaler (joblib serialized)
│
├── src/                          # Source code — all pipeline logic
│   ├── __init__.py
│   ├── preprocessing/            # Phase 1: Audio → Spectrogram conversion
│   │   ├── __init__.py
│   │   ├── preprocess_v1.py      # Pipeline V1: MinMaxScaler, ref=1.0 (absolute dB)
│   │   └── preprocess_v2.py      # Pipeline V2: StandardScaler, ref=np.max (relative dB)
│   ├── training/                 # Phase 2: Autoencoder training
│   │   ├── __init__.py
│   │   ├── train_v1.py           # Train 5-layer CNN with Sigmoid output
│   │   ├── train_v2.py           # Train 3-layer CNN with Linear output
│   │   ├── optimize_v1.py        # Hyperparameter search for V1 architecture
│   │   └── retrain_valve.py      # Valve-specific retraining (uses V2 with bottleneck=128)
│   └── evaluation/               # Phase 3: Scoring & threshold calibration
│       ├── __init__.py
│       ├── evaluate.py           # Unified evaluation — tests all strategies per machine
│       ├── optimize_scoring.py   # V2-only scoring optimization (7 strategies)
│       └── calibrate_thresholds.py  # Compute thresholds on dev data only (NEVER test data)
│
├── api/                          # Phase 4: Production deployment
│   ├── __init__.py
│   ├── app.py                    # FastAPI server with /predict, /health, /stats endpoints
│   └── monitor.py                # Drift monitoring — tracks score distributions & anomaly rates
│
├── weights/                      # Trained model checkpoints (gitignored)
│   └── <machine>/
│       ├── best_model.pth        # Best PyTorch model state dict (selected by val loss)
│       ├── metadata.pth          # Training metadata (n_frames, best_val, bottleneck_dim)
│       ├── knn.save              # (V1 only) Fitted KNN model for embedding scoring
│       └── gmm.save              # (V1 only) Fitted GMM model (optional scoring)
│
├── logs/                         # Runtime logs
│   └── predictions/              # JSONL prediction logs from API (one file per day)
│       └── predictions_YYYY-MM-DD.jsonl
│
└── outputs/                      # Generated visualizations from experiments
    ├── training_curves_*.png     # Loss curves from training
    ├── roc_curves_*.png          # ROC curves from evaluation
    └── score_distributions_*.png # Normal vs anomaly score histograms
```

---

## Detailed File Reference

### `src/preprocessing/preprocess_v1.py`
**What it does:** Converts raw `.wav` audio into scaled Log-Mel Spectrograms using Pipeline V1.

**Key choices:**
- `ref=1.0` — Preserves **absolute** dB levels across files. A louder machine stays louder in the spectrogram. This is important for machines like ToyCar and fan where volume differences carry diagnostic information.
- `MinMaxScaler(0, 1)` — Scales all spectrogram pixels to [0, 1]. This matches the Sigmoid activation in the V1 autoencoder output layer. Without this, the model literally cannot reconstruct negative dB values.
- **Domain shift handling** — Oversamples the ~10 "target" domain files to match the ~990 "source" files, so the autoencoder learns both recording environments equally.

**Run:** `conda run -n ML python src/preprocessing/preprocess_v1.py`

---

### `src/preprocessing/preprocess_v2.py`
**What it does:** Converts raw `.wav` audio into scaled Log-Mel Spectrograms using Pipeline V2.

**Key choices:**
- `ref=np.max` — Normalizes each file's max power to 0 dB. This removes volume differences and captures the **spectral shape** (relative energy distribution across frequencies). Better for machines like gearbox and slider where the fault signature is a change in frequency pattern, not volume.
- `StandardScaler` — Z-score normalization per Mel band. Instead of hard-clipping to [0, 1], this centers each frequency band to mean=0, std=1. This works with the V2 autoencoder's linear output layer (no activation = no output range constraint).
- **Circular time shifts** — Instead of simply duplicating target files, each oversampled copy is randomly time-shifted. This creates genuinely different training examples while preserving spectral content.

**Run:** `conda run -n ML python src/preprocessing/preprocess_v2.py`

---

### `src/training/train_v1.py`
**What it does:** Trains the V1 CNN Autoencoder architecture.

**Architecture (5-layer encoder):**
```
Input (1, 128, 64) → Conv(16) → Conv(32) → Conv(64) → Conv(128) → Conv(128)
→ Flatten → FC(64) → FC(1024) → Unflatten
→ DeConv(128) → DeConv(64) → DeConv(32) → DeConv(16) → DeConv(1) + Sigmoid
```
- **Bottleneck**: 64 dimensions (forces extreme compression)
- **Dropout2d**: 0.2 (prevents overfitting to specific frequency patterns)
- **LeakyReLU**: 0.2 negative slope (avoids dead neurons in the encoder)
- **Sigmoid output**: Output range [0, 1], matching MinMaxScaler input range

---

### `src/training/train_v2.py`
**What it does:** Trains the V2 CNN Autoencoder architecture.

**Architecture (3-layer encoder):**
```
Input (1, 128, 64) → Conv(32) → Conv(64) → Conv(128)
→ Flatten → FC(128) → FC(128*16*8) → Unflatten
→ DeConv(64) → DeConv(32) → DeConv(1) [Linear output]
```
- **Bottleneck**: 128 dimensions (wider for z-scored data which has more variance)
- **Linear output**: No Sigmoid — can reconstruct z-scored values (including negatives)
- **ReLU**: Standard activation (works better with StandardScaler data)

---

### `src/evaluation/evaluate.py`
**What it does:** The unified evaluation script. Loads the best model per machine (V1 or V2), tests **7 scoring strategies**, and picks the best one.

**Scoring strategies tested:**
| Strategy | Formula | When It Works Best |
|----------|---------|-------------------|
| `Recon_Mean` | Mean of patch MSEs | Most machines (general-purpose) |
| `Recon_Max` | Max of patch MSEs | Machines with localized faults (ToyCar, ToyTrain) |
| `Recon_P90` | 90th percentile MSE | Robust to outlier patches |
| `KNN_Mean` | Mean KNN distance in embedding space | When reconstruction error alone is insufficient (bearing) |
| `KNN_Max` | Max KNN distance | Extreme deviation detection |
| `Neg_Recon` | -1 × Mean MSE | When anomalies are "simpler" and reconstructed BETTER (valve) |
| `Hybrid_Mean` | Z-scored Recon + KNN | Combined evidence |

**Why Neg_Recon for valve:** Valve anomalies produce simpler, more repetitive sounds that the autoencoder actually reconstructs *better* than complex normal sounds. Negating the error flips the direction so that better reconstruction = higher anomaly score.

---

### `src/evaluation/calibrate_thresholds.py`
**What it does:** Computes the decision threshold for each machine using **ONLY development data** (training set).

**Why this matters:** A classic mistake in anomaly detection is tuning thresholds on the test set. This artificially inflates results because you're fitting your decision boundary to data you're supposed to be evaluating against. By using only training data (all normal sounds), we get an honest estimate of the threshold.

**Method:** Threshold = 95th percentile of normal training scores. This means ~5% of known-normal sounds would be flagged as anomalous (false positive rate), which is a conservative, deployment-safe calibration.

**Run:** `conda run -n ML python src/evaluation/calibrate_thresholds.py`

**Output:** `configs/thresholds.json`

---

### `api/app.py`
**What it does:** A FastAPI web server that wraps the entire pipeline into a REST API.

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload a `.wav` file + machine type → anomaly prediction |
| `/machines` | GET | List all supported machine types and their scoring configs |
| `/thresholds` | GET | Return the full threshold JSON |
| `/health` | GET | System health summary with drift alerts |
| `/stats` | GET | Detailed per-machine monitoring statistics |

**Startup behavior:** Loads all 7 models, scalers, and thresholds into GPU memory. For machines using KNN scoring (bearing), it either loads a cached KNN model or fits one from training data.

---

### `api/monitor.py`
**What it does:** Tracks production inference statistics in real-time to detect **data drift** and **model degradation**.

**What it monitors:**
- **Score distribution** — Are anomaly scores shifting? (mean, std, min, max over a rolling window)
- **Anomaly rate** — Is the system suddenly flagging >30% of clips? (alert threshold)
- **Feature drift** — Are spectrogram pixel values changing? (3-sigma deviation alert)
- **Latency** — Is inference slowing down?

**Alerts generated:**
- `HIGH_ANOMALY_RATE` — More than 30% of recent clips flagged
- `COLLAPSED_SCORES` — All scores identical (model may be dead/stuck)
- `FEATURE_DRIFT` — Current spectrogram statistics deviate >3 standard deviations from baseline

**Logging:** Every prediction is appended to `logs/predictions/predictions_YYYY-MM-DD.jsonl` for offline analysis.

---

### `configs/thresholds.json`
**What it contains:** Per-machine configuration including:
- `pipeline` — Which model version (V1 or V2)
- `strategy` — Which scoring method (Recon_Max, KNN_Mean, etc.)
- `threshold` — The calibrated decision boundary
- `score_stats` — Distribution statistics of normal training scores (mean, std, percentiles)

This file is the bridge between training and deployment. The API server reads it at startup to know how to score and threshold each machine type.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (for training; CPU works for inference)
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MarioAouad/Mechanical-Engineering-Machine-Learning-Model.git
cd Mechanical-Engineering-Machine-Learning-Model

# Create conda environment
conda create -n ML python=3.10
conda activate ML

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Download the DCASE 2024 Dataset
Download the development dataset from the [DCASE 2024 Task 2 website](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) and extract it into `data/raw/`:
```
data/raw/
├── ToyCar/
│   ├── train/    (normal .wav files)
│   └── test/     (normal + anomaly .wav files)
├── fan/
│   ├── train/
│   └── test/
└── ... (7 machine types total)
```

---

## Pipeline Walkthrough

### Step 1: Preprocess Audio
```bash
# Generate V1 spectrograms (MinMaxScaler, ref=1.0)
conda run -n ML python src/preprocessing/preprocess_v1.py

# Generate V2 spectrograms (StandardScaler, ref=np.max)
conda run -n ML python src/preprocessing/preprocess_v2.py
```
This creates `data/processed_v1/` and `data/processed_v2/` with `.npy` spectrograms and fitted scalers.

### Step 2: Train Models
```bash
# Train V1 models (5-layer CNN, used for ToyCar, fan, valve)
conda run -n ML python src/training/train_v1.py

# Train V2 models (3-layer CNN, used for ToyTrain, bearing, gearbox, slider)
conda run -n ML python src/training/train_v2.py
```
Models are saved to `weights/<machine>/best_model.pth`.

### Step 3: Evaluate & Find Best Strategy
```bash
conda run -n ML python src/evaluation/evaluate.py
```
Tests all 7 scoring strategies per machine and prints the best one.

### Step 4: Calibrate Thresholds
```bash
conda run -n ML python src/evaluation/calibrate_thresholds.py
```
Computes thresholds on development data only. Saves to `configs/thresholds.json`.

### Step 5: Deploy
```bash
conda run -n ML uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Server is live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Running the API Server

### Start the Server
```bash
conda activate ML
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict?machine_type=fan" \
  -F "audio=@data/raw/fan/test/section_00_source_test_anomaly_0001.wav"
```

### Response Example
```json
{
  "machine_type": "fan",
  "anomaly_score": 0.00512345,
  "threshold": 0.004055,
  "is_anomaly": true,
  "decision": "ANOMALY",
  "confidence": 1.25,
  "strategy": "Recon_Mean",
  "pipeline": "V1",
  "latency_ms": 245.3
}
```

### Check System Health
```bash
curl http://localhost:8000/health
```

---

## Docker Deployment

You can run the entire API server using Docker. This ensures a consistent environment regardless of your host operating system.

### Option 1: Using a Pre-built Image from Docker Hub (Recommended)
If the image is published to Docker Hub, you can pull and run it directly. *Note: You must still mount your local weights and data directories so the container can access the models and scalers.*

```bash
# Pull the image
docker pull yourusername/acoustic-anomaly-api:latest

# Run the container (mounting required directories)
docker run -d -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/logs:/app/logs \
  yourusername/acoustic-anomaly-api:latest
```

### Option 2: Building the Image Locally
You can build the Docker image yourself using the provided `Dockerfile`.

```bash
# Build the image locally
docker build -t acoustic-anomaly-api:local .

# Run the local image
docker run -d -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/logs:/app/logs \
  acoustic-anomaly-api:local
```

### Publishing to Docker Hub
To publish your own image to Docker Hub:
```bash
# 1. Login to Docker Hub
docker login

# 2. Tag your local image with your Docker Hub username
docker tag acoustic-anomaly-api:local yourusername/acoustic-anomaly-api:v1.0

# 3. Push the image
docker push yourusername/acoustic-anomaly-api:v1.0
```

---

## Monitoring & Drift Detection

The monitoring system (`api/monitor.py`) tracks three categories of drift:

### 1. Score Drift
If the anomaly score distribution shifts (e.g., mean score increases over time), it may indicate that the machine's operating conditions have changed and the model is becoming stale.

### 2. Feature Drift
If the average spectrogram pixel values change significantly (>3 standard deviations from the rolling window), it suggests the input audio characteristics are different from what the model was trained on.

### 3. Decision Drift
If the anomaly rate suddenly spikes above 30%, either:
- Something is genuinely wrong with the machines (real anomalies)
- The input data has shifted enough to invalidate the model (requires retraining)

All predictions are logged to `logs/predictions/` for offline post-hoc analysis.

---

## Best Results Per Machine

| Machine | Pipeline | Scoring Strategy | AUC | pAUC |
|---------|----------|-----------------|-----|------|
| ToyCar | V1 | Recon_Max | 0.5545 | 0.5005 |
| ToyTrain | V2 | Recon_Max | 0.6232 | 0.5084 |
| bearing | V2 | KNN_Mean | 0.5869 | 0.5347 |
| fan | V1 | Recon_Mean | 0.6131 | 0.5589 |
| gearbox | V2 | Recon_Mean | 0.5996 | 0.5258 |
| slider | V2 | Recon_Mean | 0.6235 | 0.5074 |
| valve | V1 | Neg_Recon | 0.5961 | 0.5405 |
| **Average** | — | — | **0.5996** | **0.5252** |

---

## Key Design Decisions

### Why Two Pipelines (V1 vs V2)?
Different machines respond differently to preprocessing and architecture choices:
- **V1** (MinMaxScaler + Sigmoid + 5-layer CNN) works better for machines where absolute volume matters (ToyCar, fan, valve)
- **V2** (StandardScaler + Linear + 3-layer CNN) works better for machines where spectral shape matters more than volume (ToyTrain, bearing, gearbox, slider)

### Why Per-Mel-Band Scaling?
Both V1 and V2 scale per Mel band (128 features), not per pixel (128 x 313 = 40,064 features). This preserves time-shift invariance — a spike at time t=10 is treated the same as a spike at t=50. Per-pixel scaling would destroy this invariance and cause the model to memorize pixel positions instead of acoustic patterns.

### Why Domain Oversampling?
The DCASE dataset has a severe class imbalance: ~990 source-domain recordings vs ~10 target-domain recordings. Without oversampling, the autoencoder would only learn to reconstruct source-domain sounds and fail on target-domain test files (causing low pAUC scores).

### Why Reconstruction Error Instead of Classification?
Since we have no anomaly labels during training, we can't train a classifier. Instead, we train the autoencoder to be an expert at reconstructing normal sounds. Any sound it can't reconstruct well is, by definition, abnormal. This is the fundamental principle of unsupervised anomaly detection.
