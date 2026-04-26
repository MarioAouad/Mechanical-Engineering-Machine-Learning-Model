# Acoustic Anomaly Detection for Machine Condition Monitoring

> **DCASE 2024 Challenge — Task 2: First-Shot Unsupervised Anomalous Sound Detection**

[![Docker](https://img.shields.io/badge/Docker-Available-2496ED?logo=docker&logoColor=white)](#-quick-start-with-docker)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#prerequisites)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](#api-endpoints)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#model-architectures)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

An end-to-end machine learning system that detects anomalous sounds in industrial machines — fans, valves, gearboxes, bearings, toy cars, toy trains, and sliders — by learning **normal** acoustic patterns and flagging deviations. The system is trained entirely on unlabeled normal recordings (unsupervised) and ships as a containerized REST API with a web interface, real-time inference, and production drift monitoring.

**Author:** Mario Aouad  
**Course:** Mechanical Engineering — Machine Learning  
**Date:** April 2026

---

## Table of Contents

- [Quick Start with Docker](#-quick-start-with-docker)
- [Web Interface Guide](#-web-interface-guide)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Setup & Installation (From Source)](#-setup--installation-from-source)
- [Full Pipeline Walkthrough](#-full-pipeline-walkthrough)
- [API Endpoints](#-api-endpoints)
- [Monitoring & Drift Detection](#-monitoring--drift-detection)
- [Results](#-results)
- [Key Design Decisions](#-key-design-decisions)
- [Troubleshooting](#-troubleshooting)

---

## 🐳 Quick Start with Docker

The fastest way to run the system. **No Python, no dependencies, no setup** — just Docker.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Option A: Pull from Docker Hub (Recommended)

```bash
docker pull elchamandre/acoustic-anomaly-api:latest
docker run -p 8000:8000 elchamandre/acoustic-anomaly-api:latest
```

### Option B: Build Locally

```bash
# Clone the repository
git clone https://github.com/MarioAouad/Mechanical-Engineering-Machine-Learning-Model.git
cd Mechanical-Engineering-Machine-Learning-Model

# Build the Docker image
docker build -t acoustic-anomaly-api .

# Run the container
docker run -p 8000:8000 acoustic-anomaly-api
```

### What Happens Next

1. The container starts and loads all **7 machine models** into memory (takes ~15–30 seconds)
2. You will see output confirming each model is loaded:
   ```
   Loading ToyCar [V1]...
     [OK] ToyCar ready (strategy=Recon_Max, threshold=0.001978)
   Loading ToyTrain [V2]...
     [OK] ToyTrain ready (strategy=Recon_Max, threshold=0.182543)
   ...
   ==================================================
   All 7 models loaded. Server ready.
   ==================================================
   ```
3. Open your browser and navigate to **http://localhost:8000**
4. The web interface loads automatically — you are ready to analyze audio files

### Stopping the Container

Press `Ctrl+C` in the terminal, or run:

```bash
docker ps                          # Find the container ID
docker stop <container_id>         # Stop it
```

---

## 🖥 Web Interface Guide

The application includes a built-in web UI accessible at **http://localhost:8000** after starting the server.

### Step-by-Step Usage

#### 1. Select a Machine Type

When the page loads, you will see 7 machine buttons arranged in a grid. Each button shows:
- The machine name (e.g., **Fan**, **Bearing**, **Valve**)
- The scoring pipeline and strategy used for that machine (e.g., `V1 · Recon_Mean`)

**Click the button** corresponding to the machine type your audio was recorded from.

#### 2. Upload a `.wav` Audio File

After selecting a machine:
- **Drag and drop** a `.wav` file onto the upload zone, or
- **Click** the upload zone to open a file picker

The file name and size will appear below the upload zone.

> **Note:** The system expects `.wav` audio files. Other formats (`.mp3`, `.flac`, etc.) are not supported.

#### 3. View the Prediction Result

Click **"Analyze Audio"**. The system will:
1. Convert the audio to a Log-Mel spectrogram
2. Extract overlapping patches
3. Run them through the trained autoencoder
4. Compute the anomaly score using the machine's optimal strategy
5. Compare the score against the calibrated threshold

The result card displays:
- **NORMAL** (green) or **ANOMALY** (red) — the final decision
- **Anomaly Score** — the raw numeric score
- **Threshold** — the calibrated boundary for this machine
- **Confidence** — how far the score is from the threshold (in σ units)
- **Strategy** — which scoring method was used
- **Pipeline** — V1 or V2
- **Latency** — inference time in milliseconds

#### 4. Health Dashboard

Click the **"Health"** button in the header to expand the real-time health dashboard. It shows:
- **Status** — Healthy (green) or Degraded (amber)
- **Uptime** — how long the server has been running
- **Total Requests** — cumulative prediction count
- **Anomaly Rate** — percentage of predictions flagged as anomalous
- **Machines Active** — how many machine types have been queried
- **Alerts** — any active drift alerts (see [Monitoring](#-monitoring--drift-detection))

The dashboard polls the `/health` endpoint every 5 seconds and updates automatically.

#### 5. Start a New Analysis

Click **"← Start New Analysis"** to reset and analyze a different file or machine type.

---

## 🧠 How It Works

### Core Concept

Industrial machines produce consistent sound patterns when operating normally. A failing bearing creates a distinctive grinding noise; a clogged fan develops a wobble frequency. These acoustic "fingerprints" deviate from the normal baseline.

We exploit this by training a **Convolutional Autoencoder** — a neural network that compresses and reconstructs spectrogram images — exclusively on normal sounds.

| Input Type | Reconstruction Quality | Anomaly Score |
|------------|----------------------|---------------|
| **Normal** sound | High (model knows this pattern) | Low |
| **Anomalous** sound | Low (unfamiliar pattern) | High |

### Pipeline Overview

```
Phase 1: PREPROCESS        Phase 2: TRAIN             Phase 3: EVALUATE          Phase 4: DEPLOY
───────────────────        ──────────────             ─────────────────          ───────────────
.wav audio files     ───►  Train CNN Autoencoder ───► Score test files      ───► FastAPI + Web UI
Log-Mel Spectrograms       on normal patches only     Find best strategy         with thresholds
MinMax/StandardScaler      Early stopping             per machine type           and monitoring
```

### What Makes This Challenging

| Challenge | Description |
|-----------|-------------|
| **No labeled anomalies** | Training uses only "normal" sounds — the model never sees what an anomaly looks like |
| **Domain shift** | DCASE provides ~990 "source" recordings and only ~10 "target" recordings from a different environment |
| **7 machine types** | Each machine has unique acoustic characteristics requiring its own model, scaler, and strategy |
| **Threshold calibration** | Must be computed on development data only — never on test data — to avoid overfitting the evaluation |

---

## 📁 Project Structure

```
Mechanical-Engineering-Machine-Learning-Model/
│
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build instructions
├── .dockerignore                    # Excludes large files from Docker context
├── .gitignore                       # Excludes large files from Git
│
├── configs/
│   └── thresholds.json              # Per-machine anomaly thresholds + scoring config
│
├── api/                             # Deployment layer
│   ├── app.py                       # FastAPI server (/predict, /health, /stats, /machines)
│   ├── monitor.py                   # Drift monitoring (score, feature, decision drift)
│   └── static/                      # Web interface
│       ├── index.html               # Main HTML page
│       ├── style.css                # Dark glassmorphism theme
│       └── app.js                   # Client-side logic + health dashboard
│
├── src/                             # ML pipeline source code
│   ├── preprocessing/
│   │   ├── preprocess_v1.py         # Pipeline V1: MinMaxScaler, ref=1.0 (absolute dB)
│   │   └── preprocess_v2.py         # Pipeline V2: StandardScaler, ref=np.max (relative dB)
│   ├── training/
│   │   ├── train_v1.py              # Train 5-layer CNN (Sigmoid output)
│   │   ├── train_v2.py              # Train 3-layer CNN (Linear output)
│   │   ├── optimize_v1.py           # Hyperparameter search for V1
│   │   └── retrain_valve.py         # Valve-specific retraining
│   └── evaluation/
│       ├── evaluate.py              # Unified evaluation (7 strategies per machine)
│       ├── optimize_scoring.py      # V2-only scoring optimization
│       └── calibrate_thresholds.py  # Threshold computation on dev data only
│
├── weights/                         # Trained model checkpoints (per machine)
│   └── <machine>/
│       ├── best_model.pth           # Best model (selected by validation loss)
│       ├── metadata.pth             # Training metadata (n_frames, bottleneck_dim)
│       ├── scaler.save              # Fitted scaler (for Docker — no raw data needed)
│       └── knn.save                 # KNN model (bearing only)
│
├── data/                            # Audio data (not included in Docker/Git)
│   ├── raw/                         # DCASE 2024 .wav files
│   ├── processed_v1/               # V1 spectrograms + scalers
│   └── processed_v2/               # V2 spectrograms + scalers
│
├── logs/predictions/                # JSONL prediction logs (one file per day)
└── outputs/                         # Training curves, ROC plots, score distributions
```

---

## 🛠 Setup & Installation (From Source)

> **Note:** This section is only needed if you want to retrain models or modify the pipeline. To just run the API, use the [Docker instructions](#-quick-start-with-docker).

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended for training; CPU works for inference)
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MarioAouad/Mechanical-Engineering-Machine-Learning-Model.git
cd Mechanical-Engineering-Machine-Learning-Model

# Create conda environment
conda create -n ML python=3.10
conda activate ML

# Install PyTorch with CUDA (adjust cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### Download the DCASE 2024 Dataset

Download the development dataset from the [DCASE 2024 Task 2 website](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) and extract into `data/raw/`:

```
data/raw/
├── ToyCar/
│   ├── train/     (normal .wav files)
│   └── test/      (normal + anomaly .wav files)
├── ToyTrain/
├── bearing/
├── fan/
├── gearbox/
├── slider/
└── valve/
```

---

## 🔄 Full Pipeline Walkthrough

### Step 1: Preprocess Audio → Spectrograms

```bash
# Pipeline V1: MinMaxScaler, absolute dB (ref=1.0)
python src/preprocessing/preprocess_v1.py

# Pipeline V2: StandardScaler, relative dB (ref=np.max)
python src/preprocessing/preprocess_v2.py
```

Creates `data/processed_v1/` and `data/processed_v2/` with `.npy` spectrogram arrays and fitted scalers.

### Step 2: Train Autoencoders

```bash
# V1 models (5-layer CNN — used for ToyCar, fan, valve)
python src/training/train_v1.py

# V2 models (3-layer CNN — used for ToyTrain, bearing, gearbox, slider)
python src/training/train_v2.py
```

Models are saved to `weights/<machine>/best_model.pth` with early stopping on validation loss.

### Step 3: Evaluate & Find Best Strategy

```bash
python src/evaluation/evaluate.py
```

Tests 7 scoring strategies per machine and reports AUC/pAUC for each. The best strategy per machine is selected for deployment.

### Step 4: Calibrate Thresholds

```bash
python src/evaluation/calibrate_thresholds.py
```

Computes anomaly thresholds using **only** the development (training) data. Saves the full configuration to `configs/thresholds.json`.

### Step 5: Deploy

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Server is live at http://localhost:8000. Swagger API docs available at http://localhost:8000/docs.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to the web interface |
| `/predict` | POST | Upload a `.wav` file + machine type → anomaly prediction |
| `/machines` | GET | List all supported machine types and their configurations |
| `/thresholds` | GET | Return the full threshold configuration JSON |
| `/health` | GET | System health summary with uptime, request counts, and drift alerts |
| `/stats` | GET | Detailed per-machine monitoring statistics |

### Example: Predict via cURL

```bash
curl -X POST "http://localhost:8000/predict?machine_type=fan" \
  -F "audio=@path/to/your/audio.wav"
```

### Example Response

```json
{
  "machine_type": "fan",
  "anomaly_score": 0.00512345,
  "threshold": 0.00345999,
  "is_anomaly": true,
  "decision": "ANOMALY",
  "confidence": 1.41,
  "strategy": "Recon_Mean",
  "pipeline": "V1",
  "latency_ms": 245.3
}
```

### Example: Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "uptime_seconds": 3621.4,
  "total_requests": 42,
  "total_anomalies": 5,
  "anomaly_rate": 0.119,
  "machines_served": ["fan", "valve", "bearing"],
  "active_alerts": []
}
```

---

## 📊 Monitoring & Drift Detection

The built-in monitoring system (`api/monitor.py`) continuously tracks three categories of production drift:

| Drift Type | What It Detects | Alert Trigger |
|------------|----------------|---------------|
| **Score drift** | Anomaly score distribution shifting over time | — |
| **Feature drift** | Spectrogram pixel values deviating from baseline | Mean deviates > 3σ from rolling window |
| **Decision drift** | Anomaly rate spiking | > 30% of recent clips flagged |

### Alert Types

| Alert | Meaning |
|-------|---------|
| `HIGH_ANOMALY_RATE` | More than 30% of recent clips flagged as anomalous |
| `COLLAPSED_SCORES` | All scores are identical — model may be stuck |
| `FEATURE_DRIFT` | Input audio characteristics have changed significantly |

All predictions are logged to `logs/predictions/predictions_YYYY-MM-DD.jsonl` for offline analysis.

---

## 📈 Results

### Best Performance Per Machine

| Machine | Pipeline | Strategy | AUC | pAUC |
|---------|----------|----------|-----|------|
| ToyCar | V1 | Recon_Max | 0.5545 | 0.5005 |
| ToyTrain | V2 | Recon_Max | 0.6232 | 0.5084 |
| bearing | V2 | KNN_Mean | 0.5869 | 0.5347 |
| fan | V1 | Recon_Mean | 0.6131 | 0.5589 |
| gearbox | V2 | Recon_Mean | 0.5996 | 0.5258 |
| slider | V2 | Recon_Mean | 0.6235 | 0.5074 |
| valve | V1 | Neg_Recon | 0.5961 | 0.5405 |
| **Average** | — | — | **0.5996** | **0.5252** |

### Model Architectures

| | V1 (5-Layer CNN) | V2 (3-Layer CNN) |
|---|---|---|
| **Used for** | ToyCar, fan, valve | ToyTrain, bearing, gearbox, slider |
| **Encoder** | 5 conv layers → FC(64) | 3 conv layers → FC(128) |
| **Output** | Sigmoid (range [0, 1]) | Linear (unbounded) |
| **Scaler** | MinMaxScaler (per Mel band) | StandardScaler (per Mel band) |
| **dB reference** | `ref=1.0` (absolute) | `ref=np.max` (relative) |
| **Activation** | LeakyReLU + Dropout2d | ReLU |

### Scoring Strategies

| Strategy | Formula | Best For |
|----------|---------|----------|
| `Recon_Mean` | Mean of patch MSEs | Most machines (general-purpose) |
| `Recon_Max` | Max of patch MSEs | Localized faults (ToyCar, ToyTrain) |
| `KNN_Mean` | Mean KNN distance in embedding space | When reconstruction error alone is insufficient (bearing) |
| `Neg_Recon` | −1 × Mean MSE | When anomalies are reconstructed *better* than normal (valve) |

---

## 🔑 Key Design Decisions

### Why Two Pipelines?

Different machines respond differently to preprocessing:
- **V1** (absolute dB + MinMaxScaler + Sigmoid) — works better when volume carries diagnostic information (ToyCar, fan, valve)
- **V2** (relative dB + StandardScaler + Linear) — works better when spectral shape matters more than volume (ToyTrain, bearing, gearbox, slider)

### Why Per-Mel-Band Scaling?

Both pipelines scale per Mel band (128 features), not per pixel (128 × T). This preserves **time-shift invariance** — a spike at t=10 is treated identically to a spike at t=50. Per-pixel scaling would cause the model to memorize positions instead of acoustic patterns.

### Why Neg_Recon for Valve?

Valve anomalies produce simpler, more repetitive sounds that the autoencoder actually reconstructs *better* than complex normal sounds. Negating the error flips the direction so that better reconstruction = higher anomaly score.

### Why Threshold on Dev Data Only?

A classic mistake in anomaly detection is tuning thresholds on the test set, which inflates results. Our thresholds are computed at the 85th percentile of normal training scores — ensuring an honest, deployment-safe calibration.

### Why Docker?

The container packages all models, weights, configs, and the web UI into a single portable image. An evaluator can run the system with one command (`docker run`) without installing Python, PyTorch, or any dependencies.

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port 8000 already in use** | Stop the existing process or use a different port: `docker run -p 9000:8000 ...` then visit http://localhost:9000 |
| **Container exits immediately** | Check logs: `docker logs <container_id>`. Usually a missing file or dependency issue. |
| **"Unknown machine type" error** | Use one of: `ToyCar`, `ToyTrain`, `bearing`, `fan`, `gearbox`, `slider`, `valve` (case-sensitive) |
| **Slow first prediction** | Normal — the first inference loads model layers into memory. Subsequent predictions are faster. |
| **scikit-learn version warning** | Can be safely ignored. The serialized scalers may show a version mismatch warning but function correctly. |
| **Web UI not loading** | Ensure you are visiting `http://localhost:8000` (not `https://`). Check that port 8000 is mapped correctly. |

---

## License

This project was developed as an academic deliverable for the Mechanical Engineering Machine Learning course. The DCASE 2024 dataset is provided under its own license — see the [DCASE website](https://dcase.community/challenge2024/) for details.
