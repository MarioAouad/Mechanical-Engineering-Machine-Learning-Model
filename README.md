# Mechanical-Engineering-Machine-Learning-Model

**First-Shot Acoustic Anomaly Detection Under Domain Shift**

---

## Overview

This project implements an **Unsupervised Autoencoder** for detecting anomalies in industrial machine sounds, built around the [DCASE 2024 Challenge Task 2](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) dataset.

### How It Works

1. **Audio → Image**: Raw `.wav` waveforms are converted into 2D **Log-Mel Spectrograms** — a visual "heat-map" of sound that captures frequency, time, and intensity.
2. **Learn Normal**: A Dense Autoencoder (fully-connected `nn.Linear` layers) is trained *only* on normal machine sounds. It learns to compress and reconstruct what "healthy" sounds look like.
3. **Detect Anomalies**: When an abnormal sound is fed in, the Autoencoder fails to reconstruct it accurately. The **Reconstruction Error (MSE)** spikes — this spike *is* the anomaly signal.
4. **Deploy**: A FastAPI server exposes the trained model as a REST API endpoint for real-time inference.

### Supported Machine Types

| Machine Type | Description |
|-------------|-------------|
| `fan` | Industrial cooling fans |
| `gearbox` | Mechanical gearbox assemblies |
| `bearing` | Rotating shaft bearings |
| `slider` | Linear sliding mechanisms |
| `valve` | Solenoid/pneumatic valves |
| `ToyCar` | Miniature vehicle motors |
| `ToyTrain` | Miniature locomotive motors |

---

## Folder Structure

```
Mechanical-Engineering-Machine-Learning-Model/
│
├── .gitignore                  # Ignores audio, model weights, and data files
├── requirements.txt            # Python dependencies
├── README.md                   # ← You are here
│
├── data/
│   ├── raw/                    # DCASE 2024 .wav files (git-ignored)
│   │   ├── .gitkeep            # Preserves folder on GitHub
│   │   ├── fan/
│   │   │   ├── train/          # Normal training audio
│   │   │   └── test/           # Mixed normal + anomalous audio
│   │   ├── bearing/
│   │   ├── gearbox/
│   │   ├── slider/
│   │   ├── valve/
│   │   ├── ToyCar/
│   │   └── ToyTrain/
│   │
│   └── processed/              # Log-Mel Spectrograms as .npy (git-ignored)
│       ├── .gitkeep            # Preserves folder on GitHub
│       ├── fan/
│       │   ├── X_train.npy     # Scaled training spectrograms
│       │   ├── X_val.npy       # Scaled validation spectrograms
│       │   └── scaler.save     # Fitted MinMaxScaler object
│       └── (same structure for each machine type)
│
├── notebooks/
│   ├── 01_data_prep.ipynb      # Phase 1: Audio → Log-Mel Spectrogram pipeline
│   ├── 02_model_training.ipynb # Phase 2: Autoencoder training (upcoming)
│   └── 03_evaluation.ipynb     # Phase 3: Anomaly scoring & metrics (upcoming)
│
└── src/                        # Source code for deployment
    └── app.py                  # FastAPI inference server (upcoming)
```

> **Note:** The `data/raw/` and `data/processed/` directories are intentionally empty on GitHub.
> All `.wav`, `.npy`, `.pth`, and `.save` files are listed in `.gitignore`.
> The `.gitkeep` files exist solely to preserve the folder structure when pushing to GitHub.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- (Recommended) A virtual environment

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/MarioAouad/Mechanical-Engineering-Machine-Learning-Model.git
cd Mechanical-Engineering-Machine-Learning-Model

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Usage Pipeline

### Step 1: Prepare the Data

Download the DCASE 2024 Task 2 development dataset and extract the machine-type folders into `data/raw/`:

```
data/raw/
├── fan/
│   ├── train/   ← place .wav files here
│   └── test/
├── bearing/
│   ├── train/
│   └── test/
└── (etc.)
```

### Step 2: Run the Preprocessing Pipeline

Open and execute the data preparation notebook:

```bash
jupyter notebook notebooks/01_data_prep.ipynb
```

This notebook will:
- Dynamically scan all machine types in `data/raw/`
- Convert each `.wav` file into a 128-band Log-Mel Spectrogram
- Split the data 85% Train / 15% Validation
- Apply `MinMaxScaler(0, 1)` normalization (fitted on training data only)
- Save `X_train.npy`, `X_val.npy`, and `scaler.save` into `data/processed/<machine>/`

### Step 3: Train the Autoencoder

```bash
jupyter notebook notebooks/02_autoencoder_training.ipynb
```

This notebook will:
- Define a Dense Autoencoder (`nn.Linear` layers: 40064 → 1024 → 256 → **64** → 256 → 1024 → 40064)
- Flatten `(128, 313)` spectrograms into 40,064-feature vectors
- Use `nn.Dropout(0.2)` to combat Domain Shift memorization
- Use `Sigmoid` output layer to match the `[0, 1]` scaled input range
- Train with `MSELoss` and `Adam` optimizer with `weight_decay` (L2 regularization)
- Apply Early Stopping on validation loss to prevent overfitting
- Save the best model weights as `.pth` checkpoints

### Step 4: Launch the FastAPI Inference Server

```bash
cd src
uvicorn app:app --reload
```

The server exposes API endpoints for real-time anomaly detection. Upload a `.wav` file, and the server returns an anomaly score based on reconstruction error.

---

## Current Project Architecture

| Phase | Notebook | Status | Key Technical Details |
|-------|----------|--------|----------------------|
| **Phase 1** — Data Preprocessing | `01_data_prep.ipynb` | ✅ Complete | Dynamic machine scanning, 128-band Log-Mel Spectrograms, 85/15 split, `MinMaxScaler(0,1)` with leakage prevention |
| **Phase 2** — Autoencoder Architecture | `02_autoencoder_training.ipynb` | ✅ Complete | Unsupervised PyTorch `nn.Linear` Autoencoder, **64-dim bottleneck** (626× compression), `nn.Dropout(0.2)` for Domain Shift regularization, `nn.Sigmoid` output layer, L2 regularization via `weight_decay` |
| **Phase 3** — Evaluation & Scoring | `03_evaluation.ipynb` | 🔲 Upcoming | Reconstruction Error (MSE) anomaly scoring, per-machine threshold tuning |
| **Phase 4** — API Deployment | `src/app.py` | 🔲 Upcoming | FastAPI REST endpoint for real-time `.wav` inference |

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Audio Processing | `librosa` | Load `.wav` files, compute Mel spectrograms |
| Data Science | `numpy`, `scikit-learn` | Array operations, train/val split, scaling |
| Deep Learning | `PyTorch` | Dense Autoencoder (fully-connected `nn.Linear` layers) |
| API Server | `FastAPI` + `Uvicorn` | REST API for model inference |
| Serialization | `joblib` | Save/load fitted scaler objects |

---

## Key Concepts

### Why Log-Mel Spectrograms?
Raw audio is a 1D signal. Neural networks perform much better on 2D representations where spatial patterns (frequency × time) can be learned convolutionally — similar to image recognition.

### Why Unsupervised?
In real industrial settings, anomalous sounds are **rare and unpredictable**. We can't collect enough labeled anomaly examples to train a supervised classifier. Instead, we train only on "normal" sounds and detect anything that *deviates* from normal.

### Why Reconstruction Error?
An Autoencoder compresses input into a bottleneck and reconstructs it. If the input looks like training data (normal), reconstruction is accurate (low MSE). If the input is anomalous, the model has never learned to reconstruct it — the error spikes, flagging the anomaly.

---

## License

This project is for academic and research purposes.

## Acknowledgments

- [DCASE 2024 Challenge Task 2](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) for the dataset and problem formulation.
