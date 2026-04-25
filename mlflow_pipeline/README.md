# MLflow Experiment Pipeline — Acoustic Anomaly Detection

## What Is This?

This folder contains a **self-contained experiment framework** for the DCASE 2024
Task 2 Acoustic Anomaly Detection challenge. It uses [MLflow](https://mlflow.org/)
to track every hyperparameter, loss curve, and model artifact across runs so you
can scientifically compare what works and what doesn't.

---

## The Core Concept: Unsupervised Autoencoders

Imagine teaching an AI what a **healthy** machine sounds like — and *only* what
healthy sounds like. That's exactly what an Autoencoder does:

```
Normal Sound → [Encoder] → 64 numbers (Bottleneck) → [Decoder] → Reconstructed Sound
                                                                   ↕
                                                            Compare with Original
                                                            Low Error = Normal ✓
```

When an **anomalous** sound is fed in, the Autoencoder has never learned to
reconstruct that pattern. The reconstruction error **spikes** — that spike is
our anomaly signal.

### Why "Unsupervised"?

In real factories, broken machines are **rare**. We can't collect thousands of
labeled "broken fan" recordings. So we train on normal sounds only and detect
anything that *deviates* from normal — no labels required.

---

## The Three Notebooks

| # | Notebook | What It Does |
|---|----------|-------------|
| **01** | `01_mlflow_prep.ipynb` | Converts `.wav` audio → Spectrograms (Linear or Mel). Scales data. Saves `.npy` matrices. |
| **02** | `02_mlflow_train.ipynb` | Builds & trains a CNN Autoencoder with configurable loss (Huber/L1/MSE), optimizer (Adam/AdamW), and bottleneck size. Logs everything to MLflow. |
| **03** | `03_mlflow_eval.ipynb` | Evaluates the trained model using one of three scoring methods. Computes AUC & pAUC. Logs results to MLflow. |

Each notebook has a **Control Panel** dictionary at the top — change one
parameter, re-run, and MLflow tracks the difference automatically.

---

## The Secret Weapon: Isolation Forest on Bottleneck Embeddings

Standard anomaly scoring uses **Reconstruction Error** (how badly the Autoencoder
fails to copy the input). But there's a more powerful approach:

### The Insight

The Autoencoder's bottleneck compresses each sound patch into just **64 numbers**
(a "fingerprint"). Normal sounds cluster tightly in this 64-dimensional space.
Anomalous sounds land in **empty regions** far from the cluster.

### The Method

1. Pass all **training** patches through the Encoder → extract 64-dim embeddings
2. Fit an **Isolation Forest** (or GMM) on these normal embeddings
3. Pass **test** patches through the Encoder → extract embeddings
4. Score test embeddings with the fitted model — outliers = anomalies

### Why This Solves Domain Shift

The DCASE 2024 test set contains machines recorded under **different conditions**
(speed, load, microphone position). Reconstruction Error is sensitive to these
surface-level changes. But the bottleneck embeddings capture **abstract patterns**
that are more robust to domain shift.

---

## How to Use MLflow

### 1. Install MLflow

```bash
pip install mlflow
```

### 2. Run a Notebook

Open any of the three notebooks and execute all cells. Each notebook automatically
logs its parameters, metrics, and artifacts to a local MLflow tracking directory.

### 3. Launch the Dashboard

From the `mlflow_pipeline/` directory, run:

```bash
cd mlflow_pipeline
mlflow ui
```

Then open your browser to: **http://localhost:5000**

### 4. Compare Experiments

In the MLflow UI you can:
- **Sort** runs by AUC or validation loss
- **Compare** loss curves across different hyperparameter settings
- **Download** the best model `.pth` file directly from the UI
- **Filter** by parameter (e.g., show only runs with `LOSS_FUNCTION=Huber`)

---

## Quick-Start Workflow

```bash
# 1. Ensure raw audio is in ./data/raw/<machine>/train/ and test/
# 2. Run notebooks in order:
#    01_mlflow_prep.ipynb   → generates spectrograms
#    02_mlflow_train.ipynb  → trains model, logs to MLflow
#    03_mlflow_eval.ipynb   → evaluates, logs AUC/pAUC
# 3. Launch dashboard:
mlflow ui
# 4. Open http://localhost:5000 to compare all runs
```

---

## Directory Structure

```
mlflow_pipeline/
├── README.md                    ← You are here
├── 01_mlflow_prep.ipynb         ← Data preprocessing engine
├── 02_mlflow_train.ipynb        ← Training engine + MLflow logging
├── 03_mlflow_eval.ipynb         ← Evaluation + embedding ML scoring
└── mlruns/                      ← Auto-created by MLflow (git-ignored)
```
