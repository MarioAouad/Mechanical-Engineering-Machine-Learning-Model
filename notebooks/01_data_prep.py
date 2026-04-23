#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Data Preprocessing (Audio Pipeline)
# 
# ---
# 
# ## What This Notebook Does
# 
# This notebook is the **first phase** of our Acoustic Anomaly Detection pipeline, built around the
# [DCASE 2024 Challenge Task 2](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) dataset.
# 
# ### The Core Problem
# 
# Raw audio files are **1-dimensional time-series waveforms** — just a long list of amplitude values.
# Neural networks (especially Convolutional Autoencoders) are far more effective when they can
# treat data like **2D images**, where spatial patterns become visible.
# 
# ### Our Solution: Log-Mel Spectrograms
# 
# We convert each `.wav` file into a **Log-Mel Spectrogram** — a 2D matrix where:
# 
# | Axis | Represents | Intuition |
# |------|-----------|----------|
# | **Y-axis** (rows) | Mel frequency bands | "Which pitches are present?" |
# | **X-axis** (columns) | Time frames | "When do those pitches occur?" |
# | **Cell value** | Log-scaled power (dB) | "How loud is that pitch at that moment?" |
# 
# Think of it as a **heat-map photograph of sound**. Our future Autoencoder will learn what
# "normal" heat-maps look like, and flag anything that deviates as an anomaly.
# 
# ### Pipeline Steps
# 
# 1. **Dynamically scan** `data/raw/` to discover all machine types (fan, pump, valve, etc.)
# 2. **Load** each `.wav` file at a fixed sample rate of 16 kHz
# 3. **Extract** 128-band Log-Mel Spectrograms
# 4. **Split** into 85% Training / 15% Validation sets (before scaling — to prevent data leakage)
# 5. **Normalize** using `MinMaxScaler(0, 1)` — fitted ONLY on training data
# 6. **Save** processed arrays as `.npy` files and the scaler as `scaler.save` per machine type
# 
# > **Why does this matter?** Without this preprocessing step, our Autoencoder would receive
# > inconsistent, unnormalized, high-dimensional data and fail to converge during training.

# ---
# 
# ## 1. Library Imports
# 
# We rely on a focused set of libraries, each chosen for a specific role:
# 
# | Library | Purpose |
# |---------|--------|
# | **`librosa`** | The gold-standard library for audio analysis in Python. It handles loading `.wav` files, computing Short-Time Fourier Transforms (STFT), mapping frequencies onto the perceptually-motivated Mel scale, and converting power to decibels. |
# | **`numpy`** | Our workhorse for all matrix/array operations. Spectrograms are just NumPy arrays under the hood. |
# | **`scikit-learn`** | Provides `train_test_split` for statistically sound data splitting, and `MinMaxScaler` for feature normalization. |
# | **`joblib`** | Efficiently serializes our fitted scaler object to disk so we can reuse the *exact same* mathematical transformation on unseen test data in Phase 4. |
# | **`os` / `glob`** | Standard Python modules for filesystem navigation and pattern-based file discovery. |

# In[1]:


import os
import glob

import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("All libraries imported successfully.")
print(f"  librosa  version: {librosa.__version__}")
print(f"  numpy    version: {np.__version__}")


# ---
# 
# ## 2. Directory Setup & Dynamic Machine-Type Scanning
# 
# ### Why Dynamic Scanning?
# 
# A common beginner mistake is to **hardcode** machine names like this:
# 
# ```python
# # BAD — brittle, breaks when you add/remove machines
# machines = ['fan', 'pump', 'slider', 'valve']
# ```
# 
# Instead, we **dynamically scan** the `data/raw/` directory to discover whatever machine
# folders exist at runtime. This makes our pipeline:
# 
# - **Portable** — works on anyone's machine without edits
# - **Scalable** — automatically picks up new machine types if you add more data
# - **Robust** — no risk of typos in hardcoded lists
# 
# The code below also creates the corresponding output directories under `data/processed/`
# so we have a clean, organized place to store our results.

# In[2]:


# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ---------------------------------------------------------------------------
# Dynamically discover all machine-type folders inside data/raw/
# We filter to directories only (ignore stray files like .DS_Store)
# ---------------------------------------------------------------------------
machine_types = sorted([
    d for d in os.listdir(RAW_BASE_DIR)
    if os.path.isdir(os.path.join(RAW_BASE_DIR, d))
])

print(f"Discovered {len(machine_types)} machine types:")
for i, m in enumerate(machine_types, 1):
    train_path = os.path.join(RAW_BASE_DIR, m, "train")
    n_files = len(glob.glob(os.path.join(train_path, "*.wav"))) if os.path.isdir(train_path) else 0
    print(f"  {i}. {m:<12s}  →  {n_files} training .wav files")


# ---
# 
# ## 3. Audio Loading & Log-Mel Spectrogram Extraction
# 
# This is the **most physics-heavy step** in the entire pipeline. Let's break down the two
# critical rules we enforce:
# 
# ### Rule 1: Force a Consistent Sample Rate (`sr=16000`)
# 
# Audio files can be recorded at different sample rates (44.1 kHz for music, 16 kHz for speech,
# 8 kHz for telephony). The sample rate determines **how many amplitude measurements per second**
# are captured.
# 
# If we don't force a fixed sample rate, different files will produce arrays of **different lengths**,
# and we won't be able to stack them into a single 3D NumPy array. By setting `sr=16000`,
# librosa will automatically **resample** any file that doesn't match.
# 
# The DCASE 2024 dataset is already at 16 kHz, so this is mostly a safety guard.
# 
# ### Rule 2: Extract 128 Mel Bands & Convert to Decibels
# 
# **Why Mel Scale?** Human hearing is *not* linear — we can distinguish between 100 Hz and
# 200 Hz much more easily than between 10,000 Hz and 10,100 Hz. The Mel scale compresses
# high frequencies, mimicking how our ears actually perceive pitch. This gives the neural
# network a more *meaningful* frequency representation.
# 
# **Why 128 bands?** This is the standard used in most audio ML research. It provides enough
# frequency resolution to capture the acoustic signatures of mechanical faults (bearing wear,
# fan imbalance, valve leaks) without creating excessively large matrices.
# 
# **Why convert to decibels (`power_to_db`)?** Raw Mel spectrogram values span a *huge*
# dynamic range (quiet background hum vs. loud motor). Converting to a logarithmic decibel
# scale compresses this range, making it far easier for a neural network to learn patterns.
# This is analogous to why we use log-returns in finance instead of raw prices.
# 
# ### What Comes Out
# 
# For each machine type, we produce a 3D array of shape:
# 
# ```
# X_all.shape = (num_files, 128, time_frames)
# ```
# 
# where `time_frames` depends on the audio duration and hop length (default: 512 samples).

# In[3]:


# ===========================================================================
# HYPERPARAMETERS — Centralized here for easy tuning
# ===========================================================================
SAMPLE_RATE = 16000   # Force all audio to this sample rate (Hz)
N_MELS = 128          # Number of Mel frequency bands
HOP_LENGTH = 512      # Samples between successive STFT columns
N_FFT = 2048          # Length of the FFT window
VAL_SPLIT = 0.15      # Fraction of data reserved for validation
RANDOM_STATE = 42     # Seed for reproducible train/val splits

print("Hyperparameters set:")
print(f"  Sample Rate  : {SAMPLE_RATE} Hz")
print(f"  Mel Bands    : {N_MELS}")
print(f"  Hop Length   : {HOP_LENGTH}")
print(f"  FFT Window   : {N_FFT}")
print(f"  Val Split    : {VAL_SPLIT * 100:.0f}%")
print(f"  Random Seed  : {RANDOM_STATE}")


# In[4]:


import os

# (Variables already defined above using PROJECT_ROOT)

machine_types = sorted([
    d for d in os.listdir(RAW_BASE_DIR)
    if os.path.isdir(os.path.join(RAW_BASE_DIR, d)) and not d.startswith("dev")
])



print(machine_types)

for machine in machine_types:
    print("=" * 70)
    print(f"PROCESSING: {machine}")
    print("=" * 70)

    processed_dir = os.path.join(PROCESSED_BASE_DIR, machine)
    os.makedirs(processed_dir, exist_ok=True)

    train_dir = os.path.join(RAW_BASE_DIR, machine, "train")
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))

    if len(wav_files) == 0:
        print(f"  [WARNING] No .wav files found in {train_dir}. Skipping.\n")
        continue

    print(f"  Found {len(wav_files)} .wav files in {train_dir}")

    # Step 1: Extract all unique spectrograms first
    spec_dict = {}
    for i, fpath in enumerate(wav_files):
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        # CRITICAL FIX: ref=1.0 retains absolute energy across files.
        # ref=np.max was normalizing every file to 0dB, destroying relative amplitudes!
        log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
        spec_dict[fpath] = log_mel_spec

        if (i + 1) % 200 == 0 or (i + 1) == len(wav_files):
            print(f"    Processed {i + 1}/{len(wav_files)} files...")

    # Step 2: Split into source and target to handle domain shift
    src_files = [f for f in wav_files if "source" in os.path.basename(f)]
    tgt_files = [f for f in wav_files if "target" in os.path.basename(f)]

    if not src_files and not tgt_files:
        tr_files, va_files = train_test_split(wav_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
    else:
        src_tr, src_va = train_test_split(src_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE) if src_files else ([], [])
        tgt_tr, tgt_va = train_test_split(tgt_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE) if tgt_files else ([], [])
        
        # CRITICAL FIX: Oversample target domain in training set
        # DCASE first-shot provides ~990 source and ~10 target.
        # We oversample target to match source, so Autoencoder learns to reconstruct target normal perfectly.
        oversample = max(1, len(src_tr) // max(1, len(tgt_tr))) if tgt_tr and src_tr else 1
        tr_files = src_tr + tgt_tr * oversample
        va_files = src_va + tgt_va
        print(f"    Domain Shift Fix: Oversampled {len(tgt_tr)} target train files by {oversample}x")

    X_train = np.array([spec_dict[f] for f in tr_files])
    X_val = np.array([spec_dict[f] for f in va_files])

    print(f"  Combined train shape: {X_train.shape}, val shape: {X_val.shape}")
    print(f"  Value range: [{X_train.min():.2f}, {X_train.max():.2f}] dB")

    N_train, H, W = X_train.shape
    N_val, _, _ = X_val.shape

    # Time-Independent Scaling: flatten to (-1, 128)
    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, H)
    X_val_flat = X_val.transpose(0, 2, 1).reshape(-1, H)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    X_train_scaled = X_train_scaled.reshape(N_train, W, H).transpose(0, 2, 1)
    X_val_scaled = X_val_scaled.reshape(N_val, W, H).transpose(0, 2, 1)

    print(f"\n  After MinMaxScaler(0, 1):")
    print(f"    X_train range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    if N_val > 0:
        print(f"    X_val   range: [{X_val_scaled.min():.4f}, {X_val_scaled.max():.4f}]")

    train_path = os.path.join(processed_dir, "X_train.npy")
    val_path = os.path.join(processed_dir, "X_val.npy")
    scaler_path = os.path.join(processed_dir, "scaler.save")

    np.save(train_path, X_train_scaled)
    np.save(val_path, X_val_scaled)
    joblib.dump(scaler, scaler_path)

    print(f"\n  Saved to {processed_dir}:")
    print(f"    ✓ X_train.npy   — {X_train_scaled.shape}")
    print(f"    ✓ X_val.npy     — {X_val_scaled.shape}")
    print(f"    ✓ scaler.save   — MinMaxScaler fitted on training data")
    print()


# ---
# 
# ## 4. Train / Validation Split — Why We Split *Before* Scaling
# 
# ### The Data Leakage Trap
# 
# This is one of the most common and dangerous mistakes in machine learning:
# 
# ```
# ❌ WRONG ORDER:  Scale All Data  →  Split into Train/Val
# ✅ RIGHT ORDER:  Split into Train/Val  →  Scale (fit on Train only)
# ```
# 
# **Why is the wrong order dangerous?** If you scale before splitting, the scaler's `min` and
# `max` values are computed using *all* data — including the validation set. This means your
# model indirectly "knows" statistical properties of data it's supposed to have never seen.
# Your validation metrics will look artificially good, but the model will perform worse on
# truly unseen test data.
# 
# ### The 85/15 Split
# 
# We use 85% for training and 15% for validation. The validation set serves as a
# **"practice quiz"** during training:
# 
# - After each training epoch, the model evaluates its reconstruction loss on the validation set
# - If validation loss stops improving for several consecutive epochs, **Early Stopping** kicks in
# - This prevents **overfitting** — the model memorizing training data instead of learning
#   general patterns
# 
# > **Note:** We use `random_state=42` to ensure the split is **reproducible**. Running this
# > notebook again will produce the exact same train/val partition.

# ---
# 
# ## 5. The Critical Normalization Step — Why MinMaxScaler(0, 1) Is Non-Negotiable
# 
# ### The Mathematical Trap
# 
# Log-Mel Spectrograms produced by `librosa.power_to_db` contain values in a range like
# **[-80 dB, 0 dB]**. Here's why that's a problem:
# 
# Our Autoencoder's output layer will use a **Sigmoid activation function**:
# 
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
# 
# Sigmoid can **only output values between 0 and 1**. If our input data contains -80 dB,
# the network literally cannot reconstruct it. The loss function will be enormous and will
# never converge — the model will learn nothing.
# 
# ### The Fix: MinMaxScaler
# 
# MinMaxScaler applies this transformation to every feature (pixel):
# 
# $$X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
# 
# This compresses all values into **[0, 1]**, which perfectly matches the Sigmoid output range.
# 
# ### The Reshape Dance (3D → 2D → 3D)
# 
# Scikit-learn's scalers expect 2D input `(n_samples, n_features)`, but our data is 3D
# `(n_samples, n_mels, time_frames)`. So we:
# 
# 1. **Flatten** each 2D spectrogram into a 1D vector: `(N, 128, 313)` → `(N, 40064)`
# 2. **Apply** the scaler on the 2D data
# 3. **Reshape** back to the original 3D shape
# 
# ### The Leakage Guard
# 
# ```python
# # ✅ CORRECT — scaler learns from training data ONLY
# scaler.fit_transform(X_train_flat)  # Learn min/max + apply
# scaler.transform(X_val_flat)        # Apply same min/max (no re-learning)
# 
# # ❌ WRONG — scaler re-learns from validation data
# scaler.fit_transform(X_val_flat)    # This would contaminate everything!
# ```

# ---
# 
# ## 6. Saving to Disk — Why This Saves Hours of Compute
# 
# ### The `.npy` Files
# 
# Saving processed spectrograms as `.npy` files is a massive efficiency win:
# 
# | Operation | Time (approx.) |
# |-----------|---------------|
# | Load 1000 `.wav` files + compute spectrograms | **2–5 minutes** |
# | Load `X_train.npy` from disk | **< 1 second** |
# 
# During model training (Phase 2), we'll load this data hundreds of times across epochs.
# Pre-computing saves *hours* of wasted reprocessing.
# 
# ### The `scaler.save` File
# 
# This file is **mandatory** for evaluation integrity. Here's the pipeline it enables:
# 
# ```
# Phase 1 (this notebook):  Fit scaler on train → Save scaler
# Phase 2 (training):       Load X_train.npy    → Train Autoencoder
# Phase 3 (evaluation):     Load scaler.save    → Transform test data → Compute anomaly scores
# ```
# 
# If we lost the scaler and created a new one on test data, the model's reconstruction
# errors would be measured on a *different numerical scale* — making all anomaly scores
# meaningless.

# ---
# 
# ## 7. Verification — Quick Sanity Check
# 
# Let's verify that our saved files can be loaded and contain sensible data.

# In[5]:


# ===========================================================================
# VERIFICATION — Reload saved files and confirm integrity
# ===========================================================================

print("Verification: Reloading saved files...\n")

for machine in machine_types:
    processed_dir = os.path.join(PROCESSED_BASE_DIR, machine)

    train_file = os.path.join(processed_dir, "X_train.npy")
    val_file = os.path.join(processed_dir, "X_val.npy")
    scaler_file = os.path.join(processed_dir, "scaler.save")

    # Check all 3 files exist
    if not all(os.path.exists(f) for f in [train_file, val_file, scaler_file]):
        print(f"  [{machine}] ⚠ Missing files — skipping verification")
        continue

    # Reload from disk
    X_train_check = np.load(train_file)
    X_val_check = np.load(val_file)
    scaler_check = joblib.load(scaler_file)

    # Validate value ranges are within [0, 1] for training data
    train_in_range = (X_train_check.min() >= -0.001) and (X_train_check.max() <= 1.001)
    status = "✓ PASS" if train_in_range else "✗ FAIL"

    print(f"  [{machine}] {status}")
    print(f"    X_train : {X_train_check.shape}  range=[{X_train_check.min():.4f}, {X_train_check.max():.4f}]")
    print(f"    X_val   : {X_val_check.shape}  range=[{X_val_check.min():.4f}, {X_val_check.max():.4f}]")
    print(f"    Scaler  : {type(scaler_check).__name__} (feature_range={scaler_check.feature_range})")
    print()


# ---
# 
# ## Summary & Next Steps
# 
# ### What We Accomplished
# 
# | Step | Input | Output |
# |------|-------|--------|
# | Audio Loading | `.wav` files (1D waveform) | NumPy arrays |
# | Feature Extraction | 1D waveform | 2D Log-Mel Spectrogram (128 × T) |
# | Train/Val Split | Single 3D array | Two separate 3D arrays |
# | Normalization | dB values [-80, 0] | Scaled values [0, 1] |
# | Persistence | In-memory arrays | `.npy` files + `scaler.save` |
# 
# ### Output File Structure
# 
# ```
# data/processed/
# ├── bearing/
# │   ├── X_train.npy
# │   ├── X_val.npy
# │   └── scaler.save
# ├── fan/
# │   ├── X_train.npy
# │   ├── X_val.npy
# │   └── scaler.save
# ├── gearbox/
# │   └── ...
# └── (etc. for all machine types)
# ```
# 
# ### Next: Phase 2 — Autoencoder Training (`02_model_training.ipynb`)
# 
# In the next notebook, we will:
# 1. Load these `.npy` files
# 2. Build a Convolutional Autoencoder in TensorFlow/Keras
# 3. Train it to reconstruct *normal* spectrograms
# 4. Use Early Stopping with the validation set to prevent overfitting
# 
# The key insight: anomalous sounds will produce spectrograms that the Autoencoder
# *cannot reconstruct well*, resulting in high reconstruction error — our anomaly signal.
