#!/usr/bin/env python
"""
Data Preprocessing V2 for Acoustic Anomaly Detection (DCASE 2024 Task 2)

Key changes from V1:
  - ref=np.max (per-file normalization, captures spectral shape not volume)
  - StandardScaler per mel band (z-score, no hard [0,1] clipping)
  - Target domain oversampling with random circular time shifts (not duplicates)
  - Saves to data/processed_v2/ (does NOT overwrite V1 data)
"""
import os, glob
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================================
# CONFIG
# ==========================================================================
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v2")

SAMPLE_RATE  = 16000
N_MELS       = 128
HOP_LENGTH   = 512
N_FFT        = 2048
VAL_SPLIT    = 0.15
RANDOM_STATE = 42

print(f"Data Preprocessing V2")
print(f"  Output: {PROCESSED_DIR}")
print(f"  Sample Rate: {SAMPLE_RATE}, Mel Bands: {N_MELS}, Hop: {HOP_LENGTH}, FFT: {N_FFT}")

# ==========================================================================
# DISCOVER MACHINE TYPES
# ==========================================================================
machine_types = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith("dev") and not d.startswith(".")
])
print(f"\nDiscovered {len(machine_types)} machine types: {machine_types}\n")

# ==========================================================================
# PROCESS EACH MACHINE
# ==========================================================================
for machine in machine_types:
    print("=" * 70)
    print(f"PROCESSING: {machine}")
    print("=" * 70)

    out_dir = os.path.join(PROCESSED_DIR, machine)
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(RAW_DIR, machine, "train")
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))

    if not wav_files:
        print(f"  [SKIP] No .wav files in {train_dir}\n")
        continue

    print(f"  Found {len(wav_files)} .wav files")

    # --- Step 1: Compute spectrograms ---
    specs = {}
    for i, fp in enumerate(wav_files):
        y, sr = librosa.load(fp, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        # ref=np.max: normalize each file's max to 0dB
        # This removes volume differences and captures spectral SHAPE
        log_mel = librosa.power_to_db(mel, ref=np.max)
        specs[fp] = log_mel
        if (i + 1) % 200 == 0 or (i + 1) == len(wav_files):
            print(f"    Computed {i+1}/{len(wav_files)} spectrograms...")

    # --- Step 2: Source/target split + oversampling ---
    src_files = [f for f in wav_files if "source" in os.path.basename(f)]
    tgt_files = [f for f in wav_files if "target" in os.path.basename(f)]

    if not src_files and not tgt_files:
        tr_files, va_files = train_test_split(
            wav_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE
        )
        tr_specs = [specs[f] for f in tr_files]
        va_specs = [specs[f] for f in va_files]
    else:
        src_tr, src_va = (
            train_test_split(src_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
            if src_files else ([], [])
        )
        tgt_tr, tgt_va = (
            train_test_split(tgt_files, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
            if tgt_files else ([], [])
        )

        src_tr_specs = [specs[f] for f in src_tr]
        tgt_tr_specs = [specs[f] for f in tgt_tr]

        # Oversample target with random circular time shifts to match source count
        if tgt_tr_specs and src_tr_specs:
            n_needed = len(src_tr_specs)
            rng = np.random.RandomState(RANDOM_STATE)
            augmented_tgt = []
            for i in range(n_needed):
                base = tgt_tr_specs[i % len(tgt_tr_specs)]
                shift = rng.randint(0, base.shape[1])
                augmented_tgt.append(np.roll(base, shift, axis=1))
            tr_specs = src_tr_specs + augmented_tgt
            print(f"    Domain balance: {len(src_tr_specs)} source + "
                  f"{len(augmented_tgt)} target (from {len(tgt_tr_specs)} with time shifts)")
        else:
            tr_specs = src_tr_specs + tgt_tr_specs

        va_specs = [specs[f] for f in src_va] + [specs[f] for f in tgt_va]

    X_train = np.array(tr_specs)
    X_val   = np.array(va_specs)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Raw dB range: [{X_train.min():.1f}, {X_train.max():.1f}]")

    # --- Step 3: StandardScaler per mel band ---
    N_tr, H, W = X_train.shape
    N_va = X_val.shape[0]

    # Reshape: (N, 128, W) → (N, W, 128) → (N*W, 128)
    X_tr_flat = X_train.transpose(0, 2, 1).reshape(-1, H)
    X_va_flat = X_val.transpose(0, 2, 1).reshape(-1, H)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(N_tr, W, H).transpose(0, 2, 1)
    X_va_scaled = scaler.transform(X_va_flat).reshape(N_va, W, H).transpose(0, 2, 1)

    print(f"  After StandardScaler:")
    print(f"    Train: mean={X_tr_scaled.mean():.4f}, std={X_tr_scaled.std():.4f}")
    print(f"    Val:   mean={X_va_scaled.mean():.4f}, std={X_va_scaled.std():.4f}")

    # --- Step 4: Save ---
    np.save(os.path.join(out_dir, "X_train.npy"), X_tr_scaled)
    np.save(os.path.join(out_dir, "X_val.npy"), X_va_scaled)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.save"))
    print(f"  Saved to {out_dir}/\n")

print("=" * 70)
print("DATA PREPROCESSING V2 COMPLETE")
print("=" * 70)
