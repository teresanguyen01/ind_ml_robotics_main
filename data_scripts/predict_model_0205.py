#!/usr/bin/env python3
"""
Predict mocap/angle array from capacitance files using a trained model directory.
Also computes per-file movement percentage breakdown (e.g. % Bicep curl, % Walking).
"""

import os
import re
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------
# Config defaults (can be overridden via CLI)
# ---------------------------
SENSOR_SUFFIX = "_CapacitanceTable"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# Outlier handling consistent with training defaults
DEFAULT_HANDLE_OUTLIERS = True
DEFAULT_LOWER_Q, DEFAULT_UPPER_Q = 0.01, 0.99

# baselines
PER_FILE_BASELINE_NORMALIZE = True
PER_FILE_BASELINE_METHOD = "median"

# ---------------------------
# Plotting helper
# ---------------------------

def plot_joint_angles(df, joints, output_dir, pic_name):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = cm.tab20(np.linspace(0, 1, len(joints)))
    t = np.arange(len(df))
    for name, color in zip(joints, colors):
        if name not in df.columns:
            continue
        plt.plot(t, df[name].values, label=name, color=color)
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.title('Revolute Joint Angles (Predicted)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{pic_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

# joint-only columns (no x,y,z, no quats)
JOINT_COLUMNS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
]

# ---------------------------
# Helpers (mirrors training-time cleaning)
# ---------------------------

def clean_value(v):
    s = str(v).strip()
    parts = s.split('.')
    if len(parts) > 2:
        s = '.'.join(parts[:-1])
    return s

_TS_NAME_RE = re.compile(r'(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)', re.IGNORECASE)

def _detect_header(file_path, sample_rows=5):
    peek = pd.read_csv(file_path, nrows=sample_rows, header=None,
                       dtype=str, sep=None, engine='python')
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r'[A-Za-z]', s))).any():
        return 0
    return None

def _drop_timestamp_like_columns(df):
    cols_to_drop = []
    for c in df.columns:
        name = str(c)
        if _TS_NAME_RE.search(name.strip()):
            cols_to_drop.append(c)
    for c in df.columns:
        if c in cols_to_drop:
            continue
        col = df[c].astype(str)
        digit_ratio = np.mean(col.str.fullmatch(r'-?\d{10,}').fillna(False))
        iso_ratio = np.mean(col.str.contains(r'\d{4}-\d{2}-\d{2}', na=False))
        if digit_ratio > 0.9 or iso_ratio > 0.6:
            cols_to_drop.append(c)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def _winsorize_df(df, lower_q=0.01, upper_q=0.99):
    q_low = df.quantile(lower_q, axis=0, numeric_only=True)
    q_high = df.quantile(upper_q, axis=0, numeric_only=True)
    return df.clip(lower=q_low, upper=q_high, axis=1)

def load_and_clean_csv(file_path, handle_outliers=True,
                       lower_q=0.01, upper_q=0.99, drop_first_n_cols=0):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str,
                     sep=None, engine='python')

    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

    df = df.map(clean_value)
    df = _drop_timestamp_like_columns(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    before = len(df)
    df = df.dropna(how='all')
    if len(df) != before:
        print(f"Dropped {before - len(df)} completely empty rows.")

    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    if handle_outliers and not df.empty:
        df = _winsorize_df(df, lower_q=lower_q, upper_q=upper_q)

    return df.to_numpy(dtype=float)

def _basename_without_suffix(p: Path, suffix: str):
    stem = p.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem

def _collect_files(dir_path: str):
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {dir_path}")
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS]
    if not files:
        raise RuntimeError(f"No data files (csv/tsv/txt) found in: {dir_path}")
    return sorted(files)

def add_lags(X, lags=10):
    T, F = X.shape
    out = [X]
    pad = np.repeat(X[:1], lags, axis=0)
    Xp = np.vstack([pad, X])
    for k in range(1, lags + 1):
        out.append(Xp[lags - k:lags - k + T])
    return np.hstack(out)

# ---------------------------
# Model loading & prediction
# ---------------------------

def load_models(models_dir: str):
    models_dir = Path(models_dir)
    scaler_X_path = models_dir / 'scaler_X.pkl'
    scaler_y_path = models_dir / 'scaler_y.pkl'

    if not scaler_X_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError("scaler_X.pkl or scaler_y.pkl not found in models_dir")

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    boosters = []
    for f in sorted(models_dir.glob('xgboost_model_dim_*.json')):
        m = re.search(r'_(\d+)\.json$', f.name)
        if m:
            idx = int(m.group(1))
            boosters.append((idx, f))

    if not boosters:
        raise FileNotFoundError("No model files named like xgboost_model_dim_*.json found in models_dir")

    boosters.sort(key=lambda t: t[0])
    loaded = []
    for idx, path in boosters:
        booster = xgb.Booster()
        booster.load_model(str(path))
        loaded.append(booster)

    # Movement classifier (optional)
    clf_path = models_dir / "movement_classifier.pkl"
    le_path = models_dir / "movement_label_encoder.pkl"
    if clf_path.exists() and le_path.exists():
        movement_clf = joblib.load(clf_path)
        movement_le = joblib.load(le_path)
        print("[INFO] Loaded movement classifier and label encoder.")
    else:
        movement_clf = None
        movement_le = None
        print("[WARN] Movement classifier / label encoder not found; movement percentages will be skipped.")

    return scaler_X, scaler_y, loaded, movement_clf, movement_le

def predict_array_scaled(X_scaled: np.ndarray, scaler_y, boosters):
    """
    Predict array of outputs given **scaled** input features.
    X_scaled: [T, F], boosters: list of xgboost.Booster
    Returns unscaled predictions [T, D].
    """
    if X_scaled.ndim != 2:
        raise ValueError("X_scaled must be 2D")

    dmat = xgb.DMatrix(X_scaled)

    preds = []
    for booster in boosters:
        y_scaled = booster.predict(dmat)
        preds.append(y_scaled)

    Y_scaled = np.column_stack(preds)
    Y_unscaled = scaler_y.inverse_transform(Y_scaled)
    return Y_unscaled

def predict_movement_mix(X_scaled: np.ndarray, clf: XGBClassifier, le) -> dict:
    """
    Given scaled features and a trained classifier + label encoder,
    return a dict mapping movement label -> percentage (0-100).
    """
    probs = clf.predict_proba(X_scaled)       # [T, C]
    mean_probs = probs.mean(axis=0)           # [C]

    result = {}
    for idx, cls_name in enumerate(le.classes_):
        result[str(cls_name)] = float(mean_probs[idx] * 100.0)
    return result

# ---------------------------
# Directory inference
# ---------------------------

def predict_directory(models_dir: str,
                      input_dir: str,
                      out_dir: str,
                      handle_outliers: bool = DEFAULT_HANDLE_OUTLIERS,
                      lower_q: float = DEFAULT_LOWER_Q,
                      upper_q: float = DEFAULT_UPPER_Q,
                      drop_first_n_cols: int = 0,
                      output_suffix: str = "_pred_resamp.csv"):
    os.makedirs(out_dir, exist_ok=True)

    scaler_X, scaler_y, boosters, movement_clf, movement_le = load_models(models_dir)
    n_outputs = len(boosters)
    print(f"[INFO] Loaded scalers and {n_outputs} regression model(s) from: {models_dir}")

    files = _collect_files(input_dir)
    print(f"[INFO] Found {len(files)} input file(s) in: {input_dir}")

    meta_all = []

    # angle-array header (must match your trained y-dim)
    ANGLE_ARRAY_COLUMN_NAMES = [
        "qw",
        "qx",
        "qy",
        "qz",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_yaw_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
    ]

    for fp in tqdm(files, desc="Predicting"):
        try:
            X = load_and_clean_csv(str(fp),
                                   handle_outliers=handle_outliers,
                                   lower_q=lower_q,
                                   upper_q=upper_q,
                                   drop_first_n_cols=drop_first_n_cols)
            if X.size == 0 or X.shape[0] == 0:
                print(f"[WARN] Empty after cleaning, skipping: {fp.name}")
                continue

            if PER_FILE_BASELINE_NORMALIZE:
                if PER_FILE_BASELINE_METHOD == "median":
                    baseline = np.median(X, axis=0, keepdims=True)
                elif PER_FILE_BASELINE_METHOD == "mean":
                    baseline = np.mean(X, axis=0, keepdims=True)
                else:
                    raise ValueError(f"Unknown baseline method: {PER_FILE_BASELINE_METHOD}")
                X = X - baseline

            # If you used lags in training, uncomment:
            # X = add_lags(X, lags=10)

            # Check feature dimension
            try:
                X_scaled = scaler_X.transform(X)
            except Exception as e:
                print(f"[ERROR] Feature mismatch for {fp.name}: {e}")
                continue

            # Predict angles
            Y_pred = predict_array_scaled(X_scaled, scaler_y, boosters)  # [T, D]

            # Enforce quaternion continuity for root quaternion (dims 0..3)
            def enforce_quat_continuity(Y_pred, quat_blocks):
                Yc = Y_pred.copy()
                for s, e in quat_blocks:
                    q = Yc[:, s:e]
                    if q.shape[1] != 4:
                        continue
                    # Normalize
                    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
                    for t in range(1, q.shape[0]):
                        if np.dot(q[t], q[t - 1]) < 0:
                            q[t] = -q[t]
                    Yc[:, s:e] = q
                return Yc

            Yc = enforce_quat_continuity(Y_pred, [(0, 4)])  # root quat only

            base = _basename_without_suffix(fp, SENSOR_SUFFIX)
            out_path = Path(out_dir) / f"{base}{output_suffix}"

            if Yc.shape[1] != len(ANGLE_ARRAY_COLUMN_NAMES):
                print(f"[WARN] Output dim mismatch: got {Yc.shape[1]}, expected {len(ANGLE_ARRAY_COLUMN_NAMES)}")

            df_out = pd.DataFrame(Yc, columns=ANGLE_ARRAY_COLUMN_NAMES[:Yc.shape[1]])
            df_out.to_csv(out_path, index=False)

            # Plot joint angles
            pic_name = f"{base}_joints"
            plot_joint_angles(df_out, JOINT_COLUMNS, out_dir, pic_name)

            # Movement mix, if classifier available
            movement_mix = None
            if movement_clf is not None and movement_le is not None:
                movement_mix = predict_movement_mix(X_scaled, movement_clf, movement_le)
                print(f"{fp.name} movement mix:")
                for mv, pct in movement_mix.items():
                    print(f"  {mv}: {pct:.1f}%")

            meta_entry = {
                "input_file": str(fp),
                "output_file": str(out_path),
                "n_rows": int(Yc.shape[0]),
                "n_outputs": int(Yc.shape[1]),
                "movement_mix": movement_mix,
            }
            meta_all.append(meta_entry)

        except Exception as e:
            print(f"[ERROR] Failed on {fp.name}: {e}")

    summary_path = Path(out_dir) / "prediction_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "models_dir": str(models_dir),
            "input_dir": str(input_dir),
            "out_dir": str(out_dir),
            "n_outputs": n_outputs,
            "files": meta_all
        }, f, indent=2)

    print(f"[DONE] Wrote {len(meta_all)} predicted file(s) to: {out_dir}")
    print(f"[INFO] Summary -> {summary_path}")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Create predicted angle-array (mocap-ish) files from capacitance data directory, "
                    "and compute movement percentages."
    )
    p.add_argument("--models_dir", required=True,
                   help="Directory containing scaler_X.pkl, scaler_y.pkl, xgboost_model_dim_*.json, "
                        "movement_classifier.pkl, movement_label_encoder.pkl")
    p.add_argument("--input_dir", required=True,
                   help="Directory of capacitance CSV/TSV/TXT files")
    p.add_argument("--out_dir", required=True,
                   help="Output directory to write predicted CSVs and plots")
    p.add_argument("--no_outlier_handling", action="store_true",
                   help="Disable winsorization")
    p.add_argument("--lower_q", type=float, default=DEFAULT_LOWER_Q,
                   help="Lower quantile for winsorization")
    p.add_argument("--upper_q", type=float, default=DEFAULT_UPPER_Q,
                   help="Upper quantile for winsorization")
    p.add_argument("--drop_first_n_cols", type=int, default=0,
                   help="Drop first N columns from input files before processing (if needed)")
    p.add_argument("--output_suffix", type=str, default="_pred_resamp.csv",
                   help="Suffix for output filenames")
    return p.parse_args()

def main():
    args = parse_args()
    handle_outliers = not args.no_outlier_handling

    predict_directory(
        models_dir=args.models_dir,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        handle_outliers=handle_outliers,
        lower_q=args.lower_q,
        upper_q=args.upper_q,
        drop_first_n_cols=args.drop_first_n_cols,
        output_suffix=args.output_suffix
    )

if __name__ == "__main__":
    main()
