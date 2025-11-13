#!/usr/bin/env python3
"""
Predict mocap from capacitance files using a trained model directory.

Expected model dir contents (from your training script):
  - scaler_X.pkl
  - scaler_y.pkl
  - xgboost_model_dim_1.json, xgboost_model_dim_2.json, ... (1-based indices)
  - (optional) run_metadata.json for reference only

Usage:
  python predict_mocap_dir.py \
    --models_dir MAIN_veronica_ml/ind_movements/walking/models \
    --input_dir  MAIN_veronica_ml/ind_movements/walking/sensor_new \
    --out_dir    MAIN_veronica_ml/ind_movements/walking/mocap_pred

Notes:
- Input files can be .csv/.tsv/.txt (case-insensitive).
- If an input filename ends with "_CapacitanceTable", that suffix is removed for the output base.
- Output files are named: <base>_pred_resamp.csv
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

# ---------------------------
# Config defaults (can be overridden via CLI)
# ---------------------------
SENSOR_SUFFIX = "_CapacitanceTable"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# Outlier handling consistent with training defaults
DEFAULT_HANDLE_OUTLIERS = True
DEFAULT_LOWER_Q, DEFAULT_UPPER_Q = 0.01, 0.99

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
    # autodetect delimiter and header presence
    peek = pd.read_csv(file_path, nrows=sample_rows, header=None, dtype=str, sep=None, engine='python')
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

def load_and_clean_csv(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99, drop_first_n_cols=0):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine='python')

    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

    df = df.map(clean_value)
    # df = _drop_timestamp_like_columns(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        # print(f"Dropping empty columns: {empty_cols}")
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
    for k in range(1, lags+1):
        out.append(Xp[lags-k: lags-k+T])
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

    # Collect boosters by 1-based index suffix
    boosters = []
    for f in sorted(models_dir.glob('xgboost_model_dim_*.json')):
        m = re.search(r'_(\d+)\.json$', f.name)
        if m:
            idx = int(m.group(1))
            boosters.append((idx, f))

    if not boosters:
        raise FileNotFoundError("No model files named like xgboost_model_dim_*.json found in models_dir")

    # Sort by index and load
    boosters.sort(key=lambda t: t[0])
    loaded = []
    for idx, path in boosters:
        booster = xgb.Booster()
        booster.load_model(str(path))
        loaded.append(booster)

    return scaler_X, scaler_y, loaded

def predict_array(X_raw: np.ndarray, scaler_X, scaler_y, boosters):
    """
    X_raw: numpy array [n_samples, n_features] of cleaned sensor input
    Returns: y_pred_unscaled [n_samples, n_outputs]
    """
    if X_raw.ndim != 2:
        raise ValueError("X_raw must be 2D")

    # Scale features using training scaler
    X_scaled = scaler_X.transform(X_raw)
    dmat = xgb.DMatrix(X_scaled)

    preds = []
    for booster in boosters:
        y_scaled = booster.predict(dmat)
        preds.append(y_scaled)

    Y_scaled = np.column_stack(preds)
    # Inverse transform to physical units
    Y_unscaled = scaler_y.inverse_transform(Y_scaled)
    return Y_unscaled

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
    """
    Walk input_dir for CSV/TSV/TXT files, produce a predicted mocap CSV per file in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    scaler_X, scaler_y, boosters = load_models(models_dir)
    n_outputs = len(boosters)
    print(f"[INFO] Loaded scalers and {n_outputs} model(s) from: {models_dir}")

    files = _collect_files(input_dir)
    print(f"[INFO] Found {len(files)} input file(s) in: {input_dir}")

    meta_all = []

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

            # Feature count sanity check
            try:
                # X = add_lags(X, lags=10)  # add lags consistent with training
                _ = scaler_X.transform(X[:1])
            except Exception as e:
                print(f"[ERROR] Feature mismatch for {fp.name}: {e}")
                continue

            Y_pred = predict_array(X, scaler_X, scaler_y, boosters)  # [T, D]

            # After you stack Y_pred (shape [T, D]) with 4D blocks per segment (W,X,Y,Z)
            def enforce_quat_continuity(Y_pred, quat_blocks):
                # quat_blocks: list of (start_col, end_col_exclusive) for each segment's quat
                Yc = Y_pred.copy()
                for s, e in quat_blocks:
                    q = Yc[:, s:e]  # [T,4] as (W,X,Y,Z)
                    # normalize
                    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
                    # flip signs to keep dot(q_t, q_{t-1}) >= 0
                    for t in range(1, q.shape[0]):
                        if np.dot(q[t], q[t-1]) < 0:
                            q[t] = -q[t]
                    Yc[:, s:e] = q
                return Yc
        
            Yc = enforce_quat_continuity(Y_pred, [
                (0, 4),    # base
                (4, 8),    # abdomen
                (8, 12),   # left_thigh
                (12, 16),  # right_thigh
                (16, 20),  # left_upper_arm
                (20, 24),  # right_upper_arm
                (24, 28),  # left_forearm
                (28, 32)   # right_forearm
            ])
            base = _basename_without_suffix(fp, SENSOR_SUFFIX)
            out_path = Path(out_dir) / f"{base}{output_suffix}"

            # Actual mocap target columns (without base or time_ms)
            MOCAP_COLUMN_NAMES = [
                "time_ms", "base_X", "base_Y", "base_Z", "base_W", "abdomen_X","abdomen_Y","abdomen_Z","abdomen_W",
                "left_thigh_X","left_thigh_Y","left_thigh_Z","left_thigh_W",
                "right_thigh_X","right_thigh_Y","right_thigh_Z","right_thigh_W",
                "left_upper_arm_X","left_upper_arm_Y","left_upper_arm_Z","left_upper_arm_W",
                "right_upper_arm_X","right_upper_arm_Y","right_upper_arm_Z","right_upper_arm_W",
                "left_forearm_X","left_forearm_Y","left_forearm_Z","left_forearm_W",
                "right_forearm_X","right_forearm_Y","right_forearm_Z","right_forearm_W"
            ]

            if Yc.shape[1] != len(MOCAP_COLUMN_NAMES):
                print(f"[WARN] Output dim mismatch: got {Yc.shape[1]}, expected {len(MOCAP_COLUMN_NAMES)}")

            df_out = pd.DataFrame(Yc, columns=MOCAP_COLUMN_NAMES[:Yc.shape[1]])
            df_out.to_csv(out_path, index=False)

            meta_all.append({
                "input_file": str(fp),
                "output_file": str(out_path),
                "n_rows": int(Yc.shape[0]),
                "n_outputs": int(Yc.shape[1])
            })

        except Exception as e:
            print(f"[ERROR] Failed on {fp.name}: {e}")

    # Dump a small run summary
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
    p = argparse.ArgumentParser(description="Create predicted mocap files from capacitance data directory.")
    p.add_argument("--models_dir", required=True, help="Directory containing scaler_X.pkl, scaler_y.pkl, and xgboost_model_dim_*.json")
    p.add_argument("--input_dir", required=True, help="Directory of capacitance CSV/TSV/TXT files")
    p.add_argument("--out_dir", required=True, help="Output directory to write predicted mocap CSVs")
    p.add_argument("--no_outlier_handling", action="store_true", help="Disable winsorization")
    p.add_argument("--lower_q", type=float, default=DEFAULT_LOWER_Q, help="Lower quantile for winsorization")
    p.add_argument("--upper_q", type=float, default=DEFAULT_UPPER_Q, help="Upper quantile for winsorization")
    p.add_argument("--drop_first_n_cols", type=int, default=0, help="Drop first N columns from input files before processing (if needed)")
    p.add_argument("--output_suffix", type=str, default="_pred_resamp.csv", help="Suffix for output filenames")
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