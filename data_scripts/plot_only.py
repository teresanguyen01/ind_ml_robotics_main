#!/usr/bin/env python3
"""
PLOT ONLY SCRIPT
----------------
This script does NOT train new models. 
It loads existing models from 'output_dir', runs predictions on the TEST set, 
and generates the "Nature-style" figures in PNG, SVG, and EPS formats.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (Must match your training config)
# ---------------------------
output_dir = '/home/tt/ind_ml_rawan_teresa/AA_hand_included_1221/veronica/models'

# We ONLY need Test dirs for plotting
test_sensor_dir = "/home/tt/ind_ml_rawan_teresa/AA_hand_included_1221/veronica/sensor_test"
test_mocap_dir = "/home/tt/ind_ml_rawan_teresa/AA_hand_included_1221/veronica/aa_test"

# Params used during loading
STRICT_ROW_MATCH = False
HANDLE_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.01, 0.99
PER_FILE_BASELINE_NORMALIZE = True
PER_FILE_BASELINE_METHOD = "median"

SENSOR_SUFFIX = "_CapacitanceTable"
MOCAP_SUFFIX = "_resamp"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# ---------------------------
# JOINT METADATA (Must match training)
# ---------------------------
JOINT_NAMES = [
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

JOINTS_FOR_PLOTS = [
    "waist_yaw_joint",
    "waist_pitch_joint",
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

DT_MS = 10.0

# ---------------------------
# Helpers (Same as before)
# ---------------------------
def clean_value(v):
    s = str(v).strip()
    parts = s.split('.')
    if len(parts) > 2: s = '.'.join(parts[:-1])
    return s

_TS_NAME_RE = re.compile(r'(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)', re.IGNORECASE)

def _detect_header(file_path, sample_rows=5):
    peek = pd.read_csv(file_path, nrows=sample_rows, header=None, dtype=str, sep=None, engine='python')
    if peek.iloc[0].astype(str).apply(lambda s: bool(re.search(r'[A-Za-z]', s))).any(): return 0
    return None

def _drop_timestamp_like_columns(df):
    cols_to_drop = []
    for c in df.columns:
        if _TS_NAME_RE.search(str(c).strip()): cols_to_drop.append(c)
    for c in df.columns:
        if c in cols_to_drop: continue
        col = df[c].astype(str)
        if np.mean(col.str.fullmatch(r'-?\d{10,}').fillna(False)) > 0.9: cols_to_drop.append(c)
    if cols_to_drop: df = df.drop(columns=cols_to_drop)
    return df

def _winsorize_df(df, lower_q=0.01, upper_q=0.99):
    q_low = df.quantile(lower_q, axis=0, numeric_only=True)
    q_high = df.quantile(upper_q, axis=0, numeric_only=True)
    return df.clip(lower=q_low, upper=q_high, axis=1)

def load_and_clean_csv(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99, drop_first_n_cols=0):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine='python')
    if drop_first_n_cols > 0: df = df.iloc[:, drop_first_n_cols:]
    df = df.map(clean_value)
    df = _drop_timestamp_like_columns(df)
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(how='all').fillna(df.median(numeric_only=True))
    if handle_outliers and not df.empty: df = _winsorize_df(df, lower_q, upper_q)
    return df.to_numpy(dtype=float)

def _collect_files(dir_path):
    d = Path(dir_path)
    return [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS]

def match_pairs_strict(sensor_dir, mocap_dir):
    s_files = [p for p in _collect_files(sensor_dir) if p.stem.endswith(SENSOR_SUFFIX)]
    m_files = [p for p in _collect_files(mocap_dir) if p.stem.endswith(MOCAP_SUFFIX)]
    
    def get_key(p, suffix): return p.stem[:-len(suffix)]
    smap = {get_key(p, SENSOR_SUFFIX): p for p in s_files}
    mmap = {get_key(p, MOCAP_SUFFIX): p for p in m_files}
    
    keys = sorted(set(smap.keys()).intersection(mmap.keys()))
    return [(smap[k], mmap[k], k) for k in keys]

def load_dataset_from_dirs(sensor_dir, mocap_dir):
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_list, y_list, keys, file_ids_all = [], [], [], []
    
    print(f"[INFO] Loading {len(pairs)} test files...")
    
    for s_path, m_path, key in pairs:
        X = load_and_clean_csv(str(s_path), HANDLE_OUTLIERS, LOWER_Q, UPPER_Q, 0)
        y = load_and_clean_csv(str(m_path), HANDLE_OUTLIERS, LOWER_Q, UPPER_Q, 4)
        
        # Align
        mn = min(X.shape[0], y.shape[0])
        X, y = X[:mn], y[:mn]
        
        # Norm
        if PER_FILE_BASELINE_NORMALIZE:
            base = np.median(X, axis=0, keepdims=True) if PER_FILE_BASELINE_METHOD == "median" else np.mean(X, axis=0, keepdims=True)
            X = X - base
            
        X_list.append(X)
        y_list.append(y)
        keys.append(key)
        file_ids_all.extend([key] * len(X))
        
    return np.vstack(X_list), np.vstack(y_list), np.array(file_ids_all), keys

# ---------------------------
# MAIN PLOTTING FUNCTION
# ---------------------------
def plot_figures_only():
    if not os.path.exists(output_dir):
        print(f"[ERROR] Output directory not found: {output_dir}. Run training first!")
        return

    # 1. Load Data
    X_test, y_test_full, test_file_ids, test_keys = load_dataset_from_dirs(test_sensor_dir, test_mocap_dir)
    
    # 2. Load Scalers
    print("[INFO] Loading Scalers...")
    scaler_X = joblib.load(os.path.join(output_dir, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(output_dir, 'scaler_y.pkl'))
    
    X_test_scaled = scaler_X.transform(X_test)
    
    # 3. Load Regression Models & Predict
    print("[INFO] Loading XGBoost models & Predicting...")
    num_dims = y_test_full.shape[1]
    y_test_pred_list = []
    dtest = xgb.DMatrix(X_test_scaled)
    
    for i in range(num_dims):
        model_path = os.path.join(output_dir, f'xgboost_model_dim_{i+1}.json')
        if not os.path.exists(model_path):
             print(f"[ERROR] Model {model_path} not found.")
             continue
        
        bst = xgb.Booster(model_file=model_path)
        y_test_pred_list.append(bst.predict(dtest))
        
    y_test_pred_scaled = np.column_stack(y_test_pred_list)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)
    
    # 4. Load Classifier
    print("[INFO] Loading Classifier...")
    clf = joblib.load(os.path.join(output_dir, "movement_classifier.pkl"))
    le = joblib.load(os.path.join(output_dir, "movement_label_encoder.pkl"))
    
    # 5. Setup Plotting
    plots_dir = os.path.join(output_dir, 'test_plots_svg_eps')
    os.makedirs(plots_dir, exist_ok=True)
    
    joint_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    joint_indices_for_plots = [joint_name_to_idx[name] for name in JOINTS_FOR_PLOTS if name in joint_name_to_idx]
    joint_labels_for_plots = [JOINT_NAMES[idx] for idx in joint_indices_for_plots]

    # ---------------------------------------------------------
    # PLOT LOOP
    # ---------------------------------------------------------
    print(f"[INFO] Generating plots in: {plots_dir}")
    
    # Build index lists per file key
    indices_by_file = defaultdict(list)
    for idx, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(idx)

    for fid in sorted(indices_by_file.keys()):
        print(f"Processing plots for: {fid}")
        idxs = np.array(indices_by_file[fid], dtype=int)
        
        # Data for this file
        y_true_file = y_test_full[idxs]
        y_pred_file = y_test_pred_unscaled[idxs]

        # Calculate Accuracy
        y_true_sel = y_true_file[:, joint_indices_for_plots]
        y_pred_sel = y_pred_file[:, joint_indices_for_plots]

        # Ranges from FULL test set (consistent normalization)
        full_true_sel = y_test_full[:, joint_indices_for_plots]
        joint_ranges = full_true_sel.max(axis=0) - full_true_sel.min(axis=0)
        joint_ranges[joint_ranges == 0] = 1e-6

        abs_err_file = np.abs(y_pred_sel - y_true_sel)
        norm_err_file = abs_err_file / joint_ranges
        
        mean_norm_err_per_joint = norm_err_file.mean(axis=0)
        joint_acc_pct = 100.0 * (1.0 - mean_norm_err_per_joint)
        joint_acc_pct = np.clip(joint_acc_pct, 0.0, 100.0)

        # -----------------------------------------------------
        # FIGURE 1: NATURE STYLE POLAR PLOT (YlGnBu)
        # -----------------------------------------------------
        if any(a > 0 for a in joint_acc_pct):
            acc_values = np.array(joint_acc_pct)
            N = len(joint_labels_for_plots)
            theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
            width = 2 * np.pi / N

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')
            bars = ax.bar(theta, acc_values, width=width, bottom=0.0, align='edge')

            # --- COLOR SETTINGS (NATURE STYLE: YlGnBu) ---
            cmap = plt.get_cmap("YlGnBu")
            # Normalize 50-100 so high accuracy gets the deep teal color
            norm = plt.Normalize(vmin=50, vmax=100)

            for bar, val in zip(bars, acc_values):
                bar.set_facecolor(cmap(norm(val)))
                bar.set_alpha(0.9)
                bar.set_edgecolor('white')
                bar.set_linewidth(0.5)

            ax.set_xticks(theta + width / 2)
            ax.set_xticklabels(joint_labels_for_plots, fontsize=9)
            
            # Y-Ticks
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', ''], color="gray", fontsize=8)
            ax.grid(True, alpha=0.2, color='black')

            for angle, height in zip(theta, acc_values):
                text_angle = angle + width / 2
                text_color = 'white' if height > 80 else 'black'
                ax.text(text_angle, height - 10, f"{height:.1f}%", 
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=8)

            plt.title(f"Joint prediction accuracy\nfile: {fid}", y=1.08)
            plt.tight_layout()
            
            base_polar = os.path.join(plots_dir, f"joint_accuracy_polar_{fid}")
            # SAVE AS PNG, SVG, EPS
            plt.savefig(base_polar + ".png", dpi=150, bbox_inches="tight")
            plt.savefig(base_polar + ".svg", format='svg', bbox_inches="tight")
            plt.savefig(base_polar + ".eps", format='eps', bbox_inches="tight")
            plt.close()

        # -----------------------------------------------------
        # FIGURE 2: ACCURACY OVER TIME
        # -----------------------------------------------------
        accuracy_per_frame = 100.0 * (1.0 - norm_err_file.mean(axis=1))
        accuracy_per_frame = np.clip(accuracy_per_frame, 0.0, 100.0)
        
        window_size = 15
        acc_series = pd.Series(accuracy_per_frame)
        accuracy_smoothed = acc_series.rolling(window=window_size, center=True, min_periods=1).mean().values
        time_axis_ms = np.arange(len(idxs)) * DT_MS

        plt.figure(figsize=(8, 4))
        plt.plot(time_axis_ms, accuracy_per_frame, color='gray', alpha=0.15, label='Raw')
        plt.plot(time_axis_ms, accuracy_smoothed, color='tab:blue', linewidth=2, label='Smoothed')
        plt.xlabel("Time (ms)")
        plt.ylabel("Accuracy percentage")
        plt.title(f"Prediction accuracy over time\nfile: {fid}")
        plt.ylim(60, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()

        base_acc_time = os.path.join(plots_dir, f"accuracy_over_time_{fid}")
        # SAVE AS PNG, SVG, EPS
        plt.savefig(base_acc_time + ".png", dpi=150, bbox_inches="tight")
        plt.savefig(base_acc_time + ".svg", format='svg', bbox_inches="tight")
        plt.savefig(base_acc_time + ".eps", format='eps', bbox_inches="tight")
        plt.close()
        
    print("\n[DONE] All plots generated.")

if __name__ == "__main__":
    plot_figures_only()