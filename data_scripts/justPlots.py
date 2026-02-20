#!/usr/bin/env python3
"""
Train or Load XGBoost models to predict angle arrays from sensor data.
Includes movement classification and generating publication-quality plots 
(Rose charts, smoothed time series) in PNG, SVG, and EPS formats.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder

# ---------------------------
# CONFIGURATION
# ---------------------------

# Set to FALSE to skip training and just generate plots using saved models
TRAIN_MODELS = False  

# Directories
output_dir = 'AA_hand_included_1221/veronica/models'
train_sensor_dir = "AA_hand_included_1221/veronica/sensor_train"
train_mocap_dir = "AA_hand_included_1221/veronica/aa_train"
test_sensor_dir = "AA_hand_included_1221/veronica/sensor_test"
test_mocap_dir = "AA_hand_included_1221/veronica/aa_test"

# File matching and Pre-processing
STRICT_ROW_MATCH = False
HANDLE_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.01, 0.99
PER_FILE_BASELINE_NORMALIZE = True
PER_FILE_BASELINE_METHOD = "median"  # "median" or "mean"

# XGBoost Hyperparameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'alpha': 0.1,
    'lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
}
NUM_ROUNDS = 5000
EARLY_STOP = 200

# File patterns
SENSOR_SUFFIX = "_CapacitanceTable"
MOCAP_SUFFIX = "_resamp"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# ---------------------------
# PLOTTING METADATA
# ---------------------------
JOINT_NAMES = [
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_yaw_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
]

JOINTS_FOR_PLOTS = [
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
]

DT_MS = 10.0  # Sampling period in ms

# ---------------------------
# HELPERS
# ---------------------------

def clean_value(v):
    s = str(v).strip()
    parts = s.split('.')
    if len(parts) > 2:
        s = '.'.join(parts[:-1])
    return s

_TS_NAME_RE = re.compile(r'(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)', re.IGNORECASE)

def _detect_header(file_path, sample_rows=5):
    peek = pd.read_csv(file_path, nrows=sample_rows, header=None, dtype=str, sep=None, engine='python')
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r'[A-Za-z]', s))).any():
        return 0
    return None

def _drop_timestamp_like_columns(df):
    cols_to_drop = []
    for c in df.columns:
        if _TS_NAME_RE.search(str(c).strip()):
            cols_to_drop.append(c)
    for c in df.columns:
        if c in cols_to_drop: continue
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
    df = _drop_timestamp_like_columns(df)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    df = df.dropna(how='all')
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)
    if handle_outliers and not df.empty:
        df = _winsorize_df(df, lower_q=lower_q, upper_q=upper_q)
    return df.to_numpy(dtype=float)

def _collect_files(dir_path):
    d = Path(dir_path)
    if not d.exists(): raise FileNotFoundError(f"Dir not found: {dir_path}")
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS]
    return files

def _build_keymap(files, suffix):
    m = {}
    for f in files:
        stem = f.stem
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        m[stem] = f
    return m

def match_pairs_strict(sensor_dir, mocap_dir):
    s_files = _collect_files(sensor_dir)
    m_files = _collect_files(mocap_dir)
    s_files = [p for p in s_files if p.stem.endswith(SENSOR_SUFFIX)]
    m_files = [p for p in m_files if p.stem.endswith(MOCAP_SUFFIX)]
    
    smap = _build_keymap(s_files, SENSOR_SUFFIX)
    mmap = _build_keymap(m_files, MOCAP_SUFFIX)
    
    common = sorted(set(smap.keys()) & set(mmap.keys()))
    if not common:
        raise RuntimeError("No matching keys found between sensor and mocap dirs.")
    return [(smap[k], mmap[k], k) for k in common]

def movement_from_key(key):
    k = key.lower().replace('_', ' ')
    mapping = [
        ("hip bend", "hip bend"), ("shoulder rol", "shoulder roll"),
        ("hip twist", "hip twist"), ("bicep", "bicep curl"),
        ("air punch", "air punch"), ("running", "running"),
        ("walking", "walking"), ("lhand", "lhand"), ("rhand", "rhand"),
    ]
    for substr, label in mapping:
        if substr in k: return label
    return "Unknown"

def load_dataset_from_dirs(sensor_dir, mocap_dir, **kwargs):
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_l, y_l, keys, movs, fids = [], [], [], [], []
    
    print(f"[INFO] Loading {len(pairs)} files from {sensor_dir}...")
    for s_p, m_p, k in pairs:
        X = load_and_clean_csv(str(s_p), **kwargs, drop_first_n_cols=0)
        y = load_and_clean_csv(str(m_p), **kwargs, drop_first_n_cols=4)
        
        mn = min(X.shape[0], y.shape[0])
        if kwargs.get('strict_row_match') and X.shape[0] != y.shape[0]:
            raise RuntimeError(f"Row mismatch in {k}")
        
        X, y = X[:mn], y[:mn]
        
        # Per file baseline norm
        if PER_FILE_BASELINE_NORMALIZE:
            base = np.median(X, axis=0, keepdims=True) if PER_FILE_BASELINE_METHOD == "median" else np.mean(X, axis=0, keepdims=True)
            X = X - base

        X_l.append(X)
        y_l.append(y)
        keys.append(k)
        mov_lbl = movement_from_key(k)
        movs.extend([mov_lbl]*len(X))
        fids.extend([k]*len(X))
        
    return np.vstack(X_l), np.vstack(y_l), np.array(movs), np.array(fids), keys

# ---------------------------
# MAIN LOGIC
# ---------------------------

def run_pipeline():
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'test_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. ALWAYS LOAD TEST DATA (Needed for plots)
    print("\n[INFO] Loading Test Data...")
    X_test, y_test_full, test_movements, test_file_ids, test_keys = load_dataset_from_dirs(
        test_sensor_dir, test_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )
    
    num_dims = y_test_full.shape[1]
    
    # Initialize variables for Scalers and Models
    scaler_X = None
    scaler_y = None
    regressors = []
    clf = None
    le = None

    # ---------------------------
    # A) TRAINING MODE
    # ---------------------------
    if TRAIN_MODELS:
        print("\n[INFO] TRAINING MODE: ON")
        print("[INFO] Loading Train Data...")
        X_train, y_train_full, train_movements, _, _ = load_dataset_from_dirs(
            train_sensor_dir, train_mocap_dir,
            handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
            strict_row_match=STRICT_ROW_MATCH
        )

        # Fit Scalers
        scaler_X = RobustScaler().fit(X_train)
        scaler_y = RobustScaler().fit(y_train_full)
        joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train_full)

        # Train Regressors (One per dim)
        dtrain = xgb.DMatrix(X_train_scaled)
        for i in range(num_dims):
            print(f"  Training Regressor Dim {i+1}/{num_dims}")
            dtrain.set_label(y_train_scaled[:, i])
            model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS, verbose_eval=False)
            model.save_model(os.path.join(output_dir, f'xgboost_model_dim_{i+1}.json'))
            regressors.append(model)

        # Train Classifier
        print("  Training Movement Classifier...")
        known_mask = train_movements != "Unknown"
        le = LabelEncoder().fit(train_movements[known_mask])
        y_cls_train = le.transform(train_movements[known_mask])
        clf = XGBClassifier(objective='multi:softprob', n_estimators=500, tree_method='hist')
        clf.fit(X_train_scaled[known_mask], y_cls_train)
        
        joblib.dump(clf, os.path.join(output_dir, "movement_classifier.pkl"))
        joblib.dump(le, os.path.join(output_dir, "movement_label_encoder.pkl"))

    # ---------------------------
    # B) LOADING MODE (INFERENCE)
    # ---------------------------
    else:
        print("\n[INFO] TRAINING MODE: OFF (Loading existing models)")
        try:
            scaler_X = joblib.load(os.path.join(output_dir, 'scaler_X.pkl'))
            scaler_y = joblib.load(os.path.join(output_dir, 'scaler_y.pkl'))
            le = joblib.load(os.path.join(output_dir, "movement_label_encoder.pkl"))
            clf = joblib.load(os.path.join(output_dir, "movement_classifier.pkl"))
            
            for i in range(num_dims):
                model_path = os.path.join(output_dir, f'xgboost_model_dim_{i+1}.json')
                bst = xgb.Booster()
                bst.load_model(model_path)
                regressors.append(bst)
        except FileNotFoundError as e:
            print(f"[ERROR] Could not load models. Did you run with TRAIN_MODELS=True first?\n{e}")
            return

    # ---------------------------
    # PREDICTION (Common to both modes)
    # ---------------------------
    print("\n[INFO] Generating Predictions on Test Set...")
    X_test_scaled = scaler_X.transform(X_test)
    dtest = xgb.DMatrix(X_test_scaled)
    
    y_test_pred_list = []
    for model in regressors:
        y_test_pred_list.append(model.predict(dtest))
    
    y_test_pred_scaled = np.column_stack(y_test_pred_list)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---------------------------
    # PLOTTING
    # ---------------------------
    print(f"\n[INFO] Starting Plotting Phase -> {plots_dir}")

    # Helper indices
    joint_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    joint_indices = [joint_name_to_idx[n] for n in JOINTS_FOR_PLOTS if n in joint_name_to_idx]
    joint_labels = [JOINT_NAMES[idx] for idx in joint_indices]

    # 1. Global Time Series Comparison
    cols = 4
    rows = (num_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    axes = axes.ravel()

    for dim in range(num_dims):
        ax = axes[dim]
        r2 = r2_score(y_test_full[:, dim], y_test_pred_unscaled[:, dim])
        ax.plot(y_test_full[:, dim], label='True', alpha=0.7)
        ax.plot(y_test_pred_unscaled[:, dim], label='Pred', alpha=0.7)
        ax.set_title(f'Dim {dim+1} (R²={r2:.2f})')
        ax.grid(True, alpha=0.3)
    
    for idx in range(num_dims, len(axes)): axes[idx].set_visible(False)
    plt.tight_layout()
    
    base_all = os.path.join(plots_dir, 'all_dims_comparison')
    for ext in ['png', 'svg', 'eps']: plt.savefig(f"{base_all}.{ext}", format=ext, bbox_inches='tight')
    plt.close()

    # 2. Per-File Analysis
    indices_by_file = defaultdict(list)
    for idx, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(idx)

    for fid in sorted(indices_by_file.keys()):
        idxs = np.array(indices_by_file[fid], dtype=int)
        
        # Classifier Probs
        if clf:
            probs = clf.predict_proba(X_test_scaled[idxs]).mean(axis=0)
            # Just print to console
            print(f"[{fid}] Movement Probs: ", end="")
            top3 = sorted(zip(le.classes_, probs), key=lambda x: -x[1])[:3]
            print(" | ".join([f"{c}: {p*100:.1f}%" for c, p in top3]))

        if not joint_indices: continue

        # Regression Errors
        y_true_f = y_test_full[idxs]
        y_pred_f = y_test_pred_unscaled[idxs]
        
        y_true_sel = y_true_f[:, joint_indices]
        y_pred_sel = y_pred_f[:, joint_indices]

        # Calc Range from FULL test set to avoid div by zero in small files
        full_sel = y_test_full[:, joint_indices]
        ranges = full_sel.max(axis=0) - full_sel.min(axis=0)
        ranges[ranges == 0] = 1e-6

        norm_err = np.abs(y_pred_sel - y_true_sel) / ranges
        mean_acc = 100.0 * (1.0 - norm_err.mean(axis=0))
        mean_acc = np.clip(mean_acc, 0.0, 100.0)

        # -----------------------------------------------
        # PLOT A: Polar Bar Chart (Rose Plot)
        # -----------------------------------------------
        if np.any(mean_acc > 0):
            N = len(joint_labels)
            theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
            width = 2 * np.pi / N
            
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')
            bars = ax.bar(theta, mean_acc, width=width, bottom=0.0, align='edge')
            
            # Color map
            cmap = plt.get_cmap("RdYlGn")
            for bar, val in zip(bars, mean_acc):
                bar.set_facecolor(cmap(val / 100.0))
                bar.set_alpha(0.8)
                bar.set_edgecolor('white')

            ax.set_xticks(theta + width / 2)
            ax.set_xticklabels(joint_labels, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', ''], color="gray", fontsize=8)
            ax.grid(True, alpha=0.3)

            for angle, height in zip(theta, mean_acc):
                ax.text(angle + width/2, height - 10, f"{height:.0f}%", 
                        ha='center', va='center', color='black', fontweight='bold', fontsize=8)
            
            plt.title(f"Joint Prediction Accuracy\n{fid}", y=1.08)
            
            base_pol = os.path.join(plots_dir, f"rose_acc_{fid}")
            for ext in ['png', 'svg', 'eps']: plt.savefig(f"{base_pol}.{ext}", format=ext, bbox_inches='tight')
            plt.close()

        # -----------------------------------------------
        # PLOT B: Accuracy vs Time (Smoothed)
        # -----------------------------------------------
        acc_frame = 100.0 * (1.0 - norm_err.mean(axis=1))
        acc_frame = np.clip(acc_frame, 0.0, 100.0)
        
        # Smooth
        win = 15
        acc_smooth = pd.Series(acc_frame).rolling(window=win, center=True, min_periods=1).mean().values
        t_axis = np.arange(len(idxs)) * DT_MS

        plt.figure(figsize=(8, 4))
        plt.plot(t_axis, acc_frame, color='gray', alpha=0.15, label='Raw')
        plt.plot(t_axis, acc_smooth, color='tab:blue', linewidth=2, label='Smoothed')
        
        plt.ylim(60, 100)  # FIXED RANGE
        plt.xlabel("Time (ms)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy Over Time\n{fid}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        base_time = os.path.join(plots_dir, f"time_acc_{fid}")
        for ext in ['png', 'svg', 'eps']: plt.savefig(f"{base_time}.{ext}", format=ext, bbox_inches='tight')
        plt.close()
        
    print("\n[INFO] Done. Check output folder for plots.")

if __name__ == "__main__":
    run_pipeline()