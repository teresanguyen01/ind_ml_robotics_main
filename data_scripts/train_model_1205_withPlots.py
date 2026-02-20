#!/usr/bin/env python3
"""
Train XGBoost models to predict angle arrays (mocap-like) from sensor data,
and also train a movement classifier (Bicep curl, Walking, etc.) using
movement names inferred from filenames.

Additionally, for each TEST file, print a per-file movement probability
distribution, e.g.:

 some_trial_CapacitanceTable.csv:
 air punch : 90.2%
 bicep curl: 5.3%
 walking   : 4.5%
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from matplotlib import cm

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (edit these)
# ---------------------------
# AA_Yan_mocap_included_0120/model25_toddler_0.7
output_dir = 'AA_Yan_mocap_included_0120/model25_toddler_0.7/model'
train_sensor_dir = "AA_Yan_mocap_included_0120/model25_toddler_0.7/sensor_train"  # contains *_CapacitanceTable.csv
train_mocap_dir = "AA_Yan_mocap_included_0120/model25_toddler_0.7/aa_train"      # contains *_resamp.csv
test_sensor_dir = "AA_Yan_mocap_included_0120/model25_toddler_0.7/sensor_test"
test_mocap_dir = "AA_Yan_mocap_included_0120/model25_toddler_0.7/aa_test"

# If True, per-pair X/y row counts must match exactly.
STRICT_ROW_MATCH = False

# Outlier clipping in feature/target space
HANDLE_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.01, 0.99

# Per file baseline normalization for sensor data
# This centers each sensor file so its per column baseline is near zero
PER_FILE_BASELINE_NORMALIZE = True
PER_FILE_BASELINE_METHOD = "median"  # "median" or "mean"

# XGBoost params for regression
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

# (Optional) Feature lagging (currently disabled)
LAGS = 10

# File patterns
SENSOR_SUFFIX = "_CapacitanceTable"
MOCAP_SUFFIX = "_resamp"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# ---------------------------
# NEW: joint metadata for plotting
# ---------------------------
# This list should be aligned with the mocap output columns (y arrays).
# Replace these placeholders with your real joint names in correct order.
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

# Only joints in this list will be included in the per-file pie charts
# and accuracy over time plots.
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

# Sampling period used for the time axis in ms
DT_MS = 10.0  # set to your actual mocap sampling period

# ---------------------------
# Helpers
# ---------------------------

def clean_value(v):
    s = str(v).strip()
    parts = s.split('.')
    if len(parts) > 2:
        s = '.'.join(parts[:-1])
    return s


_TS_NAME_RE = re.compile(r'(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)', re.IGNORECASE)


def _detect_header(file_path, sample_rows=5):
    """Autodetect delimiter and header presence."""
    peek = pd.read_csv(
        file_path,
        nrows=sample_rows,
        header=None,
        dtype=str,
        sep=None,
        engine='python'
    )
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r'[A-Za-z]', s))).any():
        return 0
    return None


def _drop_timestamp_like_columns(df):
    cols_to_drop = []

    # Drop obvious time-like columns by name
    for c in df.columns:
        name = str(c)
        if _TS_NAME_RE.search(name.strip()):
            cols_to_drop.append(c)

    # Drop by content pattern
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


def add_lags(X, lags=10):
    """
    X: [T, F] -> [T, F*(lags+1)], using prefix padding.
    """
    T, F = X.shape
    out = [X]
    pad = np.repeat(X[:1], lags, axis=0)
    Xp = np.vstack([pad, X])  # prefix padding
    for k in range(1, lags + 1):
        out.append(Xp[lags - k:lags - k + T])
    return np.hstack(out)


def _winsorize_df(df, lower_q=0.01, upper_q=0.99):
    q_low = df.quantile(lower_q, axis=0, numeric_only=True)
    q_high = df.quantile(upper_q, axis=0, numeric_only=True)
    return df.clip(lower=q_low, upper=q_high, axis=1)


def load_and_clean_csv(file_path, handle_outliers=True,
                       lower_q=0.01, upper_q=0.99, drop_first_n_cols=0):
    header = _detect_header(file_path)
    df = pd.read_csv(
        file_path,
        header=header,
        dtype=str,
        sep=None,
        engine='python'
    )

    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

    df = df.map(clean_value)
    df = _drop_timestamp_like_columns(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        print(f"Dropping empty columns: {empty_cols}")
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
        stem = stem[:-len(suffix)]
    return stem


def _collect_files(dir_path: str):
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {dir_path}")
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS]
    if not files:
        raise RuntimeError(f"No data files (csv/tsv/txt) found in: {dir_path}")
    return files


def _build_keymap(files, suffix_expected):
    m = {}
    for f in files:
        key = _basename_without_suffix(f, suffix_expected)
        m[key] = f
    return m


def match_pairs_strict(sensor_dir: str, mocap_dir: str):
    sensor_files = _collect_files(sensor_dir)
    mocap_files = _collect_files(mocap_dir)

    sensor_files = [p for p in sensor_files if p.stem.endswith(SENSOR_SUFFIX)]
    mocap_files = [p for p in mocap_files if p.stem.endswith(MOCAP_SUFFIX)]

    if not sensor_files:
        raise RuntimeError(f"No sensor files ending with '{SENSOR_SUFFIX}' found in {sensor_dir}")
    if not mocap_files:
        raise RuntimeError(f"No mocap files ending with '{MOCAP_SUFFIX}' found in {mocap_dir}")

    smap = _build_keymap(sensor_files, SENSOR_SUFFIX)
    mmap = _build_keymap(mocap_files, MOCAP_SUFFIX)

    s_keys = set(smap.keys())
    m_keys = set(mmap.keys())

    only_in_sensor = sorted(s_keys - m_keys)
    only_in_mocap = sorted(m_keys - s_keys)
    if only_in_sensor or only_in_mocap:
        msg = []
        if only_in_sensor:
            msg.append(
                f"Unmatched sensor keys: {only_in_sensor[:8]}{' ...' if len(only_in_sensor) > 8 else ''}"
            )
        if only_in_mocap:
            msg.append(
                f"Unmatched mocap keys: {only_in_mocap[:8]}{' ...' if len(only_in_mocap) > 8 else ''}"
            )
        raise RuntimeError("Sensor/Mocap filename mismatch within split. " + " | ".join(msg))

    keys_sorted = sorted(s_keys)
    return [(smap[k], mmap[k], k) for k in keys_sorted]


def align_X_y(X, y, key, strict_row_match=True):
    """
    Align X and y arrays by trimming the longer one to match the shorter if not strict.
    If strict, raise error on mismatch.
    """
    if X.shape[0] != y.shape[0]:
        if strict_row_match:
            raise RuntimeError(
                f"[{key}] Row mismatch: X={X.shape[0]} vs y={y.shape[0]} (strict mode)."
            )
        else:
            mn = min(X.shape[0], y.shape[0])
            print(f"[WARN] [{key}] Row mismatch -> trimming both to {mn}")
            X = X[:mn]
            y = y[:mn]
    return X, y

# ---------------------------
# Movement label extraction from filename key
# ---------------------------

def movement_from_key(key: str) -> str:
    """
    Map a file key (basename without suffix) to a high-level movement label, using
    substring matches. Adjust mappings to match your actual filename patterns.

    Example movements:
      Shoulder roll
      Hip twist
      Bicep curl
      Air punches
      Hip bend
      Running
      Walking
    """
    k = key.lower().replace('_', ' ')

    # Order matters: more specific phrases first
    mapping = [
        ("hip bend", "hip bend"),
        ("shoulder rol", "shoulder roll"),
        ("hip twist", "hip twist"),
        ("bicep", "bicep curl"),
        ("air punch", "air punch"),
        ("running", "running"),
        ("walking", "walking"),
        ("lhand", "lhand"),
        ("rhand", "rhand"),
    ]

    for substr, label in mapping:
        if substr in k:
            return label

    return "Unknown"

# ---------------------------
# Dataset loader (returns movement + file IDs per row)
# ---------------------------

def load_dataset_from_dirs(sensor_dir: str, mocap_dir: str,
                           handle_outliers=True, lower_q=0.01, upper_q=0.99,
                           strict_row_match=True):
    """
    Returns:
      X_all            : [N, F]
      y_all            : [N, D]
      movements_all    : [N]  movement label per row, e.g. "bicep curl"
      file_ids_all     : [N]  file key per row, e.g. "BicepCurlTrial01"
      keys             : list of keys in this split
    """
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_list, y_list, keys = [], [], []
    movements_all = []
    file_ids_all = []

    print(
        f"[INFO] {len(pairs)} matched pairs found in\n"
        f"  Sensor: {sensor_dir}\n  Mocap : {mocap_dir}"
    )

    for s_path, m_path, key in pairs:
        X = load_and_clean_csv(
            str(s_path),
            handle_outliers=handle_outliers,
            lower_q=lower_q,
            upper_q=upper_q,
            drop_first_n_cols=0
        )
        y = load_and_clean_csv(
            str(m_path),
            handle_outliers=handle_outliers,
            lower_q=lower_q,
            upper_q=upper_q,
            drop_first_n_cols=4
        )

        # Align X and y first
        X, y = align_X_y(X, y, key, strict_row_match=strict_row_match)

        # Safety on NaNs in y
        mask = np.all(~np.isnan(y), axis=1)
        if mask.sum() != y.shape[0]:
            dropped = int(y.shape[0] - mask.sum())
            print(f"[{key}] Dropped {dropped} rows with NaN in y.")
        X = X[mask]
        y = y[mask]

        # Per file baseline normalization for sensor data
        if PER_FILE_BASELINE_NORMALIZE:
            if PER_FILE_BASELINE_METHOD == "median":
                baseline = np.median(X, axis=0, keepdims=True)
            elif PER_FILE_BASELINE_METHOD == "mean":
                baseline = np.mean(X, axis=0, keepdims=True)
            else:
                raise ValueError(f"Unknown PER_FILE_BASELINE_METHOD: {PER_FILE_BASELINE_METHOD}")
            X = X - baseline

        X_list.append(X)
        y_list.append(y)
        keys.append(key)

        movement_label = movement_from_key(key)
        movements_all.extend([movement_label] * len(X))
        file_ids_all.extend([key] * len(X))

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    movements_all = np.array(movements_all)
    file_ids_all = np.array(file_ids_all)

    return X_all, y_all, movements_all, file_ids_all, keys

# ---------------------------
# Training / Evaluation
# ---------------------------

def choose_device_param():
    device = 'cuda'
    try:
        _ = xgb.Booster(params={'device': device})
    except Exception:
        device = 'cpu'
        print("[INFO] CUDA not available; using CPU.")
    return device


def train_and_evaluate():
    os.makedirs(output_dir, exist_ok=True)

    # Load train and test (with movement + file IDs)
    X_train, y_train_full, train_movements, train_file_ids, train_keys = load_dataset_from_dirs(
        train_sensor_dir, train_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )
    X_test, y_test_full, test_movements, test_file_ids, test_keys = load_dataset_from_dirs(
        test_sensor_dir, test_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )

    print(
        f"Shapes -> X_train: {X_train.shape}, y_train: {y_train_full.shape}, "
        f"X_test: {X_test.shape}, y_test: {y_test_full.shape}"
    )

    # Optionally add lags
    # X_train = add_lags(X_train, lags=LAGS)
    # X_test = add_lags(X_test, lags=LAGS)

    # Scale
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train_full)
    y_test_scaled = scaler_y.transform(y_test_full)

    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))

    print("Train y var (scaled):", np.var(y_train_scaled))
    print("Test  y var (scaled):", np.var(y_test_scaled))
    print("Train y var (unscaled):", np.var(y_train_full))
    print("Test  y var (unscaled):", np.var(y_test_full))

    # ---------------------------
    # Train per output regression models
    # ---------------------------
    device_param = choose_device_param()
    params = dict(XGB_PARAMS)
    params['device'] = device_param

    train_matrix = xgb.DMatrix(X_train_scaled)
    test_matrix = xgb.DMatrix(X_test_scaled)

    y_train_pred_list, y_test_pred_list = [], []
    individual_train_r2_scaled, individual_test_r2_scaled = [], []
    models = []

    for i in range(y_train_full.shape[1]):
        print(f"[INFO] Training regression model for dim {i + 1}/{y_train_full.shape[1]}")
        y_tr = y_train_scaled[:, i]
        y_te = y_test_scaled[:, i]
        train_matrix.set_label(y_tr)
        test_matrix.set_label(y_te)

        model = xgb.train(
            params,
            train_matrix,
            num_boost_round=NUM_ROUNDS,
            evals=[(test_matrix, "eval")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )

        model_path = os.path.join(output_dir, f'xgboost_model_dim_{i + 1}.json')
        model.save_model(model_path)
        models.append(model)
        print(f"[Saved] model for dim {i + 1} -> {model_path}")

        y_tr_pred = model.predict(train_matrix)
        y_te_pred = model.predict(test_matrix)
        y_train_pred_list.append(y_tr_pred)
        y_test_pred_list.append(y_te_pred)

        tr_r2 = r2_score(y_tr, y_tr_pred)
        te_r2 = r2_score(y_te, y_te_pred)
        individual_train_r2_scaled.append(tr_r2)
        individual_test_r2_scaled.append(te_r2)
        print(f"Dim {i + 1:02d} R2 train (scaled): {tr_r2:.4f} | R2 test (scaled): {te_r2:.4f}")

    # Stack scaled predictions
    y_train_pred_scaled = np.column_stack(y_train_pred_list)
    y_test_pred_scaled = np.column_stack(y_test_pred_list)

    overall_train_r2_scaled = r2_score(
        y_train_scaled, y_train_pred_scaled, multioutput='variance_weighted'
    )
    overall_test_r2_scaled = r2_score(
        y_test_scaled, y_test_pred_scaled, multioutput='variance_weighted'
    )

    # Inverse transform for unscaled evaluation
    y_train_pred_unscaled = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    overall_train_r2_unscaled = r2_score(
        y_train_full, y_train_pred_unscaled, multioutput='variance_weighted'
    )
    overall_test_r2_unscaled = r2_score(
        y_test_full, y_test_pred_unscaled, multioutput='variance_weighted'
    )

    individual_train_r2_unscaled = [
        r2_score(y_train_full[:, i], y_train_pred_unscaled[:, i])
        for i in range(y_train_full.shape[1])
    ]
    individual_test_r2_unscaled = [
        r2_score(y_test_full[:, i], y_test_pred_unscaled[:, i])
        for i in range(y_test_full.shape[1])
    ]

    print("\nScaled Individual Training R2 per dim:", individual_train_r2_scaled)
    print("Scaled Individual Testing  R2 per dim:", individual_test_r2_scaled)
    print("Scaled Overall Training R2:", overall_train_r2_scaled)
    print("Scaled Overall Testing  R2:", overall_test_r2_scaled)

    print("\nUnscaled Individual Training R2 per dim:", individual_train_r2_unscaled)
    print("Unscaled Individual Testing  R2 per dim:", individual_test_r2_unscaled)
    print("Unscaled Overall Training R2:", overall_train_r2_unscaled)
    print("Unscaled Overall Testing  R2:", overall_test_r2_unscaled)

    # ---------------------------
    # Per movement R2 (on unscaled)
    # ---------------------------
    r2_per_movement = {}
    for mv in np.unique(test_movements):
        mask = (test_movements == mv)
        if mask.sum() < 5:
            continue
        r2_mv = r2_score(
            y_test_full[mask],
            y_test_pred_unscaled[mask],
            multioutput='variance_weighted'
        )
        r2_per_movement[mv] = r2_mv

    print("\nR2 per movement (test):")
    for mv, r2_mv in r2_per_movement.items():
        print(f"  {mv}: {r2_mv:.4f}")

    # ---------------------------
    # Movement classifier (multi class, per frame)
    # ---------------------------
    print("\n[INFO] Training movement classifier...")

    # 1) Drop "Unknown" labels
    known_train_mask = train_movements != "Unknown"
    known_test_mask = test_movements != "Unknown"

    X_train_cls = X_train_scaled[known_train_mask]
    X_test_cls = X_test_scaled[known_test_mask]
    train_mov_known = train_movements[known_train_mask]
    test_mov_known = test_movements[known_test_mask]

    # 2) Encode only known labels
    le = LabelEncoder()
    all_known_movements = np.concatenate([train_mov_known, test_mov_known])
    le.fit(all_known_movements)

    y_cls_train = le.transform(train_mov_known)
    y_cls_test = le.transform(test_mov_known)

    print("Movement classes (encoder):", list(le.classes_))

    # Warn if any labels appear only in test
    train_label_set = set(train_mov_known)
    test_label_set = set(test_mov_known)
    only_in_test = sorted(test_label_set - train_label_set)
    if only_in_test:
        print(
            "[WARN] The following movement labels appear only in TEST; "
            "the classifier has never seen them during training:",
            only_in_test
        )

    # 3) Train classifier on encoded ints
    clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        random_state=42,
        tree_method='hist',
    )

    clf.fit(
        X_train_cls,
        y_cls_train,
        eval_set=[(X_test_cls, y_cls_test)],
        verbose=False,
    )

    joblib.dump(clf, os.path.join(output_dir, "movement_classifier.pkl"))
    joblib.dump(le, os.path.join(output_dir, "movement_label_encoder.pkl"))
    print(f"[Saved] movement classifier and label encoder to {output_dir}")

    # 4) Evaluation on known-label test rows
    y_cls_pred = clf.predict(X_test_cls)
    cls_report = classification_report(
        y_cls_test,
        y_cls_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
    )
    cls_cm = confusion_matrix(
        y_cls_test,
        y_cls_pred,
        labels=np.arange(len(le.classes_)),
    )

    print("\nMovement classification report:\n", cls_report)
    print("Movement confusion matrix:\n", cls_cm)

    # ---------------------------
    # Plotting (global)
    # ---------------------------
    plots_dir = os.path.join(output_dir, 'test_plots')
    os.makedirs(plots_dir, exist_ok=True)

    num_dims = y_test_full.shape[1]
    print(f"\n[INFO] Plotting {num_dims} dimensions.")

    # Map joint names to indices
    if len(JOINT_NAMES) != num_dims:
        print(
            "[WARN] len(JOINT_NAMES) does not match y_test_full.shape[1]. "
            "Check that JOINT_NAMES is aligned with your mocap output columns."
        )

    joint_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    joint_indices_for_plots = [
        joint_name_to_idx[name]
        for name in JOINTS_FOR_PLOTS
        if name in joint_name_to_idx
    ]
    joint_labels_for_plots = [
        JOINT_NAMES[idx] for idx in joint_indices_for_plots
    ]

    if not joint_indices_for_plots:
        print(
            "[WARN] JOINTS_FOR_PLOTS produced an empty index list. "
            "No per joint pies or time plots will be generated."
        )

    # 1. Time series comparison plots (all dims in a grid)
    cols = 4
    rows = (num_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    if rows == 1:
        axes = axes.ravel()
    else:
        axes = axes.ravel()

    for dim in range(num_dims):
        ax = axes[dim]
        time_steps = np.arange(len(y_test_full))
        ax.plot(time_steps, y_test_full[:, dim], label='True', alpha=0.7)
        ax.plot(time_steps, y_test_pred_unscaled[:, dim], label='Pred', alpha=0.7)
        r2_val = r2_score(y_test_full[:, dim], y_test_pred_unscaled[:, dim])
        ax.set_title(f'Dim {dim + 1} (R^2 = {r2_val:.3f})')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(num_dims, len(axes)):
        axes[idx].set_visible(False)

    plt.xlabel('Time Step')
    plt.tight_layout()
    
    # Save All Dims Comparison
    base_all_dims = os.path.join(plots_dir, 'all_dims_comparison')
    plt.savefig(base_all_dims + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(base_all_dims + ".svg", format='svg', bbox_inches='tight')
    plt.savefig(base_all_dims + ".eps", format='eps', bbox_inches='tight')
    plt.close()
    print(f"[Saved] Combined time series plot: {base_all_dims}.[png/svg/eps]")

    # 2. Per dimension MAE bar plot
    errors = y_test_pred_unscaled - y_test_full
    mae_per_dim = np.mean(np.abs(errors), axis=0)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(mae_per_dim)), mae_per_dim)
    plt.xlabel("Dimension index")
    plt.ylabel("MAE (unscaled)")
    plt.title("Per dimension MAE (test set)")
    plt.tight_layout()
    
    # Save MAE Plot
    base_mae = os.path.join(plots_dir, "mae_per_dim")
    plt.savefig(base_mae + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(base_mae + ".svg", format='svg', bbox_inches='tight')
    plt.savefig(base_mae + ".eps", format='eps', bbox_inches='tight')
    plt.close()
    print(f"[Saved] MAE per dimension plot: {base_mae}.[png/svg/eps]")

    # ---------------------------
    # Per file movement mixes (TEST SET ONLY)
    # ---------------------------
    print("\n[INFO] Per-test-file movement probability breakdown:")
    per_file_movement_mix = {}  # for metadata

    # Build index lists per file key (over all test rows)
    indices_by_file = defaultdict(list)
    for idx, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(idx)

    for fid in sorted(indices_by_file.keys()):
        idxs = np.array(indices_by_file[fid], dtype=int)
        X_f = X_test_scaled[idxs]  # use all frames for prediction
        probs = clf.predict_proba(X_f)  # [T_file, n_classes]
        mean_probs = probs.mean(axis=0)  # [n_classes]

        mix = {}
        print(f"\n File key: {fid}")
        for cls_name, p in sorted(
            zip(le.classes_, mean_probs),
            key=lambda x: -x[1]
        ):
            pct = float(p * 100.0)
            mix[str(cls_name)] = pct
            print(f"  {cls_name}: {pct:.1f}%")

        per_file_movement_mix[fid] = mix

        # Plots for the paper
        if not joint_indices_for_plots:
            continue

        y_true_file = y_test_full[idxs]
        y_pred_file = y_test_pred_unscaled[idxs]

        # Subset of joints
        y_true_sel = y_true_file[:, joint_indices_for_plots]
        y_pred_sel = y_pred_file[:, joint_indices_for_plots]

        # Compute ranges for selected joints using the full test set
        full_true_sel = y_test_full[:, joint_indices_for_plots]
        joint_ranges = full_true_sel.max(axis=0) - full_true_sel.min(axis=0)
        joint_ranges[joint_ranges == 0] = 1e-6

        # Absolute and normalized errors for this file
        abs_err_file = np.abs(y_pred_sel - y_true_sel)
        norm_err_file = abs_err_file / joint_ranges

        # 1) Polar Bar Chart (Rose Plot) for Joint Accuracy
        mean_norm_err_per_joint = norm_err_file.mean(axis=0)
        joint_acc_pct = 100.0 * (1.0 - mean_norm_err_per_joint)
        joint_acc_pct = np.clip(joint_acc_pct, 0.0, 100.0)

        if any(a > 0 for a in joint_acc_pct):
            acc_values = np.array(joint_acc_pct)
            
            # Setup Polar Plot
            N = len(joint_labels_for_plots)
            theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
            width = 2 * np.pi / N

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')

            # Bars: Theta=angle, Height=accuracy
            bars = ax.bar(theta, acc_values, width=width, bottom=0.0, align='edge')

            # --- COLOR SETTINGS (NATURE STYLE) ---
            # Use 'YlGnBu' (Yellow-Green-Blue). 
            # Low values = Light Greenish-Yellow
            # High values = Deep Teal/Navy Blue (The "Nature" look)
            cmap = plt.get_cmap("YlGnBu")
            
            # Normalize 60-100 so high accuracy gets the deep teal color
            norm = plt.Normalize(vmin=50, vmax=100) # Slightly wider range for better gradient

            for bar, val in zip(bars, acc_values):
                color_val = norm(val)
                bar.set_facecolor(cmap(color_val))
                bar.set_alpha(0.9) 
                bar.set_edgecolor('white')
                bar.set_linewidth(0.5)

            # Labels
            ax.set_xticks(theta + width / 2)
            ax.set_xticklabels(joint_labels_for_plots, fontsize=9)
            
            # Y-Ticks
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', ''], color="gray", fontsize=8)
            
            # Grid lines
            ax.grid(True, alpha=0.2, color='black')

            # Text annotations
            for angle, height in zip(theta, acc_values):
                text_angle = angle + width / 2
                
                # Use white text for the dark bars (high accuracy)
                text_color = 'white' if height > 80 else 'black'
                
                ax.text(text_angle, height - 10, f"{height:.1f}%", 
                        ha='center', va='center', color=text_color, fontweight='bold', fontsize=8)

            plt.title(f"Joint prediction accuracy\nfile: {fid}", y=1.08)
            plt.tight_layout()

            # Save Polar Plots
            base_polar = os.path.join(plots_dir, f"joint_accuracy_polar_{fid}")
            plt.savefig(base_polar + ".png", dpi=150, bbox_inches="tight")
            plt.savefig(base_polar + ".svg", format='svg', bbox_inches="tight")
            plt.savefig(base_polar + ".eps", format='eps', bbox_inches="tight")
            plt.close()
            print(f"[Saved] per joint accuracy polar plot for {fid} (png/svg/eps)")
        else:
            print(f"[INFO] No valid accuracy values for polar chart for file {fid}")

        # 2) Accuracy over time: Smoothed + Fixed Y-range
        accuracy_per_frame = 100.0 * (1.0 - norm_err_file.mean(axis=1))
        accuracy_per_frame = np.clip(accuracy_per_frame, 0.0, 100.0)

        # Smoothing
        window_size = 15
        acc_series = pd.Series(accuracy_per_frame)
        accuracy_smoothed = acc_series.rolling(window=window_size, center=True, min_periods=1).mean().values

        time_axis_ms = np.arange(len(idxs)) * DT_MS

        plt.figure(figsize=(8, 4))
        # Plot raw (faint) and smoothed (solid)
        plt.plot(time_axis_ms, accuracy_per_frame, color='gray', alpha=0.15, label='Raw')
        plt.plot(time_axis_ms, accuracy_smoothed, color='tab:blue', linewidth=2, label='Smoothed')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Accuracy percentage")
        plt.title(f"Prediction accuracy over time\nfile: {fid}")
        
        # Fixed Y-Axis limits
        plt.ylim(60, 100)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()

        # Save Accuracy vs Time Plots
        base_acc_time = os.path.join(plots_dir, f"accuracy_over_time_{fid}")
        plt.savefig(base_acc_time + ".png", dpi=150, bbox_inches="tight")
        plt.savefig(base_acc_time + ".svg", format='svg', bbox_inches="tight")
        plt.savefig(base_acc_time + ".eps", format='eps', bbox_inches="tight")
        plt.close()
        print(f"[Saved] accuracy over time plot for {fid} (png/svg/eps)")


    # ---------------------------
    # Save metadata for reproducibility
    # ---------------------------
    meta = {
        "train_sensor_dir": train_sensor_dir,
        "train_mocap_dir": train_mocap_dir,
        "test_sensor_dir": test_sensor_dir,
        "test_mocap_dir": test_mocap_dir,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "X_shape_train": list(X_train.shape),
        "y_shape_train": list(y_train_full.shape),
        "X_shape_test": list(X_test.shape),
        "y_shape_test": list(y_test_full.shape),
        "strict_row_match": STRICT_ROW_MATCH,
        "handle_outliers": HANDLE_OUTLIERS,
        "lower_q": LOWER_Q,
        "upper_q": UPPER_Q,
        "per_file_baseline_normalize": PER_FILE_BASELINE_NORMALIZE,
        "per_file_baseline_method": PER_FILE_BASELINE_METHOD,
        "xgb_params": params,
        "num_rounds": NUM_ROUNDS,
        "early_stop": EARLY_STOP,
        "overall_train_r2_unscaled": overall_train_r2_unscaled,
        "overall_test_r2_unscaled": overall_test_r2_unscaled,
        "individual_train_r2_unscaled": individual_train_r2_unscaled,
        "individual_test_r2_unscaled": individual_test_r2_unscaled,
        "overall_train_r2_scaled": overall_train_r2_scaled,
        "overall_test_r2_scaled": overall_test_r2_scaled,
        "individual_train_r2_scaled": individual_train_r2_scaled,
        "individual_test_r2_scaled": individual_test_r2_scaled,
        "r2_per_movement_test": r2_per_movement,
        "movement_classes": list(le.classes_),
        "movement_classification_report": cls_report,
        "movement_confusion_matrix": cls_cm.tolist(),
        "per_file_movement_mix_test": per_file_movement_mix,
        "note": "R2 computed on unscaled data for fair variance comparison",
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] metadata -> {os.path.join(output_dir, 'run_metadata.json')}")

    return {
        "overall_train_r2_unscaled": overall_train_r2_unscaled,
        "overall_test_r2_unscaled": overall_test_r2_unscaled,
        "individual_train_r2_unscaled": individual_train_r2_unscaled,
        "individual_test_r2_unscaled": individual_test_r2_unscaled,
        "r2_per_movement_test": r2_per_movement,
        "per_file_movement_mix_test": per_file_movement_mix,
    }

# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    train_and_evaluate()