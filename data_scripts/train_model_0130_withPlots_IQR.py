#!/usr/bin/env python3
"""
Train XGBoost regressors to predict mocap joint angles from sensor data.
Also trains a movement classifier from filenames.

Paper-grade evaluation/plots:
- Uses mocap CSV headers to align joints by name (prevents column order mistakes)
- Computes accuracy-over-time from GT vs prediction per test file
- Uses robust normalization (IQR from TRAIN only) to convert errors to a percent score
- Saves MAE/RMSE in physical units and normalized metrics to CSV
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

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
# AA_Yan_mocap_included_0120/model5_Yan_sync_long_combo1_0129/model_0130
# AA_Yan_mocap_included_0120/model6_0129_test_Vero
output_dir = "AA_Yan_mocap_included_0120/model6_0129_test_Vero/model_0130"
train_sensor_dir = "AA_Yan_mocap_included_0120/model6_0129_test_Vero/sensor_train"
train_mocap_dir = "AA_Yan_mocap_included_0120/model6_0129_test_Vero/aa_train"
test_sensor_dir = "AA_Yan_mocap_included_0120/model6_0129_test_Vero/sensor_test"
test_mocap_dir = "AA_Yan_mocap_included_0120/model6_0129_test_Vero/aa_test"

STRICT_ROW_MATCH = False

HANDLE_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.01, 0.99

PER_FILE_BASELINE_NORMALIZE = True
PER_FILE_BASELINE_METHOD = "median"  # "median" or "mean"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "alpha": 0.1,
    "lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "rmse",
}
NUM_ROUNDS = 5000
EARLY_STOP = 200

SENSOR_SUFFIX = "_CapacitanceTable"
MOCAP_SUFFIX = "_resamp"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

# Joints to include in per-file polar chart + accuracy over time
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

DT_MS = 10.0  # sampling period (ms)

# Accuracy smoothing window
SMOOTH_WINDOW = 15

# Accuracy axis limits (optional for paper consistency)
ACC_YLIM = (0, 100)  # set to (60,100) if you insist, but (0,100) is more honest.

# ---------------------------
# Helpers
# ---------------------------

def clean_value(v):
    s = str(v).strip()
    parts = s.split(".")
    if len(parts) > 2:
        s = ".".join(parts[:-1])
    return s


_TS_NAME_RE = re.compile(
    r"(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)",
    re.IGNORECASE,
)


def _detect_header(file_path, sample_rows=5):
    """Autodetect delimiter and header presence."""
    peek = pd.read_csv(
        file_path, nrows=sample_rows, header=None, dtype=str, sep=None, engine="python"
    )
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r"[A-Za-z]", s))).any():
        return 0
    return None


def _drop_timestamp_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = []

    for c in df.columns:
        name = str(c)
        if _TS_NAME_RE.search(name.strip()):
            cols_to_drop.append(c)

    for c in df.columns:
        if c in cols_to_drop:
            continue
        col = df[c].astype(str)
        digit_ratio = np.mean(col.str.fullmatch(r"-?\d{10,}").fillna(False))
        iso_ratio = np.mean(col.str.contains(r"\d{4}-\d{2}-\d{2}", na=False))
        if digit_ratio > 0.9 or iso_ratio > 0.6:
            cols_to_drop.append(c)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def _winsorize_df(df, lower_q=0.01, upper_q=0.99):
    q_low = df.quantile(lower_q, axis=0, numeric_only=True)
    q_high = df.quantile(upper_q, axis=0, numeric_only=True)
    return df.clip(lower=q_low, upper=q_high, axis=1)


def load_sensor_as_array(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine="python")
    df = df.map(clean_value)
    df = _drop_timestamp_like_columns(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    df = df.dropna(how="all")
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    if handle_outliers and not df.empty:
        df = _winsorize_df(df, lower_q=lower_q, upper_q=upper_q)

    X = df.to_numpy(dtype=float)

    if PER_FILE_BASELINE_NORMALIZE:
        if PER_FILE_BASELINE_METHOD == "median":
            baseline = np.median(X, axis=0, keepdims=True)
        elif PER_FILE_BASELINE_METHOD == "mean":
            baseline = np.mean(X, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown PER_FILE_BASELINE_METHOD: {PER_FILE_BASELINE_METHOD}")
        X = X - baseline

    return X


def load_mocap_as_dataframe(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99, drop_first_n_cols=4):
    """
    Loads mocap file as a DataFrame WITH column names, then drops first N columns,
    drops timestamp-like columns, converts to numeric, fills NaNs, winsorizes.
    """
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine="python")

    # Drop leading metadata cols if your files contain them (you had 4 before)
    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

    df = df.map(clean_value)

    # keep mocap joint names; but still drop timestamp-like columns if any survived
    df = _drop_timestamp_like_columns(df)

    # numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    df = df.dropna(how="all")
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    if handle_outliers and not df.empty:
        df = _winsorize_df(df, lower_q=lower_q, upper_q=upper_q)

    return df


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
            msg.append(f"Unmatched sensor keys: {only_in_sensor[:8]}{' ...' if len(only_in_sensor) > 8 else ''}")
        if only_in_mocap:
            msg.append(f"Unmatched mocap keys: {only_in_mocap[:8]}{' ...' if len(only_in_mocap) > 8 else ''}")
        raise RuntimeError("Sensor/Mocap filename mismatch within split. " + " | ".join(msg))

    keys_sorted = sorted(s_keys)
    return [(smap[k], mmap[k], k) for k in keys_sorted]


def align_X_y(X, y_df: pd.DataFrame, key, strict_row_match=True):
    y = y_df.to_numpy(dtype=float)
    if X.shape[0] != y.shape[0]:
        if strict_row_match:
            raise RuntimeError(f"[{key}] Row mismatch: X={X.shape[0]} vs y={y.shape[0]} (strict mode).")
        mn = min(X.shape[0], y.shape[0])
        print(f"[WARN] [{key}] Row mismatch -> trimming both to {mn}")
        X = X[:mn]
        y_df = y_df.iloc[:mn].copy()
    return X, y_df


def movement_from_key(key: str) -> str:
    k = key.lower().replace("_", " ")
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


def load_dataset_from_dirs(sensor_dir: str, mocap_dir: str, handle_outliers=True, lower_q=0.01, upper_q=0.99, strict_row_match=True):
    """
    Returns:
      X_all: [N, F] concatenated
      y_all: [N, D] concatenated (numpy)
      y_cols: list of mocap column names (D)
      movements_all: [N]
      file_ids_all: [N]
      keys: list of file keys
    """
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_list = []
    y_list = []
    keys = []
    movements_all = []
    file_ids_all = []
    y_cols = None

    print(f"[INFO] {len(pairs)} matched pairs found in\n Sensor: {sensor_dir}\n Mocap : {mocap_dir}")

    for s_path, m_path, key in pairs:
        X = load_sensor_as_array(str(s_path), handle_outliers=handle_outliers, lower_q=lower_q, upper_q=upper_q)
        y_df = load_mocap_as_dataframe(str(m_path), handle_outliers=handle_outliers, lower_q=lower_q, upper_q=upper_q, drop_first_n_cols=4)

        X, y_df = align_X_y(X, y_df, key, strict_row_match=strict_row_match)

        # Drop rows with NaN in y (safety)
        y = y_df.to_numpy(dtype=float)
        mask = np.all(~np.isnan(y), axis=1)
        if mask.sum() != y.shape[0]:
            dropped = int(y.shape[0] - mask.sum())
            print(f"[{key}] Dropped {dropped} rows with NaN in y.")
            X = X[mask]
            y_df = y_df.iloc[mask].copy()

        if y_cols is None:
            y_cols = list(y_df.columns)
        else:
            # Enforce same columns across files
            if list(y_df.columns) != y_cols:
                raise RuntimeError(
                    f"[{key}] Mocap columns differ from earlier files.\n"
                    f"Expected: {y_cols}\nGot: {list(y_df.columns)}"
                )

        X_list.append(X)
        y_list.append(y_df.to_numpy(dtype=float))
        keys.append(key)

        mv = movement_from_key(key)
        movements_all.extend([mv] * len(X))
        file_ids_all.extend([key] * len(X))

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    movements_all = np.array(movements_all)
    file_ids_all = np.array(file_ids_all)

    return X_all, y_all, y_cols, movements_all, file_ids_all, keys


def choose_device_param():
    device = "cuda"
    try:
        _ = xgb.Booster(params={"device": device})
    except Exception:
        device = "cpu"
        print("[INFO] CUDA not available; using CPU.")
    return device


def robust_scale_from_train(y_train_full: np.ndarray, eps: float = 1e-6):
    """
    Robust per-dimension scale using IQR (q75 - q25) on TRAIN ONLY.
    This is used only for normalization in reporting percent-style accuracy.
    """
    q75 = np.percentile(y_train_full, 75, axis=0)
    q25 = np.percentile(y_train_full, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr < eps] = eps
    return iqr


def sanity_check_joint_channels(y_full: np.ndarray, y_cols: list, joint_a: str, joint_b: str):
    """
    Debug helper: detect identical/duplicated channels.
    """
    if joint_a not in y_cols or joint_b not in y_cols:
        return
    i = y_cols.index(joint_a)
    j = y_cols.index(joint_b)
    same = np.allclose(y_full[:, i], y_full[:, j], atol=1e-8)
    corr = np.corrcoef(y_full[:, i], y_full[:, j])[0, 1]
    print(f"[CHECK] {joint_a} vs {joint_b}: allclose={same} corr={corr:.4f}")


def plot_accuracy_over_time(time_ms, acc_raw, acc_smooth, fid, plots_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(time_ms, acc_raw, alpha=0.2, label="Raw")
    plt.plot(time_ms, acc_smooth, linewidth=2, label="Smoothed")
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized accuracy (%)")
    plt.title(f"Prediction accuracy over time\nfile: {fid}")
    plt.ylim(*ACC_YLIM)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    base = os.path.join(plots_dir, f"accuracy_over_time_{fid}")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(base + ".eps", format="eps", bbox_inches="tight")
    plt.close()
    print(f"[Saved] accuracy over time plot for {fid} (png/svg/eps)")


def plot_joint_accuracy_polar(joint_labels, acc_values, fid, plots_dir):
    # Polar bar chart
    N = len(joint_labels)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    bars = ax.bar(theta, acc_values, width=width, bottom=0.0, align="edge")

    cmap = plt.get_cmap("YlGnBu")
    norm = plt.Normalize(vmin=50, vmax=100)

    for bar, val in zip(bars, acc_values):
        bar.set_facecolor(cmap(norm(val)))
        bar.set_alpha(0.9)
        bar.set_edgecolor("white")
        bar.set_linewidth(0.5)

    ax.set_xticks(theta + width / 2)
    ax.set_xticklabels(joint_labels, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", ""], fontsize=8)

    ax.grid(True, alpha=0.2)

    for angle, height in zip(theta, acc_values):
        text_angle = angle + width / 2
        text_color = "white" if height > 80 else "black"
        ax.text(text_angle, max(height - 10, 2), f"{height:.1f}%", ha="center", va="center",
                color=text_color, fontweight="bold", fontsize=8)

    plt.title(f"Per-joint normalized accuracy\nfile: {fid}", y=1.08)
    plt.tight_layout()

    base = os.path.join(plots_dir, f"joint_accuracy_polar_{fid}")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(base + ".eps", format="eps", bbox_inches="tight")
    plt.close()
    print(f"[Saved] per joint accuracy polar plot for {fid} (png/svg/eps)")


def train_and_evaluate():
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "test_plots")
    os.makedirs(plots_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Load train and test
    X_train, y_train_full, y_cols, train_movements, train_file_ids, train_keys = load_dataset_from_dirs(
        train_sensor_dir, train_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )
    X_test, y_test_full, y_cols_test, test_movements, test_file_ids, test_keys = load_dataset_from_dirs(
        test_sensor_dir, test_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )
    if y_cols_test != y_cols:
        raise RuntimeError("Train/test mocap columns differ. Cannot evaluate reliably.")

    print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train_full.shape}, X_test: {X_test.shape}, y_test: {y_test_full.shape}")

    # Sanity check for the “two joints same” issue (prints correlation + equality)
    sanity_check_joint_channels(y_test_full, y_cols, "waist_yaw_joint", "left_shoulder_pitch_joint")

    # Scale X and y for learning (model space)
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train_full)
    y_test_scaled = scaler_y.transform(y_test_full)

    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))

    # Train per-dimension regressors
    device_param = choose_device_param()
    params = dict(XGB_PARAMS)
    params["device"] = device_param

    train_matrix = xgb.DMatrix(X_train_scaled)
    test_matrix = xgb.DMatrix(X_test_scaled)

    y_train_pred_list = []
    y_test_pred_list = []
    individual_train_r2_scaled = []
    individual_test_r2_scaled = []
    models = []

    D = y_train_full.shape[1]

    for i in range(D):
        print(f"[INFO] Training regression model for dim {i + 1}/{D}")
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

        model_path = os.path.join(output_dir, f"xgboost_model_dim_{i + 1}.json")
        model.save_model(model_path)
        models.append(model)

        y_tr_pred = model.predict(train_matrix)
        y_te_pred = model.predict(test_matrix)
        y_train_pred_list.append(y_tr_pred)
        y_test_pred_list.append(y_te_pred)

        tr_r2 = r2_score(y_tr, y_tr_pred)
        te_r2 = r2_score(y_te, y_te_pred)
        individual_train_r2_scaled.append(tr_r2)
        individual_test_r2_scaled.append(te_r2)

    y_train_pred_scaled = np.column_stack(y_train_pred_list)
    y_test_pred_scaled = np.column_stack(y_test_pred_list)

    # Unscale predictions for meaningful error in angle units
    y_train_pred_unscaled = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    # Overall / per-dim R2 on unscaled
    overall_train_r2_unscaled = r2_score(y_train_full, y_train_pred_unscaled, multioutput="variance_weighted")
    overall_test_r2_unscaled = r2_score(y_test_full, y_test_pred_unscaled, multioutput="variance_weighted")

    individual_train_r2_unscaled = [r2_score(y_train_full[:, i], y_train_pred_unscaled[:, i]) for i in range(D)]
    individual_test_r2_unscaled = [r2_score(y_test_full[:, i], y_test_pred_unscaled[:, i]) for i in range(D)]

    print(f"[RESULT] Overall R2 train (unscaled): {overall_train_r2_unscaled:.4f}")
    print(f"[RESULT] Overall R2 test  (unscaled): {overall_test_r2_unscaled:.4f}")

    # Robust normalization scale for percent-style accuracy (TRAIN ONLY, no leakage)
    norm_scale = robust_scale_from_train(y_train_full)  # shape [D]

    # ---------------------------
    # Movement classifier (same as before)
    # ---------------------------
    print("\n[INFO] Training movement classifier...")

    known_train_mask = train_movements != "Unknown"
    known_test_mask = test_movements != "Unknown"

    X_train_cls = X_train_scaled[known_train_mask]
    X_test_cls = X_test_scaled[known_test_mask]
    train_mov_known = train_movements[known_train_mask]
    test_mov_known = test_movements[known_test_mask]

    le = LabelEncoder()
    all_known_movements = np.concatenate([train_mov_known, test_mov_known])
    le.fit(all_known_movements)

    y_cls_train = le.transform(train_mov_known)
    y_cls_test = le.transform(test_mov_known)

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        random_state=42,
        tree_method="hist",
    )

    clf.fit(X_train_cls, y_cls_train, eval_set=[(X_test_cls, y_cls_test)], verbose=False)

    joblib.dump(clf, os.path.join(output_dir, "movement_classifier.pkl"))
    joblib.dump(le, os.path.join(output_dir, "movement_label_encoder.pkl"))

    y_cls_pred = clf.predict(X_test_cls)
    cls_report = classification_report(y_cls_test, y_cls_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_)
    cls_cm = confusion_matrix(y_cls_test, y_cls_pred, labels=np.arange(len(le.classes_)))

    print("\nMovement classification report:\n", cls_report)
    print("Movement confusion matrix:\n", cls_cm)

    # ---------------------------
    # Global MAE / RMSE per dimension (paper friendly)
    # ---------------------------
    errors = y_test_pred_unscaled - y_test_full
    mae_per_dim = np.mean(np.abs(errors), axis=0)
    rmse_per_dim = np.sqrt(np.mean(errors**2, axis=0))

    df_global = pd.DataFrame({
        "joint": y_cols,
        "mae": mae_per_dim,
        "rmse": rmse_per_dim,
        "r2": individual_test_r2_unscaled
    })
    df_global.to_csv(os.path.join(tables_dir, "test_per_joint_metrics.csv"), index=False)

    # ---------------------------
    # Per file movement mixes + per-file plots and tables
    # ---------------------------
    print("\n[INFO] Per-test-file movement probability breakdown:")
    per_file_movement_mix = {}

    indices_by_file = defaultdict(list)
    for idx, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(idx)

    # Determine plot joint indices by name (from mocap headers)
    missing = [j for j in JOINTS_FOR_PLOTS if j not in y_cols]
    if missing:
        raise RuntimeError(f"These JOINTS_FOR_PLOTS are missing from mocap columns: {missing}")

    joint_indices_for_plots = [y_cols.index(j) for j in JOINTS_FOR_PLOTS]
    assert len(set(joint_indices_for_plots)) == len(joint_indices_for_plots), "Duplicate joint indices for plots."

    # Main loop per file
    for fid in sorted(indices_by_file.keys()):
        idxs = np.array(indices_by_file[fid], dtype=int)

        # movement mix
        X_f = X_test_scaled[idxs]
        probs = clf.predict_proba(X_f)
        mean_probs = probs.mean(axis=0)

        mix = {}
        print(f"\n File key: {fid}")
        for cls_name, p in sorted(zip(le.classes_, mean_probs), key=lambda x: -x[1]):
            pct = float(p * 100.0)
            mix[str(cls_name)] = pct
            print(f" {cls_name}: {pct:.1f}%")
        per_file_movement_mix[fid] = mix

        # Extract GT and pred for this file (UNSCALED angles)
        y_true_file = y_test_full[idxs]
        y_pred_file = y_test_pred_unscaled[idxs]

        # Subset joints for plots
        y_true_sel = y_true_file[:, joint_indices_for_plots]
        y_pred_sel = y_pred_file[:, joint_indices_for_plots]

        # Errors in physical units
        err = y_pred_sel - y_true_sel
        abs_err = np.abs(err)
        se = err**2

        # Normalize errors using TRAIN IQR for those joints (paper defensible)
        scale_sel = norm_scale[joint_indices_for_plots]
        norm_abs_err = abs_err / scale_sel  # [T, J]

        # Convert to percent style accuracy (0..100), then clip
        acc_per_frame = 100.0 * (1.0 - np.mean(norm_abs_err, axis=1))
        acc_per_frame = np.clip(acc_per_frame, 0.0, 100.0)

        # Smooth for presentation
        acc_series = pd.Series(acc_per_frame)
        acc_smoothed = acc_series.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean().values

        time_axis_ms = np.arange(len(idxs)) * DT_MS
        plot_accuracy_over_time(time_axis_ms, acc_per_frame, acc_smoothed, fid, plots_dir)

        # Per-joint accuracy (mean over time)
        mean_norm_abs_err_per_joint = np.mean(norm_abs_err, axis=0)  # [J]
        joint_acc_pct = 100.0 * (1.0 - mean_norm_abs_err_per_joint)
        joint_acc_pct = np.clip(joint_acc_pct, 0.0, 100.0)

        plot_joint_accuracy_polar(JOINTS_FOR_PLOTS, joint_acc_pct, fid, plots_dir)

        # Save per-frame table (paper supplement)
        # Also include MAE/RMSE per frame aggregated over selected joints
        mae_frame = np.mean(abs_err, axis=1)
        rmse_frame = np.sqrt(np.mean(se, axis=1))

        df_frame = pd.DataFrame({
            "time_ms": time_axis_ms,
            "acc_percent_norm_iqr": acc_per_frame,
            "acc_percent_norm_iqr_smoothed": acc_smoothed,
            "mae_selected_joints": mae_frame,
            "rmse_selected_joints": rmse_frame,
        })
        df_frame.to_csv(os.path.join(tables_dir, f"{fid}_frame_metrics.csv"), index=False)

        # Save per-joint summary table for this file
        df_joint = pd.DataFrame({
            "joint": JOINTS_FOR_PLOTS,
            "mae": np.mean(abs_err, axis=0),
            "rmse": np.sqrt(np.mean(se, axis=0)),
            "acc_percent_norm_iqr": joint_acc_pct,
        })
        df_joint.to_csv(os.path.join(tables_dir, f"{fid}_per_joint_metrics.csv"), index=False)

    # ---------------------------
    # Save metadata
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
        "mocap_columns": y_cols,
        "strict_row_match": STRICT_ROW_MATCH,
        "handle_outliers": HANDLE_OUTLIERS,
        "lower_q": LOWER_Q,
        "upper_q": UPPER_Q,
        "per_file_baseline_normalize": PER_FILE_BASELINE_NORMALIZE,
        "per_file_baseline_method": PER_FILE_BASELINE_METHOD,
        "xgb_params": params,
        "num_rounds": NUM_ROUNDS,
        "early_stop": EARLY_STOP,
        "overall_train_r2_unscaled": float(overall_train_r2_unscaled),
        "overall_test_r2_unscaled": float(overall_test_r2_unscaled),
        "individual_train_r2_unscaled": [float(x) for x in individual_train_r2_unscaled],
        "individual_test_r2_unscaled": [float(x) for x in individual_test_r2_unscaled],
        "individual_train_r2_scaled": [float(x) for x in individual_train_r2_scaled],
        "individual_test_r2_scaled": [float(x) for x in individual_test_r2_scaled],
        "movement_classes": list(le.classes_),
        "movement_classification_report": cls_report,
        "movement_confusion_matrix": cls_cm.tolist(),
        "per_file_movement_mix_test": per_file_movement_mix,
        "accuracy_definition": "acc% = 100*(1 - mean_j(|err_j| / IQR_train_j)), clipped to [0,100]",
        "normalization": "IQR computed from training mocap only (no test leakage)",
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] metadata -> {os.path.join(output_dir, 'run_metadata.json')}")

    return meta


if __name__ == "__main__":
    train_and_evaluate()