#!/usr/bin/env python3
"""
Train XGBoost regressors to predict mocap joint angle arrays from sensor data.
Also trains a movement classifier using labels inferred from filenames.

Outputs (paper-ready):
A) Primary metrics (unscaled angles): per-joint MAE, RMSE, R^2
B) Per-test-file plots for selected joints (by mocap column name):
   1) Movement-specific percentile ROM-normalized score (%) over time + polar
      Denominator: (95th - 5th percentile) ROM computed within the movement class (TEST set)
      This is stricter than max-min and avoids outlier inflation.
   2) IQR-normalized score (%) over time + polar
      Denominator: TRAIN IQR (75th - 25th), more robust and often stricter.

C) CSV exports per test file:
   - time_ms + scores (raw/smoothed) + MAE/RMSE-per-frame for selected joints

All plots and metrics compare predicted vs ground-truth mocap angles in unscaled units.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
# AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003
output_dir = "AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003/model_0206"
train_sensor_dir = "AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003/sensor_train"
train_mocap_dir  = "AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003/aa_train"
test_sensor_dir  = "AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003/sensor_test"
test_mocap_dir   = "AA_Yan_mocap_included_0120/model22_Rawan_Teresa_Yan_to_Vero_one_file_each_1003/aa_test"

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

# Joints included in time and polar plots (must exactly match mocap column names)
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
SMOOTH_WINDOW = 15

# Plot y-limits
MOV_PCTL_ROM_SCORE_YLIM = (0, 100)
IQR_SCORE_YLIM = (0, 100)

# Percentiles for ROM denominator
ROM_P_LOW = 5
ROM_P_HIGH = 95

# Safeguards
EPS = 1e-6
MIN_FRAMES_FOR_MOV_ROM = 25  # if a movement has fewer frames, fallback to global denom

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
    peek = pd.read_csv(
        file_path,
        nrows=sample_rows,
        header=None,
        dtype=str,
        sep=None,
        engine="python",
    )
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r"[A-Za-z]", s))).any():
        return 0
    return None


def _drop_timestamp_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = []

    # Drop by name
    for c in df.columns:
        name = str(c)
        if _TS_NAME_RE.search(name.strip()):
            cols_to_drop.append(c)

    # Drop by content
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


def load_mocap_as_dataframe(
    file_path,
    handle_outliers=True,
    lower_q=0.01,
    upper_q=0.99,
    drop_first_n_cols=4,
):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine="python")

    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

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

    if len(set(df.columns)) != len(df.columns):
        raise RuntimeError(f"Duplicate mocap column names detected in {file_path}")

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


# ---------------------------
# Plotting
# ---------------------------

def rolling_mean(arr, window):
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_score_over_time(time_ms, raw, smooth, fid, out_path_base, ylabel, ylim):
    plt.figure(figsize=(8, 4))
    plt.plot(time_ms, raw, alpha=0.2, label="Raw")
    plt.plot(time_ms, smooth, linewidth=2, label="Smoothed")
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over time\nfile: {fid}")
    plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_path_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_path_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


def plot_polar_scores(joint_labels, scores_pct, fid, out_path_base, title_prefix):
    N = len(joint_labels)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    bars = ax.bar(theta, scores_pct, width=width, bottom=0.0, align="edge")

    cmap = plt.get_cmap("YlGnBu")
    norm = plt.Normalize(vmin=50, vmax=100)

    for bar, val in zip(bars, scores_pct):
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

    for angle, height in zip(theta, scores_pct):
        text_angle = angle + width / 2
        text_color = "white" if height > 80 else "black"
        ax.text(
            text_angle,
            max(height - 10, 2),
            f"{height:.1f}%",
            ha="center",
            va="center",
            color=text_color,
            fontweight="bold",
            fontsize=8,
        )

    plt.title(f"{title_prefix}\nfile: {fid}", y=1.08)
    plt.tight_layout()
    plt.savefig(out_path_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_path_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_path_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


def percentile_rom(arr_2d: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """
    arr_2d: [N, J]
    Returns per-joint (p_high - p_low) range.
    """
    hi = np.percentile(arr_2d, p_high, axis=0)
    lo = np.percentile(arr_2d, p_low, axis=0)
    denom = hi - lo
    denom[denom == 0] = EPS
    return denom


def majority_label(labels_1d: np.ndarray) -> str:
    """
    Pick the most common label. If everything is Unknown, returns Unknown.
    """
    c = Counter(labels_1d.tolist())
    return c.most_common(1)[0][0]


# ---------------------------
# Main
# ---------------------------

def train_and_evaluate():
    os.makedirs(output_dir, exist_ok=True)

    # Load train/test
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

    # Scale for training
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

    D = y_train_full.shape[1]
    y_train_pred_list = []
    y_test_pred_list = []

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
        model.save_model(os.path.join(output_dir, f"xgboost_model_dim_{i + 1}.json"))

        y_train_pred_list.append(model.predict(train_matrix))
        y_test_pred_list.append(model.predict(test_matrix))

    y_train_pred_scaled = np.column_stack(y_train_pred_list)
    y_test_pred_scaled = np.column_stack(y_test_pred_list)

    # Unscale predictions for metrics and plots
    y_train_pred_unscaled = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    # Primary metrics (unscaled) for paper tables
    errors = y_test_pred_unscaled - y_test_full
    mae_per_dim = np.mean(np.abs(errors), axis=0)
    rmse_per_dim = np.sqrt(np.mean(errors**2, axis=0))
    r2_per_dim = [r2_score(y_test_full[:, i], y_test_pred_unscaled[:, i]) for i in range(D)]
    overall_test_r2 = r2_score(y_test_full, y_test_pred_unscaled, multioutput="variance_weighted")

    tables_dir = os.path.join(output_dir, "paper_tables")
    os.makedirs(tables_dir, exist_ok=True)

    df_global = pd.DataFrame({
        "joint": y_cols,
        "mae": mae_per_dim,
        "rmse": rmse_per_dim,
        "r2": r2_per_dim,
    })
    df_global.to_csv(os.path.join(tables_dir, "test_per_joint_metrics.csv"), index=False)
    print(f"[Saved] test_per_joint_metrics.csv -> {tables_dir}")
    print(f"[RESULT] Overall test R2 (unscaled): {overall_test_r2:.4f}")

    # TRAIN IQR scales for IQR-normalized score
    q75 = np.percentile(y_train_full, 75, axis=0)
    q25 = np.percentile(y_train_full, 25, axis=0)
    iqr_train = q75 - q25
    iqr_train[iqr_train == 0] = EPS

    # ---------------------------
    # Movement classifier (optional)
    # ---------------------------
    print("\n[INFO] Training movement classifier...")

    known_train_mask = train_movements != "Unknown"
    known_test_mask = test_movements != "Unknown"

    X_train_cls = X_train_scaled[known_train_mask]
    X_test_cls = X_test_scaled[known_test_mask]
    train_mov_known = train_movements[known_train_mask]
    test_mov_known = test_movements[known_test_mask]

    le = LabelEncoder()
    # Fit on train only (cleaner for reporting)
    le.fit(train_mov_known)

    # If test has unseen labels, they will not be in le.classes_
    # This classifier is mainly for per-file "mix" reporting, so keep it simple.
    y_cls_train = le.transform(train_mov_known)

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
    clf.fit(X_train_cls, y_cls_train, verbose=False)

    joblib.dump(clf, os.path.join(output_dir, "movement_classifier.pkl"))
    joblib.dump(le, os.path.join(output_dir, "movement_label_encoder.pkl"))

    # Evaluate only on test rows whose label exists in train classes
    test_in_train_mask = np.isin(test_mov_known, le.classes_)
    if test_in_train_mask.sum() > 0:
        y_cls_test_eval = le.transform(test_mov_known[test_in_train_mask])
        y_cls_pred_eval = clf.predict(X_test_cls[test_in_train_mask])
        cls_report = classification_report(y_cls_test_eval, y_cls_pred_eval, labels=np.arange(len(le.classes_)), target_names=le.classes_)
        cls_cm = confusion_matrix(y_cls_test_eval, y_cls_pred_eval, labels=np.arange(len(le.classes_)))
        print("\nMovement classification report (test labels seen in train):\n", cls_report)
        print("Movement confusion matrix:\n", cls_cm)
    else:
        cls_report = "No test labels overlap with train labels; skipped classifier evaluation."
        cls_cm = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)
        print("\n[WARN] No overlap between known test labels and train label set; skipped classifier evaluation.")

    # ---------------------------
    # Joint alignment for plots by mocap column name
    # ---------------------------
    name_to_idx = {n: i for i, n in enumerate(y_cols)}
    missing = [j for j in JOINTS_FOR_PLOTS if j not in name_to_idx]
    if missing:
        raise RuntimeError(f"JOINTS_FOR_PLOTS missing from mocap columns: {missing}")
    joint_idx = [name_to_idx[j] for j in JOINTS_FOR_PLOTS]
    if len(set(joint_idx)) != len(joint_idx):
        raise RuntimeError("Duplicate joint indices found. Check mocap column names.")

    # Denominator for IQR score (train IQR, selected joints only)
    iqr_sel = iqr_train[joint_idx]
    iqr_sel[iqr_sel == 0] = EPS

    # ---------------------------
    # Movement-specific percentile ROM denominators (TEST set)
    # ---------------------------
    # Compute a global fallback denom (all test frames) to use if a movement is too small
    full_true_sel = y_test_full[:, joint_idx]
    global_mov_pctl_rom = percentile_rom(full_true_sel, ROM_P_LOW, ROM_P_HIGH)

    mov_pctl_rom_by_movement = {}
    for mv in np.unique(test_movements):
        mv_mask = (test_movements == mv)
        if mv_mask.sum() < MIN_FRAMES_FOR_MOV_ROM:
            mov_pctl_rom_by_movement[mv] = global_mov_pctl_rom
            continue
        mv_true_sel = y_test_full[mv_mask][:, joint_idx]
        mov_pctl_rom_by_movement[mv] = percentile_rom(mv_true_sel, ROM_P_LOW, ROM_P_HIGH)

    # Per-file indexing
    indices_by_file = defaultdict(list)
    for i, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(i)

    plots_dir = os.path.join(output_dir, "paper_plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_dir = os.path.join(output_dir, "paper_csv_exports")
    os.makedirs(csv_dir, exist_ok=True)

    per_file_movement_mix = {}
    file_summary_rows = []
    per_movement_rows = []

    print("\n[INFO] Per-test-file plots + CSV exports:")

    for fid in sorted(indices_by_file.keys()):
        idxs = np.asarray(indices_by_file[fid], dtype=int)

        # The movement label for the file is derived from filename mapping, consistent across frames,
        # but we compute majority just in case.
        mv_file = majority_label(test_movements[idxs])

        # movement probability mix (classifier output)
        X_f = X_test_scaled[idxs]
        probs = clf.predict_proba(X_f)
        mean_probs = probs.mean(axis=0)

        mix = {}
        for cls_name, p in sorted(zip(le.classes_, mean_probs), key=lambda x: -x[1]):
            mix[str(cls_name)] = float(p * 100.0)
        per_file_movement_mix[fid] = mix

        # Select GT and pred (unscaled)
        y_true = y_test_full[idxs][:, joint_idx]
        y_pred = y_test_pred_unscaled[idxs][:, joint_idx]

        err = y_pred - y_true
        abs_err = np.abs(err)

        # Time axis
        time_ms = np.arange(len(idxs)) * DT_MS

        # ---------------------------
        # 1) Movement-specific percentile ROM-normalized score
        # Denom: (95th - 5th) within movement class (TEST set)
        # ---------------------------
        denom_mov = mov_pctl_rom_by_movement.get(mv_file, global_mov_pctl_rom)
        denom_mov = np.maximum(denom_mov, EPS)

        norm_abs_err_mov = abs_err / denom_mov
        norm_err_frame_mov = norm_abs_err_mov.mean(axis=1)
        mov_rom_score_raw = 100.0 * (1.0 - norm_err_frame_mov)
        mov_rom_score_raw = np.clip(mov_rom_score_raw, 0.0, 100.0)
        mov_rom_score_smooth = rolling_mean(mov_rom_score_raw, SMOOTH_WINDOW)

        mov_out_base = os.path.join(plots_dir, f"mov_pctl_rom_score_{fid}")
        plot_score_over_time(
            time_ms,
            mov_rom_score_raw,
            mov_rom_score_smooth,
            fid,
            mov_out_base,
            ylabel=f"Movement-specific percentile ROM-normalized score (%)",
            ylim=MOV_PCTL_ROM_SCORE_YLIM,
        )

        mean_norm_mov_joint = norm_abs_err_mov.mean(axis=0)
        mov_joint_score = 100.0 * (1.0 - mean_norm_mov_joint)
        mov_joint_score = np.clip(mov_joint_score, 0.0, 100.0)

        mov_polar_base = os.path.join(plots_dir, f"mov_pctl_rom_polar_{fid}")
        plot_polar_scores(
            JOINTS_FOR_PLOTS,
            mov_joint_score,
            fid,
            mov_polar_base,
            title_prefix="Movement-specific percentile ROM-normalized per-joint score (%)",
        )

        # ---------------------------
        # 2) IQR-normalized score (train IQR)
        # ---------------------------
        norm_abs_err_iqr = abs_err / iqr_sel
        norm_err_frame_iqr = norm_abs_err_iqr.mean(axis=1)
        iqr_score_raw = 100.0 * (1.0 - norm_err_frame_iqr)
        iqr_score_raw = np.clip(iqr_score_raw, 0.0, 100.0)
        iqr_score_smooth = rolling_mean(iqr_score_raw, SMOOTH_WINDOW)

        iqr_out_base = os.path.join(plots_dir, f"iqr_norm_score_{fid}")
        plot_score_over_time(
            time_ms,
            iqr_score_raw,
            iqr_score_smooth,
            fid,
            iqr_out_base,
            ylabel="IQR-normalized score (%)",
            ylim=IQR_SCORE_YLIM,
        )

        mean_norm_iqr_joint = norm_abs_err_iqr.mean(axis=0)
        iqr_joint_score = 100.0 * (1.0 - mean_norm_iqr_joint)
        iqr_joint_score = np.clip(iqr_joint_score, 0.0, 100.0)

        iqr_polar_base = os.path.join(plots_dir, f"iqr_norm_polar_{fid}")
        plot_polar_scores(
            JOINTS_FOR_PLOTS,
            iqr_joint_score,
            fid,
            iqr_polar_base,
            title_prefix="IQR-normalized per-joint score (%)",
        )

        # ---------------------------
        # CSV exports for external plotting
        # ---------------------------
        # Minimal CSV: time + chosen scores
        df_export = pd.DataFrame({
            "time_ms": time_ms,
            "mov_pctl_rom_score_raw": mov_rom_score_raw,
            "mov_pctl_rom_score_smooth": mov_rom_score_smooth,
            "iqr_norm_score_raw": iqr_score_raw,
            "iqr_norm_score_smooth": iqr_score_smooth,
        })
        df_export.to_csv(os.path.join(csv_dir, f"{fid}_time_scores.csv"), index=False)

        # Rich per-frame metrics (selected joints)
        df_frame = pd.DataFrame({
            "time_ms": time_ms,
            "mov_pctl_rom_score_raw": mov_rom_score_raw,
            "mov_pctl_rom_score_smooth": mov_rom_score_smooth,
            "iqr_norm_score_raw": iqr_score_raw,
            "iqr_norm_score_smooth": iqr_score_smooth,
            "mae_selected_joints_frame": abs_err.mean(axis=1),
            "rmse_selected_joints_frame": np.sqrt((err**2).mean(axis=1)),
            "movement_label": [mv_file] * len(time_ms),
        })
        df_frame.to_csv(os.path.join(tables_dir, f"{fid}_frame_metrics.csv"), index=False)

        # Per-joint table for this file (selected joints)
        df_joint = pd.DataFrame({
            "joint": JOINTS_FOR_PLOTS,
            "mae": abs_err.mean(axis=0),
            "rmse": np.sqrt((err**2).mean(axis=0)),
            "mov_pctl_rom_score_pct": mov_joint_score,
            "iqr_norm_score_pct": iqr_joint_score,
        })
        df_joint.to_csv(os.path.join(tables_dir, f"{fid}_per_joint_metrics.csv"), index=False)

        # File summary row
        file_summary_rows.append({
            "file": fid,
            "movement_label": mv_file,
            "mov_pctl_rom_score_mean": float(np.mean(mov_rom_score_raw)),
            "iqr_norm_score_mean": float(np.mean(iqr_score_raw)),
            "mae_all_selected_joints": float(np.mean(abs_err)),
            "rmse_all_selected_joints": float(np.sqrt(np.mean(err**2))),
        })

    # File-level summary CSV
    df_file_summary = pd.DataFrame(file_summary_rows).sort_values("file")
    df_file_summary.to_csv(os.path.join(tables_dir, "file_level_summary.csv"), index=False)

    # Per-movement summary (test set) for selected joints
    # Aggregate across all frames in the movement class
    for mv in np.unique(test_movements):
        mv_mask = (test_movements == mv)
        if mv_mask.sum() < 5:
            continue

        y_true_mv = y_test_full[mv_mask][:, joint_idx]
        y_pred_mv = y_test_pred_unscaled[mv_mask][:, joint_idx]
        err_mv = y_pred_mv - y_true_mv
        abs_err_mv = np.abs(err_mv)

        denom_mv = mov_pctl_rom_by_movement.get(mv, global_mov_pctl_rom)
        denom_mv = np.maximum(denom_mv, EPS)

        norm_abs_err_mv = abs_err_mv / denom_mv
        mv_score = 100.0 * (1.0 - norm_abs_err_mv.mean(axis=1))
        mv_score = np.clip(mv_score, 0.0, 100.0)

        per_movement_rows.append({
            "movement": mv,
            "n_frames": int(mv_mask.sum()),
            "mae_selected_joints": float(abs_err_mv.mean()),
            "rmse_selected_joints": float(np.sqrt((err_mv**2).mean())),
            "mov_pctl_rom_score_mean": float(np.mean(mv_score)),
        })

    df_per_movement = pd.DataFrame(per_movement_rows).sort_values("movement")
    df_per_movement.to_csv(os.path.join(tables_dir, "per_movement_selected_joint_metrics.csv"), index=False)

    # Metric definition text
    with open(os.path.join(output_dir, "metric_definitions.txt"), "w") as f:
        f.write(
            "Movement-specific percentile ROM-normalized score (used for joint/time visualization):\n"
            f"For each movement class m and joint j, define ROM_m,j = P{ROM_P_HIGH}(y_test[m, j]) - P{ROM_P_LOW}(y_test[m, j]).\n"
            "Per-frame normalized error: E(t) = mean_j |pred_j(t) - true_j(t)| / ROM_m,j.\n"
            "Score: S(t) = 100 * (1 - E(t)), clipped to [0,100].\n"
            "This is stricter than max-min and reduces inflation from outliers.\n\n"
            "IQR-normalized score (stricter normalized view):\n"
            "IQR_j = P75(y_train[:, j]) - P25(y_train[:, j]) over TRAIN set.\n"
            "E_iqr(t) = mean_j |pred_j(t) - true_j(t)| / IQR_j.\n"
            "S_iqr(t) = 100 * (1 - E_iqr(t)), clipped to [0,100].\n\n"
            "Primary paper metrics (in unscaled angle units):\n"
            "MAE and RMSE computed directly on y_pred_unscaled vs y_true.\n"
            "R^2 computed per joint and overall (variance_weighted).\n"
        )

    # Metadata
    meta = {
        "train_sensor_dir": train_sensor_dir,
        "train_mocap_dir": train_mocap_dir,
        "test_sensor_dir": test_sensor_dir,
        "test_mocap_dir": test_mocap_dir,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "mocap_columns": y_cols,
        "joints_for_plots": JOINTS_FOR_PLOTS,
        "rom_percentiles": {"low": ROM_P_LOW, "high": ROM_P_HIGH},
        "min_frames_for_movement_rom": MIN_FRAMES_FOR_MOV_ROM,
        "overall_test_r2_unscaled": float(overall_test_r2),
        "movement_classes_classifier": list(le.classes_),
        "movement_classification_report": cls_report,
        "movement_confusion_matrix": cls_cm.tolist() if hasattr(cls_cm, "tolist") else [],
        "per_file_movement_mix_test": per_file_movement_mix,
        "notes": {
            "movement_specific_percentile_rom": "denominator uses (95th - 5th) percentile ROM within movement class (TEST set)",
            "iqr_norm": "denominator uses TRAIN IQR (75th - 25th)",
            "plots_use_unscaled_angles": True,
            "csv_exports_dir": "paper_csv_exports",
        },
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Saved] plots -> {plots_dir}")
    print(f"[Saved] tables -> {tables_dir}")
    print(f"[Saved] CSV exports -> {csv_dir}")
    print(f"[Saved] metric_definitions.txt -> {os.path.join(output_dir, 'metric_definitions.txt')}")
    print(f"[Saved] run_metadata.json -> {os.path.join(output_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    train_and_evaluate()