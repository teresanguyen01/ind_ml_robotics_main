#!/usr/bin/env python3
"""
Extended evaluation script for sensor -> mocap (angle array) prediction.

Adds paper-friendly analyses on the TEST set:
1) Pearson correlation r per joint (unscaled)
   - Saves table + optional boxplot grouped by joint type (trunk vs arms)

2) Error heatmaps (time x joint) per TEST file
   - Absolute error heatmap
   - ROM-normalized error heatmap

3) Distribution plots (box/violin)
   - Per-joint absolute error distribution across all TEST frames

4) CDF of absolute error
   - Lets you report: "90% of frames are within X"

5) Per-movement MAE and ROM score
   - Uses movement labels inferred from filenames

Also keeps your existing outputs:
- Per-file ROM and IQR normalized score over time plots
- Polar plots
- Per-file CSV exports (time and scores/errors)
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
# CONFIG
# ---------------------------
output_dir = "AA_Yan_mocap_included_0120/model10_Rawan_to_Vero/model_0203_new"
train_sensor_dir = "AA_Yan_mocap_included_0120/model10_Rawan_to_Vero/sensor_train"
train_mocap_dir = "AA_Yan_mocap_included_0120/model10_Rawan_to_Vero/aa_train"
test_sensor_dir = "AA_Yan_mocap_included_0120/model10_Rawan_to_Vero/sensor_test"
test_mocap_dir = "AA_Yan_mocap_included_0120/model10_Rawan_to_Vero/aa_test"

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

ROM_SCORE_YLIM = (0, 100)
IQR_SCORE_YLIM = (0, 100)

# Heatmap settings
HEATMAP_USE_SELECTED_JOINTS = True  # if False, uses all mocap columns
HEATMAP_COLORMAP = "viridis"

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


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 2:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def joint_group(name: str) -> str:
    n = name.lower()
    if ("waist" in n) or ("hip" in n) or ("spine" in n) or ("torso" in n):
        return "trunk"
    if ("shoulder" in n) or ("elbow" in n) or ("wrist" in n) or ("hand" in n) or ("arm" in n):
        return "arms"
    return "other"


def save_error_heatmap(time_ms, joint_labels, values, title, out_base, cmap=HEATMAP_COLORMAP):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(values.T, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(im, label="Error")
    plt.xlabel("Frame index (time increases left to right)")
    plt.ylabel("Joint")
    plt.title(title)

    yticks = np.arange(len(joint_labels))
    plt.yticks(yticks, joint_labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


def plot_violin_per_joint(joint_labels, per_joint_abs_err_lists, out_base, title):
    plt.figure(figsize=(max(10, 0.6 * len(joint_labels)), 5))
    parts = plt.violinplot(per_joint_abs_err_lists, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks(np.arange(1, len(joint_labels) + 1), joint_labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Absolute error (angle units)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


def plot_box_by_group(group_to_values, out_base, title):
    groups = sorted(group_to_values.keys())
    data = [group_to_values[g] for g in groups]

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=groups, showfliers=True)
    plt.ylabel("Absolute error (angle units)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


def plot_cdf(errors_1d, out_base, title, xlabel):
    e = np.asarray(errors_1d).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0:
        return

    e_sorted = np.sort(e)
    y = np.linspace(0.0, 1.0, e_sorted.size, endpoint=True)

    plt.figure(figsize=(6, 4))
    plt.plot(e_sorted, y, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of frames")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    for q in [0.5, 0.9, 0.95]:
        xq = float(np.quantile(e_sorted, q))
        plt.axvline(xq, alpha=0.2)
        plt.text(xq, q, f"{int(q*100)}%: {xq:.4f}", rotation=90, va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(out_base + ".eps", format="eps", bbox_inches="tight")
    plt.close()


# ---------------------------
# Main
# ---------------------------

def train_and_evaluate():
    os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, "paper_plots")
    tables_dir = os.path.join(output_dir, "paper_tables")
    csv_exports_dir = os.path.join(output_dir, "paper_csv_exports")
    extra_dir = os.path.join(output_dir, "paper_extra_analysis")
    for d in [plots_dir, tables_dir, csv_exports_dir, extra_dir]:
        os.makedirs(d, exist_ok=True)

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

    # Primary metrics (unscaled)
    errors = y_test_pred_unscaled - y_test_full
    abs_errors = np.abs(errors)

    mae_per_dim = np.mean(abs_errors, axis=0)
    rmse_per_dim = np.sqrt(np.mean(errors**2, axis=0))
    r2_per_dim = [r2_score(y_test_full[:, i], y_test_pred_unscaled[:, i]) for i in range(D)]
    overall_test_r2 = r2_score(y_test_full, y_test_pred_unscaled, multioutput="variance_weighted")

    df_global = pd.DataFrame({
        "joint": y_cols,
        "mae": mae_per_dim,
        "rmse": rmse_per_dim,
        "r2": r2_per_dim,
    })
    df_global.to_csv(os.path.join(tables_dir, "test_per_joint_metrics.csv"), index=False)
    print(f"[Saved] test_per_joint_metrics.csv -> {tables_dir}")
    print(f"[RESULT] Overall test R2 (unscaled): {overall_test_r2:.4f}")

    # ---------------------------
    # NEW 1) Pearson correlation per joint (unscaled)
    # ---------------------------
    pearson_per_joint = [pearson_r(y_test_full[:, i], y_test_pred_unscaled[:, i]) for i in range(D)]
    df_corr = pd.DataFrame({
        "joint": y_cols,
        "pearson_r": pearson_per_joint,
        "group": [joint_group(j) for j in y_cols],
    })
    df_corr.to_csv(os.path.join(extra_dir, "test_per_joint_pearson_r.csv"), index=False)
    print(f"[Saved] test_per_joint_pearson_r.csv -> {extra_dir}")

    # Boxplot grouped by joint type using per-frame absolute error pooled across joints in group
    group_to_vals = defaultdict(list)
    for j, name in enumerate(y_cols):
        group_to_vals[joint_group(name)].extend(abs_errors[:, j].tolist())
    plot_box_by_group(
        group_to_vals,
        os.path.join(extra_dir, "abs_error_boxplot_by_group"),
        title="Absolute error distribution by joint group (TEST)",
    )

    # ---------------------------
    # NEW 3) Per-joint error distributions (violin)
    # ---------------------------
    per_joint_lists = [abs_errors[:, j] for j in range(D)]
    plot_violin_per_joint(
        y_cols,
        per_joint_lists,
        os.path.join(extra_dir, "abs_error_violin_per_joint"),
        title="Absolute error distribution per joint (TEST)",
    )

    # ---------------------------
    # NEW 4) CDF of absolute error
    # ---------------------------
    plot_cdf(
        abs_errors.ravel(),
        os.path.join(extra_dir, "cdf_abs_error_all_joints"),
        title="CDF of absolute joint-angle error (TEST, all joints)",
        xlabel="Absolute error (angle units)",
    )

    # ---------------------------
    # Movement classifier (kept)
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
    # Joint alignment
    # ---------------------------
    name_to_idx = {n: i for i, n in enumerate(y_cols)}

    missing = [j for j in JOINTS_FOR_PLOTS if j not in name_to_idx]
    if missing:
        raise RuntimeError(f"JOINTS_FOR_PLOTS missing from mocap columns: {missing}")

    joint_idx_sel = [name_to_idx[j] for j in JOINTS_FOR_PLOTS]

    # Denominators for plot scores
    full_true_sel = y_test_full[:, joint_idx_sel]
    rom_test_sel = full_true_sel.max(axis=0) - full_true_sel.min(axis=0)
    rom_test_sel[rom_test_sel == 0] = 1e-6

    q75 = np.percentile(y_train_full, 75, axis=0)
    q25 = np.percentile(y_train_full, 25, axis=0)
    iqr_train = q75 - q25
    iqr_train[iqr_train == 0] = 1e-6
    iqr_sel = iqr_train[joint_idx_sel]
    iqr_sel[iqr_sel == 0] = 1e-6

    # ---------------------------
    # NEW 5) Per-movement MAE and ROM score
    # ---------------------------
    movement_rows = []
    for mv in sorted(np.unique(test_movements)):
        mask = (test_movements == mv)
        if mask.sum() < 5:
            continue

        mv_abs_err = np.abs(y_test_pred_unscaled[mask] - y_test_full[mask])  # [N, D]
        mv_mae_allj = float(mv_abs_err.mean())

        mv_abs_err_sel = mv_abs_err[:, joint_idx_sel]
        mv_norm_rom = mv_abs_err_sel / rom_test_sel
        mv_score_rom = float(np.clip(100.0 * (1.0 - mv_norm_rom.mean()), 0.0, 100.0))

        movement_rows.append({
            "movement": mv,
            "n_frames": int(mask.sum()),
            "mae_all_joints": mv_mae_allj,
            "rom_norm_score_selected_joints": mv_score_rom,
        })

    df_movement = pd.DataFrame(movement_rows).sort_values("movement")
    df_movement.to_csv(os.path.join(extra_dir, "per_movement_mae_and_rom_score.csv"), index=False)

    # Bar chart for per-movement ROM score (selected joints)
    if not df_movement.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(df_movement["movement"], df_movement["rom_norm_score_selected_joints"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("ROM-normalized score (%)")
        plt.title("Per-movement ROM-normalized score (selected joints, TEST)")
        plt.ylim(0, 100)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        base = os.path.join(extra_dir, "per_movement_rom_score_bar")
        plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
        plt.savefig(base + ".eps", format="eps", bbox_inches="tight")
        plt.close()

    print(f"[Saved] per_movement_mae_and_rom_score.csv -> {extra_dir}")

    # ---------------------------
    # Per-file indexing
    # ---------------------------
    indices_by_file = defaultdict(list)
    for i, fid in enumerate(test_file_ids):
        indices_by_file[fid].append(i)

    per_file_movement_mix = {}
    file_summary_rows = []

    print("\n[INFO] Per-test-file plots and exports:")

    for fid in sorted(indices_by_file.keys()):
        idxs = np.asarray(indices_by_file[fid], dtype=int)

        # movement probability mix
        X_f = X_test_scaled[idxs]
        probs = clf.predict_proba(X_f)
        mean_probs = probs.mean(axis=0)
        mix = {}
        for cls_name, p in sorted(zip(le.classes_, mean_probs), key=lambda x: -x[1]):
            mix[str(cls_name)] = float(p * 100.0)
        per_file_movement_mix[fid] = mix

        # Select GT and pred (unscaled)
        y_true_all = y_test_full[idxs]
        y_pred_all = y_test_pred_unscaled[idxs]

        y_true_sel = y_true_all[:, joint_idx_sel]
        y_pred_sel = y_pred_all[:, joint_idx_sel]

        err_sel = y_pred_sel - y_true_sel
        abs_err_sel = np.abs(err_sel)

        time_ms = np.arange(len(idxs)) * DT_MS

        # ROM score over time (selected joints)
        norm_abs_err_rom = abs_err_sel / rom_test_sel
        norm_err_frame_rom = norm_abs_err_rom.mean(axis=1)
        rom_score_raw = np.clip(100.0 * (1.0 - norm_err_frame_rom), 0.0, 100.0)
        rom_score_smooth = rolling_mean(rom_score_raw, SMOOTH_WINDOW)

        rom_out_base = os.path.join(plots_dir, f"rom_norm_score_{fid}")
        plot_score_over_time(
            time_ms,
            rom_score_raw,
            rom_score_smooth,
            fid,
            rom_out_base,
            ylabel="Range of Motion (ROM)-normalized score (%)",
            ylim=ROM_SCORE_YLIM,
        )

        rom_joint_score = np.clip(100.0 * (1.0 - norm_abs_err_rom.mean(axis=0)), 0.0, 100.0)
        rom_polar_base = os.path.join(plots_dir, f"rom_norm_polar_{fid}")
        plot_polar_scores(
            JOINTS_FOR_PLOTS,
            rom_joint_score,
            fid,
            rom_polar_base,
            title_prefix="ROM-normalized per-joint score (%)",
        )

        # IQR score over time (selected joints)
        norm_abs_err_iqr = abs_err_sel / iqr_sel
        norm_err_frame_iqr = norm_abs_err_iqr.mean(axis=1)
        iqr_score_raw = np.clip(100.0 * (1.0 - norm_err_frame_iqr), 0.0, 100.0)
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

        iqr_joint_score = np.clip(100.0 * (1.0 - norm_abs_err_iqr.mean(axis=0)), 0.0, 100.0)
        iqr_polar_base = os.path.join(plots_dir, f"iqr_norm_polar_{fid}")
        plot_polar_scores(
            JOINTS_FOR_PLOTS,
            iqr_joint_score,
            fid,
            iqr_polar_base,
            title_prefix="IQR-normalized per-joint score (%)",
        )

        # Save per-frame table (already includes what you asked for)
        df_frame = pd.DataFrame({
            "time_ms": time_ms,
            "rom_norm_score_raw": rom_score_raw,
            "rom_norm_score_smooth": rom_score_smooth,
            "iqr_norm_score_raw": iqr_score_raw,
            "iqr_norm_score_smooth": iqr_score_smooth,
            "mae_selected_joints_frame": abs_err_sel.mean(axis=1),
            "rmse_selected_joints_frame": np.sqrt((err_sel**2).mean(axis=1)),
        })
        df_frame.to_csv(os.path.join(tables_dir, f"{fid}_frame_metrics.csv"), index=False)

        # Also save simple 2-column CSVs for external plotting
        pd.DataFrame({"time_ms": time_ms, "rom_norm_score_pct": rom_score_raw}).to_csv(
            os.path.join(csv_exports_dir, f"{fid}_rom_score.csv"), index=False
        )
        pd.DataFrame({"time_ms": time_ms, "iqr_norm_score_pct": iqr_score_raw}).to_csv(
            os.path.join(csv_exports_dir, f"{fid}_iqr_score.csv"), index=False
        )
        pd.DataFrame({"time_ms": time_ms, "mae": abs_err_sel.mean(axis=1)}).to_csv(
            os.path.join(csv_exports_dir, f"{fid}_mae.csv"), index=False
        )
        pd.DataFrame({"time_ms": time_ms, "rmse": np.sqrt((err_sel**2).mean(axis=1))}).to_csv(
            os.path.join(csv_exports_dir, f"{fid}_rmse.csv"), index=False
        )

        # Save per-joint table for this file
        df_joint = pd.DataFrame({
            "joint": JOINTS_FOR_PLOTS,
            "mae": abs_err_sel.mean(axis=0),
            "rmse": np.sqrt((err_sel**2).mean(axis=0)),
            "rom_norm_score_pct": rom_joint_score,
            "iqr_norm_score_pct": iqr_joint_score,
        })
        df_joint.to_csv(os.path.join(tables_dir, f"{fid}_per_joint_metrics.csv"), index=False)

        # ---------------------------
        # NEW 2) Error heatmaps (time x joint)
        # ---------------------------
        if HEATMAP_USE_SELECTED_JOINTS:
            hm_joint_labels = JOINTS_FOR_PLOTS
            abs_err_hm = abs_err_sel  # [T, J]
            rom_err_hm = norm_abs_err_rom  # [T, J]
        else:
            hm_joint_labels = y_cols
            abs_err_hm = np.abs(y_pred_all - y_true_all)  # [T, D]
            # ROM for all joints computed on full test set
            rom_all = (y_test_full.max(axis=0) - y_test_full.min(axis=0))
            rom_all[rom_all == 0] = 1e-6
            rom_err_hm = abs_err_hm / rom_all

        save_error_heatmap(
            time_ms,
            hm_joint_labels,
            abs_err_hm,
            title=f"Absolute error heatmap (time x joint)\nfile: {fid}",
            out_base=os.path.join(extra_dir, f"heatmap_abs_error_{fid}"),
        )
        save_error_heatmap(
            time_ms,
            hm_joint_labels,
            rom_err_hm,
            title=f"ROM-normalized absolute error heatmap (time x joint)\nfile: {fid}",
            out_base=os.path.join(extra_dir, f"heatmap_rom_norm_error_{fid}"),
        )

        file_summary_rows.append({
            "file": fid,
            "rom_norm_score_mean": float(np.mean(rom_score_raw)),
            "iqr_norm_score_mean": float(np.mean(iqr_score_raw)),
            "mae_selected_joints": float(np.mean(abs_err_sel)),
            "rmse_selected_joints": float(np.sqrt(np.mean(err_sel**2))),
        })

    df_file_summary = pd.DataFrame(file_summary_rows).sort_values("file")
    df_file_summary.to_csv(os.path.join(tables_dir, "file_level_summary.csv"), index=False)

    # Save metric definition text
    with open(os.path.join(output_dir, "metric_definitions.txt"), "w") as f:
        f.write(
            "ROM-normalized score (selected joints):\n"
            "ROM_j = max(y_test_full[:, j]) - min(y_test_full[:, j]) over the full TEST set.\n"
            "E_rom(t) = mean_j |pred_j(t) - true_j(t)| / ROM_j.\n"
            "S_rom(t) = 100 * (1 - E_rom(t)), clipped to [0,100].\n\n"
            "IQR-normalized score (selected joints):\n"
            "IQR_j = percentile75(y_train_full[:, j]) - percentile25(y_train_full[:, j]) over TRAIN set.\n"
            "E_iqr(t) = mean_j |pred_j(t) - true_j(t)| / IQR_j.\n"
            "S_iqr(t) = 100 * (1 - E_iqr(t)), clipped to [0,100].\n\n"
            "Pearson correlation (per joint): corr(y_true, y_pred) computed on unscaled angles.\n"
        )

    # Save metadata
    meta = {
        "train_sensor_dir": train_sensor_dir,
        "train_mocap_dir": train_mocap_dir,
        "test_sensor_dir": test_sensor_dir,
        "test_mocap_dir": test_mocap_dir,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "mocap_columns": y_cols,
        "joints_for_plots": JOINTS_FOR_PLOTS,
        "overall_test_r2_unscaled": float(overall_test_r2),
        "movement_classes": list(le.classes_),
        "movement_classification_report": cls_report,
        "movement_confusion_matrix": cls_cm.tolist(),
        "per_file_movement_mix_test": per_file_movement_mix,
        "notes": {
            "rom_norm": "uses full test-set ROM (max-min), intended for visualization and movement-level trend reporting",
            "iqr_norm": "uses training-set IQR, stricter normalized view",
            "pearson_r": "correlation of predicted and GT time series, unscaled",
        },
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved outputs:")
    print(f"  plots: {plots_dir}")
    print(f"  tables: {tables_dir}")
    print(f"  csv exports: {csv_exports_dir}")
    print(f"  extra analysis: {extra_dir}")
    print(f"  metadata: {os.path.join(output_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    train_and_evaluate()
