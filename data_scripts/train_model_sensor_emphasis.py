#!/usr/bin/env python3
"""
Train XGBoost regressors to predict mocap joint angle arrays from sensor data.

This version focuses on:
- Paper-ready metrics: per-joint MAE, RMSE, R^2 (unscaled)
- Feature contribution analysis:
  A) Built-in XGBoost feature importance (gain/weight/cover)
  B) Optional permutation importance (model-agnostic, more defensible but slower)

Outputs:
- paper_tables/test_per_joint_metrics.csv
- feature_analysis/global_feature_importance_gain.csv
- feature_analysis/per_joint_top_features_gain.csv
- feature_analysis/feature_to_best_joints_gain.csv
- feature_analysis/permutation_importance_global.csv (optional)
- run_metadata.json

Notes:
- Input is sensor, output is mocap/angle array.
- Per-file baseline normalization on sensor data is supported.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import joblib

# ---------------------------
# CONFIG
# ---------------------------
# AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each
output_dir = "AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each/model_sensor_weight"
train_sensor_dir = "AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each/sensor_train"
train_mocap_dir  = "AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each/aa_train"
test_sensor_dir  = "AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each/sensor_test"
test_mocap_dir   = "AA_Yan_mocap_included_0120/model21_Rawan_Teresa_Yan_to_Vero_one_file_each/aa_test"

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

# Feature analysis settings
TOP_K_FEATURES_PER_JOINT = 30
TOP_K_JOINTS_PER_FEATURE = 10

# Permutation importance (optional, slower)
DO_PERMUTATION_IMPORTANCE = True
PERM_N_FRAMES = 5000         # subsample frames from test for speed
PERM_N_REPEATS = 3           # repeats per feature
PERM_MAX_FEATURES = 80       # only compute on top-N features by gain to keep runtime reasonable

EPS = 1e-6

# ---------------------------
# Cleaning helpers
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

# ---------------------------
# Loaders that preserve column names
# ---------------------------

def load_sensor_df(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
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

    # Per-file baseline normalization
    if PER_FILE_BASELINE_NORMALIZE and not df.empty:
        if PER_FILE_BASELINE_METHOD == "median":
            baseline = df.median(numeric_only=True)
        elif PER_FILE_BASELINE_METHOD == "mean":
            baseline = df.mean(numeric_only=True)
        else:
            raise ValueError(f"Unknown PER_FILE_BASELINE_METHOD: {PER_FILE_BASELINE_METHOD}")
        df = df - baseline

    # Ensure stable column order
    df = df.loc[:, list(df.columns)]
    return df

def load_mocap_df(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99, drop_first_n_cols=4) -> pd.DataFrame:
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

# ---------------------------
# Pairing and dataset assembly
# ---------------------------

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

def align_X_y(X_df: pd.DataFrame, y_df: pd.DataFrame, key: str, strict_row_match=True):
    if len(X_df) != len(y_df):
        if strict_row_match:
            raise RuntimeError(f"[{key}] Row mismatch: X={len(X_df)} vs y={len(y_df)} (strict mode).")
        mn = min(len(X_df), len(y_df))
        print(f"[WARN] [{key}] Row mismatch -> trimming both to {mn}")
        X_df = X_df.iloc[:mn].copy()
        y_df = y_df.iloc[:mn].copy()
    return X_df, y_df

def load_dataset(sensor_dir: str, mocap_dir: str, strict_row_match=True):
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_list = []
    y_list = []
    keys = []

    sensor_cols = None
    mocap_cols = None

    print(f"[INFO] {len(pairs)} matched pairs found in\n Sensor: {sensor_dir}\n Mocap : {mocap_dir}")

    for s_path, m_path, key in pairs:
        X_df = load_sensor_df(str(s_path), handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q)
        y_df = load_mocap_df(str(m_path), handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q, drop_first_n_cols=4)

        X_df, y_df = align_X_y(X_df, y_df, key, strict_row_match=strict_row_match)

        # Drop rows with NaNs in y
        y_np = y_df.to_numpy(dtype=float)
        mask = np.all(~np.isnan(y_np), axis=1)
        if mask.sum() != len(y_df):
            dropped = int(len(y_df) - mask.sum())
            print(f"[{key}] Dropped {dropped} rows with NaN in y.")
            X_df = X_df.iloc[mask].copy()
            y_df = y_df.iloc[mask].copy()

        # Column consistency checks
        if sensor_cols is None:
            sensor_cols = list(X_df.columns)
        else:
            if list(X_df.columns) != sensor_cols:
                raise RuntimeError(f"[{key}] Sensor columns differ from earlier files.")

        if mocap_cols is None:
            mocap_cols = list(y_df.columns)
        else:
            if list(y_df.columns) != mocap_cols:
                raise RuntimeError(f"[{key}] Mocap columns differ from earlier files.")

        X_list.append(X_df.to_numpy(dtype=float))
        y_list.append(y_df.to_numpy(dtype=float))
        keys.append(key)

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)

    return X_all, y_all, sensor_cols, mocap_cols, keys

def choose_device_param():
    device = "cuda"
    try:
        _ = xgb.Booster(params={"device": device})
    except Exception:
        device = "cpu"
        print("[INFO] CUDA not available; using CPU.")
    return device

# ---------------------------
# Feature importance utilities
# ---------------------------

def booster_importance_to_df(booster: xgb.Booster, feature_names: list, importance_type: str, joint_name: str):
    """
    Returns a DataFrame with columns: joint, feature, importance_type, score
    """
    score_map = booster.get_score(importance_type=importance_type)  # keys like "f12" (or names)
    rows = []
    for k, v in score_map.items():
        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_names):
                rows.append((joint_name, feature_names[idx], importance_type, float(v)))
        else:
            # If XGBoost returns real feature names as keys
            if k in feature_names:
                rows.append((joint_name, k, importance_type, float(v)))

    return pd.DataFrame(rows, columns=["joint", "feature", "importance_type", "score"])

def normalize_group_sum(df: pd.DataFrame, group_cols: list, score_col: str = "score"):
    """
    Adds normalized_score where each group sums to 1.
    """
    df = df.copy()
    sums = df.groupby(group_cols)[score_col].transform("sum").replace(0, np.nan)
    df["normalized_score"] = df[score_col] / sums
    df["normalized_score"] = df["normalized_score"].fillna(0.0)
    return df

def sensor_group_from_feature(feature_name: str) -> str:
    """
    Map a feature column into a 'sensor id' group.
    Edit this to match your naming convention.

    Examples supported:
      - "S03_ch12" -> "S03"
      - "sensor03_cap_12" -> "sensor03"
      - "CapSensor3_12" -> "CapSensor3"
    """
    s = str(feature_name)

    m = re.match(r"^(S\d+)[\-_].*", s, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(r"^(sensor\d+)[\-_].*", s, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(r"^(CapSensor\d+).*", s, re.IGNORECASE)
    if m:
        return m.group(1)

    # fallback: treat each feature as its own group
    return s

def predict_multioutput_unscaled(models: list, dmat: xgb.DMatrix, scaler_y: RobustScaler) -> np.ndarray:
    preds_scaled = [m.predict(dmat) for m in models]
    Y_scaled = np.column_stack(preds_scaled)
    return scaler_y.inverse_transform(Y_scaled)

# ---------------------------
# Main training routine
# ---------------------------

def train_and_analyze():
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "paper_tables")
    feat_dir = os.path.join(output_dir, "feature_analysis")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    # Load data
    X_train, y_train, sensor_cols, mocap_cols, train_keys = load_dataset(
        train_sensor_dir, train_mocap_dir, strict_row_match=STRICT_ROW_MATCH
    )
    X_test, y_test, sensor_cols_t, mocap_cols_t, test_keys = load_dataset(
        test_sensor_dir, test_mocap_dir, strict_row_match=STRICT_ROW_MATCH
    )

    if sensor_cols_t != sensor_cols:
        raise RuntimeError("Train/test sensor columns differ. Cannot evaluate reliably.")
    if mocap_cols_t != mocap_cols:
        raise RuntimeError("Train/test mocap columns differ. Cannot evaluate reliably.")

    print(f"[INFO] Shapes -> X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")

    # Scale
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))

    # Train per-dimension regressors
    device_param = choose_device_param()
    params = dict(XGB_PARAMS)
    params["device"] = device_param

    train_matrix = xgb.DMatrix(X_train_scaled, feature_names=sensor_cols)
    test_matrix = xgb.DMatrix(X_test_scaled, feature_names=sensor_cols)

    D = y_train.shape[1]
    models = []
    y_test_pred_scaled_list = []

    # Collect feature importance rows
    imp_rows = []

    for i in range(D):
        joint_name = str(mocap_cols[i])

        y_tr = y_train_scaled[:, i]
        y_te = y_test_scaled[:, i]

        train_matrix.set_label(y_tr)
        test_matrix.set_label(y_te)

        print(f"[INFO] Training joint {i+1}/{D}: {joint_name}")

        booster = xgb.train(
            params,
            train_matrix,
            num_boost_round=NUM_ROUNDS,
            evals=[(test_matrix, "eval")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        booster.save_model(os.path.join(output_dir, f"xgboost_model_dim_{i + 1}.json"))
        models.append(booster)

        y_pred_sc = booster.predict(test_matrix)
        y_test_pred_scaled_list.append(y_pred_sc)

        # Collect importance types
        for imp_type in ["gain", "weight", "cover"]:
            df_imp = booster_importance_to_df(booster, sensor_cols, imp_type, joint_name)
            if not df_imp.empty:
                imp_rows.append(df_imp)

    y_test_pred_scaled = np.column_stack(y_test_pred_scaled_list)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---------------------------
    # Primary metrics (unscaled)
    # ---------------------------
    err = y_test_pred_unscaled - y_test
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    r2 = [r2_score(y_test[:, i], y_test_pred_unscaled[:, i]) for i in range(D)]
    overall_r2 = r2_score(y_test, y_test_pred_unscaled, multioutput="variance_weighted")

    df_metrics = pd.DataFrame({
        "joint": mocap_cols,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    })
    df_metrics.to_csv(os.path.join(tables_dir, "test_per_joint_metrics.csv"), index=False)
    print(f"[Saved] test_per_joint_metrics.csv -> {tables_dir}")
    print(f"[RESULT] overall test R2 (unscaled, variance_weighted): {overall_r2:.4f}")

    # ---------------------------
    # Feature importance analysis
    # ---------------------------
    if not imp_rows:
        print("[WARN] No feature importance rows were collected.")
        print("Possible causes:")
        print("  - model did not split (rare),")
        print("  - feature_names mismatch,")
        print("  - training failed earlier.")
    else:
        df_imp_all = pd.concat(imp_rows, ignore_index=True)
        out_raw = os.path.join(feat_dir, "raw_feature_importance_all.csv")
        df_imp_all.to_csv(out_raw, index=False)
        print(f"[Saved] {out_raw}")

        df_gain = df_imp_all[df_imp_all["importance_type"] == "gain"].copy()
        if df_gain.empty:
            print("[WARN] gain importance is empty. Try weight/cover or enable permutation importance.")
        else:
            # Normalize within each joint
            df_gain_norm = normalize_group_sum(df_gain, group_cols=["joint"], score_col="score")

            # A) Global importance across joints: sum normalized gain over joints
            df_global = (
                df_gain_norm.groupby("feature", as_index=False)["normalized_score"]
                .sum()
                .rename(columns={"normalized_score": "global_gain_sum"})
                .sort_values("global_gain_sum", ascending=False)
            )
            out_global = os.path.join(feat_dir, "global_feature_importance_gain.csv")
            df_global.to_csv(out_global, index=False)
            print(f"[Saved] {out_global}")

            # B) Per-joint top features
            rows = []
            for joint in df_gain_norm["joint"].unique():
                dfj = (
                    df_gain_norm[df_gain_norm["joint"] == joint]
                    .sort_values("normalized_score", ascending=False)
                    .head(TOP_K_FEATURES_PER_JOINT)
                    .copy()
                )
                dfj["rank"] = np.arange(1, len(dfj) + 1)
                rows.append(dfj[["joint", "feature", "rank", "normalized_score", "score"]])
            df_per_joint_top = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            out_per_joint = os.path.join(feat_dir, "per_joint_top_features_gain.csv")
            df_per_joint_top.to_csv(out_per_joint, index=False)
            print(f"[Saved] {out_per_joint}")

            # C) For each feature, which joints does it help most
            df_feat_joint = df_gain_norm.groupby(["feature", "joint"], as_index=False)["normalized_score"].sum()
            out_rows = []
            for feat in df_feat_joint["feature"].unique():
                dff = (
                    df_feat_joint[df_feat_joint["feature"] == feat]
                    .sort_values("normalized_score", ascending=False)
                    .head(TOP_K_JOINTS_PER_FEATURE)
                    .copy()
                )
                dff["rank"] = np.arange(1, len(dff) + 1)
                out_rows.append(dff[["feature", "joint", "rank", "normalized_score"]])
            df_feat_to_joints = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
            out_feat_to_joints = os.path.join(feat_dir, "feature_to_best_joints_gain.csv")
            df_feat_to_joints.to_csv(out_feat_to_joints, index=False)
            print(f"[Saved] {out_feat_to_joints}")

            # D) Sensor group importance (aggregate feature -> sensor id)
            df_global2 = df_global.copy()
            df_global2["sensor_group"] = df_global2["feature"].apply(sensor_group_from_feature)
            df_sensor = (
                df_global2.groupby("sensor_group", as_index=False)["global_gain_sum"]
                .sum()
                .sort_values("global_gain_sum", ascending=False)
            )
            out_sensor = os.path.join(feat_dir, "sensor_group_importance_gain.csv")
            df_sensor.to_csv(out_sensor, index=False)
            print(f"[Saved] {out_sensor}")

            # ---------------------------
            # Optional permutation importance
            # ---------------------------
            if DO_PERMUTATION_IMPORTANCE:
                print("[INFO] Computing permutation importance on a subset of test frames...")

                n = X_test_scaled.shape[0]
                take = min(PERM_N_FRAMES, n)
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=take, replace=False)

                Xp = X_test_scaled[idx].copy()
                yt = y_test[idx].copy()

                dmat_base = xgb.DMatrix(Xp, feature_names=sensor_cols)
                Y_base = predict_multioutput_unscaled(models, dmat_base, scaler_y)
                base_mae = float(np.abs(Y_base - yt).mean())

                # Restrict to top features by gain
                top_feats = df_global.head(PERM_MAX_FEATURES)["feature"].tolist()
                feat_to_idx = {f: i for i, f in enumerate(sensor_cols)}

                perm_rows = []
                for feat in top_feats:
                    if feat not in feat_to_idx:
                        continue
                    j = feat_to_idx[feat]
                    col_orig = Xp[:, j].copy()

                    maes = []
                    for _ in range(PERM_N_REPEATS):
                        perm = rng.permutation(col_orig)
                        Xp[:, j] = perm

                        dmat = xgb.DMatrix(Xp, feature_names=sensor_cols)
                        Yp = predict_multioutput_unscaled(models, dmat, scaler_y)

                        mae_perm = float(np.abs(Yp - yt).mean())
                        maes.append(mae_perm)

                    Xp[:, j] = col_orig

                    perm_rows.append({
                        "feature": feat,
                        "mae_increase": float(np.mean(maes) - base_mae),
                        "base_mae": base_mae,
                        "mae_permuted_mean": float(np.mean(maes)),
                    })

                df_perm = pd.DataFrame(perm_rows).sort_values("mae_increase", ascending=False)
                out_perm = os.path.join(feat_dir, "permutation_importance_global.csv")
                df_perm.to_csv(out_perm, index=False)
                print(f"[Saved] {out_perm}")

    # Metadata
    meta = {
        "train_sensor_dir": train_sensor_dir,
        "train_mocap_dir": train_mocap_dir,
        "test_sensor_dir": test_sensor_dir,
        "test_mocap_dir": test_mocap_dir,
        "X_shape_train": list(X_train.shape),
        "y_shape_train": list(y_train.shape),
        "X_shape_test": list(X_test.shape),
        "y_shape_test": list(y_test.shape),
        "sensor_columns": sensor_cols,
        "mocap_columns": mocap_cols,
        "per_file_baseline_normalize": PER_FILE_BASELINE_NORMALIZE,
        "per_file_baseline_method": PER_FILE_BASELINE_METHOD,
        "overall_test_r2_unscaled": float(overall_r2),
        "notes": {
            "feature_importance": "Built-in XGBoost importance (gain/weight/cover) exported; permutation importance exported if enabled.",
            "primary_metrics": "MAE/RMSE/R2 computed in unscaled mocap units.",
            "sensor_grouping": "sensor_group_importance_gain.csv aggregates global gain by a regex-based sensor_group_from_feature() function; edit it to match your column naming.",
        },
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved] run_metadata.json -> {os.path.join(output_dir, 'run_metadata.json')}")
    print("[DONE]")

if __name__ == "__main__":
    train_and_analyze()