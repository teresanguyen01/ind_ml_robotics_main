#!/usr/bin/env python3
"""
Predict mocap joint angle arrays from sensor data using a directory of per-dimension XGBoost models.
Also produces per-file joint-angle plots similar to your reference script.

Expects in model_dir:
- scaler_X.pkl
- scaler_y.pkl
- run_metadata.json  (sensor_columns, mocap_columns_joint_only)
- xgboost_model_dim_1.json ... xgboost_model_dim_D.json

Writes to out_dir:
- <file_key>_pred.csv
- <file_key>_joints.png
- all_predictions_long.csv (optional)
- prediction_run_metadata.json
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------
# Defaults (mirror training)
# ---------------------------
SENSOR_SUFFIX = "_CapacitanceTable"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

HANDLE_OUTLIERS_DEFAULT = True
LOWER_Q_DEFAULT, UPPER_Q_DEFAULT = 0.01, 0.99

PER_FILE_BASELINE_NORMALIZE_DEFAULT = True
PER_FILE_BASELINE_METHOD_DEFAULT = "median"  # "median" or "mean"

_TS_NAME_RE = re.compile(
    r"(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)",
    re.IGNORECASE,
)

# ---------------------------
# Plotting helper (like your script)
# ---------------------------

def plot_joint_angles(df: pd.DataFrame, joints, output_dir: str, pic_name: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = cm.tab20(np.linspace(0, 1, max(1, len(joints))))
    t = np.arange(len(df))

    plotted_any = False
    for name, color in zip(joints, colors):
        if name not in df.columns:
            continue
        plt.plot(t, df[name].values, label=name, color=color)
        plotted_any = True

    plt.xlabel("Frame")
    plt.ylabel("Angle (rad)")
    plt.title("Revolute Joint Angles (Predicted)")
    if plotted_any:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{pic_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] plot -> {out_path}")

# ---------------------------
# Cleaning helpers (same as training)
# ---------------------------

def clean_value(v):
    s = str(v).strip()
    parts = s.split(".")
    if len(parts) > 2:
        s = ".".join(parts[:-1])
    return s

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

def load_sensor_df(
    file_path: str,
    handle_outliers: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    baseline_normalize: bool = True,
    baseline_method: str = "median",
    drop_first_n_cols: int = 0,
) -> pd.DataFrame:
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

    if baseline_normalize and not df.empty:
        if baseline_method == "median":
            baseline = df.median(numeric_only=True)
        elif baseline_method == "mean":
            baseline = df.mean(numeric_only=True)
        else:
            raise ValueError(f"Unknown baseline_method: {baseline_method}")
        df = df - baseline

    df = df.loc[:, list(df.columns)]
    return df

# ---------------------------
# IO utilities
# ---------------------------

def _collect_sensor_files(input_dir: str):
    d = Path(input_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir}")

    files = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix in CSV_EXTS and p.stem.endswith(SENSOR_SUFFIX)
    ]
    if not files:
        raise RuntimeError(f"No sensor files ending with '{SENSOR_SUFFIX}' found in {input_dir}")
    return sorted(files)

def _load_metadata(model_dir: str):
    meta_path = Path(model_dir) / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run_metadata.json in model_dir: {model_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    sensor_cols = meta.get("sensor_columns")
    mocap_cols = meta.get("mocap_columns_joint_only")

    if not sensor_cols or not mocap_cols:
        raise RuntimeError("run_metadata.json missing 'sensor_columns' and/or 'mocap_columns_joint_only'.")

    return meta, list(sensor_cols), list(mocap_cols)

def _load_models(model_dir: str, D: int):
    models = []
    for i in range(D):
        p = Path(model_dir) / f"xgboost_model_dim_{i + 1}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        booster = xgb.Booster()
        booster.load_model(str(p))
        models.append(booster)
    return models

# ---------------------------
# Prediction
# ---------------------------

def predict_directory(
    model_dir: str,
    input_dir: str,
    out_dir: str,
    handle_outliers: bool,
    lower_q: float,
    upper_q: float,
    baseline_normalize: bool,
    baseline_method: str,
    strict_columns: bool,
    drop_first_n_cols: int,
    plot: bool,
    plot_joints: list[str] | None,
    write_aggregate_long: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    meta, sensor_cols_train, mocap_cols = _load_metadata(model_dir)
    D = len(mocap_cols)

    scaler_X_path = Path(model_dir) / "scaler_X.pkl"
    scaler_y_path = Path(model_dir) / "scaler_y.pkl"
    if not scaler_X_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError("Missing scaler_X.pkl and/or scaler_y.pkl in model_dir.")

    scaler_X = joblib.load(str(scaler_X_path))
    scaler_y = joblib.load(str(scaler_y_path))
    models = _load_models(model_dir, D)

    sensor_files = _collect_sensor_files(input_dir)

    agg_rows = []

    # Default plot joints: all predicted joints (like your JOINT_COLUMNS list usage)
    joints_to_plot = plot_joints if (plot_joints and len(plot_joints) > 0) else list(mocap_cols)

    for fpath in sensor_files:
        key = fpath.stem[: -len(SENSOR_SUFFIX)] if fpath.stem.endswith(SENSOR_SUFFIX) else fpath.stem

        X_df = load_sensor_df(
            str(fpath),
            handle_outliers=handle_outliers,
            lower_q=lower_q,
            upper_q=upper_q,
            baseline_normalize=baseline_normalize,
            baseline_method=baseline_method,
            drop_first_n_cols=drop_first_n_cols,
        )

        if X_df.empty:
            print(f"[WARN] Empty after cleaning, skipping: {fpath.name}")
            continue

        if strict_columns:
            if list(X_df.columns) != sensor_cols_train:
                raise RuntimeError(
                    f"[{key}] Sensor columns differ from training.\n"
                    f"Expected: {sensor_cols_train}\nGot: {list(X_df.columns)}"
                )
            X_df_aligned = X_df
        else:
            missing = [c for c in sensor_cols_train if c not in X_df.columns]
            extra = [c for c in X_df.columns if c not in sensor_cols_train]
            if missing:
                print(f"[WARN] [{key}] Missing {len(missing)} cols, filling with 0 (example: {missing[:5]})")
            if extra:
                print(f"[WARN] [{key}] Extra {len(extra)} cols, dropping (example: {extra[:5]})")
            X_df_aligned = X_df.reindex(columns=sensor_cols_train, fill_value=0.0)

        X = X_df_aligned.to_numpy(dtype=float)

        try:
            X_scaled = scaler_X.transform(X)
        except Exception as e:
            print(f"[ERROR] Feature mismatch for {fpath.name}: {e}")
            continue

        dmat = xgb.DMatrix(X_scaled, feature_names=sensor_cols_train)

        preds_scaled = [m.predict(dmat) for m in models]
        Y_scaled = np.column_stack(preds_scaled)
        Y = scaler_y.inverse_transform(Y_scaled)

        df_pred = pd.DataFrame(Y, columns=mocap_cols)
        df_pred.insert(0, "frame", np.arange(len(df_pred), dtype=int))
        df_pred.insert(0, "file_key", key)

        out_csv = Path(out_dir) / f"{key}_pred.csv"
        df_pred.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

        if plot:
            pic_name = f"{key}_joints"
            plot_joint_angles(df_pred, joints_to_plot, out_dir, pic_name)

        if write_aggregate_long:
            long_df = df_pred.melt(
                id_vars=["file_key", "frame"],
                value_vars=mocap_cols,
                var_name="joint",
                value_name="pred",
            )
            agg_rows.append(long_df)

    if write_aggregate_long and agg_rows:
        df_all = pd.concat(agg_rows, ignore_index=True)
        out_all = Path(out_dir) / "all_predictions_long.csv"
        df_all.to_csv(out_all, index=False)
        print(f"[Saved] {out_all}")

    run_info = {
        "model_dir": model_dir,
        "input_dir": input_dir,
        "out_dir": out_dir,
        "n_files": len(sensor_files),
        "sensor_suffix": SENSOR_SUFFIX,
        "handle_outliers": bool(handle_outliers),
        "winsorize_lower_q": float(lower_q),
        "winsorize_upper_q": float(upper_q),
        "baseline_normalize": bool(baseline_normalize),
        "baseline_method": baseline_method,
        "strict_columns": bool(strict_columns),
        "drop_first_n_cols": int(drop_first_n_cols),
        "plot": bool(plot),
        "plot_joints": joints_to_plot,
        "mocap_columns_joint_only": mocap_cols,
        "sensor_columns": sensor_cols_train,
        "training_meta_snapshot": {
            "train_sensor_dir": meta.get("train_sensor_dir"),
            "train_mocap_dir": meta.get("train_mocap_dir"),
            "excluded_mocap_columns": meta.get("excluded_mocap_columns"),
        },
    }
    with open(Path(out_dir) / "prediction_run_metadata.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[Saved] {Path(out_dir) / 'prediction_run_metadata.json'}")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Predict joint-angle mocap arrays from capacitance sensor files using trained XGBoost models."
    )
    p.add_argument("--models_dir", type=str, required=True, help="Directory containing models, scalers, run_metadata.json")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing sensor files to predict on")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for predictions and plots")

    p.add_argument("--handle_outliers", action="store_true", default=HANDLE_OUTLIERS_DEFAULT, help="Enable winsorization")
    p.add_argument("--no_handle_outliers", action="store_false", dest="handle_outliers", help="Disable winsorization")
    p.add_argument("--lower_q", type=float, default=LOWER_Q_DEFAULT)
    p.add_argument("--upper_q", type=float, default=UPPER_Q_DEFAULT)

    p.add_argument("--baseline_norm", action="store_true", default=PER_FILE_BASELINE_NORMALIZE_DEFAULT,
                   help="Enable per-file baseline normalization")
    p.add_argument("--no_baseline_norm", action="store_false", dest="baseline_norm",
                   help="Disable per-file baseline normalization")
    p.add_argument("--baseline_method", type=str, default=PER_FILE_BASELINE_METHOD_DEFAULT, choices=["median", "mean"])

    p.add_argument("--strict_columns", action="store_true", default=False,
                   help="Error if input sensor columns do not exactly match training")
    p.add_argument("--drop_first_n_cols", type=int, default=0,
                   help="Drop first N columns from each input file before processing (if needed)")

    p.add_argument("--plot", action="store_true", default=False, help="Write per-file joint plots (png)")
    p.add_argument("--no_plot", action="store_false", dest="plot", help="Disable plot output")

    # Optional: pass joints explicitly
    p.add_argument("--plot_joints", nargs="*", default=None,
                   help="List of joint column names to plot. Default: all predicted joints.")

    p.add_argument("--write_aggregate_long", action="store_true", default=True,
                   help="Write out_dir/all_predictions_long.csv")
    p.add_argument("--no_write_aggregate_long", action="store_false", dest="write_aggregate_long")

    return p.parse_args()

def main():
    args = parse_args()
    predict_directory(
        model_dir=args.models_dir,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        handle_outliers=args.handle_outliers,
        lower_q=args.lower_q,
        upper_q=args.upper_q,
        baseline_normalize=args.baseline_norm,
        baseline_method=args.baseline_method,
        strict_columns=args.strict_columns,
        drop_first_n_cols=args.drop_first_n_cols,
        plot=args.plot,
        plot_joints=args.plot_joints,
        write_aggregate_long=args.write_aggregate_long,
    )

if __name__ == "__main__":
    main()
