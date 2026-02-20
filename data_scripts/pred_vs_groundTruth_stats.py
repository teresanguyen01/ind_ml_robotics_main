#!/usr/bin/env python3
"""
evaluate_predictions.py

Evaluate predicted mocap/angle-array CSVs against ground-truth CSVs.

Assumptions
- You have two directories:
    --gt_dir: ground-truth CSV files
    --pred_dir: predicted CSV files
- Files correspond by base name:
    ground truth:   <base>.csv
    predicted:      <base>_pred*.csv or <base>_pred_* or <base>*_pred*_smoothed*.csv
  The matcher tries to find the best predicted file for each GT file by stripping
  common prediction suffixes.

Outputs (written to --out_dir)
1) per_joint_metrics.csv
   - MAE, RMSE, R^2, Pearson r, ROM and ROM-normalized errors (per file and aggregated)
2) per_file_summary.csv
   - aggregated scores per file: mean MAE/RMSE, mean Pearson r, mean ROM-score, etc.
3) error_cdf.csv
   - CDF over absolute error (all frames/joints, unscaled units)
4) time_joint_error_heatmaps/
   - heatmap images per file (time x joint) for absolute error and ROM-normalized error
5) time_series_scores/
   - per file CSV with time_ms and per-frame scores (ROM-score, MAE, RMSE)
6) plots/
   - CDF plot, boxplots, correlation bar plots

Notes
- This script only compares columns that exist in BOTH files (intersection).
- It automatically handles row mismatches by trimming to the shortest length.
- "ROM" here defaults to per-file GT ROM (max-min) unless you choose --rom_mode=test_global.
- For academic reporting:
  - Use MAE/RMSE/R^2 as primary numeric accuracy.
  - Use Pearson r to quantify motion-shape agreement.
  - Use ROM-normalized scores to visualize relative error (do not call it unqualified "accuracy").

Run
python evaluate_predictions.py \
  --gt_dir path/to/gt \
  --pred_dir path/to/pred \
  --out_dir path/to/out \
  --dt_ms 10
"""

import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-8


# ---------------------------
# File matching
# ---------------------------

PRED_SUFFIX_PATTERNS = [
    r"_pred_smoothed$",        # base_pred_smoothed
    r"_pred$",                 # base_pred
    r"_pred_.*$",              # base_pred_anything
    r"_smoothed$",             # base_smoothed
    r"_filtered$",             # base_filtered
]

def normalize_stem_for_matching(stem: str) -> str:
    s = stem
    for pat in PRED_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s)
    return s

def build_pred_index(pred_dir: Path) -> Dict[str, List[Path]]:
    """
    Map normalized base key -> list of candidate pred files.
    """
    idx: Dict[str, List[Path]] = {}
    for p in sorted(pred_dir.glob("*.csv")):
        base = normalize_stem_for_matching(p.stem)
        idx.setdefault(base, []).append(p)
    return idx

def choose_best_pred(candidates: List[Path]) -> Path:
    """
    Prefer files containing both 'pred' and 'smoothed' (if present).
    Otherwise take the shortest name (usually closest to base_pred.csv).
    """
    if len(candidates) == 1:
        return candidates[0]

    def score(p: Path) -> Tuple[int, int, int]:
        name = p.stem.lower()
        has_pred = 1 if "pred" in name else 0
        has_smoothed = 1 if "smoothed" in name else 0
        # higher is better; then prefer shorter filename
        return (has_pred + has_smoothed, has_smoothed, -len(name))

    return sorted(candidates, key=score, reverse=True)[0]


# ---------------------------
# Data loading
# ---------------------------

def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    # fill remaining NaNs with column median
    df = df.fillna(df.median(numeric_only=True))
    return df

def align_frames(gt: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(gt), len(pred))
    if len(gt) != len(pred):
        gt = gt.iloc[:n].copy()
        pred = pred.iloc[:n].copy()
    return gt, pred

def intersect_columns(gt: pd.DataFrame, pred: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    exclude_cols = exclude_cols or []
    gt_cols = set(gt.columns) - set(exclude_cols)
    pred_cols = set(pred.columns) - set(exclude_cols)
    cols = sorted(list(gt_cols.intersection(pred_cols)))
    return cols


# ---------------------------
# Metrics
# ---------------------------

def r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < EPS:
        return np.nan
    return 1.0 - ss_res / (ss_tot + EPS)

def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true - np.mean(y_true)
    y_pred = y_pred - np.mean(y_pred)
    denom = (np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + EPS
    return float(np.dot(y_true, y_pred) / denom)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rom_range(y_true: np.ndarray) -> float:
    return float(np.max(y_true) - np.min(y_true))

def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + EPS)


@dataclass
class FileEvalResult:
    base: str
    gt_path: Path
    pred_path: Path
    n_frames: int
    cols: List[str]
    # per joint arrays
    mae_per_joint: np.ndarray
    rmse_per_joint: np.ndarray
    r2_per_joint: np.ndarray
    pearson_per_joint: np.ndarray
    rom_per_joint: np.ndarray
    rom_norm_mae_per_joint: np.ndarray
    # per frame aggregates
    per_frame_mae: np.ndarray
    per_frame_rmse: np.ndarray
    per_frame_rom_score: np.ndarray  # 100*(1 - mean_j abs_err/ROM_j), clipped [0,100]


def evaluate_one_file(
    base: str,
    gt_path: Path,
    pred_path: Path,
    dt_ms: float,
    exclude_cols: Optional[List[str]] = None,
    rom_mode: str = "per_file",   # "per_file" or "test_global"
    global_rom: Optional[np.ndarray] = None,
    joints_order: Optional[List[str]] = None,
) -> FileEvalResult:
    gt_df = read_csv_numeric(gt_path)
    pred_df = read_csv_numeric(pred_path)

    gt_df, pred_df = align_frames(gt_df, pred_df)

    cols = intersect_columns(gt_df, pred_df, exclude_cols=exclude_cols)

    if not cols:
        raise RuntimeError(f"No overlapping numeric columns between\n  GT: {gt_path}\n  Pred: {pred_path}")

    # optionally order columns (useful for consistent heatmaps)
    if joints_order:
        cols = [c for c in joints_order if c in cols] + [c for c in cols if c not in set(joints_order)]

    Yt = gt_df[cols].to_numpy(dtype=float)
    Yp = pred_df[cols].to_numpy(dtype=float)

    err = Yp - Yt
    abs_err = np.abs(err)

    # ROM denominators
    if rom_mode == "test_global":
        if global_rom is None or len(global_rom) != len(cols):
            raise ValueError("rom_mode=test_global requires global_rom with same length as cols")
        rom = global_rom
    else:
        rom = (Yt.max(axis=0) - Yt.min(axis=0)).astype(float)

    rom = np.where(rom == 0, 1e-6, rom)

    # Per-joint scalar metrics
    mae_j = np.mean(abs_err, axis=0)
    rmse_j = np.sqrt(np.mean(err ** 2, axis=0))
    r2_j = np.array([r2_score_1d(Yt[:, i], Yp[:, i]) for i in range(Yt.shape[1])], dtype=float)
    pr_j = np.array([pearson_r(Yt[:, i], Yp[:, i]) for i in range(Yt.shape[1])], dtype=float)
    rom_j = rom
    rom_norm_mae_j = mae_j / (rom_j + EPS)

    # Per-frame aggregated metrics across joints
    per_frame_mae = np.mean(abs_err, axis=1)
    per_frame_rmse = np.sqrt(np.mean(err ** 2, axis=1))

    # ROM-normalized score per frame (visualization)
    norm_abs = abs_err / rom_j[None, :]
    norm_frame = np.mean(norm_abs, axis=1)
    per_frame_rom_score = 100.0 * (1.0 - norm_frame)
    per_frame_rom_score = np.clip(per_frame_rom_score, 0.0, 100.0)

    return FileEvalResult(
        base=base,
        gt_path=gt_path,
        pred_path=pred_path,
        n_frames=Yt.shape[0],
        cols=cols,
        mae_per_joint=mae_j,
        rmse_per_joint=rmse_j,
        r2_per_joint=r2_j,
        pearson_per_joint=pr_j,
        rom_per_joint=rom_j,
        rom_norm_mae_per_joint=rom_norm_mae_j,
        per_frame_mae=per_frame_mae,
        per_frame_rmse=per_frame_rmse,
        per_frame_rom_score=per_frame_rom_score,
    )


# ---------------------------
# Plots
# ---------------------------

def save_heatmap(matrix: np.ndarray, x_label: str, y_labels: List[str], title: str, out_path: Path):
    plt.figure(figsize=(10, max(4, len(y_labels) * 0.35)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=title)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.xlabel(x_label)
    plt.ylabel("Joint")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cdf(values: np.ndarray, out_path: Path, xlabel: str, title: str):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    v = np.sort(v)
    if len(v) == 0:
        return
    y = np.arange(1, len(v) + 1) / len(v)
    plt.figure(figsize=(6, 4))
    plt.plot(v, y)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of frames")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def boxplot_by_joint(df: pd.DataFrame, value_col: str, out_path: Path, title: str):
    joints = df["joint"].unique().tolist()
    data = [df[df["joint"] == j][value_col].values for j in joints]
    plt.figure(figsize=(max(8, len(joints) * 0.5), 5))
    plt.boxplot(data, labels=joints, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="Directory containing ground-truth CSVs")
    ap.add_argument("--pred_dir", required=True, help="Directory containing predicted CSVs")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--dt_ms", type=float, default=10.0, help="Sampling period in ms for time axis")
    ap.add_argument("--rom_mode", choices=["per_file", "test_global"], default="per_file",
                    help="ROM denominator mode for ROM-normalized score")
    ap.add_argument("--exclude_cols", type=str, default="",
                    help="Comma-separated column names to exclude (e.g. time_ms,frame)")
    ap.add_argument("--joints_order_file", type=str, default="",
                    help="Optional text file listing joint names in desired order (one per line)")
    ap.add_argument("--make_heatmaps", action="store_true", help="Generate time×joint heatmaps per file")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]

    joints_order = None
    if args.joints_order_file:
        joints_order = [l.strip() for l in Path(args.joints_order_file).read_text().splitlines() if l.strip()]

    pred_index = build_pred_index(pred_dir)

    gt_files = sorted(gt_dir.glob("*.csv"))
    if not gt_files:
        raise RuntimeError(f"No CSV files found in gt_dir: {gt_dir}")

    # First pass: determine a consistent column list if possible
    # We use the first matched pair intersection as reference
    reference_cols: Optional[List[str]] = None
    matched_pairs: List[Tuple[str, Path, Path]] = []

    for gt_path in gt_files:
        base = gt_path.stem
        if base not in pred_index:
            # try normalization as well (in case GT includes suffixes)
            alt = normalize_stem_for_matching(base)
            candidates = pred_index.get(alt, [])
        else:
            candidates = pred_index[base]

        if not candidates:
            print(f"[WARN] No prediction match for GT file: {gt_path.name}")
            continue

        pred_path = choose_best_pred(candidates)
        matched_pairs.append((base, gt_path, pred_path))

        if reference_cols is None:
            gt_df = read_csv_numeric(gt_path)
            pred_df = read_csv_numeric(pred_path)
            gt_df, pred_df = align_frames(gt_df, pred_df)
            cols = intersect_columns(gt_df, pred_df, exclude_cols=exclude_cols)
            if joints_order:
                cols = [c for c in joints_order if c in cols] + [c for c in cols if c not in set(joints_order)]
            reference_cols = cols

    if not matched_pairs:
        raise RuntimeError("No matched GT/pred pairs found. Check naming and directories.")

    if reference_cols is None or len(reference_cols) == 0:
        raise RuntimeError("Could not determine any overlapping numeric columns from matched pairs.")

    # If rom_mode=test_global, compute global ROM from all GT (over reference cols)
    global_rom = None
    if args.rom_mode == "test_global":
        all_gt_vals = []
        for _, gt_path, _ in matched_pairs:
            gt_df = read_csv_numeric(gt_path)
            if not set(reference_cols).issubset(set(gt_df.columns)):
                continue
            all_gt_vals.append(gt_df[reference_cols].to_numpy(dtype=float))
        if not all_gt_vals:
            raise RuntimeError("rom_mode=test_global requested but no GT values collected.")
        Y_all = np.vstack(all_gt_vals)
        global_rom = (Y_all.max(axis=0) - Y_all.min(axis=0)).astype(float)
        global_rom = np.where(global_rom == 0, 1e-6, global_rom)

    # Output folders
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    time_dir = out_dir / "time_series_scores"
    heat_dir = out_dir / "time_joint_error_heatmaps"

    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    time_dir.mkdir(exist_ok=True)
    if args.make_heatmaps:
        heat_dir.mkdir(exist_ok=True)

    per_joint_rows = []
    per_file_rows = []

    # For global CDF: collect all abs errors (unscaled)
    all_abs_errors = []

    for base, gt_path, pred_path in matched_pairs:
        res = evaluate_one_file(
            base=base,
            gt_path=gt_path,
            pred_path=pred_path,
            dt_ms=args.dt_ms,
            exclude_cols=exclude_cols,
            rom_mode=args.rom_mode,
            global_rom=global_rom,
            joints_order=reference_cols,  # enforce same ordering across files
        )

        # Per-file summary
        per_file_rows.append({
            "file": base,
            "gt_file": gt_path.name,
            "pred_file": pred_path.name,
            "n_frames": res.n_frames,
            "mae_mean": float(np.mean(res.mae_per_joint)),
            "rmse_mean": float(np.mean(res.rmse_per_joint)),
            "pearson_r_mean": float(np.nanmean(res.pearson_per_joint)),
            "r2_mean": float(np.nanmean(res.r2_per_joint)),
            "rom_score_mean": float(np.mean(res.per_frame_rom_score)),
            "rom_score_median": float(np.median(res.per_frame_rom_score)),
        })

        # Per-joint rows (file-level)
        for j, name in enumerate(res.cols):
            per_joint_rows.append({
                "file": base,
                "joint": name,
                "mae": float(res.mae_per_joint[j]),
                "rmse": float(res.rmse_per_joint[j]),
                "r2": float(res.r2_per_joint[j]) if np.isfinite(res.r2_per_joint[j]) else np.nan,
                "pearson_r": float(res.pearson_per_joint[j]) if np.isfinite(res.pearson_per_joint[j]) else np.nan,
                "rom": float(res.rom_per_joint[j]),
                "rom_norm_mae": float(res.rom_norm_mae_per_joint[j]),
            })

        # Time-series CSV per file
        time_ms = np.arange(res.n_frames, dtype=float) * float(args.dt_ms)
        df_time = pd.DataFrame({
            "time_ms": time_ms,
            "per_frame_mae": res.per_frame_mae,
            "per_frame_rmse": res.per_frame_rmse,
            "rom_norm_score": res.per_frame_rom_score,
        })
        df_time.to_csv(time_dir / f"{base}_frame_scores.csv", index=False)

        # Heatmaps (optional)
        if args.make_heatmaps:
            gt_df = read_csv_numeric(gt_path)
            pred_df = read_csv_numeric(pred_path)
            gt_df, pred_df = align_frames(gt_df, pred_df)
            gt_vals = gt_df[res.cols].to_numpy(dtype=float)
            pred_vals = pred_df[res.cols].to_numpy(dtype=float)
            err = pred_vals - gt_vals
            abs_err = np.abs(err)

            # absolute error heatmap (joint x time)
            abs_hm = abs_err.T
            save_heatmap(
                abs_hm,
                x_label="Frame",
                y_labels=res.cols,
                title=f"Absolute error (unscaled) | {base}",
                out_path=heat_dir / f"{base}_abs_error_heatmap.png",
            )

            # ROM-normalized error heatmap
            rom = res.rom_per_joint
            rom_norm = abs_err / (rom[None, :] + EPS)
            rom_hm = rom_norm.T
            save_heatmap(
                rom_hm,
                x_label="Frame",
                y_labels=res.cols,
                title=f"ROM-normalized abs error | {base}",
                out_path=heat_dir / f"{base}_rom_norm_error_heatmap.png",
            )

        # Collect abs errors for global CDF
        all_abs_errors.append(res.mae_per_joint)  # joint average errors
        # Better: per-frame per-joint
        # For CDF we’ll collect from raw arrays via reread to keep memory simple
        gt_df = read_csv_numeric(gt_path)
        pred_df = read_csv_numeric(pred_path)
        gt_df, pred_df = align_frames(gt_df, pred_df)
        cols = res.cols
        abs_err_all = np.abs(pred_df[cols].to_numpy(dtype=float) - gt_df[cols].to_numpy(dtype=float))
        all_abs_errors.append(abs_err_all.ravel())

    # Save tables
    df_joint = pd.DataFrame(per_joint_rows)
    df_file = pd.DataFrame(per_file_rows).sort_values("file")

    df_joint.to_csv(tables_dir / "per_joint_metrics.csv", index=False)
    df_file.to_csv(tables_dir / "per_file_summary.csv", index=False)

    # Aggregate per joint across all files (mean/median)
    df_joint_agg = (
        df_joint.groupby("joint", as_index=False)
        .agg({
            "mae": ["mean", "median"],
            "rmse": ["mean", "median"],
            "r2": ["mean", "median"],
            "pearson_r": ["mean", "median"],
            "rom_norm_mae": ["mean", "median"],
        })
    )
    # flatten columns
    df_joint_agg.columns = ["_".join([c for c in col if c]) for col in df_joint_agg.columns.values]
    df_joint_agg.to_csv(tables_dir / "per_joint_aggregated.csv", index=False)

    # Global CDF
    all_abs = np.concatenate([np.asarray(a).ravel() for a in all_abs_errors if a is not None])
    all_abs = all_abs[np.isfinite(all_abs)]
    all_abs = np.sort(all_abs)
    if len(all_abs) > 0:
        cdf_y = np.arange(1, len(all_abs) + 1) / len(all_abs)
        df_cdf = pd.DataFrame({"abs_error": all_abs, "cdf": cdf_y})
        df_cdf.to_csv(tables_dir / "error_cdf.csv", index=False)

        plot_cdf(
            all_abs,
            out_path=plots_dir / "abs_error_cdf.png",
            xlabel="Absolute error (unscaled angle units)",
            title="CDF of absolute error (all joints, all frames)",
        )

    # Boxplots (joint distributions across files)
    # Pearson r by joint
    boxplot_by_joint(
        df_joint.dropna(subset=["pearson_r"]),
        value_col="pearson_r",
        out_path=plots_dir / "pearson_r_by_joint_boxplot.png",
        title="Pearson correlation by joint (across files)",
    )
    # MAE by joint
    boxplot_by_joint(
        df_joint.dropna(subset=["mae"]),
        value_col="mae",
        out_path=plots_dir / "mae_by_joint_boxplot.png",
        title="MAE by joint (across files)",
    )

    print(f"\n[SAVED] tables -> {tables_dir}")
    print(f"[SAVED] plots  -> {plots_dir}")
    print(f"[SAVED] time-series CSVs -> {time_dir}")
    if args.make_heatmaps:
        print(f"[SAVED] heatmaps -> {heat_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
