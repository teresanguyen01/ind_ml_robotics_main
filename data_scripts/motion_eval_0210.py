#!/usr/bin/env python3
"""
motion_eval.py

Evaluate how closely a predicted motion CSV matches a ground-truth motion CSV.

Inputs
- Predicted CSV columns (validated by name):
  file_key (optional), frame, <JOINT_COLUMNS>
- Ground truth CSV columns (validated by name):
  time_ms, x, y, z, qw, qx, qy, qz, <JOINT_COLUMNS>

Only the joint angle columns in JOINT_COLUMNS are used for evaluation. All other columns are ignored.

Alignment
Default alignment: "normalized progress"
- Let predicted have Np frames and ground truth have Ng samples.
- Map predicted frame i to ground truth index j by normalized progress:
  progress_i = i / (Np - 1)
  j = round(progress_i * (Ng - 1))

Optional alignment: "nearest_time" (requires --pred_fps)
- Estimate predicted time_ms from frame index and pred_fps.
- Map to nearest ground truth time_ms.

Units
- This version assumes ALL joint angles are in radians (per your CSVs).
- Default thresholds are 5 deg and 10 deg converted to radians.

Angle circularity
- If joints are circular, wrapping is used to compute minimal angular error.
- If joints are NOT circular (bounded), enable --non_circular:
  - Threshold accuracies use raw error (no wrapping)
  - "circ_*" metrics are set equal to raw metrics for convenience

Outputs (saved under --output_dir/<gt_stem>/[<file_key>/])
- metrics_per_joint.csv
- metrics_summary.json
- temporal_error_stats.csv
- overall_accuracy_over_time.csv   (NEW: time series with multiple thresholds + extra stats)
- overall_accuracy_over_time_primary_only.csv (keeps old single-threshold series)
- max_joint_error_over_time.csv
- top_offending_joints.csv
- plots/...

Multi-file behavior
- If predicted CSV contains multiple file_key values, evaluate each file_key group separately,
  and also write combined tables at: --output_dir/<gt_stem>/
  - per_file_metrics_per_joint.csv
  - per_file_metrics_summary.json

Dependencies
numpy, pandas, matplotlib, seaborn, scipy
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from scipy.signal import correlate


# -----------------------------
# Configuration
# -----------------------------

# Match your predicted CSV columns (your example uses these 11 joints).
JOINT_COLUMNS: List[str] = [
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

# Default thresholds are 5 and 10 degrees converted to radians.
DEFAULT_THRESHOLDS = [math.radians(5.0), math.radians(10.0)]

PLOT_DPI = 300

# Plot RAW error over time by default.
PLOT_CIRCULAR_ERROR = False

# All your CSV angles are radians.
ANGLE_UNIT = "rad"


# -----------------------------
# Utils
# -----------------------------

def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")


def wrap_angle_diff(diff: np.ndarray, unit: str = ANGLE_UNIT) -> np.ndarray:
    """
    Minimal signed angular difference.
    Radians: wrap to [-pi, pi)
    Degrees: wrap to [-180, 180)
    """
    if unit == "deg":
        period = 360.0
        half = 180.0
    else:
        period = 2.0 * math.pi
        half = math.pi
    return (diff + half) % period - half


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def mae(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x)))


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return m, s


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """
    Returns (max_ncc, lag_at_max) in samples.
    Uses z-scored signals.
    """
    az = zscore(a)
    bz = zscore(b)
    if len(az) == 0 or len(bz) == 0:
        return float("nan"), 0

    c = correlate(az, bz, mode="full", method="auto")
    c = c / max(len(az), 1)
    lags = np.arange(-len(bz) + 1, len(az))
    idx = int(np.argmax(c))
    return float(c[idx]), int(lags[idx])


def dtw_distance(a: np.ndarray, b: np.ndarray, window: Optional[int] = None) -> float:
    """
    Classic DTW distance with squared error local cost, returning sqrt(total_cost).
    Optional Sakoe-Chiba band with half-width=window.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")

    if window is None:
        window = max(n, m)
        window = max(window, abs(n - m))

    inf = float("inf")
    D = np.full((n + 1, m + 1), inf, dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(math.sqrt(D[n, m])) if np.isfinite(D[n, m]) else float("nan")


def derivative(x: np.ndarray, dt: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([], dtype=float)
    return np.diff(x) / dt


def direction_agreement(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if len(v1) == 0 or len(v2) == 0:
        return float("nan")
    return float(np.mean(np.sign(v1) == np.sign(v2)))


# -----------------------------
# Data loading
# -----------------------------

def load_predicted_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected = ["frame"] + JOINT_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Predicted CSV is missing columns: {missing}")

    keep_cols = (["file_key"] if "file_key" in df.columns else []) + expected
    return df[keep_cols].copy()


def load_ground_truth_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected_prefix = ["time_ms", "x", "y", "z", "qw", "qx", "qy", "qz"]
    expected = expected_prefix + JOINT_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Ground truth CSV is missing columns: {missing}")

    return df[expected].copy()


# -----------------------------
# Alignment
# -----------------------------

@dataclass
class AlignedSequence:
    pred_frames: np.ndarray
    pred_time_ms: Optional[np.ndarray]
    gt_time_ms: np.ndarray
    pred_angles: pd.DataFrame
    gt_angles: pd.DataFrame


def align_normalized_progress(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> AlignedSequence:
    pred_df = pred_df.sort_values("frame").reset_index(drop=True)
    gt_df = gt_df.sort_values("time_ms").reset_index(drop=True)

    Np = len(pred_df)
    Ng = len(gt_df)
    if Np < 2 or Ng < 2:
        raise ValueError("Predicted or ground truth sequence too short to align.")

    pred_idx = np.arange(Np, dtype=float)
    progress = pred_idx / (Np - 1)
    gt_idx = np.rint(progress * (Ng - 1)).astype(int)
    gt_idx = np.clip(gt_idx, 0, Ng - 1)

    pred_angles = pred_df[JOINT_COLUMNS].copy()
    gt_angles = gt_df.loc[gt_idx, JOINT_COLUMNS].reset_index(drop=True)

    return AlignedSequence(
        pred_frames=pred_df["frame"].to_numpy(dtype=float),
        pred_time_ms=None,
        gt_time_ms=gt_df.loc[gt_idx, "time_ms"].to_numpy(dtype=float),
        pred_angles=pred_angles.reset_index(drop=True),
        gt_angles=gt_angles,
    )


def align_nearest_time(pred_df: pd.DataFrame, gt_df: pd.DataFrame, pred_fps: float) -> AlignedSequence:
    if pred_fps <= 0:
        raise ValueError("--pred_fps must be positive for nearest_time alignment.")

    pred_df = pred_df.sort_values("frame").reset_index(drop=True)
    gt_df = gt_df.sort_values("time_ms").reset_index(drop=True)

    frames = pred_df["frame"].to_numpy(dtype=float)
    f0 = float(np.min(frames))
    pred_time_ms = (frames - f0) * 1000.0 / float(pred_fps)

    gt_time_ms_full = gt_df["time_ms"].to_numpy(dtype=float)

    gt_idx = np.searchsorted(gt_time_ms_full, pred_time_ms, side="left")
    gt_idx = np.clip(gt_idx, 0, len(gt_time_ms_full) - 1)

    left = np.clip(gt_idx - 1, 0, len(gt_time_ms_full) - 1)
    right = gt_idx
    choose_left = np.abs(gt_time_ms_full[left] - pred_time_ms) <= np.abs(gt_time_ms_full[right] - pred_time_ms)
    gt_idx = np.where(choose_left, left, right).astype(int)

    pred_angles = pred_df[JOINT_COLUMNS].copy()
    gt_angles = gt_df.loc[gt_idx, JOINT_COLUMNS].reset_index(drop=True)

    return AlignedSequence(
        pred_frames=frames,
        pred_time_ms=pred_time_ms,
        gt_time_ms=gt_df.loc[gt_idx, "time_ms"].to_numpy(dtype=float),
        pred_angles=pred_angles.reset_index(drop=True),
        gt_angles=gt_angles,
    )


# -----------------------------
# Metrics
# -----------------------------

def compute_joint_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    unit: str,
    dtw_window: int,
    thresholds: Optional[List[float]] = None,
    primary_threshold: Optional[float] = None,
    non_circular: bool = False,
) -> Dict[str, float]:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)

    raw_err = pred - gt
    circ_err = wrap_angle_diff(raw_err, unit=unit)

    raw_mae = mae(raw_err)
    raw_rmse = rmse(raw_err)
    raw_mean, raw_std = mean_std(raw_err)

    if non_circular:
        circ_mae = raw_mae
        circ_rmse = raw_rmse
        circ_mean, circ_std = raw_mean, raw_std
    else:
        circ_mae = mae(circ_err)
        circ_rmse = rmse(circ_err)
        circ_mean, circ_std = mean_std(circ_err)

    corr = pearson_corr(pred, gt)
    pattern_acc_corr = 100.0 * max(0.0, corr) if np.isfinite(corr) else float("nan")

    err_for_thresholds = raw_err if non_circular else circ_err
    thr_metrics = {
        f"pct_within_{t:g}": float(100.0 * np.mean(np.abs(err_for_thresholds) <= t))
        for t in thresholds
    }
    if primary_threshold is not None:
        thr_metrics["primary_pct_accuracy"] = float(100.0 * np.mean(np.abs(err_for_thresholds) <= primary_threshold))

    pz = zscore(pred)
    gz = zscore(gt)
    cos_z = cosine_similarity(pz, gz)
    max_ncc, ncc_lag = normalized_cross_correlation(pred, gt)
    dtw_z = dtw_distance(pz, gz, window=dtw_window)

    v_pred = derivative(pred, dt=1.0)
    v_gt = derivative(gt, dt=1.0)
    a_pred = derivative(v_pred, dt=1.0)
    a_gt = derivative(v_gt, dt=1.0)

    v_err = v_pred - v_gt
    a_err = a_pred - a_gt

    out: Dict[str, float] = {
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_mean_err": raw_mean,
        "raw_std_err": raw_std,
        "pearson_r": corr,
        "circ_mae": circ_mae,
        "circ_rmse": circ_rmse,
        "circ_mean_err": circ_mean,
        "circ_std_err": circ_std,
        "pattern_acc_corr_pct": float(pattern_acc_corr),
        "cosine_sim_z": cos_z,
        "max_ncc": max_ncc,
        "ncc_lag": float(ncc_lag),
        "dtw_dist_z": float(dtw_z),
        "vel_mae": mae(v_err) if len(v_err) else float("nan"),
        "vel_rmse": rmse(v_err) if len(v_err) else float("nan"),
        "vel_pearson_r": pearson_corr(v_pred, v_gt) if len(v_pred) else float("nan"),
        "acc_mae": mae(a_err) if len(a_err) else float("nan"),
        "acc_rmse": rmse(a_err) if len(a_err) else float("nan"),
        "acc_pearson_r": pearson_corr(a_pred, a_gt) if len(a_pred) else float("nan"),
        "vel_direction_agreement": direction_agreement(v_pred, v_gt) if len(v_pred) else float("nan"),
    }
    out.update(thr_metrics)
    return out


def aggregate_metrics(per_joint_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    numeric_cols = [c for c in per_joint_df.columns if pd.api.types.is_numeric_dtype(per_joint_df[c])]
    for c in numeric_cols:
        vals = per_joint_df[c].to_numpy(dtype=float)
        out[f"mean_{c}"] = float(np.nanmean(vals))
        out[f"median_{c}"] = float(np.nanmedian(vals))
    return out


def temporal_error_stats(err_df: pd.DataFrame) -> pd.DataFrame:
    abs_err = err_df.abs()
    return pd.DataFrame({
        "mean_abs_err_across_joints": abs_err.mean(axis=1),
        "median_abs_err_across_joints": abs_err.median(axis=1),
        "max_abs_err_across_joints": abs_err.max(axis=1),
    })


def compute_error_matrix(
    pred_angles: pd.DataFrame,
    gt_angles: pd.DataFrame,
    non_circular: bool,
    unit: str = ANGLE_UNIT,
) -> pd.DataFrame:
    err = pd.DataFrame(index=pred_angles.index)
    for j in JOINT_COLUMNS:
        raw = pred_angles[j].to_numpy(dtype=float) - gt_angles[j].to_numpy(dtype=float)
        err[j] = raw if non_circular else wrap_angle_diff(raw, unit=unit)
    return err

def per_joint_accuracy_over_time(
    err_df: pd.DataFrame,
    aligned_time: np.ndarray,
    thresholds: Optional[List[float]] = None,
    include_threshold_flags: bool = True,
    include_signed_error: bool = False,
) -> pd.DataFrame:
    """
    Per-frame per-joint table:
      - abs_err_<joint> (rad)
      - acc_<joint> in [0,100] computed as 100*(1 - clip(|err|/pi, 0, 1))
      - optionally within_<thr>_<joint> (0/1) for each threshold
      - optionally signed_err_<joint>

    err_df should already be the error you want to evaluate:
      - wrapped error if non_circular is False
      - raw error if non_circular is True
    """
    out = pd.DataFrame({"aligned_time": aligned_time})
    denom = float(math.pi)

    thr_list = thresholds or []
    for j in err_df.columns:
        e = err_df[j].to_numpy(dtype=float)
        ae = np.abs(e)

        out[f"abs_err_{j}"] = ae
        out[f"acc_{j}"] = 100.0 * (1.0 - np.clip(ae / denom, 0.0, 1.0))

        if include_signed_error:
            out[f"signed_err_{j}"] = e

        if include_threshold_flags:
            for thr in thr_list:
                out[f"within_{thr:g}_{j}"] = (ae <= float(thr)).astype(int)

    return out



def max_joint_error_over_time(err_df: pd.DataFrame) -> pd.DataFrame:
    abs_err = err_df.abs()
    worst_joint = abs_err.idxmax(axis=1)
    max_abs_error = abs_err.max(axis=1)
    return pd.DataFrame({
        "max_abs_error": max_abs_error,
        "worst_joint": worst_joint,
    })


def top_offending_joints_table(err_df: pd.DataFrame) -> pd.DataFrame:
    abs_err = err_df.abs()
    max_info = max_joint_error_over_time(err_df)
    worst = max_info["worst_joint"]

    rows = []
    T = len(err_df)
    for j in JOINT_COLUMNS:
        overall = abs_err[j].to_numpy(dtype=float)
        is_worst = (worst == j).to_numpy()

        times_as_worst = int(np.sum(is_worst))
        pct_time_as_worst = 100.0 * times_as_worst / max(T, 1)

        when_worst_vals = overall[is_worst] if np.any(is_worst) else np.array([], dtype=float)

        rows.append({
            "joint": j,
            "times_as_worst": times_as_worst,
            "pct_time_as_worst": pct_time_as_worst,
            "mean_abs_err_overall": float(np.mean(overall)) if T else float("nan"),
            "p95_abs_err_overall": float(np.percentile(overall, 95)) if T else float("nan"),
            "max_abs_err_overall": float(np.max(overall)) if T else float("nan"),
            "mean_abs_err_when_worst": float(np.mean(when_worst_vals)) if len(when_worst_vals) else float("nan"),
            "max_abs_err_when_worst": float(np.max(when_worst_vals)) if len(when_worst_vals) else float("nan"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["times_as_worst", "mean_abs_err_overall"], ascending=[False, False]).reset_index(drop=True)
    return df


def overall_accuracy_over_time(err_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    abs_err = err_df.abs().to_numpy(dtype=float)  # (T, J)
    acc_pct = 100.0 * np.mean(abs_err <= float(threshold), axis=1)
    return pd.DataFrame({"overall_accuracy_pct": acc_pct})


def overall_threshold_accuracy_over_time(err_df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    abs_err = err_df.abs().to_numpy(dtype=float)  # (T, J)
    out = {}
    for thr in thresholds:
        out[f"pct_joints_within_{thr:g}"] = 100.0 * np.mean(abs_err <= float(thr), axis=1)
    return pd.DataFrame(out)


def overall_pattern_metrics_over_time(
    pred_angles: pd.DataFrame,
    gt_angles: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Sliding-window overall pattern metrics across all joints.
    """
    P = pred_angles.to_numpy(dtype=float)
    G = gt_angles.to_numpy(dtype=float)
    T = min(len(P), len(G))
    if T == 0:
        return pd.DataFrame()

    w = int(max(3, window))
    half = w // 2

    pearsons = np.full(T, np.nan, dtype=float)
    patt = np.full(T, np.nan, dtype=float)
    cosz = np.full(T, np.nan, dtype=float)
    maxncc = np.full(T, np.nan, dtype=float)
    lag = np.full(T, np.nan, dtype=float)

    p_mean = np.mean(P, axis=1)
    g_mean = np.mean(G, axis=1)

    for t in range(T):
        a = max(0, t - half)
        b = min(T, t + half + 1)

        pw = P[a:b].reshape(-1)
        gw = G[a:b].reshape(-1)

        r = pearson_corr(pw, gw)
        pearsons[t] = r
        patt[t] = 100.0 * max(0.0, r) if np.isfinite(r) else np.nan

        cosz[t] = cosine_similarity(zscore(pw), zscore(gw))

        mncc, mlag = normalized_cross_correlation(p_mean[a:b], g_mean[a:b])
        maxncc[t] = mncc
        lag[t] = float(mlag)

    return pd.DataFrame({
        "pearson_r_overall": pearsons,
        "pattern_acc_corr_pct_overall": patt,
        "cosine_sim_z_overall": cosz,
        "max_ncc_overall": maxncc,
        "ncc_lag_overall": lag,
    })


# -----------------------------
# Plotting
# -----------------------------

def save_max_joint_error_plot(out_dir: Path, x: np.ndarray, max_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.plot(x, max_df["max_abs_error"], linewidth=1.4)
    plt.title("Max joint absolute error over time (worst joint each time)")
    plt.xlabel("Aligned time (ms)" if np.all(np.isfinite(x)) else "Aligned index")
    plt.ylabel("Max |error| across joints")
    plt.tight_layout()
    plt.savefig(out_dir / "max_joint_abs_error_over_time.png", dpi=PLOT_DPI)
    plt.close()


def save_overall_accuracy_plot(out_dir: Path, x: np.ndarray, acc_df: pd.DataFrame, threshold: float) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.plot(x, acc_df["overall_accuracy_pct"], linewidth=1.4)
    plt.ylim(0, 100)
    plt.title(f"Overall accuracy over time (percent joints within {threshold:g} rad)")
    plt.xlabel("Aligned time (ms)" if np.all(np.isfinite(x)) else "Aligned index")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_accuracy_over_time.png", dpi=PLOT_DPI)
    plt.close()


def save_time_series_overlay(out_dir: Path, joint: str, x: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.plot(x, gt, label="Ground truth", linewidth=1.6)
    plt.plot(x, pred, label="Predicted", linewidth=1.4, alpha=0.9)
    plt.title(f"{joint} | Angle over time")
    plt.xlabel("Aligned time (ms)" if np.all(np.isfinite(x)) else "Aligned index")
    plt.ylabel("Angle (rad)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{joint}__timeseries.png", dpi=PLOT_DPI)
    plt.close()


def save_error_over_time(out_dir: Path, joint: str, x: np.ndarray, err: np.ndarray, label: str) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.plot(x, err, linewidth=1.2)
    plt.axhline(0.0, linewidth=1.0)
    plt.title(f"{joint} | {label} over time")
    plt.xlabel("Aligned time (ms)" if np.all(np.isfinite(x)) else "Aligned index")
    plt.ylabel(f"{label} (rad)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{joint}__error_time.png", dpi=PLOT_DPI)
    plt.close()


def save_error_distribution(out_dir: Path, joint: str, err: np.ndarray, label: str) -> None:
    plt.figure(figsize=(6.5, 3.6))
    sns.histplot(err, bins=60, kde=True)
    plt.title(f"{joint} | {label} distribution")
    plt.xlabel(f"{label} (rad)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{joint}__error_dist.png", dpi=PLOT_DPI)
    plt.close()


def save_summary_bars(out_dir: Path, metrics_df: pd.DataFrame, metric: str, title: str) -> None:
    if metric not in metrics_df.columns:
        return
    df = metrics_df.sort_values(metric, ascending=True).copy()
    plt.figure(figsize=(9.5, 5.8))
    sns.barplot(data=df, y="joint", x=metric, orient="h")
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("Joint")
    plt.tight_layout()
    plt.savefig(out_dir / f"summary__{metric}.png", dpi=PLOT_DPI)
    plt.close()


def save_temporal_error_plot(out_dir: Path, x: np.ndarray, frame_stats: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.plot(x, frame_stats["mean_abs_err_across_joints"], label="Mean |err| across joints", linewidth=1.4)
    plt.plot(x, frame_stats["median_abs_err_across_joints"], label="Median |err| across joints", linewidth=1.2, alpha=0.9)
    plt.title("Temporal error summary across joints")
    plt.xlabel("Aligned time (ms)" if np.all(np.isfinite(x)) else "Aligned index")
    plt.ylabel("Absolute error (rad)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_error_summary.png", dpi=PLOT_DPI)
    plt.close()


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(
    aligned: AlignedSequence,
    out_dir: Path,
    dtw_window_ratio: float = 0.1,
    thresholds: Optional[List[float]] = None,
    primary_acc_threshold: Optional[float] = None,
    non_circular: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    N = len(aligned.pred_angles)
    if N < 2:
        raise ValueError("Aligned sequence too short for evaluation.")

    x = aligned.gt_time_ms if aligned.gt_time_ms is not None else np.arange(N, dtype=float)
    dtw_window = max(1, int(round(dtw_window_ratio * N)))

    per_joint_rows: List[Dict[str, float]] = []
    err_df_for_plots = pd.DataFrame(index=np.arange(N))

    plots_dir = out_dir / "plots"
    safe_makedirs(plots_dir)

    thr_list = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    unit = ANGLE_UNIT

    for joint in JOINT_COLUMNS:
        p = aligned.pred_angles[joint].to_numpy(dtype=float)
        g = aligned.gt_angles[joint].to_numpy(dtype=float)

        metrics = compute_joint_metrics(
            pred=p,
            gt=g,
            unit=unit,
            dtw_window=dtw_window,
            thresholds=thr_list,
            primary_threshold=primary_acc_threshold,
            non_circular=non_circular,
        )
        metrics["joint"] = joint
        metrics["unit"] = unit
        per_joint_rows.append(metrics)

        raw_err = p - g
        circ_err = wrap_angle_diff(raw_err, unit=unit)

        plot_err = circ_err if (PLOT_CIRCULAR_ERROR and not non_circular) else raw_err
        plot_label = "Circular error" if (PLOT_CIRCULAR_ERROR and not non_circular) else "Raw error"

        err_df_for_plots[joint] = plot_err

        save_time_series_overlay(plots_dir, joint, x, p, g)
        save_error_over_time(plots_dir, joint, x, plot_err, plot_label)
        save_error_distribution(plots_dir, joint, plot_err, plot_label)

    metrics_df = pd.DataFrame(per_joint_rows)
    frame_stats = temporal_error_stats(err_df_for_plots)

    # Consistent error matrix for analysis
    err_for_analysis = compute_error_matrix(
        pred_angles=aligned.pred_angles,
        gt_angles=aligned.gt_angles,
        non_circular=non_circular,
        unit=unit,
    )

    thr_list = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

    per_joint_time_df = per_joint_accuracy_over_time(
        err_df=err_for_analysis,
        aligned_time=x,
        thresholds=thr_list,
        include_threshold_flags=True,   # keep your within_* columns too
        include_signed_error=False,
    )
    per_joint_time_df.to_csv(out_dir / "per_joint_accuracy_over_time.csv", index=False)



    # Max joint error over time + offenders
    max_df = max_joint_error_over_time(err_for_analysis)
    max_df_out = pd.DataFrame({
        "aligned_time": x,
        "max_abs_error": max_df["max_abs_error"].to_numpy(dtype=float),
        "worst_joint": max_df["worst_joint"].astype(str).to_numpy(),
    })
    max_df_out.to_csv(out_dir / "max_joint_error_over_time.csv", index=False)
    save_max_joint_error_plot(plots_dir, x, max_df)

    offenders_df = top_offending_joints_table(err_for_analysis)
    offenders_df.to_csv(out_dir / "top_offending_joints.csv", index=False)

    # Overall accuracy over time for all thresholds + extra stats
    thr_acc_df = overall_threshold_accuracy_over_time(err_for_analysis, thresholds=thr_list)
    temporal_df = temporal_error_stats(err_for_analysis)
    pattern_df = overall_pattern_metrics_over_time(
        pred_angles=aligned.pred_angles,
        gt_angles=aligned.gt_angles,
        window=max(9, int(round(0.05 * N))),
    )

    overall_time_df = pd.concat(
        [
            pd.DataFrame({"aligned_time": x}),
            thr_acc_df.reset_index(drop=True),
            temporal_df.reset_index(drop=True),
            max_df.reset_index(drop=True).rename(columns={"max_abs_error": "max_abs_error_across_joints"}),
            pattern_df.reset_index(drop=True),
        ],
        axis=1,
    )
    overall_time_df.to_csv(out_dir / "overall_accuracy_over_time.csv", index=False)

    # Keep old single-threshold series + plot
    overall_thr = primary_acc_threshold if primary_acc_threshold is not None else float(thr_list[0])
    acc_df = overall_accuracy_over_time(err_for_analysis, threshold=float(overall_thr))
    acc_df_out = pd.DataFrame({
        "aligned_time": x,
        "overall_accuracy_pct": acc_df["overall_accuracy_pct"].to_numpy(dtype=float),
    })
    acc_df_out.to_csv(out_dir / "overall_accuracy_over_time_primary_only.csv", index=False)
    save_overall_accuracy_plot(plots_dir, x, acc_df, threshold=float(overall_thr))

    # Summary plots
    save_summary_bars(plots_dir, metrics_df, "raw_mae", "Raw MAE per joint")
    save_summary_bars(plots_dir, metrics_df, "raw_rmse", "Raw RMSE per joint")
    save_summary_bars(plots_dir, metrics_df, "circ_mae", "Circular MAE per joint (raw if non-circular)")
    save_summary_bars(plots_dir, metrics_df, "circ_rmse", "Circular RMSE per joint (raw if non-circular)")
    save_summary_bars(plots_dir, metrics_df, "pearson_r", "Pearson correlation per joint")
    save_summary_bars(plots_dir, metrics_df, "pattern_acc_corr_pct", "Pattern accuracy (correlation) per joint")
    save_summary_bars(plots_dir, metrics_df, "cosine_sim_z", "Z-normalized cosine similarity per joint")
    save_summary_bars(plots_dir, metrics_df, "dtw_dist_z", "DTW distance (z-scored) per joint")
    if "primary_pct_accuracy" in metrics_df.columns:
        save_summary_bars(plots_dir, metrics_df, "primary_pct_accuracy", "Primary percentage accuracy per joint")
    save_temporal_error_plot(plots_dir, x, frame_stats)

    agg = aggregate_metrics(metrics_df)
    agg["temporal_mean_of_mean_abs_err"] = float(frame_stats["mean_abs_err_across_joints"].mean())
    agg["temporal_p95_of_mean_abs_err"] = float(np.percentile(frame_stats["mean_abs_err_across_joints"], 95))

    return metrics_df, frame_stats, agg


def print_concise_summary(agg: Dict[str, float]) -> None:
    def g(k: str) -> float:
        return float(agg.get(k, float("nan")))

    print("\n=== Evaluation summary ===")
    print(f"Raw MAE (mean across joints): {g('mean_raw_mae'):.6g}")
    print(f"Raw RMSE (mean across joints): {g('mean_raw_rmse'):.6g}")
    print(f"Circ MAE (mean across joints): {g('mean_circ_mae'):.6g}")
    print(f"Circ RMSE (mean across joints): {g('mean_circ_rmse'):.6g}")
    print(f"Pearson r (mean across joints): {g('mean_pearson_r'):.6g}")
    print(f"Pattern acc corr pct (mean joints): {g('mean_pattern_acc_corr_pct'):.6g}")
    print(f"Cosine sim z (mean across joints): {g('mean_cosine_sim_z'):.6g}")
    print(f"DTW dist z (mean across joints): {g('mean_dtw_dist_z'):.6g}")
    print(f"Vel direction agreement (mean joints): {g('mean_vel_direction_agreement'):.6g}")
    if "mean_primary_pct_accuracy" in agg:
        print(f"Primary pct accuracy (mean joints): {g('mean_primary_pct_accuracy'):.6g}")
    print(f"Temporal mean(|err|) across joints: {g('temporal_mean_of_mean_abs_err'):.6g}")
    print(f"Temporal p95(|err|) across joints: {g('temporal_p95_of_mean_abs_err'):.6g}")


# -----------------------------
# Multi-file evaluation (file_key)
# -----------------------------

def evaluate_multiple_pred_files(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    base_out_dir: Path,
    gt_stem: str,
    alignment: str,
    pred_fps: Optional[float],
    dtw_window_ratio: float,
    acc_thresholds: List[float],
    primary_acc_threshold: Optional[float],
    non_circular: bool,
) -> None:
    results_rows: List[pd.DataFrame] = []
    summaries: Dict[str, Dict[str, float]] = {}

    alignment_name: str = "normalized_progress(pred index -> nearest gt index)"

    for fk, gpred in pred_df.groupby("file_key", sort=True):
        out_dir = base_out_dir / gt_stem / str(fk)
        safe_makedirs(out_dir)
        set_plot_style()

        if alignment == "nearest_time":
            if pred_fps is None:
                raise ValueError("nearest_time alignment requires --pred_fps")
            aligned = align_nearest_time(gpred, gt_df, pred_fps=float(pred_fps))
            alignment_name = "nearest_time(frame->ms via pred_fps, then nearest gt time_ms)"
        else:
            aligned = align_normalized_progress(gpred, gt_df)
            alignment_name = "normalized_progress(pred index -> nearest gt index)"

        metrics_df, frame_stats, agg = evaluate(
            aligned,
            out_dir,
            dtw_window_ratio=float(dtw_window_ratio),
            thresholds=acc_thresholds,
            primary_acc_threshold=primary_acc_threshold,
            non_circular=non_circular,
        )

        metrics_df.insert(0, "file_key", fk)
        results_rows.append(metrics_df)

        metrics_df.to_csv(out_dir / "metrics_per_joint.csv", index=False)
        frame_stats.to_csv(out_dir / "temporal_error_stats.csv", index=False)

        summary = {
            "alignment": alignment_name,
            "non_circular": bool(non_circular),
            "plots_use_circular_error": bool(PLOT_CIRCULAR_ERROR and (not non_circular)),
            "thresholds": acc_thresholds,
            "primary_acc_threshold": primary_acc_threshold,
            "aggregated": agg,
        }
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summaries[str(fk)] = agg

    root_out = base_out_dir / gt_stem
    safe_makedirs(root_out)

    all_metrics = pd.concat(results_rows, axis=0, ignore_index=True)
    all_metrics.to_csv(root_out / "per_file_metrics_per_joint.csv", index=False)

    with open(root_out / "per_file_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alignment": alignment_name,
                "non_circular": bool(non_circular),
                "thresholds": acc_thresholds,
                "primary_acc_threshold": primary_acc_threshold,
                "summaries_by_file_key": summaries,
            },
            f,
            indent=2,
        )


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate predicted vs ground-truth motion (joint angles only).")
    p.add_argument("--predicted", required=True, type=str, help="Path to predicted CSV file.")
    p.add_argument("--ground", required=True, type=str, help="Path to ground truth CSV file.")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to save results and plots.")
    p.add_argument(
        "--alignment",
        default="progress",
        choices=["progress", "nearest_time"],
        help="Alignment strategy: progress (default) or nearest_time (requires --pred_fps).",
    )
    p.add_argument(
        "--pred_fps",
        default=None,
        type=float,
        help="Predicted FPS for nearest_time alignment. If not set, nearest_time is unavailable.",
    )
    p.add_argument(
        "--dtw_window_ratio",
        default=0.1,
        type=float,
        help="DTW Sakoe-Chiba window ratio (default 0.1).",
    )
    p.add_argument(
        "--acc_thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds for percentage accuracy, in radians.",
    )
    p.add_argument(
        "--primary_acc_threshold",
        type=float,
        default=None,
        help="If set, also report primary_pct_accuracy per joint using this threshold (radians).",
    )
    p.add_argument(
        "--non_circular",
        action="store_true",
        help="Treat joints as non-circular (bounded). Uses raw error for threshold accuracies.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    pred_path = Path(args.predicted)
    gt_path = Path(args.ground)
    base_out_dir = Path(args.output_dir)
    gt_name = Path(args.ground).stem
    out_dir = base_out_dir / gt_name

    safe_makedirs(out_dir)
    set_plot_style()

    pred_df = load_predicted_csv(pred_path)
    gt_df = load_ground_truth_csv(gt_path)

    thresholds = list(args.acc_thresholds)
    primary_thr = args.primary_acc_threshold
    non_circular = bool(args.non_circular)

    if "file_key" in pred_df.columns and pred_df["file_key"].nunique(dropna=False) > 1:
        evaluate_multiple_pred_files(
            pred_df=pred_df,
            gt_df=gt_df,
            base_out_dir=base_out_dir,
            gt_stem=gt_name,
            alignment=args.alignment,
            pred_fps=args.pred_fps,
            dtw_window_ratio=float(args.dtw_window_ratio),
            acc_thresholds=thresholds,
            primary_acc_threshold=primary_thr,
            non_circular=non_circular,
        )
        print(f"Saved per-file results under: {out_dir}")
        return

    if args.alignment == "nearest_time":
        if args.pred_fps is None:
            raise ValueError("nearest_time alignment requires --pred_fps")
        aligned = align_nearest_time(pred_df, gt_df, pred_fps=float(args.pred_fps))
        alignment_name = "nearest_time(frame->ms via pred_fps, then nearest gt time_ms)"
    else:
        aligned = align_normalized_progress(pred_df, gt_df)
        alignment_name = "normalized_progress(pred index -> nearest gt index)"

    metrics_df, frame_stats, agg = evaluate(
        aligned,
        out_dir,
        dtw_window_ratio=float(args.dtw_window_ratio),
        thresholds=thresholds,
        primary_acc_threshold=primary_thr,
        non_circular=non_circular,
    )

    metrics_df.to_csv(out_dir / "metrics_per_joint.csv", index=False)
    frame_stats.to_csv(out_dir / "temporal_error_stats.csv", index=False)

    summary = {
        "alignment": alignment_name,
        "non_circular": bool(non_circular),
        "plots_use_circular_error": bool(PLOT_CIRCULAR_ERROR and (not non_circular)),
        "thresholds": thresholds,
        "primary_acc_threshold": primary_thr,
        "unit": ANGLE_UNIT,
        "aggregated": agg,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_concise_summary(agg)


if __name__ == "__main__":
    main()
