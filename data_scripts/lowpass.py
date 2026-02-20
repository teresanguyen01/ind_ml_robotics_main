#!/usr/bin/env python3
"""
Smooth joint angle arrays in CSV files and generate plots.

- Reads all CSVs from an input directory
- Uses the provided default joint list unless --joints is given
- Applies a zero-phase low-pass filter (Butterworth/Cheby1/Cheby2) to each joint series
- Preserves a "time_ms" column if present; otherwise indexes by sample
- Writes smoothed CSVs to an output directory
- Creates plots in a plot directory using the user's plotting function

Example:
    python smooth_angles.py \
        --in_dir ./angles_in \
        --out_dir ./angles_out \
        --plots_dir ./plots \
        --method butter \
        --cutoff_frac 0.05 \
        --order 2

Cutoff guidance:
    cutoff_frac is normalized to Nyquist (0<cutoff_frac<1). Try 0.03–0.10.
"""
import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, cheby1, cheby2, filtfilt
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------- Default joints (provided by user) ----------
DEFAULT_JOINTS = [
    'waist_yaw_joint',
    'waist_pitch_joint',
    'waist_roll_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
]


# ---------- Plotting: provided by user ----------
def plot_joint_angles(df_raw, df_sm, joints, output_dir, pic_name):
    os.makedirs(output_dir, exist_ok=True)

    # x axis (frame index)
    t = np.arange(len(df_raw))

    nrows = len(joints)
    fig_h = max(3.0, 1.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, fig_h), sharex=True)

    if nrows == 1:
        axes = [axes]

    for ax, joint in zip(axes, joints):
        if joint not in df_raw.columns:
            continue

        y_raw = df_raw[joint].values
        y_sm = df_sm[joint].values if joint in df_sm.columns else None

        ax.plot(t, y_raw, label="original")
        if y_sm is not None:
            ax.plot(t, y_sm, label="smoothed")

        ax.set_title(joint)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("frame")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{pic_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


# ---------- Filtering utilities ----------
def _interp_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (and back/forward fill edges)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    idx = np.arange(n)
    mask = ~np.isnan(y)
    if mask.all():
        return y
    # If all NaN, return zeros
    if not mask.any():
        return np.zeros_like(y)
    y_interp = np.interp(idx, idx[mask], y[mask])
    return y_interp


def build_filter(method: str, cutoff_frac: float, order: int, rp: float, rs: float):
    """Return a function f(x) that applies zero-phase low-pass filtering with given specs.

    method: 'butter' | 'cheby1' | 'cheby2'
    cutoff_frac: normalized cutoff (fraction of Nyquist), 0<cutoff_frac<1
    order: filter order (integer >=1)
    rp: passband ripple (dB) for Chebyshev I
    rs: stopband attenuation (dB) for Chebyshev II
    """
    if not (0.0 < cutoff_frac < 1.0):
        raise ValueError("cutoff_frac must be between 0 and 1 (fraction of Nyquist).")
    method = method.lower()
    if method == 'butter':
        b, a = butter(order, cutoff_frac, btype='low')
    elif method == 'cheby1':
        b, a = cheby1(order, rp, cutoff_frac, btype='low')
    elif method == 'cheby2':
        b, a = cheby2(order, rs, cutoff_frac, btype='low')
    else:
        raise ValueError("Unknown method. Choose from: butter, cheby1, cheby2.")

    def apply(x: np.ndarray) -> np.ndarray:
        x = _interp_nans(x)
        # Zero-phase forward-backward filtering
        return filtfilt(b, a, x, method="pad")
    return apply


def detect_joints(df: pd.DataFrame, explicit: List[str] = None) -> List[str]:
    # Use provided list by default; warn if missing.
    if explicit is None:
        explicit = DEFAULT_JOINTS
    joints = [j for j in explicit if j in df.columns]
    missing = [j for j in explicit if j not in df.columns]
    if missing:
        print(f"Warning: missing joints in file: {missing}")
    # Fallback: any '*_joint' columns if nothing matched.
    return joints if joints else [c for c in df.columns if c.endswith('_joint')]


def compute_fs_from_time_ms(df: pd.DataFrame) -> float:
    """Compute sampling rate in Hz from 'time_ms' if present; else return NaN."""
    if 'time_ms' not in df.columns:
        return float('nan')
    t = df['time_ms'].to_numpy()
    if len(t) < 2:
        return float('nan')
    dt_ms = np.median(np.diff(t))
    if dt_ms <= 0 or np.isnan(dt_ms):
        return float('nan')
    return 1000.0 / dt_ms


def smooth_file(csv_path: str,
                out_dir: str,
                plots_dir: str,
                method: str,
                cutoff_frac: float,
                order: int,
                rp: float,
                rs: float,
                joints: List[str] = None) -> Tuple[str, str]:
    """Smooth one CSV and plot. Returns (output_csv_path, plot_png_path_prefix)."""
    df = pd.read_csv(csv_path)
    # Pick joints
    joint_cols = detect_joints(df, joints)

    if not joint_cols:
        raise ValueError(f"No joint columns found in {csv_path}. "
                         "Specify --joints or ensure columns match the default list.")

    # Build filter
    filt = build_filter(method, cutoff_frac, order, rp, rs)

    # Apply per joint
    smoothed = df.copy()
    for col in joint_cols:
        smoothed[col] = filt(df[col].to_numpy())

    # Ensure output dirs
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # File names
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"{base}.csv")

    # Write smoothed CSV
    smoothed.to_csv(out_csv, index=False)
    print(f"Saved smoothed CSV: {out_csv}")

    # Plots: raw and smoothed
    plot_joint_angles(df, smoothed, joint_cols, plots_dir, f"{base}_comparison")


    return out_csv, os.path.join(plots_dir, base)


def main():
    parser = argparse.ArgumentParser(description="Smooth joint angle arrays and plot results.")
    parser.add_argument("--in_dir", required=True, help="Directory containing input CSVs.")
    parser.add_argument("--out_dir", required=True, help="Directory to write smoothed CSVs.")
    parser.add_argument("--plots_dir", required=True, help="Directory to save plots.")
    parser.add_argument("--method", choices=["butter", "cheby1", "cheby2"], default="butter",
                        help="Low-pass filter type (default: butter).")
    parser.add_argument("--cutoff_frac", type=float, default=0.05,
                        help="Normalized cutoff as a fraction of Nyquist (e.g., 0.03–0.10).")
    parser.add_argument("--order", type=int, default=2, help="Filter order (default: 2).")
    parser.add_argument("--rp", type=float, default=0.5,
                        help="Passband ripple in dB (Chebyshev I only).")
    parser.add_argument("--rs", type=float, default=20.0,
                        help="Stopband attenuation in dB (Chebyshev II only).")
    parser.add_argument("--joints", type=str, default="",
                        help="Comma-separated list of joint column names to smooth. "
                             "Default: your provided joint list (falls back to any '*_joint' columns).")

    args = parser.parse_args()

    joints = [j.strip() for j in args.joints.split(",")] if args.joints else None

    csv_paths = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.in_dir}")

    print(f"Found {len(csv_paths)} CSV file(s) in {args.in_dir}.")
    for p in csv_paths:
        try:
            smooth_file(p, args.out_dir, args.plots_dir, args.method,
                        args.cutoff_frac, args.order, args.rp, args.rs, joints)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

if __name__ == "__main__":
    main()