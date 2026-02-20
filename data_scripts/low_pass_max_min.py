#!/usr/bin/env python3
"""
Filter/smooth joint-angle CSVs in a directory while preserving peaks/patterns,
and save comparison plots.

Input CSV structure (example):
file_key,frame,waist_yaw_joint,...,right_elbow_pitch_joint
air_punch_...,0, ...
...

Behavior:
- Ignores `file_key` for filtering (it is preserved in outputs but never filtered).
- Sorts by `frame` (numeric) before filtering.
- Smooths each joint column with a peak-preserving Savitzky-Golay filter (preferred),
  optionally preceded by a median filter to suppress spikes.
- Writes filtered CSVs to output_dir with the same filename.
- Writes plots (original vs filtered) to plot_dir (one PNG per input file).

Usage:
  python filter_mocap_csvs.py --input_dir ./in --output_dir ./out --plot_dir ./plots

Optional knobs:
  --sg_window 31 --sg_poly 3 --median_window 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Prefer SciPy if available (best peak-preserving smoothing)
try:
    from scipy.signal import savgol_filter, medfilt  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _clip_window_to_length(window: int, n: int) -> int:
    """
    Ensure window is odd and <= n (and at least 3 if possible).
    """
    if n <= 1:
        return 1
    window = _ensure_odd(max(1, window))
    if window > n:
        window = _ensure_odd(n)
    if window < 3 and n >= 3:
        window = 3
    return window


def _median_filter_1d(x: np.ndarray, median_window: int) -> np.ndarray:
    if median_window <= 1:
        return x

    n = len(x)
    w = _clip_window_to_length(median_window, n)
    if w <= 1:
        return x

    if _HAVE_SCIPY:
        # medfilt requires odd kernel size
        return medfilt(x, kernel_size=w)

    # Fallback: rolling median (centered)
    s = pd.Series(x)
    return (
        s.rolling(window=w, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _savgol_smooth_1d(x: np.ndarray, sg_window: int, sg_poly: int) -> np.ndarray:
    n = len(x)
    w = _clip_window_to_length(sg_window, n)
    if w <= 2:
        return x

    # Polyorder must be < window length
    p = int(sg_poly)
    if p >= w:
        p = max(1, w - 1)
    if p < 1:
        p = 1

    if _HAVE_SCIPY:
        # mode='interp' preserves edges better than padding for smooth signals
        return savgol_filter(x, window_length=w, polyorder=p, mode="interp")

    # Fallback: light peak-preserving smoothing:
    # rolling mean on top of a rolling median gives reasonable smoothing without crushing peaks too much.
    s = pd.Series(x)
    x_med = s.rolling(window=w, center=True, min_periods=1).median()
    x_smooth = x_med.rolling(window=w, center=True, min_periods=1).mean()
    return x_smooth.to_numpy(dtype=float)


def smooth_dataframe(
    df: pd.DataFrame,
    sg_window: int,
    sg_poly: int,
    median_window: int,
    ignore_cols: Tuple[str, ...] = ("file_key", "frame"),
) -> pd.DataFrame:
    out = df.copy()

    # Determine numeric joint columns (everything except ignored columns)
    cols = [c for c in out.columns if c not in ignore_cols]
    if not cols:
        return out

    for c in cols:
        # Convert to numeric safely; keep NaNs where conversion fails
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)

        # If all NaN or too short, skip
        if np.all(np.isnan(x)) or len(x) < 3:
            continue

        # Interpolate NaNs for filtering, then restore NaNs
        nan_mask = np.isnan(x)
        if nan_mask.any():
            s = pd.Series(x)
            x_filled = (
                s.interpolate(limit_direction="both")
                .fillna(method="bfill")
                .fillna(method="ffill")
                .to_numpy(dtype=float)
            )
        else:
            x_filled = x

        # Optional spike suppression then Savitzky-Golay smoothing
        x_med = _median_filter_1d(x_filled, median_window)
        x_sm = _savgol_smooth_1d(x_med, sg_window, sg_poly)

        # Restore NaNs where the original had NaNs
        x_sm[nan_mask] = np.nan
        out[c] = x_sm

    return out


def plot_comparison(
    df_raw: pd.DataFrame,
    df_sm: pd.DataFrame,
    out_path: Path,
    max_cols: Optional[int] = None,
    ignore_cols: Tuple[str, ...] = ("file_key", "frame"),
) -> None:
    cols = [c for c in df_raw.columns if c not in ignore_cols]
    if max_cols is not None:
        cols = cols[:max_cols]
    if not cols:
        return

    x = pd.to_numeric(df_raw["frame"], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(x)):
        x = np.arange(len(df_raw), dtype=float)

    ncols = 1
    nrows = len(cols)
    # Height scales with number of joints
    fig_h = max(3.0, 1.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, fig_h), sharex=True)

    if nrows == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        y0 = pd.to_numeric(df_raw[c], errors="coerce").to_numpy(dtype=float)
        y1 = pd.to_numeric(df_sm[c], errors="coerce").to_numpy(dtype=float)

        ax.plot(x, y0, label="original")
        ax.plot(x, y1, label="smoothed")
        ax.set_title(c)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def iter_input_files(input_dir: Path) -> List[Path]:
    exts = {".csv", ".txt"}
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--plots_dir", required=True, type=str)

    ap.add_argument("--sg_window", type=int, default=31, help="Savitzky-Golay window (odd).")
    ap.add_argument("--sg_poly", type=int, default=3, help="Savitzky-Golay polyorder.")
    ap.add_argument("--median_window", type=int, default=5, help="Median filter window (odd). Use 1 to disable.")

    ap.add_argument(
        "--max_plot_cols",
        type=int,
        default=None,
        help="Limit plotted joint columns (useful if you have many). Default: plot all.",
    )
    ap.add_argument(
        "--keep_file_key",
        action="store_true",
        help="Keep file_key column in output (default: True). This flag is here for clarity.",
    )
    ap.add_argument(
        "--drop_file_key",
        action="store_true",
        help="Drop file_key column from output entirely.",
    )

    args = ap.parse_args()

    input_dir = Path(args.in_dir)
    output_dir = Path(args.out_dir)
    plot_dir = Path(args.plots_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_dir)
    if not files:
        raise SystemExit(f"No .csv/.txt files found under: {input_dir}")

    print(f"Found {len(files)} files.")
    print(f"SciPy available: {_HAVE_SCIPY}")

    for in_path in files:
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read CSV
        df = pd.read_csv(in_path)

        # Basic validation
        if "frame" not in df.columns:
            print(f"[SKIP] Missing 'frame' column: {in_path}")
            continue

        # Sort by frame to ensure temporal order
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df = df.sort_values("frame", kind="mergesort").reset_index(drop=True)

        # Smooth joint columns (ignores file_key and frame)
        df_sm = smooth_dataframe(
            df,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            median_window=args.median_window,
            ignore_cols=("file_key", "frame"),
        )

        # Optionally drop file_key
        if args.drop_file_key and "file_key" in df_sm.columns:
            df_sm = df_sm.drop(columns=["file_key"])

        # Write filtered CSV
        df_sm.to_csv(out_path, index=False)

        # Plot comparison
        plot_path = plot_dir / rel.with_suffix(".png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(
            df_raw=df,
            df_sm=df_sm if not (args.drop_file_key) else df_sm.assign(file_key=df.get("file_key")),
            out_path=plot_path,
            max_cols=args.max_plot_cols,
            ignore_cols=("file_key", "frame"),
        )

        print(f"[OK] {in_path} -> {out_path} ; plot -> {plot_path}")

    print("Done.")


if __name__ == "__main__":
    main()
