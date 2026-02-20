#!/usr/bin/env python3
"""
Align mocap CSV to sensor CSV by:
1) Detecting the initial stillness segment in each file using a user-selected column.
2) Computing a single time offset (shift) using stillness end (motion onset) as the anchor.
3) Syncing the ENTIRE overlapping recording by timestamps with sensor as master:
   - For each sensor timestamp, take mocap values from the closest mocap timestamp.
4) Dropping only sensor rows that fall outside mocap coverage after shifting.
   (This typically drops the sensor tail if sensor lasts longer.)

Outputs:
- --out_sensor: sensor rows kept (full from beginning, except any rows outside mocap coverage)
- --out_mocap: mocap numeric columns aligned to each kept sensor timestamp (nearest neighbor), plus shifted mocap time column
- --out_merged (optional): sensor + aligned mocap in one CSV

Notes:
- Stillness detection uses ONLY the specified stillness column.
- Alignment is timestamp-based, not index-based.
- Mocap alignment includes only numeric mocap columns (besides time). Non-numeric mocap columns are not included.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Robust stats + helpers
# -----------------------------

def robust_mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (MAD), unscaled."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def ensure_numeric(series: pd.Series, name: str) -> pd.Series:
    """Coerce series to numeric; raise if everything becomes NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        raise ValueError(f"Column '{name}' is not numeric (all values became NaN after coercion).")
    return s


def validate_time_column(df: pd.DataFrame, time_col: str, label: str) -> pd.DataFrame:
    """
    Ensure time column exists, numeric, sorted ascending, unique.
    Drops duplicate timestamps (keep first) with a warning.
    """
    if time_col not in df.columns:
        raise ValueError(f"{label}: time column '{time_col}' not found in CSV header.")

    df = df.copy()
    df[time_col] = ensure_numeric(df[time_col], f"{label}.{time_col}")
    df = df[df[time_col].notna()].copy()
    if df.empty:
        raise ValueError(f"{label}: all timestamps are NaN after parsing '{time_col}'.")

    df.sort_values(time_col, inplace=True, kind="mergesort")

    dup_mask = df[time_col].duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        print(f"[WARN] {label}: dropped {dup_count} rows with duplicate timestamps in '{time_col}'.", file=sys.stderr)
        df = df.loc[~dup_mask].copy()

    t = df[time_col].to_numpy(dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{label}: timestamps in '{time_col}' are not strictly increasing after cleanup.")

    return df


def time_rolling_median(df: pd.DataFrame, time_col: str, value_col: str, smooth_ms: int) -> pd.Series:
    """Time-based rolling median using a TimedeltaIndex derived from time_col."""
    if smooth_ms <= 0:
        return df[value_col].copy()

    t0 = float(df[time_col].iloc[0])
    td_index = pd.to_timedelta(df[time_col].astype(float) - t0, unit="ms")

    s = df[value_col].copy()
    s.index = td_index

    window = f"{int(smooth_ms)}ms"
    smoothed = s.rolling(window=window, center=True, min_periods=1).median()
    return smoothed.reset_index(drop=True)


def find_first_sustained_run(
    times_ms: np.ndarray,
    condition: np.ndarray,
    min_duration_ms: int,
    start_idx: int = 0
) -> Optional[Tuple[int, int]]:
    """
    Find first run where condition is True lasting >= min_duration_ms, using timestamps for duration.
    Returns (run_start_idx, run_end_idx_exclusive) or None.
    """
    n = len(condition)
    i = max(0, start_idx)

    while i < n:
        if not condition[i]:
            i += 1
            continue

        run_start = i
        j = i + 1
        while j < n and condition[j]:
            j += 1

        dur = float(times_ms[j - 1] - times_ms[run_start]) if j - 1 >= run_start else 0.0
        if dur >= float(min_duration_ms):
            return run_start, j

        i = j

    return None


# -----------------------------
# Stillness detection
# -----------------------------

@dataclass
class StillDetectionResult:
    still_start_ms: float
    still_end_ms: float
    duration_ms: float
    used_fallback: bool
    warnings: List[str]
    threshold: float
    noise_median: float
    noise_mad: float


def detect_still_segment(
    df: pd.DataFrame,
    label: str,
    time_col: str,
    still_col: str,
    *,
    baseline_ms: int,
    target_still_ms: int,
    min_still_ms: int,
    min_active_ms: int,
    smooth_ms: int,
    k: float,
    use_diff: bool
) -> StillDetectionResult:
    """
    Detect initial stillness segment [still_start_ms, still_end_ms] for one file.
    Uses ONLY still_col for detection.

    Fallback if missing: [first_timestamp, first_timestamp + target_still_ms]
    """
    warnings: List[str] = []

    if still_col not in df.columns:
        raise ValueError(f"{label}: stillness column '{still_col}' not found in CSV header.")

    t = df[time_col].to_numpy(dtype=float)
    x = ensure_numeric(df[still_col], f"{label}.{still_col}").to_numpy(dtype=float)

    # Activity signal
    if use_diff:
        dx = np.diff(x, prepend=x[0])
        activity = np.abs(dx)
    else:
        t0 = float(t[0])
        baseline_mask = (t >= t0) & (t <= t0 + float(baseline_ms))
        if int(baseline_mask.sum()) < 5:
            warnings.append(
                f"{label}: baseline window too small ({int(baseline_mask.sum())} samples). "
                "Using first 200 samples for baseline."
            )
            baseline_slice = x[: min(len(x), 200)]
        else:
            baseline_slice = x[baseline_mask]

        baseline_slice = baseline_slice[np.isfinite(baseline_slice)]
        if baseline_slice.size == 0:
            warnings.append(f"{label}: baseline median failed (no finite values). Using fallback window.")
            still_start = float(t0)
            still_end = float(t0 + target_still_ms)
            return StillDetectionResult(
                still_start_ms=still_start,
                still_end_ms=still_end,
                duration_ms=still_end - still_start,
                used_fallback=True,
                warnings=warnings,
                threshold=float("nan"),
                noise_median=float("nan"),
                noise_mad=float("nan"),
            )

        baseline_med = float(np.median(baseline_slice))
        activity = np.abs(x - baseline_med)

    # Smooth activity
    tmp = pd.DataFrame({time_col: t, "_activity": activity})
    tmp = validate_time_column(tmp, time_col, f"{label} (activity)")
    t2 = tmp[time_col].to_numpy(dtype=float)
    smoothed = time_rolling_median(tmp, time_col, "_activity", smooth_ms).to_numpy(dtype=float)

    # Noise estimation
    t0 = float(t2[0])
    noise_mask = (t2 >= t0) & (t2 <= t0 + float(baseline_ms))
    if int(noise_mask.sum()) < 10:
        warnings.append(
            f"{label}: noise window too small ({int(noise_mask.sum())} samples). "
            "Trying first 500 ms, else first 200 samples."
        )
        noise_mask = (t2 >= t0) & (t2 <= t0 + 500.0)
        if int(noise_mask.sum()) < 10:
            noise_mask = np.zeros_like(t2, dtype=bool)
            noise_mask[: min(len(t2), 200)] = True

    noise_vals = smoothed[noise_mask]
    noise_vals = noise_vals[np.isfinite(noise_vals)]
    if noise_vals.size == 0:
        warnings.append(f"{label}: noise estimation failed (no finite values). Using fallback window.")
        still_start = float(t2[0])
        still_end = float(t2[0] + target_still_ms)
        return StillDetectionResult(
            still_start_ms=still_start,
            still_end_ms=still_end,
            duration_ms=still_end - still_start,
            used_fallback=True,
            warnings=warnings,
            threshold=float("nan"),
            noise_median=float("nan"),
            noise_mad=float("nan"),
        )

    noise_median = float(np.median(noise_vals))
    noise_mad = robust_mad(noise_vals)
    if (not np.isfinite(noise_mad)) or noise_mad == 0.0:
        warnings.append(f"{label}: MAD is zero/invalid. Adding epsilon to avoid overly strict threshold.")
        noise_mad = 1e-12

    still_threshold = noise_median + float(k) * float(noise_mad)

    is_still = smoothed < still_threshold
    is_active = ~is_still

    # Still start
    still_run = find_first_sustained_run(t2, is_still, min_duration_ms=min_still_ms, start_idx=0)
    if still_run is None:
        warnings.append(
            f"{label}: no sustained still run found (min_still_ms={min_still_ms}). "
            "Falling back to target window from first timestamp."
        )
        still_start = float(t2[0])
        still_end = float(t2[0] + target_still_ms)
        return StillDetectionResult(
            still_start_ms=still_start,
            still_end_ms=still_end,
            duration_ms=still_end - still_start,
            used_fallback=True,
            warnings=warnings,
            threshold=still_threshold,
            noise_median=noise_median,
            noise_mad=noise_mad,
        )

    still_start_idx, _ = still_run
    still_start = float(t2[still_start_idx])

    # Still end (motion onset)
    active_run = find_first_sustained_run(t2, is_active, min_duration_ms=min_active_ms, start_idx=still_start_idx)
    if active_run is None:
        warnings.append(
            f"{label}: no sustained active run found after still start (min_active_ms={min_active_ms}). "
            "Falling back to target window from still_start."
        )
        still_end = float(still_start + target_still_ms)
        used_fallback = True
    else:
        active_start_idx, _ = active_run
        still_end = float(t2[active_start_idx])
        used_fallback = False

        if still_end <= still_start:
            warnings.append(
                f"{label}: detected still_end <= still_start (end={still_end}, start={still_start}). "
                "Falling back to target window from still_start."
            )
            still_end = float(still_start + target_still_ms)
            used_fallback = True

    return StillDetectionResult(
        still_start_ms=still_start,
        still_end_ms=still_end,
        duration_ms=float(still_end - still_start),
        used_fallback=used_fallback,
        warnings=warnings,
        threshold=still_threshold,
        noise_median=noise_median,
        noise_mad=noise_mad,
    )


# -----------------------------
# Nearest timestamp alignment
# -----------------------------

def build_numeric_column_list(df: pd.DataFrame, time_col: str) -> List[str]:
    """Columns that are numeric-ish (besides time)."""
    cols: List[str] = []
    for c in df.columns:
        if c == time_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def nearest_mocap_to_sensor(
    mocap_df_shifted: pd.DataFrame,
    mocap_time_shifted_col: str,
    sensor_times: np.ndarray
) -> pd.DataFrame:
    """
    For each sensor timestamp, take mocap numeric columns from the closest mocap timestamp.
    """
    t = mocap_df_shifted[mocap_time_shifted_col].to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("mocap_shifted: no timestamps available for nearest alignment.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("mocap_shifted: timestamps must be strictly increasing for nearest alignment.")

    numeric_cols = build_numeric_column_list(mocap_df_shifted, mocap_time_shifted_col)
    if not numeric_cols:
        raise ValueError("mocap_shifted: no numeric columns to align (besides time).")

    sensor_times = sensor_times.astype(float)
    idx = np.searchsorted(t, sensor_times, side="left")

    left = np.clip(idx - 1, 0, t.size - 1)
    right = np.clip(idx, 0, t.size - 1)

    left_dist = np.abs(sensor_times - t[left])
    right_dist = np.abs(sensor_times - t[right])
    choose_right = right_dist < left_dist
    nearest_idx = np.where(choose_right, right, left)

    out = pd.DataFrame({mocap_time_shifted_col: sensor_times})
    for c in numeric_cols:
        y = pd.to_numeric(mocap_df_shifted[c], errors="coerce").to_numpy(dtype=float)
        out[c] = y[nearest_idx]

    return out


# -----------------------------
# CLI + main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect initial stillness, compute offset, then sync the full recording by timestamps (sensor master)."
    )

    p.add_argument("--mocap_csv", required=True, help="Path to mocap CSV")
    p.add_argument("--sensor_csv", required=True, help="Path to sensor CSV")

    p.add_argument("--mocap_time_col", default="time_ms", help="Mocap timestamp column (ms)")
    p.add_argument("--sensor_time_col", default="Time_ms", help="Sensor timestamp column (ms)")

    p.add_argument("--mocap_still_col", required=True, help="Mocap column used to detect stillness")
    p.add_argument("--sensor_still_col", required=True, help="Sensor column used to detect stillness")

    p.add_argument("--out_mocap", required=True, help="Output path for aligned mocap CSV")
    p.add_argument("--out_sensor", required=True, help="Output path for aligned sensor CSV")
    p.add_argument("--out_merged", default=None, help="Optional output path for merged synced CSV")

    # Detection parameters
    p.add_argument("--baseline_ms", type=int, default=1500, help="Early window for baseline/noise estimation (ms)")
    p.add_argument("--target_still_ms", type=int, default=5000, help="Fallback still window length (ms)")
    p.add_argument("--min_still_ms", type=int, default=4000, help="Minimum still run duration to accept still start (ms)")
    p.add_argument("--min_active_ms", type=int, default=150, help="Minimum active run duration to mark still end (ms)")
    p.add_argument("--smooth_ms", type=int, default=80, help="Rolling median window for activity smoothing (ms)")
    p.add_argument("--k", type=float, default=10.0, help="MAD multiplier for still threshold")

    p.add_argument("--mocap_use_diff", action="store_true", help="Use abs(x(t)-x(t-1)) activity for mocap stillness col")
    p.add_argument("--sensor_use_diff", action="store_true", help="Use abs(x(t)-x(t-1)) activity for sensor stillness col")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    mocap_df = pd.read_csv(args.mocap_csv)
    sensor_df = pd.read_csv(args.sensor_csv)

    mocap_df = validate_time_column(mocap_df, args.mocap_time_col, "mocap")
    sensor_df = validate_time_column(sensor_df, args.sensor_time_col, "sensor")

    # Detect still segments (used only for time shift)
    mocap_still = detect_still_segment(
        mocap_df, "mocap",
        time_col=args.mocap_time_col,
        still_col=args.mocap_still_col,
        baseline_ms=args.baseline_ms,
        target_still_ms=args.target_still_ms,
        min_still_ms=args.min_still_ms,
        min_active_ms=args.min_active_ms,
        smooth_ms=args.smooth_ms,
        k=args.k,
        use_diff=bool(args.mocap_use_diff),
    )

    sensor_still = detect_still_segment(
        sensor_df, "sensor",
        time_col=args.sensor_time_col,
        still_col=args.sensor_still_col,
        baseline_ms=args.baseline_ms,
        target_still_ms=args.target_still_ms,
        min_still_ms=args.min_still_ms,
        min_active_ms=args.min_active_ms,
        smooth_ms=args.smooth_ms,
        k=args.k,
        use_diff=bool(args.sensor_use_diff),
    )

    # Compute constant shift (anchor on still_end)
    shift_ms = float(sensor_still.still_end_ms - mocap_still.still_end_ms)

    mocap_time_shifted_col = f"{args.mocap_time_col}_shifted_to_sensor"
    mocap_shifted = mocap_df.copy()
    mocap_shifted[mocap_time_shifted_col] = mocap_shifted[args.mocap_time_col].astype(float) + shift_ms
    mocap_shifted = validate_time_column(mocap_shifted, mocap_time_shifted_col, "mocap_shifted")

    # Determine overlap coverage (in sensor timeline)
    mocap_min_t = float(mocap_shifted[mocap_time_shifted_col].iloc[0])
    mocap_max_t = float(mocap_shifted[mocap_time_shifted_col].iloc[-1])

    sensor_min_t = float(sensor_df[args.sensor_time_col].iloc[0])
    sensor_max_t = float(sensor_df[args.sensor_time_col].iloc[-1])

    # Keep the full overlap range so we do NOT drop the beginning stillness section.
    keep_start = max(sensor_min_t, mocap_min_t)
    keep_end = min(sensor_max_t, mocap_max_t)

    if keep_end <= keep_start:
        raise ValueError(
            "No overlapping time range between sensor and shifted mocap. "
            f"sensor=[{sensor_min_t:.3f},{sensor_max_t:.3f}] mocap_shifted=[{mocap_min_t:.3f},{mocap_max_t:.3f}]"
        )

    # Keep sensor rows within overlap coverage (this drops only the tail if sensor lasts longer)
    sensor_out = sensor_df[
        (sensor_df[args.sensor_time_col] >= keep_start) &
        (sensor_df[args.sensor_time_col] <= keep_end)
    ].copy()

    if sensor_out.empty:
        raise ValueError("Sensor output window is empty after applying overlap coverage.")

    sensor_times = sensor_out[args.sensor_time_col].to_numpy(dtype=float)

    # Align mocap to each kept sensor timestamp using closest mocap timestamp
    mocap_aligned = nearest_mocap_to_sensor(
        mocap_df_shifted=mocap_shifted,
        mocap_time_shifted_col=mocap_time_shifted_col,
        sensor_times=sensor_times,
    )

    if len(mocap_aligned) != len(sensor_out):
        raise RuntimeError(f"Row mismatch: mocap_aligned={len(mocap_aligned)} vs sensor_out={len(sensor_out)}")

    # Write outputs
    mocap_aligned.to_csv(args.out_mocap, index=False)
    sensor_out.to_csv(args.out_sensor, index=False)

    merged_path = None
    if args.out_merged:
        merged = sensor_out.merge(
            mocap_aligned,
            left_on=args.sensor_time_col,
            right_on=mocap_time_shifted_col,
            how="left",
            validate="one_to_one",
            suffixes=("_sensor", "_mocap"),
        )
        merged.to_csv(args.out_merged, index=False)
        merged_path = args.out_merged

    # Report
    print("\n=== Full Recording Sync Report ===")
    print("Inputs:")
    print(f"  mocap_csv:   {args.mocap_csv}")
    print(f"  sensor_csv:  {args.sensor_csv}")
    print(f"  mocap_time_col:   {args.mocap_time_col}")
    print(f"  sensor_time_col:  {args.sensor_time_col}")
    print(f"  mocap_still_col:  {args.mocap_still_col} (use_diff={bool(args.mocap_use_diff)})")
    print(f"  sensor_still_col: {args.sensor_still_col} (use_diff={bool(args.sensor_use_diff)})")

    print("\nDetected still segments (original timelines):")
    print(f"  mocap:  start={mocap_still.still_start_ms:.3f}  end={mocap_still.still_end_ms:.3f}  dur={mocap_still.duration_ms:.3f} ms")
    print(f"  sensor: start={sensor_still.still_start_ms:.3f}  end={sensor_still.still_end_ms:.3f}  dur={sensor_still.duration_ms:.3f} ms")

    print("\nTime shift:")
    print(f"  shift_ms = sensor_still_end - mocap_still_end = {shift_ms:.3f} ms")
    print(f"  shifted mocap time col: {mocap_time_shifted_col}")

    print("\nCoverage (sensor timeline):")
    print(f"  sensor coverage:        [{sensor_min_t:.3f}, {sensor_max_t:.3f}] ms")
    print(f"  shifted mocap coverage: [{mocap_min_t:.3f}, {mocap_max_t:.3f}] ms")
    print(f"  kept overlap range:     [{keep_start:.3f}, {keep_end:.3f}] ms")
    print(f"  final_row_count:        {len(sensor_out)}")

    print("\nAlignment rule:")
    print("  Sensor timestamps are the master grid (every output row is a sensor row).")
    print("  Mocap numeric columns are sampled at the closest shifted mocap timestamp (nearest neighbor).")
    print("  Sensor rows outside shifted mocap coverage are dropped (typically only the tail).")

    print("\nOutputs:")
    print(f"  out_sensor: {args.out_sensor}")
    print(f"  out_mocap:  {args.out_mocap}")
    if merged_path:
        print(f"  out_merged: {merged_path}")

    all_warnings = mocap_still.warnings + sensor_still.warnings
    if mocap_still.used_fallback:
        all_warnings.append("mocap: used fallback still window")
    if sensor_still.used_fallback:
        all_warnings.append("sensor: used fallback still window")
    if all_warnings:
        print("\nWarnings:")
        for w in all_warnings:
            print(f"  - {w}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
