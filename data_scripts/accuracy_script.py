#!/usr/bin/env python3
"""
Evaluate paired ground-truth vs predicted time series files.

Pairing rule:
- Predicted filename key: remove "_pred_resamp_smoothed" (before extension)
- Ground-truth filename key: remove "_resamp" (before extension)

Ground truth preprocessing (text files only: .csv/.tsv/.txt):
- Drops columns named exactly: time_ms, x, y, z
- Uses remaining header column names as joint names (they match the joint list you provided)

Plots per paired file (ONLY JOINTS_FOR_PLOTS):
- Pie chart: mean joint accuracy (%)
- Accuracy over time: per-frame mean joint accuracy (%)

Accuracy definition (per file):
- range_j = max(true[:,j]) - min(true[:,j]) for each joint j
- norm_err_tj = |pred_tj - true_tj| / max(range_j, eps)
- joint_accuracy_j = 100 * (1 - mean_t(norm_err_tj))
- accuracy_over_time_t = 100 * (1 - mean_j(norm_err_tj))

Outputs saved to plots_dir:
- joint_accuracy_pie_<key>.png
- accuracy_over_time_<key>.png
- summary_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt


PRED_SUFFIX = "_pred_resamp_smoothed"
GT_SUFFIX = "_resamp"

JOINT_NAMES = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
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


def strip_suffix_stem(filename: str, suffix: str) -> str:
    p = Path(filename)
    stem = p.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def build_key_to_path_map(folder: Path, kind: str) -> Dict[str, Path]:
    if kind not in {"pred", "gt"}:
        raise ValueError("kind must be 'pred' or 'gt'")

    key_map: Dict[str, Path] = {}
    for fp in sorted(folder.glob("*")):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue

        if kind == "pred":
            key = strip_suffix_stem(fp.name, PRED_SUFFIX)
        else:
            key = strip_suffix_stem(fp.name, GT_SUFFIX)

        if key in key_map:
            raise RuntimeError(
                f"Duplicate key '{key}' in {kind} folder:\n"
                f"  - {key_map[key]}\n"
                f"  - {fp}\n"
                "Rename files so keys are unique."
            )

        key_map[key] = fp

    return key_map


def _infer_delimiter(sample_line: str) -> str:
    if sample_line.count("\t") > sample_line.count(","):
        return "\t"
    return ","


def _ensure_2d(arr: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{path} loaded with shape {arr.shape}, expected 1D or 2D.")
    if arr.shape[0] < 2:
        raise ValueError(f"{path} has too few timesteps: shape {arr.shape}")
    return arr.astype(np.float64)


def load_timeseries(path: Path, delimiter: str = "auto") -> np.ndarray:
    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _ensure_2d(arr, path)

    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        if len(z.files) == 1:
            arr = z[z.files[0]]
        else:
            for k in ["data", "y", "X", "arr_0"]:
                if k in z.files:
                    arr = z[k]
                    break
            else:
                raise ValueError(
                    f"{path} is an .npz with multiple arrays {z.files}. "
                    "Add a key like 'data' or ensure only one array exists."
                )
        return _ensure_2d(arr, path)

    # text table
    if delimiter == "auto":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        delim = _infer_delimiter(first)
    else:
        delim = delimiter

    try:
        arr = np.loadtxt(path, delimiter=delim)
    except Exception:
        arr = np.loadtxt(path, delimiter=delim, skiprows=1)

    return _ensure_2d(arr, path)


def load_ground_truth_text_with_column_drop(
    path: Path,
    delimiter: str = "auto",
    drop_cols: Set[str] = {"time_ms", "x", "y", "z"},
) -> Tuple[np.ndarray, List[str]]:
    """
    Load GT text file WITH header.
    - Drops drop_cols by name.
    - Returns (data [T,D], kept_column_names [D]).
    """
    if delimiter == "auto":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()
        delim = _infer_delimiter(header)
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()
        delim = delimiter

    col_names = [c.strip() for c in header.split(delim)]
    keep_indices = [i for i, name in enumerate(col_names) if name not in drop_cols]
    keep_names = [col_names[i] for i in keep_indices]

    if not keep_indices:
        raise ValueError(
            f"All columns were removed from GT file {path.name}. Header: {col_names}"
        )

    data = np.loadtxt(
        path,
        delimiter=delim,
        skiprows=1,
        usecols=keep_indices,
    )

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data = _ensure_2d(data, path)
    if len(keep_names) != data.shape[1]:
        raise ValueError(
            f"Header/shape mismatch in {path.name}: kept {len(keep_names)} names "
            f"but loaded D={data.shape[1]}"
        )

    return data, keep_names


def compute_joint_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true {y_true.shape} vs pred {y_pred.shape}")

    ranges = y_true.max(axis=0) - y_true.min(axis=0)
    ranges = np.where(ranges == 0, eps, ranges)

    abs_err = np.abs(y_pred - y_true)
    norm_err = abs_err / ranges  # [T, D]

    mean_norm_err_per_joint = norm_err.mean(axis=0)  # [D]
    joint_acc = 100.0 * (1.0 - mean_norm_err_per_joint)
    joint_acc = np.clip(joint_acc, 0.0, 100.0)

    mean_norm_err_per_frame = norm_err.mean(axis=1)  # [T]
    acc_t = 100.0 * (1.0 - mean_norm_err_per_frame)
    acc_t = np.clip(acc_t, 0.0, 100.0)

    return joint_acc, acc_t


def save_joint_accuracy_pie(
    joint_acc_pct: np.ndarray,
    joint_labels: List[str],
    out_path: Path,
    title: str,
) -> None:
    if float(np.max(joint_acc_pct)) <= 0.0:
        return

    def make_autopct(values: np.ndarray):
        total = float(np.sum(values))
        def _autopct(pct: float) -> str:
            val = pct / 100.0 * total
            return f"{val:.1f}%"
        return _autopct

    plt.figure(figsize=(7, 7))
    plt.pie(
        joint_acc_pct,
        labels=joint_labels,
        autopct=make_autopct(joint_acc_pct),
        startangle=90,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_accuracy_over_time(
    acc_t: np.ndarray,
    dt_ms: float,
    out_path: Path,
    title: str,
) -> None:
    t_axis = np.arange(len(acc_t)) * float(dt_ms)
    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, acc_t)
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--plots_dir", type=str, required=True)
    ap.add_argument("--dt_ms", type=float, default=10.0)
    ap.add_argument("--delimiter", type=str, default="auto")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    gt_map = build_key_to_path_map(gt_dir, kind="gt")
    pred_map = build_key_to_path_map(pred_dir, kind="pred")

    common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    missing_in_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    missing_in_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))

    print(f"[INFO] GT files:   {len(gt_map)}")
    print(f"[INFO] Pred files: {len(pred_map)}")
    print(f"[INFO] Paired:     {len(common)}")
    if missing_in_pred:
        print(f"[WARN] Missing predictions for {len(missing_in_pred)} keys (up to 20): {missing_in_pred[:20]}")
    if missing_in_gt:
        print(f"[WARN] Missing ground truth for {len(missing_in_gt)} keys (up to 20): {missing_in_gt[:20]}")

    joint_name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}

    summary_rows: List[Dict[str, object]] = []

    for key in common:
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        print(f"\n[PAIR] {key}")
        print(f"  GT:   {gt_path.name}")
        print(f"  Pred: {pred_path.name}")

        # GT: drop time_ms, x, y, z, and capture header-based joint names
        if gt_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            y_true, gt_joint_cols = load_ground_truth_text_with_column_drop(gt_path, delimiter=args.delimiter)
        else:
            # For non-text GT, we cannot read column names. We assume columns match JOINT_NAMES order.
            y_true = load_timeseries(gt_path, delimiter=args.delimiter)
            gt_joint_cols = JOINT_NAMES[: y_true.shape[1]]

        # Pred: load numeric array
        y_pred = load_timeseries(pred_path, delimiter=args.delimiter)

        # Align time length
        T = min(y_true.shape[0], y_pred.shape[0])
        if y_true.shape[0] != y_pred.shape[0]:
            print(f"  [WARN] Length mismatch: true {y_true.shape[0]} vs pred {y_pred.shape[0]}. Truncating to {T}.")
            y_true = y_true[:T]
            y_pred = y_pred[:T]

        if y_true.shape[1] != y_pred.shape[1]:
            raise ValueError(
                f"Joint count mismatch for key '{key}': true D={y_true.shape[1]} vs pred D={y_pred.shape[1]}"
            )

        # Validate that GT joint columns are known (optional, but helpful)
        unknown = [j for j in gt_joint_cols if j not in joint_name_to_idx]
        if unknown:
            print(f"  [WARN] GT has joint columns not in JOINT_NAMES (will still use them): {unknown}")

        # Select joints for plots based on GT header order
        gt_name_to_idx = {name: i for i, name in enumerate(gt_joint_cols)}
        selected_joint_names = [j for j in JOINTS_FOR_PLOTS if j in gt_name_to_idx]
        selected_indices = [gt_name_to_idx[j] for j in selected_joint_names]

        if not selected_indices:
            print("  [WARN] No JOINTS_FOR_PLOTS found in this file's columns. Skipping plots.")
            continue

        y_true_sel = y_true[:, selected_indices]
        y_pred_sel = y_pred[:, selected_indices]

        joint_acc_pct, acc_t = compute_joint_accuracy(y_true_sel, y_pred_sel)

        pie_path = plots_dir / f"joint_accuracy_pie_{key}.png"
        time_path = plots_dir / f"accuracy_over_time_{key}.png"

        save_joint_accuracy_pie(
            joint_acc_pct=joint_acc_pct,
            joint_labels=selected_joint_names,
            out_path=pie_path,
            title=f"Joint accuracy (mean)\nfile: {key}",
        )
        save_accuracy_over_time(
            acc_t=acc_t,
            dt_ms=args.dt_ms,
            out_path=time_path,
            title=f"Accuracy over time\nfile: {key}",
        )

        row = {
            "key": key,
            "gt_file": gt_path.name,
            "pred_file": pred_path.name,
            "T_used": int(T),
            "num_joints_plotted": int(len(selected_joint_names)),
            "mean_accuracy_over_time_pct": float(np.mean(acc_t)),
            "median_accuracy_over_time_pct": float(np.median(acc_t)),
            "min_accuracy_over_time_pct": float(np.min(acc_t)),
            "max_accuracy_over_time_pct": float(np.max(acc_t)),
            "mean_joint_accuracy_pct": float(np.mean(joint_acc_pct)),
            "min_joint_accuracy_pct": float(np.min(joint_acc_pct)),
            "max_joint_accuracy_pct": float(np.max(joint_acc_pct)),
        }
        summary_rows.append(row)

        print(f"  [SAVED] {pie_path}")
        print(f"  [SAVED] {time_path}")
        print(f"  [METRIC] mean accuracy over time: {row['mean_accuracy_over_time_pct']:.2f}%")

    if summary_rows:
        summary_path = plots_dir / "summary_metrics.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[SAVED] Summary CSV: {summary_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
