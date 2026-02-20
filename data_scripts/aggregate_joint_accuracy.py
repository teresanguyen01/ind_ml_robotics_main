#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


# Must match what motion_eval.py writes
DEFAULT_THRESHOLDS = [5.0, 10.0]


def load_all_metrics_per_joint(base_out: Path) -> pd.DataFrame:
    """
    Load metrics_per_joint.csv from each subfolder under base_out.
    Adds a 'sequence' column taken from the subfolder name.
    """
    rows = []
    for sub in sorted(base_out.iterdir()):
        if not sub.is_dir():
            continue
        f = sub / "metrics_per_joint.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["sequence"] = sub.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compute_overall_joint_accuracy(all_df: pd.DataFrame, thresholds) -> pd.DataFrame:
    """
    Overall % accuracy per joint, aggregated across sequences.
    Accuracy is defined as the fraction of frames within a threshold,
    using the per-sequence 'acc_within_X' values already computed by motion_eval.py.

    Aggregation strategy: mean across sequences (each sequence contributes equally).
    """
    if all_df.empty:
        return pd.DataFrame()

    acc_cols = [f"acc_within_{t:g}" for t in thresholds]
    keep_cols = ["joint"] + [c for c in acc_cols if c in all_df.columns]

    if len(keep_cols) == 1:
        raise ValueError(
            "No acc_within_* columns found. "
            "Make sure motion_eval.py is writing acc_within_5 and acc_within_10."
        )

    tmp = all_df[keep_cols].copy()

    overall = (
        tmp.groupby("joint", as_index=False)
           .agg({c: "mean" for c in keep_cols if c != "joint"})
           .sort_values("joint")
           .reset_index(drop=True)
    )

    # Convert to percent
    for c in overall.columns:
        if c.startswith("acc_within_"):
            overall[c] = overall[c] * 100.0

    return overall


def main():
    p = argparse.ArgumentParser(description="Aggregate per-joint % accuracy across all sequences in an output directory.")
    p.add_argument("--output_dir", required=True, help="Base output dir (e.g., pred_vs_ground_0210)")
    p.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS,
                   help="Threshold(s) to aggregate, must match acc_within_* columns (default: 5 10)")
    args = p.parse_args()

    base = Path(args.output_dir)
    if not base.exists():
        raise FileNotFoundError(f"output_dir does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {base}")

    all_df = load_all_metrics_per_joint(base)
    if all_df.empty:
        raise RuntimeError(
            f"No metrics_per_joint.csv files found under: {base}\n"
            "Expected structure: base_out/<sequence_name>/metrics_per_joint.csv"
        )

    # Save concatenated per-file-per-joint table (useful for debugging and analysis)
    all_path = base / "OVERALL__all_sequences_metrics_per_joint.csv"
    all_df.to_csv(all_path, index=False)

    overall_acc = compute_overall_joint_accuracy(all_df, thresholds=args.thresholds)
    out_path = base / "OVERALL__joint_accuracy_percent.csv"
    overall_acc.to_csv(out_path, index=False)

    print(f"Loaded sequences: {all_df['sequence'].nunique()}")
    print(f"Saved: {all_path}")
    print(f"Saved: {out_path}\n")
    print(overall_acc.to_string(index=False))


if __name__ == "__main__":
    main()