#!/usr/bin/env python3

from __future__ import annotations

"""
This script takes in directories and arguments: 
- sensor_dir
- aa_dir 
- aa_train (for output)
- aa_test (for output)
- sensor_train (for output)
- sensor_test (for output)
- train fraction (0.0–1.0)
Each of the sensor files have a corresponding mocap file. If you remove "_resamp" from the angle array data and _CapacitanceTable
from the sensor data, the filenames should match. 

The script takes a train fraction and for each of the sensor and aa files, it splits the data into train and test sets 
according to the fraction, and saves them to the specified output directories. Keep the first row for the column names for both training and testing sets
as well. Choose random rows for the training set according to the fraction, and the rest go to the test set. Make sure to shuffle the data before splitting.
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SENSOR_SUFFIX = "_CapacitanceTable"
AA_SUFFIX = "_resamp"


@dataclass(frozen=True)
class Pair:
    base: str
    sensor_path: Path
    aa_path: Path


def stem_without_suffix(p: Path, suffix: str) -> str:
    """
    Return filename stem with a specific trailing suffix removed if present.

    Example:
      "trial1_CapacitanceTable.csv" -> "trial1" (suffix="_CapacitanceTable")
      "trial1_resamp.csv" -> "trial1" (suffix="_resamp")
    """
    stem = p.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def list_csv_files(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def build_index(files: List[Path], suffix_to_strip: str) -> Dict[str, Path]:
    """
    Map base name -> file path.
    If duplicates exist for a base name, raise an error.
    """
    idx: Dict[str, Path] = {}
    for p in files:
        base = stem_without_suffix(p, suffix_to_strip)
        if base in idx:
            raise ValueError(
                f"Duplicate base name after stripping suffix '{suffix_to_strip}': "
                f"'{base}'\n  - {idx[base]}\n  - {p}"
            )
        idx[base] = p
    return idx


def find_pairs(sensor_dir: Path, aa_dir: Path) -> List[Pair]:
    sensor_files = list_csv_files(sensor_dir)
    aa_files = list_csv_files(aa_dir)

    sensor_idx = build_index(sensor_files, SENSOR_SUFFIX)
    aa_idx = build_index(aa_files, AA_SUFFIX)

    common = sorted(set(sensor_idx.keys()) & set(aa_idx.keys()))
    missing_sensor = sorted(set(aa_idx.keys()) - set(sensor_idx.keys()))
    missing_aa = sorted(set(sensor_idx.keys()) - set(aa_idx.keys()))

    if missing_sensor:
        raise FileNotFoundError(
            "No matching sensor file for these angle-array bases:\n  "
            + "\n  ".join(missing_sensor)
        )
    if missing_aa:
        raise FileNotFoundError(
            "No matching angle-array file for these sensor bases:\n  "
            + "\n  ".join(missing_aa)
        )

    return [
        Pair(base=b, sensor_path=sensor_idx[b], aa_path=aa_idx[b])
        for b in common
    ]


def read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Read CSV and return (header_row, data_rows).
    Header is the first row.
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV (no header row): {path}")
        rows = [row for row in reader]
    return header, rows


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def split_rows(
    rows: List[List[str]],
    train_fraction: float,
    rng: random.Random,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Shuffle rows, then split into (train_rows, test_rows).
    Train size = floor(train_fraction * N).
    """
    n = len(rows)
    idxs = list(range(n))
    rng.shuffle(idxs)

    train_n = int(math.floor(train_fraction * n))
    train_idxs = set(idxs[:train_n])

    train_rows: List[List[str]] = []
    test_rows: List[List[str]] = []
    for i, row in enumerate(rows):
        if i in train_idxs:
            train_rows.append(row)
        else:
            test_rows.append(row)

    return train_rows, test_rows


def split_and_save_pair(
    pair: Pair,
    out_sensor_train: Path,
    out_sensor_test: Path,
    out_aa_train: Path,
    out_aa_test: Path,
    train_fraction: float,
    rng: random.Random,
) -> None:
    # Sensor
    s_header, s_rows = read_csv_rows(pair.sensor_path)
    s_train, s_test = split_rows(s_rows, train_fraction, rng)

    # Angle array
    a_header, a_rows = read_csv_rows(pair.aa_path)
    a_train, a_test = split_rows(a_rows, train_fraction, rng)

    # Output filenames match the originals
    write_csv(out_sensor_train / pair.sensor_path.name, s_header, s_train)
    write_csv(out_sensor_test / pair.sensor_path.name, s_header, s_test)
    write_csv(out_aa_train / pair.aa_path.name, a_header, a_train)
    write_csv(out_aa_test / pair.aa_path.name, a_header, a_test)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split paired sensor and angle-array CSV files into train/test sets."
    )
    parser.add_argument("--sensor_dir", type=Path, required=True, help="Input sensor CSV directory")
    parser.add_argument("--aa_dir", type=Path, required=True, help="Input angle-array CSV directory")
    parser.add_argument("--aa_train", type=Path, required=True, help="Output angle-array train directory")
    parser.add_argument("--aa_test", type=Path, required=True, help="Output angle-array test directory")
    parser.add_argument("--sensor_train", type=Path, required=True, help="Output sensor train directory")
    parser.add_argument("--sensor_test", type=Path, required=True, help="Output sensor test directory")
    parser.add_argument(
        "--train_fraction",
        type=float,
        required=True,
        help="Fraction of rows to put in train set (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible splits",
    )

    args = parser.parse_args()

    if not (0.0 <= args.train_fraction <= 1.0):
        raise ValueError(f"--train_fraction must be between 0.0 and 1.0, got {args.train_fraction}")

    for d in [args.sensor_dir, args.aa_dir]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {d}")

    rng = random.Random(args.seed)

    pairs = find_pairs(args.sensor_dir, args.aa_dir)
    if not pairs:
        raise FileNotFoundError("No CSV pairs found. Check your input directories and naming conventions.")

    # Ensure output dirs exist
    args.aa_train.mkdir(parents=True, exist_ok=True)
    args.aa_test.mkdir(parents=True, exist_ok=True)
    args.sensor_train.mkdir(parents=True, exist_ok=True)
    args.sensor_test.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        split_and_save_pair(
            pair=pair,
            out_sensor_train=args.sensor_train,
            out_sensor_test=args.sensor_test,
            out_aa_train=args.aa_train,
            out_aa_test=args.aa_test,
            train_fraction=args.train_fraction,
            rng=rng,
        )

    print(f"Processed {len(pairs)} paired file(s).")
    if args.seed is not None:
        print(f"Seed used: {args.seed}")


if __name__ == "__main__":
    main()
