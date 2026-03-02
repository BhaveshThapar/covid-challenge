"""
Analyze the extracted dataset: class balance, slice counts, source distribution.
"""
import os
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_scan_manifest, _get_sorted_slices


def analyze(data_dir, metadata_dir, split):
    """Analyze a split of the dataset."""
    entries = build_scan_manifest(data_dir, split, metadata_dir)
    if not entries:
        print(f"No entries found for split '{split}'")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset Analysis: {split} split")
    print(f"{'='*60}")

    # Class distribution
    label_counts = Counter(e["label"] for e in entries)
    label_names = {0: "Covid", 1: "Non-Covid"}
    print(f"\n  Class Distribution:")
    for label_id in sorted(label_counts):
        name = label_names.get(label_id, f"Class {label_id}")
        print(f"    {name}: {label_counts[label_id]} scans")
    print(f"    Total: {sum(label_counts.values())} scans")

    # Source distribution
    source_counts = Counter(e["source"] for e in entries)
    print(f"\n  Source Distribution:")
    for src in sorted(source_counts):
        print(f"    Source {src}: {source_counts[src]} scans")

    # Source × Class
    print(f"\n  Source × Class:")
    src_class = Counter((e["source"], e["label"]) for e in entries)
    sources = sorted(set(e["source"] for e in entries))
    labels = sorted(set(e["label"] for e in entries))
    header = f"    {'Source':>8}"
    for l in labels:
        header += f"  {label_names.get(l, f'C{l}'):>10}"
    header += f"  {'Total':>10}"
    print(header)
    for src in sources:
        row = f"    {src:>8}"
        total = 0
        for l in labels:
            c = src_class.get((src, l), 0)
            row += f"  {c:>10}"
            total += c
        row += f"  {total:>10}"
        print(row)

    # Slice statistics (sample first 50 scans for speed)
    print(f"\n  Slice Statistics (sampling up to 50 scans):")
    sample = entries[:50]
    slice_counts = []
    for entry in sample:
        slices = _get_sorted_slices(entry["scan_dir"])
        slice_counts.append(len(slices))

    if slice_counts:
        arr = np.array(slice_counts)
        print(f"    Min slices/scan:  {arr.min()}")
        print(f"    Max slices/scan:  {arr.max()}")
        print(f"    Mean slices/scan: {arr.mean():.1f}")
        print(f"    Std slices/scan:  {arr.std():.1f}")

    # Check for empty scan dirs
    empty = [e["scan_name"] for e in entries
             if not os.listdir(e["scan_dir"])]
    if empty:
        print(f"\n  ⚠ Empty scan directories: {len(empty)}")
        for name in empty[:5]:
            print(f"    - {name}")
    else:
        print(f"\n  ✓ No empty scan directories")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="all",
                        help="Split to analyze: train, val, or all")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        analyze(args.data_dir, args.metadata_dir, split)


if __name__ == "__main__":
    main()
