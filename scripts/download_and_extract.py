"""
Download dataset archives from Google Drive and organize into the standard directory structure.

Fill in the GOOGLE_DRIVE_IDS dict below with your actual file IDs before running.
A Google Drive file ID looks like: 1A2B3C4D5E6F7G8H9I0J (from the sharing URL).

Expected output:
  data/
  ├── train/
  │   ├── covid/        (from covid1.rar, covid2.rar)
  │   └── non_covid/    (from non-covid1.rar, non-covid2.rar, non-covid3.rar)
  ├── val/
  │   ├── covid/        (from Validation.zip)
  │   └── non_covid/    (from Validation.zip)
  └── metadata/
      ├── train_covid.csv
      ├── train_non_covid.csv
      ├── val_covid.csv
      └── val_non_covid.csv
"""

import os
import sys
import shutil
import zipfile
import subprocess
import argparse
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FILL IN YOUR GOOGLE DRIVE FILE IDs HERE
# Get the ID from the sharing link:
#   https://drive.google.com/file/d/<FILE_ID>/view
# ---------------------------------------------------------------------------
GOOGLE_DRIVE_IDS = {
    # Training archives
    "covid1.rar":           "1g26d6-QoPWpG-SIjkqwKCKRekgATeFCE",
    "covid2.rar":           "1JkbqK9bxKyuIhj-ZB9joRkYo-CuJCCAG",
    "non-covid1.rar":       "1e7Kv2VC0xMTtgiafyst7S-bPQ2lZ8fp_",
    "non-covid2.rar":       "1ab6fzG5_96SLjA6ETwi_F9z6z8FgPiHe",
    "non-covid3.rar":       "1gLbkKDmW5YGz73f23zf8iYMQ2iIiwmcd",
    # Validation archive
    "Validation.zip":       "1PqQB41L-7rcFRfdpes5iaF6IWVMxX-pN",
    # CSV metadata files
    "train_covid.csv":      "11zjdJztL8DATNsO21JAX9QafxKeYUvbP",
    "train_non_covid.csv":  "1nrjof8Qu55WEazgGtx2oSM_cCYO1J-Tx",
    "validation_covid.csv": "155W8e4h0t1odKspjxCiK6Cl4-81PK2zH",
    "validation_non_covid.csv": "1dhVi_0Sldyj4hrBpfRGkLV4PObyzDsj5",
    # Test set (1st challenge, unlabeled)
    "1st_challenge_test_set.zip": "1mcHL63ILDYh7IQFnMtW_0p0grmb5vhc6",
}
# ---------------------------------------------------------------------------


def download_file(file_id: str, dest_path: str):
    """Download a single file from Google Drive using gdown."""
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {dest_path}")
        return
    print(f"  Downloading -> {dest_path}")
    import gdown
    gdown.download(id=file_id, output=dest_path, quiet=False)


def download_all(datasets_dir: str, metadata_dir: str):
    """Download all archives and CSVs from Google Drive."""
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    archives = ["covid1.rar", "covid2.rar", "non-covid1.rar",
                "non-covid2.rar", "non-covid3.rar", "Validation.zip",
                "1st_challenge_test_set.zip"]
    csvs = ["train_covid.csv", "train_non_covid.csv",
            "validation_covid.csv", "validation_non_covid.csv"]

    print("\n=== Downloading archives ===")
    for fname in archives:
        file_id = GOOGLE_DRIVE_IDS[fname]
        if file_id.startswith("PLACEHOLDER"):
            print(f"  WARNING: No ID set for {fname}, skipping")
            continue
        download_file(file_id, os.path.join(datasets_dir, fname))

    print("\n=== Downloading CSVs ===")
    for fname in csvs:
        file_id = GOOGLE_DRIVE_IDS[fname]
        if file_id.startswith("PLACEHOLDER"):
            print(f"  WARNING: No ID set for {fname}, skipping")
            continue
        download_file(file_id, os.path.join(datasets_dir, fname))


# ---------------------------------------------------------------------------
# Extraction helpers (same logic as original extract_data.py)
# ---------------------------------------------------------------------------

def extract_zip(zip_path: str, dest_dir: str):
    """Extract a ZIP archive."""
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        for member in tqdm(members, desc="Unzipping"):
            zf.extract(member, dest_dir)


def extract_rar(rar_path: str, dest_dir: str):
    """Extract a RAR archive — tries system unrar then 7z."""
    print(f"Extracting {rar_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    # Try system unrar (loaded via module on Nexus)
    try:
        result = subprocess.run(
            ["unrar", "x", "-o+", rar_path, dest_dir],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print("  Extracted with unrar")
            return
        print(f"  unrar failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  unrar not found, trying 7z...")

    # Try p7zip
    try:
        result = subprocess.run(
            ["7z", "x", f"-o{dest_dir}", "-y", rar_path],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print("  Extracted with 7z")
            return
        print(f"  7z failed: {result.stderr[:200]}")
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Could not extract {rar_path}. "
        "Make sure the unrar module is loaded: `module load unrar/7.0.9`"
    )


def organize_training_rar(extract_dir: str, data_dir: str, label: str):
    """Move ct_scan_* folders to data/train/<label>/."""
    dst_dir = os.path.join(data_dir, "train", label)
    os.makedirs(dst_dir, exist_ok=True)

    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            if d.startswith("ct_scan_"):
                src = os.path.join(root, d)
                dst = os.path.join(dst_dir, d)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

    count = len([d for d in os.listdir(dst_dir) if d.startswith("ct_scan_")])
    print(f"  {label} training scans so far: {count}")


def organize_validation(extract_dir: str, data_dir: str):
    """Move validation scans from extracted zip into data/val/."""
    # The zip extracts to a folder containing covid/ and non-covid/ subdirs
    val_src = None
    for d in Path(extract_dir).rglob("covid"):
        if d.is_dir() and d.parent.name not in ("train",):
            val_src = str(d.parent)
            break
    if val_src is None:
        val_src = extract_dir

    val_covid_dst = os.path.join(data_dir, "val", "covid")
    val_noncovid_dst = os.path.join(data_dir, "val", "non_covid")
    os.makedirs(val_covid_dst, exist_ok=True)
    os.makedirs(val_noncovid_dst, exist_ok=True)

    covid_src = os.path.join(val_src, "covid")
    if os.path.exists(covid_src):
        for scan_name in sorted(os.listdir(covid_src)):
            src = os.path.join(covid_src, scan_name)
            dst = os.path.join(val_covid_dst, scan_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        print(f"  Moved covid val scans to {val_covid_dst}")

    noncovid_src = os.path.join(val_src, "non-covid")
    if os.path.exists(noncovid_src):
        for scan_name in sorted(os.listdir(noncovid_src)):
            src = os.path.join(noncovid_src, scan_name)
            dst = os.path.join(val_noncovid_dst, scan_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        print(f"  Moved non-covid val scans to {val_noncovid_dst}")


def organize_test(extract_dir: str, data_dir: str):
    """Move test scan folders (ct_scan_*) into data/test/."""
    test_dst = os.path.join(data_dir, "test")
    os.makedirs(test_dst, exist_ok=True)

    # Search for ct_scan_* dirs (flat or nested)
    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            if d.startswith("ct_scan_"):
                src = os.path.join(root, d)
                dst = os.path.join(test_dst, d)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.move(src, dst)

    count = len([x for x in os.listdir(test_dst) if os.path.isdir(os.path.join(test_dst, x))])
    print(f"  Test scans: {count}")


def copy_csvs(datasets_dir: str, metadata_dir: str):
    """Copy downloaded CSVs into data/metadata/, renaming validation ones."""
    os.makedirs(metadata_dir, exist_ok=True)
    rename_map = {
        "train_covid.csv":          "train_covid.csv",
        "train_non_covid.csv":      "train_non_covid.csv",
        "validation_covid.csv":     "val_covid.csv",
        "validation_non_covid.csv": "val_non_covid.csv",
    }
    for src_name, dst_name in rename_map.items():
        src = os.path.join(datasets_dir, src_name)
        dst = os.path.join(metadata_dir, dst_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {src_name} -> {dst_name}")


# ---------------------------------------------------------------------------
# Dataset analysis (port of the deleted scripts/analyze_data.py)
# Run with: python scripts/download_and_extract.py --skip-download --analyze
# ---------------------------------------------------------------------------

def analyze_dataset(data_dir: str, metadata_dir: str, split: str = "all"):
    """
    Print class balance, source distribution, source×class breakdown,
    and slice statistics for the extracted dataset.

    Mirrors the functionality of the deleted scripts/analyze_data.py.
    Uses build_scan_manifest and _get_sorted_slices from src/dataset.py.
    """
    try:
        import numpy as np
    except ImportError:
        print("numpy not installed — run `pip install numpy` to use analysis")
        return

    # Make src/ importable when running this script directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.dataset import build_scan_manifest, _get_sorted_slices

    splits = ["train", "val"] if split == "all" else [split]
    label_names = {0: "Covid", 1: "Non-Covid"}

    for sp in splits:
        entries = build_scan_manifest(data_dir, sp, metadata_dir)
        if not entries:
            print(f"No entries found for split '{sp}'")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset Analysis: {sp} split")
        print(f"{'='*60}")

        # Class distribution
        label_counts = Counter(e["label"] for e in entries)
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

        # Source × Class breakdown
        src_class = Counter((e["source"], e["label"]) for e in entries)
        sources = sorted(set(e["source"] for e in entries))
        labels = sorted(set(e["label"] for e in entries))
        print(f"\n  Source x Class:")
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
        slice_counts = [len(_get_sorted_slices(e["scan_dir"])) for e in sample]
        if slice_counts:
            arr = np.array(slice_counts)
            print(f"    Min slices/scan:  {arr.min()}")
            print(f"    Max slices/scan:  {arr.max()}")
            print(f"    Mean slices/scan: {arr.mean():.1f}")
            print(f"    Std slices/scan:  {arr.std():.1f}")

        # Check for empty scan directories
        empty = [e["scan_name"] for e in entries if not os.listdir(e["scan_dir"])]
        if empty:
            print(f"\n  WARNING: {len(empty)} empty scan directories:")
            for name in empty[:5]:
                print(f"    - {name}")
        else:
            print(f"\n  OK: No empty scan directories")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and extract Covid-19 dataset from Google Drive")
    parser.add_argument("--datasets-dir", default="datasets",
                        help="Where to save downloaded archives (default: datasets/)")
    parser.add_argument("--data-dir", default="data",
                        help="Output directory for organized data (default: data/)")
    parser.add_argument("--temp-dir", default="data/_temp_extract",
                        help="Scratch space for extraction (default: data/_temp_extract)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading (assume archives already in --datasets-dir)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run dataset analysis after extraction (class/source/slice stats)")
    parser.add_argument("--analyze-split", default="all", choices=["train", "val", "all"],
                        help="Which split to analyze (default: all)")
    args = parser.parse_args()

    metadata_dir = os.path.join(args.data_dir, "metadata")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 1. Download
    if not args.skip_download:
        download_all(args.datasets_dir, args.datasets_dir)

    # 2. Copy CSVs to metadata/
    print("\n=== Copying metadata CSVs ===")
    copy_csvs(args.datasets_dir, metadata_dir)

    # 3. Extract Validation.zip
    val_zip = os.path.join(args.datasets_dir, "Validation.zip")
    if os.path.exists(val_zip):
        print("\n=== Extracting Validation ===")
        extract_zip(val_zip, args.temp_dir)
        organize_validation(args.temp_dir, args.data_dir)
        shutil.rmtree(args.temp_dir, ignore_errors=True)
        os.makedirs(args.temp_dir, exist_ok=True)

    # 4. Extract training covid RARs
    print("\n=== Extracting Training Covid ===")
    for rar_name in ["covid1.rar", "covid2.rar"]:
        rar_path = os.path.join(args.datasets_dir, rar_name)
        if os.path.exists(rar_path):
            extract_rar(rar_path, args.temp_dir)
            organize_training_rar(args.temp_dir, args.data_dir, "covid")
            shutil.rmtree(args.temp_dir, ignore_errors=True)
            os.makedirs(args.temp_dir, exist_ok=True)

    # 5. Extract training non-covid RARs
    print("\n=== Extracting Training Non-Covid ===")
    for rar_name in ["non-covid1.rar", "non-covid2.rar", "non-covid3.rar"]:
        rar_path = os.path.join(args.datasets_dir, rar_name)
        if os.path.exists(rar_path):
            extract_rar(rar_path, args.temp_dir)
            organize_training_rar(args.temp_dir, args.data_dir, "non_covid")
            shutil.rmtree(args.temp_dir, ignore_errors=True)
            os.makedirs(args.temp_dir, exist_ok=True)

    # 6. Extract test set
    test_zip = os.path.join(args.datasets_dir, "1st_challenge_test_set.zip")
    if os.path.exists(test_zip):
        print("\n=== Extracting Test Set ===")
        extract_zip(test_zip, args.temp_dir)
        organize_test(args.temp_dir, args.data_dir)
        shutil.rmtree(args.temp_dir, ignore_errors=True)
        os.makedirs(args.temp_dir, exist_ok=True)

    # Cleanup
    shutil.rmtree(args.temp_dir, ignore_errors=True)

    # Summary
    print("\n=== Done ===")
    for split in ["train", "val"]:
        for label in ["covid", "non_covid"]:
            d = os.path.join(args.data_dir, split, label)
            if os.path.exists(d):
                n = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])
                print(f"  {split}/{label}: {n} scans")
    test_d = os.path.join(args.data_dir, "test")
    if os.path.exists(test_d):
        n = len([x for x in os.listdir(test_d) if os.path.isdir(os.path.join(test_d, x))])
        print(f"  test: {n} scans")
    print(f"\nMetadata files in {metadata_dir}:")
    if os.path.exists(metadata_dir):
        for f in sorted(os.listdir(metadata_dir)):
            print(f"  {f}")

    if args.analyze:
        analyze_dataset(args.data_dir, metadata_dir, args.analyze_split)


if __name__ == "__main__":
    main()