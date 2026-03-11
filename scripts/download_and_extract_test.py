"""
Download the Test dataset archive from Google Drive and organize into data/test/.

Downloads Test.zip to datasets/, then:
  - Moves CSV metadata files to datasets/
  - Moves RAR archives to datasets/ and extracts them
  - Moves ct_scan_* folders to data/test/covid/ or data/test/non_covid/
    based on which subfolder they came from

Expected output:
  data/
  └── test/
      ├── covid/        (ct_scan_* dirs)
      └── non_covid/    (ct_scan_* dirs)
  datasets/
      ├── Test.zip
      ├── test_covid.csv       (if present in zip)
      ├── test_non_covid.csv   (if present in zip)
      └── *.rar                (any RAR archives found in zip)
"""

import os
import sys
import shutil
import zipfile
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

TEST_ZIP_ID = "1mcHL63ILDYh7IQFnMtW_0p0grmb5vhc6"
TEST_ZIP_NAME = "Test.zip"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(file_id: str, dest_path: str):
    """Download a single file from Google Drive using gdown."""
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {dest_path}")
        return
    print(f"  Downloading -> {dest_path}")
    import gdown
    gdown.download(id=file_id, output=dest_path, quiet=False)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_zip(zip_path: str, dest_dir: str):
    """Extract a ZIP archive, skipping __MACOSX artifacts."""
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        for member in tqdm(members, desc="Unzipping"):
            zf.extract(member, dest_dir)


def extract_rar(rar_path: str, dest_dir: str):
    """Extract a RAR archive — tries system unrar then 7z."""
    print(f"Extracting {rar_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

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
        "Make sure unrar is loaded: `module load unrar/7.0.9`"
    )


# ---------------------------------------------------------------------------
# Organization
# ---------------------------------------------------------------------------

def move_scan_dirs(src_dir: str, dst_dir: str, label: str):
    """Move all ct_scan_* folders from src_dir into dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for item in sorted(os.listdir(src_dir)):
        if item.startswith("ct_scan_"):
            src = os.path.join(src_dir, item)
            dst = os.path.join(dst_dir, item)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
    print(f"  Moved {moved} {label} scans -> {dst_dir}")
    return moved


def find_label_dir(extract_dir: str, label: str) -> str | None:
    """
    Find the first directory named `label` (e.g. 'covid' or 'non_covid')
    in the extracted tree, excluding data/ and datasets/.
    """
    for root, dirs, _ in os.walk(extract_dir):
        # Skip already-organised destination dirs
        norm = os.path.normpath(root)
        if "data/test" in norm or "datasets" in norm:
            continue
        for d in dirs:
            if d.lower() in (label.lower(), label.lower().replace("_", "-")):
                return os.path.join(root, d)
    return None


def organize_test_zip(extract_dir: str, data_dir: str, datasets_dir: str, temp_dir: str):
    """
    Walk the extracted zip tree:
      - CSV files → copy to datasets/
      - RAR files → copy to datasets/ then extract + move ct_scan_* to test/<label>/
      - covid/ and non_covid/ (or non-covid/) dirs → move ct_scan_* to test/<label>/
    """
    test_covid_dst = os.path.join(data_dir, "test", "covid")
    test_noncovid_dst = os.path.join(data_dir, "test", "non_covid")
    os.makedirs(test_covid_dst, exist_ok=True)
    os.makedirs(test_noncovid_dst, exist_ok=True)

    # --- Move CSVs to datasets/ ---
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".csv"):
                src = os.path.join(root, fname)
                dst = os.path.join(datasets_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  CSV: {fname} -> datasets/")

    # --- Move RARs to datasets/ and extract them ---
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".rar"):
                src = os.path.join(root, fname)
                dst_rar = os.path.join(datasets_dir, fname)
                if not os.path.exists(dst_rar):
                    shutil.copy2(src, dst_rar)
                    print(f"  RAR: {fname} -> datasets/")

                # Determine label from RAR filename
                fname_lower = fname.lower()
                if "non" in fname_lower:
                    label = "non_covid"
                    label_dst = test_noncovid_dst
                else:
                    label = "covid"
                    label_dst = test_covid_dst

                rar_temp = os.path.join(temp_dir, f"rar_{fname}")
                os.makedirs(rar_temp, exist_ok=True)
                extract_rar(dst_rar, rar_temp)
                move_scan_dirs(rar_temp, label_dst, label)
                # Also walk subdirs in case RAR extracts to a subfolder
                for sub_root, sub_dirs, _ in os.walk(rar_temp):
                    for sub_d in sub_dirs:
                        if sub_d.startswith("ct_scan_"):
                            src_scan = os.path.join(sub_root, sub_d)
                            dst_scan = os.path.join(label_dst, sub_d)
                            if not os.path.exists(dst_scan):
                                shutil.move(src_scan, dst_scan)
                shutil.rmtree(rar_temp, ignore_errors=True)

    # --- Find and move covid/ subdir ---
    for label, dst in [("covid", test_covid_dst), ("non_covid", test_noncovid_dst),
                        ("non-covid", test_noncovid_dst)]:
        label_dir = find_label_dir(extract_dir, label)
        if label_dir:
            dst_label = "non_covid" if "non" in label else "covid"
            move_scan_dirs(label_dir, dst, dst_label)

    # Summary
    n_covid = len([d for d in os.listdir(test_covid_dst)
                   if os.path.isdir(os.path.join(test_covid_dst, d))])
    n_noncovid = len([d for d in os.listdir(test_noncovid_dst)
                      if os.path.isdir(os.path.join(test_noncovid_dst, d))])
    print(f"\n  test/covid:     {n_covid} scans")
    print(f"  test/non_covid: {n_noncovid} scans")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the Test dataset from Google Drive"
    )
    parser.add_argument("--datasets-dir", default="datasets",
                        help="Where to save zip/rar/csv files (default: datasets/)")
    parser.add_argument("--data-dir", default="data",
                        help="Output directory for organized data (default: data/)")
    parser.add_argument("--temp-dir", default="data/_temp_test",
                        help="Scratch space for extraction (default: data/_temp_test)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading (assume Test.zip already in --datasets-dir)")
    args = parser.parse_args()

    os.makedirs(args.datasets_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    zip_path = os.path.join(args.datasets_dir, TEST_ZIP_NAME)

    # 1. Download
    if not args.skip_download:
        print("\n=== Downloading Test.zip ===")
        download_file(TEST_ZIP_ID, zip_path)
    else:
        if not os.path.exists(zip_path):
            print(f"ERROR: {zip_path} not found and --skip-download was set.")
            sys.exit(1)

    # 2. Extract zip to temp
    print("\n=== Extracting Test.zip ===")
    extract_zip(zip_path, args.temp_dir)

    # 3. Organize: move CSVs, extract RARs, move ct_scan_* dirs
    print("\n=== Organizing test data ===")
    organize_test_zip(args.temp_dir, args.data_dir, args.datasets_dir, args.temp_dir)

    # 4. Cleanup temp
    shutil.rmtree(args.temp_dir, ignore_errors=True)

    print("\n=== Done ===")
    print(f"  Archives/metadata in: {args.datasets_dir}/")
    print(f"  Scans organized in:   {args.data_dir}/test/")


if __name__ == "__main__":
    main()
