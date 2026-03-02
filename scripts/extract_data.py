"""
Extract dataset archives and organize into the standard directory structure.

Expected output:
  data/
  ├── train/
  │   ├── covid/        (from covid1.rar, covid2.rar)
  │   └── non_covid/    (from non-covid1.rar, non-covid2.rar, non-covid3.rar)
  ├── val/
  │   ├── covid/        (from Validation.zip)
  │   └── non_covid/    (from Validation.zip)
  └── metadata/
      ├── val_covid.csv
      └── val_non_covid.csv
"""
import os
import sys
import shutil
import zipfile
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_zip(zip_path, dest_dir):
    """Extract a ZIP archive."""
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        for member in tqdm(members, desc="Unzipping"):
            zf.extract(member, dest_dir)


def extract_rar(rar_path, dest_dir):
    """Extract a RAR archive using system unrar or python rarfile."""
    print(f"Extracting {rar_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    # Try system unrar first
    try:
        result = subprocess.run(
            ["unrar", "x", "-o+", rar_path, dest_dir],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print(f"  Extracted with unrar")
            return
        else:
            print(f"  unrar failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  unrar not found, trying Python rarfile...")

    # Try p7zip
    try:
        result = subprocess.run(
            ["7z", "x", f"-o{dest_dir}", "-y", rar_path],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print(f"  Extracted with 7z")
            return
        else:
            print(f"  7z failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  7z not found either")

    # Try Python unrar-cffi (pure Python, no external binary needed)
    try:
        from unrar.cffi import rarfile as unrar_cffi
        rf = unrar_cffi.RarFile(rar_path)
        rf.extractall(dest_dir)
        print(f"  Extracted with unrar-cffi")
        return
    except Exception as e:
        print(f"  unrar-cffi failed: {e}")

    # Try Python rarfile (needs unrar binary)
    try:
        import rarfile
        rf = rarfile.RarFile(rar_path)
        rf.extractall(dest_dir)
        print(f"  Extracted with Python rarfile")
        return
    except Exception as e:
        print(f"  Python rarfile failed: {e}")

    raise RuntimeError(f"Could not extract {rar_path}. Install unrar, 7z, or pip install unrar-cffi")


def organize_validation(extract_dir, data_dir, datasets_dir):
    """Move validation data from extracted zip structure to our standard layout."""
    val_src = os.path.join(extract_dir, "val")
    if not os.path.exists(val_src):
        # Try looking one level deeper
        for d in Path(extract_dir).rglob("val"):
            if d.is_dir():
                val_src = str(d)
                break

    val_covid_dst = os.path.join(data_dir, "val", "covid")
    val_noncovid_dst = os.path.join(data_dir, "val", "non_covid")
    os.makedirs(val_covid_dst, exist_ok=True)
    os.makedirs(val_noncovid_dst, exist_ok=True)

    # Move covid scans
    covid_src = os.path.join(val_src, "covid")
    if os.path.exists(covid_src):
        for scan_name in sorted(os.listdir(covid_src)):
            src = os.path.join(covid_src, scan_name)
            dst = os.path.join(val_covid_dst, scan_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        print(f"  Moved covid validation scans to {val_covid_dst}")

    # Move non-covid scans
    noncovid_src = os.path.join(val_src, "non-covid")
    if os.path.exists(noncovid_src):
        for scan_name in sorted(os.listdir(noncovid_src)):
            src = os.path.join(noncovid_src, scan_name)
            dst = os.path.join(val_noncovid_dst, scan_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        print(f"  Moved non-covid validation scans to {val_noncovid_dst}")

    # Copy metadata CSVs
    metadata_dir = os.path.join(data_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    for csv_name, dst_name in [("validation_covid.csv", "val_covid.csv"),
                                ("validation_non_covid.csv", "val_non_covid.csv")]:
        src = os.path.join(datasets_dir, csv_name)
        dst = os.path.join(metadata_dir, dst_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {csv_name} -> {dst}")


def organize_training_rar(extract_dir, data_dir, label):
    """
    Move extracted training scans to the standard layout.
    label: 'covid' or 'non_covid'
    """
    dst_dir = os.path.join(data_dir, "train", label)
    os.makedirs(dst_dir, exist_ok=True)

    # Find scan directories (ct_scan_*) in the extracted content
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d.startswith("ct_scan_"):
                src = os.path.join(root, d)
                dst = os.path.join(dst_dir, d)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

    count = len([d for d in os.listdir(dst_dir) if d.startswith("ct_scan_")])
    print(f"  {label} training scans: {count}")


def main():
    parser = argparse.ArgumentParser(description="Extract and organize dataset")
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--temp-dir", type=str, default="data/_temp_extract")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 1. Extract validation
    if not args.skip_validation:
        print("\n=== Extracting Validation Data ===")
        extract_zip(
            os.path.join(args.datasets_dir, "Validation.zip"),
            args.temp_dir,
        )
        organize_validation(args.temp_dir, args.data_dir, args.datasets_dir)
        # Clean temp
        shutil.rmtree(args.temp_dir, ignore_errors=True)
        os.makedirs(args.temp_dir, exist_ok=True)

    # 2. Extract training covid
    if not args.skip_training:
        print("\n=== Extracting Training Covid Data ===")
        for rar_name in ["covid1.rar", "covid2.rar"]:
            rar_path = os.path.join(args.datasets_dir, rar_name)
            if os.path.exists(rar_path):
                extract_rar(rar_path, args.temp_dir)
                organize_training_rar(args.temp_dir, args.data_dir, "covid")
                shutil.rmtree(args.temp_dir, ignore_errors=True)
                os.makedirs(args.temp_dir, exist_ok=True)

        print("\n=== Extracting Training Non-Covid Data ===")
        for rar_name in ["non-covid1.rar", "non-covid2.rar", "non-covid3.rar"]:
            rar_path = os.path.join(args.datasets_dir, rar_name)
            if os.path.exists(rar_path):
                extract_rar(rar_path, args.temp_dir)
                organize_training_rar(args.temp_dir, args.data_dir, "non_covid")
                shutil.rmtree(args.temp_dir, ignore_errors=True)
                os.makedirs(args.temp_dir, exist_ok=True)

    # Cleanup
    shutil.rmtree(args.temp_dir, ignore_errors=True)

    # Summary
    print("\n=== Extraction Complete ===")
    for split in ["train", "val"]:
        for label in ["covid", "non_covid"]:
            d = os.path.join(args.data_dir, split, label)
            if os.path.exists(d):
                n = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])
                print(f"  {split}/{label}: {n} scans")


if __name__ == "__main__":
    main()
