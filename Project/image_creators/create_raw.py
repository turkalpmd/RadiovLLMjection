#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Copy raw tumor images from Project/Images/Raw/ into
Project/Images/Organized_images/ with proper naming ({base_id}.jpg).
Uses dataset.csv to know which images to include.
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, ORGANIZED_DIR
from utils import load_dataset


def create_raw_images():
    """Copy raw images from Raw/ into Organized_images/ with base_id naming."""
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        print(f"WARNING: Raw directory not found: {RAW_DIR}")
        print("Please place your raw 512x512 images in this folder first.")
        return ORGANIZED_DIR

    df = load_dataset()

    total_copied = 0
    total_skipped = 0

    print("=" * 60)
    print(f"Source : {RAW_DIR}")
    print(f"Target : {ORGANIZED_DIR}")
    print(f"Images : {len(df)}")
    print("=" * 60)

    raw_files = {f.stem.lower(): f for f in RAW_DIR.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")}

    for _, row in df.iterrows():
        base_id = row["base_id"]
        out_name = f"{base_id}.jpg"
        out_path = ORGANIZED_DIR / out_name

        if out_path.exists():
            total_skipped += 1
            continue

        # Try to find matching file in Raw/ by multiple naming patterns
        src = None
        search_patterns = [
            base_id.lower(),
            f"512_raw_{base_id}".lower(),
            base_id.split("_")[-1],
        ]
        for pattern in search_patterns:
            if pattern in raw_files:
                src = raw_files[pattern]
                break

        if src is None:
            for fname, fpath in raw_files.items():
                if base_id.lower() in fname or base_id.split("_")[-1] in fname:
                    src = fpath
                    break

        if src is None:
            print(f"  NOT FOUND: {base_id}")
            continue

        try:
            shutil.copy2(src, out_path)
            total_copied += 1
            print(f"  Copied: {src.name} -> {out_name}")
        except Exception as e:
            print(f"  ERROR: {src.name} - {e}")

    print(f"\nCopied: {total_copied}, Skipped: {total_skipped}")
    print(f"Target folder: {ORGANIZED_DIR}")
    return ORGANIZED_DIR


if __name__ == "__main__":
    create_raw_images()
