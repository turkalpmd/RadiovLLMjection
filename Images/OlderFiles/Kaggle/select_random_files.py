import argparse
import os
import random
import shutil
from pathlib import Path


def collect_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file()]
    return files


def choose_random(files: list[Path], count: int) -> list[Path]:
    if count >= len(files):
        return files
    return random.sample(files, count)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files: list[Path], destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for src in files:
        dst = destination_dir / src.name
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly copy files from class folders into SelectedFiles with the same folder names."
        )
    )

    parser.add_argument(
        "--base-raw",
        default=(
            "/home/ubuntu/RadiovLLMjection/Images/Kaggle/PMRAM Bangladeshi Brain Cancer - MRI Dataset/"
            "PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw"
        ),
        help="Path to the Raw directory containing 512Glioma, 512Meningioma, 512Normal, 512Pituitary",
    )
    parser.add_argument(
        "--dest",
        default="/home/ubuntu/RadiovLLMjection/Images/Kaggle/SelectedFiles",
        help="Destination root where SelectedFiles/<class> will be created",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="If provided, use this count for all classes (overridden by class-specific args)",
    )
    parser.add_argument("--glioma", type=int, default=None, help="Count for 512Glioma")
    parser.add_argument("--meningioma", type=int, default=None, help="Count for 512Meningioma")
    parser.add_argument("--normal", type=int, default=None, help="Count for 512Normal")
    parser.add_argument("--pituarrity", type=int, default=None, help="Count for 512Pituitary")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    base_raw = Path(args.base_raw)
    dest_root = Path(args.dest)

    classes = {
        "512Glioma": args.glioma,
        "512Meningioma": args.meningioma,
        "512Normal": args.normal,
        "512Pituitary": args.pituarrity,
    }

    # Defaults per your request: Normal=300, others=100, unless overridden
    default_all = args.count
    for cls in classes:
        if classes[cls] is None:
            if default_all is not None:
                classes[cls] = default_all
            else:
                classes[cls] = 300 if cls == "512Normal" else 100

    for cls, count in classes.items():
        src_dir = base_raw / cls
        dst_dir = dest_root / cls

        files = collect_files(src_dir)
        if not files:
            print(f"[WARN] No files found in {src_dir}")
            continue

        chosen = choose_random(files, count)
        if len(chosen) < count:
            print(
                f"[INFO] Requested {count} from {cls} but only {len(chosen)} available; copying all."
            )

        copy_files(chosen, dst_dir)
        print(f"[OK] Copied {len(chosen)} files to {dst_dir}")


if __name__ == "__main__":
    main()


