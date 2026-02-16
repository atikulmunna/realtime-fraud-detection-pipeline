"""Download the PaySim dataset and place CSV under data/raw."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def select_csv(dataset_dir: Path, preferred_name: str | None = None) -> Path:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    if preferred_name:
        wanted = dataset_dir / preferred_name
        if not wanted.exists():
            raise FileNotFoundError(f"Preferred CSV not found: {wanted}")
        return wanted

    if len(csv_files) == 1:
        return csv_files[0]

    # For multiple CSVs, pick the largest file as default.
    return max(csv_files, key=lambda p: p.stat().st_size)


def download_paysim(dataset_ref: str = "ealaxi/paysim1") -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required. Install it with `pip install kagglehub`.") from exc

    return Path(kagglehub.dataset_download(dataset_ref))


def copy_csv_to_raw(csv_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / csv_path.name
    shutil.copy2(csv_path, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PaySim and copy CSV to data/raw.")
    parser.add_argument("--dataset", default="ealaxi/paysim1", help="Kaggle dataset reference.")
    parser.add_argument("--out", default="data/raw", help="Destination directory for CSV.")
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional exact CSV filename to select from downloaded files.",
    )
    args = parser.parse_args()

    dataset_dir = download_paysim(args.dataset)
    csv_file = select_csv(dataset_dir, args.filename)
    copied = copy_csv_to_raw(csv_file, Path(args.out))
    print(f"Downloaded dataset to: {dataset_dir}")
    print(f"Selected CSV: {csv_file}")
    print(f"Copied CSV to: {copied}")


if __name__ == "__main__":
    main()
