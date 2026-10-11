"""Download the CarDekho vehicle dataset from Kaggle into data/raw/.

Kaggle's public dataset download endpoint serves this dataset without
credentials, so no API token is normally needed. If that ever changes the
script prints exact manual instructions instead of failing silently.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucpp import config  # noqa: E402
from ucpp.console import enable_utf8_stdout  # noqa: E402

enable_utf8_stdout()

DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{config.KAGGLE_DATASET_REF}"
)

MANUAL_INSTRUCTIONS = f"""
Automatic download failed. Fetch the file manually - it takes about a minute.

  1. Open {config.KAGGLE_DATASET_URL}
  2. Sign in to Kaggle (a free account is enough) and click "Download".
  3. You will get 'archive.zip' containing four CSV files.
  4. Extract ONLY this file:

         {config.RAW_CSV_NAME}

  5. Place it at exactly this path:

         {config.RAW_CSV_PATH}

  6. Re-run:  python scripts/prepare_data.py

Note: the other three CSVs in that archive are NOT interchangeable. 'car
data.csv' quotes prices in lakhs rather than rupees and includes the current
showroom price, which leaks the answer. This project uses '{config.RAW_CSV_NAME}'
only.
""".strip()


def download(force: bool = False) -> Path:
    """Fetch and extract the one CSV this project uses."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_CSV_PATH.exists() and not force:
        size = config.RAW_CSV_PATH.stat().st_size
        print(f"Already present: {config.RAW_CSV_PATH} ({size:,} bytes)")
        print("Use --force to re-download.")
        return config.RAW_CSV_PATH

    print(f"Downloading {config.KAGGLE_DATASET_REF} ...")
    request = urllib.request.Request(
        DOWNLOAD_URL, headers={"User-Agent": "used-car-price-predictor/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"ERROR: download failed ({exc})\n", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Received {len(payload):,} bytes. Extracting ...")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        print("ERROR: response was not a zip archive.\n", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        raise SystemExit(1)

    names = archive.namelist()
    if config.RAW_CSV_NAME not in names:
        print(
            f"ERROR: '{config.RAW_CSV_NAME}' not in archive. Found: {names}\n",
            file=sys.stderr,
        )
        print(MANUAL_INSTRUCTIONS, file=sys.stderr)
        raise SystemExit(1)

    # Extract by explicit name rather than extractall(): never trust paths
    # inside a downloaded archive.
    with archive.open(config.RAW_CSV_NAME) as src:
        config.RAW_CSV_PATH.write_bytes(src.read())

    print(f"Saved {config.RAW_CSV_PATH} ({config.RAW_CSV_PATH.stat().st_size:,} bytes)")
    print(f"Other files in the archive (not used): "
          f"{[n for n in names if n != config.RAW_CSV_NAME]}")
    print(f"Licence: {config.DATASET_LICENCE}")
    return config.RAW_CSV_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file exists"
    )
    args = parser.parse_args()
    download(force=args.force)
    print("\nNext:  python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
