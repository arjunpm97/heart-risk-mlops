"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:  python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_risk import config  # noqa: E402
from heart_risk.data import clean, download_raw, load_raw, missing_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.force or not config.RAW_FILE.exists():
        download_raw()
    else:
        logging.info("Raw file already present: %s", config.RAW_FILE)

    raw = load_raw()
    logging.info("Raw shape: %s; missing: %s", raw.shape, missing_report(raw).to_dict())
    df = clean(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    logging.info("Clean shape: %s; target balance: %s -> %s", df.shape,
                 df[config.TARGET].value_counts().to_dict(), config.CLEAN_FILE)


if __name__ == "__main__":
    main()
