"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heart_risk import config

logger = logging.getLogger(__name__)

# Valid code sets taken from the UCI attribute documentation
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def download_raw(dest: Path = config.RAW_FILE, timeout: int = 30) -> Path:
    """Download the Cleveland file from UCI, trying the zip bundle as a fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in config.UCI_URLS:
        try:
            logger.info("Downloading %s", url)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    content = zf.read("processed.cleveland.data")
            else:
                content = resp.content
            dest.write_bytes(content)
            logger.info("Saved %d bytes to %s", len(content), dest)
            return dest
        except Exception as err:  # try the next mirror
            logger.warning("Download from %s failed: %s", url, err)
            last_err = err
    raise RuntimeError(f"All dataset URLs failed: {last_err}")


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw comma-separated file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame.

    * coerce every column to numeric (bad tokens -> NaN)
    * set out-of-domain categorical codes to NaN (imputed later in the pipeline)
    * set physiologically impossible values (bp/chol <= 0) to NaN
    * drop exact duplicate rows and rows with a missing label
    * binarise the label: num 0 -> 0 (no disease), num 1..4 -> 1 (disease)
    """
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col, valid in VALID_CODES.items():
        if col in out.columns:
            out.loc[~out[col].isin(valid) & out[col].notna(), col] = np.nan

    for col in ("trestbps", "chol"):
        if col in out.columns:
            out.loc[out[col] <= 0, col] = np.nan

    before = len(out)
    out = out.drop_duplicates()
    out = out.dropna(subset=["num"])
    if len(out) != before:
        logger.info("Dropped %d duplicate/unlabelled rows", before - len(out))

    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])
    return out.reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Number of missing values per column (only columns that have any)."""
    counts = df.isna().sum()
    return counts[counts > 0]


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.INPUT_FEATURES], df[config.TARGET]
