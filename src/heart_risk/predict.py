"""Inference helpers shared by the CLI, the notebook and the API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from heart_risk import config

LABELS = {0: "no_heart_disease", 1: "heart_disease"}


def load_model(path: Path | str = config.MODEL_FILE):
    return joblib.load(path)


def load_metadata(path: Path | str = config.METADATA_FILE) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Order/complete columns exactly as the pipeline expects (missing -> NaN)."""
    return pd.DataFrame(list(records)).reindex(columns=config.INPUT_FEATURES).astype(float)


def risk_band(probability: float, bands: tuple[float, float] = config.RISK_BANDS) -> str:
    """Human-friendly triage bucket, independent of the binary decision threshold."""
    low, high = bands
    if probability < low:
        return "low"
    return "moderate" if probability < high else "high"


def predict_records(model, records: Iterable[dict], threshold: float = 0.5) -> list[dict]:
    frame = to_frame(records)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= threshold)
        results.append({
            "prediction": pred,
            "label": LABELS[pred],
            "confidence": round(float(p if pred else 1 - p), 4),
            "risk_probability": round(float(p), 4),
            "risk_band": risk_band(float(p)),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart-disease risk from JSON")
    parser.add_argument("input", type=Path, help="JSON file: one record or a list")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(args.model), records), indent=2))


if __name__ == "__main__":
    main()
