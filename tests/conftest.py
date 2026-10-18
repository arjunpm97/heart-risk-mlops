"""Shared fixtures. Tests are self-contained: they use a synthetic dataset that follows
the UCI schema, so they run in CI without network access or a pre-trained model."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_risk import config
from heart_risk.features import build_pipeline


def make_raw_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    # label loosely tied to known risk factors so models can learn something
    score = (df["cp"] == 4) * 1.5 + df["exang"] + df["oldpeak"] / 2 + df["ca"] - 1.5
    df["num"] = np.where(score + rng.normal(0, 0.7, n) > 0, rng.integers(1, 5, n), 0)
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture()
def clean_df(raw_df) -> pd.DataFrame:
    from heart_risk.data import clean
    return clean(raw_df)


@pytest.fixture()
def sample_record() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}


@pytest.fixture()
def trained_model_path(tmp_path, clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(clean_df[config.INPUT_FEATURES], clean_df[config.TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    (tmp_path / "metadata.json").write_text(
        '{"algorithm": "logistic_regression", "model_version": "test", "threshold": 0.5}')
    return path
