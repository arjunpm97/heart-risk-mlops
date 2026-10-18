"""Feature engineering + preprocessing pipeline (fitted on training data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from heart_risk import config


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds domain-motivated features derived from the raw inputs.

    hr_pct_of_max         : thalach / (220 - age), the share of the age-predicted
                            maximum heart rate reached during the exercise test.
    rate_pressure_product : thalach * trestbps / 1000, the "double product", a standard
                            cardiology proxy for myocardial oxygen demand.
    st_risk               : oldpeak * (1 + exang), ST depression amplified when the
                            test also induced angina.
    vessel_age_burden     : (age / 50) * (1 + ca), number of narrowed vessels weighted
                            by age (the same blockage is more serious in older patients).
    Missing inputs propagate as NaN and are imputed downstream.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        max_hr = (220 - X["age"]).replace(0, np.nan)
        X["hr_pct_of_max"] = X["thalach"] / max_hr
        X["rate_pressure_product"] = X["thalach"] * X["trestbps"] / 1000.0
        X["st_risk"] = X["oldpeak"] * (1 + X["exang"].fillna(0))
        X["vessel_age_burden"] = (X["age"] / 50.0) * (1 + X["ca"])
        return X


def build_preprocessor() -> Pipeline:
    # RobustScaler (median / IQR) because chol, oldpeak and the engineered ratios have
    # long right tails; a few extreme patients should not dominate the scale.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    columns = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
    ], verbose_feature_names_out=False)

    return Pipeline([
        ("engineer", ClinicalFeatureBuilder()),
        ("columns", columns),
    ])


def build_pipeline(estimator) -> Pipeline:
    """Full inference pipeline: raw patient record -> class probability."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def feature_names(pipeline: Pipeline) -> list[str]:
    """Output feature names of a *fitted* pipeline's preprocessing step."""
    columns = pipeline.named_steps["preprocess"].named_steps["columns"]
    return list(columns.get_feature_names_out())
