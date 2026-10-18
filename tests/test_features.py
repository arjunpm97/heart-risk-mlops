import numpy as np
import pytest

from heart_risk import config
from heart_risk.features import ClinicalFeatureBuilder, build_preprocessor


def test_engineered_features_values(sample_record):
    import pandas as pd
    out = ClinicalFeatureBuilder().fit_transform(pd.DataFrame([sample_record]))
    row = out.iloc[0]
    assert row["hr_pct_of_max"] == pytest.approx(150 / (220 - 63))
    assert row["rate_pressure_product"] == pytest.approx(150 * 145 / 1000)
    assert row["st_risk"] == pytest.approx(2.3)  # exang == 0 -> no amplification
    assert row["vessel_age_burden"] == pytest.approx(63 / 50)  # ca == 0


def test_vessel_age_burden_grows_with_vessels_and_age(sample_record):
    import pandas as pd
    frame = pd.DataFrame([sample_record, dict(sample_record, ca=3),
                          dict(sample_record, ca=3, age=75)])
    burden = ClinicalFeatureBuilder().transform(frame)["vessel_age_burden"].tolist()
    assert burden[0] < burden[1] < burden[2]


def test_feature_builder_does_not_mutate_input(clean_df):
    X = clean_df[config.INPUT_FEATURES]
    before = X.copy()
    ClinicalFeatureBuilder().transform(X)
    assert X.equals(before)


def test_preprocessor_output_is_finite_and_imputed(clean_df):
    X = clean_df[config.INPUT_FEATURES].copy()
    X.loc[:5, ["ca", "thal", "chol"]] = np.nan
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert np.isfinite(Xt).all()


def test_preprocessor_robust_scales_numeric_columns(clean_df):
    pre = build_preprocessor()
    Xt = pre.fit_transform(clean_df[config.INPUT_FEATURES])
    n_scaled = len(config.NUMERIC_FEATURES) + len(config.ENGINEERED_FEATURES)
    scaled = Xt[:, :n_scaled]
    # RobustScaler centres on the median and divides by the inter-quartile range
    assert np.allclose(np.median(scaled, axis=0), 0, atol=1e-7)
    q75, q25 = np.percentile(scaled, [75, 25], axis=0)
    assert np.allclose(q75 - q25, 1, atol=1e-6)


def test_preprocessor_one_hot_and_unknown_categories(clean_df, sample_record):
    import pandas as pd
    pre = build_preprocessor().fit(clean_df[config.INPUT_FEATURES])
    names = pre.named_steps["columns"].get_feature_names_out()
    assert any(n.startswith("cp_") for n in names)
    assert any(n.startswith("thal_") for n in names)
    unseen = dict(sample_record, restecg=99)  # category never seen in training
    out = pre.transform(pd.DataFrame([unseen])[config.INPUT_FEATURES])
    assert out.shape[1] == len(names)
