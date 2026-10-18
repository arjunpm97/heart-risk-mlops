import numpy as np
import pandas as pd

from heart_risk import config
from heart_risk.data import clean, load_raw, missing_report, split_xy


def test_load_raw_parses_question_marks_as_nan(tmp_path):
    raw = tmp_path / "raw.data"
    raw.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
                   "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,?,2\n")
    df = load_raw(raw)
    assert list(df.columns) == config.RAW_COLUMNS
    assert np.isnan(df.loc[0, "ca"])
    assert np.isnan(df.loc[1, "thal"])


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    expected = (raw_df.drop_duplicates()["num"] > 0).astype(int).to_numpy()
    assert (out[config.TARGET].to_numpy() == expected).all()


def test_clean_nulls_invalid_codes_and_impossible_values(raw_df):
    raw_df.loc[0, "thal"] = 5      # not a valid thal code
    raw_df.loc[1, "cp"] = 9        # not a valid chest-pain type
    raw_df.loc[2, "chol"] = 0      # impossible cholesterol
    out = clean(raw_df)
    assert np.isnan(out.loc[0, "thal"])
    assert np.isnan(out.loc[1, "cp"])
    assert np.isnan(out.loc[2, "chol"])


def test_clean_drops_duplicates_and_unlabelled_rows(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup)] = dup.iloc[1].copy()
    dup.loc[len(dup) - 1, "num"] = np.nan
    out = clean(dup)
    assert len(out) == len(raw_df)


def test_clean_coerces_non_numeric_tokens(raw_df):
    raw_df = raw_df.astype(object)
    raw_df.loc[3, "trestbps"] = "abc"
    out = clean(raw_df)
    assert np.isnan(out.loc[3, "trestbps"])


def test_missing_report_and_split(clean_df):
    clean_df.loc[0, "ca"] = np.nan
    assert missing_report(clean_df).to_dict() == {"ca": 1}
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.INPUT_FEATURES
    assert len(X) == len(y)
