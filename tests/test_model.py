import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from heart_risk import config
from heart_risk.features import build_pipeline, feature_names
from heart_risk.predict import load_model, predict_records, risk_band, to_frame
from heart_risk.train import evaluate, run, stability_score, tune_threshold


def test_risk_band_boundaries():
    assert risk_band(0.0) == "low"
    assert risk_band(0.2999) == "low"
    assert risk_band(0.30) == "moderate"
    assert risk_band(0.5999) == "moderate"
    assert risk_band(0.60) == "high"
    assert risk_band(1.0) == "high"


def test_stability_score_penalises_variance():
    steady = {"cv_roc_auc_mean": 0.90, "cv_roc_auc_std": 0.01}
    erratic = {"cv_roc_auc_mean": 0.91, "cv_roc_auc_std": 0.05}
    assert stability_score(steady) > stability_score(erratic)


def test_tuned_threshold_reaches_target_recall(clean_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X, y = clean_df[config.INPUT_FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    thr = tune_threshold(pipe, X, y, cv, target_recall=0.9)
    assert 0 < thr <= 1
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    recall = ((oof >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert recall >= 0.9


def test_pipeline_fits_and_returns_probabilities(clean_df):
    X, y = clean_df[config.INPUT_FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert len(feature_names(pipe)) == pipe.named_steps["model"].n_features_in_


def test_evaluate_returns_all_metrics_in_range(clean_df):
    X, y = clean_df[config.INPUT_FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    metrics = evaluate(pipe, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_saved_model_roundtrip_is_identical(trained_model_path, clean_df):
    X = clean_df[config.INPUT_FEATURES]
    original = joblib.load(trained_model_path)
    reloaded = load_model(trained_model_path)
    assert np.allclose(original.predict_proba(X), reloaded.predict_proba(X))


def test_predict_records_contract(trained_model_path, sample_record):
    model = load_model(trained_model_path)
    partial = {k: v for k, v in sample_record.items() if k not in ("ca", "thal")}
    out = predict_records(model, [sample_record, partial])
    assert len(out) == 2
    for res in out:
        assert res["prediction"] in (0, 1)
        assert 0.5 <= res["confidence"] <= 1
        assert 0 <= res["risk_probability"] <= 1
        assert res["risk_band"] == risk_band(res["risk_probability"])
    assert list(to_frame([partial]).columns) == config.INPUT_FEATURES


def test_training_run_end_to_end(tmp_path, clean_df, monkeypatch):
    # keep this synthetic-data run out of the CI run's summary page
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    data = tmp_path / "clean.csv"
    clean_df.to_csv(data, index=False)
    meta = run(fast=True, tracking_uri=(tmp_path / "mlruns").as_uri(), data_path=data,
               model_dir=tmp_path / "models", report_dir=tmp_path / "reports")
    assert (tmp_path / "models" / "model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert saved["algorithm"] == meta["algorithm"]
    assert 0 <= saved["test_metrics"]["roc_auc"] <= 1
    assert 0 < saved["threshold"] <= 1
    assert "recall" in saved["test_metrics_tuned_threshold"]
    assert (tmp_path / "reports" / "model_comparison.csv").exists()
