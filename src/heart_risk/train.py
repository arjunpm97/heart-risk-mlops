"""Model training with cross-validated hyper-parameter search and MLflow tracking.

Every grid candidate is logged as a nested MLflow run (params + CV metrics); each
model family gets a parent run holding the best params, hold-out test metrics,
diagnostic plots and the serialised sklearn pipeline. The champion (highest mean
CV ROC-AUC minus its std - chosen on CV, never on the test set) gets a decision
threshold tuned for recall on out-of-fold predictions and is exported to ``models/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics import precision_recall_curve  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)

from heart_risk import __version__, config  # noqa: E402
from heart_risk.data import load_clean, split_xy  # noqa: E402
from heart_risk.features import build_pipeline, feature_names  # noqa: E402

logger = logging.getLogger(__name__)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict
    notes: str = ""


@dataclass
class ModelResult:
    name: str
    run_id: str
    best_params: dict
    cv: dict
    test: dict
    pipeline: object = field(repr=False)


def candidates(fast: bool = False) -> list[Candidate]:
    """Model families and their search spaces. ``fast`` shrinks grids for CI/tests."""
    rs = config.RANDOM_STATE
    if fast:
        return [
            Candidate("logistic_regression",
                      LogisticRegression(max_iter=2000, solver="liblinear", random_state=rs),
                      {"model__C": [0.1, 1.0]}),
            Candidate("random_forest",
                      RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
                      {"model__max_depth": [4, None]}),
        ]
    return [
        Candidate(
            "logistic_regression",
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=rs),
            {
                "model__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
            "Linear baseline; interpretable coefficients, L1/L2 regularisation.",
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [4, 6, None],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced_subsample"],
            },
            "Bagged trees; captures non-linear interactions, robust to scaling.",
        ),
        Candidate(
            "extra_trees",
            ExtraTreesClassifier(random_state=rs, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
            "Randomised split points; lower variance than a random forest on n~300.",
        ),
    ]


def _git_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _serving_requirements() -> list[str]:
    """Pinned packages needed to load the logged pipeline. The project's own code is
    bundled with the model via code_paths, so the model is self-contained."""
    return [f"scikit-learn=={sklearn.__version__}", f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}", f"joblib=={joblib.__version__}"]


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def stability_score(cv: dict) -> float:
    """Champion criterion: mean CV ROC-AUC penalised by its fold-to-fold spread, so a
    model that is good on average but erratic across folds does not win."""
    return cv["cv_roc_auc_mean"] - cv["cv_roc_auc_std"]


def tune_threshold(pipeline, X: pd.DataFrame, y: pd.Series, cv,
                   target_recall: float = config.TARGET_RECALL) -> float:
    """Highest probability threshold whose out-of-fold recall is >= target_recall.

    Uses cross_val_predict on the training split only, so the test set stays untouched.
    """
    oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    _, recall, thresholds = precision_recall_curve(y, oof)
    ok = thresholds[recall[:-1] >= target_recall]
    # round *down* so that rounding can never push a borderline positive below the cut-off
    return float(np.floor(ok.max() * 1000) / 1000) if len(ok) else 0.5


def evaluate(pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def _diagnostic_plots(pipeline, name: str, X_test, y_test, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(out_dir / f"confusion_matrix_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[0], name=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("ROC curve (test)")
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1], name=name)
    axes[1].set_title("Precision-recall (test)")
    paths.append(out_dir / f"roc_pr_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Impurity importance"
    else:
        values, label = model.coef_.ravel(), "Coefficient (standardised)"
    imp = pd.Series(values, index=names).sort_values(key=np.abs).tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=np.where(imp.values >= 0, "#d62828", "#457b9d"))
    ax.set_xlabel(label)
    ax.set_title(f"Top features - {name}")
    paths.append(out_dir / f"feature_importance_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def _cv_summary(search: GridSearchCV, idx: int) -> dict:
    res = search.cv_results_
    out = {}
    for metric in SCORING:
        out[f"cv_{metric}_mean"] = float(res[f"mean_test_{metric}"][idx])
        out[f"cv_{metric}_std"] = float(res[f"std_test_{metric}"][idx])
    return out


def train_candidate(cand: Candidate, X_train, y_train, X_test, y_test,
                    cv, plot_dir: Path, data_md5: str) -> ModelResult:
    search = GridSearchCV(build_pipeline(cand.estimator), cand.grid, scoring=SCORING,
                          refit="roc_auc", cv=cv, n_jobs=-1, return_train_score=True)
    with mlflow.start_run(run_name=cand.name) as parent:
        mlflow.set_tags({"model_family": cand.name, "stage": "tuning",
                         "git_sha": _git_sha(), "notes": cand.notes})
        mlflow.log_params({"cv_folds": cv.get_n_splits(), "search": "grid",
                           "refit_metric": "roc_auc", "n_candidates":
                           int(np.prod([len(v) for v in cand.grid.values()])),
                           "train_rows": len(X_train), "test_rows": len(X_test),
                           "data_md5": data_md5})
        start = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        # one nested run per hyper-parameter combination
        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(run_name=f"{cand.name}-{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                mlflow.log_metrics(_cv_summary(search, i))
                mlflow.log_metric("cv_train_roc_auc_mean",
                                  float(search.cv_results_["mean_train_roc_auc"][i]))
                mlflow.log_metric("rank_roc_auc",
                                  int(search.cv_results_["rank_test_roc_auc"][i]))

        best = search.best_estimator_
        cv_metrics = _cv_summary(search, search.best_index_)
        test_metrics = evaluate(best, X_test, y_test)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)

        for path in _diagnostic_plots(best, cand.name, X_test, y_test, plot_dir):
            mlflow.log_artifact(str(path), artifact_path="plots")
        cv_table = pd.DataFrame(search.cv_results_).drop(columns=["params"])
        cv_path = plot_dir / f"cv_results_{cand.name}.csv"
        cv_table.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), artifact_path="cv")

        signature = infer_signature(X_train, best.predict_proba(X_train))
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_train.head(3),
                                 code_paths=[str(Path(__file__).resolve().parent)],
                                 pip_requirements=_serving_requirements())
        logger.info("%-20s cv_roc_auc=%.4f test_roc_auc=%.4f best=%s", cand.name,
                    cv_metrics["cv_roc_auc_mean"], test_metrics["roc_auc"], best_params)
        return ModelResult(cand.name, parent.info.run_id, best_params, cv_metrics,
                           test_metrics, best)


def _comparison_plot(results: list[ModelResult], X_test, y_test, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        RocCurveDisplay.from_estimator(res.pipeline, X_test, y_test, ax=ax, name=res.name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC comparison on hold-out test set")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_ci_summary(summary: pd.DataFrame, champion: str, threshold: float,
                      tuned: dict) -> None:
    """On GitHub Actions, render the model comparison on the workflow run's summary page."""
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if not target:
        return
    lines = ["## Model comparison", "",
             "| model | CV ROC-AUC | CV std | stability | test ROC-AUC | test recall |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        mark = " (champion)" if r["model"] == champion else ""
        lines.append(f"| {r['model']}{mark} | {r.cv_roc_auc_mean:.3f} | {r.cv_roc_auc_std:.3f} "
                     f"| {r.stability_score:.3f} | {r.test_roc_auc:.3f} | {r.test_recall:.3f} |")
    lines += ["", f"Decision threshold tuned for recall >= {config.TARGET_RECALL}: "
                  f"**{threshold}** (test recall {tuned['recall']:.3f}, "
                  f"precision {tuned['precision']:.3f})"]
    with open(target, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run(fast: bool = False, tracking_uri: str | None = None,
        data_path: Path = config.CLEAN_FILE, model_dir: Path = config.MODELS_DIR,
        report_dir: Path = config.FIGURES_DIR.parent) -> dict:
    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", (config.PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    df = load_clean(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    plot_dir = report_dir / "training"
    data_md5 = _file_md5(Path(data_path))

    results = [train_candidate(c, X_train, y_train, X_test, y_test, cv, plot_dir, data_md5)
               for c in candidates(fast)]

    champion = max(results, key=lambda r: stability_score(r.cv))
    threshold = tune_threshold(champion.pipeline, X_train, y_train, cv)
    tuned_test = evaluate(champion.pipeline, X_test, y_test, threshold)
    client = mlflow.tracking.MlflowClient()
    client.set_tag(champion.run_id, "champion", "true")
    with mlflow.start_run(run_id=champion.run_id):
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metric("stability_score", stability_score(champion.cv))
        mlflow.log_metrics({f"test_tuned_{k}": v for k, v in tuned_test.items()})

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion.pipeline, model_dir / "model.joblib")
    metadata = {
        "model_name": "heart-risk-classifier",
        "model_version": __version__,
        "algorithm": champion.name,
        "best_params": champion.best_params,
        "selection_rule": "max(cv_roc_auc_mean - cv_roc_auc_std)",
        "cv_metrics": champion.cv,
        "test_metrics": champion.test,
        "test_metrics_tuned_threshold": tuned_test,
        "features": config.INPUT_FEATURES,
        "threshold": threshold,
        "target_recall": config.TARGET_RECALL,
        "risk_bands": list(config.RISK_BANDS),
        "mlflow_run_id": champion.run_id,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "data_md5": data_md5,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    summary = pd.DataFrame([
        {"model": r.name, **r.cv, "stability_score": stability_score(r.cv),
         **{f"test_{k}": v for k, v in r.test.items()},
         "best_params": json.dumps(r.best_params, default=str), "run_id": r.run_id}
        for r in results
    ])
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "model_comparison.csv", index=False)
    _write_ci_summary(summary, champion.name, threshold, tuned_test)
    comparison_png = _comparison_plot(results, X_test, y_test, plot_dir / "roc_comparison.png")
    with mlflow.start_run(run_id=champion.run_id):
        mlflow.log_artifact(str(comparison_png), artifact_path="plots")
        mlflow.log_artifact(str(model_dir / "metadata.json"), artifact_path="export")

    logger.info("Champion: %s (run %s)", champion.name, champion.run_id)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart-disease classifiers")
    parser.add_argument("--fast", action="store_true", help="small grids (CI smoke run)")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = run(fast=args.fast, tracking_uri=args.tracking_uri)
    keys = ("algorithm", "threshold", "cv_metrics", "test_metrics",
            "test_metrics_tuned_threshold")
    print(json.dumps({k: meta[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
