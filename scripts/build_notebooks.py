"""Build and execute the project notebooks (EDA, training/tracking, inference).

Usage: python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 140)"""


def eda() -> nbformat.NotebookNode:
    return new_notebook(cells=[
        new_markdown_cell(
            "# 01 - Data acquisition & EDA\n"
            "UCI Heart Disease, Cleveland subset (303 patients, 13 attributes + `num`). "
            "`num` 0 = no disease, 1-4 = disease, binarised into `target`."),
        new_code_cell(SETUP),
        new_markdown_cell("## 1. Acquire\nThe download script fetches the file from UCI "
                          "(falling back to the zip bundle) and writes the cleaned CSV."),
        new_code_cell("from heart_risk import config\n"
                      "from heart_risk.data import download_raw, load_raw, clean, missing_report\n"
                      "if not config.RAW_FILE.exists():\n    download_raw()\n"
                      "raw = load_raw()\nprint(raw.shape)\nraw.head()"),
        new_markdown_cell("## 2. Data quality\nMissing values are encoded as `?` in the raw file."),
        new_code_cell("print('Missing per column:', missing_report(raw).to_dict())\n"
                      "print('Duplicate rows:', raw.duplicated().sum())\nraw.describe().T"),
        new_markdown_cell(
            "Only `ca` (4) and `thal` (2) are missing (<2% of rows). Rather than dropping rows "
            "from an already small dataset, they are kept as NaN and imputed **inside** the "
            "model pipeline (median / most-frequent), so the imputation statistics are learned "
            "from the training fold only and re-applied identically at serving time."),
        new_code_cell("df = clean(raw)\ndf.to_csv(config.CLEAN_FILE, index=False)\n"
                      "df[config.TARGET].value_counts()"),
        new_markdown_cell("## 3. Visual exploration"),
        new_code_cell("from IPython.display import Image, display\n"
                      "from heart_risk.eda import run_eda\n"
                      "figs = run_eda()\nfor name, path in figs.items():\n"
                      "    display(Image(filename=str(path)))"),
        new_markdown_cell(
            "## 4. Findings that drive modelling\n"
            "* **Class balance** is 54% / 46%: mild, so accuracy is meaningful, but ROC-AUC is "
            "used as the primary selection metric and `class_weight='balanced'` is searched.\n"
            "* **Strongest signals**: `thal` (reversible defect), `ca` (vessels coloured), "
            "`cp` = asymptomatic, `exang`, `oldpeak`, and *low* `thalach` "
            "(|Spearman rho| 0.4-0.5 with the target).\n"
            "* `chol`, `fbs` and `trestbps` are weak on their own, and `chol` / `oldpeak` are "
            "right-skewed, so numeric features use a **RobustScaler** (median / IQR).\n"
            "* Diseased patients reach a lower share of their age-predicted max heart rate "
            "(scatter plot), motivating `hr_pct_of_max = thalach / (220 - age)`; blood pressure "
            "is combined with heart rate into the **rate-pressure product** (cardiac workload).\n"
            "* Men have a higher disease rate than women in every age band, and up to 70 risk "
            "rises with age (age-band chart; 71-80 has only 6 patients). With `ca` being a top "
            "signal, this motivates `vessel_age_burden = (age / 50) * (1 + ca)`.\n"
            "* `cp`, `restecg`, `slope`, `thal` are nominal codes, so they are one-hot encoded "
            "rather than treated as numbers. `restecg = 1` has only 4 rows, so the encoder uses "
            "`handle_unknown='ignore'`."),
        new_code_cell("df.groupby(config.TARGET)"
                      "[['age','thalach','oldpeak','ca']].mean().round(2)"),
    ])


def training() -> nbformat.NotebookNode:
    return new_notebook(cells=[
        new_markdown_cell(
            "# 02 - Feature engineering, model training & experiment tracking\n"
            "Training is implemented in `src/heart_risk/train.py` (also run by CI). "
            "This notebook re-runs it on demand and analyses the MLflow experiment."),
        new_code_cell(SETUP),
        new_markdown_cell("## 1. Preprocessing pipeline\nEngineered features -> impute -> "
                          "scale numerics / one-hot categoricals. Everything is one sklearn "
                          "`Pipeline`, so the saved artefact accepts raw patient records."),
        new_code_cell("from heart_risk.features import build_preprocessor\n"
                      "from heart_risk.data import load_clean, split_xy\n"
                      "X, y = split_xy(load_clean())\npre = build_preprocessor().fit(X)\n"
                      "names = pre.named_steps['columns'].get_feature_names_out()\n"
                      "print(len(names), 'model features:')\nprint(list(names))"),
        new_markdown_cell("## 2. Train (set `RETRAIN = True` to create new MLflow runs)"),
        new_code_cell("RETRAIN = False\nfrom heart_risk import config\n"
                      "if RETRAIN or not config.MODEL_FILE.exists():\n"
                      "    from heart_risk.train import run\n    run()"),
        new_markdown_cell("## 3. Experiment summary from MLflow"),
        new_code_cell(
            "import mlflow\nmlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())\n"
            "runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME])\n"
            "parents = runs[runs['tags.mlflow.parentRunId'].isna()]\n"
            "cols = ['tags.mlflow.runName','metrics.cv_roc_auc_mean','metrics.cv_roc_auc_std',"
            "'metrics.cv_f1_mean','metrics.test_accuracy','metrics.test_precision',"
            "'metrics.test_recall','metrics.test_roc_auc','metrics.decision_threshold',"
            "'metrics.test_tuned_recall','tags.champion']\n"
            "parents = parents.assign(stability=parents['metrics.cv_roc_auc_mean']"
            " - parents['metrics.cv_roc_auc_std'])\n"
            "parents[cols + ['stability']].sort_values('stability', ascending=False).round(4)"),
        new_code_cell(
            "children = runs[runs['tags.mlflow.parentRunId'].notna()]\n"
            "print('nested hyper-parameter runs logged:', len(children))\n"
            "children.groupby(children['tags.mlflow.runName'].str.rsplit('-', n=1).str[0])"
            "['metrics.cv_roc_auc_mean'].describe().round(4)"),
        new_code_cell("from IPython.display import Image, display\n"
                      "display(Image(filename=str(ROOT/'reports/training/roc_comparison.png')))\n"
                      "for m in ['logistic_regression','random_forest','extra_trees']:\n"
                      "    display(Image(filename=str(\n"
                      "        ROOT/f'reports/training/feature_importance_{m}.png')))"),
        new_markdown_cell(
            "## 4. Model selection and threshold\nThe champion maximises **mean CV ROC-AUC minus "
            "its standard deviation** across the 5 folds (a stability-penalised score), computed "
            "on the training split only; the 61-row test split is never used for choosing. "
            "Logistic regression wins this rule and is also the most interpretable model, which "
            "matters for a clinical risk score.\n\n"
            "Because missing a sick patient is costlier than a false alarm, the decision "
            "threshold is then tuned on **out-of-fold** predictions to the highest value that "
            "still gives >= 90% recall. The tuned threshold and its test metrics are logged to "
            "MLflow (`decision_threshold`, `test_tuned_*`) and stored in `models/metadata.json`, "
            "where the API picks them up."),
    ])


def inference() -> nbformat.NotebookNode:
    return new_notebook(cells=[
        new_markdown_cell("# 03 - Inference\nLoad the packaged pipeline and score raw records; "
                          "then call the same model through the REST API."),
        new_code_cell(SETUP),
        new_code_cell("import json\nfrom heart_risk.predict import load_model, load_metadata, "
                      "predict_records\nmodel = load_model()\nmeta = load_metadata()\n"
                      "print(meta['algorithm'], meta['test_metrics'])\n"
                      "records = json.loads((ROOT/'scripts/sample_input.json').read_text())\n"
                      "pd.DataFrame(predict_records(model, records))"),
        new_markdown_cell("Missing optional fields (`ca`, `thal`) are imputed by the pipeline:"),
        new_code_cell("partial = {k: v for k, v in records[1].items() if k not in ('ca','thal')}\n"
                      "predict_records(model, [partial])"),
        new_markdown_cell("## Through the API (in-process test client; use `requests` against "
                          "http://localhost:8000 for the container)"),
        new_code_cell("from fastapi.testclient import TestClient\nfrom api.main import create_app\n"
                      "client = TestClient(create_app())\n"
                      "r = client.post('/predict', json=records[0])\nr.status_code, r.json()"),
    ])


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")
    for name, builder in [("01_eda", eda), ("02_training_and_tracking", training),
                          ("03_inference", inference)]:
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        executor.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        path = NB_DIR / f"{name}.ipynb"
        nbformat.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
