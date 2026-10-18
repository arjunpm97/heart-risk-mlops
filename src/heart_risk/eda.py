"""Exploratory data analysis - produces the figures used in the notebook and report."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_risk import config  # noqa: E402
from heart_risk.data import load_clean, load_raw, missing_report  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {0: "#457b9d", 1: "#d62828"}
ACCENT = "#d62828"
plt.style.use("seaborn-v0_8-whitegrid")
LABELS = {0: "No disease", 1: "Disease"}

CATEGORY_NAMES = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120", 1: ">120"},
    "restecg": {0: "normal", 1: "ST-T abn.", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "up", 2: "flat", 3: "down"},
    "thal": {3: "normal", 6: "fixed", 7: "reversible"},
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.annotate(f"{val} ({val / counts.sum():.0%})",
                    (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom")
    ax.set_title("Class balance (target)")
    ax.set_ylabel("Patients")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out_dir, "01_class_balance.png")


def plot_missing(raw: pd.DataFrame, out_dir: Path) -> Path:
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(miss.index, miss.values, color="#1d3557")
    ax.set_title("Missing values per column (raw file, '?' tokens)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out_dir, "02_missing_values.png")


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        for cls in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == cls, col].dropna(), ax=ax,
                         color=PALETTE[cls], label=LABELS[cls], kde=True,
                         stat="density", alpha=0.45, bins=20)
        ax.set_title(col)
        ax.set_xlabel("")
    axes[0, 0].legend()
    fig.suptitle("Distribution of numeric features by diagnosis", fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir, "03_numeric_histograms.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle only
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.75}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearman correlation matrix (incl. target)")
    return _save(fig, out_dir, "04_correlation_heatmap.png")


def plot_categorical_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = list(CATEGORY_NAMES)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), cols):
        sub = df[[col, config.TARGET]].dropna()
        rate = sub.groupby(col)[config.TARGET].agg(["mean", "size"])
        names = [CATEGORY_NAMES[col].get(int(k), str(k)) for k in rate.index]
        ax.bar(names, rate["mean"], color=ACCENT)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{col}: disease rate")
        ax.tick_params(axis="x", rotation=30)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Share of patients with heart disease per category", fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir, "05_categorical_disease_rate.png")


def plot_age_thalach(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in (0, 1):
        sub = df[df[config.TARGET] == cls]
        ax.scatter(sub["age"], sub["thalach"], c=PALETTE[cls], label=LABELS[cls],
                   alpha=0.7, edgecolor="white", linewidth=0.5)
    ages = pd.Series(range(28, 80))
    ax.plot(ages, 220 - ages, "--", color="grey", label="Age-predicted max HR")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Max heart rate achieved (thalach)")
    ax.set_title("Exercise heart-rate response vs age")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out_dir, "06_age_vs_thalach.png")


def plot_age_band_risk(df: pd.DataFrame, out_dir: Path) -> Path:
    """Disease rate per age decade, split by sex - shows the risk gap between men and women."""
    bands = pd.cut(df["age"], bins=[28, 40, 50, 60, 70, 80],
                   labels=["29-40", "41-50", "51-60", "61-70", "71-80"])
    table = (df.assign(age_band=bands, sex_name=df["sex"].map({0: "female", 1: "male"}))
               .groupby(["age_band", "sex_name"], observed=False)[config.TARGET]
               .agg(["mean", "size"]).reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    xs = np.arange(table["age_band"].nunique())
    for offset, (sex, colour) in zip((-width / 2, width / 2),
                                     (("female", "#a8dadc"), ("male", "#1d3557"))):
        sub = table[table["sex_name"] == sex]
        ax.bar(xs + offset, sub["mean"].fillna(0), width, label=sex, color=colour)
        for x, m, n in zip(xs + offset, sub["mean"].fillna(0), sub["size"]):
            ax.text(x, m + 0.02, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(xs, table["age_band"].cat.categories)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Age band (years)")
    ax.set_ylabel("Share with heart disease")
    ax.set_title("Disease rate by age band and sex")
    ax.legend()
    return _save(fig, out_dir, "07_age_band_risk.png")


def run_eda(out_dir: Path = config.FIGURES_DIR) -> dict[str, Path]:
    raw = load_raw()
    df = load_clean()
    logger.info("Missing (clean): %s", missing_report(df).to_dict())
    figures = {
        "class_balance": plot_class_balance(df, out_dir),
        "missing": plot_missing(raw, out_dir),
        "histograms": plot_numeric_histograms(df, out_dir),
        "correlation": plot_correlation(df, out_dir),
        "categorical": plot_categorical_rates(df, out_dir),
        "age_thalach": plot_age_thalach(df, out_dir),
        "age_band_risk": plot_age_band_risk(df, out_dir),
    }
    df.describe().T.round(2).to_csv(out_dir.parent / "summary_statistics.csv")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, path in run_eda(args.out).items():
        print(f"{name:14s} -> {path}")


if __name__ == "__main__":
    main()
