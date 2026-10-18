"""Central configuration: paths, column definitions and training constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"

UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data",
    # Fallback: the official zip bundle of the same dataset (UCI id 45)
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
]

# Column names as documented in heart-disease.names (14 attributes used by all studies)
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous / ordinal features -> imputed + scaled
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Engineered features (created by ClinicalFeatureBuilder) -> imputed + robust-scaled
ENGINEERED_FEATURES = ["hr_pct_of_max", "rate_pressure_product", "st_risk", "vessel_age_burden"]
# Nominal codes -> imputed + one-hot encoded
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
# Already-binary flags -> imputed, passed through
BINARY_FEATURES = ["sex", "fbs", "exang"]

INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Screening use-case: the decision threshold is tuned on out-of-fold predictions so that
# at least this share of diseased patients is flagged (missing a sick patient costs more
# than a false alarm).
TARGET_RECALL = 0.90
# Probability cut-offs for the risk band returned by the API: low < 0.30 <= moderate < 0.60 <= high
RISK_BANDS = (0.30, 0.60)
EXPERIMENT_NAME = "heart-risk-classification"
