"""Request / response models. Ranges follow the UCI attribute documentation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest-pain type")
    trestbps: float = Field(..., gt=50, lt=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=80, lt=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., gt=50, lt=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels coloured (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")


class BatchRequest(BaseModel):
    records: list[PatientRecord] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    risk_probability: float = Field(..., description="Probability of heart disease")
    risk_band: Literal["low", "moderate", "high"] = Field(
        ..., description="Triage bucket: low < 0.30 <= moderate < 0.60 <= high")


class PredictionResponse(Prediction):
    # pydantic reserves the "model_" prefix; model_version is our own field name
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    model_version: str
    request_id: str
