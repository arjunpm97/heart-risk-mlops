"""FastAPI model-serving app with structured request logging and Prometheus metrics.

Endpoints: POST /predict, POST /predict/batch, GET /health, GET /ready,
GET /model-info, GET /metrics (Prometheus exposition format).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
)

from api.schemas import BatchRequest, BatchResponse, PatientRecord, PredictionResponse
from heart_risk.predict import load_metadata, load_model, predict_records

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class JsonFormatter(logging.Formatter):
    """One JSON object per line - easy to ship to Loki/ELK/CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                 "logger": record.name, "msg": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


def _configure_logger() -> logging.Logger:
    log = logging.getLogger("heart_api")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            log.addHandler(file_handler)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    log.propagate = False
    return log


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("api_requests_total", "HTTP requests",
                                ["method", "endpoint", "status"], registry=self.registry)
        self.latency = Histogram("api_request_latency_seconds", "Request latency",
                                 ["endpoint"], registry=self.registry,
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
        self.predictions = Counter("model_predictions_total", "Predictions by class",
                                   ["label"], registry=self.registry)
        self.bands = Counter("model_risk_band_total", "Predictions by risk band",
                             ["band"], registry=self.registry)
        self.risk = Histogram("model_risk_probability", "Predicted disease probability",
                              registry=self.registry,
                              buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        self.in_flight = Gauge("api_in_flight_requests", "Requests being served",
                               registry=self.registry)
        self.model_loaded = Gauge("model_loaded", "1 if the model is loaded",
                                  registry=self.registry)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL))
    log = _configure_logger()
    metrics = Metrics()

    app = FastAPI(title="Heart Disease Risk API", version="1.0.0",
                  description="Predicts heart-disease risk from 13 clinical attributes.")
    app.state.model = None
    app.state.metadata = {}

    try:
        app.state.model = load_model(model_path)
        app.state.metadata = load_metadata(model_path.with_name("metadata.json"))
        metrics.model_loaded.set(1)
        log.info("model loaded", extra={"extra_fields": {
            "model_path": str(model_path), "algorithm": app.state.metadata.get("algorithm")}})
    except Exception as err:  # keep serving /health with model_loaded=0
        metrics.model_loaded.set(0)
        log.error("model load failed", extra={"extra_fields": {"error": str(err)}})

    def model_version() -> str:
        meta = app.state.metadata
        return f"{meta.get('algorithm', 'unknown')}-{meta.get('model_version', '0')}"

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        endpoint = request.url.path
        metrics.in_flight.inc()
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            metrics.in_flight.dec()
            if endpoint != "/metrics":
                metrics.requests.labels(request.method, endpoint, str(status)).inc()
                metrics.latency.labels(endpoint).observe(elapsed)
                log.info("request", extra={"extra_fields": {
                    "request_id": request_id, "method": request.method, "path": endpoint,
                    "status": status, "latency_ms": round(elapsed * 1000, 2),
                    "client": request.client.host if request.client else None}})

    def _predict(records: list[dict], request: Request) -> list[dict]:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        results = predict_records(app.state.model, records,
                                  threshold=app.state.metadata.get("threshold", 0.5))
        for rec, res in zip(records, results):
            metrics.predictions.labels(res["label"]).inc()
            metrics.bands.labels(res["risk_band"]).inc()
            metrics.risk.observe(res["risk_probability"])
            log.info("prediction", extra={"extra_fields": {
                "request_id": request.state.request_id, "input": rec, **res}})
        return results

    @app.get("/")
    def root():
        return {"service": "heart-risk-api", "docs": "/docs", "predict": "POST /predict"}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.get("/ready")
    def ready():
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info():
        return app.state.metadata or {"detail": "no metadata"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: PatientRecord, request: Request):
        result = _predict([record.model_dump()], request)[0]
        return {**result, "model_version": model_version(),
                "request_id": request.state.request_id}

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        results = _predict([r.model_dump() for r in batch.records], request)
        return {"predictions": results, "model_version": model_version(),
                "request_id": request.state.request_id}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
