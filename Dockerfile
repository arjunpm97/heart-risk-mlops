# Build stage: compile wheels so the runtime image stays small
FROM python:3.13-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt

# Runtime stage
FROM python:3.13-slim
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib \
    PORT=8000

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels \
 && useradd --create-home --uid 1001 appuser

COPY src/ ./src/
COPY api/ ./api/
COPY models/model.joblib models/metadata.json ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
