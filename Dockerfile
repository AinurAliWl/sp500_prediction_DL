FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SP500_ARTIFACT_DIR=/app/models/returns_mlflow_v1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/returns_mlflow_v1/lstm.pt models/returns_mlflow_v1/transformer.pt models/returns_mlflow_v1/feature_scaler.joblib models/returns_mlflow_v1/target_scaler.joblib models/returns_mlflow_v1/config.json ./models/returns_mlflow_v1/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
