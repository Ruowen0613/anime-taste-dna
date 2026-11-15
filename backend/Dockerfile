FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT_PATH=/app/data/weighted_als_hybrid.npz

WORKDIR /app

COPY backend/pyproject.toml /tmp/backend/pyproject.toml
COPY backend/app /tmp/backend/app
RUN python -m pip install --no-cache-dir /tmp/backend

COPY backend/model/weighted-als-hybrid-v1.json /app/model-manifest.json
COPY backend/scripts/verify_model_artifact.py /app/scripts/verify_model_artifact.py
COPY data/artifacts/weighted-als-hybrid-v1/weighted_als_hybrid.npz /app/data/weighted_als_hybrid.npz

RUN python /app/scripts/verify_model_artifact.py \
    --artifact /app/data/weighted_als_hybrid.npz \
    --manifest /app/model-manifest.json

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
