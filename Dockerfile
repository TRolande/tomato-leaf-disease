FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KERAS_HOME=/opt/keras
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the ImageNet weights at BUILD time into a shared cache, so the container never needs internet at start.
RUN python -c "import tensorflow as tf; tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)"

# Run as a non-root user. Uploads, the database and retrained models are written under /app/data and /app/models.
RUN useradd --create-home --uid 1000 app
COPY src ./src
COPY tomato_leaf_checking_app ./tomato_leaf_checking_app
COPY models ./models
COPY data ./data
RUN mkdir -p data/uploads && chown -R app:app data models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

# One worker: the model, the SQLite database and the retraining thread all live in this process.
CMD ["uvicorn", "tomato_leaf_checking_app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
