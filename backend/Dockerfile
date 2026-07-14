FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CNN_CHECKPOINT_PATH=/app/ml/saved_models/cnn_efficientnet_b0.pth
ENV VIT_CHECKPOINT_PATH=/app/ml/saved_models/vit_deit_tiny.pth
ENV ENSEMBLE_CONFIG_PATH=/app/ml/saved_models/ensemble_config.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
