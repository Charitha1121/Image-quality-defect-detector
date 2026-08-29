#!/bin/sh

echo "=== AuraVision Server Starting ==="

# Check if model weight files are present
if [ ! -f "backend/models/classical_rf.joblib" ] || [ ! -f "backend/models/cnn_model.pth" ]; then
    echo "Trained models not found. Running end-to-end training pipeline first..."
    python -m ml.train
else
    echo "Pre-trained models found. Skipping training phase."
fi

echo "Starting FastAPI application server on port 8000..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
