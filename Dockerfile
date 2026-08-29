FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
# We install PyTorch CPU-only version specifically to avoid pulling massive CUDA binaries (reduces image size by ~4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sqlalchemy \
    pydantic \
    python-multipart \
    scikit-learn \
    opencv-python-headless \
    numpy \
    joblib \
    pandas

# Copy application directories
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY ml/ /app/ml/

# Create media dirs
RUN mkdir -p /app/static/uploads /app/static/heatmaps /app/backend/models

# Expose port
EXPOSE 8000

# Copy and make entrypoint script executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
