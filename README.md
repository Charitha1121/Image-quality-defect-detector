# AuraVision: AI-Powered Image Quality & Defect Detection

AuraVision is a production-grade, deployable full-stack application that automatically evaluates the visual quality of input images, detects common degradations (focus blur, exposure clipping, sensor noise, compression artifacts), and identifies local physical defects (scratches, sensor smudges, pixel column failures) using a hybrid machine learning pipeline.

---

## Key Features
*   **Engineered Image Quality Features**: Fast extraction of Laplacian variance (sharpness), grayscale histograms (exposure), RMS contrast, pixel-difference statistics (noise), and FFT energy distribution (high-frequency spectral density).
*   **Dual-Branch Decision Model**:
    *   **Classical Branch**: Random Forest model mapping engineered feature vectors to degradation categories. Highly interpretable and fast ($<5\text{ms}$).
    *   **Deep Learning Branch**: A lightweight PyTorch convolutional neural network (CNN) designed for spatial anomaly patterns and local defect detection.
*   **Model Explainability**: Grad-CAM saliency heatmaps highlighting the exact spatial regions directing the CNN's defect decisions.
*   **Robust Heuristics Fallback**: A computer-vision fallback system (using Hough line transform and shape analysis) that takes over if model weights are missing, preventing application start crashes.
*   **Modern Glassmorphism Dashboard**: A beautiful single-page dashboard serving drag-and-drop file uploads, circular progress dials, original vs Grad-CAM heatmap toggles, issue checklists, and paginated historical analysis retrieval.
*   **Dockerized Deployment**: Single command setup using Docker Compose with persistent databases, media folders, and health check indicators.

---

## Directory Structure

```
image-quality-detector/
├── ml/
│   ├── dataset.py        # Procedural clean base images & synthetic degradation generator
│   ├── features.py       # Grayscale, Laplacian, Noise, and FFT feature extraction
│   ├── classical.py      # Random Forest training on engineered features
│   ├── deep_learning.py  # PyTorch CNN model and Grad-CAM implementation
│   └── train.py          # Orchestrates dataset synthesis and model training
├── backend/
│   └── app/
│       ├── database.py   # SQLite database engine and session setup
│       ├── models.py     # SQLAlchemy database schema (AnalysisResult)
│       ├── schemas.py    # Pydantic serialization and validation schemas
│       ├── pipeline.py   # Combined inference logic and CV heuristics fallback
│       └── main.py       # FastAPI application, REST endpoints, and Static mounts
├── frontend/
│   ├── index.html        # Front-end HTML5 template
│   ├── style.css         # Custom stylesheet (Glassmorphism layout, animations)
│   └── app.js            # Frontend JavaScript (Uploads, rendering, history)
├── Dockerfile            # Container definition (slim Debian + PyTorch CPU + Headless CV)
├── entrypoint.sh         # Startup script executing training check and Uvicorn
└── docker-compose.yml    # Compose file mapping ports, volumes, and healthchecks
```

---

## Getting Started

### 1. Run via Docker Compose (Recommended)
This approach handles dependencies and training internally within the container.

1.  Make sure you have Docker and Docker Compose installed.
2.  Navigate to the project root and run:
    ```bash
    docker-compose up --build
    ```
3.  On first startup, the container will detect that model weight files are missing and automatically run the dataset generation and model training pipeline. This takes approximately 1-2 minutes.
4.  Once completed, the FastAPI application will launch. Open your browser and navigate to:
    ```
    http://localhost:8000
    ```

### 2. Local Environment Execution
If you wish to run on your host system:

1.  **Install dependencies**:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install fastapi uvicorn sqlalchemy pydantic python-multipart scikit-learn opencv-python-headless numpy joblib pandas
    ```
2.  **Train the models**:
    ```bash
    python -m ml.train
    ```
    This generates a synthetic dataset of 700 images, trains both the classical Random Forest and CNN models, saves weights to `backend/models/`, and generates a `metrics.json` file.
3.  **Start the FastAPI server**:
    ```bash
    python -m uvicorn backend.app.main:app --reload
    ```
4.  Navigate to `http://127.0.0.1:8000` in your web browser.

---

## API Documentation

FastAPI automatically generates interactive Swagger docs under `http://localhost:8000/docs`.

### Key Endpoints

#### 1. System Health
*   **Endpoint**: `GET /health`
*   **Description**: Returns database connection and model load statuses.
*   **Example Response**:
    ```json
    {
      "status": "healthy",
      "database": "connected",
      "models": {
        "classical_loaded": true,
        "cnn_loaded": true,
        "mode": "hybrid"
      }
    }
    ```

#### 2. Analyze Image
*   **Endpoint**: `POST /api/analyze`
*   **Content-Type**: `multipart/form-data`
*   **Request Body**: `file` (Binary Image File)
*   **Example Request**:
    ```bash
    curl -X POST "http://localhost:8000/api/analyze" \
      -H "accept: application/json" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@sample_image.png;type=image/png"
    ```
*   **Example Response**:
    ```json
    {
      "id": 1,
      "filename": "sample_image.png",
      "timestamp": "2026-08-27T17:15:30",
      "quality_score": 82.5,
      "quality_label": "ACCEPTABLE",
      "issues": [
        {
          "type": "noise",
          "severity": "low",
          "confidence": 0.71
        }
      ],
      "features": {
        "blur_laplacian_var": 184.2,
        "brightness_mean": 118.5,
        "brightness_std": 62.4,
        "contrast_rms": 62.4,
        "noise_std": 9.15,
        "fft_high_freq_ratio": 0.125
      },
      "heatmap_url": "/static/heatmaps/heatmap_sample_image.png"
    }
    ```

#### 3. Fetch History List
*   **Endpoint**: `GET /api/results`
*   **Query Parameters**: `page` (default 1), `limit` (default 10), `label` (optional: ACCEPTABLE, DEGRADED, DEFECTIVE)
*   **Description**: Retrieves a paginated history list of analyzed records sorted newest-first.

---

## Database Schema (SQLite)

The application uses SQLite as its metadata persistence layer. The `analysis_results` table is structured as follows:

| Column | Type | Key | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | Primary Key | Autoincrement record ID |
| `filename` | VARCHAR | Index | Original uploaded filename |
| `timestamp` | DATETIME | - | Date and time analysis was completed (UTC) |
| `quality_score` | FLOAT | - | Computed overall quality score [0.0 - 100.0] |
| `quality_label` | VARCHAR | - | Final assessment tag (`ACCEPTABLE`, `DEGRADED`, `DEFECTIVE`) |
| `original_image_path` | VARCHAR | - | Local relative file path of original upload |
| `heatmap_image_path` | VARCHAR | - | Local relative path of Grad-CAM overlay image |
| `issues` | JSON | - | Array of detected issues (type, severity, confidence) |
| `features` | JSON | - | Key-value dictionary of raw statistical parameters |
