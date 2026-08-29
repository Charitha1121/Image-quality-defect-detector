import os
import uuid
import shutil
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import text
from backend.app.database import engine, Base, get_db
from backend.app.models import AnalysisResult
from backend.app.schemas import AnalysisResponse
from backend.app.pipeline import (
    analyze_image_pipeline,
    load_models,
    UPLOAD_DIR,
    HEATMAP_DIR,
    CLASSICAL_MODEL_PATH,
    CNN_MODEL_PATH,
    classical_model_data,
    cnn_model
)

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Image Quality & Defect Detection API",
    description="REST API to analyze images for sharpness, brightness, noise, corruption, and local defects.",
    version="1.0.0"
)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media upload folders
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

# Helper to check if file type is allowed
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    """
    Endpoint verifying backend, database, and machine learning models status.
    """
    health_status = {
        "status": "healthy",
        "database": "connected",
        "models": {
            "classical_loaded": classical_model_data is not None,
            "cnn_loaded": cnn_model is not None,
            "mode": "hybrid" if (classical_model_data and cnn_model) else "heuristic_fallback"
        }
    }

    # Check DB
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"

    return health_status

@app.post("/api/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded image, validates it, runs the quality pipeline,
    persists result in SQLite, and returns analysis report.
    """
    # 1. Validate MIME Type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Only JPEG, PNG and WebP are allowed."
        )

    # 2. Save file temporarily to read and validate format integrity
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    if not file_ext:
        file_ext = ".jpg"

    temp_filename = f"{file_id}{file_ext}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # 3. Validate image integrity (corruption check)
    import cv2
    img_check = cv2.imread(temp_filepath)
    if img_check is None:
        # Delete file if invalid
        os.remove(temp_filepath)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupt or not a readable image."
        )

    # 4. Run Analysis Pipeline
    try:
        report = analyze_image_pipeline(temp_filepath, file.filename)
    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing image: {str(e)}"
        )

    # 5. Persist to DB
    analysis_record = AnalysisResult(
        filename=file.filename,
        quality_score=report["quality_score"],
        quality_label=report["quality_label"],
        original_image_path=report["original_relative_path"],
        heatmap_image_path=report["heatmap_relative_path"],
        issues=report["issues"],
        features=report["features"]
    )

    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    return AnalysisResponse(
        id=analysis_record.id,
        filename=analysis_record.filename,
        timestamp=analysis_record.timestamp,
        quality_score=analysis_record.quality_score,
        quality_label=analysis_record.quality_label,
        issues=analysis_record.issues,
        features=analysis_record.features,
        original_image_url=analysis_record.original_image_path,
        heatmap_url=analysis_record.heatmap_image_path
    )

@app.get("/api/results", response_model=List[AnalysisResponse])
def get_results(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    label: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieves previous analysis results sorted by date (newest first).
    Allows filtering by label (ACCEPTABLE, DEGRADED, DEFECTIVE).
    """
    offset = (page - 1) * limit
    query = db.query(AnalysisResult)

    if label:
        query = query.filter(AnalysisResult.quality_label == label.upper())

    results = query.order_by(AnalysisResult.timestamp.desc()).offset(offset).limit(limit).all()

    response = []
    for r in results:
        response.append(AnalysisResponse(
            id=r.id,
            filename=r.filename,
            timestamp=r.timestamp,
            quality_score=r.quality_score,
            quality_label=r.quality_label,
            issues=r.issues,
            features=r.features,
            original_image_url=r.original_image_path,
            heatmap_url=r.heatmap_image_path
        ))
    return response

@app.get("/api/results/{result_id}", response_model=AnalysisResponse)
def get_result_by_id(result_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single detailed analysis result by its unique ID.
    """
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result with ID {result_id} not found."
        )

    return AnalysisResponse(
        id=result.id,
        filename=result.filename,
        timestamp=result.timestamp,
        quality_score=result.quality_score,
        quality_label=result.quality_label,
        issues=result.issues,
        features=result.features,
        original_image_url=result.original_image_path,
        heatmap_url=result.heatmap_image_path
    )

# Mount frontend files at the root route AFTER API routes are defined
# This ensures API endpoints are correctly intercepted and frontend handles other routes.
FRONTEND_DIR = "frontend"
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: Frontend folder '{FRONTEND_DIR}' does not exist yet. Root path won't serve HTML.")