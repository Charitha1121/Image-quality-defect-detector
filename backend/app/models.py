import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from backend.app.database import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    quality_score = Column(Float)
    quality_label = Column(String)  # ACCEPTABLE, DEGRADED, DEFECTIVE
    
    original_image_path = Column(String)
    heatmap_image_path = Column(String, nullable=True)
    
    # Store issues list and raw features as JSON
    issues = Column(JSON)
    features = Column(JSON)
