from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class Issue(BaseModel):
    type: str          # e.g., "blur", "underexposure", "overexposure", "noise", "corruption", "defect"
    severity: str      # e.g., "low", "high", "scratch", "spot", "sensor_line", "none"
    confidence: float  # probability [0.0, 1.0]

class AnalysisResponse(BaseModel):
    id: int
    filename: str
    timestamp: datetime
    quality_score: float
    quality_label: str
    issues: List[Issue]
    features: Dict[str, float]
    original_image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    

    class Config:
        orm_mode = True

    @classmethod
    def from_orm(cls, obj):
        # Handle transformation or base URL mapping if needed
        return super().from_orm(obj)
