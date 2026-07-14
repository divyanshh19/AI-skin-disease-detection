"""Pydantic request/response models for the API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str
    display_name: str
    probability: float


class DiseaseInfo(BaseModel):
    label: str
    display_name: str
    description: str
    is_malignant_risk: bool
    recommendation: str


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_display_name: str
    confidence: float = Field(..., description="Calibrated confidence of the top prediction, 0-1")
    class_probabilities: List[ClassProbability]
    cnn_top_label: str
    vit_top_label: str
    models_agree: bool
    disease_info: DiseaseInfo
    gradcam_image_base64: Optional[str] = Field(
        None, description="Base64-encoded PNG of the Grad-CAM overlay")
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    cnn_model_loaded: bool
    vit_model_loaded: bool
    ensemble_config_loaded: bool
    version: str = "1.0.0"
