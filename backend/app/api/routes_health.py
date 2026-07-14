from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.inference import inference_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if inference_service.is_ready else "degraded",
        cnn_model_loaded=inference_service.cnn_model is not None,
        vit_model_loaded=inference_service.vit_model is not None,
        ensemble_config_loaded=inference_service.ensemble_config is not None,
    )
