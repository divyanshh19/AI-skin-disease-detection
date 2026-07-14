from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import PredictionResponse
from app.services.preprocessing import preprocess_for_inference
from app.services.inference import inference_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(status_code=415,
                             detail=f"Unsupported content type: {file.content_type}. "
                                    f"Allowed: {settings.allowed_content_types}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(status_code=413,
                             detail=f"File too large ({size_mb:.1f}MB). "
                                    f"Max allowed: {settings.max_upload_size_mb}MB")

    if not inference_service.is_ready:
        raise HTTPException(status_code=503,
                             detail="Models are not loaded. Check server health endpoint.")

    try:
        input_tensor, display_rgb_float = preprocess_for_inference(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = inference_service.predict(input_tensor, display_rgb_float, generate_explanation=True)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictionResponse(**result)
