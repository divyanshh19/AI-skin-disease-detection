"""Application configuration, loaded from environment variables / .env file."""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Skin Disease Detection API"
    api_v1_prefix: str = "/api/v1"

    cnn_checkpoint_path: str = os.getenv(
        "CNN_CHECKPOINT_PATH", "ml/saved_models/cnn_efficientnet_b0.pth")
    vit_checkpoint_path: str = os.getenv(
        "VIT_CHECKPOINT_PATH", "ml/saved_models/vit_deit_tiny.pth")
    ensemble_config_path: str = os.getenv(
        "ENSEMBLE_CONFIG_PATH", "ml/saved_models/ensemble_config.json")

    target_image_size: int = 224
    max_upload_size_mb: int = 10
    allowed_content_types: tuple = ("image/jpeg", "image/png", "image/jpg")

    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"


settings = Settings()
