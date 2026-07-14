"""
Thin wrapper around ml/preprocessing/cv_pipeline.py for use inside the API layer.
Kept separate from the ml/ package so the API doesn't need to import training code.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml"))

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from preprocessing.cv_pipeline import remove_hair, denoise, enhance_contrast_clahe

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_for_inference(image_bytes: bytes):
    """
    Returns:
      input_tensor: (1, 3, 224, 224) normalized tensor, ready for the model
      display_rgb_float: (224, 224, 3) float32 [0,1] RGB image for Grad-CAM overlay
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Ensure it's a valid JPEG/PNG file.")

    cleaned = remove_hair(img_bgr)
    cleaned = denoise(cleaned)
    cleaned = enhance_contrast_clahe(cleaned)

    cleaned_resized = cv2.resize(cleaned, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(cleaned_resized, cv2.COLOR_BGR2RGB)
    display_rgb_float = rgb.astype(np.float32) / 255.0

    pil_img = Image.fromarray(rgb)
    input_tensor = _inference_transform(pil_img).unsqueeze(0)

    return input_tensor, display_rgb_float
