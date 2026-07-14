"""
OpenCV preprocessing pipeline for dermoscopic images.

Steps (in order):
  1. Hair removal (DullRazor-style: blackhat morphology + inpainting)
  2. Denoising (Non-local means)
  3. Resize (to target size, preserving aspect ratio via center-crop/pad)
  4. Contrast enhancement (CLAHE on L channel of LAB)
  5. Normalization (to [0,1] float32, then ImageNet mean/std)

This module is used both at training time (as a fixed pre-augmentation step,
before torchvision transforms) and at inference time (in the FastAPI service),
so behavior must be identical in both places.
"""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor-style hair removal:
      - Convert to grayscale
      - Blackhat morphological transform to highlight hair (thin dark structures)
      - Threshold to create a hair mask
      - Inpaint the original image using that mask
    """ 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return inpainted


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Non-local means denoising, tuned for dermoscopic images (mild strength)."""
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=6, hColor=6,
                                            templateWindowSize=7, searchWindowSize=21)


def resize_with_pad(img_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize preserving aspect ratio, padding with black to reach a square target_size."""
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def enhance_contrast_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB color space to boost local contrast
    without blowing out color (important for lesion border visibility)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR->RGB, scale to [0,1], then standardize with ImageNet mean/std.
    Returns float32 array of shape (H, W, 3) in RGB, standardized."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    standardized = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return standardized


def preprocess_image(img_bgr: np.ndarray, target_size: int = 224,
                      return_normalized: bool = True) -> np.ndarray:
    """
    Full pipeline. If return_normalized=False, returns the resized+enhanced
    uint8 BGR image (useful for display / Grad-CAM overlay backgrounds),
    otherwise returns the normalized float32 RGB tensor-ready array.
    """
    img = remove_hair(img_bgr)
    img = denoise(img)
    img = resize_with_pad(img, target_size)
    img = enhance_contrast_clahe(img)
    if return_normalized:
        return normalize(img)
    return img


def load_and_preprocess(path: str, target_size: int = 224) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return preprocess_image(img_bgr, target_size)
