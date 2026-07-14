"""
Grad-CAM explainability service.

Uses the `grad-cam` (pytorch-grad-cam) package. For the CNN (EfficientNet-B0),
we target the last convolutional block. For the ViT (DeiT-tiny), Grad-CAM
requires a reshape_transform since transformer activations are token sequences,
not spatial feature maps - we use the standard patch-token reshape approach.
"""
import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def _vit_reshape_transform(tensor, height=14, width=14):
    """Reshape DeiT-tiny's (batch, tokens, dim) output into a spatial (batch, dim, H, W)
    map for Grad-CAM, dropping the [CLS] token (first token)."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_gradcam_cnn(model: torch.nn.Module, input_tensor: torch.Tensor,
                          rgb_img_float: np.ndarray, target_class: int) -> np.ndarray:
    """rgb_img_float must be HxWx3, float32, values in [0,1] (the un-normalized
    display image, for overlay purposes)."""
    target_layers = [model.features[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)


def generate_gradcam_vit(model: torch.nn.Module, input_tensor: torch.Tensor,
                          rgb_img_float: np.ndarray, target_class: int) -> np.ndarray:
    """model here is the ViTWithHead wrapper; we target the last transformer block
    of the backbone and use the patch-token reshape transform."""
    target_layers = [model.backbone.blocks[-1].norm1]
    with GradCAM(model=model, target_layers=target_layers,
                 reshape_transform=_vit_reshape_transform) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)


def overlay_to_base64(overlay_rgb_uint8: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB array as a base64 PNG string."""
    img = Image.fromarray(overlay_rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
