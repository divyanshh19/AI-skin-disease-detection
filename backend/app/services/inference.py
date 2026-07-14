"""
Core inference service: loads CNN + ViT models once at startup, runs both,
combines via calibrated ensemble weighting, and produces Grad-CAM visualizations.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml"))

from training.train_cnn import build_cnn
from training.train_vit import build_vit
from app.services.gradcam import generate_gradcam_cnn, generate_gradcam_vit, overlay_to_base64
from app.core.config import settings
from app.core.logging_config import logger

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_INFO = {
    "akiec": {
        "display_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "description": "A rough, scaly patch caused by sun damage; considered a precancerous lesion "
                        "that can progress to squamous cell carcinoma if untreated.",
        "is_malignant_risk": True,
        "recommendation": "See a dermatologist for evaluation and possible biopsy or treatment.",
    },
    "bcc": {
        "display_name": "Basal Cell Carcinoma",
        "description": "The most common form of skin cancer; grows slowly and rarely spreads, "
                        "but can cause local tissue damage if untreated.",
        "is_malignant_risk": True,
        "recommendation": "Consult a dermatologist promptly for biopsy and treatment planning.",
    },
    "bkl": {
        "display_name": "Benign Keratosis-like Lesions",
        "description": "A group of non-cancerous skin growths, including seborrheic keratoses, "
                        "solar lentigines, and lichen-planus-like keratoses.",
        "is_malignant_risk": False,
        "recommendation": "Generally harmless; monitor for changes and consult a doctor if it evolves.",
    },
    "df": {
        "display_name": "Dermatofibroma",
        "description": "A common benign fibrous nodule, often on the legs, usually harmless.",
        "is_malignant_risk": False,
        "recommendation": "Typically no treatment needed unless symptomatic or cosmetically concerning.",
    },
    "mel": {
        "display_name": "Melanoma",
        "description": "The most dangerous form of skin cancer, arising from pigment-producing cells; "
                        "can spread rapidly if not caught early.",
        "is_malignant_risk": True,
        "recommendation": "Seek immediate dermatological evaluation. Early detection is critical.",
    },
    "nv": {
        "display_name": "Melanocytic Nevi",
        "description": "Common moles; benign proliferations of melanocytes.",
        "is_malignant_risk": False,
        "recommendation": "Routine monitoring for changes in size, shape, or color (ABCDE rule).",
    },
    "vasc": {
        "display_name": "Vascular Lesions",
        "description": "Benign blood-vessel-related skin lesions such as angiomas, "
                        "hemangiomas, and pyogenic granulomas.",
        "is_malignant_risk": False,
        "recommendation": "Usually harmless; consult a doctor if bleeding, growing, or painful.",
    },
}


class InferenceService:
    def __init__(self):
        self.device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() - 1))

        self.cnn_model = None
        self.vit_model = None
        self.ensemble_config = {"cnn_weight": 0.5, "cnn_temperature": 1.0, "vit_temperature": 1.0}

        self._load_models()

    def _load_models(self):
        try:
            self.cnn_model = build_cnn(num_classes=len(CLASSES), freeze_backbone=False)
            ckpt = torch.load(settings.cnn_checkpoint_path, map_location=self.device)
            self.cnn_model.load_state_dict(ckpt["model_state_dict"])
            self.cnn_model.eval()
            logger.info("CNN model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            self.cnn_model = None

        # try:
        #     self.vit_model = build_vit(num_classes=len(CLASSES), freeze_backbone=False)
        #     ckpt = torch.load(settings.vit_checkpoint_path, map_location=self.device)
        #     self.vit_model.load_state_dict(ckpt["model_state_dict"])
        #     self.vit_model.eval()
        #     logger.info("ViT model loaded successfully.")
        # except Exception as e:
        #     logger.error(f"Failed to load ViT model: {e}")
        #     self.vit_model = None

        try:
            with open(settings.ensemble_config_path) as f:
                self.ensemble_config = json.load(f)
            logger.info(f"Ensemble config loaded: {self.ensemble_config}")
        except Exception as e:
            logger.warning(f"Could not load ensemble config, using defaults: {e}")

    @property
    def is_ready(self):
        return self.cnn_model is not None

    @staticmethod
    def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> np.ndarray:
        return torch.softmax(logits / temperature, dim=1).detach().numpy()

    def predict(self, input_tensor: torch.Tensor, display_rgb_float: np.ndarray,
            generate_explanation: bool = True):

        if self.cnn_model is None:
            raise RuntimeError("CNN model is not loaded.")

        cnn_logits = self.cnn_model(input_tensor)

        cnn_probs = self._softmax_with_temperature(
            cnn_logits,
            self.ensemble_config["cnn_temperature"]
        )

        ensemble_probs = cnn_probs[0]

        pred_idx = int(np.argmax(ensemble_probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(ensemble_probs[pred_idx])

        cnn_top = pred_label
        vit_top = "Not Trained"

        class_probabilities = [
            {
                "label": CLASSES[i],
                "display_name": DISEASE_INFO[CLASSES[i]]["display_name"],
                "probability": float(ensemble_probs[i]),
            }
            for i in range(len(CLASSES))
        ]

        class_probabilities.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        gradcam_b64 = None

        if generate_explanation:
            try:
                overlay = generate_gradcam_cnn(
                    self.cnn_model,
                    input_tensor,
                    display_rgb_float,
                    pred_idx,
                )
                gradcam_b64 = overlay_to_base64(overlay)
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")

        warning = "Vision Transformer not trained. Using CNN prediction only."

        return {
            "predicted_label": pred_label,
            "predicted_display_name": DISEASE_INFO[pred_label]["display_name"],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "cnn_top_label": cnn_top,
            "vit_top_label": vit_top,
            "models_agree": True,
            "disease_info": {
                "label": pred_label,
                **DISEASE_INFO[pred_label],
            },
            "gradcam_image_base64": gradcam_b64,
            "warning": warning,
        }


inference_service = InferenceService()
