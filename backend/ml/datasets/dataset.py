"""
PyTorch Dataset for HAM10000 (+ optional ISIC subset augmentation).

Expects:
  data/ham10000/metadata.csv   (columns: image_id, dx, image_path, ...)
  data/ham10000/images/*.jpg

Handles:
  - Stratified train/val/test split (by dx, patient-aware where lesion_id exists,
    to avoid leakage of the same lesion across splits)
  - Class-balanced sampling (via WeightedRandomSampler) to counter HAM10000's
    heavy skew toward 'nv' (melanocytic nevi, ~67% of the data)
  - Applies the OpenCV preprocessing pipeline once (cached), then torchvision
    augmentation transforms per-epoch
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cv_pipeline import remove_hair, denoise, enhance_contrast_clahe
import cv2

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_splits(metadata_csv: str, test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(metadata_csv)
    # group by lesion_id if present, to avoid the same lesion appearing in both
    # train and test (HAM10000 has multiple images per lesion)
    group_col = "lesion_id" if "lesion_id" in df.columns else "image_id"
    groups = df[group_col].unique()

    train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=random_state)
    train_groups, val_groups = train_test_split(train_groups, test_size=val_size / (1 - test_size),
                                                 random_state=random_state)

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    return train_df, val_df, test_df


class SkinLesionDataset(Dataset):
    """
    Applies deterministic OpenCV cleanup (hair removal, denoise, CLAHE) at
    __getitem__ time, then hands off to torchvision transforms (resize,
    augmentation, tensor conversion, normalization).
    """

    def __init__(self, df: pd.DataFrame, images_root: str, target_size: int = 224,
                 train: bool = False, apply_cv_pipeline: bool = True):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.target_size = target_size
        self.apply_cv_pipeline = apply_cv_pipeline

        if train:
            self.transform = T.Compose([
                T.Resize((target_size, target_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((target_size, target_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, os.path.basename(row["image_path"]))
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        if self.apply_cv_pipeline:
            img_bgr = remove_hair(img_bgr)
            img_bgr = denoise(img_bgr)
            img_bgr = enhance_contrast_clahe(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img)

        label = CLASS_TO_IDX[row["dx"]]
        return tensor, label

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights per class, for WeightedRandomSampler or loss weighting."""
        counts = self.df["dx"].value_counts().reindex(CLASSES, fill_value=0)
        weights = 1.0 / counts.replace(0, 1)
        weights = weights / weights.sum()
        return torch.tensor(weights.values, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        """Per-sample weight for WeightedRandomSampler, balancing class frequency each epoch."""
        counts = self.df["dx"].value_counts()
        w = self.df["dx"].apply(lambda c: 1.0 / counts[c]).values
        return w


def make_weighted_sampler(dataset: SkinLesionDataset) -> WeightedRandomSampler:
    weights = dataset.sample_weights()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
