"""
Finds the optimal ensemble weight (cnn_weight in [0,1]) on the validation set
by grid search, maximizing macro F1. Also fits temperature scaling per model
for better-calibrated confidence scores (so "85% confidence" actually means
~85% empirical accuracy).

Usage:
  python train_ensemble_calibration.py --data-root ../../../data/ham10000
Outputs:
  ../saved_models/ensemble_config.json   {"cnn_weight": 0.55, "cnn_temperature": 1.3, "vit_temperature": 1.1}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, build_splits, CLASSES
from training.train_cnn import build_cnn
from training.train_vit import build_vit


@torch.no_grad()
def get_logits(model, loader, device):
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        all_logits.append(model(x).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Simple 1D temperature scaling fit via gradient descent on NLL."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def softmax(x, axis=1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main(args):
    device = torch.device("cpu")
    metadata_csv = os.path.join(args.data_root, "metadata.csv")
    images_root = os.path.join(args.data_root, "images")
    _, val_df, _ = build_splits(metadata_csv)
    val_ds = SkinLesionDataset(val_df, images_root, target_size=224, train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    cnn = build_cnn(num_classes=len(CLASSES), freeze_backbone=False)
    cnn.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device)["model_state_dict"])
    cnn.eval()

    vit = build_vit(num_classes=len(CLASSES), freeze_backbone=False)
    vit.load_state_dict(torch.load(args.vit_checkpoint, map_location=device)["model_state_dict"])
    vit.eval()

    cnn_logits, labels = get_logits(cnn, val_loader, device)
    vit_logits, _ = get_logits(vit, val_loader, device)

    cnn_temp = fit_temperature(cnn_logits, labels)
    vit_temp = fit_temperature(vit_logits, labels)
    print(f"Fitted temperatures -> CNN: {cnn_temp:.3f}, ViT: {vit_temp:.3f}")

    cnn_probs = softmax(cnn_logits / cnn_temp)
    vit_probs = softmax(vit_logits / vit_temp)

    best_weight, best_f1 = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.05):
        ensemble_probs = w * cnn_probs + (1 - w) * vit_probs
        preds = ensemble_probs.argmax(axis=1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weight = w

    print(f"Best ensemble cnn_weight={best_weight:.2f} (macro F1={best_f1:.4f})")

    config = {"cnn_weight": float(best_weight), "cnn_temperature": cnn_temp, "vit_temperature": vit_temp}
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved ensemble config to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../../../data/ham10000")
    parser.add_argument("--cnn-checkpoint", type=str, default="../saved_models/cnn_efficientnet_b0.pth")
    parser.add_argument("--vit-checkpoint", type=str, default="../saved_models/vit_deit_tiny.pth")
    parser.add_argument("--out", type=str, default="../saved_models/ensemble_config.json")
    args = parser.parse_args()
    main(args)
