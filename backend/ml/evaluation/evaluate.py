"""
Evaluate a trained model (CNN or ViT) or an ensemble of both on the held-out
test split. Computes accuracy, precision, recall, F1 (macro and per-class),
confusion matrix, and multi-class ROC-AUC (one-vs-rest).

Usage:
  python evaluate.py --data-root ../../../data/ham10000 --model cnn --checkpoint ../saved_models/cnn_efficientnet_b0.pth
  python evaluate.py --data-root ../../../data/ham10000 --model ensemble \
      --cnn-checkpoint ../saved_models/cnn_efficientnet_b0.pth \
      --vit-checkpoint ../saved_models/vit_deit_tiny.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, build_splits, CLASSES
from training.train_cnn import build_cnn
from training.train_vit import build_vit


def load_model(kind, checkpoint_path, device):
    if kind == "cnn":
        model = build_cnn(num_classes=len(CLASSES), freeze_backbone=False)
    elif kind == "vit":
        model = build_vit(num_classes=len(CLASSES), freeze_backbone=False)
    else:
        raise ValueError(kind)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def get_probs(model, loader, device):
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(y_true, probs, class_names, title, out_dir):
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== {title} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    try:
        y_true_onehot = np.eye(len(class_names))[y_true]
        roc_auc = roc_auc_score(y_true_onehot, probs, average="macro", multi_class="ovr")
        print(f"ROC-AUC (macro, OvR): {roc_auc:.4f}")
    except ValueError as e:
        roc_auc = None
        print(f"ROC-AUC could not be computed: {e}")

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"confusion_matrix_{title.replace(' ', '_').lower()}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def main(args):
    device = torch.device("cpu")
    metadata_csv = os.path.join(args.data_root, "metadata.csv")
    images_root = os.path.join(args.data_root, "images")
    _, _, test_df = build_splits(metadata_csv)
    test_ds = SkinLesionDataset(test_df, images_root, target_size=224, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.model == "cnn":
        model = load_model("cnn", args.checkpoint, device)
        probs, labels = get_probs(model, test_loader, device)
        compute_metrics(labels, probs, CLASSES, "CNN (EfficientNet-B0)", args.out_dir)

    elif args.model == "vit":
        model = load_model("vit", args.checkpoint, device)
        probs, labels = get_probs(model, test_loader, device)
        compute_metrics(labels, probs, CLASSES, "ViT (DeiT-tiny)", args.out_dir)

    elif args.model == "ensemble":
        cnn_model = load_model("cnn", args.cnn_checkpoint, device)
        vit_model = load_model("vit", args.vit_checkpoint, device)
        cnn_probs, labels = get_probs(cnn_model, test_loader, device)
        vit_probs, _ = get_probs(vit_model, test_loader, device)

        compute_metrics(labels, cnn_probs, CLASSES, "CNN (EfficientNet-B0)", args.out_dir)
        compute_metrics(labels, vit_probs, CLASSES, "ViT (DeiT-tiny)", args.out_dir)

        ensemble_probs = args.cnn_weight * cnn_probs + (1 - args.cnn_weight) * vit_probs
        compute_metrics(labels, ensemble_probs, CLASSES,
                         f"Ensemble (cnn_weight={args.cnn_weight})", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../../../data/ham10000")
    parser.add_argument("--model", choices=["cnn", "vit", "ensemble"], required=True)
    parser.add_argument("--checkpoint", type=str, help="For --model cnn or vit")
    parser.add_argument("--cnn-checkpoint", type=str, default="../saved_models/cnn_efficientnet_b0.pth")
    parser.add_argument("--vit-checkpoint", type=str, default="../saved_models/vit_deit_tiny.pth")
    parser.add_argument("--cnn-weight", type=float, default=0.5,
                         help="Weight given to CNN in ensemble average (ViT gets 1-weight)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="../evaluation/results")
    args = parser.parse_args()
    main(args)
