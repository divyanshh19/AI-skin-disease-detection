"""
Train a CNN (EfficientNet-B0, ImageNet-pretrained) on HAM10000 via transfer learning.

CPU strategy: freeze the backbone entirely, train only a new classifier head
for the first N epochs, then optionally unfreeze the last block for fine-tuning
at a low LR. This keeps the backward pass cheap on CPU.

Usage:
  python train_cnn.py --data-root ../../../data/ham10000 --epochs 10
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import SkinLesionDataset, build_splits, make_weighted_sampler, CLASSES
from training.train_utils import train_model


def build_cnn(num_classes=7, freeze_backbone=True):
    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_last_block(model):
    """Unfreeze the final MBConv block of EfficientNet-B0 for light fine-tuning."""
    for param in model.features[-1].parameters():
        param.requires_grad = True
    return model


def main(args):
    device = torch.device("cpu")
    metadata_csv = os.path.join(args.data_root, "metadata.csv")
    images_root = os.path.join(args.data_root, "images")

    train_df, val_df, test_df = build_splits(metadata_csv)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_ds = SkinLesionDataset(train_df, images_root, target_size=224, train=True,
                                  apply_cv_pipeline=not args.skip_cv)
    val_ds = SkinLesionDataset(val_df, images_root, target_size=224, train=False,
                                apply_cv_pipeline=not args.skip_cv)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    model = build_cnn(num_classes=len(CLASSES), freeze_backbone=True).to(device)
    class_weights = train_ds.class_weights()

    print("=== Phase 1: training classifier head only ===")
    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=1e-3,
                device=device, checkpoint_path=args.checkpoint, class_weights=class_weights,
                patience=args.patience)

    if args.fine_tune_epochs > 0:
        print("=== Phase 2: fine-tuning last block at low LR ===")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = unfreeze_last_block(model)
        train_model(model, train_loader, val_loader, epochs=args.fine_tune_epochs, lr=1e-5,
                    device=device, checkpoint_path=args.checkpoint, class_weights=class_weights,
                    patience=args.patience)

    print(f"Best CNN checkpoint saved at: {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../../../data/ham10000")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="../saved_models/cnn_efficientnet_b0.pth")
    parser.add_argument("--skip-cv", action="store_true", help="Skip OpenCV pipeline (faster, for debugging)")
    args = parser.parse_args()
    main(args)
