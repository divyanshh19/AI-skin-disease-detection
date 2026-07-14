"""Shared training utilities: CPU-friendly training loop, early stopping, checkpointing."""
import os
import time
import torch
from torch import nn
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """One epoch of train (optimizer given) or eval (optimizer=None)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, desc="train" if is_train else "eval", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return total_loss / total_samples, total_correct / total_samples


def train_model(model, train_loader, val_loader, epochs, lr, device, checkpoint_path,
                 class_weights=None, patience=5):
    """Full training loop with early stopping and best-checkpoint saving.
    Uses CPU thread tuning since this is designed to run without a GPU."""
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "epoch": epoch}, checkpoint_path)
            print(f"  -> saved new best checkpoint to {checkpoint_path}")

        if stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return history
