"""Shared training loop.

    from train import fit
    history = fit(model, loaders, epochs=15, lr=1e-3, device="cuda", ckpt_path="checkpoints/baseline_driverwise.pt")

or from the command line:

    python train.py --model baseline --split splits/driverwise.csv --epochs 15
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import SEED, make_loaders
from eval import evaluate, summarize
from models import build_model, count_parameters, model_size_mb


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit(
    model: nn.Module,
    loaders: dict,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    ckpt_path: str | Path | None = None,
    patience: int = 5,
) -> dict:
    """Train with AdamW and a cosine schedule. Keeps the weights with the best val accuracy.

    Stops early if val accuracy has not improved for `patience` epochs.
    The test loader is never touched here. Returns a history dict of per-epoch lists.
    """
    device = device or get_device()
    model.to(device)
    if hasattr(model, "parameter_groups"):
        param_groups = model.parameter_groups(lr, weight_decay)
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": 0}
    best_acc, stale = -1.0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += len(targets)
        scheduler.step()

        val = evaluate(model, loaders["val"], device)
        history["train_loss"].append(loss_sum / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        print(f"epoch {epoch:2d}/{epochs}  train loss {loss_sum / seen:.3f} acc {correct / seen:.3f}  "
              f"val loss {val['loss']:.3f} acc {val['accuracy']:.3f}  ({time.time() - start:.0f}s)")

        if val["accuracy"] > best_acc:
            best_acc, stale = val["accuracy"], 0
            history["best_epoch"] = epoch
            if ckpt_path:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop: no val improvement for {patience} epochs")
                break

    if ckpt_path and Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--split", default="splits/driverwise.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Fine-tuning backbone learning rate; defaults to 10%% of --lr")
    parser.add_argument("--frozen", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze the pretrained backbone and train only the 10-class head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", default=".",
                        help="Directory for checkpoints/ and results/ (use Google Drive on Colab)")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    transfer_models = {"resnet18", "mobilenet_v2", "mobilenet"}
    if args.model in transfer_models:
        mode = "frozen" if args.frozen else "fine_tuned"
        model = build_model(args.model, pretrained=True, frozen=args.frozen)
        if args.backbone_lr is not None:
            model.backbone_lr_multiplier = args.backbone_lr / args.lr
        tag = f"{args.model}_{mode}_{Path(args.split).stem}"
    else:
        model = build_model(args.model)
        tag = f"{args.model}_{Path(args.split).stem}"
    loaders = make_loaders(args.split, args.data_dir, args.batch_size, args.workers)
    print(f"{tag}: {count_parameters(model):,} trainable parameters, "
          f"{model_size_mb(model):.2f} MB on {device}")

    output_dir = Path(args.output_dir)
    history = fit(model, loaders, args.epochs, args.lr, device=device,
                  ckpt_path=output_dir / "checkpoints" / f"{tag}.pt", patience=args.patience)
    test = summarize(evaluate(model, loaders["test"], device))
    print("test:", test)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{tag}.json").write_text(
        json.dumps({"args": vars(args), "history": history, "test": test}, indent=2)
    )


if __name__ == "__main__":
    main()
