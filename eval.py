"""Metrics and confusion matrix shared by every experiment.

    from eval import evaluate, summarize, plot_confusion_matrix
    result = evaluate(model, loaders["test"], device)
    print(summarize(result))
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from data import CLASS_NAMES


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str | torch.device) -> dict:
    """Run the model over a loader. Returns loss, accuracy, and every label and prediction."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, labels, preds = 0.0, [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        labels.append(targets.cpu())
        preds.append(logits.argmax(1).cpu())
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def summarize(result: dict) -> dict:
    """Headline numbers for the results table."""
    return {
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "macro_f1": float(f1_score(result["y_true"], result["y_pred"], average="macro", zero_division=0)),
    }


def per_class_report(result: dict) -> str:
    return classification_report(
        result["y_true"], result["y_pred"],
        labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0, digits=3,
    )


def accuracy_by_driver(result: dict, subjects) -> dict[str, float]:
    """Accuracy per driver. `subjects` is the subject column of the evaluated split, in loader order."""
    subjects = np.asarray(subjects)
    correct = result["y_true"] == result["y_pred"]
    return {str(s): float(correct[subjects == s].mean()) for s in np.unique(subjects)}


def plot_confusion_matrix(result: dict, path: str | None = None, normalize: bool = True):
    """Draw the confusion matrix. Saves to `path` if given (use figures/...)."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=range(len(CLASS_NAMES)))
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ticks = range(len(CLASS_NAMES))
    ax.set_xticks(ticks, [f"c{i}" for i in ticks])
    ax.set_yticks(ticks, [f"c{i} {n}" for i, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 * cm.max() else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
