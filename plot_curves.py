"""Learning-curve figures for the transfer-learning runs (Experiment 2).

Reads results/*_driverwise.json and writes:
  figures/transfer_curves.png        train vs val loss/accuracy for each run
  figures/transfer_val_accuracy.png  val accuracy of every run plus the baseline

Usage: python plot_curves.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
FIGURES = Path("figures")

TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
BLUE = "#257de0"    # train / ResNet-18
ORANGE = "#f47230"  # val / MobileNetV2
GRAY = "#8c8c8c"    # baseline reference

RUNS = [
    ("resnet18_frozen_driverwise", "ResNet-18, frozen"),
    ("resnet18_fine_tuned_driverwise", "ResNet-18, fine-tuned"),
    ("mobilenet_v2_frozen_driverwise", "MobileNetV2, frozen"),
    ("mobilenet_v2_fine_tuned_driverwise", "MobileNetV2, fine-tuned"),
]

plt.rcParams.update({
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 11,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def load(tag: str) -> dict:
    return json.loads((RESULTS / f"{tag}.json").read_text())


def per_run_curves() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True, sharey="row")
    for col, (tag, title) in enumerate(RUNS):
        run = load(tag)
        h, test = run["history"], run["test"]
        epochs = range(1, len(h["val_loss"]) + 1)
        best = h["best_epoch"]
        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row, col]
            ax.plot(epochs, h[f"train_{metric}"], color=BLUE, lw=2, label="train")
            ax.plot(epochs, h[f"val_{metric}"], color=ORANGE, lw=2, label="validation")
            ax.plot(best, h[f"val_{metric}"][best - 1], "o", ms=8, color=ORANGE,
                    mec="white", mew=2, zorder=3, label="best epoch (saved)")
            if col == 0:
                ax.set_ylabel("loss" if metric == "loss" else "accuracy")
                # direct labels on the first panel of each row
                for key, color, name in [(f"train_{metric}", BLUE, "train"),
                                         (f"val_{metric}", ORANGE, "val")]:
                    ax.annotate(name, (len(epochs), h[key][-1]), xytext=(4, 0),
                                textcoords="offset points", va="center", color=TEXT, fontsize=9)
            if row == 0:
                ax.set_title(f"{title}\ntest acc {test['accuracy']:.3f} · macro-F1 {test['macro_f1']:.3f}")
            else:
                ax.set_xlabel("epoch")
                ax.set_ylim(0.3, 1.02)
            ax.set_xlim(0.5, 16.5)
            ax.set_xticks([1, 5, 10, 15])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Transfer learning on the driver-wise split: train vs validation",
                 y=1.06, fontsize=13, color=TEXT)
    fig.tight_layout()
    path = FIGURES / "transfer_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


def val_accuracy_comparison() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    series = [(tag, title, BLUE if "resnet" in tag else ORANGE,
               "-" if "fine_tuned" in tag else "--") for tag, title in RUNS]
    series.append(("baseline_driverwise", "Baseline CNN", GRAY, ":"))
    ends = []
    for tag, title, color, style in series:
        run = load(tag)
        val = run["history"]["val_acc"]
        epochs = range(1, len(val) + 1)
        ax.plot(epochs, val, color=color, ls=style, lw=2,
                label=f"{title} (test {run['test']['accuracy']:.3f})")
        ends.append([len(val), val[-1], title])
    # direct labels at line ends, nudged apart so they never overlap
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.035)
    for x, y, title in ends:
        ax.annotate(title, (15.3, y), va="center", color=TEXT, fontsize=9, annotation_clip=False)
    ax.set_xlim(0.5, 15.2)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (held-out drivers)")
    ax.set_title("Validation accuracy by epoch: fine-tuning vs frozen backbone",
                 loc="left", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    fig.tight_layout()
    path = FIGURES / "transfer_val_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    per_run_curves()
    val_accuracy_comparison()
