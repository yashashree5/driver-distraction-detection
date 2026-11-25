"""Models for every experiment.

    from models import build_model
    model = build_model("baseline")

Member 2 owns BaselineCNN. Member 3 adds ResNet-18 and MobileNetV2 here and
registers them in MODELS, so train.py and eval.py work for all of them.
"""
from __future__ import annotations

import torch
from torch import nn

NUM_CLASSES = 10


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm and ReLU, then halve the image size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch (Experiment 1).

    Four conv blocks (32, 64, 128, 256 channels), global average pooling,
    dropout, then one linear layer. Input is (N, 3, 224, 224).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


# name -> function that returns a fresh model; Member 3 adds "resnet18" etc.
MODELS = {
    "baseline": BaselineCNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
