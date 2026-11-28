"""Models for every experiment.

    from models import build_model
    model = build_model("baseline")

Member 2 owns BaselineCNN. Member 3 adds ResNet-18 and MobileNetV2 here and
registers them in MODELS, so train.py and eval.py work for all of them.
"""
from __future__ import annotations

import torch
import time
from torch import nn
from torchvision import models

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


class TransferLearningModel(nn.Module):
    """Mixin helpers for pretrained torchvision classifiers."""

    backbone_lr_multiplier = 0.1

    def backbone_parameters(self):
        return ()

    def head_parameters(self):
        return ()

    def parameter_groups(self, lr: float, weight_decay: float = 1e-4):
        backbone = list(self.backbone_parameters())
        head = list(self.head_parameters())
        if not backbone:
            return [{"params": head, "lr": lr, "weight_decay": weight_decay}]
        head_ids = {id(p) for p in head}
        backbone = [p for p in backbone if id(p) not in head_ids]
        return [
            {"params": backbone, "lr": lr * self.backbone_lr_multiplier,
             "weight_decay": weight_decay},
            {"params": head, "lr": lr, "weight_decay": weight_decay},
        ]


class ResNet18(TransferLearningModel):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.backbone.fc.parameters()}
        return (p for p in self.backbone.parameters()
                if id(p) not in head_ids and p.requires_grad)

    def head_parameters(self):
        return (p for p in self.backbone.fc.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetV2(TransferLearningModel):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.backbone.classifier.parameters()}
        return (p for p in self.backbone.parameters()
                if id(p) not in head_ids and p.requires_grad)

    def head_parameters(self):
        return (p for p in self.backbone.classifier.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    frozen: bool = True,
    fine_tuned: bool | None = None,
) -> nn.Module:
    if fine_tuned is not None:
        frozen = not fine_tuned
    return ResNet18(num_classes=num_classes, pretrained=pretrained, frozen=frozen)


def mobilenet_v2(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    frozen: bool = True,
    fine_tuned: bool | None = None,
) -> nn.Module:
    if fine_tuned is not None:
        frozen = not fine_tuned
    return MobileNetV2(num_classes=num_classes, pretrained=pretrained, frozen=frozen)


# name -> function that returns a fresh model.
MODELS = {
    "baseline": BaselineCNN,
    "resnet18": resnet18,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet": mobilenet_v2,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate serialized parameter size, including no checkpoint metadata."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Return average milliseconds per image on the requested device."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    model.to(device)
    model.eval()
    sample = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(sample)
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            model(sample)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        started = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms / iterations / batch_size
