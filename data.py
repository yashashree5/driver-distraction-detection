"""Dataset, transforms and data loaders shared by every experiment.

    from data import make_loaders
    loaders = make_loaders("splits/driverwise.csv", data_dir="data")
    for images, labels in loaders["train"]:
        ...

Everyone uses the same resize, normalisation and augmentation from this file,
so the numbers in the results table can be compared.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
SEED = 42
MAX_FILES_TO_CHECK = 200

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("path", "label", "split")

# the position in this tuple is the numeric label
CLASS_NAMES = (
    "safe driving",
    "texting (right)",
    "phone call (right)",
    "texting (left)",
    "phone call (left)",
    "operating radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
)


def build_transforms(train: bool) -> transforms.Compose:
    """Resize and normalise an image. Training images also get light augmentation.

    There is no horizontal flip on purpose. A flip would turn "texting (right)"
    into "texting (left)" and make the label wrong.
    """
    steps: list[Callable] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class DriverDataset(Dataset):
    """The images of one split (train, val or test) from a split file.

    `samples` keeps the rows of the split file, so the path and driver of
    image i are in `samples.iloc[i]`.
    """

    def __init__(
        self,
        split_csv: str | Path,
        split: str,
        data_dir: str | Path = "data",
        transform: Callable | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")

        frame = pd.read_csv(split_csv)
        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{split_csv} is missing columns: {missing}")

        samples = frame[frame["split"] == split].reset_index(drop=True)
        if samples.empty:
            raise ValueError(f"{split_csv} has no rows for the '{split}' split")

        self.data_dir = Path(data_dir)
        self.samples = samples
        self.paths = [self.data_dir / path for path in samples["path"]]
        self.labels = [int(label) for label in samples["label"]]
        self.transform = transform if transform is not None else build_transforms(train=split == "train")

        # checking every file is slow on a mounted Drive, so look at a spread of them
        step = max(1, len(self.paths) // MAX_FILES_TO_CHECK)
        checked = self.paths[::step]
        not_found = [path for path in checked if not path.exists()]
        if not_found:
            raise FileNotFoundError(
                f"{len(not_found)} of {len(checked)} images checked are missing, for example {not_found[0]}. "
                f"Unzip statefarm_train.zip into '{self.data_dir}' first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


def seed_worker(worker_id: int) -> None:
    """Give each loader worker its own fixed seed for numpy and random."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    split_csv: str | Path,
    data_dir: str | Path = "data",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """Return the train, val and test loaders for one split file.

    Only the train loader is shuffled, and its order depends only on the seed.

    The augmentation is random too. With num_workers > 0 it also follows the
    seed. With num_workers = 0 it follows the global torch seed, so call
    torch.manual_seed(seed) at the start of training.
    """
    loaders = {}
    for split in SPLITS:
        is_train = split == "train"
        loaders[split] = DataLoader(
            DriverDataset(split_csv, split, data_dir),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders
