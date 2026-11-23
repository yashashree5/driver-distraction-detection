import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

import data

IMAGES_PER_SPLIT = {"train": 12, "val": 4, "test": 6}
LABEL_STEP = 25


@pytest.fixture
def tiny_dataset(tmp_path):
    """A few small images on disk plus a split file that points at them.

    The red value of each image is its label times 25, so a test can read the
    label back from the pixels.
    """
    data_dir = tmp_path / "data"
    rows = []
    img_id = 0
    for split, n_images in IMAGES_PER_SPLIT.items():
        for i in range(n_images):
            label = i % 10
            rel_path = f"imgs/train/c{label}/img_{img_id}.jpg"
            full_path = data_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (label * LABEL_STEP, 120, 200)).save(full_path)
            rows.append({"path": rel_path, "classname": f"c{label}", "label": label,
                         "subject": f"p{img_id % 3:03d}", "split": split})
            img_id += 1
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    return split_csv, data_dir


# transforms

def test_eval_transform_shape_and_type():
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    tensor = data.build_transforms(train=False)(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_shape():
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    assert data.build_transforms(train=True)(image).shape == (3, 224, 224)


def test_eval_transform_uses_imagenet_normalisation():
    image = Image.new("RGB", (640, 480), (255, 0, 127))
    tensor = data.build_transforms(train=False)(image)
    expected = (torch.tensor([255, 0, 127]) / 255 - torch.tensor(data.IMAGENET_MEAN)) / torch.tensor(data.IMAGENET_STD)
    assert torch.allclose(tensor.mean(dim=(1, 2)), expected, atol=1e-4)


def test_eval_transform_is_not_random():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    transform = data.build_transforms(train=False)
    assert torch.equal(transform(image), transform(image))


def test_no_flip_in_the_transform_list():
    for train in (True, False):
        steps = data.build_transforms(train=train).transforms
        assert not any(isinstance(s, (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip)) for s in steps)


def test_train_transform_never_swaps_left_and_right():
    # left half white, right half black: a flip would make the right side the bright one
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[:, :320] = 255
    image = Image.fromarray(pixels)
    transform = data.build_transforms(train=True)
    torch.manual_seed(0)
    for _ in range(50):
        tensor = transform(image)
        assert tensor[:, :, :112].mean() > tensor[:, :, 112:].mean()


# dataset

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_length(tiny_dataset, split):
    split_csv, data_dir = tiny_dataset
    assert len(data.DriverDataset(split_csv, split, data_dir)) == IMAGES_PER_SPLIT[split]


def test_dataset_item(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, "val", data_dir)
    image, label = dataset[1]
    assert image.shape == (3, 224, 224)
    assert label == 1
    assert isinstance(label, int)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_pairs_each_image_with_its_own_label(tiny_dataset, split):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, split, data_dir, transform=transforms.ToTensor())
    for image, label in dataset:
        red = image[0].mean().item() * 255
        assert round(red / LABEL_STEP) == label


def test_val_images_are_not_augmented(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, "val", data_dir)
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_train_images_are_augmented(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, "train", data_dir)
    torch.manual_seed(0)
    assert not torch.equal(dataset[0][0], dataset[0][0])


def test_dataset_keeps_the_rows_of_its_split(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, "test", data_dir)
    assert set(dataset.samples["split"]) == {"test"}
    assert {"path", "label", "subject"} <= set(dataset.samples.columns)


def test_dataset_uses_a_custom_transform(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    dataset = data.DriverDataset(split_csv, "val", data_dir, transform=transforms.ToTensor())
    image, _ = dataset[0]
    assert image.shape == (3, 48, 64)


def test_dataset_rejects_unknown_split(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    with pytest.raises(ValueError, match="validation"):
        data.DriverDataset(split_csv, "validation", data_dir)


def test_dataset_rejects_missing_column(tiny_dataset, tmp_path):
    split_csv, data_dir = tiny_dataset
    broken = tmp_path / "broken.csv"
    pd.read_csv(split_csv).drop(columns="label").to_csv(broken, index=False)
    with pytest.raises(ValueError, match="label"):
        data.DriverDataset(broken, "train", data_dir)


def test_dataset_reports_missing_images(tiny_dataset, tmp_path):
    split_csv, _ = tiny_dataset
    with pytest.raises(FileNotFoundError, match="12 of 12"):
        data.DriverDataset(split_csv, "train", tmp_path / "nowhere")


# loaders

def test_make_loaders_returns_all_three_splits(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    loaders = data.make_loaders(split_csv, data_dir, batch_size=4, num_workers=0)
    assert set(loaders) == {"train", "val", "test"}
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64


def test_val_and_test_loaders_keep_file_order(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    loaders = data.make_loaders(split_csv, data_dir, batch_size=4, num_workers=0)
    for split in ("val", "test"):
        labels = torch.cat([batch_labels for _, batch_labels in loaders[split]]).tolist()
        assert labels == loaders[split].dataset.samples["label"].tolist()


def test_train_loader_order_depends_only_on_the_seed(tiny_dataset):
    split_csv, data_dir = tiny_dataset

    def label_order(seed):
        loader = data.make_loaders(split_csv, data_dir, batch_size=4, num_workers=0, seed=seed)["train"]
        return torch.cat([batch_labels for _, batch_labels in loader]).tolist()

    assert label_order(42) == label_order(42)
    assert label_order(42) != label_order(7)


def test_loaders_with_workers_repeat_for_the_same_seed(tiny_dataset):
    # this is the Colab setting: the batches come from worker processes
    split_csv, data_dir = tiny_dataset

    def first_batch():
        loader = data.make_loaders(split_csv, data_dir, batch_size=4, num_workers=2)["train"]
        return next(iter(loader))

    images_a, labels_a = first_batch()
    images_b, labels_b = first_batch()
    assert torch.equal(labels_a, labels_b)
    assert torch.equal(images_a, images_b)


def test_train_loader_is_shuffled(tiny_dataset):
    split_csv, data_dir = tiny_dataset
    loader = data.make_loaders(split_csv, data_dir, batch_size=4, num_workers=0)["train"]
    labels = torch.cat([batch_labels for _, batch_labels in loader]).tolist()
    assert sorted(labels) == sorted(loader.dataset.samples["label"].tolist())
    assert labels != loader.dataset.samples["label"].tolist()
