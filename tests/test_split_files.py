"""Checks on the two split files that are committed to the repo.

Most of these tests only need the CSV files in splits/. The last one also
needs data/driver_imgs_list.csv and is skipped when the dataset is not there.
"""
from pathlib import Path

import pandas as pd
import pytest

import make_splits as ms

REPO = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO / "splits"
DRIVER_LIST = REPO / "data" / "driver_imgs_list.csv"
COLUMNS = ["path", "classname", "label", "subject", "split"]
N_IMAGES = 22424
N_DRIVERS = 26
DRIVERS_PER_SPLIT = {"train": 18, "val": 4, "test": 4}


@pytest.fixture(scope="module")
def driverwise():
    return pd.read_csv(SPLITS_DIR / "driverwise.csv")


@pytest.fixture(scope="module")
def random_split():
    return pd.read_csv(SPLITS_DIR / "random.csv")


@pytest.mark.parametrize("name", ["driverwise.csv", "random.csv"])
def test_file_is_complete(name):
    frame = pd.read_csv(SPLITS_DIR / name)
    assert list(frame.columns) == COLUMNS
    assert len(frame) == N_IMAGES
    assert frame["path"].is_unique
    assert frame["subject"].nunique() == N_DRIVERS
    assert set(frame["split"]) == {"train", "val", "test"}
    assert not frame.isna().any().any()


@pytest.mark.parametrize("name", ["driverwise.csv", "random.csv"])
def test_labels_match_the_folder(name):
    frame = pd.read_csv(SPLITS_DIR / name)
    assert (frame["classname"] == "c" + frame["label"].astype(str)).all()
    assert (frame["path"].str.split("/").str[2] == frame["classname"]).all()
    assert frame["label"].between(0, 9).all()


def test_driverwise_has_no_driver_in_two_splits(driverwise):
    assert (driverwise.groupby("subject")["split"].nunique() == 1).all()


def test_driverwise_driver_counts(driverwise):
    assert driverwise.groupby("split")["subject"].nunique().to_dict() == DRIVERS_PER_SPLIT


def test_driverwise_every_class_in_every_split(driverwise):
    assert (driverwise.groupby("split")["classname"].nunique() == 10).all()


def test_driverwise_keeps_p072_in_train(driverwise):
    assert set(driverwise.loc[driverwise["subject"] == "p072", "split"]) == {"train"}


def test_random_split_has_every_driver_in_every_split(random_split):
    assert (random_split.groupby("split")["subject"].nunique() == N_DRIVERS).all()


def test_both_files_cover_the_same_images(driverwise, random_split):
    assert set(driverwise["path"]) == set(random_split["path"])


def test_both_files_have_the_same_sizes(driverwise, random_split):
    assert driverwise["split"].value_counts().to_dict() == random_split["split"].value_counts().to_dict()


def test_files_match_what_make_splits_builds(tmp_path, driverwise, random_split):
    if not DRIVER_LIST.exists():
        pytest.skip("the dataset is not in data/, so the files cannot be rebuilt")
    ms.main(["--driver-list", str(DRIVER_LIST), "--out-dir", str(tmp_path)])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "driverwise.csv"), driverwise)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "random.csv"), random_split)
