import numpy as np
import pandas as pd
import pytest

import make_splits as ms

N_DRIVERS = 26
N_VAL = 4
N_TEST = 4


def fake_driver_list(n_drivers=N_DRIVERS, seed=0):
    """A small copy of driver_imgs_list.csv where drivers have different sizes."""
    rng = np.random.default_rng(seed)
    rows = []
    img_id = 0
    for d in range(n_drivers):
        for c in range(10):
            for _ in range(int(rng.integers(3, 12))):
                rows.append({"subject": f"p{d:03d}", "classname": f"c{c}", "img": f"img_{img_id}.jpg"})
                img_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def images():
    return ms.prepare_driver_list(fake_driver_list())


@pytest.fixture
def driverwise(images):
    return ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42)


@pytest.fixture
def random_split(images):
    return ms.random_split(images, val_frac=0.15, test_frac=0.15, seed=42)


# prepare_driver_list

def test_prepare_adds_path_and_label():
    raw = pd.DataFrame({"subject": ["p002"], "classname": ["c7"], "img": ["img_5.jpg"]})
    row = ms.prepare_driver_list(raw).iloc[0]
    assert row["path"] == "imgs/train/c7/img_5.jpg"
    assert row["label"] == 7


def test_prepare_rejects_missing_column():
    raw = pd.DataFrame({"subject": ["p002"], "img": ["img_5.jpg"]})
    with pytest.raises(ValueError, match="classname"):
        ms.prepare_driver_list(raw)


def test_prepare_rejects_duplicate_images():
    raw = pd.DataFrame({"subject": ["p002", "p012"], "classname": ["c0", "c0"], "img": ["a.jpg", "a.jpg"]})
    with pytest.raises(ValueError, match="duplicate"):
        ms.prepare_driver_list(raw)


# driver-wise split

def test_driverwise_no_driver_in_two_splits(driverwise):
    splits_per_driver = driverwise.groupby("subject")["split"].nunique()
    assert (splits_per_driver == 1).all()


def test_driverwise_driver_counts(driverwise):
    drivers = driverwise.groupby("split")["subject"].nunique()
    assert drivers["val"] == N_VAL
    assert drivers["test"] == N_TEST
    assert drivers["train"] == N_DRIVERS - N_VAL - N_TEST


def test_driverwise_keeps_every_image_once(images, driverwise):
    assert len(driverwise) == len(images)
    assert driverwise["path"].is_unique
    assert set(driverwise["path"]) == set(images["path"])


def test_driverwise_has_only_known_split_names(driverwise):
    assert set(driverwise["split"]) == {"train", "val", "test"}


def test_driverwise_every_class_in_every_split(driverwise):
    classes = driverwise.groupby("split")["classname"].nunique()
    assert (classes == 10).all()


def test_driverwise_keep_in_train(images):
    kept = ("p000", "p013")
    split = ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42, keep_in_train=kept)
    assert set(split.loc[split["subject"].isin(kept), "split"]) == {"train"}


def test_driverwise_same_seed_gives_same_split(images):
    first = ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42)
    second = ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_driverwise_val_and_test_are_close_to_their_share(driverwise):
    share = driverwise["split"].value_counts(normalize=True)
    assert share["val"] == pytest.approx(N_VAL / N_DRIVERS, abs=0.03)
    assert share["test"] == pytest.approx(N_TEST / N_DRIVERS, abs=0.03)


def test_driverwise_does_not_change_the_input(images):
    before = images.copy()
    ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42)
    pd.testing.assert_frame_equal(images, before)


@pytest.mark.parametrize("n_val, n_test", [(0, 4), (4, 0), (13, 13), (20, 10)])
def test_driverwise_rejects_bad_driver_counts(images, n_val, n_test):
    with pytest.raises(ValueError):
        ms.driverwise_split(images, n_val=n_val, n_test=n_test, seed=42)


def test_driverwise_rejects_unknown_keep_in_train(images):
    with pytest.raises(ValueError, match="p999"):
        ms.driverwise_split(images, n_val=N_VAL, n_test=N_TEST, seed=42, keep_in_train=("p999",))


# random split

def test_random_keeps_every_image_once(images, random_split):
    assert len(random_split) == len(images)
    assert set(random_split["path"]) == set(images["path"])


def test_random_sizes_match_the_fractions(images, random_split):
    counts = random_split["split"].value_counts()
    # rounding happens once per class, so allow one image per class
    assert counts["val"] == pytest.approx(0.15 * len(images), abs=10)
    assert counts["test"] == pytest.approx(0.15 * len(images), abs=10)


def test_random_is_stratified_by_class(random_split):
    test_share = (random_split["split"] == "test").groupby(random_split["classname"]).mean()
    val_share = (random_split["split"] == "val").groupby(random_split["classname"]).mean()
    assert np.allclose(test_share, 0.15, atol=0.01)
    assert np.allclose(val_share, 0.15, atol=0.01)


def test_random_same_seed_gives_same_split(images):
    first = ms.random_split(images, val_frac=0.15, test_frac=0.15, seed=42)
    second = ms.random_split(images, val_frac=0.15, test_frac=0.15, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_random_mixes_drivers_across_splits(random_split):
    train_drivers = set(random_split.loc[random_split["split"] == "train", "subject"])
    test_drivers = set(random_split.loc[random_split["split"] == "test", "subject"])
    assert len(train_drivers & test_drivers) == N_DRIVERS


@pytest.mark.parametrize("val_frac, test_frac", [(0.0, 0.15), (0.15, 0.0), (0.5, 0.5), (-0.1, 0.2)])
def test_random_rejects_bad_fractions(images, val_frac, test_frac):
    with pytest.raises(ValueError):
        ms.random_split(images, val_frac=val_frac, test_frac=test_frac, seed=42)


# both splits together

def test_both_splits_have_the_same_columns(driverwise, random_split):
    assert list(driverwise.columns) == ms.OUTPUT_COLUMNS
    assert list(random_split.columns) == ms.OUTPUT_COLUMNS


def test_main_writes_both_files(tmp_path):
    driver_list = tmp_path / "driver_imgs_list.csv"
    fake_driver_list().to_csv(driver_list, index=False)
    out_dir = tmp_path / "splits"

    ms.main(["--driver-list", str(driver_list), "--out-dir", str(out_dir), "--keep-in-train"])

    driverwise = pd.read_csv(out_dir / "driverwise.csv")
    random_split = pd.read_csv(out_dir / "random.csv")
    assert set(driverwise["path"]) == set(random_split["path"])
    # the random split is built with the same sizes as the driver-wise one
    sizes_a = driverwise["split"].value_counts()
    sizes_b = random_split["split"].value_counts()
    assert (sizes_a - sizes_b).abs().max() <= 10
