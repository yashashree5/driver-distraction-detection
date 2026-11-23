"""Build the two split files that every experiment uses.

    python make_splits.py

driverwise.csv holds out whole drivers, so no driver is in more than one of
train, val and test. random.csv ignores the drivers and is only used in
Experiment 3 to show how much a random split inflates accuracy.

Both files list every labeled image once and have the same columns:
path, classname, label, subject, split.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("subject", "classname", "img")
OUTPUT_COLUMNS = ["path", "classname", "label", "subject", "split"]

# p072 has only 2 images of c7 and 6 of c5 (see notebooks/00_eda.ipynb),
# so it would be a weak validation or test driver.
KEEP_IN_TRAIN = ("p072",)
N_TRIALS = 2000


def prepare_driver_list(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the Kaggle driver list and add the image path and the numeric label."""
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"driver list is missing columns: {missing}")
    if raw["img"].duplicated().any():
        raise ValueError("driver list has duplicate image names")

    images = raw.assign(
        path="imgs/train/" + raw["classname"] + "/" + raw["img"],
        label=raw["classname"].str[1:].astype(int),
    )
    # sort so the result does not depend on the row order of the input file
    return images.sort_values("path").reset_index(drop=True)[["path", "classname", "label", "subject"]]


def pick_drivers(
    images: pd.DataFrame,
    n_val: int,
    n_test: int,
    seed: int = SEED,
    keep_in_train: Sequence[str] = (),
    n_trials: int = N_TRIALS,
) -> dict[str, list[str]]:
    """Choose which drivers go to val and which go to test.

    Drivers have very different numbers of images, so one random draw can give
    a test set that is much too small or that has very little of one class.
    We try many random draws and keep the one where val and test are closest
    to their fair share of the images and to the class mix of the whole
    dataset. Model accuracy is never used to choose.
    """
    counts = images.groupby(["subject", "classname"]).size().unstack(fill_value=0).sort_index()
    drivers = list(counts.index)

    unknown = sorted(set(keep_in_train) - set(drivers))
    if unknown:
        raise ValueError(f"keep_in_train has drivers that are not in the data: {unknown}")
    free = [d for d in drivers if d not in set(keep_in_train)]
    if n_val < 1 or n_test < 1 or n_trials < 1:
        raise ValueError("n_val, n_test and n_trials must be at least 1")
    if n_val + n_test >= len(drivers) or n_val + n_test > len(free):
        raise ValueError(f"cannot hold out {n_val} + {n_test} drivers from {len(drivers)}")

    table = counts.loc[free].to_numpy()
    total = counts.to_numpy().sum()
    class_share = counts.to_numpy().sum(axis=0) / total

    def gap(rows: np.ndarray, n_drivers: int) -> float:
        picked = table[rows].sum(axis=0)
        size_gap = abs(picked.sum() / total - n_drivers / len(drivers))
        class_gap = np.abs(picked / picked.sum() - class_share).max()
        return size_gap + class_gap

    rng = np.random.default_rng(seed)
    best_score, best_order = np.inf, None
    for _ in range(n_trials):
        order = rng.permutation(len(free))
        score = gap(order[:n_val], n_val) + gap(order[n_val:n_val + n_test], n_test)
        if score < best_score:
            best_score, best_order = score, order

    val = sorted(free[i] for i in best_order[:n_val])
    test = sorted(free[i] for i in best_order[n_val:n_val + n_test])
    train = sorted(set(drivers) - set(val) - set(test))
    return {"train": train, "val": val, "test": test}


def driverwise_split(
    images: pd.DataFrame,
    n_val: int,
    n_test: int,
    seed: int = SEED,
    keep_in_train: Sequence[str] = (),
) -> pd.DataFrame:
    """Give every image the split of its driver."""
    drivers = pick_drivers(images, n_val, n_test, seed, keep_in_train)
    split_of = {driver: split for split, names in drivers.items() for driver in names}
    return images.assign(split=images["subject"].map(split_of))[OUTPUT_COLUMNS]


def random_split(images: pd.DataFrame, val_frac: float, test_frac: float, seed: int = SEED) -> pd.DataFrame:
    """Shuffle the images inside each class and cut them into train, val and test.

    Drivers are ignored, so the same driver ends up in all three sets.
    """
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and add up to less than 1")

    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=images.index)
    for _, rows in images.groupby("classname", sort=True):
        order = rng.permutation(rows.index.to_numpy())
        n_test = round(len(order) * test_frac)
        n_val = round(len(order) * val_frac)
        split.loc[order[:n_test]] = "test"
        split.loc[order[n_test:n_test + n_val]] = "val"
    return images.assign(split=split)[OUTPUT_COLUMNS]


def summarize(split: pd.DataFrame) -> pd.DataFrame:
    """One row per split: images, share of all images, drivers, smallest class."""
    grouped = split.groupby("split")
    summary = pd.DataFrame({
        "images": grouped.size(),
        "share": (grouped.size() / len(split)).round(3),
        "drivers": grouped["subject"].nunique(),
        "smallest_class": grouped["classname"].agg(lambda names: names.value_counts().min()),
    })
    return summary.reindex(SPLITS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build splits/driverwise.csv and splits/random.csv")
    parser.add_argument("--driver-list", type=Path, default=Path("data/driver_imgs_list.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("splits"))
    parser.add_argument("--n-val", type=int, default=4, help="drivers held out for validation")
    parser.add_argument("--n-test", type=int, default=4, help="drivers held out for test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--keep-in-train", nargs="*", default=list(KEEP_IN_TRAIN),
                        help="drivers that must stay in the training set")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    images = prepare_driver_list(pd.read_csv(args.driver_list))

    driverwise = driverwise_split(images, args.n_val, args.n_test, args.seed, args.keep_in_train)
    # use the same sizes for the random split, so Experiment 3 compares like with like
    share = driverwise["split"].value_counts(normalize=True)
    shuffled = random_split(images, share["val"], share["test"], args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    driverwise.to_csv(args.out_dir / "driverwise.csv", index=False)
    shuffled.to_csv(args.out_dir / "random.csv", index=False)

    for name, split in (("driverwise", driverwise), ("random", shuffled)):
        print(f"\n{name}.csv")
        print(summarize(split).to_string())
    for split_name in ("val", "test"):
        drivers = sorted(driverwise.loc[driverwise["split"] == split_name, "subject"].unique())
        print(f"\n{split_name} drivers: {', '.join(drivers)}")


if __name__ == "__main__":
    main()
