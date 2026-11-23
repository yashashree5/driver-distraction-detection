# Driver Distraction Detection

Deep Learning final project. We classify dashboard-camera images of drivers into 10 behaviour
classes using the [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection)
dataset, and measure how much a random train/test split inflates accuracy compared with a
split that holds out whole drivers.

## The four experiments

| # | Experiment | Owner |
|---|---|---|
| 1 | Baseline CNN trained from scratch | Member 2 |
| 2 | ResNet-18 and MobileNetV2 with transfer learning, frozen vs fine-tuned | Member 3 |
| 3 | Random split vs driver-wise split (leakage) | Member 1 |
| 4 | Grad-CAM heatmaps | Member 4 |

## Classes

| Label | Behaviour | Label | Behaviour |
|---|---|---|---|
| c0 | safe driving | c5 | operating radio |
| c1 | texting (right) | c6 | drinking |
| c2 | phone call (right) | c7 | reaching behind |
| c3 | texting (left) | c8 | hair and makeup |
| c4 | phone call (left) | c9 | talking to passenger |

## Repo layout

```
.
├── README.md
├── requirements.txt
├── data.py            Member 1        dataset, transforms, data loaders
├── make_splits.py     Member 1        builds the two split files
├── train.py           Member 2        shared training loop
├── eval.py            Member 2        shared metrics and confusion matrix
├── models.py          Members 2, 3    baseline CNN, ResNet-18, MobileNetV2
├── gradcam.py         Member 4        Grad-CAM
├── splits/            driverwise.csv, random.csv (the only splits anyone uses)
├── notebooks/         00_eda, then one notebook per experiment (01 to 04)
├── figures/           every plot that goes in the slides
├── tests/             unit tests
├── data/              NOT in git: the dataset (see below)
└── checkpoints/       NOT in git: saved models live in the shared Drive
```

What exists so far: the EDA notebook with its charts in `figures/`, the two split files,
`make_splits.py` and `data.py`. The other Python files are added by their owners.

## Getting the data

The dataset is not in this repo. It is one zip in the team's shared Google Drive folder:
`DL Final Project/data/statefarm_train.zip` (911 MB: all 22,424 labeled training images plus
`driver_imgs_list.csv`).

Unzip it into `data/` so the folder looks like this:

```
data/
├── driver_imgs_list.csv        subject, classname, img
└── imgs/train/c0 ... c9/       22,424 images, 640x480
```

**On your own machine**

```bash
unzip -q statefarm_train.zip -d data
```

**On Colab** (first add a shortcut to the shared `DL Final Project` folder in your My Drive)

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/yashashree5/driver-distraction-detection.git
%cd driver-distraction-detection
!unzip -q "/content/drive/MyDrive/DL Final Project/data/statefarm_train.zip" -d data
```

Unzipping onto the Colab disk is much faster than reading 22,424 small files from Drive.

## Setup

```bash
pip install -r requirements.txt
```

## Loading the data in your code

`data.py` gives you the train, val and test loaders for a split file. It already does the
resize to 224×224, the ImageNet normalisation and the augmentation, so do not add your own.

```python
from data import make_loaders

loaders = make_loaders("splits/driverwise.csv", data_dir="data")
for images, labels in loaders["train"]:
    ...  # images: (32, 3, 224, 224), labels: (32,)
```

For Experiment 3, pass `"splits/random.csv"` instead. The split files are explained in
[splits/README.md](splits/README.md).

Two things to know:

- Call `torch.manual_seed(42)` at the start of training, so the augmentation is the same in
  every run.
- On a Mac, a plain script that uses the loader hangs unless the code is under
  `if __name__ == "__main__":`. Notebooks and Colab are fine. You can also pass
  `num_workers=0`.

To run the tests:

```bash
pytest
```

## Shared rules, so the numbers compare

- **Splits:** only `splits/driverwise.csv` and `splits/random.csv`. Nobody makes their own split.
- **Main number:** test accuracy on the held-out drivers of the driver-wise split. Also report
  macro-F1 and log loss.
- **Input:** 224×224, ImageNet normalisation, the same augmentation for every model.
- **No horizontal flip.** It turns "texting (right)" into "texting (left)" and breaks the labels.
- **Seed:** 42 for every run. Batch size 32 unless the GPU runs out of memory.
- **Storage:** code in this repo; datasets and checkpoints in the shared Drive folder.
- **Colab:** save checkpoints to Drive every epoch so a disconnect does not lose the run.

## Dataset facts (from `notebooks/00_eda.ipynb`)

- 22,424 images, 10 classes, 26 drivers, all 640×480 RGB.
- Classes are roughly balanced: 1,911 to 2,489 images each.
- Drivers are uneven: 346 to 1,237 images each.
- Every driver has every class, but driver `p072` has only 2 images of `c7` and 6 of `c5`.
