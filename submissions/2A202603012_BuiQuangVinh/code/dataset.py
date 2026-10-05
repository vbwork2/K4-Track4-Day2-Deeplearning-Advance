"""DeepWeeds data loading and fixed-fold validation."""

from __future__ import annotations

import hashlib
import random
import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly Acacia",
    "Rubber Vine",
    "Siam Weed",
    "Snake Weed",
    "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOTAL = 17509
OFFICIAL_CONFLICT_HASHES = {
    "labels.csv": "6fb95b89fd9d384f94e185a5cab6c5da7c987649399c90618ecc60acfb0112eb",
    "train_subset0.csv": "a2e63f561a545837c8cabaa276a957ccf1293ec985c9d8fbc49d67b4fd5abdb2",
}


def load_split(labels_dir: str | Path, fold: int = 0):
    if fold != 0:
        raise ValueError("Core evaluation is fixed to official fold 0.")
    path = Path(labels_dir)
    canonical_path = path / "labels.csv"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Official labels file is missing: {canonical_path}")
    canonical = pd.read_csv(canonical_path)
    _validate_labels(canonical, "labels")
    if "Species" not in canonical.columns:
        raise ValueError("Official labels.csv must contain the Species column")
    if len(canonical) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} labels.csv rows, found {len(canonical)}"
        )
    frames = tuple(
        pd.read_csv(path / f"{split}_subset0.csv") for split in ("train", "val", "test")
    )
    for name, frame in zip(("train", "val", "test"), frames):
        _validate_labels(frame, name)
        mapped = (
            canonical.set_index("Filename").Label.reindex(frame.Filename).to_numpy()
        )
        if pd.isna(mapped).any():
            missing = frame.loc[pd.isna(mapped), "Filename"].head(5).tolist()
            raise ValueError(
                f"{name} split contains filenames absent from labels.csv: {missing}"
            )
        mismatch = mapped != frame.Label.to_numpy()
        if mismatch.any():
            conflicts = [
                {
                    "split": name,
                    "Filename": str(row.Filename),
                    "split_label": int(row.Label),
                    "labels_csv_label": int(label),
                }
                for row, label in zip(
                    frame.loc[mismatch].itertuples(), mapped[mismatch]
                )
            ]
            known_conflict = [
                {
                    "split": "train",
                    "Filename": "20170714-110407-3.jpg",
                    "split_label": 0,
                    "labels_csv_label": 1,
                }
            ]
            # Preserve the official split label for this verified upstream inconsistency.
            verified_source = conflicts == known_conflict and all(
                hashlib.sha256((path / filename).read_bytes()).hexdigest() == digest
                for filename, digest in OFFICIAL_CONFLICT_HASHES.items()
            )
            if not verified_source:
                raise ValueError(
                    f"{name} split labels disagree with official labels.csv; "
                    f"{len(conflicts)} conflicts; examples: {conflicts[:5]}"
                )
            frame.attrs["official_label_discrepancies"] = conflicts
            warnings.warn(
                "Official DeepWeeds CSV inconsistency: 20170714-110407-3.jpg "
                "has train label 0 and labels.csv label 1. Source checksums verified; "
                "preserving the official train label 0.",
                UserWarning,
                stacklevel=2,
            )
    return frames


def _validate_labels(frame: pd.DataFrame, name: str) -> None:
    # Official split CSVs contain Filename and Label; Species is only in labels.csv.
    required = {"Filename", "Label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} CSV must contain {sorted(required)}")
    if frame.Filename.duplicated().any():
        raise ValueError(f"{name} CSV contains duplicate filenames")
    labels = pd.to_numeric(frame.Label, errors="raise")
    if (
        labels.isna().any()
        or ((labels < 0) | (labels >= NUM_CLASSES) | (labels % 1 != 0)).any()
    ):
        raise ValueError(f"{name} CSV contains labels outside [0, 8]")
    frame["Label"] = labels.astype("int64")


def check_split(train_df, val_df, test_df, images_dir: str | Path) -> dict:
    frames = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in frames.items():
        _validate_labels(frame, name)
    names = {key: set(value.Filename.astype(str)) for key, value in frames.items()}
    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    if any(overlap.values()):
        raise ValueError("Official split files overlap by Filename")
    all_names = set.union(*names.values())
    if len(all_names) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} unique fold-0 images, found {len(all_names)}"
        )
    images = Path(images_dir)
    missing = [f for f in all_names if not (images / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CSV image files are missing; examples: {missing[:5]}"
        )
    per_class = {
        split: {CLASS_NAMES[i]: int((df.Label == i).sum()) for i in range(NUM_CLASSES)}
        for split, df in frames.items()
    }
    counts = {split: len(df) for split, df in frames.items()}
    ratios = {split: counts[split] / EXPECTED_TOTAL for split in frames}
    report = {
        "n": counts,
        "ratios": ratios,
        "per_class": per_class,
        "overlap": {key: len(value) for key, value in overlap.items()},
        "union": len(all_names),
        "missing_files": 0,
        "official_label_discrepancies": [
            conflict
            for df in frames.values()
            for conflict in df.attrs.get("official_label_discrepancies", [])
        ],
    }
    if (
        abs(ratios["train"] - 0.6) > 0.01
        or abs(ratios["val"] - 0.2) > 0.01
        or abs(ratios["test"] - 0.2) > 0.01
    ):
        raise ValueError(f"Fold 0 split proportions are outside tolerance: {ratios}")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if img_size < 32:
        raise ValueError("img_size must be at least 32")
    if train:
        ops = [
            T.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ]
        if aug == "color":
            ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            ops.append(T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))
        elif aug == "randaug":
            ops.append(
                T.RandAugment(
                    num_ops=2, magnitude=9, interpolation=InterpolationMode.BILINEAR
                )
            )
        elif aug != "basic":
            raise ValueError(f"Unknown augmentation: {aug}")
        ops.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    else:
        resize_size = max(img_size, round(img_size * 256 / 224))
        ops = [
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    return T.Compose(ops)


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        _validate_labels(df, "dataset")
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / str(row.Filename)) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row.Label), str(row.Filename)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def make_loader(
    df,
    images_dir,
    transform,
    batch_size: int,
    train: bool,
    sampler: str | None = None,
    num_workers: int = 2,
    seed: int = 0,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError(
            "batch_size must be positive and num_workers must be non-negative"
        )
    if train and len(df) < batch_size:
        raise ValueError("Training set must contain at least one full batch")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(seed)
    weighted_sampler = None
    if sampler not in (None, "balanced"):
        raise ValueError("sampler must be None or 'balanced'")
    if sampler == "balanced":
        counts = df.Label.value_counts().to_dict()
        weights = torch.as_tensor(
            [1.0 / counts[int(label)] for label in df.Label], dtype=torch.double
        )
        weighted_sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and weighted_sampler is None,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=False,
    )
