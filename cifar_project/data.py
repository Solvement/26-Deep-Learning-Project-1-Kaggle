"""Strict CSV/image adapters and a reproducible, stratified holdout split."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate CSV columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows in {path}")
    return rows


def unique_values(rows, key):
    values = [row[key] for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{key} values must be nonempty and unique")
    return values


def image_transforms(training):
    # A fixed normalization shared by all experiments, not statistics fit on hidden test data.
    operations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


class CsvImages(Dataset):
    def __init__(self, csv_path, image_dir, training=False, labeled=True):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir).resolve()
        self.rows = read_csv(csv_path, {"filename", "label"} if labeled else {"filename", "id"})
        unique_values(self.rows, "filename")
        if "id" in self.rows[0]:
            unique_values(self.rows, "id")
        self.paths = []
        self.labels = []
        for row in self.rows:
            path = (self.image_dir / row["filename"]).resolve()
            if not path.is_relative_to(self.image_dir):
                raise ValueError("Image path escapes image directory")
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)
            if labeled:
                try:
                    label = int(row["label"])
                except ValueError as exc:
                    raise ValueError("Labels must be integers from 0 to 9") from exc
                if not 0 <= label <= 9:
                    raise ValueError("Labels must be integers from 0 to 9")
                self.labels.append(label)
        self.labeled = labeled
        self.transform = image_transforms(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            if image.size != (32, 32):
                raise ValueError(f"Expected a 32 x 32 image: {self.paths[index]}")
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index] if self.labeled else index

    def fingerprint(self):
        digest = hashlib.sha256(self.csv_path.read_bytes())
        for row, path in zip(self.rows, self.paths):
            digest.update(row["filename"].encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()


def stratified_split(labels, fraction, seed):
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    labels = np.asarray(labels)
    if set(labels.tolist()) != set(range(10)):
        raise ValueError("Training data must contain all 10 CIFAR classes")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("Each class needs at least two examples for a holdout split")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return sorted(train), sorted(validation)


def research_split(labels, validation_fraction, confirmation_fraction, seed):
    """Reserve a confirmation set that training/search must never evaluate."""
    if not 0 < validation_fraction < 0.5 or not 0 < confirmation_fraction < 0.5:
        raise ValueError("Development and confirmation fractions must each be between 0 and 0.5")
    if validation_fraction + confirmation_fraction >= 1:
        raise ValueError("Holdout fractions leave no training data")
    labels = np.asarray(labels)
    if set(labels.tolist()) != set(range(10)):
        raise ValueError("Training data must contain all 10 CIFAR classes")
    rng = np.random.default_rng(seed)
    training, validation, confirmation = [], [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_validation = max(1, round(len(indices) * validation_fraction))
        n_confirmation = max(1, round(len(indices) * confirmation_fraction))
        if n_validation + n_confirmation >= len(indices):
            raise ValueError("Each class needs enough examples for training and both holdouts")
        validation.extend(indices[:n_validation].tolist())
        confirmation.extend(indices[n_validation:n_validation + n_confirmation].tolist())
        training.extend(indices[n_validation + n_confirmation:].tolist())
    return {"training": sorted(training), "validation": sorted(validation), "confirmation": sorted(confirmation)}
