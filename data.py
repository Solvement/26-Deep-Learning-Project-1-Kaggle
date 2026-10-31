"""Only staff-provided Kaggle training data and unlabeled test-image loading.

Accepted inputs: train_images/ + audited train_labels.csv, or official_train.npz
exported privately from Kaggle with official_source_manifest.json. External
CIFAR archives and the old external-data caches are never read.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
SOURCE_TYPE = "official_kaggle_competition_training"
CSV_SHA256 = {
    "train_labels.csv": "b596f4cbb12107ecac987e3a61123af40408a6bb83719a1dbb5cc3ce70968e93",
    "test.csv": "d0db3026e3a7bc397d6f59b7ff6825960cc32a5e2aef57df63c9140b2ad31e5d",
    "sample_submission.csv": "83aa7ddd58d46026c3b3f870f507c33b178ebfbdf24e0a69d9f7a9d6d8cfff59",
}
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _official_csv(folder: Path, name: str, columns: list[str]) -> list[dict]:
    path = folder / name
    if file_hash(path) != CSV_SHA256[name]:
        raise RuntimeError(f"Official competition CSV SHA-256 mismatch: {name}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected {name} columns: {reader.fieldnames}")
        rows = list(reader)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {name}.")
    return rows


def _safe_image_path(folder: Path, filename: str) -> Path:
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Competition image filename must be a simple basename: {filename}")
    return folder / filename


def _image_pixels(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError(f"Competition image is not 32x32 RGB: {path.name}")
        return np.asarray(image, dtype=np.uint8)


def _validate_training_arrays(images, labels, ids) -> None:
    if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
        raise RuntimeError("Official training images must be uint8 NHWC with exactly 50,000 examples.")
    if labels.shape != (50000,) or labels.dtype != np.int64 or np.any((labels < 0) | (labels > 9)):
        raise RuntimeError("Official training labels must be 50,000 int64 values in 0..9.")
    if ids.shape != (50000,) or ids.dtype.kind not in "iuUS" or len(np.unique(ids)) != 50000:
        raise RuntimeError("Official training IDs must be 50,000 unique integers or strings.")
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise RuntimeError("Official training data must have exactly 5,000 examples per class.")


def _load_official_training(folder: Path):
    npz_path = folder / "official_train.npz"
    if npz_path.exists():
        source_path = folder / "official_source_manifest.json"
        if not source_path.exists():
            raise FileNotFoundError("official_train.npz requires its audited official_source_manifest.json.")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("competition") != COMPETITION or source.get("source") != "official_kaggle_competition":
            raise RuntimeError("The training export manifest is not from this official competition.")
        if source.get("train_csv_sha256") != CSV_SHA256["train_labels.csv"]:
            raise RuntimeError("The export does not match the audited official train_labels.csv.")
        if source.get("no_external_data") is not True or source.get("hidden_labels_loaded") is not False:
            raise RuntimeError("The export manifest does not confirm official-only inputs and no hidden labels.")
        npz_sha256 = file_hash(npz_path)
        if source.get("archive_sha256") != npz_sha256:
            raise RuntimeError("Official training export NPZ SHA-256 mismatch.")
        with np.load(npz_path, allow_pickle=False) as arrays:
            if not {"images", "labels", "ids"}.issubset(arrays.files):
                raise RuntimeError("official_train.npz must contain images, labels and ids arrays.")
            images, labels, ids = arrays["images"], arrays["labels"], arrays["ids"]
        _validate_training_arrays(images, labels, ids)
        rows = _official_csv(folder, "train_labels.csv", ["id", "filename", "label"])
        if [str(i) for i in ids] != [row["id"] for row in rows] or not np.array_equal(
            labels, np.array([int(row["label"]) for row in rows], dtype=np.int64)
        ):
            raise RuntimeError("Exported IDs/labels do not match official CSV row order.")
        provenance = {
            "kind": "audited_competition_kaggle_export", "official_train_npz_sha256": npz_sha256,
            "official_source_manifest_sha256": file_hash(source_path),
            "train_labels_csv_sha256": source["train_csv_sha256"],
            "ordered_training_png_sha256": source["ordered_training_png_sha256"],
            "export_kernel": source["export_kernel"],
        }
        return np.ascontiguousarray(images), labels, ids, provenance
    if not (folder / "train_labels.csv").exists() or not (folder / "train_images").is_dir():
        raise FileNotFoundError(
            "Supply official train_images/ + train_labels.csv, or official_train.npz + "
            "official_source_manifest.json + train_labels.csv. External CIFAR caches are ignored."
        )
    rows = _official_csv(folder, "train_labels.csv", ["id", "filename", "label"])
    if len(rows) != 50000:
        raise RuntimeError("Official train_labels.csv must have 50,000 rows.")
    pixels, digest = [], hashlib.sha256()
    for i, row in enumerate(rows):
        path = _safe_image_path(folder / "train_images", row["filename"])
        pixels.append(_image_pixels(path))
        digest.update(row["filename"].encode() + b"\0" + bytes.fromhex(file_hash(path)))
        if (i + 1) % 10000 == 0:
            print(f"Official training images loaded: {i + 1}/50000", flush=True)
    images = np.stack(pixels)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    ids = np.array([row["id"] for row in rows])
    _validate_training_arrays(images, labels, ids)
    provenance = {
        "kind": "official_competition_pngs", "train_labels_csv_sha256": CSV_SHA256["train_labels.csv"],
        "ordered_training_png_sha256": digest.hexdigest(),
    }
    return images, labels, ids, provenance


def _stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts, validation_parts, confirmation_parts = [], [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 5000:
            raise RuntimeError(f"Expected 5000 official training examples for class {label}; got {len(indices)}.")
        indices = rng.permutation(indices)
        validation_parts.append(indices[:500])
        confirmation_parts.append(indices[500:1000])
        train_parts.append(indices[1000:])
    train = rng.permutation(np.concatenate(train_parts)).astype(np.int64)
    validation = rng.permutation(np.concatenate(validation_parts)).astype(np.int64)
    confirmation = rng.permutation(np.concatenate(confirmation_parts)).astype(np.int64)
    return train, validation, confirmation


def prepare_training_data(data_dir: str | Path | None = None, seed: int = 2026) -> dict:
    """Cache only official inputs. Perform no network downloads."""
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "training_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_type") == SOURCE_TYPE and manifest.get("split", {}).get("schema") == "train_dev_confirmation_v1":
            if manifest["split"]["seed"] != seed:
                raise ValueError("A different split already exists. Use a separate data directory to change the seed.")
            load_training_arrays(folder, verify=True)
            return manifest

    images, labels, ids, provenance = _load_official_training(folder)
    train, validation, confirmation = _stratified_split(labels, seed)
    _save_npy(folder / "official_train_images.npy", images)
    _save_npy(folder / "official_train_labels.npy", labels)
    _save_npy(folder / "official_train_ids.npy", ids)
    temporary = folder / "official_split_indices.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, train=train, validation=validation, confirmation=confirmation,
                            seed=np.array(seed, dtype=np.int64))
    temporary.replace(folder / "official_split_indices.npz")
    paths = ["official_train_images.npy", "official_train_labels.npy", "official_train_ids.npy", "official_split_indices.npz"]
    manifest = {
        "source_type": SOURCE_TYPE,
        "competition": COMPETITION,
        "source_provenance": provenance,
        "external_training_data_loaded": False,
        "official_test_labels_loaded": False,
        "images": {"count": len(images), "shape": list(images.shape), "dtype": str(images.dtype), "layout": "NHWC"},
        "labels": {"dtype": str(labels.dtype), "class_names": CLASS_NAMES, "class_counts": np.bincount(labels, minlength=10).tolist()},
        "split": {
            "schema": "train_dev_confirmation_v1",
            "seed": seed,
            "algorithm": "numpy.default_rng(seed); per-class permutation; first 500 validation, next 500 one-time confirmation, remaining 4000 training; shuffle concatenated splits",
            "train_count": len(train),
            "validation_count": len(validation),
            "confirmation_count": len(confirmation),
            "train_class_counts": np.bincount(labels[train], minlength=10).tolist(),
            "validation_class_counts": np.bincount(labels[validation], minlength=10).tolist(),
            "confirmation_class_counts": np.bincount(labels[confirmation], minlength=10).tolist(),
            "indices_sha256": hashlib.sha256(train.astype("<i8").tobytes() + validation.astype("<i8").tobytes()
                                              + confirmation.astype("<i8").tobytes()).hexdigest(),
            "train_indices_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(validation.astype("<i8").tobytes()).hexdigest(),
            "confirmation_indices_sha256": hashlib.sha256(confirmation.astype("<i8").tobytes()).hexdigest(),
        },
        "files_sha256": {name: file_hash(folder / name) for name in paths},
    }
    _write_json(manifest_path, manifest)
    load_training_arrays(folder, verify=True)
    return manifest


def load_training_arrays(data_dir: str | Path | None = None, verify: bool = True):
    """Return official images, labels, 40k train IDs and 5k development IDs.

    The 5k confirmation IDs are never returned to epoch-wise training or model
    selection. Use load_confirmation_indices only for the one-time final check.
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    manifest_path = folder / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Supply official Kaggle inputs, then run python Project1/prepare_data.py.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_type") != SOURCE_TYPE or manifest.get("competition") != COMPETITION:
        raise RuntimeError("Training cache is not verified official Kaggle data. External CIFAR caches cannot be used.")
    if manifest.get("split", {}).get("schema") != "train_dev_confirmation_v1":
        raise RuntimeError("Old training split cannot be used. Run prepare_data.py to build the 40k/5k/5k split.")
    if manifest.get("source_provenance", {}).get("train_labels_csv_sha256") != CSV_SHA256["train_labels.csv"]:
        raise RuntimeError("Training provenance does not match the audited official label CSV.")
    if verify:
        for name, expected in manifest["files_sha256"].items():
            if file_hash(folder / name) != expected:
                raise RuntimeError(f"Training cache SHA-256 mismatch: {name}")
    images = np.load(folder / "official_train_images.npy", allow_pickle=False)
    labels = np.load(folder / "official_train_labels.npy", allow_pickle=False)
    ids = np.load(folder / "official_train_ids.npy", allow_pickle=False)
    _validate_training_arrays(images, labels, ids)
    with np.load(folder / "official_split_indices.npz", allow_pickle=False) as split:
        if not {"train", "validation", "confirmation", "seed"}.issubset(split.files):
            raise RuntimeError("The official split must include independent train, validation and confirmation indices.")
        train, validation, confirmation = split["train"].copy(), split["validation"].copy(), split["confirmation"].copy()
        if int(split["seed"]) != manifest["split"]["seed"]:
            raise RuntimeError("The official split seed does not match its manifest.")
    if train.shape != (40000,) or validation.shape != (5000,) or confirmation.shape != (5000,) or not np.array_equal(
        np.sort(np.concatenate([train, validation, confirmation])), np.arange(50000)
    ):
        raise RuntimeError("The train/development/confirmation split is not a disjoint complete partition.")
    if not np.array_equal(np.bincount(labels[train], minlength=10), np.full(10, 4000)) or not np.array_equal(
        np.bincount(labels[validation], minlength=10), np.full(10, 500)
    ) or not np.array_equal(
        np.bincount(labels[confirmation], minlength=10), np.full(10, 500)
    ):
        raise RuntimeError("The train/development/confirmation split is not stratified.")
    split_hash = hashlib.sha256(train.astype("<i8").tobytes() + validation.astype("<i8").tobytes()
                                + confirmation.astype("<i8").tobytes()).hexdigest()
    if split_hash != manifest["split"]["indices_sha256"]:
        raise RuntimeError("The official split fingerprint does not match the manifest.")
    return images, labels, train, validation


def load_confirmation_indices(data_dir: str | Path | None = None) -> np.ndarray:
    """Return the independent 5k confirmation indices after full cache validation.

    Call only after freezing the candidate; these examples must never be used
    for epoch-wise checkpoint selection or subsequent hyperparameter tuning.
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    load_training_arrays(folder, verify=True)
    with np.load(folder / "official_split_indices.npz", allow_pickle=False) as split:
        return split["confirmation"].copy()


def load_kaggle_test(data_dir: str | Path | None = None, test_images_dir: str | Path | None = None):
    """Load official PNGs or an audited private NPZ, in sample submission order."""
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    image_folder = Path(test_images_dir) if test_images_dir is not None else folder / "test_images"
    rows = _official_csv(folder, "test.csv", ["id", "filename"])
    sample = _official_csv(folder, "sample_submission.csv", ["id", "label"])
    ids = [row["id"] for row in sample]
    by_id = {row["id"]: row["filename"] for row in rows}
    if len(rows) != 2000 or len(ids) != 2000 or set(ids) != set(by_id):
        raise ValueError("Official test/sample IDs must match exactly and contain 2,000 examples.")
    archive_path = folder / "official_test.npz"
    if archive_path.exists() and test_images_dir is None:
        source_path = folder / "official_test_source_manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("source") != "official_kaggle_competition" or source.get("competition") != COMPETITION:
            raise RuntimeError("The test export manifest is not from this official competition.")
        if source.get("test_csv_sha256") != CSV_SHA256["test.csv"] or source.get(
            "sample_submission_csv_sha256"
        ) != CSV_SHA256["sample_submission.csv"]:
            raise RuntimeError("The test export does not match the audited official CSV files.")
        if source.get("no_external_data") is not True or source.get("hidden_labels_loaded") is not False or source.get(
            "test_labels_in_archive"
        ) is not False:
            raise RuntimeError("The test export must contain official images and no hidden labels.")
        archive_sha256 = file_hash(archive_path)
        if archive_sha256 != source.get("archive_sha256"):
            raise RuntimeError("Official test export NPZ SHA-256 mismatch.")
        with np.load(archive_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"images", "ids"}:
                raise RuntimeError("Official test export must contain only images and ids arrays.")
            pixels, exported_ids = arrays["images"], arrays["ids"]
        if pixels.shape != (2000, 32, 32, 3) or pixels.dtype != np.uint8:
            raise RuntimeError("Official test export must contain 2,000 uint8 NHWC 32x32 RGB images.")
        if exported_ids.shape != (2000,) or exported_ids.dtype.kind not in "iuUS" or [
            str(i) for i in exported_ids
        ] != [row["id"] for row in rows]:
            raise RuntimeError("Exported test IDs do not match official test.csv row order.")
        positions = {str(identifier): i for i, identifier in enumerate(exported_ids)}
        pixels = pixels[np.array([positions[identifier] for identifier in ids])]
        provenance = {
            "kind": "audited_competition_kaggle_export", "archive_sha256": archive_sha256,
            "source_manifest_sha256": file_hash(source_path),
            "ordered_test_png_sha256": source["ordered_test_png_sha256"],
            "export_kernel": source["export_kernel"],
        }
    else:
        image_pixels, digest = [], hashlib.sha256()
        for identifier in ids:
            filename = by_id[identifier]
            path = _safe_image_path(image_folder, filename)
            image_pixels.append(_image_pixels(path))
            digest.update(filename.encode() + b"\0" + bytes.fromhex(file_hash(path)))
        pixels = np.stack(image_pixels)
        provenance = {"kind": "official_competition_pngs", "ordered_test_png_sha256": digest.hexdigest()}
    manifest = {
        "competition": COMPETITION,
        "count": len(ids), "shape": [len(ids), 32, 32, 3], "dtype": "uint8", "layout": "NHWC",
        "order": "sample_submission.csv ID order", "test_csv_sha256": file_hash(folder / "test.csv"),
        "sample_submission_sha256": file_hash(folder / "sample_submission.csv"),
        "source_provenance": provenance,
        "labels_loaded": False,
    }
    _write_json(folder / "kaggle_test_manifest.json", manifest)
    return ids, pixels


def to_device_images(images: np.ndarray, device="cuda"):
    """Keep uint8 NCHW images on the GPU; normalize only individual minibatches."""
    try:
        from .runtime import configure_runtime
    except ImportError:
        from runtime import configure_runtime
    configure_runtime()
    import torch

    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("Expected uint8 NHWC 32x32 RGB images.")
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, memory_format=torch.channels_last)
