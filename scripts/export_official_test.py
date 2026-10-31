"""Pack the attached official Kaggle test images in a private CPU kernel.

No internet, external data, pretrained weights, or hidden labels are used. The
output is competition data and must stay private. The sample submission is
copied byte-for-byte solely as a format template; its label values are not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
KERNEL = "kevin1220/saber-official-test-export-20261030"


def export_test(input_root: Path, output_dir: Path, expected_rows: int = 2000) -> dict:
    candidates = [
        path
        for path in input_root.rglob("test.csv")
        if (path.parent / "test_images").is_dir()
        and (path.parent / "sample_submission.csv").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one attached official test.csv; found {len(candidates)}")
    test_csv = candidates[0]
    root = test_csv.parent
    with test_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"id", "filename"}:
            raise ValueError("Expected only id and filename columns in official test.csv")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} official test rows, got {len(rows)}")
    ids = np.asarray([row["id"] for row in rows], dtype=str)
    filenames = [row["filename"] for row in rows]
    if len(set(ids)) != len(rows) or len(set(filenames)) != len(rows):
        raise ValueError("Official test.csv has duplicate IDs or filenames")
    if any(
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.endswith(".png")
        for filename in filenames
    ):
        raise ValueError("Official test.csv contains unsafe filenames")

    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    ordered_png_digest = hashlib.sha256()
    image_files = []
    for index, filename in enumerate(filenames):
        path = root / "test_images" / filename
        raw = path.read_bytes()
        image_sha256 = hashlib.sha256(raw).hexdigest()
        ordered_png_digest.update(filename.encode("utf-8") + b"\0" + bytes.fromhex(image_sha256))
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(f"Invalid official test image dimensions/mode: {filename}")
            images[index] = np.asarray(image)
        image_files.append({
            "path": f"test_images/{filename}",
            "bytes": len(raw),
            "sha256": image_sha256,
        })
        if (index + 1) % 500 == 0:
            print(f"Packed {index + 1}/{len(rows)} official test images", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "official_test.npz"
    np.savez_compressed(archive_path, images=images, ids=ids)
    test_csv_bytes = test_csv.read_bytes()
    sample_bytes = (root / "sample_submission.csv").read_bytes()
    (output_dir / "test.csv").write_bytes(test_csv_bytes)
    (output_dir / "sample_submission.csv").write_bytes(sample_bytes)
    manifest = {
        "source": "official_kaggle_competition",
        "competition": COMPETITION,
        "export_kernel": KERNEL,
        "rows": len(rows),
        "images_shape": list(images.shape),
        "images_dtype": str(images.dtype),
        "ids_dtype": str(ids.dtype),
        "row_order": "official test.csv order",
        "test_csv_sha256": hashlib.sha256(test_csv_bytes).hexdigest(),
        "sample_submission_csv_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "ordered_test_png_sha256": ordered_png_digest.hexdigest(),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "no_external_data": True,
        "hidden_labels_loaded": False,
        "test_labels_in_archive": False,
        "files": image_files,
    }
    (output_dir / "official_test_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--expected-test-rows", type=int, default=2000)
    args = parser.parse_args()
    export_test(args.input_root, args.output_dir, args.expected_test_rows)


if __name__ == "__main__":
    main()
