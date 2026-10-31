"""Download this competition's held-out images through authenticated Kaggle API.

The competition's bulk archive endpoint currently returns HTTP 404, while the
individual-file endpoint works. This script uses at most four workers, retains
valid local files, and writes SHA-256 provenance into the ignored data directory.
It reads filenames and IDs only; it never retrieves or searches for test labels.

Install the official CLI with ``pip install kaggle`` and configure credentials
using Kaggle's supported environment variables or ~/.kaggle configuration. No
credentials are accepted as command-line arguments or written to the manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any
import zlib


COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_worker_state = threading.local()


def _api() -> Any:
    """Give every worker its own authenticated client; never print credentials."""

    if not hasattr(_worker_state, "api"):
        from kaggle.api.kaggle_api_extended import KaggleApi

        client = KaggleApi()
        client.authenticate()
        _worker_state.api = client
    return _worker_state.api


def _status_code(error: Exception) -> int | None:
    status = getattr(getattr(error, "response", None), "status_code", None)
    status = status or getattr(error, "status_code", None) or getattr(error, "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient(error: Exception) -> bool:
    status = _status_code(error)
    if status is not None:
        return status in (408, 429, 500, 502, 503, 504)
    # Do not retry authorization, a missing endpoint, or arbitrary coding errors.
    return any(
        word in type(error).__name__.lower()
        for word in ("timeout", "connection", "protocol", "chunkedencoding")
    )


def _valid_png(path: Path) -> bool:
    """Reject partial/cache files using PNG dimensions, chunk CRCs and IEND.

    Pixel decoding is validated separately by data.py/Pillow. This stdlib-only
    check also works inside the small environment containing the Kaggle CLI.
    """

    try:
        if path.stat().st_size < 57:
            return False
        blob = path.read_bytes()
        if not blob.startswith(PNG_SIGNATURE):
            return False
        cursor = len(PNG_SIGNATURE)
        saw_header = False
        saw_data = False
        while cursor + 12 <= len(blob):
            size = struct.unpack_from(">I", blob, cursor)[0]
            end = cursor + 12 + size
            if end > len(blob):
                return False
            kind = blob[cursor + 4 : cursor + 8]
            payload = blob[cursor + 8 : cursor + 8 + size]
            crc = struct.unpack_from(">I", blob, cursor + 8 + size)[0]
            if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
                return False
            if not saw_header:
                if kind != b"IHDR" or size != 13:
                    return False
                if struct.unpack_from(">II", payload) != (32, 32):
                    return False
                saw_header = True
            if kind == b"IDAT":
                saw_data = True
            if kind == b"IEND":
                return size == 0 and saw_header and saw_data and end == len(blob)
            cursor = end
    except (OSError, struct.error):
        return False
    return False


def _download_file(competition: str, remote_name: str, directory: Path) -> None:
    for attempt in range(2):
        try:
            _api().competition_download_file(
                competition,
                remote_name,
                path=str(directory),
                force=True,
                quiet=True,
            )
            return
        except Exception as error:
            if attempt == 0 and _is_transient(error):
                time.sleep(1.0)
                continue
            status = _status_code(error)
            # Exception messages may contain signed URLs: retain only safe facts.
            detail = f" HTTP {status}" if status is not None else ""
            raise RuntimeError(
                f"Download failed for {remote_name}: {type(error).__name__}{detail}"
            ) from None


def _fetch_image(competition: str, directory: Path, filename: str) -> dict[str, Any]:
    remote_name = f"test_images/{filename}"
    local = directory / "test_images" / filename
    cached = _valid_png(local)
    if not cached:
        _download_file(competition, remote_name, directory)
        if not _valid_png(local):
            raise RuntimeError(f"Downloaded file is not a complete 32x32 PNG: {remote_name}")
    blob = local.read_bytes()
    return {
        "path": remote_name,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "cached": cached,
    }


def _write_manifest(
    directory: Path,
    competition: str,
    records: list[dict[str, Any]],
    expected: int,
    errors: list[str],
) -> Path:
    output = directory / "kaggle_manifest.json"
    document = {
        "competition": competition,
        "source": "Kaggle competition individual-file API",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_test_images": expected,
        "verified_test_images": len(records),
        "complete": len(records) == expected and not errors,
        "total_image_bytes": sum(record["bytes"] for record in records),
        "files": [
            {key: value for key, value in record.items() if key != "cached"}
            for record in sorted(records, key=lambda record: record["path"])
        ],
        "errors": errors,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--competition", default=COMPETITION)
    args = parser.parse_args()
    directory = args.data_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_images").mkdir(exist_ok=True)
    for metadata_name in ("test.csv", "sample_submission.csv"):
        metadata_path = directory / metadata_name
        if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
            _download_file(args.competition, metadata_name, directory)

    with (directory / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "filename"}.issubset(reader.fieldnames):
            raise ValueError("test.csv must contain id and filename columns")
        rows = list(reader)
    filenames = [row["filename"] for row in rows]
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("test.csv has no filenames or has duplicate filenames")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("test.csv has duplicate IDs")
    if any(
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.endswith(".png")
        for filename in filenames
    ):
        raise ValueError("test.csv contains an unsafe or non-PNG filename")

    print(f"Verifying/downloading {len(filenames)} test images with {args.workers} workers", flush=True)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    cached = 0
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch_image, args.competition, directory, filename): filename
            for filename in filenames
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                record = future.result()
                records.append(record)
                cached += int(record["cached"])
                consecutive_failures = 0
            except Exception as error:
                # _fetch_image errors contain only filenames and safe HTTP facts.
                message = f"{futures[future]}: {error}"
                errors.append(message)
                consecutive_failures += 1
                if len(errors) <= 5:
                    print(message, flush=True)
                if consecutive_failures >= 8:
                    for pending in futures:
                        pending.cancel()
                    print(
                        "Stopping after 8 consecutive failures; valid cached files "
                        "are preserved. Check Kaggle service availability before rerunning.",
                        flush=True,
                    )
                    break
            if completed % 100 == 0 or completed == len(filenames):
                _write_manifest(directory, args.competition, records, len(filenames), errors)
                print(
                    f"Progress {completed}/{len(filenames)}: {len(records)} verified, "
                    f"{cached} cached, {len(errors)} failed; "
                    f"elapsed {time.monotonic() - started:.0f}s",
                    flush=True,
                )
    manifest = _write_manifest(directory, args.competition, records, len(filenames), errors)
    if errors:
        raise SystemExit(f"{len(errors)} downloads failed; rerun to reuse valid files. Manifest: {manifest}")
    print(f"Complete: {len(records)} verified test images. Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
