"""Run privately on Kaggle CPU to export only the attached official inputs.

No network, external datasets, weights, or hidden labels are used. Outputs stay
private: do not publish this kernel's generated data as a public dataset.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
roots = list(Path('/kaggle/input').rglob('train_labels.csv'))
assert len(roots) == 1, f'Expected one official train_labels.csv: {roots}'
root = roots[0].parent
rows = list(csv.DictReader((root / 'train_labels.csv').open()))
assert len(rows) == 50000
images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
labels = np.empty(len(rows), dtype=np.int64)
digest = hashlib.sha256()
for i, row in enumerate(rows):
    path = root / 'train_images' / row['filename']
    raw = path.read_bytes()
    digest.update(row['filename'].encode() + b'\0' + hashlib.sha256(raw).digest())
    with Image.open(path) as im:
        assert im.mode == 'RGB' and im.size == (32, 32)
        images[i] = np.asarray(im)
    labels[i] = int(row['label'])
    if (i+1) % 10000 == 0:
        print(f'Packed {i+1}/50000 official training images', flush=True)
assert np.array_equal(np.bincount(labels), np.full(10, 5000))
np.savez_compressed('official_train.npz', images=images, labels=labels,
                    ids=np.asarray([r['id'] for r in rows]))
manifest = {
    'source': 'official_kaggle_competition', 'competition': COMPETITION,
    'rows': len(rows), 'class_counts': np.bincount(labels).tolist(),
    'train_csv_sha256': hashlib.sha256((root/'train_labels.csv').read_bytes()).hexdigest(),
    'ordered_training_png_sha256': digest.hexdigest(),
    'archive_sha256': hashlib.sha256(Path('official_train.npz').read_bytes()).hexdigest(),
    'export_kernel': 'kevin1220/saber-official-data-export-20261030',
    'no_external_data': True, 'hidden_labels_loaded': False,
}
Path('official_source_manifest.json').write_text(json.dumps(manifest, indent=2))
Path('train_labels.csv').write_bytes((root/'train_labels.csv').read_bytes())
print(json.dumps(manifest, indent=2), flush=True)
