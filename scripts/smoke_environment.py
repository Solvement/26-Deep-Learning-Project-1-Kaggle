"""Validate CPU ML dependencies; this is not a competition training pipeline."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(2)
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x / 16.0, y, test_size=0.2, random_state=42, stratify=y)
train_x = torch.tensor(x_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)
test_x = torch.tensor(x_test, dtype=torch.float32)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
losses = []
model.train()
for _ in range(12):
    total = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    losses.append(total / len(train_y))
model.eval()
with torch.no_grad():
    logits = model(test_x)
    predicted = logits.argmax(dim=1).numpy()
accuracy = accuracy_score(y_test, predicted)
if not losses[-1] < losses[0] * 0.3:
    raise RuntimeError(f"Training loss did not decrease enough: {losses}")
if accuracy <= 0.85:
    raise RuntimeError(f"Held-out accuracy must exceed 0.85; got {accuracy}")
checkpoint = out / 'digits-model.pt'
torch.save(model.state_dict(), checkpoint)
restored = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
restored.eval()
with torch.no_grad():
    torch.testing.assert_close(restored(test_x), logits)
image_tensor = ToTensor()(Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)))
if image_tensor.shape != (3, 8, 8):
    raise RuntimeError(f"Unexpected image tensor shape: {image_tensor.shape}")
frame = pd.DataFrame({'label': y_test, 'prediction': predicted})
frame.to_csv(out / 'predictions.csv', index=False)
if len(pd.read_csv(out / 'predictions.csv')) != len(y_test):
    raise RuntimeError('CSV round-trip changed the number of rows')
plt.plot(range(1, 13), losses)
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.savefig(out / 'training-loss.png')
plt.close()
result = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'training_samples': len(y_train), 'test_samples': len(y_test), 'epochs': 12, 'initial_loss': losses[0], 'final_loss': losses[-1], 'test_accuracy': accuracy, 'checkpoint_roundtrip': 'passed', 'torchvision_transform': 'passed', 'csv_and_plot': 'passed'}
(out / 'training-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
