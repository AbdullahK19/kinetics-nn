"""
eval_ensemble.py — Evaluate ensemble accuracy on the test set.
Usage:
    PYTHONPATH=src python3 eval_ensemble.py
"""
import json
import numpy as np
from pathlib import Path
import torch
from src.ensemble_predict import load_model_from_dir, load_scaler, load_class_names, normalize

ENSEMBLE_DIRS = ["models/ensemble_1", "models/ensemble_2", "models/ensemble_3"]
DATA_DIR      = Path("data")
DEVICE        = torch.device("cpu")

# ── Load models ───────────────────────────────────────────────────────────────
print(f"\nLoading {len(ENSEMBLE_DIRS)} ensemble models...")
models, cfgs = [], []
for d in ENSEMBLE_DIRS:
    m, cfg = load_model_from_dir(Path(d), DEVICE)
    models.append(m)
    cfgs.append(cfg)
    print(f"  ✓ {d}  (epoch {cfg.get('best_epoch','?')}, val_acc={cfg.get('best_val_acc',0)*100:.2f}%)")

class_names = load_class_names(DATA_DIR)

# ── Load test set ─────────────────────────────────────────────────────────────
# test.npz is already normalised by preprocess.py — do NOT re-normalise
print("\nLoading test set...")
test = np.load(DATA_DIR / "test.npz")
X_norm = test["X"].astype(np.float32)   # (N, seq_len, features) — already normalised
y_test = test["y"].astype(np.int64)
print(f"  {len(y_test)} samples, {len(np.unique(y_test))} classes")

# ── Batch ensemble inference ──────────────────────────────────────────────────
print("\nRunning ensemble inference...")
X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)

all_model_probs = []
with torch.no_grad():
    for i, model in enumerate(models):
        logits = model(X_tensor)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_model_probs.append(probs)
        print(f"  Model {i+1} done — individual acc: {(probs.argmax(1) == y_test).mean()*100:.2f}%")

# Average probabilities
mean_probs = np.mean(all_model_probs, axis=0)   # (N, 28)
preds      = mean_probs.argmax(axis=1)

# ── Results ───────────────────────────────────────────────────────────────────
correct = (preds == y_test).sum()
total   = len(y_test)
acc     = correct / total * 100

print(f"\n{'='*55}")
print(f"  ENSEMBLE TEST ACCURACY: {acc:.2f}%  ({correct}/{total})")
print(f"{'='*55}")

# Per-class breakdown
print(f"\n{'Class':>5}  {'Name':<35} {'Recall':>8}  {'N':>6}")
print("-" * 60)
classes = sorted(np.unique(y_test))
for c in classes:
    mask    = y_test == c
    recall  = (preds[mask] == c).mean() * 100
    name    = class_names.get(c, f"Class {c}")
    print(f"  {c:>3}  {name:<35} {recall:>7.1f}%  {mask.sum():>6}")

print(f"\nEnsemble: {len(ENSEMBLE_DIRS)} models averaged")
