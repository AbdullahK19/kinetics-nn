# Quick Start Guide

Get predictions from the trained 28-class ensemble in under 5 minutes.

---

## Step 1: Install dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm openpyxl
```

---

## Step 2: Run a prediction

The model is already trained. No setup needed — just run:

```bash
# Predict mechanism from reactor conditions (PFR, A → C reaction)
PYTHONPATH=src python3 src/predict.py \
  --ode --reactor PFR \
  --mechanism 0 --A0 0.3 --B0 0.1 \
  --T 550 --P 100000 --rc k1=5e-5
```

**Example output:**
```
Mechanism Predictions (ranked by probability, 0.5% cutoff, top-10 max):

 1. Equation:   A -> C
    Type:        simple
    Likelihood:   91.56%

 2. Equation:   A -> C -> D
    Type:        sequential
    Likelihood:    0.84%
```

---

## Step 3: Use the ensemble (higher accuracy)

The 3-model ensemble gives 93.63% test accuracy:

```bash
PYTHONPATH=src python3 src/ensemble_predict.py \
  --ensemble_dirs models/ensemble_1 models/ensemble_2 models/ensemble_3 \
  --ode --reactor PFR \
  --mechanism 4 --A0 0.2 --B0 0.15 \
  --T 550 --P 100000 --rc k1=1e-10
```

---

## Key parameters

| Parameter | Flag | Range | Notes |
|-----------|------|-------|-------|
| Reactor type | `--reactor` | PFR / CSTR / CSTR_cascade | |
| Molar flow A | `--A0` | 0.05–0.5 mol/s | |
| Molar flow B | `--B0` | 0.05–0.5 mol/s | |
| Temperature | `--T` | 350–750 K | |
| Pressure | `--P` | 50,000–200,000 Pa | |
| 1st-order rate | `--rc k1=` | 1e-6 – 1e-4 | Groups A, G, I |
| 2nd-order rate | `--rc k1=` | 1e-11 – 5e-10 | Groups B, C, H |

---

## Evaluate on test set

```bash
# Single model
PYTHONPATH=src python3 eval_ensemble.py
# → prints per-class recall + overall accuracy

# Noise robustness (0–50% Gaussian noise, 101 levels, ~9 min)
PYTHONPATH=src python3 noise_robustness.py
# → saves results/noise_robustness.xlsx
```

---

## Retrain from scratch

```bash
# 1. Generate dataset (500K runs, ~2–3 hours)
PYTHONPATH=src python3 src/dataset_builder.py --n_runs 500000

# 2. Preprocess
PYTHONPATH=src python3 src/preprocess.py

# 3. Train (use MPS on Apple Silicon for ~4–6 hours)
PYTHONPATH=src python3 src/train.py \
  --model_type rescnn --epochs 300 \
  --hidden_channels 64 128 256 --fc_hidden 256 --blocks_per_stage 2 \
  --dropout 0.25 --lr 0.0003 --weight_decay 0.001 --batch_size 64 \
  --optimizer adamw --scheduler cosine --label_smoothing 0.1 \
  --class_weights --augment --patience 80 --focal --focal_gamma 2.0 \
  --device mps --seed 42 --save_dir models/ensemble_1
```

---

## File locations

| File | Purpose |
|------|---------|
| `models/ensemble_1/best_model.pt` | Trained weights (seed 42, 92.99% val) |
| `models/ensemble_2/best_model.pt` | Trained weights (seed 123, 92.90% val) |
| `models/ensemble_3/best_model.pt` | Trained weights (seed 456, 92.94% val) |
| `data/scaler.json` | z-score normalisation parameters |
| `data/dataset_info.json` | Class names and dataset statistics |
| `results/noise_robustness.xlsx` | Noise robustness results |

---

## Troubleshooting

**`ModuleNotFoundError`** — Always prefix commands with `PYTHONPATH=src`

**`ODE solver failed`** — Rate constants too extreme. Stay within the ranges above.

**Low confidence for all classes** — Parameters outside training distribution (check T, P, A0, B0 ranges).

**Classes 17/18 confused** — These are genuinely hard to distinguish. Check rank 2 — the correct class is usually there.
