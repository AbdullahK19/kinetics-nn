# Kinetics Neural Network — Reaction Mechanism Classifier

A deep learning pipeline that identifies **which chemical reaction mechanism** is operating inside a reactor, purely from concentration profiles. Feed it reactor outputs; it tells you the mechanism.

**93.63% test accuracy** (3-model ensemble) across 28 mechanism classes, covering unimolecular, bimolecular, sequential, parallel, reversible, and split-product reactions.

---

## What it does

Given a set of reactor conditions (temperature, pressure, initial flows, rate constants), the model:

1. Runs an ODE simulation to produce a concentration profile
2. Classifies the profile into one of 28 reaction mechanisms
3. Returns a ranked list of candidates with probabilities

---

## Quick start (5 minutes)

### 1. Install dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm
```

### 2. Run a prediction

```bash
# Predict mechanism for A → C in a PFR
PYTHONPATH=src python3 src/predict.py \
  --ode --reactor PFR \
  --mechanism 0 --A0 0.3 --B0 0.1 \
  --T 550 --P 100000 --rc k1=5e-5
```

**Output:**
```
Mechanism Predictions (ranked by probability, 0.5% cutoff, top-10 max):
======================================================================

 1. Equation:   A -> C
    Type:        simple
    Likelihood:   91.56%

 2. Equation:   A -> C -> D
    Type:        sequential
    Likelihood:    0.84%
```

---

## The 28 mechanism classes

### Group A — Unimolecular (1st order)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 0 | `A → C` | k1 |
| 1 | `A → D` | k1 |
| 2 | `B → C` | k1 |
| 3 | `B → D` | k1 |

### Group B — Bimolecular A+B (2nd order)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 4 | `A + B → C` | k1 |
| 5 | `A + B → D` | k1 |

### Group C — Bimolecular homogeneous (2nd order)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 6 | `2A → C` | k1 |
| 7 | `2A → D` | k1 |
| 8 | `2B → C` | k1 |
| 9 | `2B → D` | k1 |

### Group D — Sequential (C is intermediate)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 10 | `A + B → C → D` | k1, k2 |
| 11 | `2A → C → D` | k1, k2 |
| 12 | `A → C → D` | k1, k2 |
| 13 | `B → C → D` | k1, k2 |

### Group E — Parallel
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 14 | `A+B → C  ;  2A → D` | k1, k2 |
| 15 | `A → C  ;  B → D` | k1, k2 |
| 16 | `2A → C  ;  2B → D` | k1, k2 |

### Group F — Parallel-Sequential (hardest)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 17 | `A+B → C → D  ;  2A → D` | k1, k2, k3 |
| 18 | `2A → C → D  ;  A+B → D` | k1, k2, k3 |
| 19 | `A → C → D  ;  B → D` | k1, k2, k3 |

### Group G — Reversible Simple (1st order both ways)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 20 | `A ⇌ C` | k1, k2 |
| 21 | `A ⇌ D` | k1, k2 |
| 22 | `B ⇌ C` | k1, k2 |
| 23 | `B ⇌ D` | k1, k2 |

### Group H — Reversible Bimolecular (2nd order forward, 1st order reverse)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 24 | `A + B ⇌ C` | k1, k2 |
| 25 | `A + B ⇌ D` | k1, k2 |

### Group I — Split Product (one reactant, two simultaneous products)
| Class | Equation | Rate constants |
|-------|----------|----------------|
| 26 | `A → C + D` | k1 |
| 27 | `B → C + D` | k1 |

---

## Rate constant ranges

Rate constants must stay within these ranges for reliable predictions (the model was trained on these ranges):

| Order | Applicable to | Valid range | Units |
|-------|--------------|-------------|-------|
| 1st order | Groups A, D (k2), E (k1/k2 for class 15), F (class 19), G (k1, k2), I | `1e-6` – `1e-4` | mol s⁻¹ m⁻³ Pa⁻¹ |
| 2nd order | Groups B, C, D (k1), E (k1/k2 for 14/16), F, H (k1) | `1e-11` – `5e-10` | mol s⁻¹ m⁻³ Pa⁻² |

Per-mechanism breakdown:

| Class | k1 | k2 | k3 |
|-------|-----|-----|-----|
| 0–3 | 1st order | — | — |
| 4–5 | 2nd order | — | — |
| 6–9 | 2nd order | — | — |
| 10–11 | 2nd order | 1st order | — |
| 12–13 | 1st order | 1st order | — |
| 14 | 2nd order | 2nd order | — |
| 15 | 1st order | 1st order | — |
| 16 | 2nd order | 2nd order | — |
| 17 | 2nd order (k1), 2nd order (k2) | 1st order (k3) | — |
| 18 | 2nd order (k1), 1st order (k2) | 2nd order (k3) | — |
| 19 | 1st order (k1), 1st order (k2) | 1st order (k3) | — |
| 20–23 | 1st order | 1st order | — |
| 24–25 | 2nd order | 1st order | — |
| 26–27 | 1st order | — | — |

---

## Full predict.py reference

```bash
PYTHONPATH=src python3 src/predict.py \
  --ode \
  --reactor    PFR | CSTR | CSTR_cascade \
  --mechanism  0-27 \
  --A0         <mol/s>   # initial molar flow of A, range: 0.05–0.5 \
  --B0         <mol/s>   # initial molar flow of B, range: 0.05–0.5 \
  --T          <K>       # temperature, range: 350–750 K \
  --P          <Pa>      # pressure, range: 50000–200000 Pa \
  --rc         k1=<val> [k2=<val>] [k3=<val>] \
  [--V_total   1.0]      # reactor volume in m³ \
  [--n_cstr    50]       # number of CSTR stages (CSTR_cascade only) \
  [--threshold 0.005]    # minimum probability to show (default 0.5%) \
  [--json_output]        # print JSON instead of formatted text \
  [--I0        0.0]      # inert molar flow rate (mol/s)
```

### Example commands for every reactor type

```bash
# PFR — simple unimolecular
PYTHONPATH=src python3 src/predict.py --ode --reactor PFR \
  --mechanism 0 --A0 0.2 --B0 0.1 --T 550 --P 100000 --rc k1=5e-5

# CSTR cascade — sequential with intermediate
PYTHONPATH=src python3 src/predict.py --ode --reactor CSTR_cascade \
  --mechanism 10 --A0 0.3 --B0 0.25 --T 600 --P 120000 --rc k1=1.5e-10 k2=2e-5

# Single CSTR — parallel mechanism
PYTHONPATH=src python3 src/predict.py --ode --reactor CSTR \
  --mechanism 15 --A0 0.25 --B0 0.2 --T 500 --P 80000 --rc k1=3e-5 k2=8e-5

# Reversible (equilibrium) — PFR
PYTHONPATH=src python3 src/predict.py --ode --reactor PFR \
  --mechanism 20 --A0 0.3 --B0 0.1 --T 550 --P 100000 --rc k1=5e-5 k2=2.5e-5

# Split product A → C + D — PFR
PYTHONPATH=src python3 src/predict.py --ode --reactor PFR \
  --mechanism 26 --A0 0.3 --B0 0.1 --T 550 --P 100000 --rc k1=5e-5

# JSON output for scripting
PYTHONPATH=src python3 src/predict.py --ode --reactor PFR \
  --mechanism 12 --A0 0.15 --B0 0.1 --T 550 --P 100000 \
  --rc k1=1e-5 k2=5e-5 --json_output
```

---

## Using from Python

```python
import sys
sys.path.insert(0, 'src')
from predict import MechanismPredictor, format_predictions

predictor = MechanismPredictor(
    model_path='models/best_model.pt',
    scaler_path='data/scaler.json',
    config_path='models/training_config.json',
)

predictions = predictor.predict_from_ode(
    reactor_type='PFR',            # 'PFR', 'CSTR', or 'CSTR_cascade'
    mechanism_class=10,            # class used to run the ODE simulation
    rate_constants={'k1': 1.5e-10, 'k2': 2e-5},
    A0=0.3,                        # mol/s
    B0=0.25,                       # mol/s
    T=600.0,                       # K
    P=120_000.0,                   # Pa
    V_total=1.0,                   # m³
    threshold=0.005,               # minimum probability to include
)

print(format_predictions(predictions))

# Or access results directly:
for pred in predictions:
    print(f"{pred['equation']:40s}  {pred['probability']*100:.1f}%")
```

---

## Ensemble inference

The ensemble averages softmax probabilities from all 3 models, giving 93.63% accuracy vs 92.9–93.2% for any individual model.

```bash
# Evaluate ensemble on test set
PYTHONPATH=src python3 eval_ensemble.py

# Single ODE prediction using ensemble
PYTHONPATH=src python3 src/ensemble_predict.py \
  --ensemble_dirs models/ensemble_1 models/ensemble_2 models/ensemble_3 \
  --ode --reactor PFR --mechanism 4 \
  --A0 0.2 --B0 0.1 --T 550 --P 100000 --rc k1=1e-10

# Noise robustness analysis (takes ~9 minutes on MPS)
PYTHONPATH=src python3 noise_robustness.py
```

---

## Retrain from scratch

If you want to regenerate data and retrain the model:

```bash
# 1. Generate 100K simulation runs (all 28 classes)
PYTHONPATH=src python3 src/dataset_builder.py --n_runs 100000 --save_dir ./data

# 2. Preprocess (split + normalise)
PYTHONPATH=src python3 src/preprocess.py --data_path ./data/dataset.npz --save_dir ./data

# 3. Train (runs for ~250 epochs with early stopping)
PYTHONPATH=src python3 src/train.py \
  --model_type rescnn \
  --epochs 250 --batch_size 64 \
  --hidden_channels 64 128 256 --fc_hidden 256 --blocks_per_stage 2 \
  --dropout 0.25 --lr 0.0003 --weight_decay 0.001 \
  --optimizer adamw --scheduler cosine \
  --label_smoothing 0.1 --class_weights --augment \
  --focal --focal_gamma 2.0 \
  --swa_epochs 20 --patience 80 \
  --save_dir ./models --device cpu

# 4. Resume if training is interrupted
PYTHONPATH=src python3 src/train.py [same args as above] --resume
```

For persistent training that survives terminal closure:
```bash
nohup PYTHONPATH=src python3 src/train.py [args] >> train.log 2>&1 &
tail -f train.log   # monitor progress
```

---

## Project structure

```
kinetics_nn/
├── src/
│   ├── ode_solver.py        # 28-class mechanism registry + PFR/CSTR solvers
│   ├── dataset_builder.py   # Dataset generation (ODE solver, 500K runs)
│   ├── preprocess.py        # z-score normalisation, train/val/test split
│   ├── model.py             # ResConv1D + SE attention (and CNN/GRU/LSTM alternatives)
│   ├── train.py             # Training: FocalLoss, SWA, TTA, cosine LR, --resume
│   ├── predict.py           # Single-model inference: MechanismPredictor class + CLI
│   ├── ensemble_predict.py  # 3-model ensemble inference CLI
│   ├── validate_excel.py    # Gallery plots + ODE vs Excel cross-validation
│   └── excel_driver.py      # Excel integration (optional, classes 0–2 only)
├── data/
│   ├── train.npz            # Training samples (331,300)
│   ├── val.npz              # Validation samples (70,993)
│   ├── test.npz             # Test samples (70,994)
│   ├── scaler.json          # z-score normalisation parameters
│   └── dataset_info.json    # Dataset statistics + feature names
├── models/
│   ├── ensemble_1/          # Seed 42 — best val 92.99% (epoch 72)
│   │   ├── best_model.pt
│   │   └── training_config.json
│   ├── ensemble_2/          # Seed 123 — best val 92.90% (epoch 25)
│   │   ├── best_model.pt
│   │   └── training_config.json
│   └── ensemble_3/          # Seed 456 — best val 92.94% (epoch 83)
│       ├── best_model.pt
│       └── training_config.json
├── results/
│   ├── noise_robustness.xlsx      # Full noise analysis (3 sheets, colour-coded)
│   ├── noise_overall.csv          # Overall accuracy vs noise level
│   ├── noise_per_class_recall.csv # Per-class recall vs noise level
│   └── noise_breakpoints.csv      # Per-class noise breakpoints
├── eval_ensemble.py         # Evaluate ensemble on test set
├── noise_robustness.py      # Noise robustness analysis (0–50% in 0.5% steps)
├── run_ensemble_mps.sh      # Sequential MPS training script
├── validation/
│   ├── mechanism_gallery_pfr.png   # Concentration profiles for all 28 classes
│   ├── mechanism_gallery_cstr.png
│   └── intermediate_detail.png
└── configs/
    ├── default_config.json         # 28-class production config
    └── smoke_test_config.json
```

---

## Model architecture

**ResConv1DClassifier** — residual 1D CNN with Squeeze-and-Excitation attention

| Component | Detail |
|-----------|--------|
| Input | (batch, 128, 16) — 128 grid points × 16 features |
| Stem | Conv1d(16→64, kernel=7) |
| Stage 1 | MaxPool + 2× ResBlock(64) + SE attention |
| Stage 2 | Conv1d(64→128, k=1) + MaxPool + 2× ResBlock(128) + SE attention |
| Stage 3 | Conv1d(128→256, k=1) + MaxPool + 2× ResBlock(256) + SE attention |
| Head | AdaptiveAvgPool → FC(256) → FC(128) → FC(28) |
| Parameters | 1,275,444 |

**16 input features per grid point:**
`[A, B, C, D, n_total, reactor_type, T_norm, P_norm, dA/dV, dB/dV, dC/dV, dD/dV, xA, xB, xC, xD]`

**Training techniques:**
- Focal Loss (γ=2.0) — focuses learning on hard misclassified examples
- Stochastic Weight Averaging (SWA, 20 epochs) — smooths the loss landscape
- Test-Time Augmentation (TTA, 5 augments) — averages predictions for stability
- Cosine annealing with warm restarts (T₀=30, T_mult=2)
- Oversampling of hard classes 14, 17, 18 (2× weight during data generation)

---

## Performance

### Ensemble test results (70,994 samples)

| Model | Val Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Model 1 (seed 42) | 92.99% | 93.03% |
| Model 2 (seed 123) | 92.90% | 92.91% |
| Model 3 (seed 456) | 92.94% | 93.16% |
| **Ensemble (averaged)** | — | **93.63%** |

### Per-class recall (ensemble, test set)

| Class | Equation | Recall |
|-------|----------|--------|
| 0 | A → C | 93.4% |
| 1 | A → D | 93.6% |
| 2 | B → C | 92.4% |
| 3 | B → D | 92.0% |
| 4 | A+B → C | 84.1% |
| 5 | A+B → D | 89.2% |
| 6 | 2A → C | **100.0%** |
| 7 | 2A → D | 99.0% |
| 8 | 2B → C | **100.0%** |
| 9 | 2B → D | **100.0%** |
| 10 | A+B → C → D | 97.2% |
| 11 | 2A → C → D | 97.6% |
| 12 | A → C → D | 97.2% |
| 13 | B → C → D | 99.0% |
| 14 | A+B→C ; 2A→D | 94.9% |
| 15 | A→C ; B→D | 98.4% |
| 16 | 2A→C ; 2B→D | 98.6% |
| 17 | A+B→C→D ; 2A→D | 82.3% |
| 18 | 2A→C→D ; A+B→D | 88.9% |
| 19 | A→C→D ; B→D | 96.2% |
| 20 | A ⇌ C | 88.8% |
| 21 | A ⇌ D | 89.7% |
| 22 | B ⇌ C | 89.9% |
| 23 | B ⇌ D | 92.5% |
| 24 | A+B ⇌ C | 92.5% |
| 25 | A+B ⇌ D | 89.0% |
| 26 | A → C + D | **100.0%** |
| 27 | B → C + D | **100.0%** |

Split-product classes (26–27) and second-order homogeneous classes (6, 8, 9) achieve perfect or near-perfect recall — they produce uniquely distinguishable concentration profiles. Classes 17 and 18 remain the hardest due to overlapping parallel-sequential signatures.

### Noise robustness (ensemble)

Zero-mean Gaussian noise was added at 101 levels (0–50%) to evaluate robustness. Results are in `results/noise_robustness.xlsx`.

| Noise level | Ensemble accuracy |
|-------------|-------------------|
| 0% | 93.63% |
| 5% | 88.81% |
| 10% | 82.25% |
| 25% | 54.35% |
| 50% | 29.78% |

**Most robust classes** (recall >80% past 25% noise): `2A→C`, `2A→D` (survive to 50%), `A→C+D`, `B→C+D`.
**Most fragile classes**: bimolecular reactions (`A+B→*`) break first, typically below 10% noise.

---

## Parameter guide

| Parameter | CLI flag | Valid range | Notes |
|-----------|----------|-------------|-------|
| Reactor type | `--reactor` | PFR, CSTR, CSTR_cascade | CSTR_cascade = 50 tanks in series |
| Molar flow A | `--A0` | 0.05 – 0.5 mol/s | Outside range → extrapolation |
| Molar flow B | `--B0` | 0.05 – 0.5 mol/s | Outside range → extrapolation |
| Temperature | `--T` | 350 – 750 K | Model uses Arrhenius-derived rates |
| Pressure | `--P` | 50,000 – 200,000 Pa | 0.5–2 bar |
| Reactor volume | `--V_total` | any | Default 1.0 m³ |
| Inert flow | `--I0` | 0.0 – 0.2 mol/s | Default 0 |

---

## Troubleshooting

**`ModuleNotFoundError`** — Always prefix commands with `PYTHONPATH=src`

**`ODE solver failed`** — Stiff kinetics with extreme rate constants can fail. Try rate constants closer to the midpoint of the valid range.

**`CSTR convergence warning`** — Rare; solver falls back to Euler step. Increase `--n_cstr` (e.g., 100) for smoother approximation.

**Low confidence for all classes** — Your parameters may be outside the training distribution. Check that T, P, A0, B0 are within the ranges above.

**Classes 17/18 confused** — These two mechanisms are genuinely hard to distinguish at certain rate constant ratios. Check rank 2 in the output — the correct class is usually there.

**Reversible vs irreversible confused** — Reversible classes (20–25) may look similar to their irreversible counterparts (0–5) if the equilibrium constant is very large (Keq >> 1). Use k2 values that give a visible reverse reaction (Keq between 0.1 and 10).
