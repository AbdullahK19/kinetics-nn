# Kinetics Neural Network — Reaction Mechanism Classifier

A deep learning pipeline that identifies **which chemical reaction mechanism** is operating inside a reactor, purely from concentration profiles. Feed it reactor outputs; it tells you the mechanism.

**94.49% validation accuracy** across 20 mechanism classes, including complex parallel-sequential reactions.

---

## What it does

Given a set of reactor conditions (temperature, pressure, initial flows, rate constants), the model:

1. Runs an ODE simulation to produce a concentration profile
2. Classifies the profile into one of 20 reaction mechanisms
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

## The 20 mechanism classes

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

---

## Rate constant ranges

Rate constants must stay within these ranges for reliable predictions (the model was trained on these ranges):

| Order | Applicable to | Valid range | Units |
|-------|--------------|-------------|-------|
| 1st order | Groups A, D (k2), E (k1/k2 for class 15), F (class 19) | `1e-6` – `1e-4` | mol s⁻¹ m⁻³ Pa⁻¹ |
| 2nd order | Groups B, C, D (k1), E (k1/k2 for 14/16), F | `1e-11` – `5e-10` | mol s⁻¹ m⁻³ Pa⁻² |

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

---

## Full predict.py reference

```bash
PYTHONPATH=src python3 src/predict.py \
  --ode \
  --reactor    PFR | CSTR | CSTR_cascade \
  --mechanism  0-19 \
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

# Parallel-sequential (hardest class) — PFR
PYTHONPATH=src python3 src/predict.py --ode --reactor PFR \
  --mechanism 17 --A0 0.3 --B0 0.2 --T 550 --P 100000 \
  --rc k1=2e-10 k2=1e-10 k3=5e-5

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

## Retrain from scratch

If you want to regenerate data and retrain the model:

```bash
# 1. Generate 50K simulation runs
PYTHONPATH=src python3 src/dataset_builder.py --n_runs 50000 --save_dir ./data

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
│   ├── ode_solver.py       # 20-class mechanism registry + PFR/CSTR solvers
│   ├── dataset_builder.py  # Dataset generation (ODE solver)
│   ├── preprocess.py       # z-score normalisation, train/val/test split
│   ├── model.py            # ResConv1D + SE attention (and CNN/GRU/LSTM alternatives)
│   ├── train.py            # Training: FocalLoss, SWA, TTA, cosine LR, --resume
│   ├── predict.py          # Inference: MechanismPredictor class + CLI
│   ├── validate_excel.py   # Gallery plots + ODE vs Excel cross-validation
│   └── excel_driver.py     # Excel integration (optional, classes 0–2 only)
├── data/
│   ├── train.npz           # 33,352 training samples
│   ├── val.npz             # 7,147 validation samples
│   ├── test.npz            # 7,147 test samples
│   ├── scaler.json         # z-score normalisation parameters
│   └── dataset_info.json   # Dataset statistics + feature names
├── models/
│   ├── best_model.pt           # Trained model (SWA-averaged, 94.49% val acc)
│   ├── training_config.json    # Architecture + training hyperparameters
│   ├── classification_report.txt
│   ├── test_report.txt         # Per-class results on held-out test set
│   ├── training_curves.png
│   └── confusion_matrix.png
├── validation/
│   ├── mechanism_gallery_pfr.png   # Concentration profiles for all 20 classes
│   ├── mechanism_gallery_cstr.png
│   └── intermediate_detail.png
└── configs/
    ├── default_config.json       # 20-class production config
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
| Head | AdaptiveAvgPool → FC(256) → FC(128) → FC(20) |
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

### Test set (7,147 held-out samples, never seen during training)

| Metric | Standard | TTA ×5 |
|--------|----------|--------|
| Overall accuracy | 94.26% | 94.49% |
| Macro avg F1 | 0.96 | 0.96 |

### Per-class F1 scores

| Class | Equation | F1 |
|-------|----------|----|
| 0 | A → C | 0.99 |
| 1 | A → D | 1.00 |
| 2 | B → C | 0.99 |
| 3 | B → D | 1.00 |
| 4 | A+B → C | 0.97 |
| 5 | A+B → D | 0.96 |
| 6 | 2A → C | 0.99 |
| 7 | 2A → D | 0.96 |
| 8 | 2B → C | 1.00 |
| 9 | 2B → D | 1.00 |
| 10 | A+B → C → D | 0.91 |
| 11 | 2A → C → D | 0.95 |
| 12 | A → C → D | 0.97 |
| 13 | B → C → D | 0.98 |
| 14 | A+B→C ; 2A→D | 0.91 |
| 15 | A→C ; B→D | 0.97 |
| 16 | 2A→C ; 2B→D | 0.96 |
| 17 | A+B→C→D ; 2A→D | **0.82** |
| 18 | 2A→C→D ; A+B→D | **0.85** |
| 19 | A→C→D ; B→D | 0.96 |

Classes 17 and 18 are the hardest — they are structurally very similar to each other and to simpler mechanisms. When the model misses, it always places the correct class at rank 2.

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
