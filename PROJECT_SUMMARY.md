# Project Summary

## What was built

An end-to-end deep learning pipeline for identifying chemical reaction mechanisms from concentration profiles in PFR and CSTR reactors.

- **28 mechanism classes** — simple, sequential, parallel, parallel-sequential, reversible, and split-product reactions
- **92.24% validation accuracy** on 14,176 held-out samples
- **No Excel required** — Python ODE solver covers all mechanisms
- **Ranked predictions** with probabilities and configurable threshold

---

## Final model performance

| | Value |
|--|-------|
| Validation accuracy | 92.24% |
| Test accuracy | 92.21% |
| Macro avg F1 | 0.93 |

Trained on 100,000 ODE-simulated runs (~70,000 train / ~15,000 val / ~15,000 test).

---

## Architecture

**ResConv1DClassifier** — residual 1D CNN with Squeeze-and-Excitation attention blocks.

- Input: (batch, 128 timesteps, 16 features)
- 3 stages: channels 64 → 128 → 256, with 2 ResBlocks + SE attention per stage
- 3-layer FC head: 256 → 128 → 28 classes
- **1,275,444 parameters**
- Trained with: Focal Loss (γ=2.0), SWA (20 epochs), cosine warm restarts, mixup, TTA

---

## Source files

| File | Purpose |
|------|---------|
| `src/ode_solver.py` | Registry of all 28 mechanisms (Groups A–I), PFR and CSTR cascade ODE solvers |
| `src/dataset_builder.py` | ODE simulation, 16-channel feature construction, oversampling, Keq constraint for reversible classes |
| `src/preprocess.py` | z-score scaler, stratified 70/15/15 train/val/test split |
| `src/model.py` | ResConv1DClassifier (+ CNN, GRU, LSTM alternatives) |
| `src/train.py` | Training loop: FocalLoss, SWA, TTA, --resume, cosine LR |
| `src/predict.py` | MechanismPredictor class + CLI |
| `src/validate_excel.py` | Gallery plots, ODE vs Excel cross-validation (classes 0–2) |
| `src/excel_driver.py` | Excel integration via xlwings (optional, validation only) |

---

## Key design decisions

### Why 16 features instead of 6?
Derivatives (dA/dV etc.) capture the *rate of change* of each species, which is more discriminative than raw concentrations alone — especially for sequential mechanisms where C rises then falls. Mole fractions capture the compositional signature independently of absolute flow rates.

**16 channels:** `[A, B, C, D, n_total, reactor_type, T_norm, P_norm, dA/dV, dB/dV, dC/dV, dD/dV, xA, xB, xC, xD]`

### Why ResConv1D instead of plain CNN?
Residual connections allow deeper networks without gradient vanishing. The SE attention blocks let the model learn which channels (species/features) matter most for each class. This gave a significant accuracy boost over the baseline CNN.

### Why Focal Loss?
Classes 17 and 18 (parallel-sequential) are the hardest — they share sub-reactions with simpler classes. Focal Loss down-weights easy examples (simple mechanisms at 99%+ probability) and focuses gradient updates on the hard boundary cases.

### Why SWA?
Stochastic Weight Averaging averages model weights across the final training epochs, which smooths the loss landscape and improves generalisation.

### Why oversample classes 14, 17, 18?
These three classes are harder to classify (they contain sub-reactions present in simpler classes). Giving them 2× sampling weight during data generation ensures the model sees more boundary examples during training.

### Why Arrhenius-sampled rate constants?
Sampling `k_ref` at T=550K then computing `k(T)` via Arrhenius ensures rate constants are physically self-consistent across the temperature range 350–750K, which is what would be observed in a real reactor.

### Why Keq constraints for reversible classes (20–25)?
Without constraining the equilibrium constant, a reversible mechanism with Keq >> 1 produces a profile indistinguishable from an irreversible one. Constraining Keq to [0.1, 10] ensures the reverse reaction is visible and the mechanism is distinguishable during training.

---

## Training improvements over v0.2

| Aspect | v0.2 (original) | Current |
|--------|----------------|---------|
| Classes | 5 | 28 |
| Features | 6 (raw concentrations only) | 16 (+ derivatives, mole fractions, T, P) |
| Architecture | Plain CNN, 49K params | ResConv1D + SE, 1.275M params |
| Dataset | 3,000 runs | 100,000 runs |
| Loss | CrossEntropy | Focal Loss (γ=2.0) |
| Post-training | None | SWA (20 epochs) |
| Inference | Single pass | TTA ×5 |
| Val accuracy | ~83% (1K samples) | **92.24%** |

---

## Known limitations

1. **Classes 17 and 18** — 84% F1. These parallel-sequential mechanisms are hard to distinguish from each other and from simpler sub-mechanisms at certain rate constant ratios. The correct answer is always in the top-2.

2. **Reversible vs irreversible** — Classes 20–25 (reversible) and 0–5 (irreversible) share the same stoichiometry; they differ only by the presence of a reverse reaction. Very large Keq values make these look near-identical.

3. **Simulation-only training** — The model has never seen real experimental data. Real measurements include sensor noise, baseline drift, and imperfect sampling that are absent from ODE simulations.

4. **No input range validation** — Passing T=1000K or A0=5 mol/s will not raise an error; the model will extrapolate outside its training distribution and give unreliable output. Keep parameters within the ranges listed in README.md.

5. **Excel cross-validation** — Requires xlwings + Microsoft Excel (macOS/Windows only). The ODE solver is the primary path; Excel is only used for cross-checking classes 0–2.

---

## File outputs after training

| File | Contents |
|------|---------|
| `models/best_model.pt` | SWA-averaged model checkpoint (includes optimizer state) |
| `models/training_config.json` | All hyperparameters + best_val_acc + architecture spec |
| `models/classification_report.txt` | Per-class precision/recall/F1 on validation set |
| `models/test_report.txt` | Per-class results on held-out test set (standard + TTA) |
| `models/training_curves.png` | Loss and accuracy vs epoch |
| `models/confusion_matrix.png` | 28×28 confusion matrix on validation set |
| `data/scaler.json` | 16-feature mean/std for z-score normalisation |
| `data/split_info.json` | Train/val/test sizes and split parameters |
