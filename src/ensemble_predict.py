"""
ensemble_predict.py — Ensemble inference across multiple trained models.

Loads best_model.pt from each ensemble directory, averages their softmax
probabilities, and returns the combined prediction.

Usage (ODE-based):
    PYTHONPATH=src python3 src/ensemble_predict.py \\
        --ensemble_dirs models/ensemble_1 models/ensemble_2 models/ensemble_3 \\
        --ode --reactor PFR --mechanism 4 \\
        --A0 0.2 --B0 0.1 --I0 0.05 --T 550 --P 100000 --rc k1=1e-10

Usage (profile file):
    PYTHONPATH=src python3 src/ensemble_predict.py \\
        --ensemble_dirs models/ensemble_1 models/ensemble_2 models/ensemble_3 \\
        --profile my_profile.npy
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch


# ── helpers copied/adapted from predict.py ──────────────────────────────────

def load_model_from_dir(model_dir: Path, device: torch.device):
    """Load best_model.pt + training_config.json from a directory."""
    from model import create_model

    config_path = model_dir / "training_config.json"
    ckpt_path   = model_dir / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    model = create_model(
        model_type      = cfg.get("model_type", "rescnn"),
        num_features    = cfg.get("num_features", 16),
        num_classes     = cfg.get("num_classes", 28),
        seq_length      = cfg.get("grid_length", 128),
        hidden_channels = cfg.get("hidden_channels", [64, 128, 256]),
        fc_hidden       = cfg.get("fc_hidden", 256),
        dropout         = cfg.get("dropout", 0.25),
        blocks_per_stage= cfg.get("blocks_per_stage", 2),
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def load_scaler(data_dir: Path):
    scaler_path = data_dir / "scaler.json"
    with open(scaler_path) as f:
        s = json.load(f)
    return np.array(s["mean"]), np.array(s["std"])


def normalize(profile: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (profile - mean) / (std + 1e-8)


def load_class_names(data_dir: Path) -> dict:
    info_path = data_dir / "dataset_info.json"
    with open(info_path) as f:
        d = json.load(f)
    return {int(k): v for k, v in d["mechanism_classes"].items()}


# ── ODE profile generation (mirrors predict.py logic) ───────────────────────

def build_profile_from_ode(args, cfg) -> np.ndarray:
    """Run ODE solver and return raw (grid_length, num_features) array.

    Mirrors predict.py preprocess_profile() feature construction exactly.
    Normalization is applied by the caller using normalize().
    """
    from ode_solver import solve_pfr, solve_cstr_cascade
    from dataset_builder import DatasetBuilder

    grid_length = cfg.get("grid_length", 128)

    # Parse rate constants
    rate_constants = {}
    if args.rc:
        for item in args.rc:
            k, v = item.split("=")
            rate_constants[k.strip()] = float(v.strip())

    mech_id = args.mechanism
    n_cstr  = getattr(args, "n_cstr", 50)
    n0  = np.array([args.A0, args.B0, args.C0, args.D0])
    nI0 = args.I0

    if args.reactor.upper() == "PFR":
        profile = solve_pfr(mech_id, n0, nI0, rate_constants, args.P,
                            V_total=1.0, n_points=200)
    elif args.reactor.upper() == "CSTR":
        profile = solve_cstr_cascade(mech_id, n0, nI0, rate_constants, args.P,
                                     V_total=1.0, n_stages=1)
    else:
        profile = solve_cstr_cascade(mech_id, n0, nI0, rate_constants, args.P,
                                     V_total=1.0, n_stages=n_cstr)

    # Interpolate to fixed grid: (grid_length, 5) — [A, B, C, D, n_total]
    builder = DatasetBuilder(n_runs=0, grid_length=grid_length, use_ode_solver=True)
    profile_interp = builder.interpolate_profile(profile, args.reactor)

    # Encode scalar features (mirrors predict.py constants exactly)
    _REACTOR_ENCODING = {'PFR': 0.0, 'CSTR': 0.5, 'CSTR_cascade': 1.0}
    _T_MIN, _T_MAX = 350.0, 750.0
    _P_MIN, _P_MAX = 50_000.0, 200_000.0

    reactor_enc = _REACTOR_ENCODING.get(args.reactor, 0.0)
    T_norm = (args.T - _T_MIN) / (_T_MAX - _T_MIN)
    P_norm = (args.P - _P_MIN) / (_P_MAX - _P_MIN)

    reactor_feature = np.full((grid_length, 1), reactor_enc)
    T_feature       = np.full((grid_length, 1), T_norm)
    P_feature       = np.full((grid_length, 1), P_norm)

    # Derivative features: dA/dV, dB/dV, dC/dV, dD/dV
    derivatives = np.gradient(profile_interp[:, :4], axis=0)

    # Mole fraction features: xA, xB, xC, xD
    n_tot_safe = np.maximum(profile_interp[:, 4:5], 1e-10)
    mole_fractions = profile_interp[:, :4] / n_tot_safe

    features = np.concatenate(
        [profile_interp, reactor_feature, T_feature, P_feature,
         derivatives, mole_fractions], axis=1
    )  # (grid_length, 16)

    return features


# ── ensemble inference ───────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict(models, profile_norm: np.ndarray, device: torch.device,
                     n_tta: int = 0):
    """
    Average softmax probabilities over all models (and optionally TTA passes).
    profile_norm: (grid_length, num_features) already normalised
    """
    x = torch.tensor(profile_norm, dtype=torch.float32).unsqueeze(0).to(device)
    # shape: (1, grid_length, num_features)

    all_probs = []
    for model in models:
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        all_probs.append(probs)

        if n_tta > 0:
            # Simple augmentation: add small Gaussian noise
            for _ in range(n_tta):
                noise = torch.randn_like(x) * 0.05
                logits_aug = model(x + noise)
                all_probs.append(torch.softmax(logits_aug, dim=-1).squeeze(0).cpu().numpy())

    mean_probs = np.mean(all_probs, axis=0)
    return mean_probs


def print_predictions(probs: np.ndarray, class_names: dict, threshold: float,
                      top_k: int = 5):
    ranked = np.argsort(probs)[::-1]
    print(f"\n{'Rank':<5} {'Class':>5}  {'Mechanism':<35} {'Confidence':>11}")
    print("-" * 60)
    shown = 0
    for rank, idx in enumerate(ranked):
        p = probs[idx]
        if p < threshold and shown >= 1:
            break
        name = class_names.get(idx, f"Class {idx}")
        marker = " <-- predicted" if rank == 0 else ""
        print(f"  {rank+1:<4} {idx:>5}  {name:<35} {p*100:>9.2f}%{marker}")
        shown += 1
        if shown >= top_k:
            break
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ensemble mechanism predictor")
    p.add_argument("--ensemble_dirs", nargs="+", required=True,
                   help="Directories containing best_model.pt + training_config.json")
    p.add_argument("--data_dir", default="./data",
                   help="Directory with scaler.json and dataset_info.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.005,
                   help="Minimum probability to display (default 0.5%%)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--tta", type=int, default=0,
                   help="Test-time augmentation passes per model (default 0)")

    # Profile source: either a file or ODE generation
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--profile", help=".npy file with (grid_length, num_features) profile")
    src.add_argument("--ode", action="store_true", help="Generate profile via ODE solver")

    # ODE arguments
    p.add_argument("--reactor",   default="PFR", choices=["PFR", "CSTR", "CSTR_cascade"])
    p.add_argument("--mechanism", type=int, default=0,
                   help="True mechanism class ID (0-27); used to run ODE solver")
    p.add_argument("--A0",  type=float, default=0.1)
    p.add_argument("--B0",  type=float, default=0.1)
    p.add_argument("--C0",  type=float, default=0.0)
    p.add_argument("--D0",  type=float, default=0.0)
    p.add_argument("--I0",  type=float, default=0.0)
    p.add_argument("--T",   type=float, default=500.0, help="Temperature (K)")
    p.add_argument("--P",   type=float, default=101325.0, help="Pressure (Pa)")
    p.add_argument("--rc",  nargs="+", help="Rate constants e.g. k1=1e-5 k2=2e-5")
    p.add_argument("--n_cstr", type=int, default=5)

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # Load all models
    print(f"\nLoading {len(args.ensemble_dirs)} ensemble models...")
    models = []
    cfgs   = []
    for d in args.ensemble_dirs:
        model, cfg = load_model_from_dir(Path(d), device)
        models.append(model)
        cfgs.append(cfg)
        print(f"  ✓ {d}  (epoch {cfg.get('best_epoch','?')}, val_acc={cfg.get('best_val_acc',0)*100:.2f}%)")

    cfg = cfgs[0]   # use first config for grid/feature dims

    # Load scaler and class names
    mean, std = load_scaler(data_dir)
    class_names = load_class_names(data_dir)

    # Build profile
    if args.ode:
        print(f"\nGenerating ODE profile (reactor={args.reactor}, mechanism={args.mechanism})...")
        profile_raw = build_profile_from_ode(args, cfg)
    else:
        loaded = np.load(args.profile)
        if isinstance(loaded, np.ndarray):
            profile_raw = loaded
        else:
            # .npz file — take the first stored array
            profile_raw = loaded[list(loaded.keys())[0]]

    profile_norm = normalize(profile_raw, mean, std)

    # Ensemble prediction
    probs = ensemble_predict(models, profile_norm, device, n_tta=args.tta)

    predicted_class = int(np.argmax(probs))
    predicted_name  = class_names.get(predicted_class, f"Class {predicted_class}")

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE PREDICTION  ({len(models)} models" +
          (f" × {args.tta+1} TTA passes" if args.tta else "") + ")")
    print(f"  Predicted: [{predicted_class}] {predicted_name}")
    print(f"  Confidence: {probs[predicted_class]*100:.2f}%")
    print(f"{'='*60}")

    print_predictions(probs, class_names, args.threshold, args.top_k)

    if args.ode:
        true_name = class_names.get(args.mechanism, f"Class {args.mechanism}")
        correct = (predicted_class == args.mechanism)
        print(f"  True mechanism: [{args.mechanism}] {true_name}")
        print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
        if not correct:
            true_prob = probs[args.mechanism]
            rank = int(np.sum(probs > true_prob)) + 1
            print(f"  True class probability: {true_prob*100:.2f}% (rank {rank})")
        print()


if __name__ == "__main__":
    main()
