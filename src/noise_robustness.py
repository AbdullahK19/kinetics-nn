"""
Noise Robustness Analysis
=========================
Evaluates how well the trained 28-class mechanism classifier holds up
when increasing levels of Gaussian noise are added to the test data.

Noise model
-----------
For each sample X (shape: 128 × 16), noise is signal-proportional:
    noise_std = (p / 100) × RMS(X)
    X_noisy   = X + N(0, noise_std²)

This is equivalent to a signal-to-noise ratio analysis and is physically
meaningful — it models instrument/measurement noise that scales with the
signal magnitude.

Outputs (saved to validation/)
-------------------------------
- overall_accuracy_vs_noise.png   : overall test accuracy at each noise level
- group_accuracy_vs_noise.png     : per-group accuracy (groups A–I) vs noise
- class_recall_heatmap.png        : 28 classes × noise levels, colour = recall

Usage
-----
    PYTHONPATH=src python3 src/noise_robustness.py
    PYTHONPATH=src python3 src/noise_robustness.py --noise_levels 0 5 10 25 50
    PYTHONPATH=src python3 src/noise_robustness.py --seed 123 --batch_size 256
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import recall_score

from model import create_model
from preprocess import StandardScaler
from ode_solver import MECHANISM_REGISTRY

# ── Mechanism group definitions ────────────────────────────────────────────────
GROUPS = {
    'A (simple)':        list(range(0, 6)),
    'B (consecutive)':   list(range(6, 11)),
    'C (parallel)':      list(range(11, 14)),
    'D (mixed)':         list(range(14, 20)),
    'G (rev. uni.)':     list(range(20, 24)),
    'H (rev. bi.)':      list(range(24, 26)),
    'I (split-prod.)':   list(range(26, 28)),
}

# Colours for the group plot — one per group
GROUP_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2']

DEFAULT_NOISE_LEVELS = [0, 1, 2, 5, 10, 20, 30, 50]


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(model_path: str, config_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = json.load(f)

    num_classes  = cfg.get('num_classes', len(MECHANISM_REGISTRY))
    num_features = cfg.get('num_features', 16)

    extra = {}
    if 'hidden_channels' in cfg:
        extra['hidden_channels'] = tuple(cfg['hidden_channels'])
    if 'fc_hidden' in cfg:
        extra['fc_hidden'] = cfg['fc_hidden']
    if 'blocks_per_stage' in cfg:
        extra['blocks_per_stage'] = cfg['blocks_per_stage']

    model = create_model(
        model_type=cfg['model_type'],
        num_features=num_features,
        num_classes=num_classes,
        seq_length=cfg.get('grid_length', 128),
        dropout=0.0,
        **extra
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, num_classes


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batched(model, X_tensor: torch.Tensor, batch_size: int,
                    device: torch.device) -> np.ndarray:
    """Run batched inference; return predicted class indices."""
    preds = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        logits = model(batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ── Noise application ──────────────────────────────────────────────────────────

def add_noise(X: np.ndarray, noise_pct: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add signal-proportional Gaussian noise.
    noise_std per sample = (noise_pct / 100) × RMS of that sample.
    """
    if noise_pct == 0:
        return X
    # RMS amplitude per sample: shape (N, 1, 1)
    rms = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
    rms = np.clip(rms, 1e-8, None)          # avoid div-by-zero on zero samples
    noise_std = (noise_pct / 100.0) * rms
    noise = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


# ── Per-class recall helper ────────────────────────────────────────────────────

def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    recalls = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            recalls[c] = np.nan
        else:
            recalls[c] = (y_pred[mask] == c).mean()
    return recalls


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_overall_accuracy(noise_levels, accuracies, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, accuracies, 'o-', color='#1f77b4',
            linewidth=2, markersize=7, markerfacecolor='white',
            markeredgewidth=2, label='Overall accuracy')

    # Annotate values
    for x, y in zip(noise_levels, accuracies):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8)

    # Reference lines
    for thresh, style, label in [(90, '--', '90%'), (80, ':', '80%'),
                                  (70, '-.', '70%'), (50, '--', '50%')]:
        ax.axhline(thresh, color='grey', linestyle=style, linewidth=0.8,
                   alpha=0.6, label=label)

    ax.set_xlabel('Added noise level (% of signal RMS)', fontsize=12)
    ax.set_ylabel('Test accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy vs. Noise Level', fontsize=14, fontweight='bold')
    ax.set_xlim(-1, max(noise_levels) + 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_group_accuracy(noise_levels, group_accs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for (group_name, accs), colour in zip(group_accs.items(), GROUP_COLOURS):
        ax.plot(noise_levels, accs, 'o-', color=colour, linewidth=2,
                markersize=6, label=group_name)

    ax.set_xlabel('Added noise level (% of signal RMS)', fontsize=12)
    ax.set_ylabel('Group accuracy (%)', fontsize=12)
    ax.set_title('Per-Group Accuracy vs. Noise Level', fontsize=14,
                 fontweight='bold')
    ax.set_xlim(-1, max(noise_levels) + 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_recall_heatmap(noise_levels, recall_matrix: np.ndarray,
                        class_names: list, out_path: Path):
    """
    recall_matrix: shape (num_classes, num_noise_levels)
    """
    n_classes, n_noise = recall_matrix.shape
    fig_h = max(10, n_classes * 0.38)
    fig, ax = plt.subplots(figsize=(max(8, n_noise * 1.2), fig_h))

    im = ax.imshow(recall_matrix * 100, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, interpolation='nearest')

    # Axis labels
    ax.set_xticks(range(n_noise))
    ax.set_xticklabels([f'{p}%' for p in noise_levels], fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=7.5)
    ax.set_xlabel('Noise level (% of signal RMS)', fontsize=11)
    ax.set_ylabel('Mechanism class', fontsize=11)
    ax.set_title('Per-Class Recall vs. Noise Level', fontsize=13,
                 fontweight='bold')

    # Annotate cells
    for r in range(n_classes):
        for c in range(n_noise):
            val = recall_matrix[r, c]
            if not np.isnan(val):
                text_colour = 'black' if 0.25 < val < 0.85 else 'white'
                ax.text(c, r, f'{val*100:.0f}',
                        ha='center', va='center',
                        fontsize=6.5, color=text_colour)

    # Draw group separators
    group_boundaries = [5.5, 10.5, 13.5, 19.5, 23.5, 25.5]
    for b in group_boundaries:
        ax.axhline(b, color='white', linewidth=1.8)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('Recall (%)', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Noise robustness analysis for the 28-class mechanism classifier')
    parser.add_argument('--model_path',  default='models/best_model.pt')
    parser.add_argument('--config_path', default='models/training_config.json')
    parser.add_argument('--scaler_path', default='data/scaler.json')
    parser.add_argument('--test_path',   default='data/test.npz')
    parser.add_argument('--dataset_info',default='data/dataset_info.json')
    parser.add_argument('--out_dir',     default='validation')
    parser.add_argument('--noise_levels', nargs='+', type=float,
                        default=DEFAULT_NOISE_LEVELS,
                        help='Noise levels in %% of signal RMS')
    parser.add_argument('--batch_size',  type=int, default=512)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--device',      default='cpu')
    args = parser.parse_args()

    rng    = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading test data...")
    test_data = np.load(args.test_path)
    X_test = test_data['X'].astype(np.float32)   # (N, 128, 16)
    y_test = test_data['y'].astype(int)           # (N,)
    print(f"  Test samples: {len(y_test):,}")

    # ── Load class names ───────────────────────────────────────────────────────
    with open(args.dataset_info) as f:
        info = json.load(f)
    mech_classes = info['mechanism_classes']       # dict: str(int) -> str
    num_classes = len(mech_classes)
    class_names = [mech_classes[str(i)] for i in range(num_classes)]

    # ── Load model ─────────────────────────────────────────────────────────────
    print("Loading model...")
    model, _ = load_model(args.model_path, args.config_path, device)
    print(f"  Model loaded from {args.model_path}")

    noise_levels = sorted(args.noise_levels)

    # Storage
    overall_accuracies = []
    group_accs = {g: [] for g in GROUPS}
    recall_matrix = np.zeros((num_classes, len(noise_levels)))

    # ── Noise sweep ────────────────────────────────────────────────────────────
    print(f"\nRunning noise sweep: {noise_levels}%")
    print("-" * 55)

    for ni, noise_pct in enumerate(noise_levels):
        # Add noise
        X_noisy = add_noise(X_test, noise_pct, rng)
        X_tensor = torch.from_numpy(X_noisy)

        # Inference
        y_pred = predict_batched(model, X_tensor, args.batch_size, device)

        # Overall accuracy
        acc = 100.0 * (y_pred == y_test).mean()
        overall_accuracies.append(acc)

        # Per-class recall
        recalls = per_class_recall(y_test, y_pred, num_classes)
        recall_matrix[:, ni] = recalls

        # Per-group accuracy
        for group_name, class_ids in GROUPS.items():
            mask = np.isin(y_test, class_ids)
            if mask.sum() > 0:
                g_acc = 100.0 * (y_pred[mask] == y_test[mask]).mean()
            else:
                g_acc = float('nan')
            group_accs[group_name].append(g_acc)

        print(f"  Noise {noise_pct:5.1f}%  →  Overall accuracy: {acc:.2f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_overall_accuracy(
        noise_levels, overall_accuracies,
        out_dir / 'overall_accuracy_vs_noise.png'
    )

    plot_group_accuracy(
        noise_levels, group_accs,
        out_dir / 'group_accuracy_vs_noise.png'
    )

    plot_recall_heatmap(
        noise_levels, recall_matrix, class_names,
        out_dir / 'class_recall_heatmap.png'
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n── Summary ────────────────────────────────────────────────────")
    print(f"{'Noise':>8}  {'Accuracy':>10}")
    for p, a in zip(noise_levels, overall_accuracies):
        print(f"  {p:5.1f}%   {a:8.2f}%")

    # Find approximate "breaking point" (first noise level where acc < 70%)
    for p, a in zip(noise_levels, overall_accuracies):
        if a < 70.0:
            print(f"\n  Breaking point (< 70% accuracy): ~{p}% noise")
            break
    else:
        print(f"\n  Model stays above 70% accuracy across all tested noise levels.")

    print("\nDone. All plots saved to:", out_dir.resolve())


if __name__ == '__main__':
    main()
