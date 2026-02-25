"""
Validation & analysis script.

1. Compares the Python ODE solver against the Excel workbook for Classes 0-2 (original).
2. Visualises concentration profiles for all 20 mechanism classes.
3. Highlights the intermediate rise-then-fall behaviour for sequential classes.

Usage (Excel validation -- requires xlwings + Excel):
    PYTHONPATH=src python3 src/validate_excel.py --workbook data/PFR-CSTR-cascade_Calculation-macro.xlsm

Usage (profile gallery only -- no Excel needed):
    PYTHONPATH=src python3 src/validate_excel.py --gallery_only
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse
import json
import math

from ode_solver import (
    solve_pfr, solve_cstr_cascade, MECHANISM_REGISTRY, get_mechanism_names,
)


# ------------------------------------------------------------------
# Representative test parameters for each mechanism class
# ------------------------------------------------------------------

def _get_test_params() -> Dict[int, Dict[str, float]]:
    """Return representative rate constants for each of the 20 mechanisms."""
    return {
        # Group A: Unimolecular (1st order)
        0:  {'k1': 5e-5},
        1:  {'k1': 5e-5},
        2:  {'k1': 5e-5},
        3:  {'k1': 5e-5},
        # Group B: Bimolecular heterogeneous
        4:  {'k1': 2.5e-10},
        5:  {'k1': 2.5e-10},
        # Group C: Bimolecular homogeneous
        6:  {'k1': 2.5e-10},
        7:  {'k1': 2.5e-10},
        8:  {'k1': 2.5e-10},
        9:  {'k1': 2.5e-10},
        # Group D: Sequential
        10: {'k1': 2.5e-10, 'k2': 5e-5},
        11: {'k1': 2.5e-10, 'k2': 5e-5},
        12: {'k1': 5e-5, 'k2': 5e-5},
        13: {'k1': 5e-5, 'k2': 5e-5},
        # Group E: Parallel
        14: {'k1': 2.0e-10, 'k2': 1.5e-10},
        15: {'k1': 5e-5, 'k2': 5e-5},
        16: {'k1': 2.5e-10, 'k2': 2.5e-10},
        # Group F: Parallel-sequential
        17: {'k1': 2.0e-10, 'k2': 1.5e-10, 'k3': 5e-5},
        18: {'k1': 2.5e-10, 'k2': 5e-5, 'k3': 2.0e-10},
        19: {'k1': 5e-5, 'k2': 5e-5, 'k3': 5e-5},
    }


# ------------------------------------------------------------------
# Gallery: all mechanism profiles
# ------------------------------------------------------------------

def _compute_grid(n: int):
    """Compute (nrows, ncols) for a subplot grid that fits n plots."""
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_mechanism_gallery(save_dir: str = './validation') -> None:
    """Plot representative PFR concentration profiles for all mechanisms."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    P = 100_000
    V_total = 1.0
    nI0 = 0.05
    n0 = np.array([0.2, 0.15, 0.0, 0.0])

    test_params = _get_test_params()
    n_mechs = len(MECHANISM_REGISTRY)
    nrows, ncols = _compute_grid(n_mechs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (cid, mech) in enumerate(sorted(MECHANISM_REGISTRY.items())):
        ax = axes[i]
        rc = test_params[cid]

        pfr = solve_pfr(cid, n0, nI0, rc, P, V_total, n_points=200)

        ax.plot(pfr['V'], pfr['A'], 'b-', linewidth=1.5, label='A')
        ax.plot(pfr['V'], pfr['B'], 'r-', linewidth=1.5, label='B')
        ax.plot(pfr['V'], pfr['C'], 'g-', linewidth=1.5, label='C')
        ax.plot(pfr['V'], pfr['D'], 'm-', linewidth=1.5, label='D')

        ax.set_xlabel('Volume (m$^3$)', fontsize=8)
        ax.set_ylabel('Molar flow (mol/s)', fontsize=8)
        ax.set_title(f'Class {cid}: {mech.name}\n({mech.type_label})', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Highlight intermediate behaviour
        if mech.type_label in ('sequential', 'parallel_sequential'):
            c_max_idx = np.argmax(pfr['C'])
            c_max_val = pfr['C'][c_max_idx]
            if c_max_val > 1e-6:
                ax.annotate(
                    f'C peak',
                    xy=(pfr['V'][c_max_idx], c_max_val),
                    xytext=(pfr['V'][c_max_idx] + V_total * 0.1, c_max_val * 1.15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=0.8),
                    fontsize=7, color='green',
                )

    # Hide unused subplots
    for j in range(n_mechs, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Mechanism Gallery: PFR Concentration Profiles (20 Classes)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = save_dir / 'mechanism_gallery_pfr.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gallery saved to: {path}")


def plot_cstr_gallery(save_dir: str = './validation') -> None:
    """Plot representative CSTR cascade profiles for all mechanisms."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    P = 100_000
    V_total = 1.0
    nI0 = 0.05
    n0 = np.array([0.2, 0.15, 0.0, 0.0])
    n_stages = 50

    test_params = _get_test_params()
    n_mechs = len(MECHANISM_REGISTRY)
    nrows, ncols = _compute_grid(n_mechs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (cid, mech) in enumerate(sorted(MECHANISM_REGISTRY.items())):
        ax = axes[i]
        rc = test_params[cid]

        cstr = solve_cstr_cascade(cid, n0, nI0, rc, P, V_total, n_stages)

        ax.plot(cstr['V'], cstr['A'], 'b-o', markersize=1, linewidth=1, label='A')
        ax.plot(cstr['V'], cstr['B'], 'r-o', markersize=1, linewidth=1, label='B')
        ax.plot(cstr['V'], cstr['C'], 'g-o', markersize=1, linewidth=1, label='C')
        ax.plot(cstr['V'], cstr['D'], 'm-o', markersize=1, linewidth=1, label='D')

        ax.set_xlabel('Cumulative Volume (m$^3$)', fontsize=8)
        ax.set_ylabel('Molar flow (mol/s)', fontsize=8)
        ax.set_title(f'Class {cid}: {mech.name}\n({mech.type_label})', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n_mechs, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Mechanism Gallery: CSTR Cascade Profiles (20 Classes)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = save_dir / 'mechanism_gallery_cstr.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gallery saved to: {path}")


# ------------------------------------------------------------------
# Excel validation (original Classes 0-2 mapping)
# ------------------------------------------------------------------

def validate_against_excel(
    workbook_path: str,
    save_dir: str = './validation',
) -> Dict:
    """
    Compare ODE solver output against Excel for the original 3 mechanisms.
    Requires xlwings and Microsoft Excel.

    NOTE: The Excel workbook uses the old class numbering (A+B->C, 2A->D, parallel).
    These correspond to new classes 4, 7, 14 respectively.
    """
    from excel_driver import (
        open_workbook, close_workbook, set_inputs, recalc,
        read_pfr_profile, read_cstr_profile,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    P = 100_000
    V_total = 1.0
    nI0 = 0.05

    # Excel only supports the original 3 mechanisms.
    # Map to new class IDs: A+B->C=4, 2A->D=7, parallel(A+B->C;2A->D)=14
    test_cases = [
        {'name': 'A+B->C (new class 4)', 'class_id': 4,
         'params': {'nA0': 0.2, 'nB0': 0.1, 'nI0': 0.05,
                    'k1': 1e-10, 'k2': 0.0, 'p': P}},
        {'name': '2A->D (new class 7)', 'class_id': 7,
         'params': {'nA0': 0.2, 'nB0': 0.1, 'nI0': 0.05,
                    'k1': 0.0, 'k2': 2e-10, 'p': P}},
        {'name': 'A+B->C ; 2A->D (new class 14)', 'class_id': 14,
         'params': {'nA0': 0.2, 'nB0': 0.15, 'nI0': 0.05,
                    'k1': 1e-10, 'k2': 1e-10, 'p': P}},
    ]

    # Map Excel rate constant names to new ODE solver names
    # Excel uses k1 for A+B->C and k2 for 2A->D
    # New registry: class 4 uses k1 (A+B->C), class 7 uses k1 (2A->D),
    #   class 14 uses k1 (A+B->C) and k2 (2A->D)
    ode_rc_mapping = {
        4:  lambda p: {'k1': p['k1']},           # A+B->C
        7:  lambda p: {'k1': p['k2']},           # 2A->D (Excel k2 → new k1)
        14: lambda p: {'k1': p['k1'], 'k2': p['k2']},  # parallel
    }

    print(f"Opening workbook: {workbook_path}")
    app, wb = open_workbook(workbook_path, visible=False)

    results = []

    try:
        for tc in test_cases:
            print(f"\nValidating: {tc['name']}")
            params = tc['params']
            cid = tc['class_id']

            # --- Excel ---
            set_inputs(wb, params)
            recalc(wb, wait_time=1.0)
            excel_pfr = read_pfr_profile(wb)

            # --- ODE solver ---
            n0 = np.array([params['nA0'], params['nB0'], 0.0, 0.0])
            rc = ode_rc_mapping[cid](params)
            ode_pfr = solve_pfr(cid, n0, nI0, rc, P, V_total,
                                n_points=len(excel_pfr['A']))

            # --- Compare ---
            species_to_check = ['A', 'B', 'C', 'D']
            max_errors = {}

            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            for idx, spec in enumerate(species_to_check):
                excel_vals = excel_pfr[spec]
                ode_vals = ode_pfr[spec]

                min_len = min(len(excel_vals), len(ode_vals))
                e = excel_vals[:min_len]
                o = ode_vals[:min_len]

                abs_err = np.abs(e - o)
                max_abs = np.max(abs_err)
                denom = np.maximum(np.abs(e), 1e-15)
                max_rel = np.max(abs_err / denom) * 100

                max_errors[spec] = {'max_abs': float(max_abs),
                                    'max_rel_pct': float(max_rel)}

                ax = axes[idx]
                ax.plot(excel_pfr['V'][:min_len], e, 'b-', label=f'{spec} (Excel)')
                ax.plot(ode_pfr['V'][:min_len], o, 'r--', label=f'{spec} (ODE)')
                ax.set_xlabel('V (m$^3$)')
                ax.set_ylabel(f'n({spec}) (mol/s)')
                ax.set_title(f'{spec}: max err = {max_abs:.2e}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Excel vs ODE: {tc['name']}", fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            path = save_dir / f"validation_class{cid}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Plot saved: {path}")

            for spec, errs in max_errors.items():
                print(f"  {spec}: max_abs={errs['max_abs']:.2e}, "
                      f"max_rel={errs['max_rel_pct']:.2f}%")

            results.append({
                'test': tc['name'],
                'class_id': cid,
                'errors': max_errors,
            })

    finally:
        close_workbook(app, wb, save=False)

    summary_path = save_dir / 'validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nValidation summary saved to: {summary_path}")

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate ODE solver and visualise mechanism profiles'
    )
    parser.add_argument('--workbook', type=str, default=None,
                        help='Path to Excel workbook (for Excel validation)')
    parser.add_argument('--gallery_only', action='store_true',
                        help='Only generate profile galleries (no Excel needed)')
    parser.add_argument('--save_dir', type=str, default='./validation')

    args = parser.parse_args()

    # Always generate galleries
    print("Generating mechanism galleries...")
    plot_mechanism_gallery(args.save_dir)
    plot_cstr_gallery(args.save_dir)

    # Excel validation (optional)
    if not args.gallery_only and args.workbook:
        print("\nRunning Excel validation...")
        validate_against_excel(args.workbook, args.save_dir)
    elif not args.gallery_only:
        print("\nSkipping Excel validation (no --workbook provided).")
        print("Use: --workbook <path> to enable, or --gallery_only to suppress this message.")

    print("\nDone!")


if __name__ == '__main__':
    main()
