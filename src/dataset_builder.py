"""
Dataset builder for generating labeled training data.

Supports two simulation backends:
  1. Python ODE solver (default) -- no Excel dependency, supports all 20 mechanisms
  2. Excel workbook via xlwings   -- original backend, classes 0-2 only

See ode_solver.py for the full 20-class mechanism taxonomy.
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from ode_solver import (
    solve_pfr, solve_cstr_cascade, MECHANISM_REGISTRY, get_mechanism_names,
    R_GAS, arrhenius_k,
)


class DatasetBuilder:
    """Build labeled dataset from reactor simulations."""

    def __init__(
        self,
        workbook_path: str = '',
        n_runs: int = 1000,
        grid_length: int = 128,
        random_seed: int = 42,
        save_dir: str = './data',
        use_ode_solver: bool = True,
        mechanism_classes_to_generate: List[int] = None,
        P: float = 100_000,
        V_total: float = 1.0,
        n_cstr_stages: int = 50,
    ):
        self.workbook_path = workbook_path
        self.n_runs = n_runs
        self.grid_length = grid_length
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_ode_solver = use_ode_solver

        np.random.seed(random_seed)

        # Which mechanism classes to include
        self.mechanism_classes_to_generate = (
            mechanism_classes_to_generate
            if mechanism_classes_to_generate is not None
            else list(MECHANISM_REGISTRY.keys())
        )

        # Mechanism name lookup (used by metadata and save_dataset)
        self.mechanism_classes = {
            cid: MECHANISM_REGISTRY[cid].name
            for cid in self.mechanism_classes_to_generate
        }

        # Reactor parameters for ODE solver
        self.P = P
        self.V_total = V_total
        self.n_cstr_stages = n_cstr_stages

        # Oversampling for hard classes (complex multi-step/parallel mechanisms)
        self.oversample_classes = {14, 17, 18}
        self.oversample_factor = 2.0

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def sample_parameters(self, mechanism_class: int) -> Dict[str, float]:
        """
        Sample random parameters for a given mechanism class.

        Rate constants are computed via the Arrhenius equation:
            k(T) = A_pre * exp(-Ea / (R * T))

        A reference k at T_ref=550 K is sampled from the mechanism's existing
        ranges, then A_pre is back-calculated so that k(T_ref) == k_ref.
        Finally, k at the sampled T is computed.
        """
        T_REF = 550.0  # K — reference temperature for A_pre back-calculation

        nA0 = np.random.uniform(0.05, 0.5)
        nB0 = np.random.uniform(0.05, 0.5)
        nI0 = np.random.uniform(0.0, 0.2)
        P   = np.random.uniform(50_000, 200_000)   # Pa (variable pressure)
        T   = np.random.uniform(350, 750)           # K  (variable temperature)

        mechanism = MECHANISM_REGISTRY[mechanism_class]

        # Sample rate constants via Arrhenius equation
        rate_constants = {}
        arrhenius_meta = {}
        for rc_name, (lo, hi) in mechanism.rate_constant_ranges.items():
            Ea    = np.random.uniform(15_000, 60_000)      # J/mol activation energy
            k_ref = np.random.uniform(lo, hi)              # k at T_ref
            A_pre = k_ref / np.exp(-Ea / (R_GAS * T_REF)) # back-calculate A_pre
            k_T   = arrhenius_k(A_pre, Ea, T)              # actual k at sampled T
            rate_constants[rc_name] = k_T
            arrhenius_meta[f'Ea_{rc_name}']    = Ea
            arrhenius_meta[f'A_pre_{rc_name}'] = A_pre
            arrhenius_meta[f'k_ref_{rc_name}'] = k_ref

        params = {
            'nA0': nA0,
            'nB0': nB0,
            'nI0': nI0,
            'p':   P,
            'T':   T,
            **rate_constants,
            **arrhenius_meta,
        }
        return params

    # ------------------------------------------------------------------
    # Profile interpolation
    # ------------------------------------------------------------------

    def interpolate_profile(
        self,
        profile: Dict[str, np.ndarray],
        reactor_type: str,
    ) -> np.ndarray:
        """
        Interpolate concentration profile to a fixed grid length.

        Returns:
            Array of shape (grid_length, 5) with columns [A, B, C, D, n_total]
        """
        if reactor_type == 'PFR':
            x = profile['z'] if 'z' in profile else profile['V']
        else:
            # Both 'CSTR' (single) and 'CSTR_cascade' use volumetric axis
            x = profile['V']

        # Normalise independent variable to [0, 1]
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x_norm = (x - x_min) / (x_max - x_min)
        else:
            x_norm = np.zeros_like(x)

        grid = np.linspace(0, 1, self.grid_length)

        species = ['A', 'B', 'C', 'D', 'n_total']
        interpolated = np.zeros((self.grid_length, len(species)))

        for i, spec in enumerate(species):
            if spec in profile and len(profile[spec]) > 0:
                if len(x_norm) == 1:
                    interpolated[:, i] = profile[spec][0]
                else:
                    interpolated[:, i] = np.interp(grid, x_norm, profile[spec])
            else:
                interpolated[:, i] = 0.0

        return interpolated

    # ------------------------------------------------------------------
    # Simulation backends
    # ------------------------------------------------------------------

    def run_simulation_ode(
        self,
        params: Dict[str, float],
        mechanism_class: int,
        reactor_type: str,
    ) -> Dict[str, np.ndarray]:
        """Run simulation using the Python ODE solver.

        Reactor types:
            'PFR'          -- Plug flow reactor (ODE IVP)
            'CSTR'         -- Single CSTR (CSTR cascade with n_stages=1)
            'CSTR_cascade' -- CSTR cascade (n_stages as configured)
        """
        n0  = np.array([params['nA0'], params['nB0'], 0.0, 0.0])
        nI0 = params['nI0']
        P   = params['p']   # Use sampled pressure, not the fixed self.P

        mechanism = MECHANISM_REGISTRY[mechanism_class]
        rc = {name: params[name] for name in mechanism.rate_constant_names}

        if reactor_type == 'PFR':
            return solve_pfr(
                mechanism_class, n0, nI0, rc,
                P, self.V_total,
                n_points=max(self.grid_length, 200),
            )
        elif reactor_type == 'CSTR':
            # Single well-mixed CSTR (n_stages=1)
            return solve_cstr_cascade(
                mechanism_class, n0, nI0, rc,
                P, self.V_total,
                n_stages=1,
            )
        else:
            # CSTR cascade
            return solve_cstr_cascade(
                mechanism_class, n0, nI0, rc,
                P, self.V_total,
                n_stages=self.n_cstr_stages,
            )

    def run_simulation_excel(self, wb, params, reactor_type):
        """Run simulation using Excel workbook (classes 0-2 only)."""
        from excel_driver import set_inputs, recalc, read_pfr_profile, read_cstr_profile

        set_inputs(wb, params)
        recalc(wb, wait_time=0.2)

        if reactor_type == 'PFR':
            return read_pfr_profile(wb)
        else:
            return read_cstr_profile(wb)

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self, visible: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate complete dataset.

        Returns:
            (X, y, metadata)
            X: shape (n_runs, grid_length, 16) -- [A, B, C, D, n_total, reactor_type, T_norm, P_norm, dA/dV, dB/dV, dC/dV, dD/dV, xA, xB, xC, xD]
            y: shape (n_runs,) -- mechanism class labels
            metadata: list of parameter dicts
        """
        classes = self.mechanism_classes_to_generate
        print(f"Generating dataset: {self.n_runs} runs, classes {classes}")

        wb = None
        app = None

        # Only open Excel if using the Excel backend
        if not self.use_ode_solver:
            from excel_driver import open_workbook, close_workbook
            print(f"Opening workbook: {self.workbook_path}")
            app, wb = open_workbook(self.workbook_path, visible=visible)

        X_list = []
        y_list = []
        metadata_list = []

        # Reactor type encoding: PFR=0.0, CSTR single=0.5, CSTR cascade=1.0
        REACTOR_ENCODING = {'PFR': 0.0, 'CSTR': 0.5, 'CSTR_cascade': 1.0}
        REACTOR_TYPES = list(REACTOR_ENCODING.keys())

        # Normalisation bounds (must match predict.py)
        T_MIN, T_MAX = 350.0, 750.0
        P_MIN, P_MAX = 50_000.0, 200_000.0

        try:
            for i in tqdm(range(self.n_runs), desc="Generating data"):
                # Weighted sampling: oversample hard classes
                weights = np.ones(len(classes))
                for idx_c, c in enumerate(classes):
                    if c in self.oversample_classes:
                        weights[idx_c] = self.oversample_factor
                weights /= weights.sum()
                mechanism_class = int(np.random.choice(classes, p=weights))
                params = self.sample_parameters(mechanism_class)
                reactor_type = np.random.choice(REACTOR_TYPES)
                reactor_type_encoded = REACTOR_ENCODING[reactor_type]

                try:
                    if self.use_ode_solver:
                        profile = self.run_simulation_ode(
                            params, mechanism_class, reactor_type
                        )
                    else:
                        if mechanism_class > 2:
                            raise ValueError(
                                f"Mechanism class {mechanism_class} "
                                "requires the ODE solver (use --use_ode)"
                            )
                        # Excel backend only supports PFR/CSTR (not cascade)
                        excel_type = 'CSTR' if reactor_type != 'PFR' else 'PFR'
                        profile = self.run_simulation_excel(
                            wb, params, excel_type
                        )

                    profile_interp = self.interpolate_profile(
                        profile, reactor_type
                    )

                    # Skip runs with negligible conversion (<5%)
                    # Check both A (col 0) and B (col 1) — use max conversion
                    max_conversion = 0.0
                    for sp_col in (0, 1):  # A, B
                        sp_start = profile_interp[0, sp_col]
                        sp_end = profile_interp[-1, sp_col]
                        if sp_start > 1e-10:
                            conv = 1.0 - sp_end / sp_start
                            max_conversion = max(max_conversion, conv)
                    if max_conversion < 0.05:
                        continue  # retry with next random draw

                    # Build 16-channel feature vector:
                    # [A, B, C, D, n_total, reactor_type, T_norm, P_norm,
                    #  dA/dV, dB/dV, dC/dV, dD/dV, xA, xB, xC, xD]
                    T_val = params.get('T', 550.0)
                    P_val = params.get('p', self.P)
                    T_norm = (T_val - T_MIN) / (T_MAX - T_MIN)
                    P_norm = (P_val - P_MIN) / (P_MAX - P_MIN)

                    reactor_feature = np.full((self.grid_length, 1), reactor_type_encoded)
                    T_feature       = np.full((self.grid_length, 1), T_norm)
                    P_feature       = np.full((self.grid_length, 1), P_norm)

                    # Derivative features: dA/dV, dB/dV, dC/dV, dD/dV
                    derivatives = np.gradient(profile_interp[:, :4], axis=0)

                    # Mole fraction features: xA, xB, xC, xD
                    n_tot = profile_interp[:, 4:5]
                    n_tot_safe = np.maximum(n_tot, 1e-10)
                    mole_fractions = profile_interp[:, :4] / n_tot_safe

                    features = np.concatenate(
                        [profile_interp, reactor_feature, T_feature, P_feature,
                         derivatives, mole_fractions], axis=1
                    )  # shape: (grid_length, 16)

                    X_list.append(features)
                    y_list.append(mechanism_class)

                    meta = {
                        'run_id': i,
                        'mechanism_class': mechanism_class,
                        'mechanism_equation': self.mechanism_classes[mechanism_class],
                        'reactor_type': reactor_type,
                        **params,
                    }
                    metadata_list.append(meta)

                except Exception as e:
                    print(f"\nWarning: Run {i} failed: {e}")
                    print(f"  Params: {params}, Reactor: {reactor_type}")

        finally:
            if wb is not None:
                from excel_driver import close_workbook
                print("\nClosing workbook...")
                close_workbook(app, wb, save=False)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        print(f"\nDataset generated:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Class distribution: {np.bincount(y, minlength=max(classes)+1)}")

        return X, y, metadata_list

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_dataset(
        self, X: np.ndarray, y: np.ndarray, metadata: List[Dict],
    ) -> None:
        """Save dataset to disk."""
        dataset_path = self.save_dir / 'dataset.npz'
        np.savez_compressed(dataset_path, X=X, y=y)
        print(f"\nDataset saved to: {dataset_path}")

        metadata_path = self.save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

        info = {
            'n_runs': len(y),
            'grid_length': self.grid_length,
            'num_features': X.shape[2],
            'num_classes': len(np.unique(y)),
            'class_distribution': {
                int(k): int(v)
                for k, v in zip(*np.unique(y, return_counts=True))
            },
            'mechanism_classes': self.mechanism_classes,
            'feature_names': [
                'A', 'B', 'C', 'D', 'n_total',
                'reactor_type', 'T_norm', 'P_norm',
                'dA/dV', 'dB/dV', 'dC/dV', 'dD/dV',
                'xA', 'xB', 'xC', 'xD',
            ],
        }
        info_path = self.save_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Dataset info saved to: {info_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset from reactor simulations'
    )
    parser.add_argument(
        '--workbook', type=str,
        default='data/PFR-CSTR-cascade_Calculation-macro.xlsm',
        help='Path to Excel workbook (only used with --use_excel)',
    )
    parser.add_argument('--n_runs', type=int, default=1000)
    parser.add_argument('--grid_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument(
        '--visible', action='store_true',
        help='Show Excel window (debugging)',
    )
    parser.add_argument(
        '--use_excel', action='store_true',
        help='Use Excel instead of ODE solver (classes 0-2 only)',
    )
    parser.add_argument(
        '--mechanisms', type=int, nargs='+',
        default=list(range(len(MECHANISM_REGISTRY))),
        help='Mechanism classes to include (default: all 20)',
    )
    parser.add_argument('--V_total', type=float, default=1.0,
                        help='Total reactor volume in m^3')
    parser.add_argument('--n_cstr', type=int, default=50,
                        help='Number of CSTR cascade stages')

    args = parser.parse_args()

    builder = DatasetBuilder(
        workbook_path=args.workbook,
        n_runs=args.n_runs,
        grid_length=args.grid_length,
        random_seed=args.seed,
        save_dir=args.save_dir,
        use_ode_solver=not args.use_excel,
        mechanism_classes_to_generate=args.mechanisms,
        V_total=args.V_total,
        n_cstr_stages=args.n_cstr,
    )

    X, y, metadata = builder.generate_dataset(visible=args.visible)
    builder.save_dataset(X, y, metadata)
    print("\nDataset generation complete!")


if __name__ == '__main__':
    main()
