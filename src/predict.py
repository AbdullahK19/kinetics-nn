"""
Prediction script for mechanism classification.

This module:
- Loads trained model
- Accepts reactor parameters and runs ODE simulation
- Returns ranked mechanism predictions with probabilities
- Supports configurable confidence threshold (default 0.5%)
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List

from model import create_model
from preprocess import StandardScaler
from ode_solver import MECHANISM_REGISTRY
from dataset_builder import DatasetBuilder

# Normalisation bounds — must match dataset_builder.py
_T_MIN, _T_MAX = 350.0, 750.0
_P_MIN, _P_MAX = 50_000.0, 200_000.0
# Reactor type encoding — must match dataset_builder.py
_REACTOR_ENCODING = {'PFR': 0.0, 'CSTR': 0.5, 'CSTR_cascade': 1.0}


class MechanismPredictor:
    """Predictor for mechanism classification."""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        config_path: str,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load scaler
        self.scaler = StandardScaler()
        self.scaler.load(scaler_path)

        # Determine num_classes and num_features from config or defaults
        num_classes  = self.config.get('num_classes', len(MECHANISM_REGISTRY))
        num_features = self.config.get('num_features', 8)

        # Optional architecture overrides stored in training_config.json
        extra_kwargs = {}
        if 'hidden_channels' in self.config:
            extra_kwargs['hidden_channels'] = tuple(self.config['hidden_channels'])
        if 'fc_hidden' in self.config:
            extra_kwargs['fc_hidden'] = self.config['fc_hidden']
        if 'blocks_per_stage' in self.config:
            extra_kwargs['blocks_per_stage'] = self.config['blocks_per_stage']

        # Create model (architecture must match the saved checkpoint exactly)
        self.model = create_model(
            model_type=self.config['model_type'],
            num_features=num_features,
            num_classes=num_classes,
            seq_length=self.config.get('grid_length', 128),
            dropout=0.0,  # No dropout during inference
            **extra_kwargs
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Mechanism class labels from registry
        self.mechanism_classes = {
            cid: {"equation": mech.name, "type": mech.type_label}
            for cid, mech in MECHANISM_REGISTRY.items()
            if cid < num_classes
        }

        print("Predictor initialized successfully!")
        print(f"Model type: {self.config['model_type']}")
        print(f"Classes: {num_classes}, Features: {num_features}")
        print(f"Device: {self.device}")

    def preprocess_profile(
        self,
        profile: Dict[str, np.ndarray],
        reactor_type: str,
        T: float = 550.0,
        P: float = 100_000.0,
        grid_length: int = 128
    ) -> np.ndarray:
        """
        Preprocess concentration profile for model input.

        Builds a 16-channel feature vector per time-step:
        [A, B, C, D, n_total, reactor_type, T_norm, P_norm,
         dA/dV, dB/dV, dC/dV, dD/dV, xA, xB, xC, xD]

        Args:
            profile:      Dict with keys 'A','B','C','D','n_total','V'/'z'
            reactor_type: One of 'PFR', 'CSTR', 'CSTR_cascade'
            T:            Temperature in K (used for T_norm feature)
            P:            Pressure in Pa (used for P_norm feature)
            grid_length:  Number of grid points to interpolate to
        """
        builder = DatasetBuilder(
            n_runs=0,
            grid_length=grid_length,
            use_ode_solver=True,
        )

        profile_interp = builder.interpolate_profile(profile, reactor_type)

        reactor_type_encoded = _REACTOR_ENCODING.get(reactor_type, 0.0)

        T_norm = (T - _T_MIN) / (_T_MAX - _T_MIN)
        P_norm = (P - _P_MIN) / (_P_MAX - _P_MIN)

        reactor_feature = np.full((grid_length, 1), reactor_type_encoded)
        T_feature       = np.full((grid_length, 1), T_norm)
        P_feature       = np.full((grid_length, 1), P_norm)

        # Derivative features: dA/dV, dB/dV, dC/dV, dD/dV
        derivatives = np.gradient(profile_interp[:, :4], axis=0)

        # Mole fraction features: xA, xB, xC, xD
        n_tot = profile_interp[:, 4:5]
        n_tot_safe = np.maximum(n_tot, 1e-10)
        mole_fractions = profile_interp[:, :4] / n_tot_safe

        features = np.concatenate(
            [profile_interp, reactor_feature, T_feature, P_feature,
             derivatives, mole_fractions], axis=1
        )  # (grid_length, 16)

        features = features[np.newaxis, ...]  # (1, grid_length, 16)
        features = self.scaler.transform(features)

        return features

    def predict_from_profile(
        self,
        profile: Dict[str, np.ndarray],
        reactor_type: str,
        T: float = 550.0,
        P: float = 100_000.0,
        threshold: float = 0.005
    ) -> List[Dict]:
        """
        Predict mechanism from concentration profile.

        Returns ranked list of up to 10 mechanisms whose predicted
        probability is >= threshold (0.5% cutoff).

        Args:
            profile:      Concentration profile dict from ODE solver
            reactor_type: 'PFR', 'CSTR', or 'CSTR_cascade'
            T:            Temperature in K
            P:            Pressure in Pa
            threshold:    Minimum probability to include (default 0.005 = 0.5%)
        """
        X = self.preprocess_profile(
            profile, reactor_type, T, P, self.config.get('grid_length', 128)
        )

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        predictions = []
        for class_idx in np.argsort(probs)[::-1]:
            prob = float(probs[class_idx])
            if prob >= threshold:
                info = self.mechanism_classes.get(
                    int(class_idx),
                    {"equation": f"Unknown (class {class_idx})", "type": "unknown"}
                )
                pred = {
                    'equation': info['equation'],
                    'type': info['type'],
                    'probability': prob,
                    'class_id': int(class_idx)
                }
                predictions.append(pred)

        # Cap at top 10 as per dissertation specification
        return predictions[:10]

    def predict_from_ode(
        self,
        reactor_type: str,
        mechanism_class: int,
        rate_constants: Dict[str, float],
        A0: float,
        B0: float,
        T: float = 550.0,
        P: float = 100_000.0,
        I0: float = 0.0,
        V_total: float = 1.0,
        n_cstr_stages: int = 50,
        threshold: float = 0.005,
    ) -> List[Dict]:
        """
        Predict mechanism from a fresh ODE simulation.

        Args:
            reactor_type:    'PFR', 'CSTR', or 'CSTR_cascade'
            mechanism_class: Mechanism class ID (0-27)
            rate_constants:  Dict of rate constants for the mechanism
            A0, B0:          Initial molar flow rates (mol/s)
            T:               Temperature in K (default 550 K)
            P:               Pressure in Pa (default 100,000 Pa)
            I0:              Inert molar flow rate (mol/s)
            V_total:         Total reactor volume (m^3)
            n_cstr_stages:   Number of CSTR stages (CSTR_cascade only)
            threshold:       Probability cutoff (default 0.5%)
        """
        from ode_solver import solve_pfr, solve_cstr_cascade

        n0  = np.array([A0, B0, 0.0, 0.0])
        nI0 = I0
        mech_class = mechanism_class

        mech = MECHANISM_REGISTRY[mech_class]
        print(f"Mechanism class {mech_class}: {mech.name}")
        print(f"Rate constants: {rate_constants}")
        print(f"Temperature: {T:.1f} K, Pressure: {P/1000:.1f} kPa")

        if reactor_type == 'PFR':
            profile = solve_pfr(
                mech_class, n0, nI0, rate_constants, P, V_total, n_points=200
            )
        elif reactor_type == 'CSTR':
            profile = solve_cstr_cascade(
                mech_class, n0, nI0, rate_constants, P, V_total, n_stages=1
            )
        else:
            profile = solve_cstr_cascade(
                mech_class, n0, nI0, rate_constants, P, V_total, n_stages=n_cstr_stages
            )

        return self.predict_from_profile(profile, reactor_type, T, P, threshold)


def format_predictions(predictions: List[Dict]) -> str:
    """Format predictions as human-readable string."""
    if not predictions:
        return "No mechanisms above the 0.5% likelihood threshold."

    output = []
    output.append("\nMechanism Predictions (ranked by probability, 0.5% cutoff, top-10 max):")
    output.append("=" * 70)

    for i, pred in enumerate(predictions, 1):
        output.append(f"\n{i:2d}. Equation:   {pred['equation']}")
        output.append(f"    Type:        {pred['type']}")
        output.append(f"    Likelihood:  {pred['probability']*100:6.2f}%")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Predict reaction mechanism from concentration profile'
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--ode', action='store_true',
                             help='Predict from ODE simulation')
    input_group.add_argument('--profile', type=str,
                             help='Path to saved profile (.npz)')

    # Model paths
    parser.add_argument('--model_path', type=str, default='./models/best_model.pt')
    parser.add_argument('--scaler_path', type=str, default='./data/scaler.json')
    parser.add_argument('--config_path', type=str, default='./models/training_config.json')

    # Reactor / simulation parameters
    parser.add_argument('--reactor', type=str,
                        choices=['PFR', 'CSTR', 'CSTR_cascade'],
                        help='Reactor type: PFR, CSTR (single), or CSTR_cascade')
    parser.add_argument('--mechanism', type=int,
                        help='Mechanism class ID (0-27) for ODE simulation')
    parser.add_argument('--A0', type=float, help='Initial molar flow of A (mol/s)')
    parser.add_argument('--B0', type=float, help='Initial molar flow of B (mol/s)')
    parser.add_argument('--I0', type=float, default=0.0,
                        help='Inert molar flow rate (mol/s, default 0)')
    parser.add_argument('--T', type=float, default=550.0,
                        help='Temperature in K (default 550 K)')
    parser.add_argument('--P', type=float, default=100_000.0,
                        help='Pressure in Pa (default 100,000 Pa = 1 bar)')
    parser.add_argument('--rc', type=str, nargs='+', default=[],
                        help='Rate constants as name=value pairs (e.g. k1=1e-10 k2=5e-5)')
    parser.add_argument('--V_total', type=float, default=1.0,
                        help='Total reactor volume in m^3 (default 1.0)')
    parser.add_argument('--n_cstr', type=int, default=50,
                        help='Number of CSTR cascade stages (default 50)')

    # Prediction options
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Probability cutoff (default 0.005 = 0.5%%)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--json_output', action='store_true',
                        help='Print raw JSON instead of formatted output')

    args = parser.parse_args()

    # Validate ODE mode parameters
    if args.ode:
        if args.reactor is None:
            parser.error("--ode requires --reactor (PFR, CSTR, or CSTR_cascade)")
        if args.mechanism is None:
            parser.error("--ode requires --mechanism (class ID 0-27)")
        if args.A0 is None or args.B0 is None:
            parser.error("--ode requires --A0 and --B0")
        if not args.rc:
            parser.error("--ode requires --rc (e.g. --rc k1=1e-10)")

    # Create predictor
    predictor = MechanismPredictor(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        config_path=args.config_path,
        device=args.device
    )

    # Make prediction
    if args.ode:
        # Parse rate constants from --rc name=value pairs
        rate_constants = {}
        for rc_str in args.rc:
            name, val = rc_str.split('=')
            rate_constants[name.strip()] = float(val.strip())

        mech = MECHANISM_REGISTRY[args.mechanism]
        print(f"\nRunning ODE simulation with parameters:")
        print(f"  Mechanism:   Class {args.mechanism}: {mech.name}")
        print(f"  Reactor:     {args.reactor}")
        print(f"  A0={args.A0}, B0={args.B0}, I0={args.I0}")
        print(f"  Temperature: {args.T} K")
        print(f"  Pressure:    {args.P/1000:.1f} kPa")
        print(f"  Rate constants: {rate_constants}")
        print(f"  V_total={args.V_total} m^3")

        predictions = predictor.predict_from_ode(
            reactor_type=args.reactor,
            mechanism_class=args.mechanism,
            rate_constants=rate_constants,
            A0=args.A0,
            B0=args.B0,
            T=args.T,
            P=args.P,
            I0=args.I0,
            V_total=args.V_total,
            n_cstr_stages=args.n_cstr,
            threshold=args.threshold,
        )

    else:
        # Load from saved profile (.npz)
        if args.reactor is None:
            parser.error("--profile requires --reactor (PFR, CSTR, or CSTR_cascade)")

        profile_data = np.load(args.profile)
        profile = {key: profile_data[key]
                   for key in ['A', 'B', 'C', 'D', 'n_total', 'V', 'z']
                   if key in profile_data}

        predictions = predictor.predict_from_profile(
            profile, args.reactor, args.T, args.P, args.threshold
        )

    # Output predictions
    if args.json_output:
        print(json.dumps(predictions, indent=2))
    else:
        print(format_predictions(predictions))


if __name__ == '__main__':
    main()
