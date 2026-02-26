"""
ODE-based reactor solvers for PFR and CSTR cascade reactors.

Provides Python-native simulation of chemical reaction mechanisms,
replacing the Excel dependency for dataset generation and enabling
complex/intermediate reaction mechanisms.

28 Mechanism Classes (Groups A-I):

Group A — Unimolecular:
    Class  0: A -> C
    Class  1: A -> D
    Class  2: B -> C
    Class  3: B -> D

Group B — Bimolecular Heterogeneous:
    Class  4: A + B -> C
    Class  5: A + B -> D

Group C — Bimolecular Homogeneous:
    Class  6: 2A -> C
    Class  7: 2A -> D
    Class  8: 2B -> C
    Class  9: 2B -> D

Group D — Sequential with Intermediate C:
    Class 10: A + B -> C -> D
    Class 11: 2A -> C -> D
    Class 12: A -> C -> D
    Class 13: B -> C -> D

Group E — Parallel:
    Class 14: A+B -> C ; 2A -> D
    Class 15: A -> C ; B -> D
    Class 16: 2A -> C ; 2B -> D

Group F — Parallel-Sequential:
    Class 17: A+B -> C -> D ; 2A -> D
    Class 18: 2A -> C -> D ; A+B -> D
    Class 19: A -> C -> D ; B -> D

Group G — Reversible Simple (1st order both ways):
    Class 20: A <-> C
    Class 21: A <-> D
    Class 22: B <-> C
    Class 23: B <-> D

Group H — Reversible Bimolecular (2nd order forward, 1st order reverse):
    Class 24: A + B <-> C
    Class 25: A + B <-> D

Group I — Split Product (one reactant, two simultaneous products):
    Class 26: A -> C + D
    Class 27: B -> C + D
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R_GAS = 8.314  # Universal gas constant, J/(mol·K)


def arrhenius_k(A_pre: float, Ea: float, T: float) -> float:
    """
    Compute Arrhenius rate constant: k(T) = A_pre * exp(-Ea / (R * T))

    Args:
        A_pre: Pre-exponential factor (same units as k)
        Ea:    Activation energy (J/mol)
        T:     Temperature (K)

    Returns:
        Rate constant at temperature T
    """
    return A_pre * np.exp(-Ea / (R_GAS * T))


# ---------------------------------------------------------------------------
# Mechanism registry -- single source of truth for all mechanism definitions
# ---------------------------------------------------------------------------

# Species index mapping: A=0, B=1, C=2, D=3

# Rate constant ranges by reaction order:
#   1st order (unimolecular): k in mol·s⁻¹·m⁻³·Pa⁻¹, range [1e-6, 1e-4]
#   2nd order (bimolecular):  k in mol·s⁻¹·m⁻³·Pa⁻², range [1e-11, 5e-10]
_RANGE_1ST = (1e-6, 1e-4)
_RANGE_2ND = (1e-11, 5e-10)


class Mechanism:
    """Defines a reaction mechanism with stoichiometry and rate expressions.

    Attributes:
        class_id: Integer class identifier.
        name: Human-readable equation string.
        type_label: Category label (simple, parallel, sequential, parallel_sequential).
        stoich_matrix: Shape (4, n_reactions) stoichiometry matrix [A, B, C, D].
        rate_expressions: List of (rc_name, [(species_idx, power), ...]) tuples.
            Each entry defines one elementary rate law:
                rate = k * product(p_species ** power)
            Example: k1·pA·pB → ('k1', [(0, 1), (1, 1)])
            Example: k2·pA²   → ('k2', [(0, 2)])
        rate_constant_names: Ordered list of rate constant names.
        rate_constant_ranges: Dict mapping rc_name → (lo, hi) sampling range.
    """

    def __init__(
        self,
        class_id: int,
        name: str,
        type_label: str,
        stoich_matrix: np.ndarray,
        rate_expressions: List[Tuple[str, List[Tuple[int, int]]]],
        rate_constant_ranges: Dict[str, Tuple[float, float]],
    ):
        self.class_id = class_id
        self.name = name
        self.type_label = type_label
        self.stoich_matrix = stoich_matrix          # shape (4, n_reactions)
        self.rate_expressions = rate_expressions
        self.rate_constant_names = [expr[0] for expr in rate_expressions]
        self.rate_constant_ranges = rate_constant_ranges


# --- Group A: Unimolecular (1st order) ---

_mech_0 = Mechanism(
    class_id=0, name="A -> C", type_label="simple",
    stoich_matrix=np.array([[-1], [0], [1], [0]]),
    rate_expressions=[('k1', [(0, 1)])],           # k1·pA
    rate_constant_ranges={'k1': _RANGE_1ST},
)

_mech_1 = Mechanism(
    class_id=1, name="A -> D", type_label="simple",
    stoich_matrix=np.array([[-1], [0], [0], [1]]),
    rate_expressions=[('k1', [(0, 1)])],           # k1·pA
    rate_constant_ranges={'k1': _RANGE_1ST},
)

_mech_2 = Mechanism(
    class_id=2, name="B -> C", type_label="simple",
    stoich_matrix=np.array([[0], [-1], [1], [0]]),
    rate_expressions=[('k1', [(1, 1)])],           # k1·pB
    rate_constant_ranges={'k1': _RANGE_1ST},
)

_mech_3 = Mechanism(
    class_id=3, name="B -> D", type_label="simple",
    stoich_matrix=np.array([[0], [-1], [0], [1]]),
    rate_expressions=[('k1', [(1, 1)])],           # k1·pB
    rate_constant_ranges={'k1': _RANGE_1ST},
)

# --- Group B: Bimolecular Heterogeneous (2nd order, A+B) ---

_mech_4 = Mechanism(
    class_id=4, name="A + B -> C", type_label="simple",
    stoich_matrix=np.array([[-1], [-1], [1], [0]]),
    rate_expressions=[('k1', [(0, 1), (1, 1)])],   # k1·pA·pB
    rate_constant_ranges={'k1': _RANGE_2ND},
)

_mech_5 = Mechanism(
    class_id=5, name="A + B -> D", type_label="simple",
    stoich_matrix=np.array([[-1], [-1], [0], [1]]),
    rate_expressions=[('k1', [(0, 1), (1, 1)])],   # k1·pA·pB
    rate_constant_ranges={'k1': _RANGE_2ND},
)

# --- Group C: Bimolecular Homogeneous (2nd order, 2X -> Y) ---

_mech_6 = Mechanism(
    class_id=6, name="2A -> C", type_label="simple",
    stoich_matrix=np.array([[-2], [0], [1], [0]]),
    rate_expressions=[('k1', [(0, 2)])],           # k1·pA²
    rate_constant_ranges={'k1': _RANGE_2ND},
)

_mech_7 = Mechanism(
    class_id=7, name="2A -> D", type_label="simple",
    stoich_matrix=np.array([[-2], [0], [0], [1]]),
    rate_expressions=[('k1', [(0, 2)])],           # k1·pA²
    rate_constant_ranges={'k1': _RANGE_2ND},
)

_mech_8 = Mechanism(
    class_id=8, name="2B -> C", type_label="simple",
    stoich_matrix=np.array([[0], [-2], [1], [0]]),
    rate_expressions=[('k1', [(1, 2)])],           # k1·pB²
    rate_constant_ranges={'k1': _RANGE_2ND},
)

_mech_9 = Mechanism(
    class_id=9, name="2B -> D", type_label="simple",
    stoich_matrix=np.array([[0], [-2], [0], [1]]),
    rate_expressions=[('k1', [(1, 2)])],           # k1·pB²
    rate_constant_ranges={'k1': _RANGE_2ND},
)

# --- Group D: Sequential with Intermediate C ---

_mech_10 = Mechanism(
    class_id=10, name="A + B -> C -> D", type_label="sequential",
    stoich_matrix=np.array([
        [-1,  0],   # A: -1 from r1
        [-1,  0],   # B: -1 from r1
        [ 1, -1],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 1), (1, 1)]),   # k1·pA·pB
        ('k2', [(2, 1)]),           # k2·pC
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_1ST},
)

_mech_11 = Mechanism(
    class_id=11, name="2A -> C -> D", type_label="sequential",
    stoich_matrix=np.array([
        [-2,  0],   # A: -2 from r1
        [ 0,  0],   # B: unaffected
        [ 1, -1],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 2)]),           # k1·pA²
        ('k2', [(2, 1)]),           # k2·pC
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_1ST},
)

_mech_12 = Mechanism(
    class_id=12, name="A -> C -> D", type_label="sequential",
    stoich_matrix=np.array([
        [-1,  0],   # A: -1 from r1
        [ 0,  0],   # B: unaffected
        [ 1, -1],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 1)]),           # k1·pA
        ('k2', [(2, 1)]),           # k2·pC
    ],
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

_mech_13 = Mechanism(
    class_id=13, name="B -> C -> D", type_label="sequential",
    stoich_matrix=np.array([
        [ 0,  0],   # A: unaffected
        [-1,  0],   # B: -1 from r1
        [ 1, -1],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(1, 1)]),           # k1·pB
        ('k2', [(2, 1)]),           # k2·pC
    ],
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

# --- Group E: Parallel ---

_mech_14 = Mechanism(
    class_id=14, name="A + B -> C ; 2A -> D", type_label="parallel",
    stoich_matrix=np.array([
        [-1, -2],   # A: -1 from r1, -2 from r2
        [-1,  0],   # B: -1 from r1
        [ 1,  0],   # C: +1 from r1
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 1), (1, 1)]),   # k1·pA·pB
        ('k2', [(0, 2)]),           # k2·pA²
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_2ND},
)

_mech_15 = Mechanism(
    class_id=15, name="A -> C ; B -> D", type_label="parallel",
    stoich_matrix=np.array([
        [-1,  0],   # A: -1 from r1
        [ 0, -1],   # B: -1 from r2
        [ 1,  0],   # C: +1 from r1
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 1)]),           # k1·pA
        ('k2', [(1, 1)]),           # k2·pB
    ],
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

_mech_16 = Mechanism(
    class_id=16, name="2A -> C ; 2B -> D", type_label="parallel",
    stoich_matrix=np.array([
        [-2,  0],   # A: -2 from r1
        [ 0, -2],   # B: -2 from r2
        [ 1,  0],   # C: +1 from r1
        [ 0,  1],   # D: +1 from r2
    ]),
    rate_expressions=[
        ('k1', [(0, 2)]),           # k1·pA²
        ('k2', [(1, 2)]),           # k2·pB²
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_2ND},
)

# --- Group F: Parallel-Sequential ---

_mech_17 = Mechanism(
    class_id=17, name="A + B -> C -> D ; 2A -> D", type_label="parallel_sequential",
    stoich_matrix=np.array([
        [-1, -2,  0],   # A: -1 from r1, -2 from r2
        [-1,  0,  0],   # B: -1 from r1
        [ 1,  0, -1],   # C: +1 from r1, -1 from r3 (intermediate)
        [ 0,  1,  1],   # D: +1 from r2, +1 from r3
    ]),
    rate_expressions=[
        ('k1', [(0, 1), (1, 1)]),   # k1·pA·pB
        ('k2', [(0, 2)]),           # k2·pA²
        ('k3', [(2, 1)]),           # k3·pC
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_2ND, 'k3': _RANGE_1ST},
)

_mech_18 = Mechanism(
    class_id=18, name="2A -> C -> D ; A + B -> D", type_label="parallel_sequential",
    stoich_matrix=np.array([
        [-2,  0, -1],   # A: -2 from r1, -1 from r3
        [ 0,  0, -1],   # B: -1 from r3
        [ 1, -1,  0],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1,  1],   # D: +1 from r2, +1 from r3
    ]),
    rate_expressions=[
        ('k1', [(0, 2)]),           # k1·pA²
        ('k2', [(2, 1)]),           # k2·pC
        ('k3', [(0, 1), (1, 1)]),   # k3·pA·pB
    ],
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_1ST, 'k3': _RANGE_2ND},
)

_mech_19 = Mechanism(
    class_id=19, name="A -> C -> D ; B -> D", type_label="parallel_sequential",
    stoich_matrix=np.array([
        [-1,  0,  0],   # A: -1 from r1
        [ 0,  0, -1],   # B: -1 from r3
        [ 1, -1,  0],   # C: +1 from r1, -1 from r2 (intermediate)
        [ 0,  1,  1],   # D: +1 from r2, +1 from r3
    ]),
    rate_expressions=[
        ('k1', [(0, 1)]),           # k1·pA
        ('k2', [(2, 1)]),           # k2·pC
        ('k3', [(1, 1)]),           # k3·pB
    ],
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST, 'k3': _RANGE_1ST},
)


# --- Group G: Reversible Simple (1st order both ways) ---

_mech_20 = Mechanism(
    class_id=20, name="A <-> C", type_label="reversible",
    stoich_matrix=np.array([[-1, 1], [0, 0], [1, -1], [0, 0]]),
    rate_expressions=[('k1', [(0, 1)]), ('k2', [(2, 1)])],   # k1·pA / k2·pC
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

_mech_21 = Mechanism(
    class_id=21, name="A <-> D", type_label="reversible",
    stoich_matrix=np.array([[-1, 1], [0, 0], [0, 0], [1, -1]]),
    rate_expressions=[('k1', [(0, 1)]), ('k2', [(3, 1)])],   # k1·pA / k2·pD
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

_mech_22 = Mechanism(
    class_id=22, name="B <-> C", type_label="reversible",
    stoich_matrix=np.array([[0, 0], [-1, 1], [1, -1], [0, 0]]),
    rate_expressions=[('k1', [(1, 1)]), ('k2', [(2, 1)])],   # k1·pB / k2·pC
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

_mech_23 = Mechanism(
    class_id=23, name="B <-> D", type_label="reversible",
    stoich_matrix=np.array([[0, 0], [-1, 1], [0, 0], [1, -1]]),
    rate_expressions=[('k1', [(1, 1)]), ('k2', [(3, 1)])],   # k1·pB / k2·pD
    rate_constant_ranges={'k1': _RANGE_1ST, 'k2': _RANGE_1ST},
)

# --- Group H: Reversible Bimolecular (2nd order forward, 1st order reverse) ---

_mech_24 = Mechanism(
    class_id=24, name="A + B <-> C", type_label="reversible",
    stoich_matrix=np.array([[-1, 1], [-1, 1], [1, -1], [0, 0]]),
    rate_expressions=[('k1', [(0, 1), (1, 1)]), ('k2', [(2, 1)])],  # k1·pA·pB / k2·pC
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_1ST},
)

_mech_25 = Mechanism(
    class_id=25, name="A + B <-> D", type_label="reversible",
    stoich_matrix=np.array([[-1, 1], [-1, 1], [0, 0], [1, -1]]),
    rate_expressions=[('k1', [(0, 1), (1, 1)]), ('k2', [(3, 1)])],  # k1·pA·pB / k2·pD
    rate_constant_ranges={'k1': _RANGE_2ND, 'k2': _RANGE_1ST},
)

# --- Group I: Split Product (one reactant, two simultaneous products) ---

_mech_26 = Mechanism(
    class_id=26, name="A -> C + D", type_label="simple",
    stoich_matrix=np.array([[-1], [0], [1], [1]]),
    rate_expressions=[('k1', [(0, 1)])],                     # k1·pA
    rate_constant_ranges={'k1': _RANGE_1ST},
)

_mech_27 = Mechanism(
    class_id=27, name="B -> C + D", type_label="simple",
    stoich_matrix=np.array([[0], [-1], [1], [1]]),
    rate_expressions=[('k1', [(1, 1)])],                     # k1·pB
    rate_constant_ranges={'k1': _RANGE_1ST},
)


MECHANISM_REGISTRY: Dict[int, Mechanism] = {
    m.class_id: m for m in [
        _mech_0, _mech_1, _mech_2, _mech_3,
        _mech_4, _mech_5,
        _mech_6, _mech_7, _mech_8, _mech_9,
        _mech_10, _mech_11, _mech_12, _mech_13,
        _mech_14, _mech_15, _mech_16,
        _mech_17, _mech_18, _mech_19,
        _mech_20, _mech_21, _mech_22, _mech_23,
        _mech_24, _mech_25,
        _mech_26, _mech_27,
    ]
}


# ---------------------------------------------------------------------------
# Rate computation (generic — reads from mechanism.rate_expressions)
# ---------------------------------------------------------------------------

def compute_rates(
    class_id: int,
    n: np.ndarray,
    n_total: float,
    P: float,
    rate_constants: Dict[str, float],
) -> np.ndarray:
    """
    Compute the reaction rate vector for a given mechanism.

    Rates use partial pressures: p_i = (n_i / n_total) * P
    matching the Excel workbook convention.

    Args:
        class_id: Mechanism class ID (0-27)
        n: Molar flow rates [nA, nB, nC, nD]
        n_total: Total molar flow rate (including inert)
        P: Total pressure (Pa)
        rate_constants: Dict of rate constants (e.g. {'k1': 1e-10})

    Returns:
        Array of reaction rates (length = number of elementary reactions)
    """
    if n_total <= 0:
        n_total = 1e-30  # avoid division by zero

    mechanism = MECHANISM_REGISTRY[class_id]

    # Compute partial pressures for all 4 species
    pp = np.maximum(n, 0.0) / n_total * P  # [pA, pB, pC, pD]

    rates = np.empty(len(mechanism.rate_expressions))
    for i, (rc_name, species_powers) in enumerate(mechanism.rate_expressions):
        r = rate_constants[rc_name]
        for species_idx, power in species_powers:
            r *= pp[species_idx] ** power
        rates[i] = r

    return rates


# ---------------------------------------------------------------------------
# PFR solver
# ---------------------------------------------------------------------------

def solve_pfr(
    mechanism_class_id: int,
    n0: np.ndarray,
    nI0: float,
    rate_constants: Dict[str, float],
    P: float,
    V_total: float,
    n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Solve PFR as an ODE initial-value problem: dn_i/dV = sum(nu_ij * r_j).

    Args:
        mechanism_class_id: Mechanism class (0-27)
        n0: Initial molar flow rates [nA0, nB0, nC0, nD0]
        nI0: Inert molar flow rate (constant)
        rate_constants: Dict of rate constants
        P: Total pressure (Pa)
        V_total: Total reactor volume (m^3)
        n_points: Number of output points along the reactor

    Returns:
        Dict with keys 'V', 'A', 'B', 'C', 'D', 'n_total'
    """
    mechanism = MECHANISM_REGISTRY[mechanism_class_id]
    stoich = mechanism.stoich_matrix  # (4, n_reactions)

    def rhs(V, n):
        n_safe = np.maximum(n, 0.0)
        n_total = np.sum(n_safe) + nI0
        rates = compute_rates(mechanism_class_id, n_safe, n_total, P, rate_constants)
        return stoich @ rates

    V_eval = np.linspace(0, V_total, n_points)

    sol = solve_ivp(
        rhs,
        t_span=(0, V_total),
        y0=n0,
        t_eval=V_eval,
        method='Radau',      # implicit solver handles stiff sequential kinetics
        rtol=1e-8,
        atol=1e-10,
        max_step=V_total / 50,
    )

    if not sol.success:
        raise RuntimeError(f"PFR solver failed: {sol.message}")

    n_total = np.sum(np.maximum(sol.y, 0.0), axis=0) + nI0

    return {
        'V': sol.t,
        'z': sol.t,          # alias so interpolation code can use either key
        'A': sol.y[0],
        'B': sol.y[1],
        'C': sol.y[2],
        'D': sol.y[3],
        'n_total': n_total,
    }


# ---------------------------------------------------------------------------
# CSTR cascade solver
# ---------------------------------------------------------------------------

def solve_cstr_cascade(
    mechanism_class_id: int,
    n0: np.ndarray,
    nI0: float,
    rate_constants: Dict[str, float],
    P: float,
    V_total: float,
    n_stages: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Solve a CSTR cascade (series of identical CSTRs).

    For each stage the steady-state mass balance is:
        n_i,out = n_i,in + V_stage * sum(nu_ij * r_j(n_out))

    This is a system of nonlinear equations solved with fsolve per stage.

    Args:
        mechanism_class_id: Mechanism class (0-27)
        n0: Initial molar flow rates [nA0, nB0, nC0, nD0]
        nI0: Inert molar flow rate (constant)
        rate_constants: Dict of rate constants
        P: Total pressure (Pa)
        V_total: Total reactor volume (m^3)
        n_stages: Number of CSTR stages

    Returns:
        Dict with keys 'V', 'stage', 'A', 'B', 'C', 'D', 'n_total'
    """
    mechanism = MECHANISM_REGISTRY[mechanism_class_id]
    stoich = mechanism.stoich_matrix
    V_stage = V_total / n_stages

    # Storage: n_stages + 1 points (inlet + each stage outlet)
    n_arr = np.zeros((4, n_stages + 1))
    n_arr[:, 0] = n0
    V_arr = np.linspace(0, V_total, n_stages + 1)

    n_in = n0.copy()

    for stage in range(n_stages):
        def residual(n_out):
            n_out_safe = np.maximum(n_out, 0.0)
            n_total = np.sum(n_out_safe) + nI0
            rates = compute_rates(
                mechanism_class_id, n_out_safe, n_total, P, rate_constants
            )
            return n_out - n_in - V_stage * (stoich @ rates)

        # Initial guess: inlet values
        n_out, info, ier, msg = fsolve(residual, n_in, full_output=True)

        if ier != 1:
            # Fallback: forward Euler step from inlet
            n_total_in = np.sum(np.maximum(n_in, 0.0)) + nI0
            rates_in = compute_rates(
                mechanism_class_id, np.maximum(n_in, 0.0),
                n_total_in, P, rate_constants
            )
            n_out = n_in + V_stage * (stoich @ rates_in)

        n_out = np.maximum(n_out, 0.0)
        n_arr[:, stage + 1] = n_out
        n_in = n_out.copy()

    n_total = np.sum(np.maximum(n_arr, 0.0), axis=0) + nI0

    return {
        'stage': np.arange(n_stages + 1),
        'V': V_arr,
        'A': n_arr[0],
        'B': n_arr[1],
        'C': n_arr[2],
        'D': n_arr[3],
        'n_total': n_total,
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_mechanism_names() -> Dict[int, str]:
    """Return {class_id: name} mapping for all registered mechanisms."""
    return {cid: m.name for cid, m in MECHANISM_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    P = 100_000          # Pa
    V_total = 1.0        # m^3
    nI0 = 0.05

    print(f"Testing ODE solver for all {len(MECHANISM_REGISTRY)} mechanism classes...\n")

    for cid, mech in sorted(MECHANISM_REGISTRY.items()):
        print(f"Class {cid:2d}: {mech.name} ({mech.type_label})")

        # Build default rate constants at midpoint of ranges
        rc = {}
        for name, (lo, hi) in mech.rate_constant_ranges.items():
            rc[name] = (lo + hi) / 2

        n0 = np.array([0.2, 0.15, 0.0, 0.0])

        # PFR
        pfr = solve_pfr(cid, n0, nI0, rc, P, V_total, n_points=100)
        print(f"  PFR outlet: A={pfr['A'][-1]:.6f}, B={pfr['B'][-1]:.6f}, "
              f"C={pfr['C'][-1]:.6f}, D={pfr['D'][-1]:.6f}")

        # CSTR
        cstr = solve_cstr_cascade(cid, n0, nI0, rc, P, V_total, n_stages=50)
        print(f"  CSTR outlet: A={cstr['A'][-1]:.6f}, B={cstr['B'][-1]:.6f}, "
              f"C={cstr['C'][-1]:.6f}, D={cstr['D'][-1]:.6f}")

        # Check intermediate rise-then-fall for sequential mechanisms
        if mech.type_label in ("sequential", "parallel_sequential"):
            c_max = np.max(pfr['C'])
            c_end = pfr['C'][-1]
            if c_max > 1e-10:
                print(f"  Intermediate C: max={c_max:.6f}, end={c_end:.6f} "
                      f"(rise-then-fall: {c_end < c_max * 0.99})")

        print()

    print("All mechanisms solved successfully!")
