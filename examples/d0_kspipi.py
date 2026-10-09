"""End-to-end example: export a D0 -> KS pi+ pi- model from Laura++, then
generate time-dependent tagged events with SM charm mixing.

The export step needs ROOT and Laura++; everything after it is pure NumPy.
"""

import numpy as np

from tdalitz import Coefficient, MixingParameters, apply_time_resolution, generate_pair
from tdalitz.exporters.laura import LauraModel, export

# BaBar isobar model (arXiv:1004.5053): magnitudes are sqrt(fit fraction) in the
# unit-integral convention, phases are the published relative phases.
COEFFICIENTS = {
    "K*-(892)":    Coefficient(np.sqrt(0.570),  2.331),
    "rho0(770)":   Coefficient(np.sqrt(0.211),  0.000),
    "K*-_0(1430)": Coefficient(np.sqrt(0.061),  1.497),
    "K*-_2(1430)": Coefficient(np.sqrt(0.019),  2.498),
}


def main(library: str, out: str = "d0_kspipi.npz", n_points: int = 500_000):
    model = LauraModel(
        parent="D0",
        daughters=("K_S0", "pi+", "pi-"),
        # (name, bachelor index): pi+pi- -> 1, KS pi- -> 2, KS pi+ -> 3
        resonances=[("K*-(892)", 2), ("rho0(770)", 1),
                    ("K*-_0(1430)", 2), ("K*-_2(1430)", 2)],
        barrier_radii={"Parent": 5.0, "Kstar": 1.5, "Light": 1.5},
        library=library,
    )
    table = export(model, n_points=n_points, seed=1)
    table.save(out)
    print("fit fractions:", {k: round(v, 3)
                             for k, v in table.fit_fractions(COEFFICIENTS).items()})

    # SM charm: x, y from HFLAV; q/p = 1, so this sample is CP conserving.
    # HFLAV's y has the opposite sign to this package's (see docs/conventions.md).
    mixing = MixingParameters(x=0.0041, y=-0.0064, qp=1.0 + 0j)
    rng = np.random.default_rng(0)
    plus, minus = generate_pair(table, COEFFICIENTS, mixing, 50_000, rng)
    plus = apply_time_resolution(plus, sigma_tau=0.05, rng=rng)
    print(f"generated {len(plus)} D0 and {len(minus)} D0bar decays")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "libLaura++")
