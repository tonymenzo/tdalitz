"""B0 -> pi+ pi- pi0 amplitudes from the BaBar time-dependent Dalitz analysis.

BaBar (arXiv:1304.3503, Table XIV) reports the fit as 27 "U and I"
coefficients (U_+^+ = 1 fixes the normalization) of the amplitudes

    A_3pi    = f_+ A_+    + f_- A_-    + f_0 A_0,
    Abar_3pi = f_+ Abar_+ + f_- Abar_- + f_0 Abar_0,

with kappa in {+, -, 0} for the rho+, rho-, rho0 and q/p absorbed into Abar
(|q/p| = 1).  Their definitions (Eqs. 12-16):

    U_k^pm          = |A_k|^2 pm |Abar_k|^2
    U_ks^{pm,Re}    = Re[A_k A_s* pm Abar_k Abar_s*]     (Im likewise)
    I_k             = Im[Abar_k A_k*]
    I_ks^{Re}       = Re[Abar_k A_s* - Abar_s A_k*]
    I_ks^{Im}       = Im[Abar_k A_s* + Abar_s A_k*]

This script finds the six complex amplitudes (11 real parameters; the global
phase is fixed by A_+ real and positive) that best reproduce the 27 central
values, by least squares with the total (stat + syst) uncertainties, ignoring
correlations.  As a check it recomputes BaBar's quasi-two-body parameters
(Table XVIII) from the fitted amplitudes.

    python scripts/fit_b0_3pi_amplitudes.py
Output: src/tdalitz/b0_3pi_amplitudes.json (read by tdalitz.channels)
"""

import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "src" / "tdalitz" / "b0_3pi_amplitudes.json"

#: BaBar Table XIV: name -> (value, stat, syst).  Pair indices as printed.
TABLE = {
    "I_0": (-0.042, 0.038, 0.022), "I_-": (-0.00, 0.06, 0.03),
    "I_-0^Im": (-0.61, 0.43, 0.46), "I_-0^Re": (0.4, 0.6, 0.8),
    "I_+": (0.05, 0.06, 0.03),
    "I_+0^Im": (-0.04, 0.36, 0.43), "I_+0^Re": (0.5, 0.5, 0.7),
    "I_+-^Im": (-0.5, 0.7, 0.9), "I_+-^Re": (-0.6, 0.8, 1.0),
    "U_0^-": (0.04, 0.05, 0.03), "U_0^+": (0.225, 0.030, 0.020),
    "U_-0^-Im": (0.53, 0.44, 0.52), "U_-0^-Re": (0.49, 0.35, 0.37),
    "U_-0^+Im": (-0.39, 0.20, 0.24), "U_-0^+Re": (-0.05, 0.17, 0.18),
    "U_-^-": (-0.27, 0.10, 0.06), "U_-^+": (1.22, 0.07, 0.05),
    "U_+0^-Im": (0.10, 0.29, 0.45), "U_+0^-Re": (0.30, 0.32, 0.38),
    "U_+0^+Im": (0.41, 0.16, 0.17), "U_+0^+Re": (0.01, 0.15, 0.19),
    "U_+-^-Im": (1.1, 0.5, 0.8), "U_+-^-Re": (-0.5, 0.5, 0.8),
    "U_+-^+Im": (-0.07, 0.26, 0.26), "U_+-^+Re": (-0.19, 0.25, 0.33),
    "U_+^-": (0.25, 0.09, 0.07),
    "U_+^+": (1.0, 1e-4, 0.0),          # fixed in the BaBar fit
}

#: BaBar Table XVIII, for the cross-check.
Q2B = {"A_rhopi": -0.100, "C": 0.016, "DeltaC": 0.234, "S": 0.053, "DeltaS": 0.054,
       "C00": 0.19, "S00": -0.37, "f00": 0.092}

KAPPA = ("+", "-", "0")


def observables(a, ab):
    """The 27 U/I values for amplitudes a[k], ab[k], k in KAPPA."""
    out = {}
    for k in KAPPA:
        out[f"U_{k}^+"] = abs(a[k]) ** 2 + abs(ab[k]) ** 2
        out[f"U_{k}^-"] = abs(a[k]) ** 2 - abs(ab[k]) ** 2
        out[f"I_{k}"] = np.imag(ab[k] * np.conj(a[k]))
    for k, s in (("+", "-"), ("+", "0"), ("-", "0")):
        for sign, label in ((+1, "+"), (-1, "-")):
            z = a[k] * np.conj(a[s]) + sign * ab[k] * np.conj(ab[s])
            out[f"U_{k}{s}^{label}Re"] = z.real
            out[f"U_{k}{s}^{label}Im"] = z.imag
        out[f"I_{k}{s}^Re"] = np.real(ab[k] * np.conj(a[s]) - ab[s] * np.conj(a[k]))
        out[f"I_{k}{s}^Im"] = np.imag(ab[k] * np.conj(a[s]) + ab[s] * np.conj(a[k]))
    return out


def unpack(p):
    a = {"+": p[0] + 0j, "-": p[1] + 1j * p[2], "0": p[3] + 1j * p[4]}
    ab = {"+": p[5] + 1j * p[6], "-": p[7] + 1j * p[8], "0": p[9] + 1j * p[10]}
    return a, ab


def residuals(p):
    pred = observables(*unpack(p))
    return np.array([(pred[n] - v) / np.hypot(st, sy) for n, (v, st, sy) in TABLE.items()])


def q2b(a, ab):
    """BaBar's quasi-two-body parameters, Eqs. (17)-(22) and the rho0 pi0 ones."""
    u = observables(a, ab)
    Up, Um = u["U_+^+"], u["U_-^+"]
    Cp, Cm = u["U_+^-"] / Up, u["U_-^-"] / Um
    Sp, Sm = 2 * u["I_+"] / Up, 2 * u["I_-"] / Um
    return {"A_rhopi": (Up - Um) / (Up + Um), "C": (Cp + Cm) / 2, "DeltaC": (Cp - Cm) / 2,
            "S": (Sp + Sm) / 2, "DeltaS": (Sp - Sm) / 2,
            "C00": u["U_0^-"] / u["U_0^+"], "S00": 2 * u["I_0"] / u["U_0^+"],
            "f00": u["U_0^+"] / (Up + Um + u["U_0^+"])}


def fit(n_starts=400, seed=1):
    """Global least squares from many random starts; returns all distinct minima."""
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(n_starts):
        p0 = rng.normal(scale=0.7, size=11)
        p0[0] = abs(p0[0]) + 0.1
        r = least_squares(residuals, p0, bounds=([0] + [-np.inf] * 10, np.inf))
        sols.append((2 * r.cost, r.x))
    sols.sort(key=lambda t: t[0])
    distinct = []
    for chi2, p in sols:
        if not any(abs(chi2 - c) < 1e-3 for c, _ in distinct):
            distinct.append((chi2, p))
    return distinct


def main():
    sols = fit()
    chi2, p = sols[0]
    a, ab = unpack(p)
    ndf = len(TABLE) - 11
    out = {
        "source": "BaBar, arXiv:1304.3503, Table XIV (U and I coefficients)",
        "convention": "BaBar: Abar includes q/p; amplitudes multiply f_kappa(m, theta_kappa) "
                      "with BaBar's helicity angles; A_+ real",
        "chi2": chi2, "ndf": ndf,
        "A": {k: [a[k].real, a[k].imag] for k in KAPPA},
        "Abar": {k: [ab[k].real, ab[k].imag] for k in KAPPA},
        "q2b_fitted": q2b(a, ab), "q2b_babar": Q2B,
        "other_minima_chi2": [c for c, _ in sols[1:6]],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"chi2/ndf = {chi2:.2f}/{ndf}; next minima: {[round(c, 2) for c, _ in sols[1:6]]}")
    for name, v in out["q2b_fitted"].items():
        print(f"  {name:8s} fitted {v:+.3f}   BaBar {Q2B[name]:+.3f}")
    for k in KAPPA:
        print(f"  A_{k} = {a[k]:.3f}   Abar_{k} = {ab[k]:.3f}")


if __name__ == "__main__":
    main()
