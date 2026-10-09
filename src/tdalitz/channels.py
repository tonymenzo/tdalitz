"""The decay channels used in the paper, as Laura++ models.

D0, B0 and Bs -> KS pi+ pi- have daughters ordered (KS, pi+, pi-), and
B0 -> pi+ pi- pi0 has (pi0, pi+, pi-): in every channel CP exchanges daughters
2 and 3, so the CP conjugate is the mirrored point (``mirror=True`` in the
exporter).  For the KS pi+ pi- channels isobar magnitudes are
``sqrt(fit fraction)``: in the unit-integral normalization every component
carries, the fit-fraction *ratios* are then exact.

Mixing parameters use this package's convention (see docs/conventions.md and
``tests/test_conventions.py``):

    |P_L> = p|P0> + q|P0bar>,   |P_H> = p|P0> - q|P0bar>,
    x = (m_H - m_L) / Gamma,    y = (Gamma_L - Gamma_H) / (2 Gamma).

Sources
-------
D0:  isobar content, fit fractions and phases from the BaBar model of
     arXiv:1004.5053 (Table I), keeping 6 of its components.  Omitted: the
     pi pi S-wave (K-matrix, 15.4%), K*(1680)-, f2(1270), and the DCS
     K0*(1430)+ and K2*(1430)+ (each <= 0.3%).  x, y: HFLAV 2023
     (x = 0.407%, y = 0.645%); q/p = 1, the Standard Model point.
     HFLAV defines y = (Gamma_2 - Gamma_1)/(2 Gamma) with D_2 = p D0 - q D0bar,
     the opposite sign to ours, so y enters here negative.
B0:  the BaBar time-dependent Dalitz analysis (arXiv:0905.3615): model content,
     lineshape parameters and barrier radii from its Table I; B0 and B0bar
     amplitudes (Solution I) from Table IV, scaled to its fit fractions (Table V).
     x_d from HFLAV (PDG 2024); sin 2beta = 0.709 from HFLAV 2023.
Bs:  representative model (no measured amplitude analysis); Delta m_s,
     Gamma_s, Delta Gamma_s from HFLAV (PDG 2024); mixing phase at the Standard
     Model value phi_s = -2 beta_s = -0.0370, i.e. q/p = exp(-i phi_s).
B0 -> pi+ pi- pi0:  the rho+, rho- and rho0 (770) amplitudes of the BaBar
     time-dependent Dalitz analysis (arXiv:1304.3503), fitted to its U and I
     coefficients by scripts/fit_b0_3pi_amplitudes.py (written to
     src/tdalitz/b0_3pi_amplitudes.json).  BaBar's rho(1450) admixture is not reported and
     is omitted.  Generated in the coherent (B-factory, Delta t) mode.
"""

from __future__ import annotations

import json
import pathlib

from dataclasses import dataclass

import numpy as np

from .amplitude import Coefficient
from .exporters.laura import LauraModel
from .mixing import MixingParameters

# Masses are not set here: the exporter takes them from Laura++'s particle
# database so that tabulated kinematics coincide exactly with Laura++'s.
KS_PI_PI = ("K_S0", "pi+", "pi-")

#: Blatt-Weisskopf radii (GeV^-1) shared by all channels.
BARRIER_RADII = {"Parent": 5.0, "Light": 1.5, "Kstar": 1.5, "Charmonium": 1.5}


@dataclass(frozen=True)
class Channel:
    key: str
    title: str
    model: LauraModel
    fit_fractions: dict           # target fit fraction per component
    coefficients: dict            # Coefficient per component
    mixing: MixingParameters
    notes: str = ""
    coherent: bool = False        # generate Delta t of a coherent pair (B factory)


def _coefficients(fit_fractions: dict, phases: dict) -> dict:
    return {name: Coefficient(float(np.sqrt(ff)), phases[name])
            for name, ff in fit_fractions.items()}


# ----------------------------------------------------------------------- D0
_D0_FF = {"K*-(892)": 0.570, "rho0(770)": 0.211, "K*-_0(1430)": 0.061,
          "K*-_2(1430)": 0.019, "omega(782)": 0.006, "K*+(892)": 0.006}
_D0_PHASE = {"K*-(892)": 2.331, "rho0(770)": 0.000, "K*-_0(1430)": 1.497,
             "K*-_2(1430)": 2.498, "omega(782)": 2.046, "K*+(892)": -0.768}
D0 = Channel(
    key="d0",
    title=r"$D^0\to K_S^0\pi^+\pi^-$",
    model=LauraModel(
        parent="D0", daughters=KS_PI_PI,
        resonances=[("K*-(892)", 2), ("rho0(770)", 1), ("K*-_0(1430)", 2),
                    ("K*-_2(1430)", 2), ("omega(782)", 1), ("K*+(892)", 3)],
        barrier_radii=BARRIER_RADII,
    ),
    fit_fractions=_D0_FF,
    coefficients=_coefficients(_D0_FF, _D0_PHASE),
    mixing=MixingParameters(x=0.00407, y=-0.00645, qp=1.0 + 0j),
    notes="SM charm: CP conserving (q/p = 1, no weak phases) -- the null channel.",
)

# ----------------------------------------------------------------------- B0
_SIN_2BETA = 0.709                                        # HFLAV 2023
_BETA = 0.5 * np.arcsin(_SIN_2BETA)
_QP_B0 = np.exp(-2j * _BETA)

#: BaBar arXiv:0905.3615, Table IV, Solution I: (|c|, arg c, |cbar|, arg cbar),
#: phases in degrees, cbar including q/p.  BaBar's components are not normalized
#: to unit integral, so only ratios within a component and phases carry over.
_BABAR_B0 = {
    "f_0(980)": (4.0, 0.0, 3.7, -73.9), "rho0(770)": (0.10, 35.6, 0.11, 15.3),
    "K*+(892)": (0.154, -138.7, 0.125, 163.1), "K*+_0(1430)": (6.9, -151.7, 7.6, 136.2),
    "f_2(1270)": (0.014, 5.8, 0.011, -24.0), "f_0(1370)": (1.41, 43.2, 1.24, 31.6),
    "NonReson": (2.6, 35.3, 2.7, 36.1), "chi_c0": (0.33, 61.4, 0.44, 15.1),
}
#: BaBar Table V, Solution I: fit fractions, (|c|^2 + |cbar|^2) <|F|^2> / total.
_B0_FF = {"f_0(980)": 0.138, "rho0(770)": 0.086, "K*+(892)": 0.110, "K*+_0(1430)": 0.452,
          "f_2(1270)": 0.023, "f_0(1370)": 0.036, "NonReson": 0.115, "chi_c0": 0.0104}

#: Sign of each Laura++ component relative to BaBar's F_k, whose angular factors
#: are T_1 = -4 p.q and T_2 = (8/3)[3 (p.q)^2 - |p|^2 |q|^2], with p the bachelor and
#: q the KS (KS pi-), pi+ (KS pi+) or pi- (pi+ pi-) daughter.  Measured from the
#: table (tests/test_channels.py); +1 for the S-waves, whose lineshapes match.
HELICITY_SIGN_B0 = {"f_0(980)": +1, "rho0(770)": -1, "K*+(892)": -1, "K*+_0(1430)": +1,
                    "f_2(1270)": +1, "f_0(1370)": +1, "NonReson": +1, "chi_c0": +1}


def _coefficients_b0() -> dict:
    """BaBar's amplitudes -> (c, cbar) per Laura++ component.

    Laura++'s F_k = lambda_k s_k F_k^BaBar with lambda_k > 0, so c_k = s_k c_k^BaBar / lambda_k
    keeps BaBar's phases and cbar/c.  The positive scales are fixed by BaBar's fit
    fractions, which for unit-integral components are |c_k|^2 + |cbar_k|^2.  BaBar's
    cbar includes q/p, so ours is divided by it.
    """
    out = {}
    for name, (m, ph, mb, phb) in _BABAR_B0.items():
        c = m * np.exp(1j * np.radians(ph))
        cb = mb * np.exp(1j * np.radians(phb))
        scale = np.sqrt(_B0_FF[name] / (abs(c) ** 2 + abs(cb) ** 2))
        sign = HELICITY_SIGN_B0[name]
        out[name] = Coefficient.from_complex(sign * scale * c, sign * scale * cb / _QP_B0)
    return out


B0 = Channel(
    key="b0",
    title=r"$B^0\to K_S^0\pi^+\pi^-$",
    model=LauraModel(
        parent="B0", daughters=KS_PI_PI,
        resonances=[("K*+(892)", 3), ("K*+_0(1430)", 3), ("rho0(770)", 1),
                    ("f_0(980)", 1), ("f_2(1270)", 1), ("f_0(1370)", 1),
                    ("chi_c0", 1), ("NonReson", 0)],
        lineshapes={"K*+_0(1430)": "LASS", "rho0(770)": "GS",
                    "f_0(980)": "Flatte", "NonReson": "FlatNR"},
        # BaBar Table I: r = 2.0 (B), 5.3 (rho), 3.6 (K*) GeV^-1.  The f2(1270) shares
        # the Light category with the rho, so it gets 5.3 instead of BaBar's 3.0.
        barrier_radii={"Parent": 2.0, "Light": 5.3, "Kstar": 3.6, "Charmonium": 1.5},
        # BaBar Table I masses and widths; f_0(1370) stands in for their fX(1300).
        # The Flatte couplings and LASS parameters are Laura++'s defaults, which equal BaBar's.
        mass_width_overrides={"rho0(770)": (0.7755, 0.1464, 1),
                              "K*+_0(1430)": (1.415, 0.300, 0),
                              "f_0(980)": (0.965, 0.070, 0),
                              "f_2(1270)": (1.2754, 0.1852, 2),
                              "f_0(1370)": (1.471, 0.097, 0),
                              "chi_c0": (3.41475, 0.0104, 0)},
    ),
    fit_fractions=_B0_FF,
    coefficients=_coefficients_b0(),
    mixing=MixingParameters(x=0.7697, y=0.0, qp=_QP_B0),
    notes="BaBar 2009 amplitudes (Solution I): direct and mixing-induced CPV.",
)

# ----------------------------------------------------------------------- Bs
_GAMMA_S, _DM_S, _DGAMMA_S = 0.6581, 17.765, 0.083        # ps^-1, HFLAV PDG 2024
_PHI_S = -0.0370                                          # SM, -2 beta_s
_BS_FF = {"K*+(892)": 0.25, "K*-(892)": 0.10, "rho0(770)": 0.20,
          "f_0(980)": 0.20, "f_2(1270)": 0.10, "NonReson": 0.15}
_BS_PHASE = {"K*+(892)": 0.0, "K*-(892)": 1.0, "rho0(770)": 2.0,
             "f_0(980)": -1.0, "f_2(1270)": 0.5, "NonReson": 1.0}
BS = Channel(
    key="bs",
    title=r"$B_s^0\to K_S^0\pi^+\pi^-$",
    model=LauraModel(
        parent="B_s0", daughters=KS_PI_PI,
        resonances=[("K*+(892)", 3), ("K*-(892)", 2), ("rho0(770)", 1),
                    ("f_0(980)", 1), ("f_2(1270)", 1), ("NonReson", 0)],
        lineshapes={"rho0(770)": "GS", "f_0(980)": "Flatte", "NonReson": "FlatNR"},
        barrier_radii=BARRIER_RADII,
    ),
    fit_fractions=_BS_FF,
    coefficients=_coefficients(_BS_FF, _BS_PHASE),
    mixing=MixingParameters(x=_DM_S / _GAMMA_S, y=_DGAMMA_S / (2 * _GAMMA_S),
                            qp=np.exp(-1j * _PHI_S)),
    notes="Representative amplitude at the SM mixing phase; the regime (x_s ~ 27) is the point.",
)

# ----------------------------------------------------------- B0 -> pi+ pi- pi0
_3PI = json.loads((pathlib.Path(__file__).resolve().parent / "b0_3pi_amplitudes.json").read_text())
_A = {k: complex(*v) for k, v in _3PI["A"].items()}         # BaBar convention:
_AB = {k: complex(*v) for k, v in _3PI["Abar"].items()}     # Abar includes q/p

#: Sign of each Laura++ component relative to BaBar's f_kappa, whose helicity
#: angles are pi0 vs the recoiling charged pion (rho+-) and pi+ vs the recoiling
#: pi0 (rho0).  Measured from the table (tests/test_channels.py).
HELICITY_SIGN_3PI = {"rho+(770)": +1, "rho-(770)": -1, "rho0(770)": +1}
_QP_3PI = _QP_B0


def _coefficients_3pi() -> dict:
    """BaBar amplitudes -> (c, cbar) per Laura++ component.

    A = sum_k A_k f_k with F_k = s_k f_k gives c_k = A_k / s_k.  The CP
    conjugate is evaluated at the mirrored point, where Laura++'s rho+ becomes
    s_+ f_-, its rho- becomes s_- f_+, and its rho0 becomes -s_0 f_0 (the pi+ <->
    pi- exchange reverses the rho0 helicity angle).  Matching (q/p) Abar to
    BaBar's sum_k Abar_k f_k then fixes cbar.
    """
    s = HELICITY_SIGN_3PI
    qp = _QP_3PI
    return {
        "rho+(770)": Coefficient.from_complex(_A["+"] / s["rho+(770)"],
                                              _AB["-"] / (qp * s["rho+(770)"])),
        "rho-(770)": Coefficient.from_complex(_A["-"] / s["rho-(770)"],
                                              _AB["+"] / (qp * s["rho-(770)"])),
        "rho0(770)": Coefficient.from_complex(_A["0"] / s["rho0(770)"],
                                              -_AB["0"] / (qp * s["rho0(770)"])),
    }


_COEFFS_3PI = _coefficients_3pi()
_FF_3PI = {k: abs(c.c) ** 2 / sum(abs(v.c) ** 2 for v in _COEFFS_3PI.values())
           for k, c in _COEFFS_3PI.items()}
B0_3PI = Channel(
    key="b0_3pi",
    title=r"$B^0\to\pi^+\pi^-\pi^0$",
    model=LauraModel(
        parent="B0", daughters=("pi0", "pi+", "pi-"),
        resonances=[("rho+(770)", 3), ("rho-(770)", 2), ("rho0(770)", 1)],
        lineshapes={"rho+(770)": "GS", "rho-(770)": "GS", "rho0(770)": "GS"},
        barrier_radii=BARRIER_RADII,
        # BaBar's rho masses and widths (arXiv:1304.3503, Table V).
        mass_width_overrides={"rho+(770)": (0.7755, 0.1482, 1),
                              "rho-(770)": (0.7755, 0.1482, 1),
                              "rho0(770)": (0.7731, 0.1480, 1)},
    ),
    fit_fractions=_FF_3PI,
    coefficients=_COEFFS_3PI,
    mixing=MixingParameters(x=0.7697, y=0.0, qp=_QP_3PI),
    notes="BaBar rho pi amplitudes; direct and mixing-induced CPV; coherent Delta t.",
    coherent=True,
)

CHANNELS = {ch.key: ch for ch in (D0, B0, BS, B0_3PI)}
