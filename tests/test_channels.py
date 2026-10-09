"""Physics checks on the decay channels used in the paper.

These guard the *inputs*: mixing parameters in this package's convention
(x = (m_H - m_L)/Gamma, y = (Gamma_L - Gamma_H)/(2 Gamma); see
test_conventions.py), CP-violation settings, and the isobar-model bookkeeping.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tdalitz.channels import CHANNELS  # noqa: E402

M_KS, M_PI = 0.497611, 0.13957039


def test_paper_channels_are_defined():
    assert set(CHANNELS) == {"d0", "b0", "bs", "b0_3pi"}


def test_only_b0_3pi_is_coherent():
    assert {k for k, ch in CHANNELS.items() if ch.coherent} == {"b0_3pi"}


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_final_state_is_ks_pi_pi_in_mirror_order(key):
    """Daughters 2 and 3 must be the pion pair, since the CP-conjugate
    amplitude is built by exchanging them."""
    assert CHANNELS[key].model.daughters == ("K_S0", "pi+", "pi-")


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_resonances_sit_in_the_pair_their_charge_requires(key):
    """Laura++ addresses a resonance by its bachelor: pi+ pi- -> 1,
    KS pi- -> 2 (bachelor pi+), KS pi+ -> 3 (bachelor pi-).  A flat
    nonresonant term belongs to no pair and takes index 0."""
    for name, bachelor in CHANNELS[key].model.resonances:
        if name == "NonReson":
            assert bachelor == 0
        elif name.startswith("K*-"):
            assert bachelor == 2, name
        elif name.startswith("K*+"):
            assert bachelor == 3, name
        else:
            assert bachelor == 1, name


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_coefficients_cover_the_model_and_reproduce_fit_fraction_ratios(key):
    """With unit-integral components, FF_r / FF_s = (|c_r|^2 + |cbar_r|^2) / (...)_s
    exactly (BaBar's definition, which reduces to |c_r|^2 / |c_s|^2 when |cbar| = |c|)."""
    ch = CHANNELS[key]
    assert set(ch.coefficients) == set(ch.model.component_names)
    names = list(ch.fit_fractions)
    mags2 = np.array([abs(ch.coefficients[n].c) ** 2 + abs(ch.coefficients[n].c_bar) ** 2
                      for n in names])
    ff = np.array([ch.fit_fractions[n] for n in names])
    assert np.allclose(mags2 / mags2.sum(), ff / ff.sum())


def test_d0_is_cp_conserving():
    ch = CHANNELS["d0"]
    assert ch.mixing.qp == pytest.approx(1.0 + 0j)
    assert all(c.weak == 0.0 for c in ch.coefficients.values())


def test_d0_heavier_eigenstate_is_the_shorter_lived():
    """HFLAV: D1 = p D0 + q D0bar, D2 = p D0 - q D0bar, x = (m2 - m1)/Gamma > 0,
    y = (Gamma2 - Gamma1)/(2 Gamma) > 0, so the heavier state (D2, our H) has
    the larger width.  In our convention y = (Gamma_L - Gamma_H)/(2 Gamma) < 0."""
    mix = CHANNELS["d0"].mixing
    assert mix.x > 0.0
    assert mix.y < 0.0
    # HFLAV 2023 (CKM 2023 WG7): x = 0.407%, y = 0.645%.
    assert mix.x == pytest.approx(0.00407, abs=1e-6)
    assert mix.y == pytest.approx(-0.00645, abs=1e-6)


def test_b0_mixing_matches_world_averages():
    """HFLAV: x_d = 0.7697, Delta Gamma_d ~ 0; HFLAV 2023: sin 2beta = 0.709.
    With q/p = exp(-2 i beta), a CP-even component gives S = -sin 2beta, so
    sin 2beta = -Im(q/p)."""
    mix = CHANNELS["b0"].mixing
    assert mix.x == pytest.approx(0.7697, abs=1e-3)
    assert mix.y == 0.0
    assert abs(mix.qp) == pytest.approx(1.0)
    assert -mix.qp.imag == pytest.approx(0.709, abs=1e-4)


def test_bs_light_eigenstate_is_the_shorter_lived():
    """HFLAV (PDG 2024): Delta m_s = 17.765/ps, Gamma_s = 0.6581/ps,
    Delta Gamma_s = Gamma_L - Gamma_H = +0.083/ps, i.e. y > 0 here."""
    mix = CHANNELS["bs"].mixing
    assert mix.x == pytest.approx(17.765 / 0.6581, abs=0.01)
    assert mix.y == pytest.approx(0.083 / (2 * 0.6581), abs=5e-4)
    assert abs(mix.qp) == pytest.approx(1.0)


def test_bs_mixing_phase_is_the_standard_model_value():
    """phi_s = -2 beta_s = -0.0370 in the SM.  In our convention, where
    q/p = exp(-2i beta) for B0 (phi_d = 2 beta), q/p = exp(-i phi_s) for Bs."""
    assert np.angle(CHANNELS["bs"].mixing.qp) == pytest.approx(0.0370, abs=1e-4)


# ------------------------------------------------------------- B0 -> pi+ pi- pi0
B3PI_TABLE = pathlib.Path(__file__).resolve().parents[1] / "tables" / "b0_3pi.npz"
needs_3pi_table = pytest.mark.skipif(not B3PI_TABLE.exists(), reason="b0_3pi table not present")


def test_b0_3pi_final_state_puts_the_charged_pions_in_mirror_order():
    """CP exchanges pi+ and pi-, which must be daughters 2 and 3."""
    assert CHANNELS["b0_3pi"].model.daughters == ("pi0", "pi+", "pi-")


def test_b0_3pi_mixing_matches_the_b0_channel():
    a, b = CHANNELS["b0_3pi"].mixing, CHANNELS["b0"].mixing
    assert (a.x, a.y) == (b.x, b.y)
    assert a.qp == pytest.approx(b.qp)


def _babar_amplitudes():
    import json
    d = json.loads((pathlib.Path(__file__).resolve().parents[1] / "src" / "tdalitz"
                    / "b0_3pi_amplitudes.json").read_text())
    return ({k: complex(*v) for k, v in d["A"].items()},
            {k: complex(*v) for k, v in d["Abar"].items()})


def _helicity_cosines(table):
    """BaBar's helicity angles (arXiv:1304.3503, Sec. II A) at every pool point."""
    fs = table.final_state
    M = fs.parent
    m1, m2, m3 = fs.daughters                     # pi0, pi+, pi-
    s12, s13 = table.s12, table.s13
    s23 = fs.s23(s12, s13)

    def cos_between(s_pair, ma, mb, mc, s_ac):    # angle(a, c) in the (ab) frame
        r = np.sqrt(s_pair)
        ea, ec = (s_pair + ma**2 - mb**2) / (2 * r), (M**2 - s_pair - mc**2) / (2 * r)
        pa, pc = np.sqrt(np.maximum(ea**2 - ma**2, 0)), np.sqrt(np.maximum(ec**2 - mc**2, 0))
        return (ma**2 + mc**2 + 2 * ea * ec - s_ac) / (2 * pa * pc)

    return ({"rho+(770)": -cos_between(s12, m1, m2, m3, s13),
             "rho-(770)": -cos_between(s13, m1, m3, m2, s12),
             "rho0(770)": -cos_between(s23, m2, m3, m1, s12)},
            {"rho+(770)": s12, "rho-(770)": s13, "rho0(770)": s23})


@needs_3pi_table
def test_b0_3pi_helicity_signs_match_the_table():
    """The sign of each Laura++ component relative to BaBar's f_kappa, measured
    below the rho peak where the Breit-Wigner is nearly real and positive."""
    from tdalitz import AmplitudeTable
    from tdalitz.channels import HELICITY_SIGN_3PI
    table = AmplitudeTable.load(B3PI_TABLE)
    cos, pair_s = _helicity_cosines(table)
    for name in table.components:
        m = np.sqrt(pair_s[name])
        sel = (m > 0.45) & (m < 0.62) & (np.abs(cos[name]) > 0.3)
        r = table.amp[sel, table.index(name)] / cos[name][sel]
        assert np.all(np.sign(r.real) == HELICITY_SIGN_3PI[name])


#: The component whose CP image populates the same band in Abar.
_CP_PARTNER = {"rho+(770)": "rho-(770)", "rho-(770)": "rho+(770)", "rho0(770)": "rho0(770)"}


def _isolated_band_cp_parameters(table, ch, band):
    """C and S of one rho band from pool integrals of the generator's rate,
    keeping only the band's own component in A and its CP partner in Abar.

    BaBar's C_kappa = U_kappa^- / U_kappa^+ and S_kappa = 2 I_kappa / U_kappa^+
    are properties of the amplitudes alone; isolating the band removes the
    interference with the other bands' tails, which is physics, not part of
    these parameters.  S involves the product of a component with its mirrored
    partner, so it tests the helicity signs, the CP mirror and q/p together.
    """
    from tdalitz import Coefficient
    from dataclasses import replace
    coeffs = {name: Coefficient.from_complex(c.c if name == band else 0.0,
                                             c.c_bar if name == _CP_PARTNER[band] else 0.0)
              for name, c in ch.coefficients.items()}
    fs = table.final_state
    m = {"rho+(770)": np.sqrt(table.s12), "rho-(770)": np.sqrt(table.s13),
         "rho0(770)": np.sqrt(fs.s23(table.s12, table.s13))}
    sel = (m[band] > 0.6) & (m[band] < 0.95)
    a, a_bar = table.total(coeffs)
    b = ch.mixing.qp * a_bar                       # BaBar's Abar includes q/p
    a, b = a[sel], b[sel]
    up = np.sum(np.abs(a) ** 2 + np.abs(b) ** 2)
    return {"U+": up, "C": np.sum(np.abs(a) ** 2 - np.abs(b) ** 2) / up,
            "S": 2 * np.sum(np.imag(b * np.conj(a))) / up}


@needs_3pi_table
@pytest.mark.parametrize("band, kappa", [("rho+(770)", "+"), ("rho-(770)", "-"), ("rho0(770)", "0")])
def test_b0_3pi_reproduces_babar_cp_parameters_in_each_band(band, kappa):
    from tdalitz import AmplitudeTable
    table = AmplitudeTable.load(B3PI_TABLE)
    a, ab = _babar_amplitudes()
    up = abs(a[kappa]) ** 2 + abs(ab[kappa]) ** 2
    got = _isolated_band_cp_parameters(table, CHANNELS["b0_3pi"], band)
    assert got["C"] == pytest.approx((abs(a[kappa]) ** 2 - abs(ab[kappa]) ** 2) / up, abs=1e-3)
    assert got["S"] == pytest.approx(2 * np.imag(ab[kappa] * np.conj(a[kappa])) / up, abs=1e-3)


@needs_3pi_table
def test_b0_3pi_reproduces_the_babar_charge_asymmetry():
    """A_rhopi = (U_+^+ - U_-^+) / (U_+^+ + U_-^+); the two charged bands have
    identical acceptance, so their summed rates must reproduce it."""
    from tdalitz import AmplitudeTable
    table = AmplitudeTable.load(B3PI_TABLE)
    a, ab = _babar_amplitudes()
    up, um = (abs(a[k]) ** 2 + abs(ab[k]) ** 2 for k in ("+", "-"))
    plus = _isolated_band_cp_parameters(table, CHANNELS["b0_3pi"], "rho+(770)")["U+"]
    minus = _isolated_band_cp_parameters(table, CHANNELS["b0_3pi"], "rho-(770)")["U+"]
    assert (plus - minus) / (plus + minus) == pytest.approx((up - um) / (up + um), abs=2e-3)


# ------------------------------------------------------- B0 -> KS pi+ pi- (BaBar 2009)
#: BaBar arXiv:0905.3615, Table IV, Solution I: (|c|, arg c [deg], |cbar|, arg cbar [deg]);
#: cbar includes q/p.  Table V: fit fractions in percent.
BABAR_B0 = {
    "f_0(980)": (4.0, 0.0, 3.7, -73.9), "rho0(770)": (0.10, 35.6, 0.11, 15.3),
    "K*+(892)": (0.154, -138.7, 0.125, 163.1), "K*+_0(1430)": (6.9, -151.7, 7.6, 136.2),
    "f_2(1270)": (0.014, 5.8, 0.011, -24.0), "f_0(1370)": (1.41, 43.2, 1.24, 31.6),
    "NonReson": (2.6, 35.3, 2.7, 36.1), "chi_c0": (0.33, 61.4, 0.44, 15.1),
}
BABAR_B0_FF = {"f_0(980)": 13.8, "rho0(770)": 8.6, "K*+(892)": 11.0, "K*+_0(1430)": 45.2,
               "f_2(1270)": 2.3, "f_0(1370)": 3.6, "NonReson": 11.5, "chi_c0": 1.04}
B0_TABLE = pathlib.Path(__file__).resolve().parents[1] / "tables" / "b0.npz"
needs_b0_table = pytest.mark.skipif(not B0_TABLE.exists(), reason="b0 table not present")


def _babar(name):
    m, ph, mb, phb = BABAR_B0[name]
    return m * np.exp(1j * np.radians(ph)), mb * np.exp(1j * np.radians(phb))


def test_b0_model_follows_babar_2009():
    m = CHANNELS["b0"].model
    assert set(m.component_names) == set(BABAR_B0)          # no omega(782)
    assert m.mass_width_overrides["f_0(1370)"][:2] == (1.471, 0.097)      # their fX(1300)
    assert m.barrier_radii["Parent"] == 2.0 and m.barrier_radii["Kstar"] == 3.6
    assert m.lineshapes["K*+_0(1430)"] == "LASS" and m.lineshapes["f_0(980)"] == "Flatte"


@pytest.mark.parametrize("name", sorted(BABAR_B0))
def test_b0_cbar_over_c_is_babars(name):
    """(q/p) cbar / c is free of normalization and sign conventions: it must equal
    BaBar's cbar / c (whose cbar includes q/p) exactly."""
    ch = CHANNELS["b0"]
    c = ch.coefficients[name]
    c_b, cbar_b = _babar(name)
    assert ch.mixing.qp * c.c_bar / c.c == pytest.approx(cbar_b / c_b, rel=1e-12)


def test_b0_relative_phases_are_babars_up_to_the_helicity_signs():
    from tdalitz.channels import HELICITY_SIGN_B0
    ch = CHANNELS["b0"]
    ref = "f_0(980)"
    for name in BABAR_B0:
        flip = HELICITY_SIGN_B0[name] * HELICITY_SIGN_B0[ref]
        ours = ch.coefficients[name].c / ch.coefficients[ref].c
        theirs = _babar(name)[0] / _babar(ref)[0]
        assert ours / abs(ours) == pytest.approx(flip * theirs / abs(theirs), abs=1e-12)


def test_b0_fit_fractions_are_babars():
    ch = CHANNELS["b0"]
    w = {n: abs(c.c) ** 2 + abs(c.c_bar) ** 2 for n, c in ch.coefficients.items()}
    total, ff_total = sum(w.values()), sum(BABAR_B0_FF.values())
    for n in BABAR_B0:
        assert w[n] / total == pytest.approx(BABAR_B0_FF[n] / ff_total, rel=1e-12)


def test_b0_reproduces_babar_quasi_two_body_parameters():
    """BaBar Table V, Solution I, from the coefficients alone."""
    ch = CHANNELS["b0"]

    def c_cbar(n):
        c = ch.coefficients[n]
        return c.c, ch.mixing.qp * c.c_bar                   # BaBar's cbar includes q/p

    c, cb = c_cbar("K*+(892)")
    assert (abs(cb) ** 2 - abs(c) ** 2) / (abs(cb) ** 2 + abs(c) ** 2) == pytest.approx(-0.21, abs=0.01)
    c, cb = c_cbar("f_0(980)")
    assert (abs(c) ** 2 - abs(cb) ** 2) / (abs(c) ** 2 + abs(cb) ** 2) == pytest.approx(0.08, abs=0.01)
    assert np.degrees(0.5 * np.angle(c * np.conj(cb))) == pytest.approx(36.0, abs=1.5)
    c, cb = c_cbar("rho0(770)")
    assert np.degrees(0.5 * np.angle(c * np.conj(cb))) == pytest.approx(10.2, abs=1.5)


def _b0_helicity_cosines(table):
    """cos of BaBar's helicity angle, between the bachelor (p) and the resonance
    daughter q: pi+ for K*+ (KS pi+), pi- for the pi+ pi- resonances
    (arXiv:0905.3615, Sec. II).  Daughters are (KS, pi+, pi-)."""
    fs = table.final_state
    M = fs.parent
    m1, m2, m3 = fs.daughters
    s12, s13 = table.s12, table.s13
    s23 = fs.s23(s12, s13)

    def cos_between(s_pair, ma, mb, mc, s_ac):    # angle(a, c) in the (ab) frame
        r = np.sqrt(s_pair)
        ea, ec = (s_pair + ma**2 - mb**2) / (2 * r), (M**2 - s_pair - mc**2) / (2 * r)
        pa, pc = np.sqrt(np.maximum(ea**2 - ma**2, 0)), np.sqrt(np.maximum(ec**2 - mc**2, 0))
        return (ma**2 + mc**2 + 2 * ea * ec - s_ac) / (2 * pa * pc)

    return ({"K*+(892)": cos_between(s12, m2, m1, m3, s23),      # pi+ vs bachelor pi- in (KS pi+)
             "rho0(770)": cos_between(s23, m3, m2, m1, s13),     # pi- vs bachelor KS in (pi+ pi-)
             "f_2(1270)": cos_between(s23, m3, m2, m1, s13)},
            {"K*+(892)": s12, "rho0(770)": s23, "f_2(1270)": s23})


@needs_b0_table
def test_b0_helicity_signs_match_the_table():
    """Sign of each Laura++ P- and D-wave component relative to BaBar's angular
    factor (T_1 = -4 p.q, T_2 = (8/3)[3 (p.q)^2 - |p|^2 |q|^2]), measured below
    the resonance peak where the Breit-Wigner is nearly real and positive."""
    from tdalitz import AmplitudeTable
    from tdalitz.channels import HELICITY_SIGN_B0
    table = AmplitudeTable.load(B0_TABLE)
    cos, pair_s = _b0_helicity_cosines(table)
    below = {"K*+(892)": (0.70, 0.84), "rho0(770)": (0.50, 0.66), "f_2(1270)": (0.95, 1.12)}
    for name, (lo, hi) in below.items():
        m = np.sqrt(pair_s[name])
        x = cos[name]
        babar = -x if name != "f_2(1270)" else 3 * x ** 2 - 1
        sel = (m > lo) & (m < hi) & (np.abs(babar) > 0.3)
        r = table.amp[sel, table.index(name)] / babar[sel]
        assert np.all(np.sign(r.real) == HELICITY_SIGN_B0[name]), name
