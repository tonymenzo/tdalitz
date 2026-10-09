"""Validation of the Laura++ exporter against Laura++ itself.

Skipped automatically when ROOT or a Laura++ build is unavailable, so the rest
of the suite stays dependency-free.  Point ``TDALITZ_LAURA_LIB`` at the compiled
``libLaura++`` to enable.
"""

import os

import numpy as np
import pytest

from tdalitz import Coefficient
from tdalitz.exporters.laura import LauraModel, export

LIB = os.environ.get("TDALITZ_LAURA_LIB")
pytestmark = pytest.mark.skipif(LIB is None, reason="TDALITZ_LAURA_LIB not set")

RESONANCES = [("K*-(892)", 2), ("rho0(770)", 1), ("K*-_0(1430)", 2)]
COEFFS = {
    "K*-(892)":    Coefficient(0.7550, 2.331),
    "rho0(770)":   Coefficient(0.4593, 0.000),
    "K*-_0(1430)": Coefficient(0.2470, 1.497),
}


def _model():
    return LauraModel(
        parent="D0", daughters=("K_S0", "pi+", "pi-"),
        resonances=RESONANCES,
        barrier_radii={"Parent": 5.0, "Kstar": 1.5, "Light": 1.5},
        library=LIB,
    )


def test_sum_of_components_reproduces_laura_total():
    """sum_r c_r F_r from the table must equal Laura++'s own total amplitude
    when Laura++ is initialised with the same coefficients."""
    import ROOT

    table = export(_model(), n_points=2000, seed=3, progress=False)
    ours, _ = table.total(COEFFS)

    # Build the same model in Laura++ with the full coefficient set.
    ROOT.gSystem.Load(LIB)
    daughters = ROOT.LauDaughters("D0", "K_S0", "pi+", "pi-", False)
    vetoes = ROOT.LauVetoes()          # must outlive eff: PyROOT holds a raw pointer
    eff = ROOT.LauEffModel(daughters, vetoes)
    maker = ROOT.LauResonanceMaker.get()
    maker.setDefaultBWRadius(ROOT.LauBlattWeisskopfFactor.Parent, 5.0)
    maker.setDefaultBWRadius(ROOT.LauBlattWeisskopfFactor.Kstar, 1.5)
    maker.setDefaultBWRadius(ROOT.LauBlattWeisskopfFactor.Light, 1.5)
    dyn = ROOT.LauIsobarDynamics(daughters, eff)
    for name, bachelor in RESONANCES:
        dyn.addResonance(name, bachelor, ROOT.LauAbsResonance.RelBW)
    vec = ROOT.std.vector("LauComplex")()
    for name, _ in RESONANCES:
        c = COEFFS[name].c
        vec.push_back(ROOT.LauComplex(c.real, c.imag))
    dyn.initialise(vec)

    s23 = table.final_state.s23(table.s12, table.s13)
    reference = np.empty(len(table), dtype=complex)
    for i in range(len(table)):
        dyn.calcLikelihoodInfo(float(table.s13[i]), float(s23[i]))
        a = dyn.getEvtDPAmp()
        reference[i] = complex(a.re(), a.im())

    assert np.allclose(ours, reference, rtol=1e-9, atol=1e-12)


def test_amp_bar_equals_amp_at_mirrored_points():
    """Fbar_r(s12, s13) must be exactly F_r evaluated at the CP-mirrored point
    (s13, s12).  Export the same pool twice -- once mirrored -- and compare."""
    fs_points = export(_model(), n_points=300, seed=5,
                       progress=False)
    s12, s13 = fs_points.s12, fs_points.s13
    direct = export(_model(), points=(s12, s13), progress=False)
    mirrored = export(_model(), points=(s13, s12), progress=False)
    assert np.allclose(direct.amp_bar, mirrored.amp, rtol=1e-12, atol=1e-14)
    assert np.allclose(direct.amp, mirrored.amp_bar, rtol=1e-12, atol=1e-14)


def test_mirrored_export_uses_a_mirror_symmetric_pool():
    """With mirror=True the pool is closed under s12 <-> s13, so a CP-conserving
    model is CP symmetric on the pool itself: Fbar at a point equals F at its
    partner, exactly, and no pool fluctuation can fake a CP asymmetry."""
    table = export(_model(), n_points=400, seed=6, progress=False)
    half = len(table) // 2
    assert np.array_equal(table.s12[half:], table.s13[:half])
    assert np.array_equal(table.amp_bar[half:], table.amp[:half])
    assert np.array_equal(table.amp_bar[:half], table.amp[half:])


def test_export_uses_supplied_points_verbatim():
    s12 = np.array([1.50, 0.90, 1.30])       # verified inside the D0 -> KS pi pi plot
    s13 = np.array([0.90, 1.20, 0.80])
    table = export(_model(), points=(s12, s13), progress=False)
    assert np.array_equal(table.s12, s12) and np.array_equal(table.s13, s13)


def test_supplied_points_outside_the_dalitz_plot_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        export(_model(), points=(np.array([9.0]), np.array([9.0])), progress=False)


def test_export_without_mirror_requires_a_conjugate_model():
    """Without the CP-mirror symmetry, Fbar must come from a second model --
    silently reusing F would set Abar = A, which is wrong physics."""
    with pytest.raises(ValueError, match="conjugate_model"):
        export(_model(), n_points=10, mirror=False, progress=False)


def test_conjugate_model_supplies_amp_bar():
    """With mirror=False, amp_bar is the conjugate model evaluated at the same
    points; using the model as its own conjugate must reproduce amp exactly."""
    table = export(_model(), n_points=200, seed=2, mirror=False,
                   conjugate_model=_model(), progress=False)
    assert np.allclose(table.amp_bar, table.amp, rtol=1e-12, atol=1e-14)


def test_export_takes_kinematics_from_laura():
    """Laura++ evaluates amplitudes with its own particle masses, so the table's
    kinematics must use exactly those; otherwise each tabulated point is
    shifted in the third invariant and the CP mirror is inexact."""
    import ROOT
    ROOT.gSystem.Load(LIB)
    d = ROOT.LauDaughters("D0", "K_S0", "pi+", "pi-", False)
    table = export(_model(), n_points=20, seed=1, progress=False)
    assert table.final_state.parent == d.getMassParent()
    assert table.final_state.daughters == (d.getMassDaug1(), d.getMassDaug2(),
                                           d.getMassDaug3())
