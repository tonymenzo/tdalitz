"""Laura++-level checks of the paper channels (skipped without ROOT/Laura++)."""

import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tdalitz.channels import CHANNELS  # noqa: E402
from laura_reference import laura_total  # noqa: E402
from tdalitz.exporters.laura import LauraModel, export  # noqa: E402

LIB = os.environ.get("TDALITZ_LAURA_LIB")
pytestmark = pytest.mark.skipif(LIB is None, reason="TDALITZ_LAURA_LIB not set")


def _with_lib(model: LauraModel) -> LauraModel:
    return LauraModel(**{**model.__dict__, "library": LIB})


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_table_reproduces_laura_total_amplitude(key):
    ch = CHANNELS[key]
    model = _with_lib(ch.model)
    table = export(model, n_points=400, seed=11, progress=False)
    ours, _ = table.total(ch.coefficients)
    ref = laura_total(model, {n: c.c for n, c in ch.coefficients.items()},
                      table.s13, table.final_state.s23(table.s12, table.s13), LIB)
    assert np.allclose(ours, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_table_reproduces_laura_conjugate_amplitude(key):
    """Abar = sum_r cbar_r F_r evaluated at the pion-exchanged point."""
    ch = CHANNELS[key]
    model = _with_lib(ch.model)
    table = export(model, n_points=400, seed=13, progress=False)
    _, ours = table.total(ch.coefficients)
    ref = laura_total(model, {n: c.c_bar for n, c in ch.coefficients.items()},
                      table.s12, table.final_state.s23(table.s12, table.s13), LIB)
    assert np.allclose(ours, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_pipi_components_have_definite_cp(key):
    """Exchanging the pions leaves s(pi pi) fixed and flips the helicity angle,
    so a pi pi resonance of spin L obeys Fbar_r = (-1)^L F_r exactly.  This is
    what makes f0 KS CP-even and rho0 KS CP-odd, and it only holds if the
    table's kinematics coincide with Laura++'s."""
    ch = CHANNELS[key]
    table = export(_with_lib(ch.model), n_points=400, seed=12, progress=False)
    spins = {name: spin for name, (_, _, spin) in table.metadata["poles"].items()}
    spins["NonReson"] = 0
    pipi = [name for name, bachelor in ch.model.resonances if bachelor in (0, 1)]
    for name in pipi:
        col = table.index(name)
        sign = (-1) ** spins[name]
        assert np.allclose(table.amp_bar[:, col], sign * table.amp[:, col],
                           rtol=1e-9, atol=1e-12), name


def test_export_records_resolved_pole_parameters():
    """The table must carry the mass, width and spin Laura++ actually used, so
    documentation can quote them from the source of truth."""
    ch = CHANNELS["b0"]
    table = export(_with_lib(ch.model), n_points=50, seed=1,
                   progress=False)
    poles = table.metadata["poles"]
    assert set(poles) == set(ch.model.component_names) - {"NonReson"}
    mass, width, spin = poles["f_0(1370)"]            # overridden to play fX(1300)
    assert (mass, width, spin) == pytest.approx(ch.model.mass_width_overrides["f_0(1370)"])
    assert poles["rho0(770)"][2] == 1 and poles["f_2(1270)"][2] == 2


@pytest.mark.parametrize("key", ["d0", "b0", "bs"])
def test_export_records_the_barrier_radii_laura_applied(key):
    """Laura++'s database carries resonance-specific radii (e.g. 5.3 GeV^-1 for
    the rho); the category defaults we set must be what it actually applies.
    For spin 0 the Blatt-Weisskopf factor is identically 1 (and LASS/Flatte
    build none, reporting -1), so the radius is irrelevant there."""
    ch = CHANNELS[key]
    table = export(_with_lib(ch.model), n_points=50, seed=1, progress=False)
    radii = table.metadata["radii"]
    spins = {name: spin for name, (_, _, spin) in table.metadata["poles"].items()}
    assert set(radii) == set(ch.model.component_names)
    for name, (res_radius, par_radius) in radii.items():
        if spins.get(name, 0) == 0:
            continue
        cat = "Charmonium" if name == "chi_c0" else ("Kstar" if name.startswith("K*") else "Light")
        assert res_radius == pytest.approx(ch.model.barrier_radii[cat]), name
        assert par_radius == pytest.approx(ch.model.barrier_radii["Parent"]), name


#: CP conjugates of the K pi resonances, with their bachelor in (KS, pi+, pi-).
_CONJ = {"K*+(892)": ("K*-(892)", 2), "K*-(892)": ("K*+(892)", 3),
         "K*+_0(1430)": ("K*-_0(1430)", 2), "K*-_2(1430)": ("K*+_2(1430)", 3)}


@pytest.mark.parametrize("key, parent_bar", [("b0", "B0_bar"), ("bs", "B_s0_bar")])
def test_mirror_versus_laura_antiparticle_model(key, parent_bar):
    """Laura++'s time-dependent code builds Abar from a separate antiparticle
    model evaluated at the same point.  Relative to our mirrored Fbar_r this is
    identical for pi pi waves and differs by (-1)^L for K pi waves (Laura++'s
    helicity angles for the (12) and (13) pairs are measured from different
    daughters), up to Laura++'s numerical normalization (< 1%)."""
    from dataclasses import replace
    ch = CHANNELS[key]
    model = _with_lib(ch.model)
    ours = export(model, n_points=400, seed=14, progress=False)
    res = [_CONJ.get(n, (n, b)) for n, b in model.resonances]
    conj = replace(model, parent=parent_bar, resonances=res,
                   lineshapes={_CONJ.get(n, (n,))[0]: s for n, s in model.lineshapes.items()},
                   mass_width_overrides={_CONJ.get(n, (n,))[0]: v
                                         for n, v in model.mass_width_overrides.items()})
    theirs = export(conj, points=(ours.s12, ours.s13), mirror=False, conjugate_model=conj,
                    progress=False).amp
    spins = {n: s for n, (_, _, s) in ours.metadata["poles"].items()}
    for col, (name, bachelor) in enumerate(model.resonances):
        ratio = ours.amp_bar[:, col] / theirs[:, col]
        if bachelor in (0, 1):                                  # pi pi and nonresonant
            assert np.allclose(ratio, 1.0, rtol=1e-9), name
        else:
            sign = (-1) ** spins[name]
            assert np.allclose(ratio, ratio[0], rtol=1e-9), name   # a constant ...
            assert ratio[0].real * sign == pytest.approx(1.0, abs=0.01), name   # ... (-1)^L
