"""Export per-component amplitudes from a Laura++ isobar model.

This is the only module in the package that depends on Laura++ and ROOT.  It
drives Laura++ through PyROOT to evaluate each isobar component on a pool of
phase-space points and writes the result as a plain
:class:`~tdalitz.amplitude.AmplitudeTable`, after which nothing downstream needs
ROOT at all.

Components are evaluated one at a time: the model is initialised with a unit
coefficient on the component of interest and zero on the rest, so the returned
column is the bare ``F_r`` including Laura++'s lineshape, Blatt-Weisskopf
barriers, angular factor and its unit-integral (``fNorm``) normalization.  Free
coefficients are applied later, in NumPy.

The CP-conjugate amplitudes ``Fbar_r`` are obtained by evaluating the same
components at the CP-mirrored Dalitz point.  This is correct for a final state
that CP maps onto itself up to the exchange of two daughters -- e.g.
``KS pi+ pi-``, where CP exchanges the pions and hence ``s12 <-> s13``.  For a
final state without that symmetry, pass ``mirror=False`` together with a
``conjugate_model`` describing the CP-conjugate decay.

Example
-------
>>> from tdalitz.exporters.laura import LauraModel, export
>>> model = LauraModel(
...     parent="D0", daughters=("K_S0", "pi+", "pi-"),
...     resonances=[("K*-(892)", 2), ("rho0(770)", 1), ("K*-_0(1430)", 2)],
...     barrier_radii={"Parent": 5.0, "Kstar": 1.5, "Light": 1.5},
... )
>>> table = export(model, n_points=1_000_000, seed=1)   # doctest: +SKIP
>>> table.save("d0_kspipi.npz")                          # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..amplitude import AmplitudeTable
from ..kinematics import FinalState

__all__ = ["LauraModel", "export"]


@dataclass
class LauraModel:
    """Specification of a Laura++ isobar model.

    Parameters
    ----------
    parent:
        Laura++ name of the parent, e.g. ``"D0"``.
    daughters:
        Laura++ names of the three daughters, in the order that fixes the
        Dalitz convention (Laura++ stores ``m13Sq`` and ``m23Sq``).
    resonances:
        Sequence of ``(name, bachelor_index)`` pairs.  The bachelor index is
        the Laura++ ``resPairAmpInt``: the daughter *not* in the resonance.
    lineshapes:
        Optional ``{name: lineshape}`` overrides; the default is ``"RelBW"``.
        Any Laura++ ``LauAbsResonance`` shape name is accepted, e.g. ``"GS"``,
        ``"Flatte"``, ``"LASS_BW"``, ``"KMatrix"``.
    barrier_radii:
        Optional ``{category: radius}`` in GeV^-1, where category is a
        ``LauBlattWeisskopfFactor`` name such as ``"Parent"`` or ``"Kstar"``.
    mass_width_overrides:
        Optional ``{name: (mass, width, spin)}`` to pin pole parameters.
    library:
        Path to the compiled ``libLaura++`` shared library.
    """

    parent: str
    daughters: tuple
    resonances: list
    lineshapes: dict = field(default_factory=dict)
    barrier_radii: dict = field(default_factory=dict)
    mass_width_overrides: dict = field(default_factory=dict)
    library: str = "libLaura++"

    @property
    def component_names(self) -> tuple:
        return tuple(name for name, _ in self.resonances)


def _evaluate(dynamics, m13_sq, m23_sq) -> np.ndarray:
    """Evaluate the currently active component at each Dalitz point."""
    out = np.empty(len(m13_sq), dtype=complex)
    for i in range(len(m13_sq)):
        dynamics.calcLikelihoodInfo(float(m13_sq[i]), float(m23_sq[i]))
        amp = dynamics.getEvtDPAmp()
        out[i] = complex(amp.re(), amp.im())
    return out


def _laura_masses(model: LauraModel) -> tuple:
    """(parent, m1, m2, m3) exactly as Laura++ uses them (its particle database).

    The table's kinematics must coincide with Laura++'s: it reconstructs the
    third invariant from these masses, so any other choice shifts every
    tabulated point and breaks the exactness of the CP mirror.
    """
    import ROOT  # imported lazily so the package works without ROOT

    ROOT.gSystem.Load(model.library)
    d = ROOT.LauDaughters(model.parent, *model.daughters, False)
    return (d.getMassParent(), d.getMassDaug1(), d.getMassDaug2(), d.getMassDaug3())


def _evaluate_components(model: LauraModel, point_sets, progress: bool) -> list:
    """Evaluate every component of ``model`` on each set of Dalitz points.

    ``point_sets`` is a sequence of ``(m13_sq, m23_sq)`` array pairs, in Laura++'s
    coordinates.  Returns ``(arrays, poles, radii)``: one ``(N, R)`` complex
    array per set; ``{name: [mass, width, spin]}`` as resolved by Laura++
    (database value or override) for every component that has a pole; and
    ``{name: [resonance radius, parent radius]}`` of the Blatt-Weisskopf
    factors Laura++ applies, in GeV^-1.

    The Laura++ model is built once and re-initialised with a unit coefficient
    on one component at a time; the per-component ``fNorm`` normalization is
    independent of the coefficients, so each pass returns that component's bare
    ``F_r``.  Every ROOT object is kept alive in this scope: ``LauIsobarDynamics``
    holds raw pointers to the daughters, vetoes and efficiency model, so letting
    Python collect any of them early leaves it with dangling references.
    """
    import ROOT  # imported lazily so the package works without ROOT

    ROOT.gSystem.Load(model.library)

    daughters = ROOT.LauDaughters(model.parent, *model.daughters, False)
    vetoes = ROOT.LauVetoes()
    efficiency = ROOT.LauEffModel(daughters, vetoes)

    maker = ROOT.LauResonanceMaker.get()
    for category, radius in model.barrier_radii.items():
        maker.setDefaultBWRadius(getattr(ROOT.LauBlattWeisskopfFactor, category), radius)

    dynamics = ROOT.LauIsobarDynamics(daughters, efficiency)
    poles, radii = {}, {}
    for name, bachelor in model.resonances:
        shape = model.lineshapes.get(name, "RelBW")
        resonance = dynamics.addResonance(
            name, bachelor, getattr(ROOT.LauAbsResonance, shape)
        )
        if name in model.mass_width_overrides:
            mass, width, spin = model.mass_width_overrides[name]
            resonance.changeResonance(mass, width, spin)
        if resonance.getMass() > 0.0:
            poles[name] = [float(resonance.getMass()), float(resonance.getWidth()),
                           int(resonance.getSpin())]
        radii[name] = [float(resonance.getResRadius()), float(resonance.getParRadius())]

    n_comp = len(model.resonances)
    outputs = [np.empty((len(m13), n_comp), dtype=complex) for m13, _ in point_sets]
    for col, (name, _) in enumerate(model.resonances):
        if progress:
            print(f"  [{col + 1}/{n_comp}] evaluating {name}", flush=True)
        unit = ROOT.std.vector("LauComplex")()
        for i in range(n_comp):
            unit.push_back(ROOT.LauComplex(1.0 if i == col else 0.0, 0.0))
        dynamics.initialise(unit)
        for out, (m13, m23) in zip(outputs, point_sets):
            out[:, col] = _evaluate(dynamics, m13, m23)
    return outputs, poles, radii


def export(model: LauraModel, n_points: int = 1_000_000, seed: int = 0,
           points: tuple = None, mirror: bool = True,
           conjugate_model: LauraModel = None,
           progress: bool = True) -> AmplitudeTable:
    """Tabulate every component of a Laura++ model on a phase-space pool.

    Parameters
    ----------
    model:
        The Laura++ model specification.
    n_points:
        Size of the phase-space pool drawn when ``points`` is not given.  Make
        this much larger than any sample you intend to generate.  With
        ``mirror=True`` the pool is mirror symmetric (see
        :meth:`~tdalitz.kinematics.FinalState.sample_phase_space`), so an odd
        ``n_points`` is rounded up by one.
    seed:
        Seed for the phase-space draw, so tables are reproducible.
    points:
        Optional ``(s12, s13)`` arrays to evaluate at instead of a random pool.
        Every point must lie inside the physical Dalitz region.  Note that a
        table used for generation must carry *flat* phase-space weight, so
        supply uniformly distributed points if you intend to generate from it.
    mirror:
        Build ``Fbar_r`` by evaluating at the CP-mirrored point
        (``s12 <-> s13``).  Correct when CP exchanges daughters 2 and 3.
    conjugate_model:
        Required when ``mirror=False``: a model of the CP-conjugate decay,
        with the same component names, whose daughters occupy the same slots
        so that it is evaluated at the same ``(s12, s13)``.
    """
    if mirror and conjugate_model is not None:
        raise ValueError("pass either mirror=True or a conjugate_model, not both")
    if not mirror:
        if conjugate_model is None:
            raise ValueError(
                "mirror=False requires a conjugate_model supplying Fbar_r; "
                "reusing F_r would set Abar = A"
            )
        if conjugate_model.component_names != model.component_names:
            raise ValueError("conjugate_model must have the same component names")

    masses = _laura_masses(model)
    final_state = FinalState(
        parent=masses[0], daughters=tuple(masses[1:]), names=tuple(model.daughters)
    )
    if points is None:
        if mirror:
            n_points += n_points % 2
        s12, s13 = final_state.sample_phase_space(
            n_points, np.random.default_rng(seed), symmetric=mirror)
        origin = f"n_points={n_points}, seed={seed}, symmetric_pool={mirror}"
    else:
        s12, s13 = (np.asarray(v, dtype=float) for v in points)
        if s12.shape != s13.shape or s12.ndim != 1:
            raise ValueError("points must be two 1-D arrays of equal length")
        if not final_state.inside(s12, s13).all():
            raise ValueError("some supplied points lie outside the physical Dalitz region")
        origin = f"{len(s12)} supplied points"
    s23 = final_state.s23(s12, s13)

    # Laura++ addresses the plot by (m13Sq, m23Sq).  CP exchanges daughters 2
    # and 3, i.e. s12 <-> s13, and leaves s23 invariant, so the mirrored point
    # is (m13Sq, m23Sq) = (s12, s23).
    if mirror:
        (amp, amp_bar), poles, radii = _evaluate_components(
            model, [(s13, s23), (s12, s23)], progress)
    else:
        (amp,), poles, radii = _evaluate_components(model, [(s13, s23)], progress)
        (amp_bar,), _, _ = _evaluate_components(conjugate_model, [(s13, s23)], progress)

    return AmplitudeTable(
        final_state=final_state,
        s12=s12, s13=s13,
        components=model.component_names,
        amp=amp, amp_bar=amp_bar,
        provenance=(
            f"Laura++ export: parent={model.parent}, daughters={model.daughters}, "
            f"{origin}, mirror={mirror}"
        ),
        metadata={"resonances": [list(r) for r in model.resonances],
                  "poles": poles,
                  "radii": radii,
                  "lineshapes": dict(model.lineshapes),
                  "barrier_radii": dict(model.barrier_radii)},
    )
