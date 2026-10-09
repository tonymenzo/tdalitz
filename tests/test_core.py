"""Core tests.  These exercise the whole pipeline with a synthetic amplitude
table and therefore run without ROOT or Laura++ -- which is exactly the
decoupling the package is built around."""

import numpy as np
import pytest

from tdalitz import (
    AmplitudeTable, Coefficient, Events, FinalState, MixingParameters,
    apply_mistag, apply_time_resolution, dilution_factor,
    generate_tagged, tagged_rate,
)

M_D0, M_KS, M_PI = 1.86484, 0.497611, 0.13957039


def toy_table(n=60_000, seed=7):
    """A two-component synthetic table: Breit-Wigners in the (13) and (23) pairs.

    ``amp_bar`` is built by evaluating the same components at the CP-mirrored
    point, mimicking what the Laura++ exporter does for a KS pi+ pi- final state.
    """
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI), ("K_S0", "pi+", "pi-"))
    rng = np.random.default_rng(seed)
    s12, s13 = fs.sample_phase_space(n, rng)

    def bw(s, m, w):
        return 1.0 / (m * m - s - 1j * m * w)

    def components(a, b):
        s23 = fs.s23(a, b)
        return np.column_stack([bw(b, 0.89167, 0.0514), bw(s23, 0.77526, 0.1474)])

    return AmplitudeTable(
        final_state=fs, s12=s12, s13=s13,
        components=("Kstar", "rho"),
        amp=components(s12, s13),
        amp_bar=components(s13, s12),          # CP mirror: s12 <-> s13
        provenance="synthetic test model",
    )


COEFFS = {"Kstar": Coefficient(1.0, 0.0), "rho": Coefficient(0.6, 1.1)}


# --------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------
def test_phase_space_points_are_inside():
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(5000, np.random.default_rng(0))
    assert fs.inside(s12, s13).all()


def test_square_dalitz_round_trip():
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(5000, np.random.default_rng(1))
    mp, tp = fs.to_square(s12, s13)
    assert mp.min() >= 0.0 and mp.max() <= 1.0
    assert tp.min() >= 0.0 and tp.max() <= 1.0
    back12, back13 = fs.from_square(mp, tp)
    assert np.allclose(back12, s12, atol=1e-8)
    assert np.allclose(back13, s13, atol=1e-8)


# --------------------------------------------------------------------------
# amplitude table
# --------------------------------------------------------------------------
def test_total_is_linear_in_coefficients():
    table = toy_table(5000)
    a1, _ = table.total({"Kstar": Coefficient(1.0)})
    a2, _ = table.total({"Kstar": Coefficient(2.0)})
    assert np.allclose(a2, 2.0 * a1)


def test_fit_fractions_are_positive_and_bounded():
    table = toy_table(20_000)
    ff = table.fit_fractions(COEFFS)
    assert set(ff) == {"Kstar", "rho"}
    assert all(0.0 < v < 1.5 for v in ff.values())


def test_save_load_round_trip(tmp_path):
    table = toy_table(3000)
    table.save(tmp_path / "t.npz")
    back = AmplitudeTable.load(tmp_path / "t.npz")
    assert back.components == table.components
    assert np.allclose(back.amp, table.amp)
    assert np.allclose(back.s12, table.s12)
    assert back.final_state.parent == pytest.approx(table.final_state.parent)


def test_unknown_component_raises():
    with pytest.raises(KeyError):
        toy_table(1000).total({"not_a_resonance": Coefficient(1.0)})


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------
def test_cp_conservation_relates_the_two_tags():
    """With q/p = 1 and no weak phases, the antimeson rate equals the meson
    rate with A and Abar exchanged -- i.e. evaluated at the CP-mirrored point."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=500) + 1j * rng.normal(size=500)
    a_bar = rng.normal(size=500) + 1j * rng.normal(size=500)
    mix = MixingParameters(x=0.77, y=0.0, qp=1.0 + 0j)
    tau = rng.exponential(1.0, 500)
    assert np.allclose(tagged_rate(a, a_bar, mix, tau, +1),
                       tagged_rate(a_bar, a, mix, tau, -1))


def test_mixing_phase_breaks_the_relation():
    rng = np.random.default_rng(4)
    a = rng.normal(size=400) + 1j * rng.normal(size=400)
    a_bar = rng.normal(size=400) + 1j * rng.normal(size=400)
    mix = MixingParameters(x=0.77, y=0.0, qp=np.exp(-0.7j))
    tau = rng.exponential(1.0, 400)
    assert not np.allclose(tagged_rate(a, a_bar, mix, tau, +1),
                           tagged_rate(a_bar, a, mix, tau, -1))


def test_g_factors_reduce_to_trig_when_dgamma_zero():
    mix = MixingParameters(x=1.3, y=0.0)
    tau = np.linspace(0.0, 5.0, 50)
    from tdalitz import g_factors
    g_p, g_m = g_factors(mix, tau)
    assert np.allclose(g_p.real, np.cos(1.3 * tau / 2.0))
    assert np.allclose(g_m.imag, np.sin(1.3 * tau / 2.0))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def test_generated_events_are_physical():
    table = toy_table()
    mix = MixingParameters(x=0.77, y=0.0, qp=np.exp(-0.7j))
    ev = generate_tagged(table, COEFFS, mix, +1, 4000, np.random.default_rng(11))
    assert len(ev) == 4000
    assert table.final_state.inside(ev.s12, ev.s13).all()
    assert (ev.tag == +1).all()
    # tau is drawn from Exp(1) and reweighted by a bounded rate
    assert 0.5 < ev.tau.mean() < 1.6
    assert (ev.tau >= 0.0).all()


def test_generation_is_reproducible():
    table = toy_table()
    mix = MixingParameters(x=0.77)
    kw = dict(table=table, coefficients=COEFFS, mixing=mix, tag=+1, n_events=800)
    a = generate_tagged(**kw, rng=np.random.default_rng(5))
    b = generate_tagged(**kw, rng=np.random.default_rng(5))
    assert np.array_equal(a.s12, b.s12) and np.array_equal(a.tau, b.tau)


def test_cp_conserving_model_gives_tag_symmetric_sample():
    """With q/p = 1 and no weak phases the mirrored antimeson sample matches the
    meson sample, so the tag carries no information."""
    table = toy_table(120_000)
    mix = MixingParameters(x=0.77, y=0.0, qp=1.0 + 0j)
    rng = np.random.default_rng(21)
    plus = generate_tagged(table, COEFFS, mix, +1, 20_000, rng)
    minus = generate_tagged(table, COEFFS, mix, -1, 20_000, rng)
    # CP-align the antimeson sample by mirroring its Dalitz coordinates
    mirrored_mean = minus.s13.mean()
    assert plus.s12.mean() == pytest.approx(mirrored_mean, rel=0.02)


# --------------------------------------------------------------------------
# detector effects
# --------------------------------------------------------------------------
def test_resolution_damps_the_oscillation_as_predicted():
    """Smearing by sigma must damp <cos(x tau)> by exp(-x^2 sigma^2 / 2)."""
    # x is chosen so that <cos(x tau)> is O(0.2) for exponential times; at
    # larger x the moment is tiny and the ratio estimator becomes too noisy to
    # test the damping at the percent level.
    rng = np.random.default_rng(31)
    x, sigma = 2.0, 0.4
    tau = rng.exponential(1.0, 400_000)
    ev = Events(np.zeros_like(tau), np.zeros_like(tau), tau, np.ones_like(tau, dtype=np.int8))
    smeared = apply_time_resolution(ev, sigma, rng)
    ratio = np.mean(np.cos(x * smeared.tau)) / np.mean(np.cos(x * tau))
    assert ratio == pytest.approx(dilution_factor(x, sigma), rel=0.03)


def test_mistag_flips_the_expected_fraction():
    rng = np.random.default_rng(41)
    n = 200_000
    ev = Events(np.zeros(n), np.zeros(n), np.ones(n), np.ones(n, dtype=np.int8))
    out = apply_mistag(ev, 0.2, rng)
    assert np.mean(out.tag == -1) == pytest.approx(0.2, abs=0.005)


def test_mistag_bounds_are_enforced():
    ev = Events(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3, dtype=np.int8))
    with pytest.raises(ValueError):
        apply_mistag(ev, 0.7, np.random.default_rng(0))


# --------------------------------------------------------------------------
# exactness of generation: time distribution vs the analytic density
# --------------------------------------------------------------------------
def constant_table(a, a_bar, n=1000):
    """Pool whose amplitude is the same constant at every Dalitz point, so the
    generated decay-time density is known exactly."""
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(n, np.random.default_rng(0))
    col = np.ones((n, 1), dtype=complex)
    return AmplitudeTable(fs, s12, s13, ("const",), a * col, a_bar * col)


def exact_moments(a, a_bar, mix, tag, x):
    """<tau>, <cos x tau>, <sin x tau> under exp(-tau) * tagged_rate, by quadrature."""
    tau = np.linspace(0.0, 45.0, 900_001)
    amp, amp_bar = np.full_like(tau, a, dtype=complex), np.full_like(tau, a_bar, dtype=complex)
    w = np.exp(-tau) * tagged_rate(amp, amp_bar, mix, tau, tag)
    norm = np.trapezoid(w, tau)
    return (np.trapezoid(w * tau, tau) / norm,
            np.trapezoid(w * np.cos(x * tau), tau) / norm,
            np.trapezoid(w * np.sin(x * tau), tau) / norm)


@pytest.mark.parametrize("x, y", [(0.77, 0.0), (26.9, 0.064), (0.0041, 0.0064)])
@pytest.mark.parametrize("tag", [+1, -1])
def test_time_distribution_matches_exact_density(x, y, tag):
    """The generated tau distribution must reproduce the analytic tagged density,
    including in the fast-oscillation (x ~ 27) and Delta-Gamma regimes."""
    a, a_bar = 1.0 + 0j, 0.8 * np.exp(1.0j)
    mix = MixingParameters(x=x, y=y, qp=np.exp(-0.4j))
    n = 200_000
    ev = generate_tagged(constant_table(a, a_bar), {"const": Coefficient(1.0)},
                         mix, tag, n, np.random.default_rng(99))
    m_tau, m_cos, m_sin = exact_moments(a, a_bar, mix, tag, x)
    tol = 5.0 / np.sqrt(n)                     # 5 sigma; each moment has std <= ~1
    assert ev.tau.mean() == pytest.approx(m_tau, abs=5.0 * tol)
    assert np.mean(np.cos(x * ev.tau)) == pytest.approx(m_cos, abs=tol)
    assert np.mean(np.sin(x * ev.tau)) == pytest.approx(m_sin, abs=tol)


def test_generation_never_clips_the_acceptance():
    """The rejection ceiling must bound the rate everywhere, or the sample is
    silently biased.  Probe a rate with a narrow peak in tau that a coarse
    tau-scan misses: huge x makes the peaks arbitrarily narrow in tau."""
    from tdalitz.generate import rate_ceiling
    a, a_bar = 1.0 + 0j, 1.0 * np.exp(0.3j)
    mix = MixingParameters(x=5000.0, y=0.2, qp=np.exp(-1.1j))
    table = constant_table(a, a_bar, n=50)
    amp, amp_bar = table.total({"const": Coefficient(1.0)})
    for tag in (+1, -1):
        ceiling = rate_ceiling(amp, amp_bar, mix, tag)
        tau = np.linspace(0.0, 60.0, 2_000_001)
        rate = tagged_rate(np.full_like(tau, a, dtype=complex),
                           np.full_like(tau, a_bar, dtype=complex), mix, tau, tag)
        assert np.all(rate * np.exp(-abs(mix.y) * tau) <= ceiling * (1 + 1e-12))


# --------------------------------------------------------------------------
# serialization safety
# --------------------------------------------------------------------------
def test_saved_table_loads_without_pickle(tmp_path):
    """Tables are shared between people, so reading one must not execute code:
    every array in the file must be loadable with allow_pickle=False."""
    path = toy_table(500).save(tmp_path / "t.npz")
    with np.load(path, allow_pickle=False) as d:
        for key in d.files:
            _ = d[key]
    assert AmplitudeTable.load(path).components == ("Kstar", "rho")


# --------------------------------------------------------------------------
# mirror-symmetric pools
# --------------------------------------------------------------------------
def test_symmetric_pool_is_closed_under_the_exchange_of_daughters_2_and_3():
    """Second half of the pool is the first half with s12 <-> s13, exactly."""
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(1000, np.random.default_rng(3), symmetric=True)
    assert len(s12) == 1000
    assert np.array_equal(s12[500:], s13[:500]) and np.array_equal(s13[500:], s12[:500])
    assert fs.inside(s12, s13).all()


def test_symmetric_pool_needs_equal_masses_and_an_even_size():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="equal"):
        FinalState(M_D0, (M_PI, M_KS, M_PI)).sample_phase_space(10, rng, symmetric=True)
    with pytest.raises(ValueError, match="even"):
        FinalState(M_D0, (M_KS, M_PI, M_PI)).sample_phase_space(11, rng, symmetric=True)
