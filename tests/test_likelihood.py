"""Tests for tdalitz.likelihood: exact per-event densities, expected KL
divergences, Asimov significance and the information per event.

The decisive check ties the information to an independent closed form: for a
constant amplitude with |lambda| = 1 the tagged time densities are
exp(-tau) [1 -+ S sin(x tau)], whose per-event information for S is
V_x = 2x^2/(1+4x^2) - x^2/(1+x^2)^2 (notes/dalitz_movers_dist_t.tex, the
time-axis likelihood Z = R sqrt(N V_x))."""

import numpy as np
import pytest

from tdalitz import (AmplitudeTable, Coefficient, FinalState, MixingParameters,
                     apply_mistag, apply_time_resolution, generate_tagged, tagged_rate)

M_D0, M_KS, M_PI = 1.86484, 0.497611, 0.13957039


def toy_table(n=4000, seed=7):
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(n, np.random.default_rng(seed), symmetric=True)

    def comps(a, b):
        bw = lambda s, m, w: 1.0 / (m * m - s - 1j * m * w)
        return np.column_stack([bw(b, 0.89167, 0.0514), bw(fs.s23(a, b), 0.77526, 0.1474)])

    return AmplitudeTable(fs, s12, s13, ("Kstar", "rho"), comps(s12, s13), comps(s13, s12))


def constant_table(a=1.0 + 0j, a_bar=1.0 + 0j, n=200):
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(n, np.random.default_rng(0))
    col = np.ones((n, 1), dtype=complex)
    return AmplitudeTable(fs, s12, s13, ("c",), a * col, a_bar * col)


COEFFS = {"Kstar": Coefficient(1.0, 0.0), "rho": Coefficient(0.6, 1.1)}


def beta_family(x, y=0.0):
    """theta -> model with q/p = exp(-2 i theta); theta = 0 is CP conserving."""
    return lambda theta: (COEFFS, MixingParameters(x=x, y=y, qp=np.exp(-2j * theta)))


# ------------------------------------------------------------ pool index
def test_generated_events_carry_their_pool_index():
    t = toy_table()
    ev = generate_tagged(t, COEFFS, MixingParameters(x=0.77), +1, 500, np.random.default_rng(1))
    assert np.array_equal(ev.s12, t.s12[ev.index]) and np.array_equal(ev.s13, t.s13[ev.index])
    sub = ev[10:20]
    assert np.array_equal(sub.index, ev.index[10:20])
    rng = np.random.default_rng(2)
    assert np.array_equal(apply_time_resolution(ev, 0.1, rng).index, ev.index)
    assert np.array_equal(apply_mistag(ev, 0.3, rng).index, ev.index)


# ------------------------------------------------------------ density
@pytest.mark.parametrize("tag", [+1, -1])
def test_density_is_normalised(tag):
    """Averaged over the (flat) pool and integrated over tau, the density is 1."""
    from tdalitz.likelihood import density
    integrate = pytest.importorskip("scipy.integrate")
    t = toy_table(400)
    mix = MixingParameters(x=2.0, y=0.1, qp=0.9 * np.exp(-0.6j))
    tau = np.linspace(0.0, 60.0, 30_001)
    f = density(t, COEFFS, mix, tag, np.arange(len(t))[:, None], tau[None, :])
    per_point = integrate.simpson(f, x=tau, axis=1)
    assert per_point.mean() == pytest.approx(1.0, rel=1e-8)


def test_event_density_uses_each_events_tag():
    from tdalitz.likelihood import density, event_density
    t = toy_table()
    mix = MixingParameters(x=0.77, qp=np.exp(-0.76j))
    rng = np.random.default_rng(3)
    plus = generate_tagged(t, COEFFS, mix, +1, 50, rng)
    minus = generate_tagged(t, COEFFS, mix, -1, 50, rng)
    both = type(plus)(*[np.concatenate([getattr(plus, f), getattr(minus, f)])
                        for f in ("s12", "s13", "tau", "tag", "index")])
    got = event_density(t, COEFFS, mix, both)
    assert np.allclose(got[:50], density(t, COEFFS, mix, +1, plus.index, plus.tau))
    assert np.allclose(got[50:], density(t, COEFFS, mix, -1, minus.index, minus.tau))


def test_event_density_needs_pool_indices():
    from tdalitz import Events
    from tdalitz.likelihood import event_density
    t = toy_table()
    ev = Events(t.s12[:3], t.s13[:3], np.ones(3), np.ones(3, dtype=np.int8))
    with pytest.raises(ValueError, match="index"):
        event_density(t, COEFFS, MixingParameters(x=1.0), ev)


# ------------------------------------------------------------ KL divergence
def test_kl_vanishes_between_identical_models():
    from tdalitz.likelihood import expected_kl
    t = toy_table()
    model = (COEFFS, MixingParameters(x=0.77, qp=np.exp(-0.7j)))
    for tag in (+1, -1):
        kl, _ = expected_kl(t, model, model, tag)
        assert kl == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("x, y", [(0.77, 0.0), (27.0, 0.06), (0.0041, -0.0064)])
@pytest.mark.parametrize("tag", [+1, -1])
def test_kl_matches_independent_quadrature(x, y, tag):
    """Constant amplitude: the KL is a one-dimensional integral over tau."""
    from tdalitz.likelihood import expected_kl
    integrate = pytest.importorskip("scipy.integrate")
    a, a_bar = 1.0 + 0j, 0.7 * np.exp(0.9j)
    t = constant_table(a, a_bar)
    alt = ({"c": Coefficient(1.0)}, MixingParameters(x=x, y=y, qp=np.exp(-0.8j)))
    null = ({"c": Coefficient(1.0)}, MixingParameters(x=x, y=y, qp=1.0 + 0j))

    def normalized(model):
        mix = model[1]
        r = lambda s: np.exp(-s) * tagged_rate(np.array([a]), np.array([a_bar]), mix, s, tag)[0]
        norm = integrate.quad(r, 0, np.inf, limit=2000)[0]
        return lambda s: r(s) / norm

    f_alt, f_null = normalized(alt), normalized(null)
    ref = integrate.quad(lambda s: f_alt(s) * np.log(f_alt(s) / f_null(s)),
                         0, 60 / (1 - abs(y)), limit=5000, epsabs=1e-13)[0]
    kl, _ = expected_kl(t, alt, null, tag)
    assert kl == pytest.approx(ref, rel=1e-4, abs=1e-12)


def test_kl_matches_monte_carlo_log_likelihood_ratio():
    """Expected log-likelihood ratio over events generated under the alternative."""
    from tdalitz.likelihood import event_density, expected_kl
    t = toy_table()
    alt = (COEFFS, MixingParameters(x=0.77, qp=np.exp(-0.76j)))
    null = (COEFFS, MixingParameters(x=0.77, qp=1.0 + 0j))
    rng = np.random.default_rng(5)
    for tag in (+1, -1):
        ev = generate_tagged(t, *alt, tag, 200_000, rng)
        llr = np.log(event_density(t, *alt, ev) / event_density(t, *null, ev))
        kl, _ = expected_kl(t, alt, null, tag)
        assert llr.mean() == pytest.approx(kl, abs=5 * llr.std() / np.sqrt(len(llr)))


# ------------------------------------------------------------ Asimov / information
def test_asimov_significance_is_sqrt_n_times_summed_kl():
    from tdalitz.likelihood import asimov_significance, expected_kl
    t = toy_table()
    alt = (COEFFS, MixingParameters(x=0.77, qp=np.exp(-0.76j)))
    null = (COEFFS, MixingParameters(x=0.77, qp=1.0 + 0j))
    kl = sum(expected_kl(t, alt, null, tag)[0] for tag in (+1, -1))
    z, _ = asimov_significance(t, alt, null, n_events=1000)
    assert z == pytest.approx(np.sqrt(1000 * kl), rel=1e-12)


@pytest.mark.parametrize("x", [0.77, 3.0, 27.0])
def test_information_reproduces_the_notes_time_axis_result(x):
    """Constant amplitude, |lambda| = 1, q/p = exp(i theta): S = sin(theta), C = 0,
    and the per-event information for theta at 0 is V_x."""
    from tdalitz.likelihood import information
    t = constant_table()
    family = lambda theta: ({"c": Coefficient(1.0)}, MixingParameters(x=x, qp=np.exp(1j * theta)))
    v_x = 2 * x**2 / (1 + 4 * x**2) - x**2 / (1 + x**2) ** 2
    info, _ = information(t, family, 0.0)
    assert info == pytest.approx(v_x, rel=1e-4)


def test_small_signal_significance_is_theta_sqrt_n_information():
    from tdalitz.likelihood import asimov_significance, information
    t = toy_table()
    fam = beta_family(0.77)
    info, _ = information(t, fam, 0.0)
    for beta in (0.01, 0.03):
        z, _ = asimov_significance(t, fam(beta), fam(0.0), n_events=10_000)
        assert z == pytest.approx(beta * np.sqrt(10_000 * info), rel=0.02)


def test_information_error_estimate_is_honest():
    """Subsampling the pool: the quoted error covers the full-pool value."""
    from tdalitz.likelihood import information
    t = toy_table(20_000)
    fam = beta_family(0.77)
    full, _ = information(t, fam, 0.0)
    sub, err = information(t, fam, 0.0, n_points=2_000, rng=np.random.default_rng(9))
    assert err > 0
    assert sub == pytest.approx(full, abs=5 * err)
