"""Coefficients with independent particle and antiparticle values, and the
coherent (B-factory) decay-time mode.

At a B factory the B0 Bbar0 pair is produced in a coherent C-odd state and the
observable is the decay-time difference Delta t, which takes both signs.  The
tagged rate as a function of Delta t has the same form as the single-meson rate
with tau -> Delta t, times exp(-|Delta t|), where the tag is the flavour of the
signal B at Delta t = 0 (opposite to the flavour of the tag-side B).
"""

import numpy as np
import pytest

from tdalitz import Coefficient, MixingParameters, generate_tagged, tagged_rate
from tdalitz import likelihood
from test_core import toy_table


# ------------------------------------------------------------ coefficients
def test_from_complex_sets_both_values_exactly():
    c, c_bar = 0.3 - 0.4j, -0.1 + 0.2j
    coeff = Coefficient.from_complex(c, c_bar)
    assert coeff.c == pytest.approx(c)
    assert coeff.c_bar == pytest.approx(c_bar)


def test_from_complex_allows_unequal_magnitudes():
    coeff = Coefficient.from_complex(1.0, 0.5j)
    assert abs(coeff.c) == pytest.approx(1.0)
    assert abs(coeff.c_bar) == pytest.approx(0.5)


def test_from_complex_matches_the_phase_parameterization_when_it_can():
    a, delta, phi = 0.7, 1.1, 0.3
    ref = Coefficient(a, delta, phi)
    coeff = Coefficient.from_complex(ref.c, ref.c_bar)
    assert coeff.c == pytest.approx(ref.c)
    assert coeff.c_bar == pytest.approx(ref.c_bar)


def test_default_coefficients_are_unchanged():
    coeff = Coefficient(0.7, 1.1, 0.3)
    assert coeff.c == pytest.approx(0.7 * np.exp(1.4j))
    assert coeff.c_bar == pytest.approx(0.7 * np.exp(0.8j))


def test_table_total_uses_the_independent_antiparticle_value():
    table = toy_table(2_000)
    coeffs = {"Kstar": Coefficient.from_complex(1.0, 0.5), "rho": Coefficient.from_complex(0.3j, 0.3j)}
    a, a_bar = table.total(coeffs)
    np.testing.assert_allclose(a, table.amp[:, 0] + 0.3j * table.amp[:, 1])
    np.testing.assert_allclose(a_bar, 0.5 * table.amp_bar[:, 0] + 0.3j * table.amp_bar[:, 1])


# ------------------------------------------------------------ coherent mode
COEFFS = {"Kstar": Coefficient(1.0, 0.0), "rho": Coefficient(0.8, 1.2)}
MIX = MixingParameters(x=0.77, y=0.0, qp=np.exp(-0.79j))


def analytic_moments(table, coefficients, mixing, tag, functions, grid=None):
    """E[f(Delta t)] under exp(-|dt|) <R_tag(dt)>_pool, by quadrature."""
    a, a_bar = table.total(coefficients)
    dt = np.linspace(-25, 25, 20_001) if grid is None else grid
    rate = np.array([tagged_rate(a, a_bar, mixing, t, tag).mean() for t in dt]) * np.exp(-np.abs(dt))
    norm = np.trapezoid(rate, dt)
    return [np.trapezoid(f(dt) * rate, dt) / norm for f in functions]


def test_coherent_times_take_both_signs():
    table = toy_table(20_000)
    ev = generate_tagged(table, COEFFS, MIX, +1, 20_000, np.random.default_rng(1), coherent=True)
    assert np.any(ev.tau < 0) and np.any(ev.tau > 0)
    assert ev.coherent


@pytest.mark.parametrize("tag", [+1, -1])
def test_coherent_generation_matches_the_analytic_density(tag):
    table = toy_table(20_000)
    n = 200_000
    ev = generate_tagged(table, COEFFS, MIX, tag, n, np.random.default_rng(2), coherent=True)
    x = MIX.x
    functions = [lambda t: np.sin(x * t), lambda t: np.cos(x * t), np.abs, np.sign]
    expected = analytic_moments(table, COEFFS, MIX, tag, functions)
    for f, mu in zip(functions, expected):
        values = f(ev.tau)
        err = values.std() / np.sqrt(n)
        assert values.mean() == pytest.approx(mu, abs=5 * err)


def test_coherent_generation_with_a_width_difference():
    table = toy_table(20_000)
    mix = MixingParameters(x=27.0, y=0.06, qp=np.exp(0.5j))
    n = 200_000
    ev = generate_tagged(table, COEFFS, mix, +1, n, np.random.default_rng(3), coherent=True)
    functions = [lambda t: np.sin(27.0 * t), lambda t: np.cos(27.0 * t), np.abs, np.sign]
    expected = analytic_moments(table, COEFFS, mix, +1, functions,
                                grid=np.linspace(-30, 30, 400_001))
    for f, mu in zip(functions, expected):
        values = f(ev.tau)
        assert values.mean() == pytest.approx(mu, abs=5 * values.std() / np.sqrt(n))


def test_single_meson_mode_is_the_default():
    table = toy_table(5_000)
    ev = generate_tagged(table, COEFFS, MIX, +1, 5_000, np.random.default_rng(5))
    assert np.all(ev.tau >= 0)
    assert not ev.coherent


def test_likelihood_refuses_coherent_events():
    table = toy_table(5_000)
    ev = generate_tagged(table, COEFFS, MIX, +1, 500, np.random.default_rng(6), coherent=True)
    with pytest.raises(NotImplementedError):
        likelihood.log_likelihood(table, COEFFS, MIX, ev)


def test_resolution_refuses_to_drop_negative_delta_t():
    from tdalitz import apply_time_resolution
    table = toy_table(5_000)
    rng = np.random.default_rng(7)
    ev = generate_tagged(table, COEFFS, MIX, +1, 500, rng, coherent=True)
    with pytest.raises(ValueError):
        apply_time_resolution(ev, 0.1, rng, allow_negative=False)
    assert apply_time_resolution(ev, 0.1, rng).coherent


# ------------------------------------------------- exact coherent expectations
from tdalitz.validation import (expected_tau_histogram, pearson_chi2, pool_weights,  # noqa: E402
                                time_density, time_integrated_rate)


def _quadrature(table, coefficients, mixing, tag, edges, n=200_001):
    """Bin probabilities of Delta t by brute-force quadrature over the pool."""
    a, a_bar = table.total(coefficients)
    dt = np.linspace(-40, 40, n)
    rate = np.array([tagged_rate(a, a_bar, mixing, t, tag).sum() for t in dt]) * np.exp(-np.abs(dt))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(dt))])
    return np.diff(np.interp(edges, dt, cum)) / cum[-1]


@pytest.mark.parametrize("tag", [+1, -1])
@pytest.mark.parametrize("mixing", [MIX, MixingParameters(x=3.0, y=0.08, qp=0.95 * np.exp(0.4j))])
def test_coherent_delta_t_histogram_matches_quadrature(tag, mixing):
    table = toy_table(3_000)
    edges = np.linspace(-6, 6, 25)
    exact = expected_tau_histogram(table, COEFFS, mixing, tag, edges, coherent=True)
    np.testing.assert_allclose(exact, _quadrature(table, COEFFS, mixing, tag, edges), atol=2e-5)


def test_coherent_delta_t_histogram_sums_to_one():
    table = toy_table(3_000)
    probs = expected_tau_histogram(table, COEFFS, MIX, +1, [-80.0, -1.0, 0.0, 2.0, 80.0],
                                   coherent=True)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_coherent_time_integrated_rate_keeps_only_even_terms():
    table = toy_table(500)
    a, a_bar = table.total(COEFFS)
    dt = np.linspace(-40, 40, 400_001)
    i = 7
    rate = np.array([tagged_rate(a[i:i + 1], a_bar[i:i + 1], MIX, t, +1)[0] for t in dt])
    numeric = np.trapezoid(rate * np.exp(-np.abs(dt)), dt)
    assert time_integrated_rate(a, a_bar, MIX, +1, coherent=True)[i] == pytest.approx(numeric, rel=1e-6)


def test_coherent_time_density_integrates_to_one():
    table = toy_table(3_000)
    dt = np.linspace(-40, 40, 400_001)
    dens = time_density(table, COEFFS, MIX, -1, dt, coherent=True)
    assert np.trapezoid(dens, dt) == pytest.approx(1.0, abs=1e-6)


def test_generated_coherent_sample_matches_the_exact_expectations():
    table = toy_table(20_000)
    rng = np.random.default_rng(11)
    ev = generate_tagged(table, COEFFS, MIX, +1, 100_000, rng, coherent=True)
    edges = np.concatenate([[-80.0], np.linspace(-6, 6, 49), [80.0]])
    counts = np.histogram(ev.tau, bins=edges)[0]
    probs = expected_tau_histogram(table, COEFFS, MIX, +1, edges, coherent=True)
    chi2, ndf = pearson_chi2(counts, probs)
    stats = pytest.importorskip("scipy.stats")
    assert stats.chi2.sf(chi2, ndf) > 1e-3
    w = pool_weights(table, COEFFS, MIX, +1, coherent=True)
    e12 = np.linspace(table.s12.min(), table.s12.max(), 41)
    chi2, ndf = pearson_chi2(np.histogram(ev.s12, e12)[0], np.histogram(table.s12, e12, weights=w)[0])
    assert stats.chi2.sf(chi2, ndf) > 1e-3


def test_coherent_folded_phase_is_not_supported():
    from tdalitz.validation import expected_phase_histogram
    table = toy_table(500)
    with pytest.raises(NotImplementedError):
        expected_phase_histogram(table, COEFFS, MIX, +1, np.linspace(0, 6.28, 5), coherent=True)
