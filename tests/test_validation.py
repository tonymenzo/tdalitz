"""Tests for tdalitz.validation: exact expected distributions and a goodness-
of-fit statistic.  The expectations are computed analytically from the table,
independently of the accept/reject generator, so comparing a generated sample
to them is a genuine test of generation."""

import numpy as np
import pytest

from tdalitz import (AmplitudeTable, Coefficient, FinalState, MixingParameters,
                     generate_tagged, tagged_rate)
from tdalitz.validation import (expected_histogram, expected_tau_histogram,
                                pearson_chi2, pool_weights, time_density,
                                time_integrated_rate)

M_D0, M_KS, M_PI = 1.86484, 0.497611, 0.13957039


def toy_table(n=80_000, seed=7):
    fs = FinalState(M_D0, (M_KS, M_PI, M_PI))
    s12, s13 = fs.sample_phase_space(n, np.random.default_rng(seed))

    def comps(a, b):
        bw = lambda s, m, w: 1.0 / (m * m - s - 1j * m * w)
        return np.column_stack([bw(b, 0.89167, 0.0514), bw(fs.s23(a, b), 0.77526, 0.1474)])

    return AmplitudeTable(fs, s12, s13, ("Kstar", "rho"), comps(s12, s13), comps(s13, s12))


COEFFS = {"Kstar": Coefficient(1.0, 0.0), "rho": Coefficient(0.6, 1.1, weak=0.4)}
MIX = MixingParameters(x=2.0, y=0.1, qp=0.95 * np.exp(-0.6j))


# ---------------------------------------------------------------- rates
@pytest.mark.parametrize("tag", [+1, -1])
def test_time_integrated_rate_matches_quadrature(tag):
    rng = np.random.default_rng(1)
    a = rng.normal(size=20) + 1j * rng.normal(size=20)
    ab = rng.normal(size=20) + 1j * rng.normal(size=20)
    tau = np.linspace(0.0, 60.0, 600_001)
    numeric = np.array([np.trapezoid(np.exp(-tau) * tagged_rate(ai, bi, MIX, tau, tag), tau)
                        for ai, bi in zip(a, ab)])
    assert np.allclose(time_integrated_rate(a, ab, MIX, tag), numeric, rtol=1e-6)


def test_time_integrated_rate_closed_form_without_width_difference():
    """y = 0, |q/p| = 1: integral = q0 [1 + C/(1+x^2) - S x/(1+x^2)]."""
    a, ab, x = 0.8 - 0.3j, 0.5 + 0.6j, 3.0
    qp = np.exp(-0.9j)
    lam = qp * ab / a
    c = (1 - abs(lam) ** 2) / (1 + abs(lam) ** 2)
    s = 2 * lam.imag / (1 + abs(lam) ** 2)
    q0 = 0.5 * (abs(a) ** 2 + abs(ab) ** 2)
    got = time_integrated_rate(np.array([a]), np.array([ab]),
                               MixingParameters(x=x, y=0.0, qp=qp), +1)[0]
    assert got == pytest.approx(q0 * (1 + c / (1 + x * x) - s * x / (1 + x * x)))


# ---------------------------------------------------------------- pool weights
@pytest.mark.parametrize("tag", [+1, -1])
def test_pool_weights_are_normalised_time_integrated_rates(tag):
    t = toy_table(5000)
    w = pool_weights(t, COEFFS, MIX, tag)
    a, ab = t.total(COEFFS)
    ref = time_integrated_rate(a, ab, MIX, tag)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, ref / ref.sum())


def test_expected_histogram_sums_to_one():
    t = toy_table(5000)
    w = pool_weights(t, COEFFS, MIX, +1)
    p = expected_histogram(t.s12, w, np.linspace(t.s12.min(), t.s12.max(), 40))
    assert p.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- time density
@pytest.mark.parametrize("tag", [+1, -1])
def test_time_density_normalisation(tag):
    t = toy_table(5000)
    tau = np.linspace(0.0, 60.0, 400_001)
    full = time_density(t, COEFFS, MIX, tag, tau)
    assert np.trapezoid(full, tau) == pytest.approx(1.0, rel=1e-6)
    mask = t.s13 < 1.0
    part = time_density(t, COEFFS, MIX, tag, tau, mask=mask)
    frac = pool_weights(t, COEFFS, MIX, tag)[mask].sum()
    assert np.trapezoid(part, tau) == pytest.approx(frac, rel=1e-6)


def test_time_density_equals_brute_force_pool_average():
    t = toy_table(400)
    tau = np.linspace(0.0, 8.0, 41)
    a, ab = t.total(COEFFS)
    brute = np.array([np.exp(-ti) * tagged_rate(a, ab, MIX, ti, -1).sum() for ti in tau])
    brute /= time_integrated_rate(a, ab, MIX, -1).sum()
    assert np.allclose(time_density(t, COEFFS, MIX, -1, tau), brute, rtol=1e-10)


def test_expected_tau_histogram_matches_time_density_integral():
    t = toy_table(3000)
    bins = np.linspace(0.0, 6.0, 25)
    p = expected_tau_histogram(t, COEFFS, MIX, +1, bins)
    fine = np.linspace(0.0, 6.0, 240_001)
    dens = time_density(t, COEFFS, MIX, +1, fine)
    ref = [np.trapezoid(dens[(fine >= lo) & (fine <= hi)], fine[(fine >= lo) & (fine <= hi)])
           for lo, hi in zip(bins[:-1], bins[1:])]
    assert np.allclose(p, ref, rtol=1e-4)


def test_expected_tau_histogram_of_folded_phase_sums_to_one():
    t = toy_table(3000)
    mix = MixingParameters(x=27.0, y=0.06, qp=np.exp(-0.4j))
    bins = np.linspace(0.0, 2 * np.pi, 41)
    p = expected_tau_histogram(t, COEFFS, mix, +1, bins,
                               transform=lambda tau: np.mod(27.0 * tau, 2 * np.pi))
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- chi2
def test_pearson_chi2_is_calibrated():
    """Counts drawn from the expected probabilities give chi2/ndf ~ 1."""
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(50) * 5.0)
    ratios = []
    for _ in range(300):
        counts = rng.multinomial(20_000, probs)
        chi2, ndf = pearson_chi2(counts, probs)
        ratios.append(chi2 / ndf)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)


def test_pearson_chi2_pools_sparse_bins():
    probs = np.array([0.5, 0.49, 0.009, 0.001])
    counts = np.array([500, 490, 9, 1])
    chi2, ndf = pearson_chi2(counts, probs, min_expected=20.0)
    assert ndf == 2                       # two dense bins + one pooled bin - 1
    assert chi2 == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- the generator
@pytest.mark.parametrize("tag", [+1, -1])
def test_generator_reproduces_expected_dalitz_and_time_distributions(tag):
    """The accept/reject generator must agree with the analytic expectation in
    the Dalitz plot, in decay time, and jointly in a Dalitz region."""
    t = toy_table()
    n = 100_000
    ev = generate_tagged(t, COEFFS, MIX, tag, n, np.random.default_rng(11 + tag))
    w = pool_weights(t, COEFFS, MIX, tag)

    for var_pool, var_ev in ((t.s12, ev.s12), (t.s13, ev.s13)):
        bins = np.linspace(var_pool.min(), var_pool.max(), 50)
        chi2, ndf = pearson_chi2(np.histogram(var_ev, bins)[0], expected_histogram(var_pool, w, bins))
        assert chi2 / ndf < 1.6

    bins = np.linspace(0.0, 8.0, 50)
    counts = np.histogram(ev.tau, bins)[0]
    probs = expected_tau_histogram(t, COEFFS, MIX, tag, bins)
    chi2, ndf = pearson_chi2(counts, probs)
    assert chi2 / ndf < 1.6

    region = t.s13 < 1.0
    ev_region = ev.s13 < 1.0
    probs = expected_tau_histogram(t, COEFFS, MIX, tag, bins, mask=region)
    chi2, ndf = pearson_chi2(np.histogram(ev.tau[ev_region], bins)[0], probs)
    assert chi2 / ndf < 1.6


# ---------------------------------------------------------------- asymmetries
def test_binned_asymmetry_and_its_error():
    from tdalitz.validation import binned_asymmetry
    a, err = binned_asymmetry(np.array([60, 50, 0]), np.array([40, 50, 0]))
    assert a[0] == pytest.approx(0.2) and a[1] == pytest.approx(0.0)
    assert err[0] == pytest.approx(np.sqrt((1 - 0.04) / 100))
    assert np.isnan(a[2]) and np.isnan(err[2])


def test_expected_asymmetry_from_probabilities():
    from tdalitz.validation import expected_asymmetry
    assert expected_asymmetry(np.array([0.3, 0.2]), np.array([0.1, 0.2])) == pytest.approx([0.5, 0.0])


def test_asymmetry_chi2_is_calibrated():
    """Counts drawn from the expected per-tag probabilities give chi2/ndf ~ 1."""
    from tdalitz.validation import asymmetry_chi2
    rng = np.random.default_rng(5)
    p_plus = rng.dirichlet(np.ones(40) * 8.0)
    p_minus = rng.dirichlet(np.ones(40) * 8.0)
    ratios = []
    for _ in range(300):
        chi2, ndf = asymmetry_chi2(rng.multinomial(30_000, p_plus),
                                   rng.multinomial(30_000, p_minus), p_plus, p_minus)
        ratios.append(chi2 / ndf)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06)


def test_asymmetry_chi2_accepts_region_joint_probabilities():
    """Inside a region the per-tag probabilities are joint with "in the region"
    and sum to the region's share, which differs between tags.  Histograms of
    the in-region events must still give chi2/ndf ~ 1."""
    from tdalitz.validation import asymmetry_chi2
    rng = np.random.default_rng(6)
    # 40 in-region bins plus one "outside" bin; the region holds 30% vs 60%.
    p_plus = np.append(0.3 * rng.dirichlet(np.ones(40) * 8.0), 0.7)
    p_minus = np.append(0.6 * rng.dirichlet(np.ones(40) * 8.0), 0.4)
    ratios = []
    for _ in range(300):
        n_p = rng.multinomial(60_000, p_plus)[:40]
        n_m = rng.multinomial(60_000, p_minus)[:40]
        chi2, ndf = asymmetry_chi2(n_p, n_m, p_plus[:40], p_minus[:40])
        ratios.append(chi2 / ndf)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06)


# ---------------------------------------------------------------- folded phase
def _folded_reference(t, mix, tag, bins, mask=None):
    """Brute-force folded-phase probabilities, independent of the closed-form
    antiderivative: trapezoid-integrate the exact density over each decay-time
    interval that maps into each phase bin, period by period, to tau = 40."""
    from tdalitz.validation import time_density
    out = np.zeros(len(bins) - 1)
    for k in range(int(np.ceil(mix.x * 40.0 / (2 * np.pi)))):
        for j, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
            tau = np.linspace(2 * np.pi * k + a, 2 * np.pi * k + b, 51) / mix.x
            out[j] += np.trapezoid(time_density(t, COEFFS, mix, tag, tau, mask=mask), tau)
    return out


@pytest.mark.parametrize("nbins", [40, 32, 7])
@pytest.mark.parametrize("tag", [+1, -1])
def test_expected_phase_histogram_is_exact(nbins, tag):
    """Exact for any binning, including edges that are not commensurate with
    the 1/400-period cells of the transform method."""
    from tdalitz.validation import expected_phase_histogram
    t = toy_table(3000)
    mix = MixingParameters(x=27.0, y=0.06, qp=np.exp(-0.4j))
    bins = np.linspace(0.0, 2 * np.pi, nbins + 1)
    p = expected_phase_histogram(t, COEFFS, mix, tag, bins)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(p, _folded_reference(t, mix, tag, bins), rtol=2e-4, atol=1e-7)


def test_expected_phase_histogram_with_region_and_slow_oscillation():
    from tdalitz.validation import expected_phase_histogram
    t = toy_table(3000)
    mix = MixingParameters(x=0.77, y=0.0, qp=np.exp(-0.76j))
    mask = t.final_state.s23(t.s12, t.s13) < 0.8
    bins = np.linspace(0.0, 2 * np.pi, 13)
    p = expected_phase_histogram(t, COEFFS, mix, -1, bins, mask=mask)
    assert np.allclose(p, _folded_reference(t, mix, -1, bins, mask=mask), rtol=2e-4, atol=1e-7)
