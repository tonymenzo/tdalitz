"""First-principles checks of the physics conventions in docs/conventions.md.

Each test fixes the meaning of a parameter by comparing the package against an
independent calculation, rather than against itself.
"""

import numpy as np
import pytest

from tdalitz import MixingParameters, tagged_rate


def exact_evolution(a, a_bar, m_l, m_h, gam_l, gam_h, p, q, tau, tag):
    """|<f|P(tau)>|^2 from the eigenstate decomposition of the effective
    Hamiltonian, with |P_L> = p|P0> + q|P0bar> and |P_H> = p|P0> - q|P0bar>
    (basis vectors (p, q) and (p, -q)), eigenvalues mu = m - i Gamma / 2."""
    mu_l, mu_h = m_l - 0.5j * gam_l, m_h - 0.5j * gam_h
    e_l, e_h = np.exp(-1j * mu_l * tau), np.exp(-1j * mu_h * tau)
    if tag == +1:            # |P0> = (|P_L> + |P_H>) / (2p)
        c0 = 0.5 * (e_l + e_h)
        c0bar = 0.5 * (q / p) * (e_l - e_h)
    else:                    # |P0bar> = (|P_L> - |P_H>) / (2q)
        c0 = 0.5 * (p / q) * (e_l - e_h)
        c0bar = 0.5 * (e_l + e_h)
    return np.abs(c0 * a + c0bar * a_bar) ** 2


@pytest.mark.parametrize("tag", [+1, -1])
@pytest.mark.parametrize("seed", range(4))
def test_parameters_mean_what_the_conventions_say(tag, seed):
    """x = (m_H - m_L)/Gamma, y = (Gamma_L - Gamma_H)/(2 Gamma), q/p as in
    |P_L,H> = p|P0> +- q|P0bar>, with Gamma = (Gamma_L + Gamma_H)/2 = 1."""
    rng = np.random.default_rng(seed)
    a = complex(*rng.normal(size=2))
    a_bar = complex(*rng.normal(size=2))
    p = complex(*rng.normal(size=2))
    q = complex(*rng.normal(size=2))           # |q/p| != 1 allowed
    m_l, dm = rng.uniform(-1, 1), rng.uniform(0.1, 3.0)
    dgam = rng.uniform(-0.8, 0.8)              # Gamma_L - Gamma_H
    gam_l, gam_h = 1.0 + dgam / 2, 1.0 - dgam / 2
    tau = np.linspace(0.0, 6.0, 301)

    expected = exact_evolution(a, a_bar, m_l, m_l + dm, gam_l, gam_h, p, q, tau, tag)
    mix = MixingParameters(x=dm, y=dgam / 2.0, qp=q / p)
    got = np.exp(-tau) * tagged_rate(np.full_like(tau, a, dtype=complex),
                                     np.full_like(tau, a_bar, dtype=complex),
                                     mix, tau, tag)
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("seed", range(3))
def test_rate_expansion_in_c_and_s(seed):
    """For y = 0 and |q/p| = 1 the rates take the standard form
        R_+ = q0 [1 + C cos(x tau) - S sin(x tau)],
        R_- = q0 [1 - C cos(x tau) + S sin(x tau)],
    with lambda = (q/p) Abar / A, C = (1 - |lambda|^2)/(1 + |lambda|^2),
    S = 2 Im(lambda)/(1 + |lambda|^2), q0 = (|A|^2 + |Abar|^2)/2."""
    rng = np.random.default_rng(seed)
    a, a_bar = complex(*rng.normal(size=2)), complex(*rng.normal(size=2))
    qp = np.exp(1j * rng.uniform(-np.pi, np.pi))
    x = rng.uniform(0.2, 30.0)
    lam = qp * a_bar / a
    c = (1 - abs(lam) ** 2) / (1 + abs(lam) ** 2)
    s = 2 * lam.imag / (1 + abs(lam) ** 2)
    q0 = 0.5 * (abs(a) ** 2 + abs(a_bar) ** 2)
    tau = np.linspace(0.0, 5.0, 200)
    mix = MixingParameters(x=x, y=0.0, qp=qp)
    A, B = np.full_like(tau, a, dtype=complex), np.full_like(tau, a_bar, dtype=complex)
    assert np.allclose(tagged_rate(A, B, mix, tau, +1),
                       q0 * (1 + c * np.cos(x * tau) - s * np.sin(x * tau)))
    assert np.allclose(tagged_rate(A, B, mix, tau, -1),
                       q0 * (1 - c * np.cos(x * tau) + s * np.sin(x * tau)))


def test_relabelling_the_eigenstates_is_a_symmetry():
    """(x, y, q/p) -> (-x, -y, -q/p) swaps the L/H labels and must leave every
    rate unchanged: only the sign of x*y and of lambda are physical."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=50) + 1j * rng.normal(size=50)
    B = rng.normal(size=50) + 1j * rng.normal(size=50)
    tau = rng.exponential(1.0, 50)
    m1 = MixingParameters(x=0.8, y=0.15, qp=0.9 * np.exp(0.7j))
    m2 = MixingParameters(x=-0.8, y=-0.15, qp=-0.9 * np.exp(0.7j))
    for tag in (+1, -1):
        assert np.allclose(tagged_rate(A, B, m1, tau, tag), tagged_rate(A, B, m2, tau, tag))


def test_cp_phase_convention_trades_against_q_over_p():
    """Changing the CP phase convention flips the sign of Abar; the physics is
    unchanged provided q/p flips with it (lambda is invariant)."""
    rng = np.random.default_rng(8)
    A = rng.normal(size=50) + 1j * rng.normal(size=50)
    B = rng.normal(size=50) + 1j * rng.normal(size=50)
    tau = rng.exponential(1.0, 50)
    m1 = MixingParameters(x=1.3, y=0.05, qp=np.exp(-0.7j))
    m2 = MixingParameters(x=1.3, y=0.05, qp=-np.exp(-0.7j))
    for tag in (+1, -1):
        assert np.allclose(tagged_rate(A, B, m1, tau, tag), tagged_rate(A, -B, m2, tau, tag))


@pytest.mark.parametrize("eta, sign", [(+1, -1.0), (-1, +1.0)])
def test_standard_model_sign_of_s_for_cp_eigenstates(eta, sign):
    """With Abar(s12, s13) = A(s13, s12) and q/p = exp(-2 i beta), a CP-even
    component (symmetric under the pion exchange, e.g. f0 KS) must give
    S = -sin 2beta and a CP-odd one (antisymmetric, e.g. rho0 KS) S = +sin 2beta,
    i.e. S_f = -eta_f sin 2beta as in the Standard Model."""
    beta = 0.3876
    a = 0.7 - 0.2j
    a_bar = eta * a                            # mirror image of a (anti)symmetric wave
    lam = np.exp(-2j * beta) * a_bar / a
    s = 2 * lam.imag / (1 + abs(lam) ** 2)
    assert s == pytest.approx(sign * np.sin(2 * beta))


def _laura_timedep_rate(a, a_bar, tag, t, gamma, dm, dgamma, phi_mix, eta):
    """Signal rate of Laura++'s LauTimeDepFitModel, transcribed from the
    unreleased timedep-branch (commit 5e03d93, src/LauTimeDepFitModel.cc):
    perfect tagging, no production asymmetry, eta = +1 (CPEven) / -1 (CPOdd)."""
    inter = a_bar * np.conj(a) * np.exp(-1j * phi_mix)      # phiMixComplex_ = e^{-i phiMix}
    e = np.exp(-gamma * t)
    return e * ((abs(a) ** 2 + abs(a_bar) ** 2) * np.cosh(0.5 * dgamma * t)
                + 2 * eta * inter.real * np.sinh(0.5 * dgamma * t)
                + tag * ((abs(a) ** 2 - abs(a_bar) ** 2) * np.cos(dm * t)
                         - 2 * eta * inter.imag * np.sin(dm * t)))


@pytest.mark.parametrize("eta", [+1, -1])
def test_mapping_to_laura_timedep_branch(eta):
    """Laura++'s time-dependent branch equals this package (up to a factor 2)
    with x = Delta m tau, y = -Delta Gamma tau / 2, q/p = eta exp(-i phiMix)."""
    rng = np.random.default_rng(11)
    gamma, dm, dgamma, phi = 1 / 1.480, 17.69, 0.100, -0.0364   # their Bs example
    t = np.linspace(0.0, 8.0, 801)
    for _ in range(4):
        a, a_bar = complex(*rng.normal(size=2)), complex(*rng.normal(size=2))
        mix = MixingParameters(x=dm / gamma, y=-dgamma / (2 * gamma), qp=eta * np.exp(-1j * phi))
        A = np.full_like(t, a, dtype=complex)
        B = np.full_like(t, a_bar, dtype=complex)
        for tag in (+1, -1):
            ours = 2 * np.exp(-gamma * t) * tagged_rate(A, B, mix, gamma * t, tag)
            assert np.allclose(ours, _laura_timedep_rate(a, a_bar, tag, t, gamma, dm, dgamma, phi, eta),
                               rtol=1e-10)
