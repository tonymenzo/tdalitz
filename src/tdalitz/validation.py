"""Exact expected distributions, for validating generated samples.

Everything here is computed analytically from an amplitude table, without
generating events, so comparing a generated sample with these expectations
tests the generator rather than re-running it.

The basis is an exact decomposition of the tagged rate.  For a meson tagged at
production (``tag = +1``) write, at each Dalitz point,

    P = |A|^2,    Q = |q/p|^2 |Abar|^2,    Z = conj(A) (q/p) Abar,

and for an antimeson (``tag = -1``)

    P = |Abar|^2, Q = |p/q|^2 |A|^2,       Z = conj(Abar) (p/q) A.

Then, with ``g_plus, g_minus`` from :mod:`tdalitz.mixing`,

    R(tau) = 1/2 (P + Q) cosh(y tau) + 1/2 (P - Q) cos(x tau)
             - Re(Z) sinh(y tau) - Im(Z) sin(x tau),

and every decay-time integral against ``exp(-tau)`` has a closed form.  Because
this expression is linear in ``(P, Q, Z)``, sums over any set of pool points --
a whole Dalitz plot or one region of it -- reduce to three numbers.

Normalization follows :func:`tdalitz.generate.generate_tagged`: each tag is
normalized separately, i.e. the expectations describe a sample of fixed size
per tag.

With ``coherent=True`` the time variable is the decay-time difference
``Delta t`` of a coherent pair (B factory), with density
``exp(-|Delta t|) R(Delta t)`` on the whole real line.  For ``Delta t < 0`` the
integrand is the ``tau > 0`` one with the odd terms (``sinh``, ``sin``, i.e.
``Z``) reversed in sign, so the same closed forms apply; integrated over all
``Delta t`` only the even terms remain, doubled.
"""

from __future__ import annotations

import numpy as np

from .amplitude import AmplitudeTable
from .mixing import MixingParameters

__all__ = [
    "time_integrated_rate", "pool_weights", "expected_histogram",
    "time_density", "expected_tau_histogram", "expected_phase_histogram",
    "pearson_chi2",
    "binned_asymmetry", "expected_asymmetry", "asymmetry_chi2",
]


def _pqz(amp, amp_bar, mixing: MixingParameters, tag: int):
    """Per-point coefficients (P, Q, Z) of the rate decomposition."""
    if tag == +1:
        return (np.abs(amp) ** 2, abs(mixing.qp) ** 2 * np.abs(amp_bar) ** 2,
                np.conj(amp) * mixing.qp * amp_bar)
    if tag == -1:
        return (np.abs(amp_bar) ** 2, np.abs(amp) ** 2 / abs(mixing.qp) ** 2,
                np.conj(amp_bar) * amp / mixing.qp)
    raise ValueError("tag must be +1 or -1")


def _integrals(mixing: MixingParameters):
    """int_0^inf exp(-tau) {cosh y tau, cos x tau, sinh y tau, sin x tau} d tau."""
    x, y = mixing.x, mixing.y
    if abs(y) >= 1.0:
        raise ValueError("|y| must be below 1")
    return 1.0 / (1 - y * y), 1.0 / (1 + x * x), y / (1 - y * y), x / (1 + x * x)


def time_integrated_rate(amp, amp_bar, mixing: MixingParameters, tag: int,
                         coherent: bool = False) -> np.ndarray:
    """``int_0^inf exp(-tau) R(tau) d tau`` at each point, in closed form.

    With ``coherent=True``, ``int exp(-|dt|) R(dt) d dt`` over the real line.
    """
    p, q, z = _pqz(amp, amp_bar, mixing, tag)
    ch, co, sh, si = _integrals(mixing)
    if coherent:
        return (p + q) * ch + (p - q) * co
    return 0.5 * (p + q) * ch + 0.5 * (p - q) * co - z.real * sh - z.imag * si


def pool_weights(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                 tag: int, coherent: bool = False) -> np.ndarray:
    """Probability that a generated event sits at each pool point.

    The pool is flat in phase space, so this is the time-integrated rate at
    each point, normalized to sum to one.
    """
    amp, amp_bar = table.total(coefficients)
    w = time_integrated_rate(amp, amp_bar, mixing, tag, coherent)
    return w / w.sum()


def expected_histogram(values, weights, bins) -> np.ndarray:
    """Expected bin probabilities of any per-point quantity (e.g. a mass)."""
    return np.histogram(values, bins=bins, weights=weights)[0]


def _sums(table, coefficients, mixing, tag, mask, coherent=False):
    amp, amp_bar = table.total(coefficients)
    p, q, z = _pqz(amp, amp_bar, mixing, tag)
    norm = time_integrated_rate(amp, amp_bar, mixing, tag, coherent).sum()
    if mask is not None:
        p, q, z = p[mask], q[mask], z[mask]
    return p.sum(), q.sum(), z.sum(), norm


def time_density(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                 tag: int, tau, mask=None, coherent: bool = False) -> np.ndarray:
    """Decay-time density of generated events, ``dN / (N d tau)``.

    With ``mask`` (a boolean array over the pool) this is the joint density of
    decay time and "event lies in the region", so it integrates to the region's
    share of the sample rather than to one.
    """
    sp, sq, sz, norm = _sums(table, coefficients, mixing, tag, mask, coherent)
    tau = np.asarray(tau, dtype=float)
    x, y = mixing.x, mixing.y
    rate = (0.5 * (sp + sq) * np.cosh(y * tau) + 0.5 * (sp - sq) * np.cos(x * tau)
            - sz.real * np.sinh(y * tau) - sz.imag * np.sin(x * tau))
    return np.exp(-np.abs(tau)) * rate / norm


def _antiderivative(sp, sq, sz, mixing, tau):
    """Closed-form antiderivative of exp(-tau) R(tau) (up to a constant)."""
    x, y = mixing.x, mixing.y
    e_m, e_p = np.exp(-(1 - y) * tau) / (1 - y), np.exp(-(1 + y) * tau) / (1 + y)
    i_ch = -0.5 * (e_m + e_p)
    i_sh = -0.5 * (e_m - e_p)
    e = np.exp(-tau) / (1 + x * x)
    i_co = e * (x * np.sin(x * tau) - np.cos(x * tau))
    i_si = -e * (np.sin(x * tau) + x * np.cos(x * tau))
    return 0.5 * (sp + sq) * i_ch + 0.5 * (sp - sq) * i_co - sz.real * i_sh - sz.imag * i_si


def _coherent_cumulative(sp, sq, sz, mixing, dt):
    """int_0^dt exp(-|t|) R(t) dt for either sign of dt (zero at dt = 0).

    For dt < 0 substitute u = -t: the integrand becomes exp(-u) R with Z -> -Z.
    """
    dt = np.asarray(dt, dtype=float)
    pos = _antiderivative(sp, sq, sz, mixing, np.maximum(dt, 0.0)) - \
        _antiderivative(sp, sq, sz, mixing, 0.0)
    neg = -(_antiderivative(sp, sq, -sz, mixing, np.maximum(-dt, 0.0)) -
            _antiderivative(sp, sq, -sz, mixing, 0.0))
    return np.where(dt >= 0.0, pos, neg)


def expected_tau_histogram(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                           tag: int, bins, mask=None, transform=None,
                           tau_max: float = None, max_step: float = 1e-3,
                           coherent: bool = False) -> np.ndarray:
    """Expected bin probabilities of the decay time, or of a function of it.

    Parameters
    ----------
    bins:
        Bin edges, in ``tau`` or -- with ``transform`` -- in the transformed
        variable.
    mask:
        Optional boolean region over the pool; probabilities are then joint
        with "event lies in the region".
    transform:
        Optional function of ``tau`` to histogram instead.  The ``tau`` axis is
        cut into fine cells, each cell's probability is integrated exactly, and
        the cell is assigned to the bin containing the transform of its
        midpoint -- an approximation, accurate to a fraction of one cell per
        bin edge.  For the folded phase use :func:`expected_phase_histogram`,
        which is exact.
    coherent:
        Bins are in ``Delta t`` of a coherent pair, on the whole real line
        (no ``transform``).
    """
    sp, sq, sz, norm = _sums(table, coefficients, mixing, tag, mask, coherent)
    bins = np.asarray(bins, dtype=float)
    if coherent:
        if transform is not None:
            raise NotImplementedError("transform is not supported in the coherent mode")
        return np.diff(_coherent_cumulative(sp, sq, sz, mixing, bins)) / norm
    if transform is None:
        f = _antiderivative(sp, sq, sz, mixing, bins)
        return np.diff(f) / norm

    if tau_max is None:
        tau_max = 45.0 / (1.0 - abs(mixing.y))
    step = max_step if mixing.x == 0 else min(max_step, 2 * np.pi / abs(mixing.x) / 400)
    edges = np.linspace(0.0, tau_max, int(np.ceil(tau_max / step)) + 1)
    cell = np.diff(_antiderivative(sp, sq, sz, mixing, edges)) / norm
    mid = 0.5 * (edges[:-1] + edges[1:])
    return np.histogram(transform(mid), bins=bins, weights=cell)[0]


def expected_phase_histogram(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                             tag: int, bins, mask=None, tau_max: float = None,
                             coherent: bool = False) -> np.ndarray:
    """Exact expected bin probabilities of the folded phase ``x tau mod 2 pi``.

    The phase bin ``[a, b]`` is the union over periods ``k`` of the decay-time
    intervals ``[(2 pi k + a) / x, (2 pi k + b) / x]``, each integrated with the
    closed-form antiderivative, so the result is exact for any binning (up to
    the neglected tail beyond ``tau_max``, of order ``exp(-tau_max (1 - |y|))``).

    Parameters
    ----------
    bins:
        Phase bin edges within ``[0, 2 pi]``.
    mask:
        Optional boolean region over the pool, as for
        :func:`expected_tau_histogram`.
    """
    if coherent:
        raise NotImplementedError("the folded phase is implemented for tau >= 0 only")
    x = mixing.x
    if x <= 0.0:
        raise ValueError("the folded phase needs x > 0")
    bins = np.asarray(bins, dtype=float)
    if bins.min() < 0.0 or bins.max() > 2 * np.pi:
        raise ValueError("phase bins must lie within [0, 2 pi]")
    if tau_max is None:
        tau_max = 45.0 / (1.0 - abs(mixing.y))
    sp, sq, sz, norm = _sums(table, coefficients, mixing, tag, mask)
    periods = np.arange(int(np.ceil(x * tau_max / (2 * np.pi))))[:, None]
    edges_tau = (2 * np.pi * periods + bins[None, :]) / x
    f = _antiderivative(sp, sq, sz, mixing, edges_tau)
    return np.diff(f, axis=1).sum(axis=0) / norm


def pearson_chi2(counts, probs, min_expected: float = 5.0) -> tuple:
    """Pearson ``chi^2`` and degrees of freedom for a histogram *shape*.

    Expected counts are ``N * probs / sum(probs)`` with ``N = sum(counts)``, so
    only the shape is tested (one degree of freedom is spent on ``N``).  Bins
    expecting fewer than ``min_expected`` events are pooled into a single bin.
    """
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    expected = counts.sum() * probs / probs.sum()
    dense = expected >= min_expected
    obs = list(counts[dense])
    exp = list(expected[dense])
    if np.any(~dense) and expected[~dense].sum() > 0:
        obs.append(counts[~dense].sum())
        exp.append(expected[~dense].sum())
    obs, exp = np.array(obs), np.array(exp)
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    return chi2, len(obs) - 1


def binned_asymmetry(counts_plus, counts_minus) -> tuple:
    """Per-bin tag asymmetry ``(n+ - n-) / (n+ + n-)`` and its binomial error.

    Empty bins give NaN for both.
    """
    a = np.asarray(counts_plus, dtype=float)
    b = np.asarray(counts_minus, dtype=float)
    n = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        asym = (a - b) / n
        err = np.sqrt((1.0 - asym ** 2) / n)
    return asym, err


def expected_asymmetry(probs_plus, probs_minus) -> np.ndarray:
    """Asymmetry of two per-tag bin probabilities (equal sample sizes)."""
    p = np.asarray(probs_plus, dtype=float)
    m = np.asarray(probs_minus, dtype=float)
    return (p - m) / (p + m)


def asymmetry_chi2(counts_plus, counts_minus, probs_plus, probs_minus,
                   min_count: int = 10) -> tuple:
    """``chi^2`` and degrees of freedom of a binned tag asymmetry.

    Each tag's probabilities are normalized over the histogram and scaled to
    that tag's observed count, so they may be joint with a region (as from
    ``mask=``) and need not sum to one.  Conditional on the per-bin total
    ``n = n+ + n-``, ``n+`` is then binomial with probability
    ``pi = e+ / (e+ + e-)`` for expected counts ``e+-``.  Bins with fewer than
    ``min_count`` events are skipped; one degree of freedom is spent on the
    fixed split of the total between the tags.
    """
    a = np.asarray(counts_plus, dtype=float)
    b = np.asarray(counts_minus, dtype=float)
    pp = np.asarray(probs_plus, dtype=float)
    pm = np.asarray(probs_minus, dtype=float)
    ep = a.sum() * pp / pp.sum()
    em = b.sum() * pm / pm.sum()
    n = a + b
    use = (n >= min_count) & (ep + em > 0)
    pi = ep[use] / (ep[use] + em[use])
    chi2 = float(np.sum((a[use] - n[use] * pi) ** 2 / (n[use] * pi * (1 - pi))))
    return chi2, int(use.sum()) - 1
