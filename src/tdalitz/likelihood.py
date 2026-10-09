"""Exact likelihoods, KL divergences and Asimov significances.

The amplitude table is a complete, normalized model, so the *fully informed*
likelihood of a tagged sample -- the Neyman-Pearson ceiling that any
model-independent test can be compared against -- is available exactly.

Density
    For a meson tagged ``tag``, the joint density of Dalitz point and true
    decay time is

        f_tag(point, tau) = exp(-tau) R_tag(point, tau) / N_tag,

    with ``R_tag`` the tagged rate of :func:`tdalitz.mixing.tagged_rate` and
    ``N_tag`` its time-integrated pool average.  It is a density per unit
    decay time and per unit *fraction of phase space* (the pool is flat), and
    each tag is normalized separately -- matching generation, which fixes the
    number of events per tag.

Asimov significance
    For ``N`` events split equally between the tags, the median discovery
    significance of the likelihood-ratio test of ``alt`` against ``null`` is

        Z = sqrt( N * sum_tag KL(f_tag^alt || f_tag^null) ),

    and for a one-parameter family ``theta -> model`` with the null at
    ``theta0``, ``Z -> |theta - theta0| sqrt(N I)`` for small signals, where
    ``I = lim sum_tag KL / (theta - theta0)^2`` is the information per event.

KL divergences are integrated over the pool (optionally a random subset of
it) and over tau on a grid fine enough to resolve the oscillation (Simpson's
rule, at least 32 nodes per period).  Normalizations are taken on the same
points and grid, so the result is the exact KL of the discretized model:
non-negative, and exactly zero between identical models.  The quoted error
is the spread over independent batches of points, i.e. the Monte Carlo error
of the pool as an integration rule.

Everything here uses the *true* decay time: it is the ceiling, before
resolution.
"""

from __future__ import annotations

import numpy as np

from .amplitude import AmplitudeTable
from .generate import Events
from .mixing import MixingParameters, tagged_rate
from .validation import time_integrated_rate

__all__ = ["density", "event_density", "log_likelihood",
           "expected_kl", "asimov_significance", "information"]


# ----------------------------------------------------------------- density
def _normalization(table, coefficients, mixing, tag):
    amp, amp_bar = table.total(coefficients)
    return float(np.mean(time_integrated_rate(amp, amp_bar, mixing, tag))), amp, amp_bar


def density(table: AmplitudeTable, coefficients, mixing: MixingParameters, tag: int,
            index, tau) -> np.ndarray:
    """Normalized density ``f_tag`` at pool rows ``index`` and true times ``tau``."""
    norm, amp, amp_bar = _normalization(table, coefficients, mixing, tag)
    index = np.asarray(index)
    tau = np.asarray(tau, dtype=float)
    return np.exp(-tau) * tagged_rate(amp[index], amp_bar[index], mixing, tau, tag) / norm


def event_density(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                  events: Events) -> np.ndarray:
    """``f_tag`` at every event, each with its own production tag.

    The events must carry their pool ``index`` (as generated events do).  Use
    the true decay time: this is the ideal-resolution density.
    """
    if events.index is None:
        raise ValueError("events carry no pool index; generate them from the table")
    if events.coherent:
        raise NotImplementedError("densities are implemented for the single-meson decay time only")
    out = np.empty(len(events))
    for tag in (+1, -1):
        sel = events.tag == tag
        if np.any(sel):
            out[sel] = density(table, coefficients, mixing, tag,
                               events.index[sel], events.tau[sel])
    return out


def log_likelihood(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                   events: Events) -> float:
    """``sum_i log f_{tag_i}(point_i, tau_i)`` -- the fully informed likelihood."""
    return float(np.sum(np.log(event_density(table, coefficients, mixing, events))))


# --------------------------------------------------------------------- KL
def _tau_grid(*mixings):
    """Simpson nodes and weights (times exp(-tau)) resolving every oscillation."""
    y = max(abs(m.y) for m in mixings)
    x = max(abs(m.x) for m in mixings)
    tau_max = 30.0 / (1.0 - y)
    step = 0.02 if x == 0 else min(0.02, 2 * np.pi / x / 32)
    n = int(np.ceil(tau_max / step))
    n += n % 2                                          # Simpson needs an even count
    tau = np.linspace(0.0, tau_max, n + 1)
    w = np.full(n + 1, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return tau, w * (tau[1] - tau[0]) / 3.0 * np.exp(-tau)


def _rate_terms(amp, amp_bar, mixing, tag):
    """(P, Q, Z) of the rate decomposition in tdalitz.validation."""
    if tag == +1:
        return (np.abs(amp) ** 2, abs(mixing.qp) ** 2 * np.abs(amp_bar) ** 2,
                np.conj(amp) * mixing.qp * amp_bar)
    return (np.abs(amp_bar) ** 2, np.abs(amp) ** 2 / abs(mixing.qp) ** 2,
            np.conj(amp_bar) * amp / mixing.qp)


def _rate(terms, mixing, tau):
    p, q, z = (t[:, None] for t in terms)
    x, y = mixing.x, mixing.y
    return (0.5 * (p + q) * np.cosh(y * tau) + 0.5 * (p - q) * np.cos(x * tau)
            - z.real * np.sinh(y * tau) - z.imag * np.sin(x * tau))


def _kl_sums(table, alt, null, tag, idx, tau, w, chunk):
    """Per-point sums: mass under alt, mass under null, and sum w Ra log(Ra/R0)."""
    a_alt = table.total(alt[0])
    a_null = table.total(null[0])
    out = np.empty((3, len(idx)))
    for lo in range(0, len(idx), chunk):
        sl = idx[lo:lo + chunk]
        ra = _rate(_rate_terms(a_alt[0][sl], a_alt[1][sl], alt[1], tag), alt[1], tau)
        r0 = _rate(_rate_terms(a_null[0][sl], a_null[1][sl], null[1], tag), null[1], tau)
        out[0, lo:lo + chunk] = ra @ w
        out[1, lo:lo + chunk] = r0 @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(ra > 0, ra * np.log(ra / r0), 0.0)   # 0 log 0 = 0
        out[2, lo:lo + chunk] = integrand @ w
    return out


def _kl_from_sums(sums):
    mass_alt, mass_null, cross = sums.mean(axis=1)
    return cross / mass_alt + np.log(mass_null / mass_alt)


def _points(table, n_points, rng):
    if n_points is None or n_points >= len(table):
        return np.arange(len(table))
    rng = np.random.default_rng(0) if rng is None else rng
    return np.sort(rng.choice(len(table), n_points, replace=False))


def expected_kl(table: AmplitudeTable, alt: tuple, null: tuple, tag: int,
                n_points: int = 200_000, rng: np.random.Generator = None,
                n_batches: int = 10) -> tuple:
    """``KL(f_tag^alt || f_tag^null)`` per event, and its integration error.

    Parameters
    ----------
    alt, null:
        ``(coefficients, mixing)`` pairs.
    n_points:
        Integrate over a random subset of this many pool points (``None`` for
        the whole pool).
    """
    if tag not in (+1, -1):
        raise ValueError("tag must be +1 or -1")
    idx = _points(table, n_points, rng)
    tau, w = _tau_grid(alt[1], null[1])
    sums = _kl_sums(table, alt, null, tag, idx, tau, w, chunk=max(1, 4_000_000 // len(tau)))
    kl = float(_kl_from_sums(sums))
    batches = np.array_split(np.arange(len(idx)), min(n_batches, len(idx)))
    spread = np.array([_kl_from_sums(sums[:, b]) for b in batches])
    return kl, float(spread.std(ddof=1) / np.sqrt(len(batches))) if len(batches) > 1 else 0.0


def asimov_significance(table: AmplitudeTable, alt: tuple, null: tuple, n_events: int,
                        **kl_options) -> tuple:
    """Median likelihood-ratio significance for ``n_events`` in total (half per tag).

    ``Z = sqrt(n_events * sum_tag KL_tag)``; returns ``(Z, error)``.
    """
    kls = [expected_kl(table, alt, null, tag, **kl_options) for tag in (+1, -1)]
    total = sum(k for k, _ in kls)
    err = np.sqrt(sum(e ** 2 for _, e in kls))
    z = float(np.sqrt(n_events * max(total, 0.0)))
    return z, float(n_events * err / (2 * z)) if z > 0 else 0.0


def information(table: AmplitudeTable, family, theta0: float, step: float = 1e-3,
                **kl_options) -> tuple:
    """Information per event ``I`` for the parameter of ``family`` at ``theta0``.

    ``family(theta)`` returns a ``(coefficients, mixing)`` pair.  With ``N``
    events split equally between the tags, the small-signal significance is
    ``|theta - theta0| sqrt(N I)``.  Computed as the symmetric second difference
    of the summed KL, ``[KL(+step) + KL(-step)] / (2 step^2)``; returns
    ``(I, error)``.
    """
    null = family(theta0)
    values, errors = [], []
    for sign in (+1, -1):
        alt = family(theta0 + sign * step)
        for tag in (+1, -1):
            kl, err = expected_kl(table, alt, null, tag, **kl_options)
            values.append(kl)
            errors.append(err)
    scale = 1.0 / (2 * step ** 2)
    return float(sum(values) * scale), float(np.sqrt(sum(e ** 2 for e in errors)) * scale)
