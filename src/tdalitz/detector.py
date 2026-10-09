"""Detector effects: decay-time resolution and flavour mistag.

These are the two effects that matter most for a time-dependent Dalitz
measurement, and both act only on the generated events -- never on the
amplitude -- so they are applied after generation.

Decay-time resolution matters enormously at large ``x``: smearing with a
Gaussian of width ``sigma_tau`` damps an oscillation of angular frequency ``x``
by the factor ``exp(-x^2 sigma_tau^2 / 2)``, so a fast-oscillating ``Bs`` sample
loses most of its mixing information unless the resolution is excellent.
Mistagging dilutes any tag-odd asymmetry by ``D = 1 - 2 omega``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .generate import Events

__all__ = ["apply_time_resolution", "apply_mistag", "dilution_factor"]


def dilution_factor(x: float, sigma_tau: float) -> float:
    """Analytic damping ``exp(-x^2 sigma_tau^2 / 2)`` of an ``x``-oscillation.

    Useful as a cross-check on a smeared sample, and for judging in advance
    whether a channel's oscillation survives a given resolution.
    """
    return float(np.exp(-0.5 * (x * sigma_tau) ** 2))


def apply_time_resolution(events: Events, sigma_tau: float,
                          rng: np.random.Generator,
                          allow_negative: bool = True) -> Events:
    """Smear the decay time with a Gaussian of width ``sigma_tau`` (lifetimes).

    Parameters
    ----------
    sigma_tau:
        Resolution in units of the lifetime, i.e. ``sigma_t * Gamma``.
    allow_negative:
        Keep negative reconstructed times, as a real measurement would.  Set
        ``False`` to discard them (this biases the sample and is offered only
        for quick comparisons).
    """
    if sigma_tau < 0.0:
        raise ValueError("sigma_tau must be non-negative")
    if events.coherent and not allow_negative:
        raise ValueError("coherent events have Delta t of both signs; negative values are physical")
    if sigma_tau == 0.0:
        return events
    smeared = events.tau + rng.normal(0.0, sigma_tau, len(events))
    out = replace(events, tau=smeared)
    if not allow_negative:
        out = out[smeared > 0.0]
    return out


def apply_mistag(events: Events, omega: float, rng: np.random.Generator) -> Events:
    """Flip the production tag of a random fraction ``omega`` of events.

    The resulting tag-odd asymmetry is diluted by ``D = 1 - 2 * omega``.
    """
    if not 0.0 <= omega <= 0.5:
        raise ValueError("omega must lie in [0, 0.5]")
    if omega == 0.0:
        return events
    flip = rng.random(len(events)) < omega
    tag = np.where(flip, -events.tag, events.tag).astype(np.int8)
    return replace(events, tag=tag)
