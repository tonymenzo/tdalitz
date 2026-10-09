"""Time-dependent tagged event generation from an amplitude table.

The joint density of a tagged event is

    dGamma_tag / d(PS) d tau  ~  exp(-tau) * |A g_plus + (q/p)^tag Abar g_minus|^2,

flat in the Dalitz variables apart from the amplitude.  Because the amplitude
table already holds points drawn uniformly over the physical region, generation
reduces to accept/reject over (pool index, proper time) pairs.

The rejection envelope is exact, not scanned.  Since
``|g_plus|, |g_minus| <= cosh(y tau / 2)``, the squared time-evolved amplitude
obeys

    R(tau) <= cosh^2(y tau / 2) * (|A| + |q/p|^tag |Abar|)^2,

and ``cosh^2(y tau / 2) exp(-|y| tau) <= 1``.  Proposing ``tau`` from
``Exp(1 - |y|)`` and accepting with probability ``R(tau) exp(-|y| tau) / C``,
with ``C = max over the pool of (|A| + |q/p|^tag |Abar|)^2``, therefore samples
the target density exactly for every ``tau``, with no clipping and no
interpolation of the amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amplitude import AmplitudeTable
from .mixing import MixingParameters, tagged_rate

__all__ = ["Events", "generate_tagged", "generate_pair", "rate_ceiling"]


@dataclass
class Events:
    """A generated sample of tagged decays.

    Attributes
    ----------
    s12, s13:
        Dalitz coordinates.
    tau:
        Proper decay time in lifetimes.  This is the *true* time; detector
        effects are applied separately (see :mod:`tdalitz.detector`).
    tag:
        Production flavour, ``+1`` or ``-1``, one entry per event.
    index:
        Row of the amplitude table each event was drawn from, or ``None`` for
        events not generated from a table.  It lets exact per-event densities
        be evaluated (see :mod:`tdalitz.likelihood`).
    coherent:
        ``True`` if ``tau`` is the decay-time difference of a coherent pair
        (B-factory mode, see :func:`generate_tagged`), which takes both signs.
    """

    s12: np.ndarray
    s13: np.ndarray
    tau: np.ndarray
    tag: np.ndarray
    index: np.ndarray = None
    coherent: bool = False

    def __len__(self) -> int:
        return len(self.tau)

    def __getitem__(self, key) -> "Events":
        index = None if self.index is None else self.index[key]
        return Events(self.s12[key], self.s13[key], self.tau[key], self.tag[key], index,
                      self.coherent)

    def as_dict(self) -> dict:
        out = {"s12": self.s12, "s13": self.s13, "tau": self.tau, "tag": self.tag}
        if self.index is not None:
            out["index"] = self.index
        return out


def rate_ceiling(amp, amp_bar, mixing: MixingParameters, tag: int) -> float:
    """Exact bound ``C`` with ``R(tau) * exp(-|y| tau) <= C`` for all ``tau >= 0``.

    ``R`` is :func:`~tdalitz.mixing.tagged_rate` and the bound holds at every
    point of the pool; see the module docstring for the derivation.
    """
    if tag not in (+1, -1):
        raise ValueError("tag must be +1 or -1")
    r = abs(mixing.qp)
    if tag == +1:
        bound = np.abs(amp) + r * np.abs(amp_bar)
    else:
        bound = np.abs(amp_bar) + np.abs(amp) / r
    return float(np.max(bound) ** 2) + 1e-300


def generate_tagged(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                    tag: int, n_events: int, rng: np.random.Generator,
                    batch: int = 100_000, coherent: bool = False) -> Events:
    """Generate ``n_events`` decays of a meson tagged as ``tag`` at production.

    Parameters
    ----------
    table:
        Amplitude table supplying the phase-space pool and components.
    coefficients:
        Mapping ``{component name: Coefficient}``.
    mixing:
        Mixing parameters (``x``, ``y``, ``q/p``).
    tag:
        ``+1`` (meson) or ``-1`` (antimeson) at production.
    n_events:
        Number of events to return.
    rng:
        NumPy random generator, for reproducibility.
    coherent:
        Generate the decay-time difference ``Delta t`` of a coherent C-odd
        pair (B factories) instead of the decay time from production.  The
        density is ``exp(-|Delta t|) R_tag(Delta t)`` for ``Delta t`` of either
        sign, where ``tag`` is the flavour of the signal meson at
        ``Delta t = 0``, i.e. opposite to the flavour of the tag-side meson.
        Returned in ``Events.tau``.

    Notes
    -----
    Events are drawn from the tabulated pool, so the pool should be much larger
    than the requested sample to keep repeats negligible.
    """
    y = abs(mixing.y)
    if y >= 1.0:
        raise ValueError("|y| must be below 1 for a normalizable decay-time density")
    amp, amp_bar = table.total(coefficients)
    ceiling = rate_ceiling(amp, amp_bar, mixing, tag)
    n_pool = len(table)

    out_idx, out_tau, have = [], [], 0
    while have < n_events:
        idx = rng.integers(0, n_pool, batch)
        tau = rng.exponential(1.0 / (1.0 - y), batch)
        if coherent:
            # |g_pm(dt)| <= cosh(y dt / 2) holds for either sign of dt, so the
            # same envelope bounds the two-sided density.
            tau = tau * rng.choice((-1.0, 1.0), batch)
        weight = tagged_rate(amp[idx], amp_bar[idx], mixing, tau, tag) * np.exp(-y * np.abs(tau))
        keep = rng.random(batch) * ceiling < weight
        if np.any(keep):
            out_idx.append(idx[keep])
            out_tau.append(tau[keep])
            have += int(keep.sum())

    idx = np.concatenate(out_idx)[:n_events]
    tau = np.concatenate(out_tau)[:n_events]
    return Events(
        s12=table.s12[idx], s13=table.s13[idx], tau=tau,
        tag=np.full(n_events, tag, dtype=np.int8), index=idx, coherent=coherent,
    )


def generate_pair(table: AmplitudeTable, coefficients, mixing: MixingParameters,
                  n_events: int, rng: np.random.Generator, coherent: bool = False) -> tuple:
    """Convenience wrapper returning ``(meson, antimeson)`` samples of equal size."""
    plus = generate_tagged(table, coefficients, mixing, +1, n_events, rng, coherent=coherent)
    minus = generate_tagged(table, coefficients, mixing, -1, n_events, rng, coherent=coherent)
    return plus, minus
