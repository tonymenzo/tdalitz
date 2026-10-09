"""Neutral-meson mixing and the time-dependent tagged decay rate.

Time is measured in lifetimes, ``tau = Gamma * t``, and mixing is parameterized
by

    x = Delta m / Gamma,        y = Delta Gamma / (2 Gamma),

together with the complex ratio ``q/p``.  A state born as a flavour eigenstate
evolves as

    |P0(tau)>    = g_plus |P0> + (q/p)      g_minus |P0bar>,
    |P0bar(tau)> = g_plus |P0bar> + (p/q)   g_minus |P0>,

with (dropping a common ``exp(-tau/2)`` that is absorbed into the exponential
proposal used when generating)

    g_plus(tau)  =  cosh( (y - i x) tau / 2 ),
    g_minus(tau) = -sinh( (y - i x) tau / 2 ).

For ``y = 0`` these reduce to ``cos(x tau / 2)`` and ``i sin(x tau / 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MixingParameters", "g_factors", "tagged_rate"]


@dataclass(frozen=True)
class MixingParameters:
    """Mixing and CP-violation parameters of the neutral meson.

    Parameters
    ----------
    x:
        ``Delta m / Gamma``.  Sets how many oscillations occur per lifetime.
    y:
        ``Delta Gamma / (2 Gamma)``.  Non-zero for charm and for ``Bs``.
    qp:
        The ratio ``q/p``.  ``abs(qp) != 1`` is CP violation in mixing;
        ``arg(qp) != 0`` drives mixing-induced CP violation.
    """

    x: float
    y: float = 0.0
    qp: complex = 1.0 + 0.0j

    @property
    def is_cp_conserving_mixing(self) -> bool:
        return np.isclose(abs(self.qp), 1.0) and np.isclose(np.angle(self.qp), 0.0)


def g_factors(mixing: MixingParameters, tau) -> tuple:
    """Return ``(g_plus, g_minus)`` at the given proper times."""
    w = 0.5 * (mixing.y - 1j * mixing.x) * np.asarray(tau, dtype=float)
    return np.cosh(w), -np.sinh(w)


def tagged_rate(amp, amp_bar, mixing: MixingParameters, tau, tag: int) -> np.ndarray:
    """Time-dependent decay rate of a tagged meson, up to ``exp(-tau)``.

    Parameters
    ----------
    amp, amp_bar:
        Complex decay amplitudes ``A`` and ``Abar`` at the Dalitz points.
    mixing:
        Mixing parameters.
    tau:
        Proper times in lifetimes, broadcastable against ``amp``.
    tag:
        ``+1`` for a meson at production, ``-1`` for an antimeson.

    Returns
    -------
    The squared time-evolved amplitude.  The common ``exp(-tau)`` factor is
    omitted, since generation proposes ``tau`` from an exponential.
    """
    if tag not in (+1, -1):
        raise ValueError("tag must be +1 or -1")
    g_p, g_m = g_factors(mixing, tau)
    if tag == +1:
        evolved = amp * g_p + mixing.qp * amp_bar * g_m
    else:
        evolved = amp_bar * g_p + (1.0 / mixing.qp) * amp * g_m
    return np.abs(evolved) ** 2
