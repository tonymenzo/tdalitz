"""Three-body Dalitz-plot kinematics.

Conventions
-----------
A spin-0 parent of mass ``M`` decays to three spin-0 daughters with masses
``(m1, m2, m3)``.  The two free coordinates are the invariant masses squared

    s12 = m^2(d1 d2),    s13 = m^2(d1 d3),

and the third follows from the exact constraint

    s12 + s13 + s23 = M^2 + m1^2 + m2^2 + m3^2.

Three-body phase space is flat in ``(s12, s13)``, so a uniform draw over the
physical region is phase-space distributed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FinalState", "kallen", "breakup_momentum"]


def kallen(x, y, z):
    """Kallen (triangle) function lambda(x, y, z)."""
    return x * x + y * y + z * z - 2.0 * (x * y + y * z + z * x)


def breakup_momentum(s, ma, mb):
    """Momentum of either daughter in the rest frame of a pair of mass^2 ``s``.

    Returns zero below threshold rather than raising, so the function is safe
    to evaluate on points outside the physical region.
    """
    s = np.asarray(s, dtype=float)
    lam = kallen(s, ma * ma, mb * mb)
    return np.sqrt(np.maximum(lam, 0.0)) / (2.0 * np.sqrt(np.maximum(s, 1e-12)))


@dataclass(frozen=True)
class FinalState:
    """A spin-0 parent decaying to three spin-0 daughters.

    Parameters
    ----------
    parent:
        Parent mass in GeV.
    daughters:
        Daughter masses ``(m1, m2, m3)`` in GeV.
    names:
        Optional daughter labels, used only for reporting.
    """

    parent: float
    daughters: tuple
    names: tuple = ("d1", "d2", "d3")

    @property
    def s_total(self) -> float:
        """``M^2 + m1^2 + m2^2 + m3^2``; the sum of the three invariants."""
        m1, m2, m3 = self.daughters
        return self.parent ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2

    def s23(self, s12, s13):
        """The remaining invariant, from the constraint.

        Written as ``s_total - (s12 + s13)`` so that it is bit-for-bit
        symmetric under ``s12 <-> s13`` (IEEE addition commutes).
        """
        return self.s_total - (np.asarray(s12) + np.asarray(s13))

    @property
    def bounds(self) -> tuple:
        """Rectangular hull ``(s12_lo, s12_hi, s13_lo, s13_hi)`` of the region."""
        m1, m2, m3 = self.daughters
        M = self.parent
        return ((m1 + m2) ** 2, (M - m3) ** 2, (m1 + m3) ** 2, (M - m2) ** 2)

    def inside(self, s12, s13) -> np.ndarray:
        """Boolean mask: are the points inside the physical Dalitz region?

        The test uses the daughter energies in the ``(12)`` rest frame, where a
        physical point requires a real scattering angle.
        """
        m1, m2, m3 = self.daughters
        M = self.parent
        s12 = np.asarray(s12, dtype=float)
        s13 = np.asarray(s13, dtype=float)

        lo12, hi12, lo13, hi13 = self.bounds
        ok = (s12 > lo12) & (s12 < hi12) & (s13 > lo13) & (s13 < hi13)

        with np.errstate(invalid="ignore", divide="ignore"):
            root = np.sqrt(np.where(s12 > 0.0, s12, np.nan))
            e2 = (s12 - m1 ** 2 + m2 ** 2) / (2.0 * root)
            e3 = (M ** 2 - s12 - m3 ** 2) / (2.0 * root)
            p2 = np.sqrt(np.maximum(e2 ** 2 - m2 ** 2, 0.0))
            p3 = np.sqrt(np.maximum(e3 ** 2 - m3 ** 2, 0.0))
            s23 = self.s23(s12, s13)
            hi = (e2 + e3) ** 2 - (p2 - p3) ** 2
            lo = (e2 + e3) ** 2 - (p2 + p3) ** 2
            within = (s23 >= lo) & (s23 <= hi)

        return ok & np.where(np.isfinite(within), within, False)

    def sample_phase_space(self, n: int, rng: np.random.Generator,
                           batch: int = 200_000, symmetric: bool = False) -> tuple:
        """Draw ``n`` points uniformly over the physical region (flat phase space).

        With ``symmetric=True`` only ``n / 2`` points are drawn and the second
        half of the pool is the first with ``s12 <-> s13``.  The pool is then
        closed under the exchange of daughters 2 and 3 -- the CP mirror of a
        final state such as ``KS pi+ pi-`` -- so a CP-symmetric model stays
        exactly CP symmetric on the pool.  Requires ``m2 == m3`` and even ``n``.
        """
        if symmetric:
            if self.daughters[1] != self.daughters[2]:
                raise ValueError("a symmetric pool needs equal masses for daughters 2 and 3")
            if n % 2:
                raise ValueError("a symmetric pool needs an even number of points")
            a, b = self.sample_phase_space(n // 2, rng, batch)
            return np.concatenate([a, b]), np.concatenate([b, a])

        lo12, hi12, lo13, hi13 = self.bounds
        out12, out13, have = [], [], 0
        while have < n:
            a = rng.uniform(lo12, hi12, batch)
            b = rng.uniform(lo13, hi13, batch)
            keep = self.inside(a, b)
            if np.any(keep):
                out12.append(a[keep])
                out13.append(b[keep])
                have += int(keep.sum())
        return np.concatenate(out12)[:n], np.concatenate(out13)[:n]

    def to_square(self, s12, s13) -> tuple:
        """Square-Dalitz coordinates ``(m', theta')`` in ``[0, 1]^2``.

        Built from the ``(23)`` pair mass and its helicity angle.  The mapping
        stretches the kinematic edges -- where resonances of an elongated Dalitz
        plot accumulate -- into the interior, which is what makes a transport or
        binning metric well conditioned.
        """
        m1, m2, m3 = self.daughters
        M = self.parent
        s23 = self.s23(s12, s13)
        m23 = np.sqrt(np.maximum(s23, 0.0))
        lo, hi = (m2 + m3), (M - m1)
        frac = (m23 - lo) / (hi - lo)
        m_prime = np.arccos(np.clip(2.0 * frac - 1.0, -1.0, 1.0)) / np.pi

        e1, e2, p1, p2 = self._helicity_frame(s23, m23)
        cos_hel = (m1 ** 2 + m2 ** 2 + 2.0 * e1 * e2 - np.asarray(s12)) / (2.0 * p1 * p2)
        theta_prime = np.arccos(np.clip(cos_hel, -1.0, 1.0)) / np.pi
        return m_prime, theta_prime

    def from_square(self, m_prime, theta_prime) -> tuple:
        """Inverse of :meth:`to_square`."""
        m1, m2, m3 = self.daughters
        M = self.parent
        lo, hi = (m2 + m3), (M - m1)
        m23 = lo + (hi - lo) * 0.5 * (1.0 + np.cos(np.pi * np.asarray(m_prime)))
        s23 = m23 ** 2
        cos_hel = np.cos(np.pi * np.asarray(theta_prime))
        e1, e2, p1, p2 = self._helicity_frame(s23, m23)
        s12 = m1 ** 2 + m2 ** 2 + 2.0 * (e1 * e2 - p1 * p2 * cos_hel)
        return s12, self.s_total - s23 - s12

    def _helicity_frame(self, s23, m23):
        """Energies/momenta of d1 and d2 in the ``(23)`` rest frame."""
        m1, m2, m3 = self.daughters
        M = self.parent
        e1 = (M ** 2 - s23 - m1 ** 2) / (2.0 * m23)
        e2 = (s23 + m2 ** 2 - m3 ** 2) / (2.0 * m23)
        p1 = np.sqrt(np.maximum(e1 ** 2 - m1 ** 2, 0.0))
        p2 = np.sqrt(np.maximum(e2 ** 2 - m2 ** 2, 0.0))
        return e1, e2, p1, p2
