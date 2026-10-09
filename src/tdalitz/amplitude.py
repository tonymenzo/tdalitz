"""The amplitude interchange format.

An :class:`AmplitudeTable` stores the *per-component* complex decay amplitudes
of an isobar (or any other) model, tabulated on a pool of phase-space points:

    F_r(s12, s13)        for the particle,
    Fbar_r(s12, s13)     for the CP-conjugate decay.

Storing components separately -- rather than one summed amplitude -- is what
makes the table reusable: the physical amplitudes

    A    = sum_r  c_r     F_r,
    Abar = sum_r  cbar_r  Fbar_r,

are rebuilt at run time from :class:`Coefficient` values, so scanning
magnitudes, strong phases or weak phases costs a re-weighted sum in NumPy and
never requires re-running the amplitude code that produced the table.

The format is deliberately plain (an ``.npz`` of arrays plus metadata), so a
table can be produced by any amplitude package.  A Laura++ exporter ships with
this library; see :mod:`tdalitz.exporters.laura`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kinematics import FinalState

__all__ = ["Coefficient", "AmplitudeTable"]


@dataclass(frozen=True)
class Coefficient:
    """Complex isobar coefficient, split into CP-even and CP-odd phases.

    The particle and antiparticle coefficients are

        c     = magnitude * exp(i * (phase + weak)),
        cbar  = magnitude * exp(i * (phase - weak)),

    so a non-zero ``weak`` phase introduces direct CP violation localized on
    this component, while ``phase`` (the strong phase) is CP-even.

    This form keeps ``|c| = |cbar|``.  A component with a rate asymmetry of
    its own (e.g. from interfering tree and penguin amplitudes) needs
    independent values: use :meth:`from_complex`.
    """

    magnitude: float
    phase: float = 0.0
    weak: float = 0.0
    value: complex = None        # explicit c, set by from_complex
    value_bar: complex = None    # explicit cbar, set by from_complex

    @classmethod
    def from_complex(cls, c, c_bar) -> "Coefficient":
        """A coefficient with independent particle and antiparticle values."""
        c, c_bar = complex(c), complex(c_bar)
        return cls(abs(c), float(np.angle(c)), 0.0, value=c, value_bar=c_bar)

    @property
    def c(self) -> complex:
        if self.value is not None:
            return self.value
        return self.magnitude * np.exp(1j * (self.phase + self.weak))

    @property
    def c_bar(self) -> complex:
        if self.value_bar is not None:
            return self.value_bar
        return self.magnitude * np.exp(1j * (self.phase - self.weak))


@dataclass
class AmplitudeTable:
    """Per-component complex amplitudes on a pool of phase-space points.

    Parameters
    ----------
    final_state:
        Parent and daughter masses defining the Dalitz plot.
    s12, s13:
        Phase-space points, shape ``(N,)``.  Drawn uniformly over the physical
        region, so the pool carries flat phase-space weight.
    components:
        Component names, length ``R``.
    amp, amp_bar:
        Complex arrays of shape ``(N, R)`` holding ``F_r`` and ``Fbar_r``.
    provenance:
        Free-form note recording how the table was produced.
    """

    final_state: FinalState
    s12: np.ndarray
    s13: np.ndarray
    components: tuple
    amp: np.ndarray
    amp_bar: np.ndarray
    provenance: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n, r = len(self.s12), len(self.components)
        for name, arr in (("amp", self.amp), ("amp_bar", self.amp_bar)):
            if arr.shape != (n, r):
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {(n, r)} "
                    f"({n} points x {r} components)"
                )
            if not np.iscomplexobj(arr):
                raise TypeError(f"{name} must be complex")

    def __len__(self) -> int:
        return len(self.s12)

    def index(self, name: str) -> int:
        """Column index of a named component."""
        return self.components.index(name)

    def total(self, coefficients) -> tuple:
        """Build ``(A, Abar)`` on the pool from a mapping of coefficients.

        Parameters
        ----------
        coefficients:
            Mapping ``{component name: Coefficient}``.  Components absent from
            the mapping are omitted from the sum (coefficient zero).
        """
        unknown = set(coefficients) - set(self.components)
        if unknown:
            raise KeyError(f"unknown components: {sorted(unknown)}")
        a = np.zeros(len(self), dtype=complex)
        a_bar = np.zeros(len(self), dtype=complex)
        for name, coeff in coefficients.items():
            col = self.index(name)
            a += coeff.c * self.amp[:, col]
            a_bar += coeff.c_bar * self.amp_bar[:, col]
        return a, a_bar

    def fit_fractions(self, coefficients) -> dict:
        """Fit fractions ``int |c_r F_r|^2 / int |A|^2`` over the pool.

        The pool is uniform in phase space, so the integrals are plain means.
        Fractions do not sum to one; the deficit is the net interference.
        """
        a, _ = self.total(coefficients)
        denom = float(np.mean(np.abs(a) ** 2))
        return {
            name: float(np.mean(np.abs(coeff.c * self.amp[:, self.index(name)]) ** 2)) / denom
            for name, coeff in coefficients.items()
        }

    def save(self, path) -> Path:
        """Write the table to a compressed ``.npz`` file.

        Every entry is a plain numeric or string array, so the file can be read
        with ``allow_pickle=False``: loading a table never executes code.
        Metadata is stored as JSON and must therefore be JSON-serializable.
        """
        path = Path(path)
        m1, m2, m3 = self.final_state.daughters
        np.savez_compressed(
            path,
            s12=self.s12, s13=self.s13,
            amp=self.amp, amp_bar=self.amp_bar,
            components=np.array(self.components, dtype=str),
            parent=self.final_state.parent,
            daughters=np.array([m1, m2, m3]),
            names=np.array(self.final_state.names, dtype=str),
            provenance=np.array(self.provenance, dtype=str),
            metadata=np.array(json.dumps(self.metadata), dtype=str),
        )
        return path

    @classmethod
    def load(cls, path) -> "AmplitudeTable":
        """Read a table written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as d:
            fs = FinalState(
                parent=float(d["parent"]),
                daughters=tuple(float(x) for x in d["daughters"]),
                names=tuple(str(x) for x in d["names"]),
            )
            return cls(
                final_state=fs,
                s12=d["s12"], s13=d["s13"],
                components=tuple(str(x) for x in d["components"]),
                amp=d["amp"], amp_bar=d["amp_bar"],
                provenance=str(d["provenance"]),
                metadata=json.loads(str(d["metadata"])),
            )
