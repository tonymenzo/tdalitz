"""tdalitz -- time-dependent event generation from tabulated Dalitz amplitudes.

The package separates the two halves of a time-dependent Dalitz-plot
simulation:

* an **amplitude table** (:class:`~tdalitz.amplitude.AmplitudeTable`) holding
  the per-component complex amplitudes of an isobar model, produced once by
  whatever amplitude code you trust, and
* a **pure-NumPy core** that applies coefficients, neutral-meson mixing,
  tagging and detector effects to generate time-dependent tagged events.

Only the exporters depend on an external amplitude package; everything else
needs NumPy alone.
"""

from .amplitude import AmplitudeTable, Coefficient
from .detector import apply_mistag, apply_time_resolution, dilution_factor
from .generate import Events, generate_pair, generate_tagged
from .kinematics import FinalState
from . import likelihood  # noqa: F401  (tdalitz.likelihood)
from .mixing import MixingParameters, g_factors, tagged_rate

__version__ = "0.1.0"

__all__ = [
    "AmplitudeTable", "Coefficient", "FinalState",
    "MixingParameters", "g_factors", "tagged_rate",
    "Events", "generate_tagged", "generate_pair",
    "apply_time_resolution", "apply_mistag", "dilution_factor",
]
