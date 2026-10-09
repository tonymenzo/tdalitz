# Making amplitude tables with Laura++

`laura.py` evaluates the components of a Laura++ isobar model on a phase-space pool and writes an amplitude table.
It is the only part of `tdalitz` that needs ROOT and Laura++.
Users who only generate events never need it; the reference tables are attached to the releases.

## Requirements

* ROOT with PyROOT
* a Laura++ build; pass the path to `libLaura++` as `library=`

## Example

```python
from tdalitz.exporters.laura import LauraModel, export

model = LauraModel(
    parent="D0", daughters=("K_S0", "pi+", "pi-"),
    resonances=[("K*-(892)", 2), ("rho0(770)", 1), ("K*-_0(1430)", 2)],
    barrier_radii={"Parent": 5.0, "Kstar": 1.5, "Light": 1.5},
    library="/path/to/libLaura++",
)
table = export(model, n_points=1_000_000, seed=1)
table.save("d0_kspipi.npz")
```

Particle and resonance names are Laura++'s.
Each resonance is given as `(name, bachelor)`, where the bachelor is the index (1, 2 or 3) of the daughter not produced by the resonance; `0` is used for non-resonant terms.
Optional fields:

| field | contents |
| --- | --- |
| `lineshapes` | lineshape per component, e.g. `{"rho0(770)": "GS", "f_0(980)": "Flatte"}` (default: relativistic Breit-Wigner) |
| `barrier_radii` | Blatt-Weisskopf radii in GeV⁻¹ per Laura++ category (`"Parent"`, `"Kstar"`, `"Light"`, ...) |
| `mass_width_overrides` | `{name: (mass, width, spin)}` to replace the Laura++ database values |

## What the exporter does

Each component is evaluated alone with a unit coefficient, so each column of the table is the bare $F_r$: lineshape, Blatt-Weisskopf barriers, angular factor and Laura++'s normalization $\int |F_r|^2\, ds_{12}\, ds_{13} = 1$.

$\bar F_r$ is evaluated at the mirrored point $(s_{13}, s_{12})$, which is correct when CP exchanges daughters 2 and 3 (as for $K_S^0\pi^+\pi^-$).
The pool is then mirror-symmetric, so a CP-conserving model is exactly CP-symmetric on it.
For other final states, pass `mirror=False, conjugate_model=...` with a model of the CP-conjugate decay.

A 1M-point table takes about a minute.

## Reference tables

The three reference tables were made with

```bash
TDALITZ_LAURA_LIB=/path/to/libLaura++ python scripts/export_tables.py
```

from the models in `src/tdalitz/channels.py`; the script reads the library path from `TDALITZ_LAURA_LIB`.

## Things to know

* Laura++ writes `integ.dat` into the working directory; it can be deleted.
* Barrier radii set per category override the Laura++ database per-resonance values. `LauResonanceMaker` is a process-wide singleton, so a category left out inherits the value from an earlier export in the same process.
