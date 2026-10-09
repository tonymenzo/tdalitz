"""Export the amplitude tables of the paper channels from Laura++.

Needs ROOT and a Laura++ build; set TDALITZ_LAURA_LIB to the library path.
Tables are written to tables/<key>.npz (not version-controlled).
"""

import os
import pathlib
import sys
import time

from dataclasses import replace

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tdalitz.exporters.laura import export          # noqa: E402
from tdalitz.channels import CHANNELS               # noqa: E402

N_POINTS = int(os.environ.get("TDALITZ_POOL", 1_000_000))


def main():
    lib = os.environ["TDALITZ_LAURA_LIB"]
    out = REPO / "tables"
    out.mkdir(exist_ok=True)
    for key, ch in CHANNELS.items():
        t0 = time.perf_counter()
        print(f"[{key}] exporting {len(ch.model.resonances)} components on "
              f"{N_POINTS:,} points", flush=True)
        table = export(replace(ch.model, library=lib), n_points=N_POINTS, seed=2026)
        path = table.save(out / f"{key}.npz")
        print(f"[{key}] wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB) "
              f"in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
