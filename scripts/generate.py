"""Generate time-dependent, flavour-tagged events for a reference channel.

    python scripts/generate.py b0                         # 10k events per tag
    python scripts/generate.py b0 --n-events 5000 --seed 1 --out b0.npz
    python scripts/generate.py b0 --sigma-tau 0.05 --mistag 0.2
    python scripts/generate.py --list                     # available channels
    python scripts/generate.py --help

No install needed: only NumPy and the amplitude table of the channel
(tables/<channel>.npz). The models, coefficients and mixing parameters
are defined in src/tdalitz/channels.py.

The output .npz holds both tags, one entry per event:

    s12, s13   m^2(h pi+), m^2(h pi-) in GeV^2, h = KS (or pi0 for b0_3pi)
    tau        decay time in lifetimes (smeared if --sigma-tau > 0); for the
               coherent channel b0_3pi, the decay-time difference Delta t of
               the B pair, which takes both signs
    tag        production tag as measured: +1 = P0, -1 = P0bar (after --mistag);
               for b0_3pi, the signal flavour at Delta t = 0 (opposite to the
               tag-side B)
    true_tag   production tag before mistag
    index      row of the amplitude table each event came from

plus the settings used (channel, coherent, seed, n_events_per_tag, sigma_tau,
mistag, x, y, qp, table_provenance).
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                   # noqa: E402

from tdalitz import (AmplitudeTable, apply_mistag,                   # noqa: E402
                     apply_time_resolution, generate_pair)
from tdalitz.channels import CHANNELS                                # noqa: E402

TABLES = REPO / "tables"

#: Channel descriptions shown by --list.
DESCRIPTION = {
    "d0": "D0 -> KS pi+ pi-   6 components of the BaBar 2010 model; SM charm mixing, "
          "CP conserved (the null case)",
    "b0": "B0 -> KS pi+ pi-   9 components, BaBar/Belle content, strong phases assumed; "
          "x = 0.77, q/p = exp(-2i beta), sin 2beta = 0.709",
    "bs": "Bs0 -> KS pi+ pi-  6 illustrative components; x = 27, y = 0.063, SM q/p = exp(0.037i)",
    "b0_3pi": "B0 -> pi+ pi- pi0  BaBar 2013 rho pi amplitudes, direct + mixing-induced CPV; "
              "coherent Delta t (B factory)",
}


def parse(argv):
    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate time-dependent, flavour-tagged events for both production tags.")
    p.add_argument("channel", nargs="?", choices=sorted(CHANNELS),
                   help="reference channel (see --list)")
    p.add_argument("--list", action="store_true", help="list the channels and exit")
    p.add_argument("--n-events", type=int, default=10_000,
                   help="events per tag (default: 10000)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--sigma-tau", type=float, default=0.0,
                   help="Gaussian decay-time resolution, in lifetimes (default: 0, true time)")
    p.add_argument("--mistag", type=float, default=0.0,
                   help="probability of flipping each tag (default: 0)")
    p.add_argument("--table", help="amplitude table (default: tables/<channel>.npz)")
    p.add_argument("--out", help="output file (default: events_<channel>.npz)")
    args = p.parse_args(argv)
    if not args.list and args.channel is None:
        p.error("a channel is required (one of: " + ", ".join(sorted(CHANNELS)) + ")")
    return args


def list_channels():
    for key in CHANNELS:
        found = "table found" if (TABLES / f"{key}.npz").exists() else "table missing"
        print(f"{key}  {DESCRIPTION.get(key, CHANNELS[key].title)}  [{found}]")


def main(argv=None):
    args = parse(argv)
    if args.list:
        list_channels()
        return None

    key = args.channel
    table_path = Path(args.table) if args.table else TABLES / f"{key}.npz"
    if not table_path.exists():
        print(f"amplitude table not found: {table_path}\n"
              "Download the reference tables from the latest release:\n"
              "    gh release download --pattern '*.npz' --dir tables\n"
              "(see the README).", file=sys.stderr)
        sys.exit(1)

    ch = CHANNELS[key]
    table = AmplitudeTable.load(table_path)
    rng = np.random.default_rng(args.seed)

    plus, minus = generate_pair(table, ch.coefficients, ch.mixing, args.n_events, rng,
                                coherent=ch.coherent)
    true_tag = np.concatenate([plus.tag, minus.tag])
    events = [apply_mistag(apply_time_resolution(ev, args.sigma_tau, rng), args.mistag, rng)
              for ev in (plus, minus)]

    out = Path(args.out) if args.out else Path(f"events_{key}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        s12=np.concatenate([e.s12 for e in events]),
        s13=np.concatenate([e.s13 for e in events]),
        tau=np.concatenate([e.tau for e in events]),
        tag=np.concatenate([e.tag for e in events]),
        true_tag=true_tag,
        index=np.concatenate([e.index for e in events]),
        channel=key, coherent=ch.coherent, seed=args.seed, n_events_per_tag=args.n_events,
        sigma_tau=args.sigma_tau, mistag=args.mistag,
        x=ch.mixing.x, y=ch.mixing.y, qp=complex(ch.mixing.qp),
        table_provenance=table.provenance,
    )
    print(f"{key}: wrote {2 * args.n_events} events ({args.n_events} per tag) to {out}")
    return out


if __name__ == "__main__":
    main()
