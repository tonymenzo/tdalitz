"""Tests for the ready-to-run generation script, scripts/generate.py.

The real amplitude tables are not version-controlled, so each test writes a
small synthetic table that carries the channel's component names and the
K_S pi+ pi- kinematics.  That is enough to exercise everything the scripts do:
argument handling, generation for both tags, detector options and the output
file.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tdalitz import AmplitudeTable, FinalState

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

import generate                                         # noqa: E402
from tdalitz.channels import CHANNELS               # noqa: E402

PARENT = {"d0": 1.86484, "b0": 5.27966, "bs": 5.36692, "b0_3pi": 5.27966}
M_KS, M_PI, M_PI0 = 0.497611, 0.13957039, 0.1349768
DAUGHTERS = {"d0": (M_KS, M_PI, M_PI), "b0": (M_KS, M_PI, M_PI), "bs": (M_KS, M_PI, M_PI),
             "b0_3pi": (M_PI0, M_PI, M_PI)}
KEYS = ("d0", "b0", "bs", "b0_3pi")


def synthetic_table(path, key, n=4_000, seed=1):
    """Mirror-symmetric table with random smooth amplitudes for channel ``key``."""
    fs = FinalState(PARENT[key], DAUGHTERS[key], CHANNELS[key].model.daughters)
    s12, s13 = fs.sample_phase_space(n, np.random.default_rng(seed), symmetric=True)
    names = CHANNELS[key].model.component_names
    rng = np.random.default_rng(seed + 1)
    k = rng.normal(size=(len(names), 2)) + 1j * rng.normal(size=(len(names), 2))

    def amps(a, b):
        return np.column_stack([1.0 + c[0] * a / fs.parent**2 + c[1] * b / fs.parent**2 for c in k])

    AmplitudeTable(fs, s12, s13, tuple(names), amps(s12, s13), amps(s13, s12),
                   provenance="synthetic test table").save(path)
    return path


def run(key, tmp_path, *args):
    table = synthetic_table(tmp_path / f"{key}.npz", key)
    out = tmp_path / f"events_{key}.npz"
    generate.main([key, "--table", str(table), "--out", str(out), *args])
    with np.load(out, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


@pytest.mark.parametrize("key", KEYS)
def test_writes_both_tags(tmp_path, key):
    ev = run(key, tmp_path, "--n-events", "300", "--seed", "4")
    for name in ("s12", "s13", "tau", "tag", "true_tag", "index"):
        assert len(ev[name]) == 600, name
    assert np.sum(ev["true_tag"] == 1) == 300
    assert np.sum(ev["true_tag"] == -1) == 300
    np.testing.assert_array_equal(ev["tag"], ev["true_tag"])     # no mistag by default
    if CHANNELS[key].coherent:
        assert np.any(ev["tau"] < 0) and np.any(ev["tau"] > 0)   # Delta t of both signs
    else:
        assert np.all(ev["tau"] >= 0)                            # no resolution by default
    assert bool(ev["coherent"]) == CHANNELS[key].coherent
    assert str(ev["channel"]) == key


def test_settings_are_recorded(tmp_path):
    ev = run("b0", tmp_path, "--n-events", "100", "--seed", "9",
             "--sigma-tau", "0.05", "--mistag", "0.2")
    assert int(ev["seed"]) == 9
    assert int(ev["n_events_per_tag"]) == 100
    assert float(ev["sigma_tau"]) == 0.05
    assert float(ev["mistag"]) == 0.2
    mixing = CHANNELS["b0"].mixing
    assert float(ev["x"]) == mixing.x and float(ev["y"]) == mixing.y
    assert complex(ev["qp"]) == complex(mixing.qp)
    assert "synthetic test table" in str(ev["table_provenance"])


def test_same_seed_same_events(tmp_path):
    a = run("d0", tmp_path, "--n-events", "200", "--seed", "3")
    b = run("d0", tmp_path, "--n-events", "200", "--seed", "3")
    c = run("d0", tmp_path, "--n-events", "200", "--seed", "4")
    np.testing.assert_array_equal(a["tau"], b["tau"])
    assert not np.array_equal(a["tau"], c["tau"])


def test_detector_options(tmp_path):
    ev = run("bs", tmp_path, "--n-events", "2000", "--seed", "2",
             "--sigma-tau", "0.2", "--mistag", "0.3")
    assert np.any(ev["tau"] < 0)                                  # smeared times can be negative
    flipped = np.mean(ev["tag"] != ev["true_tag"])
    assert 0.25 < flipped < 0.35


def test_missing_table_explains_what_to_do(tmp_path, capsys):
    with pytest.raises(SystemExit):
        generate.main(["b0", "--table", str(tmp_path / "nope.npz")])
    err = capsys.readouterr().err
    assert "nope.npz" in err and "gh release download" in err


@pytest.mark.parametrize("key", KEYS)
def test_runs_as_a_script_without_install(tmp_path, key):
    """The scripts must work from any directory with no install and no PYTHONPATH."""
    table = synthetic_table(tmp_path / f"{key}.npz", key)
    out = tmp_path / "events.npz"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / "generate.py"), key,
         "--table", str(table), "--out", str(out), "--n-events", "50"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    with np.load(out, allow_pickle=False) as f:
        assert len(f["tau"]) == 100


def test_list_shows_every_channel(capsys):
    generate.main(["--list"])
    out = capsys.readouterr().out
    for key in KEYS:
        assert key in out


def test_channel_is_required(capsys):
    with pytest.raises(SystemExit):
        generate.main([])
    assert "channel" in capsys.readouterr().err


def test_unknown_channel_is_rejected(capsys):
    with pytest.raises(SystemExit):
        generate.main(["k0"])
    assert "k0" in capsys.readouterr().err


def test_output_directory_is_created(tmp_path):
    table = synthetic_table(tmp_path / "d0.npz", "d0")
    out = tmp_path / "toys" / "seed3" / "d0.npz"
    generate.main(["d0", "--table", str(table), "--out", str(out), "--n-events", "20"])
    assert out.exists()
