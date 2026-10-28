"""A run's XLA flags take effect as JAX opens its backend (`dew.devices.flags_hold`): one holds where
XLA would read its value; a process prepared with flags after its backend opened is refused
(`dew.training.prepare_process`), not appended to no effect, and one whose flags already hold is
prepared again as often as its run asks (row 1901)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from dew.devices import flags_hold
from dew.training import prepare_process

ROOT = Path(__file__).resolve().parents[1]
STATED = "--xla_dump_hlo_pass_re=fusion=all --xla_dump_hlo_as_text"
"""Flags no lane sets (tests/lane_environment.py), one valued with an '=' of its own."""


def test_a_flag_holds_as_xla_reads_it_its_last_occurrence_winning(monkeypatch):
    """A flag holds where XLA would read the value asked of it: a later occurrence overrides an earlier
    one, in XLA_FLAGS and in the flags asked alike, a bare flag reads as true, and a value may hold
    an '='."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_pass_re=x --xla_dump_hlo_as_text "
                                    "--xla_dump_hlo_pass_re=fusion=all")
    assert not flags_hold("--xla_dump_hlo_pass_re=x")
    assert flags_hold(STATED)
    assert flags_hold("--xla_dump_hlo_pass_re=x --xla_dump_hlo_pass_re=fusion=all")


def test_flags_that_do_not_hold_after_the_backend_opened_are_refused_where_the_process_is_prepared(
        monkeypatch):
    """Once JAX's backend opened, preparing the process with flags that do not hold is refused, not
    appended to no effect."""
    jax.devices()
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_as_text")
    with pytest.raises(ValueError):
        prepare_process(xla_flags=STATED)
    assert os.environ["XLA_FLAGS"] == "--xla_dump_hlo_as_text"


PREPARED = """
import json, sys
import jax
from dew.devices import xla_flag
from dew.training import prepare_process

flags, cache = sys.argv[1], sys.argv[2]
jax.devices()
for _ in range(2):
    prepare_process(xla_flags=flags, compilation_cache_dir=cache)
print(json.dumps({"pass": xla_flag("xla_dump_hlo_pass_re"), "text": xla_flag("xla_dump_hlo_as_text"),
                  "cache": jax.config.jax_compilation_cache_dir}))
"""
"""A fresh process started with the run's flags, its backend opened, prepared twice with them and a
compilation cache, as a recipe and its `run()` each prepare it: what XLA reads, and the cache."""


def test_a_process_started_with_its_runs_flags_is_prepared_again_after_its_backend_opened(tmp_path):
    """A process whose flags were set before it started carries them: prepared with them twice after
    its backend opened, as a recipe and then its run prepare it, it is refused neither time, and the
    flags and the cache directory it keeps are the run's."""
    cache = str(tmp_path / "cache")
    env = {**os.environ, "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} {STATED}".strip(),
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", PREPARED, STATED, cache], capture_output=True, text=True,
                          env=env, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    kept = json.loads(done.stdout.strip().splitlines()[-1])
    assert kept == {"pass": "fusion=all", "text": "true", "cache": cache}
