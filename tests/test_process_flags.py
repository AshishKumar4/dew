"""A run's XLA flags hold where XLA_FLAGS carried them before the process loaded jaxlib
(`dew._devices.flags_hold`): one holds where XLA would read its value; a process prepared with flags after
its backend opened is refused (`dew.training.prepare_process`), not appended to no effect, one prepared
with them before is warned they are unproven where jaxlib had loaded, and one whose flags already hold is
prepared again as often as its run asks (row 1901)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from dew.training import prepare_process

ROOT = Path(__file__).resolve().parents[1]
STATED = "--xla_dump_hlo_pass_re=fusion=all --xla_dump_hlo_as_text"
"""Flags no lane sets (tests/lane_environment.py), one valued with an '=' of its own."""


HOLDS = """
import json, sys
from dew._devices import flags_hold

print(json.dumps([flags_hold(flags) for flags in sys.argv[1:]]))
"""
"""A fresh process, its backend never opened, asking whether each set of flags holds."""


def test_a_flag_holds_as_xla_reads_it_its_last_occurrence_winning():
    """A flag holds where XLA would read the value asked of it: a later occurrence overrides an earlier
    one, in XLA_FLAGS and in the flags asked alike, a bare flag reads as true, and a value may hold
    an '='."""
    env = {**os.environ, "XLA_FLAGS": "--xla_dump_hlo_pass_re=x --xla_dump_hlo_as_text "
                                      "--xla_dump_hlo_pass_re=fusion=all",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    asked = ["--xla_dump_hlo_pass_re=x", STATED, "--xla_dump_hlo_pass_re=x --xla_dump_hlo_pass_re=fusion=all"]
    done = subprocess.run([sys.executable, "-c", HOLDS, *asked], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [False, True, True]


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
from dew._devices import xla_flag
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
    # jax came first, so the flags are unproven, and the process is told so, naming them, once however
    # often it is prepared.
    assert done.stderr.count(STATED) == 1


BY_HAND = """
import json, os, sys
import dew
from dew.training import prepare_process

flags = sys.argv[1]
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()
for _ in range(2):
    prepare_process(xla_flags=flags)
print(json.dumps("prepared"))
"""
"""A fresh process that writes the run's flags into XLA_FLAGS by hand after importing Dew, then prepares
with them twice, its backend opening in the first."""


def test_flags_written_by_hand_before_the_backend_opened_are_prepared_with_again_after_it():
    """Flags XLA_FLAGS carried while the backend was closed are what a backend that opened since may have
    read: a process prepared with them before its backend opened is prepared with them again after."""
    env = {**os.environ, "XLA_FLAGS": "",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", BY_HAND, STATED], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == "prepared"


SET_LATE = """
import json, os, sys
import dew
import jax
from dew.training import prepare_process

flags = sys.argv[1]
jax.devices()
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()
try:
    prepare_process(xla_flags=flags)
except ValueError:
    print(json.dumps("refused"))
else:
    print(json.dumps("prepared"))
"""
"""A fresh process whose backend opened without the run's flags, which then writes them into its
environment itself and prepares with them."""


def test_flags_written_into_the_environment_after_the_backend_opened_do_not_hold(tmp_path):
    """Flags written into the environment after the backend opened match the environment and nothing
    the backend runs with, so preparing with them is refused (row 1984). Before, the environment alone
    was read, and the run went on without them."""
    env = {**os.environ, "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", SET_LATE, STATED], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == "refused"


OPENED_ON = """
import json, os, sys
import dew
import jax
from dew._devices import flags_hold
from dew.training import prepare_process

imported, opened = sys.argv[1], sys.argv[2]
os.environ["XLA_FLAGS"] = opened
jax.devices()
held = [flags_hold(opened), flags_hold(imported)]
try:
    prepare_process(xla_flags=opened)
except ValueError:
    held.append("refused")
else:
    held.append("prepared")
print(json.dumps(held))
"""
"""A fresh process imported under one XLA_FLAGS whose backend opens under others, written by hand between
Dew's import, which loaded jaxlib, and the opening."""


def test_flags_written_by_hand_after_jaxlib_loaded_are_unproven():
    """XLA parses XLA_FLAGS once per runtime it loads, at its first use of its debug options, which can
    come before a backend opens: flags in the environment before the process loaded jaxlib hold, and flags
    written by hand after it are unproven, so a process prepared with them once its backend opened is
    refused (row 2047). Dew no longer watches JAX open its backend."""
    imported, opened = "--xla_dump_hlo_pass_re=imported", "--xla_dump_hlo_pass_re=opened"
    env = {**os.environ, "XLA_FLAGS": imported,
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", OPENED_ON, imported, opened], capture_output=True, text=True,
                          env=env, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [False, True, "refused"]


OWN_FLAGS = """
import json, sys
if sys.argv[1] == "jax first":
    import jax
import dew
from dew._devices import flags_hold

print(json.dumps(flags_hold("--xla_allow_excess_precision=false")))
"""
"""A fresh process importing Dew, after jax where asked."""


@pytest.mark.parametrize(("order", "held"), [("dew first", True), ("jax first", False)])
def test_dews_own_flags_hold_where_dew_is_imported_before_jax(order, held):
    """`import dew` writes its own flags into XLA_FLAGS before anything it imports loads jaxlib, so they hold
    for every runtime the process loads; where jax was imported first, they are unproven."""
    env = {**os.environ, "XLA_FLAGS": "",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", OWN_FLAGS, order], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) is held


LATE = """
import json, logging, sys
import dew
from dew._devices import flags_hold, xla_flag
from dew.training import prepare_process

logging.basicConfig(level=logging.WARNING)
prepare_process(xla_flags=sys.argv[1])
print(json.dumps([flags_hold(sys.argv[1]), xla_flag("xla_dump_hlo_pass_re")]))
"""
"""A fresh process importing Dew, which loads jaxlib, then preparing with the run's flags before its
backend opens: a library caller's late flags."""


def test_a_library_caller_preparing_with_flags_after_jaxlib_loaded_is_told_they_are_unproven():
    """Flags a process is prepared with after it loaded jaxlib and before its backend opened are written
    into XLA_FLAGS, as XLA may still read them, and the process is warned that they are unproven, where
    setting XLA_FLAGS before the process starts would make them certain."""
    env = {**os.environ, "XLA_FLAGS": "",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", LATE, STATED], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [False, "fusion=all"]
    # The warning names the flags it cannot prove, once.
    assert done.stderr.count(STATED) == 1


HELD = """
import json, sys
import dew
from dew._devices import flags_hold, hold_xla_flags

hold_xla_flags(sys.argv[1])
print(json.dumps(flags_hold(sys.argv[1])))
"""
"""A fresh process importing Dew, which loads jaxlib, then asking for the run's flags to hold, as Dew's
command line does before it trains a run."""


def test_a_command_lines_flags_hold_as_it_runs_again_with_them_set_before_jaxlib_loads():
    """`hold_xla_flags` runs the process again from its start with the flags in XLA_FLAGS where they do not
    hold, as `dew train` does with a run's `trainer.xla_flags`: the process that goes on holds them."""
    env = {**os.environ, "XLA_FLAGS": "",
           "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", HELD, STATED], capture_output=True, text=True, env=env,
                          cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines() == ["true"]
