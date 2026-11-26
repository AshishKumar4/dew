"""A run's native fault record (row 6497, `dew.training.faults`): a fatal signal in a fit leaves every
thread's Python stack in its run directory."""

import faulthandler
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from affine_run import batches
from affine_training import frozen_trainer

from dew.data.dataset import Loader

CRASHING_FIT = """
import ctypes
import faulthandler
import json
import os
import signal
import sys

import jax
import optax
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Checkpoints, Trainer

mode = sys.argv[2]
seen = {}


def replaced():
    # A native handler installed over the record's for SIGSEGV during the fit (another library's): libc's
    # abort, so a signal it takes ends the process by SIGABRT.
    libc = ctypes.CDLL(None)
    action = ctypes.create_string_buffer(256)
    ctypes.memmove(action, ctypes.byref(ctypes.c_void_p(ctypes.cast(libc.abort, ctypes.c_void_p).value)),
                   ctypes.sizeof(ctypes.c_void_p))
    assert libc.sigaction(int(signal.SIGSEGV), action, None) == 0


def crashing_batches(partition):
    for index, batch in enumerate(batches()):
        if index == 0:
            seen["ucx during the fit"] = os.environ.get("UCX_ERROR_SIGNALS")
            if mode == "replaced":
                replaced()
        if index == 12 and mode == "plain":
            faulthandler._sigsegv()
        yield batch


Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), checkpoints=Checkpoints(sys.argv[1])).fit(
    Loader(crashing_batches, None, None, 8), steps=16, log_every=1)
seen["enabled after the fit"] = faulthandler.is_enabled()
print(json.dumps(seen), flush=True)
if mode == "replaced":
    faulthandler._sigsegv()
"""


def _fit(tmp_path: Path, mode: str, ucx: str | None = None, *,
         held: str | None = None) -> tuple[subprocess.CompletedProcess, str]:
    """A fit of `CRASHING_FIT` in a process of its own (no fault handler enabled before it,
    `UCX_ERROR_SIGNALS` `ucx` where given, else unset), in `mode`: 'none' crashing nowhere, 'plain' faulting
    in its data reader past its first step, 'replaced' faulting past the fit's end, a native handler
    installed over the record's during it; its run directory holding a regular file named `held` before it
    where given; its outcome and its record."""
    script = tmp_path / "crashing_fit.py"
    script.write_text(textwrap.dedent(CRASHING_FIT))
    run = tmp_path / "run"
    if held is not None:
        run.mkdir()
        (run / held).write_text("an unrelated file beside the checkpoints\n")
    inherited = {name: value for name, value in os.environ.items()
                 if name not in ("PYTHONFAULTHANDLER", "UCX_ERROR_SIGNALS")}
    environment = {**inherited, "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), *sys.path]),
                   **({} if ucx is None else {"UCX_ERROR_SIGNALS": ucx})}
    done = subprocess.run([sys.executable, str(script), str(run), mode], capture_output=True, text=True,
                          env=environment, timeout=600, check=False)
    records = sorted((run / "faults").glob("*.txt")) if (run / "faults").is_dir() else []
    return done, "".join(record.read_text() for record in records)


def _seen(done: subprocess.CompletedProcess) -> dict:
    """What `CRASHING_FIT` reported of its process, its last line's JSON."""
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_fault_past_a_fits_first_step_leaves_every_threads_stack_in_its_run_directory(tmp_path):
    """A segfault in a fit's data reader past its first step, in whatever thread reads it: the run
    directory's fault record holds the fatal signal and the reading thread's Python stack."""
    done, record = _fit(tmp_path, "plain")
    assert done.returncode == -signal.SIGSEGV, (done.returncode, done.stderr[-2000:])
    assert "Segmentation fault" in record and "crashing_batches" in record, (record, done.stderr[-2000:])


@pytest.mark.parametrize("ucx", [None, "SIGSEGV,SIGBUS"])
def test_a_process_owns_its_fatal_signals_from_its_start_asking_ucx_for_none_it_names(tmp_path, ucx):
    """A process readied by a fit, whose fault handler no other owner enabled, owns its fatal signals
    through Python's handler for good (never disabled, so a handler installed over it hands back to it),
    asking UCX for no handler of its own where its environment names no policy (an empty
    `UCX_ERROR_SIGNALS`), keeping the one it names (the review of toolkit/fault-record 4d757c7c4a)."""
    done, _ = _fit(tmp_path, "none", ucx)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert _seen(done) == {"ucx during the fit": "" if ucx is None else ucx, "enabled after the fit": True}
    assert list((tmp_path / "run" / "faults").glob("*.txt"))


def test_a_run_directory_holding_a_file_named_faults_fits_its_faults_on_standard_error(tmp_path):
    """A run directory may hold a regular file named `faults` beside its checkpoints (a valid checkpoint
    root's unrelated file): the fit runs, no record kept, the handler writing to standard error as before
    (Train 1290's review, row 13017: the record's directory raised FileExistsError before the fit read its
    checkpoints)."""
    done, record = _fit(tmp_path, "none", held="faults")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert _seen(done)["enabled after the fit"] and not record, (record, done.stderr[-2000:])
    assert (tmp_path / "run" / "faults").is_file()


def test_a_handler_installed_over_the_record_during_a_fit_outlives_it(tmp_path):
    """A native handler another library installed over the record's during a fit is the process's after
    it (libc's abort here), which a later segfault ends the process through, by SIGABRT."""
    done, _ = _fit(tmp_path, "replaced")
    assert done.returncode == -signal.SIGABRT, (done.returncode, done.stdout[-2000:], done.stderr[-2000:])


def test_a_fit_keeps_the_fault_handler_another_owner_enabled(tmp_path):
    """Pytest's own fault handler, enabled before the fit, stays its owner's: the fit records nothing of its
    own and leaves it enabled."""
    assert faulthandler.is_enabled()
    frozen_trainer(tmp_path).fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
    assert faulthandler.is_enabled()
    assert not (tmp_path / "run" / "faults").exists()
