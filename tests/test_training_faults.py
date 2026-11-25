"""A run's native fault record (row 6497, `dew.training.faults`): a fatal signal in a fit leaves every
thread's Python stack in its run directory."""

import faulthandler
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

from affine_run import batches
from affine_training import frozen_trainer

from dew.data.dataset import Loader

CRASHING_FIT = """
import ctypes
import faulthandler
import os
import signal
import sys

import jax
import optax
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Checkpoints, Trainer

mode = sys.argv[2]


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
            print("UCX during the fit:", repr(os.environ.get("UCX_ERROR_SIGNALS")), flush=True)
            if mode == "replaced":
                replaced()
        if index == 12 and mode == "plain":
            faulthandler._sigsegv()
        yield batch


Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), checkpoints=Checkpoints(sys.argv[1])).fit(
    Loader(crashing_batches, None, None, 8), steps=16, log_every=1)
print("handler enabled after the fit:", faulthandler.is_enabled(), flush=True)
print("UCX after the fit:", repr(os.environ.get("UCX_ERROR_SIGNALS")), flush=True)
if mode == "replaced":
    faulthandler._sigsegv()
"""


def _fit(tmp_path: Path, mode: str) -> tuple[subprocess.CompletedProcess, str]:
    """A fit of `CRASHING_FIT` in a process of its own (no fault handler enabled before it), in `mode`:
    'none' crashing nowhere, 'plain' faulting in its data reader past its first step, 'replaced' faulting
    past the fit's end, a native handler installed over the record's during it; its outcome and its
    record."""
    script = tmp_path / "crashing_fit.py"
    script.write_text(textwrap.dedent(CRASHING_FIT))
    run = tmp_path / "run"
    inherited = {name: value for name, value in os.environ.items() if name != "PYTHONFAULTHANDLER"}
    environment = {**inherited, "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), *sys.path])}
    done = subprocess.run([sys.executable, str(script), str(run), mode], capture_output=True, text=True,
                          env=environment, timeout=600, check=False)
    records = sorted((run / "faults").glob("*.txt"))
    return done, "".join(record.read_text() for record in records)


def test_a_fault_past_a_fits_first_step_leaves_every_threads_stack_in_its_run_directory(tmp_path):
    """A segfault in a fit's data reader past its first step, in whatever thread reads it: the run
    directory's fault record holds the fatal signal and the reading thread's Python stack."""
    done, record = _fit(tmp_path, "plain")
    assert done.returncode != 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Segmentation fault" in record and "crashing_batches" in record, (record, done.stderr[-2000:])


def test_a_fit_owns_the_fatal_signals_alone_and_ends_as_it_found_the_process(tmp_path):
    """A fit of a process whose fault handler is off records into its run directory, asking UCX for no
    handler of its own (an empty `UCX_ERROR_SIGNALS`: one owner, where UCX's and the record's chained
    re-raised into a spent handler, the review of toolkit/fault-record 82965eac9), and ends with the handler
    off and the environment as it was."""
    done, _ = _fit(tmp_path, "none")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "UCX during the fit: ''" in done.stdout, done.stdout[-2000:]
    assert "handler enabled after the fit: False" in done.stdout and "UCX after the fit: None" in done.stdout
    assert list((tmp_path / "run" / "faults").glob("*.txt"))


def test_a_handler_installed_over_the_record_during_a_fit_outlives_it(tmp_path):
    """A native handler another library installed over the record's during a fit is the process's after
    it: the fit's end disables its own and keeps that one (libc's abort here), which a later segfault ends
    the process through, by SIGABRT."""
    done, _ = _fit(tmp_path, "replaced")
    assert done.returncode == -signal.SIGABRT, (done.returncode, done.stdout[-2000:], done.stderr[-2000:])


def test_a_fit_keeps_the_fault_handler_another_owner_enabled(tmp_path):
    """Pytest's own fault handler, enabled before the fit, stays its owner's: the fit records nothing of its
    own and leaves it enabled."""
    assert faulthandler.is_enabled()
    frozen_trainer(tmp_path).fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
    assert faulthandler.is_enabled()
    assert not (tmp_path / "run" / "faults").exists()
