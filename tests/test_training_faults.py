"""A run's native fault record (row 6497, `dew.training.faults`): a fatal signal in a fit leaves every
thread's Python stack in its run directory."""

import faulthandler
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
import signal
import sys

import jax
import optax
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Checkpoints, Trainer

mode = sys.argv[2]


def replaced():
    # A native handler installed over the record's for SIGSEGV, as NCCL's UCX transport installs its own
    # as its first collectives start: libc's abort, so a signal handed on to it ends the process by SIGABRT.
    libc = ctypes.CDLL(None)
    action = ctypes.create_string_buffer(256)
    ctypes.memmove(action, ctypes.byref(ctypes.c_void_p(ctypes.cast(libc.abort, ctypes.c_void_p).value)),
                   ctypes.sizeof(ctypes.c_void_p))
    assert libc.sigaction(int(signal.SIGSEGV), action, None) == 0


def crashing_batches(partition):
    for index, batch in enumerate(batches()):
        if index == 0 and mode == "replaced":
            replaced()
        if index == 12 and mode != "none":
            faulthandler._sigsegv()
        yield batch


Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), checkpoints=Checkpoints(sys.argv[1])).fit(
    Loader(crashing_batches, None, None, 8), steps=16, log_every=1)
print("handler enabled after the fit:", faulthandler.is_enabled())
"""


def _fit(tmp_path: Path, mode: str) -> tuple[subprocess.CompletedProcess, str]:
    """A fit of `CRASHING_FIT` in a process of its own (no fault handler enabled before it), in `mode`:
    'none' crashing nowhere, 'plain' faulting in its data reader past its first step, 'replaced' so after
    a native handler was installed over the record's before its first step; its outcome and its record."""
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


@pytest.mark.parametrize("mode", ["plain", "replaced"])
def test_a_fault_past_a_fits_first_step_leaves_every_threads_stack_in_its_run_directory(tmp_path, mode):
    """A segfault in a fit's data reader past its first step, in whatever thread reads it: the run
    directory's fault record holds the fatal signal and the reading thread's Python stack. Where a native
    handler replaced the record's before the first step (UCX's, as NCCL starts), the record is installed
    again past it once the step has run (`faults.Recording.rearmed`) and hands the signal on to it: the
    process ends by that handler's SIGABRT (the review of toolkit/fault-record 213979906: CPython's
    `faulthandler.enable` installs nothing once enabled)."""
    done, record = _fit(tmp_path, mode)
    assert done.returncode != 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Segmentation fault" in record and "crashing_batches" in record, (record, done.stderr[-2000:])
    if mode == "replaced":
        assert done.returncode == -signal.SIGABRT, (done.returncode, done.stderr[-2000:])


def test_a_fit_leaves_a_process_with_no_fault_handler_with_none(tmp_path):
    """A fit of a process whose fault handler is off records into its run directory and turns it off again
    as it ends, so nothing after it writes its stacks to the run's file."""
    done, _ = _fit(tmp_path, "none")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "handler enabled after the fit: False" in done.stdout, done.stdout[-2000:]
    assert list((tmp_path / "run" / "faults").glob("*.txt"))


def test_a_fit_keeps_the_fault_handler_another_owner_enabled(tmp_path):
    """Pytest's own fault handler, enabled before the fit, stays its owner's: the fit records nothing of its
    own and leaves it enabled."""
    assert faulthandler.is_enabled()
    frozen_trainer(tmp_path).fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
    assert faulthandler.is_enabled()
    assert not (tmp_path / "run" / "faults").exists()
