"""A run's native fault record (row 6497, `dew.training.faults`): a fatal signal in a fit leaves every
thread's Python stack in its run directory."""

import faulthandler
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from affine_run import batches
from affine_training import frozen_trainer

from dew.data.dataset import Loader

CRASHING_FIT = """
import faulthandler
import sys

import jax
import optax
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Checkpoints, Trainer


def crashing_batches(partition):
    for index, batch in enumerate(batches()):
        if index == 3:
            faulthandler._sigsegv()
        yield batch


Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), checkpoints=Checkpoints(sys.argv[1])).fit(
    Loader(crashing_batches, None, None, 8), steps=8, log_every=1)
"""


def test_a_fault_past_a_fits_first_step_leaves_every_threads_stack_in_its_run_directory(tmp_path):
    """A segfault in a fit's data reader after its first step, in whatever thread reads it: the process dies,
    and the run directory's fault record holds the fatal signal and the reading thread's Python stack, the
    handler armed again past the first step's (`Trainer.fit`, `faults.armed`)."""
    script = tmp_path / "crashing_fit.py"
    script.write_text(textwrap.dedent(CRASHING_FIT))
    run = tmp_path / "run"
    tests = Path(__file__).parent
    environment = {**os.environ, "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join([str(tests), *sys.path])}
    done = subprocess.run([sys.executable, str(script), str(run)], capture_output=True, text=True,
                          env=environment, timeout=600, check=False)
    assert done.returncode != 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = (run / "faults" / "process-0.txt").read_text()
    assert "Segmentation fault" in record, (record, done.stderr[-2000:])
    assert "crashing_batches" in record, record


def test_a_fit_leaves_the_processs_fault_handler_as_it_found_it(tmp_path):
    """A fit arms its record and, done, gives the process its own handler back (pytest's, to standard
    error), so nothing after it writes its stacks to a run's file."""
    enabled = faulthandler.is_enabled()
    frozen_trainer(tmp_path).fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
    assert (tmp_path / "run" / "faults" / "process-0.txt").exists()
    assert faulthandler.is_enabled() == enabled
