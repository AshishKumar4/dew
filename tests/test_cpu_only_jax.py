"""A run's setup stops a process whose JAX opened on the CPU beside an NVIDIA
GPU, with the install that fixes it (`dew.devices`), and `import dew`
does not. The GPU is nvidia-smi's answer, so a stand-in nvidia-smi plays every
machine on a host without one."""

import os
import subprocess
import sys

import pytest

from dew.builds import newest_build
from dew.devices import cpu_only_refusal, nvidia_gpus, xla_flag


def stand_in(directory, printed: str, status: int = 0):
    smi = directory / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\nprintf '{printed}'\nexit {status}\n")
    smi.chmod(0o755)
    return f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


@pytest.mark.parametrize("printed, status, gpus", [
    ("580.82.07, 8.9\\n580.82.07, 8.6\\n", 0, ("580.82.07", (8, 6))),   # the least SM of the GPUs
    ("470.256.02, [N/A]\\n", 0, None),                                 # a driver too old to report an SM
    ("", 9, None),                                                       # nvidia-smi failing
])
def test_nvidia_smi_reads_as_the_driver_and_the_least_sm(tmp_path, monkeypatch, printed, status, gpus):
    monkeypatch.setenv("PATH", stand_in(tmp_path, printed, status))
    assert nvidia_gpus() == gpus


def test_a_host_with_no_nvidia_smi_lists_no_gpu(tmp_path, monkeypatch):
    """Where no nvidia-smi is on PATH, no GPU is listed, and nothing runs to ask."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert nvidia_gpus() is None


def test_a_bare_xla_flag_reads_true_and_its_last_occurrence_wins(monkeypatch):
    """`xla_flag` reads XLA_FLAGS as XLA's own parser does: a bare `--name` is 'true', a repeated flag
    its last value, and an absent one None."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_a --xla_b=1 --xla_b=2 --xla_c=x --xla_c")
    read = [xla_flag(name) for name in ("xla_a", "xla_b", "xla_c", "xla_d")]
    assert read == ["true", "2", "true", None]


LATE = """
import logging
import os

records = []


class Kept(logging.Handler):
    def emit(self, record):
        records.append(record)


logging.getLogger("dew").addHandler(Kept())
import jax

jax.devices()
before = os.environ.get("XLA_FLAGS")
import dew
assert os.environ.get("XLA_FLAGS") == before, "import dew set XLA flags no open backend reads"
assert [(record.name, record.levelno) for record in records] == [("dew.devices", logging.WARNING)], records
"""


def test_dew_imported_after_the_backend_opened_warns_and_sets_no_flag():
    """A process that opens JAX's backend before `import dew`: Dew's numerical policy can no longer
    take effect, so the import logs one warning of it, with the restart that applies it, and leaves
    XLA_FLAGS as it was."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (os.path.join(root, "src"),
                                                                     os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", LATE], capture_output=True, text=True, env=env,
                          timeout=300, cwd=root)
    assert done.returncode == 0, done.stderr[-2000:]


REFUSED = 3
"""The status the script ends with where the run's setup refuses the machine."""

READIED = f"""
import logging
import sys

import dew
assert not logging.getLogger("dew").handlers, "import dew configured Dew's logger"
sys.path.insert(0, "tests")


def run():
    if sys.argv[1] == "prepare":
        from dew.training import prepare_process
        prepare_process(multi_host=False)
        return
    from affine_run import fit_one_step
    fit_one_step()


from dew.devices import CPUOnlyJax

try:
    run()
except CPUOnlyJax as refused:
    print(refused, file=sys.stderr)
    sys.exit({REFUSED})
finally:
    assert logging.getLogger("dew").handlers, "the run left Dew's logger without its handler"
"""


@pytest.mark.parametrize("platforms", [None, "cpu"])
@pytest.mark.parametrize("entry", ["prepare", "fit"])
def test_a_run_stops_where_the_gpu_would_sit_idle_and_import_dew_does_not(tmp_path, platforms, entry):
    """A stand-in nvidia-smi reports driver 580 and two GPUs beside a JAX that opens on the CPU, the
    one device it is shown being none a host has:
    `import dew` leaves the process alone, its logger and its hardware, and a run's setup,
    `prepare_process` or a `Trainer.fit` no `prepare_process` came before, stops with what
    `cpu_only_refusal` says of this environment's JAX, the plain refusal where it finds an install
    that should run. Either way the run gives Dew's logger its handler. JAX_PLATFORMS=cpu runs."""
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    env["PATH"] = stand_in(tmp_path, "580.82.07, 8.9\\n580.82.07, 8.6\\n")
    # A device that no host has, which leaves the GPUs to JAX by name: its CUDA plugin opens
    # none, on a host with GPUs too, and JAX falls back to the CPU.
    env["CUDA_VISIBLE_DEVICES"] = "999"
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # This checkout's Dew, ahead of any the environment installs.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (os.path.join(root, "src"),
                                                      os.environ.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", READIED, entry], capture_output=True, text=True, env=env,
                          timeout=300, cwd=root)
    assert done.returncode == (0 if platforms == "cpu" else REFUSED), done.stderr[-2000:]


@pytest.mark.parametrize("driver, cuda, newest", [
    ("580.82.07", 13000, "cuda13"),    # a driver that runs CUDA 13.0 itself
    ("570.124.06", 13000, "cuda13"),   # an older one under NVIDIA's forward-compatibility package
    ("570.124.06", 12080, "cuda12"),   # the same driver without it: CUDA 12.8
    ("580.82.07", None, None),         # GPUs listed, and no CUDA driver library loads
])
def test_a_build_runs_where_the_user_mode_driver_runs_its_cuda(driver, cuda, newest):
    """JAX's cuda13 build starts where the CUDA version the user-mode driver reports is 13.0 or
    newer, as its plugin checks: a 580 driver's own, or a 570's under NVIDIA's cuda-compat, whose
    kernel driver's major is then no bar. A CUDA 12.8 user-mode driver runs the cuda12 build alone,
    and a machine whose driver library does not load runs none."""
    gpus = (driver, (9, 0))
    for installed in (["cuda13"], ["cuda12"]):
        runs = cuda is not None and newest is not None and (installed == ["cuda12"] or newest == "cuda13")
        assert (cpu_only_refusal({}, gpus, installed, cuda) is None) == runs
    if cuda is not None:
        assert newest_build(gpus, cuda) == newest


ROUTED = """
import logging

import optax
import jax

received = []


class Received(logging.Handler):
    def emit(self, record):
        received.append(record.getMessage())


logging.getLogger().addHandler(Received())
dew_logger = logging.getLogger("dew")
dew_logger.addHandler(logging.NullHandler())
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Trainer
from dew.training.runtime import ready_process

for _ in range(2):
    Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0)).fit(
        Loader(lambda partition: batches(), None, None, 8), steps=1, log_every=1)
    ready_process()
logging.getLogger("dew.training").warning("through the application's handler")
assert [type(handler) for handler in dew_logger.handlers] == [logging.NullHandler], dew_logger.handlers
assert "through the application's handler" in received, received
"""


def test_an_application_that_routes_dews_records_receives_them_through_its_handler():
    """An application that gives Dew's logger a handler of its own before a run's setup keeps it:
    two fits and setups attach none of Dew's, and Dew's records reach the application's root
    handler."""
    tests = os.path.dirname(os.path.abspath(__file__))
    # This checkout's Dew, ahead of any the environment installs, and its test helpers.
    path = os.pathsep.join(filter(None, (os.path.join(os.path.dirname(tests), "src"), tests,
                                         os.environ.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", ROUTED], capture_output=True, text=True, timeout=300,
                          cwd=tests, env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path})
    assert done.returncode == 0, done.stderr[-2000:]
