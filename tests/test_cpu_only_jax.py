"""A run's setup stops a process whose JAX opened on the CPU beside an NVIDIA
GPU, with the install that fixes it (`dew.telemetry.devices`), and `import dew`
does not. The GPU is nvidia-smi's answer, so a stand-in nvidia-smi plays every
machine on a host without one."""

import os
import subprocess
import sys

import pytest

from dew.telemetry.builds import newest_build
from dew.telemetry.devices import cpu_only_refusal, nvidia_gpus


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
    import jax
    import optax
    from affine_run import Regression, batches
    from dew.data.dataset import Loader
    from dew.training import Trainer
    Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0)).fit(
        Loader(lambda partition: batches(), None, None, 8), steps=1, log_every=1)


try:
    run()
except RuntimeError as refused:
    print(refused, file=sys.stderr)
    sys.exit({REFUSED})
finally:
    assert logging.getLogger("dew").handlers, "the run left Dew's logger without its handler"
"""


@pytest.mark.parametrize("platforms", [None, "cpu"])
@pytest.mark.parametrize("entry", ["prepare", "fit"])
def test_a_run_stops_where_the_gpu_would_sit_idle_and_import_dew_does_not(tmp_path, platforms, entry):
    """A stand-in nvidia-smi reports driver 580 and two GPUs beside a JAX that opens on the CPU:
    `import dew` leaves the process alone, its logger and its hardware, and a run's setup,
    `prepare_process` or a `Trainer.fit` no `prepare_process` came before, stops with what
    `cpu_only_refusal` says of this environment's JAX, the plain refusal where it finds an install
    that should run. Either way the run gives Dew's logger its handler. JAX_PLATFORMS=cpu runs."""
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    env["PATH"] = stand_in(tmp_path, "580.82.07, 8.9\\n580.82.07, 8.6\\n")
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    done = subprocess.run([sys.executable, "-c", READIED, entry], capture_output=True, text=True, env=env,
                          timeout=300, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
