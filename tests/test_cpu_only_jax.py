"""A run's setup stops a process whose JAX opened on the CPU beside an NVIDIA
GPU, with the install that fixes it (`dew.telemetry.devices`), and `import dew`
does not. The GPU is nvidia-smi's answer, so a stand-in nvidia-smi plays every
machine on a host without one."""

import json
import os
import subprocess
import sys

import pytest

from dew.telemetry.devices import nvidia_gpus


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
    from affine_run import fit_one_step
    fit_one_step()


from dew.telemetry.devices import CPUOnlyJax

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


DRIVER = """
int cuDriverGetVersion(int *version) {{
    *version = {version};
    return {status};
}}
"""
"""A user-mode driver library's one call a run's setup makes: the CUDA version it supports, or an
error (999, CUDA_ERROR_UNKNOWN) where it reports none."""

SETUP = f"""
import json
import sys

from dew.telemetry.devices import CPUOnlyJax
from dew.training import prepare_process

try:
    prepare_process(multi_host=False)
except CPUOnlyJax as refused:
    print(json.dumps({{"reason": refused.reason, "cuda": refused.cuda, "runs": list(refused.runs),
                      "install": refused.install}}))
    sys.exit({REFUSED})
"""


def machine(directory, driver: str, library: int | str, installed: tuple[str, ...]) -> dict[str, str]:
    """A process environment on a machine with an NVIDIA GPU of SM 9.0 on `driver` (a stand-in
    nvidia-smi), whose user-mode driver library (a stand-in libcuda.so.1 ahead of any on the
    loader's path) reports the CUDA version `library`, or fails to (`"error"`), or does not load
    (`"unloadable"`), and whose JAX's CUDA builds are `installed` alone (a `jax_plugins` package
    ahead of the environment's), each a plugin that opens no GPU, as one does on a host it cannot
    start on."""
    driven = directory / "driver"
    driven.mkdir()
    if library == "unloadable":
        (driven / "libcuda.so.1").write_bytes(b"not a shared object")
    else:
        version, status = (0, 999) if library == "error" else (library, 0)
        (driven / "driver.c").write_text(DRIVER.format(version=version, status=status))
        subprocess.run(["cc", "-shared", "-fPIC", "-o", str(driven / "libcuda.so.1"),
                        str(driven / "driver.c")], check=True)
    plugins = directory / "plugins"
    (plugins / "jax_plugins").mkdir(parents=True)
    (plugins / "jax_plugins" / "__init__.py").write_text("")
    for build in installed:
        (plugins / "jax_plugins" / f"xla_{build}").mkdir()
        (plugins / "jax_plugins" / f"xla_{build}" / "__init__.py").write_text("def initialize():\n    pass\n")
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = os.pathsep.join(filter(None, (str(driven), os.environ.get("LD_LIBRARY_PATH"))))
    env.update(PATH=stand_in(directory, f"{driver}, 9.0\\n"), CUDA_VISIBLE_DEVICES="999",
               LD_LIBRARY_PATH=loader,
               PYTHONPATH=os.pathsep.join(filter(None, (os.path.join(root, "src"), str(plugins),
                                                        os.environ.get("PYTHONPATH")))))
    return env


@pytest.mark.parametrize("driver, library, installed, reason, runs, install", [
    # A driver that runs CUDA 13.0 itself: a build runs here, so its plugin's own start failed.
    ("580.82.07", 13000, ("cuda13",), "plugin failed", ("cuda13",), None),
    ("580.82.07", 13000, ("cuda12",), "plugin failed", ("cuda12",), None),  # CUDA 13.0 runs cuda12 too
    ("570.124.06", 13000, ("cuda13",), "plugin failed", ("cuda13",), None),  # 570 under cuda-compat
    ("570.124.06", 13000, (), "none installed", (), "cuda13"),  # which installs the newer build
    ("570.124.06", 12080, ("cuda13",), "another build runs", (), "cuda12"),  # without it: CUDA 12.8
    ("570.124.06", 12080, ("cuda12",), "plugin failed", ("cuda12",), None),
    ("580.82.07", 13000, (), "none installed", (), "cuda13"),
    ("535.104.05", 11080, ("cuda12",), "no build runs", (), None),  # CUDA 11.8 runs no build of JAX
    ("580.82.07", "error", ("cuda13",), "no driver library", (), None),  # a library that reports none
    ("580.82.07", "unloadable", ("cuda13",), "no driver library", (), None),  # and one that won't load
])
def test_a_runs_setup_reads_the_machines_driver_and_builds_and_names_the_build_that_runs(
        tmp_path, driver, library, installed, reason, runs, install):
    """A run's setup (`prepare_process`) in a fresh process on a machine with an NVIDIA GPU whose JAX
    opens none reads the driver from nvidia-smi, the CUDA version from the user-mode driver library
    and the builds of JAX installed, and stops with what it found: the installed builds that run
    here, as JAX's cuda13 build starts where the user-mode driver reports CUDA 13.0 or newer (a 580
    driver's own, or a 570's under cuda-compat) and a CUDA 12.8 driver runs the cuda12 build alone,
    and where none does, the build to install, none where no driver library reports a version, and
    which of the refusal's cases it chose."""
    done = subprocess.run([sys.executable, "-c", SETUP], capture_output=True, text=True,
                          env=machine(tmp_path, driver, library, installed), timeout=300)
    assert done.returncode == REFUSED, done.stderr[-2000:]
    cuda = library if isinstance(library, int) else None
    assert json.loads(done.stdout.splitlines()[-1]) == {"reason": reason, "cuda": cuda, "runs": list(runs),
                                                        "install": install}


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


STEPPED = """
import jax
import jax.numpy as jnp

from dew.training import prepare_process

prepare_process(multi_host=False)
print(jax.devices()[0].platform, float(jax.jit(lambda x: (x * x).sum())(jnp.arange(4.0))))
"""


@pytest.mark.gpu(reason="the machine's own driver and JAX's own CUDA plugin")
def test_a_runs_setup_on_a_gpu_machine_starts_the_cuda_plugin_and_steps_on_the_gpu(tmp_path):
    """On a machine whose driver and installed build run, a run's setup in a fresh process passes
    through the real CUDA plugin's start, with no stand-in, and the process computes on the GPU."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (os.path.join(root, "src"),
                                                                     os.environ.get("PYTHONPATH"))))}
    env.pop("JAX_PLATFORMS", None)
    done = subprocess.run([sys.executable, "-c", STEPPED], capture_output=True, text=True, env=env,
                          timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["gpu", "14.0"]
