"""A run's setup stops a process whose JAX opened on the CPU beside an NVIDIA
GPU, with the install that fixes it (`dew.devices`), and `import dew`
does not. The GPU is nvidia-smi's answer, its CUDA version the user-mode
driver's (libcuda's `cuDriverGetVersion`) and JAX's CUDA builds the plugins
installed, so stand-ins for nvidia-smi, the driver library and the plugins
play every machine on a host without one. `import dew` keeps every rounding a
program states, unless the run or a backend opened before it decided
otherwise."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from dew.builds import CUDA_BUILDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ahead(*directories: str) -> str:
    """A PYTHONPATH that reads `directories` first: this checkout's Dew ahead of any the environment
    installs, and its test helpers."""
    return os.pathsep.join(filter(None, (*directories, os.environ.get("PYTHONPATH"))))


def stand_in(directory, printed: str, status: int = 0):
    smi = directory / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\nprintf '{printed}'\nexit {status}\n")
    smi.chmod(0o755)
    return f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


def stand_in_driver(directory, version: int) -> str:
    """The loader's path to a user-mode CUDA driver library, libcuda.so.1, whose `cuDriverGetVersion`
    reports `version` (13.0 as 13000), built in `directory` by the host's C compiler."""
    compiler = shutil.which("cc")
    assert compiler is not None, "the stand-in driver library is built by the host's C compiler, cc"
    source = directory / "libcuda.c"
    source.write_text(f"int cuDriverGetVersion(int *version) {{ *version = {version}; return 0; }}\n")
    subprocess.run([compiler, "-shared", "-fPIC", "-o", str(directory / "libcuda.so.1"), str(source)],
                   check=True)
    return os.pathsep.join(filter(None, (str(directory), os.environ.get("LD_LIBRARY_PATH"))))


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
    from affine_training import fit_one_step
    fit_one_step()


from dew.devices import CPUOnlyJax

try:
    run()
except CPUOnlyJax as refused:
    with open(sys.argv[2], "w") as said:
        said.write(str(refused))
    sys.exit({REFUSED})
finally:
    assert logging.getLogger("dew").handlers, "the run left Dew's logger without its handler"
"""


def readied(directory, entry: str, **environment: str) -> tuple[subprocess.CompletedProcess, str | None]:
    """A fresh process's run (`READIED`), set up by `prepare_process` and a step after it, or by a
    `Trainer.fit` alone (`entry`), on `environment` over this one's, without its JAX_PLATFORMS and
    with a device no host has, which leaves the GPUs to JAX by name: its CUDA plugin opens none, on a
    host with GPUs too, and JAX falls back to the CPU. The process, and the refusal it stopped with."""
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    env["CUDA_VISIBLE_DEVICES"] = "999"
    env["PYTHONPATH"] = ahead(os.path.join(ROOT, "src"))
    env.update(environment)
    said = directory / "refusal"
    done = subprocess.run([sys.executable, "-c", READIED, entry, str(said)], capture_output=True, text=True,
                          env=env, timeout=300, cwd=ROOT)
    return done, said.read_text() if said.is_file() else None


@pytest.mark.parametrize("platforms", [None, "cpu"])
@pytest.mark.parametrize("entry", ["prepare", "fit"])
def test_a_run_stops_where_the_gpu_would_sit_idle_and_import_dew_does_not(tmp_path, platforms, entry):
    """A stand-in nvidia-smi reports driver 580 and two GPUs beside a JAX that opens on the CPU:
    `import dew` leaves the process alone, its logger and its hardware, and a run's setup,
    `prepare_process` or a `Trainer.fit` no `prepare_process` came before, stops with what
    `cpu_only_refusal` says of this environment's JAX, the plain refusal where it finds an install
    that should run. Either way the run gives Dew's logger its handler. JAX_PLATFORMS=cpu trains."""
    path = stand_in(tmp_path, "580.82.07, 8.9\\n580.82.07, 8.6\\n")
    chosen = {} if platforms is None else {"JAX_PLATFORMS": platforms}
    done, refusal = readied(tmp_path, entry, PATH=path, **chosen)
    assert done.returncode == (0 if platforms == "cpu" else REFUSED), done.stderr[-2000:]
    assert (refusal is None) == (platforms == "cpu")


def stand_in_builds(directory, builds) -> str:
    """The PYTHONPATH entry of JAX's CUDA builds `builds` installed, in place of this environment's,
    as `cuda_builds` and JAX's plugin discovery find them: a `jax_plugins` package over the namespace
    one, each build's plugin in it registering no backend, as a build that opens no GPU here."""
    plugins = directory / "jax_plugins"
    plugins.mkdir()
    (plugins / "__init__.py").write_text("")
    for build in builds:
        (plugins / f"xla_{build}").mkdir()
        (plugins / f"xla_{build}" / "__init__.py").write_text("def initialize():\n    pass\n")
    return str(directory)


MACHINES = {
    # nvidia-smi's rows, the CUDA version the user-mode driver reports, the builds that run there,
    # newest first, and the builds installed.
    "a-580-driver-runs-cuda-13-itself": ("580.82.07, 9.0\\n580.82.07, 9.0\\n", 13000,
                                         ("cuda13", "cuda12"), ()),
    "a-570-under-nvidias-cuda-compat": ("570.124.06, 9.0\\n", 13000, ("cuda13", "cuda12"), ()),
    "the-same-570-without-it": ("570.124.06, 9.0\\n", 12080, ("cuda12",), ("cuda13",)),
    "the-least-sm-of-the-gpus": ("580.82.07, 8.9\\n580.82.07, 7.0\\n", 13000, ("cuda12",), ("cuda13",)),
    "a-cuda-no-build-runs-on": ("520.61.05, 8.0\\n", 11080, (), ()),
    "an-installed-build-that-runs": ("580.82.07, 9.0\\n", 13000, ("cuda13", "cuda12"), ("cuda13",)),
}


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_a_runs_setup_installs_the_cuda_build_the_user_mode_driver_runs(tmp_path, machine):
    """`prepare_process` on a machine whose GPUs nvidia-smi lists and whose driver library reports a
    CUDA version, JAX having opened on the CPU, refuses it with the install that runs there: JAX's
    cuda13 build where that version is 13.0 or newer, as its plugin checks, a 580 driver's own or a
    570's under NVIDIA's cuda-compat, whose kernel driver's major is then no bar; the cuda12 build
    alone under CUDA 12.8, or beside a GPU older than cuda13's least SM, in place of an installed
    cuda13 that cannot run; and none where no build runs, nor where an installed build runs, whose JAX
    should have opened the GPU."""
    printed, cuda, runs, installed = MACHINES[machine]
    for part in ("driver", "plugins"):
        (tmp_path / part).mkdir()
    done, refusal = readied(tmp_path, "prepare", PATH=stand_in(tmp_path, printed),
                            LD_LIBRARY_PATH=stand_in_driver(tmp_path / "driver", cuda),
                            PYTHONPATH=ahead(stand_in_builds(tmp_path / "plugins", installed),
                                             os.path.join(ROOT, "src")))
    assert done.returncode == REFUSED and refusal is not None, done.stderr[-2000:]
    installs = {build for build, *_ in CUDA_BUILDS if f"dewml[{build}]" in refusal}
    replaced = {build for build, *_ in CUDA_BUILDS if f"jax-{build}-plugin" in refusal}
    expected = set() if set(runs) & set(installed) else set(runs[:1])
    assert (installs, replaced) == (expected, set(installed) if expected else set()), refusal


@pytest.mark.parametrize("smi", ["none-on-the-path", "a-driver-too-old-to-report-an-sm", "failing"])
def test_a_host_without_a_gpu_jax_could_use_trains_on_the_cpu(tmp_path, smi):
    """A host where nvidia-smi lists no GPU a CUDA build could run on, none being on the PATH, its
    driver too old to report an SM (`[N/A]`), or nvidia-smi failing, is no GPU sitting idle: its
    run's setup and a step after it run on the CPU JAX opened."""
    if smi == "none-on-the-path":
        (tmp_path / "empty").mkdir()
        path = str(tmp_path / "empty")
    else:
        path = stand_in(tmp_path, *{"a-driver-too-old-to-report-an-sm": ("470.256.02, [N/A]\\n", 0),
                                    "failing": ("", 9)}[smi])
    done, refusal = readied(tmp_path, "prepare", PATH=path)
    assert done.returncode == 0 and refusal is None, (refusal, done.stderr[-2000:])


ROUNDED = """
import json
import logging
import sys

held = []


class Held(logging.Handler):
    def emit(self, record):
        held.append([record.name, record.levelno])


logging.getLogger("dew").addHandler(Held())
if sys.argv[1] == "after-a-computation":
    import jax.numpy as jnp
    jnp.zeros(2).block_until_ready()
import dew
import jax
import jax.numpy as jnp
import numpy as np

a = jax.random.normal(jax.random.key(0), (4096,), jnp.bfloat16)
b = (jax.random.normal(jax.random.key(1), (4096,)) * 0.37).astype(jnp.bfloat16)
summed = np.asarray(jax.jit(lambda a, b: (a + b).astype(jnp.float32))(a, b))
print(json.dumps({"rounded": bool(np.array_equal(summed, summed.astype(jnp.bfloat16).astype(np.float32))),
                  "warnings": [name for name, level in held if level >= logging.WARNING]}))
"""


@pytest.mark.parametrize("imported, named, rounded, warned", [
    ("first", None, True, False),
    ("first", "--xla_allow_excess_precision", False, False),
    ("first", "--xla_allow_excess_precision=true", False, False),
    ("after-a-computation", None, False, True),
], ids=["imported-first", "a-bare-flag-allows-excess-precision", "the-flag-allows-it", "imported-late"])
def test_import_dew_keeps_every_bf16_rounding_a_program_states(imported, named, rounded, warned):
    """A bf16 sum a program casts up to fp32 is a bf16 value where `import dew` came before the first
    computation: XLA's default lets the fusion skip the rounding, and Dew keeps it. A run that names
    XLA's policy keeps its own, a bare flag reading as true as XLA reads it, and a process whose
    backend opened before `import dew` keeps XLA's default, which Dew then says once at warning level."""
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_allow_excess_precision")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": " ".join([*flags, *filter(None, [named])]),
           "PYTHONPATH": ahead(os.path.join(ROOT, "src"))}
    done = subprocess.run([sys.executable, "-c", ROUNDED, imported], capture_output=True, text=True, env=env,
                          timeout=300, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.splitlines()[-1])
    assert seen["rounded"] is rounded
    assert seen["warnings"] == (["dew.devices"] if warned else [])


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
    tests = os.path.join(ROOT, "tests")
    path = ahead(os.path.join(ROOT, "src"), tests)
    done = subprocess.run([sys.executable, "-c", ROUTED], capture_output=True, text=True, timeout=300,
                          cwd=tests, env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path})
    assert done.returncode == 0, done.stderr[-2000:]
