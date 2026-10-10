"""`import dew` stops a process that would run JAX on the CPU beside an
NVIDIA GPU, with the install that fixes it (`dew.telemetry.devices`). The
GPU is nvidia-smi's answer, so a stand-in nvidia-smi plays every machine on a
host without one."""

import os
import subprocess
import sys

import pytest

from dew.telemetry.devices import CUDA_BUILDS, cpu_only_refusal, cuda_builds


@pytest.mark.parametrize("gpus, installed, wanted", [
    (("580.82.07", (8, 9)), [], 'uv pip install "dewml[cuda13]"'),
    (("575.57.08", (8, 0)), [], 'uv pip install "dewml[cuda12]"'),
    (("580.82.07", (7, 0)), [], 'uv pip install "dewml[cuda12]"'),      # CUDA 13 needs SM 7.5
    (("570.10", (8, 6)), ["cuda13"], "uv pip uninstall jax-cuda13-plugin jax-cuda13-pjrt && "
                                     'uv pip install "dewml[cuda12]"'),  # a build the driver cannot run
    (("520.61.05", (8, 0)), [], "Update the NVIDIA driver"),
    (("580.82.07", (5, 0)), [], "Update the NVIDIA driver"),           # older than either build's SM
])
def test_a_gpu_beside_cpu_only_jax_names_the_build_its_driver_runs(gpus, installed, wanted):
    """The newest build whose least driver and SM the machine meets
    (`CUDA_BUILDS`): cuda13 from driver 580 and SM 7.5, cuda12 from 525 and
    5.2. The message names the GPU and how to run on the CPU instead."""
    refusal = cpu_only_refusal({}, gpus, installed)
    assert refusal is not None and wanted in refusal, refusal
    assert gpus[0] in refusal and "JAX_PLATFORMS=cpu" in refusal


@pytest.mark.parametrize("env, gpus, installed", [
    ({"JAX_PLATFORMS": "cpu"}, ("580.82.07", (8, 9)), []),      # the CPU asked for
    ({}, None, []),                                             # no NVIDIA GPU answers
    ({}, ("580.82.07", (8, 9)), ["cuda13"]),
    ({}, ("580.82.07", (8, 9)), ["cuda12"]),                    # an older build the driver runs
    ({"JAX_PLATFORMS": "cuda,cpu"}, ("575.57.08", (8, 0)), ["cuda12"]),
])
def test_a_machine_that_runs_jax_where_it_asked_passes(env, gpus, installed):
    assert cpu_only_refusal(env, gpus, installed) is None


def test_the_rule_is_newest_build_first():
    """The rule docs/installation.md states and dewml.dev's install script applies."""
    assert CUDA_BUILDS == (("cuda13", 580, (7, 5)), ("cuda12", 525, (5, 2)))


@pytest.mark.parametrize("platforms", [None, "cpu"])
def test_import_dew_stops_where_the_gpu_would_sit_idle(tmp_path, platforms):
    """A stand-in nvidia-smi reports driver 580 and two GPUs: `import dew`
    stops with what `cpu_only_refusal` says of this environment's JAX (the
    cuda13 install where no CUDA build is present), before any backend opens.
    JAX_PLATFORMS=cpu imports."""
    smi = tmp_path / "nvidia-smi"
    smi.write_text('#!/bin/sh\nprintf "580.82.07, 8.9\\n580.82.07, 8.6\\n"\n')
    smi.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    done = subprocess.run([sys.executable, "-c", "import dew"], capture_output=True, text=True, env=env,
                          timeout=120)
    expected = None if platforms == "cpu" else cpu_only_refusal({}, ("580.82.07", (8, 6)), cuda_builds())
    if expected is None:
        assert done.returncode == 0, done.stderr[-2000:]
    else:
        assert done.returncode != 0 and expected in done.stderr, done.stderr[-2000:]
