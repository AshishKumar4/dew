"""`import dew` stops a process that would run JAX on the CPU beside an
NVIDIA GPU, with the install that fixes it (`dew.telemetry.devices`). The
GPU is nvidia-smi's answer, so a stand-in nvidia-smi plays every machine on a
host without one."""

import os
import subprocess
import sys

import pytest

from dew.telemetry.devices import cpu_only_refusal, cuda_builds, nvidia_gpus


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


@pytest.mark.parametrize("platforms", [None, "cpu"])
def test_import_dew_stops_where_the_gpu_would_sit_idle(tmp_path, platforms):
    """A stand-in nvidia-smi reports driver 580 and two GPUs: `import dew`
    stops with what `cpu_only_refusal` says of this environment's JAX (the
    cuda13 install where no CUDA build is present), before any backend opens.
    JAX_PLATFORMS=cpu imports."""
    env = {key: value for key, value in os.environ.items() if key != "JAX_PLATFORMS"}
    env["PATH"] = stand_in(tmp_path, "580.82.07, 8.9\\n580.82.07, 8.6\\n")
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    done = subprocess.run([sys.executable, "-c", "import dew"], capture_output=True, text=True, env=env,
                          timeout=120)
    expected = None if platforms == "cpu" else cpu_only_refusal({}, ("580.82.07", (8, 6)), cuda_builds())
    if expected is None:
        assert done.returncode == 0, done.stderr[-2000:]
    else:
        assert done.returncode != 0, done.stderr[-2000:]
