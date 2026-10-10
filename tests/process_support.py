"""CPU worker sessions, pool reports and process-group cleanup."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("multiprocess_worker.py")
DEVICES = 8


def prepare_worker(args, *, multi_host: bool | None = False) -> None:
    """Join a test pool through the same Open MPI detection as a launched run."""
    if args.coordinator:
        os.environ.update({
            "OMPI_MCA_orte_hnp_uri": f"0.0;tcp://{args.coordinator}",
            "OMPI_COMM_WORLD_SIZE": str(args.processes),
            "OMPI_COMM_WORLD_RANK": str(args.process_id),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(args.process_id),
            "JAX_COORDINATOR_ADDRESS": args.coordinator,
        })
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=True if args.coordinator else multi_host)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def worker_env(devices: int) -> dict[str, str]:
    """This checkout's source, CPU devices and faulthandler in each worker."""
    return {**os.environ, "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONFAULTHANDLER": "1",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}


def start_process(command: Sequence[str], devices: int, *,
                  environment: Mapping[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command, cwd=REPO_ROOT, env={**worker_env(devices), **(environment or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def spawn(mode: str, out: Path, processes: int = 1, process_id: int = 0,
          coordinator: str | None = None, devices: int | None = None, *,
          script: Path = WORKER, **flags) -> subprocess.Popen[str]:
    command = [sys.executable, str(script), mode, "--out", str(out),
               "--processes", str(processes), "--process-id", str(process_id)]
    if coordinator is not None:
        command += ["--coordinator", coordinator]
    for name, value in flags.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not None:
            command += [flag, str(value)]
    return start_process(command, DEVICES // processes if devices is None else devices)


def terminate(process: subprocess.Popen[str]) -> None:
    """Kill the session even if its leader already exited and left descendants."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=60)


class ProcessGroup:
    """Own started workers, including those launched before a later start fails."""

    def __init__(self) -> None:
        self.running: list[subprocess.Popen[str]] = []

    def __enter__(self) -> "ProcessGroup":
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 traceback: TracebackType | None) -> None:
        for process in self.running:
            terminate(process)
        if error is not None:
            print(worker_outputs(self.running))
        else:
            for process in self.running:
                process.communicate(timeout=60)


def worker_outputs(pool: Sequence[subprocess.Popen[str]]) -> str:
    outputs = []
    for index, process in enumerate(pool):
        output = process.communicate(timeout=60)[0]
        outputs.append(f"--- process {index}, exit {process.returncode}\n{output}")
    return "\n".join(outputs)


def stuck(pool: Sequence[subprocess.Popen[str]], what: str) -> str:
    """Abort workers for faulthandler stacks before killing their whole sessions."""
    for process in pool:
        with contextlib.suppress(ProcessLookupError):
            os.kill(process.pid, signal.SIGABRT)
    for process in pool:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            terminate(process)
    for process in pool:
        terminate(process)
    return f"the pool did not {what}\n" + worker_outputs(pool)


def report_of(process: subprocess.Popen[str], out: Path, timeout: float = 600,
              pool: Sequence[subprocess.Popen[str]] | None = None) -> dict:
    try:
        log = process.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        pytest.fail(stuck([process] if pool is None else pool,
                          f"let {out.name} finish within {timeout}s"))
    assert process.returncode == 0, f"{out.name} exited {process.returncode}\n{log}"
    return json.loads(out.read_text())


def run_worker(mode: str, out: Path, **flags) -> dict:
    with ProcessGroup() as pool:
        pool.running.append(spawn(mode, out, **flags))
        return report_of(pool.running[0], out)


def run_pool(mode: str | Path, directory: Path, processes: int, *, timeout: float = 600,
             start: Callable[..., subprocess.Popen[str]] = spawn, **flags) -> list[dict]:
    """Launch each rank and return reports in rank order, retaining all failure logs."""
    directory.mkdir(parents=True, exist_ok=True)
    coordinator = f"127.0.0.1:{free_port()}"
    outs = [directory / f"process{index}.json" for index in range(processes)]
    with ProcessGroup() as pool:
        for index, out in enumerate(outs):
            pool.running.append(start(mode, out, processes=processes, process_id=index,
                                      coordinator=coordinator, **flags))
        return [report_of(process, out, timeout=timeout, pool=pool.running)
                for process, out in zip(pool.running, outs, strict=True)]
