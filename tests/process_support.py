"""CPU worker sessions, pool reports, process-group cleanup and the local HTTP server a pool reads."""

import argparse
import contextlib
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType

import pytest
from lane_environment import MESH_DEVICES, without_deterministic_ops
from tool_support import load

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("multiprocess_worker.py")
groups = load("child_groups", "tools/armada")
"""The children's group ownership Dew's CI uses too."""


READ_ONLY = "--xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_READ"
"""The flag `gpu-run -n 0.5` adds beside other work on a GPU, under which no process measures."""

def keeping_picks() -> str:
    """This process's XLA flags without the cuda lane's deterministic-ops flag: a process keeps and
    measures picks only without it (`dew.cache.keeps_picks`)."""
    return without_deterministic_ops(os.environ.get("XLA_FLAGS", ""))


def fresh(script: str, *arguments: str, cache_dir, flags: str, source: Path = REPO_ROOT / "src"):
    """`script`'s last printed JSON, run by a fresh interpreter on the package at `source` (this
    checkout's) and these tests, compiling into `cache_dir` under XLA `flags`."""
    environment = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(cache_dir), "XLA_FLAGS": flags,
                   "PYTHONPATH": os.pathsep.join((str(source), str(REPO_ROOT / "tests")))}
    run = subprocess.run([sys.executable, "-c", script, *arguments], env=environment, check=False,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.splitlines()[-1])


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


def worker_parser(modes: Mapping[str, Callable], doc: str | None) -> argparse.ArgumentParser:
    """A worker's command line: the `modes` it runs, the file its JSON result goes to, and its place in
    the pool (`prepare_worker`). A worker adds its modes' own options."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("mode", choices=sorted(modes))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinator", help="host:port of the jax.distributed service")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--fsdp-size", type=int, default=1)
    return parser


def serve_mode(modes: Mapping[str, Callable], args) -> None:
    """In a worker: join the pool and write the mode's result as JSON to `args.out`."""
    prepare_worker(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(modes[args.mode](args)))


@contextlib.contextmanager
def serving(handle):
    """A local HTTP server answering each GET with `handle(request)`, and its address."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            handle(self)

        def log_message(self, *arguments):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


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
    return groups.start(command, cwd=REPO_ROOT, env={**worker_env(devices), **(environment or {})},
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def spawn(mode: str, out: Path, processes: int = 1, process_id: int = 0,
          coordinator: str | None = None, devices: int | None = None, *,
          script: Path = WORKER, environment: Mapping[str, str] | None = None,
          **flags) -> subprocess.Popen[str]:
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
    return start_process(command, MESH_DEVICES // processes if devices is None else devices,
                         environment=environment)


def terminate(process: subprocess.Popen[str]) -> None:
    """Kill the session even if its leader already exited and left descendants,
    and return once none of them still runs (tools/armada/child_groups.py's `stop`)."""
    running = groups.stop(process, settle=120.0)
    assert not running, f"session {process.pid} still runs {running} after SIGKILL"


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


def assert_exits_cleanly(program: str) -> None:
    """`program` runs to its end in a fresh CPU interpreter over this checkout's source and exits 0;
    its output is the failure's message. A worker left running at exit aborts it in finalization."""
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=300, env={**os.environ, "JAX_PLATFORMS": "cpu",
                                            "PYTHONPATH": str(REPO_ROOT / "src")})
    assert done.returncode == 0, done.stdout + done.stderr


def dew_train(directory, environment: Mapping[str, str], *arguments: str) -> None:
    """`dew train *arguments` run from `directory` in `environment`, as a user types it; it must succeed."""
    finished = subprocess.run([sys.executable, "-m", "dew.cli.main", "train", *arguments],
                              cwd=directory, env=environment, capture_output=True, text=True, timeout=900)
    assert finished.returncode == 0, f"--- stdout ---\n{finished.stdout}\n--- stderr ---\n{finished.stderr}"


def logged_losses(run_directory) -> list[float]:
    """The training losses a run in `run_directory` logged, in step order (its tracking's scalars)."""
    rows = (Path(run_directory) / "tracking" / "scalars.jsonl").read_text().splitlines()
    scalars = [json.loads(row)["scalars"] for row in rows]
    return [row["train/loss"] for row in scalars if "train/loss" in row]
