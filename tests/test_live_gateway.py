"""The live container's Kernel Gateway (site/live/container/gateway_manager.py) as start-gateway.sh starts
it, its kernels plain Python ones: the real jupyter-kernelgateway, jupyter-server's culler and ipykernel."""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from jupyter_client.manager import KernelManager
from process_support import free_port
from websockets.asyncio.client import connect

CONTAINER = Path(__file__).resolve().parents[1] / "site" / "live" / "container"
TOKEN = "token"
PLAIN = '''import asyncio
import os
from pathlib import Path

from kernel_gateway.services.kernels.manager import KernelGatewayIOLoopKernelManager
from traitlets import default

from gateway_manager import LimitedMappingKernelManager


class Relaunching(KernelGatewayIOLoopKernelManager):
    """A plain kernel whose restart says it began (RELAUNCHED) and then, as DEW_TEST_RELAUNCH says, takes
    three seconds in its shutdown (`slow-stop`) or its relaunch (`slow`), or its relaunch raises (`fail`) or
    never ends (`hang`)."""

    launches = 0

    async def _async_cleanup_resources(self, restart=False):
        if restart and os.environ["DEW_TEST_RELAUNCH"] == "slow-stop":
            Path(os.environ["RELAUNCHED"]).touch()
            await asyncio.sleep(3)
        return await super()._async_cleanup_resources(restart=restart)

    async def _async_launch_kernel(self, kernel_cmd, **kwargs):
        self.launches += 1
        if self.launches > 1 and os.environ["DEW_TEST_RELAUNCH"] != "slow-stop":
            Path(os.environ["RELAUNCHED"]).touch()
            if os.environ["DEW_TEST_RELAUNCH"] == "fail":
                raise RuntimeError("the relaunch failed")
            await asyncio.sleep(3600 if os.environ["DEW_TEST_RELAUNCH"] == "hang" else 3)
        return await super()._async_launch_kernel(kernel_cmd, **kwargs)


class Plain(LimitedMappingKernelManager):
    """The live gateway's kernels, each a plain Python kernel rather than an isolated guest's."""

    @default("kernel_manager_class")
    def _kernel_manager_class_default(self):
        return "plain.Relaunching"
'''
"""The gateway's mapping manager with the isolation left out: a guest's needs root, its uids and mounts."""
OBSERVED = 2.0
"""Seconds a kernel's removal may take to be seen after its shutdown: the kernel's cleanup and removal from
the manager, a poll's 0.1 s and the HTTP call that reads it."""


def kernel(port: int, identifier: str) -> dict | None:
    """The gateway's model of kernel `identifier`, or None once it holds no such kernel."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/kernels/{identifier}",
                                     headers={"Authorization": f"token {TOKEN}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def created(port: int) -> str:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/kernels", data=b"{}", method="POST",
                                     headers={"Authorization": f"token {TOKEN}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["id"]


def serving(port: int) -> bool:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api",
                                     headers={"Authorization": f"token {TOKEN}"})
    try:
        with urllib.request.urlopen(request, timeout=5):
            return True
    except OSError:
        return False


def process(connection: Path) -> int:
    """The process of the kernel whose connection file is `connection`: of the processes running now, the one
    whose command names it, one that ends between the listing and its reading passed over (ledger 2691)."""

    def command(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return b""

    found, = [int(path.parent.name) for path in Path("/proc").glob("[0-9]*/cmdline")
              if str(connection).encode() in command(path)]
    return found


def deleted(port: int, identifier: str) -> int:
    """The gateway's status for deleting kernel `identifier`."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/kernels/{identifier}", method="DELETE",
                                     headers={"Authorization": f"token {TOKEN}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


@contextlib.contextmanager
def gateway(tmp_path: Path, idle: int, *options: str, relaunch: str = "slow") -> Iterator[int]:
    """The live gateway's mapping manager (`PLAIN`) served by the real Kernel Gateway on a free port, which it
    yields once it answers: `idle` its live idle time, `options` more of its configuration; its connection
    files in `tmp_path`."""
    (tmp_path / "plain.py").write_text(PLAIN)
    port, log = free_port(), tmp_path / "gateway.log"
    served = subprocess.Popen(
        [sys.executable, "-m", "kernel_gateway", "--KernelGatewayApp.kernel_manager_class=plain.Plain",
         "--KernelGatewayApp.ip=127.0.0.1", f"--KernelGatewayApp.port={port}", *options],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(CONTAINER)]),
             "KG_AUTH_TOKEN": TOKEN, "DEW_LIVE_IDLE_SECONDS": str(idle),
             "JUPYTER_RUNTIME_DIR": str(tmp_path), "DEW_TEST_RELAUNCH": relaunch,
             "RELAUNCHED": str(tmp_path / "relaunched")},
        stdout=subprocess.DEVNULL, stderr=log.open("w"))
    try:
        deadline = time.monotonic() + 60
        while not serving(port):
            assert served.poll() is None and time.monotonic() < deadline, log.read_text()
            time.sleep(0.2)
        yield port
    finally:
        served.terminate()
        served.wait(30)


def test_a_kernel_no_connection_holds_is_culled_once_idle_for_the_live_contracts_idle_time(tmp_path):
    """A kernel no one connects to, as one whose create the bridge gave up on while the gateway went on, is
    culled by jupyter-server's own culler once idle for the live contract's idle time (DEW_LIVE_IDLE_SECONDS),
    within a tenth of it more, and not before; a kernel a channel holds, as the bridge's contexts do,
    stays."""
    idle = 4
    # Culled on the culler's first look past the idle time, which comes every tenth of it, and gone from the
    # gateway once jupyter-client's shutdown of it returns, within its own wait; then its cleanup, the
    # manager's removal and this poll's own reading of it (`OBSERVED`).
    bound = idle + max(1, idle // 10) + KernelManager.shutdown_wait_time.default_value + OBSERVED
    with gateway(tmp_path, idle) as port:

        async def scenario() -> float:
            """Seconds from the unconnected kernel's last activity to the gateway holding it no more."""
            held = created(port)
            async with connect(f"ws://127.0.0.1:{port}/api/kernels/{held}/channels",
                               additional_headers={"Authorization": f"token {TOKEN}"}):
                left = created(port)
                while (model := kernel(port, left)) is not None:
                    last = datetime.fromisoformat(model["last_activity"]).timestamp()
                    assert time.time() - last <= bound, "the idle kernel no connection holds stayed"
                    await asyncio.sleep(0.1)
                gone = time.time()
                assert kernel(port, held) is not None
            return gone - last

        assert idle <= asyncio.run(scenario()) <= bound


def test_a_kernel_whose_restarts_failed_leaves_no_connection_file(tmp_path):
    """A kernel that dies and cannot be restarted, as one the guests' OOM killer stops under memory pressure,
    leaves the gateway with its connection file removed, as a kernel deleted does: none outlives its page,
    where jupyter-server dropping it left the file the relay smoke counts as a context left open (ledger
    843). Its restarts are given up at once here (`restart_limit` 0)."""
    with gateway(tmp_path, 300, "--KernelRestarter.restart_limit=0") as port:
        identifier = created(port)
        connection = tmp_path / f"kernel-{identifier}.json"
        assert connection.is_file()
        os.kill(process(connection), signal.SIGKILL)
        deadline = time.monotonic() + 60
        while kernel(port, identifier) is not None or connection.exists():
            assert time.monotonic() < deadline, f"the dead kernel's connection file stayed: {connection}"
            time.sleep(0.2)


@pytest.mark.parametrize("relaunch", ["slow-stop", "slow", "fail", "hang"])
def test_a_kernel_deleted_while_it_restarts_or_after_its_relaunch_failed_leaves_no_connection_file(
        tmp_path, relaunch):
    """A kernel deleted while it restarts, in the restart's shutdown or its relaunch, as the bridge deletes a
    context it hears restarting, ends once the restart settles, where the delete had failed and kept it; one
    whose relaunch raised, which no restarter ends, is released by its delete, and one whose relaunch hangs
    is released once the server's bound on a start passes (`kernel_info_timeout`). Either way no connection
    file is left (ledger 2516)."""
    options = ("--KernelRestarter.time_to_dead=0.5", "--MappingKernelManager.kernel_info_timeout=5")
    with gateway(tmp_path, 300, *options, relaunch=relaunch) as port:
        identifier = created(port)
        connection = tmp_path / f"kernel-{identifier}.json"
        os.kill(process(connection), signal.SIGKILL)
        deadline = time.monotonic() + 60
        while not (tmp_path / "relaunched").exists():
            assert time.monotonic() < deadline, "the restarter never relaunched the kernel"
            time.sleep(0.1)
        if relaunch == "fail":
            time.sleep(1)
        assert deleted(port, identifier) == 204
        assert kernel(port, identifier) is None and not connection.exists()


def test_a_restarted_kernel_keeps_its_guest_uid_and_a_shut_one_returns_it(tmp_path, monkeypatch):
    """A kernel's relaunch, a restart's, runs as the guest uid the kernel holds, and only its full shutdown
    returns that uid to the pool: each guard kill's restart had taken a fresh uid over the held one, and after
    about 75 in a container's life no new context found one (ledger 2517)."""
    monkeypatch.syspath_prepend(str(CONTAINER))
    import gateway_manager

    pool = gateway_manager._uids.qsize()
    kernel = gateway_manager.LimitedKernelManager(connection_file=str(tmp_path / "kernel-k.json"))
    uid = kernel._guest_uid()
    assert kernel._guest_uid() == uid and gateway_manager._uids.qsize() == pool - 1
    kernel.cleanup_connection_file()
    assert gateway_manager._uids.qsize() == pool and kernel._guest_uid() is not None
