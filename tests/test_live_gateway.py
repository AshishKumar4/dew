"""The live container's Kernel Gateway (site/live/container/gateway_manager.py) as start-gateway.sh starts
it, its kernels plain Python ones: the real jupyter-kernelgateway, jupyter-server's culler and ipykernel."""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from jupyter_client.manager import KernelManager
from process_support import free_port
from websockets.asyncio.client import connect

CONTAINER = Path(__file__).resolve().parents[1] / "site" / "live" / "container"
TOKEN = "token"
PLAIN = '''from traitlets import default

from gateway_manager import LimitedMappingKernelManager


class Plain(LimitedMappingKernelManager):
    """The live gateway's kernels, each a plain Python kernel rather than an isolated guest's."""

    @default("kernel_manager_class")
    def _kernel_manager_class_default(self):
        return "kernel_gateway.services.kernels.manager.KernelGatewayIOLoopKernelManager"
'''
"""The gateway's mapping manager with the isolation left out: a guest's needs root, its uids and mounts."""


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


def test_a_kernel_no_connection_holds_is_culled_once_idle_for_the_live_contracts_idle_time(tmp_path):
    """A kernel no one connects to, as one whose create the bridge gave up on while the gateway went on, is
    culled by jupyter-server's own culler once idle for the live contract's idle time (DEW_LIVE_IDLE_SECONDS),
    within a tenth of it more, and not before; a kernel a channel holds, as the bridge's contexts do, stays
    (ledger 843)."""
    idle = 4
    (tmp_path / "plain.py").write_text(PLAIN)
    port, log = free_port(), tmp_path / "gateway.log"
    gateway = subprocess.Popen(
        [sys.executable, "-m", "kernel_gateway", "--KernelGatewayApp.kernel_manager_class=plain.Plain",
         "--KernelGatewayApp.ip=127.0.0.1", f"--KernelGatewayApp.port={port}"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(CONTAINER)]),
             "KG_AUTH_TOKEN": TOKEN, "DEW_LIVE_IDLE_SECONDS": str(idle),
             "JUPYTER_RUNTIME_DIR": str(tmp_path)},
        stdout=subprocess.DEVNULL, stderr=log.open("w"))
    try:
        deadline = time.monotonic() + 60
        while not serving(port):
            assert gateway.poll() is None and time.monotonic() < deadline, log.read_text()
            time.sleep(0.2)

        async def scenario() -> float:
            """Seconds from the unconnected kernel's last activity to the gateway holding it no more."""
            held = created(port)
            async with connect(f"ws://127.0.0.1:{port}/api/kernels/{held}/channels",
                               additional_headers={"Authorization": f"token {TOKEN}"}):
                left = created(port)
                deadline = time.monotonic() + 3 * idle
                while (model := kernel(port, left)) is not None:
                    last = datetime.fromisoformat(model["last_activity"]).timestamp()
                    assert time.monotonic() < deadline, "the idle kernel no connection holds stayed"
                    await asyncio.sleep(0.1)
                gone = time.time()
                assert kernel(port, held) is not None
            return gone - last

        # Culled on the culler's first look past the idle time, which comes every tenth of it, and gone from
        # the gateway once jupyter-client's shutdown of it returns, within its own wait.
        shutdown = KernelManager.shutdown_wait_time.default_value
        assert idle <= asyncio.run(scenario()) <= idle + max(1, idle // 10) + shutdown
    finally:
        gateway.terminate()
        gateway.wait(30)
