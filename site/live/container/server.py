"""One Jupyter kernel behind one WebSocket, for the "Run live" buttons on dewml.dev.

The Worker in front of this container checks the visitor and forwards a single
WebSocket to /ws. This server starts an IPython kernel as the unprivileged user
`kernel`, runs the cells the page sends, one at a time, and streams back what
they print and plot. The container holds no secrets and has no network access.

The server exits, which stops the container, when the page disconnects, after
DEW_LIVE_IDLE_SECONDS without a request while no cell runs, or after
DEW_LIVE_WALL_SECONDS in all. The kernel runs under RLIMIT_CPU of
DEW_LIVE_CPU_SECONDS, so a runaway cell kills the kernel and not the server.

Messages from the page, as JSON:
    {"op": "execute", "id": <any>, "code": <str>}
    {"op": "interrupt"} and {"op": "restart"}
Messages to the page, each with the id of the cell they belong to:
    {"type": "stream", "name": "stdout" | "stderr", "text": <str>}
    {"type": "display", "text": <str>, "png": <base64 str>}   (either key may be absent)
    {"type": "error", "ename", "evalue", "traceback": [<str>]}
    {"type": "clear"}
    {"type": "done", "status": "ok" | "error" | "aborted", "count": <int or null>}
and, without an id: {"type": "ready"}, {"type": "restarted"}, {"type": "closing", "reason": <str>}.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import time
from typing import Any

from jupyter_client import AsyncKernelManager
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

IDLE_SECONDS = int(os.environ.get("DEW_LIVE_IDLE_SECONDS", "300"))
WALL_SECONDS = int(os.environ.get("DEW_LIVE_WALL_SECONDS", "1200"))
CPU_SECONDS = int(os.environ.get("DEW_LIVE_CPU_SECONDS", "900"))
CONNECT_SECONDS = 60  # a container nobody connects to within a minute exits
MAX_CODE = 100_000  # characters in one cell
MAX_OUTPUT = 2_000_000  # characters of output from one cell; the rest is dropped
MAX_MESSAGE = 900_000  # characters in one WebSocket message; Workers relay at most 1 MiB
KERNEL_USER = pwd.getpwnam("kernel")
WORKDIR = os.path.join(KERNEL_USER.pw_dir, "work")


class SandboxedKernelManager(AsyncKernelManager):
    """Starts the kernel as the `kernel` user, under CPU and process limits."""

    async def _async_launch_kernel(self, kernel_cmd: list[str], **kw: Any) -> None:
        os.chown(self.connection_file, KERNEL_USER.pw_uid, KERNEL_USER.pw_gid)
        limited = ["prlimit", f"--cpu={CPU_SECONDS}", "--nproc=512", "--", *kernel_cmd]
        env = {
            "HOME": KERNEL_USER.pw_dir,
            "PATH": os.environ["PATH"],
            "LANG": "C.UTF-8",
            "JAX_PLATFORMS": "cpu",
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "PYTHONUNBUFFERED": "1",
            # Two notices about this image rather than the reader's code: it has no
            # PyTorch, and no ipywidgets for tqdm's notebook progress bars.
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "PYTHONWARNINGS": "ignore:IProgress not found",
        }
        await super()._async_launch_kernel(
            limited, **{**kw, "env": env, "cwd": WORKDIR},
            user=KERNEL_USER.pw_uid, group=KERNEL_USER.pw_gid, extra_groups=[],
        )


def outputs_of(kind: str, content: dict[str, Any]) -> list[dict[str, Any]]:
    """The page messages for one kernel output, each small enough to relay."""
    if kind == "stream":
        text = content["text"]
        return [{"type": "stream", "name": content["name"], "text": text[i:i + MAX_MESSAGE]}
                for i in range(0, len(text), MAX_MESSAGE)]
    if kind in ("display_data", "execute_result", "update_display_data"):
        data = content.get("data", {})
        out: dict[str, Any] = {"type": "display"}
        if "image/png" in data:
            if len(data["image/png"]) <= MAX_MESSAGE:
                out["png"] = data["image/png"]
            else:
                out["text"] = "[An image too large to send here was not shown.]"
        if "text/plain" in data and "text" not in out:
            out["text"] = data["text/plain"][:MAX_MESSAGE]
        return [out] if len(out) > 1 else []
    if kind == "error":
        return [{"type": "error", "ename": content["ename"], "evalue": content["evalue"][:MAX_MESSAGE // 4],
                 "traceback": [line[:MAX_MESSAGE // 4] for line in content["traceback"][-40:]]}]
    if kind == "clear_output":
        return [{"type": "clear"}]
    return []


class Session:
    """The one page connected to this container, and its kernel."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.last_request = self.started
        self.busy = False
        self.generation = 0  # counts kernel restarts; a cell from an older kernel is aborted
        self.manager = SandboxedKernelManager(kernel_name="python3")
        self.client: Any = None
        self.kernel_ready = asyncio.Event()
        self.socket: ServerConnection | None = None
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed = asyncio.Event()

    async def start_kernel(self) -> None:
        await self.manager.start_kernel()
        self.client = self.manager.client()
        self.client.start_channels()
        await self.client.wait_for_ready(timeout=60)
        self.kernel_ready.set()

    async def send(self, message: dict[str, Any]) -> None:
        if self.socket is not None:
            try:
                await self.socket.send(json.dumps(message))
            except ConnectionClosed:
                pass

    async def execute(self, cell: Any, code: str) -> None:
        """Run one cell and forward its outputs until the kernel reports it idle."""
        generation = self.generation
        self.busy = True
        sent = 0
        truncated = False
        status = "ok"
        count = None
        msg_id = self.client.execute(code, store_history=True, allow_stdin=False)
        while True:
            try:
                message = await self.client.get_iopub_msg(timeout=1)
            except Exception:
                message = None
            if self.generation != generation:
                status = "aborted"
                break
            if message is None:
                if not await self.manager.is_alive():
                    status = "error"
                    await self.send({"id": cell, "type": "error", "ename": "KernelDied",
                                     "evalue": "The kernel stopped, most likely out of memory or CPU time. Restart it to go on.",
                                     "traceback": []})
                    break
                continue
            if message["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "execute_input":
                count = content.get("execution_count")
            if kind == "error":
                status = "error"
            for out in outputs_of(kind, content):
                size = sum(len(v) for v in out.values() if isinstance(v, str))
                if sent + size > MAX_OUTPUT:
                    if not truncated:
                        truncated = True
                        await self.send({"id": cell, "type": "stream", "name": "stderr",
                                         "text": "\n[The rest of this cell's output is not shown.]\n"})
                    continue
                sent += size
                await self.send({"id": cell, **out})
        if status != "aborted":
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    reply = await self.client.get_shell_msg(timeout=max(0.1, deadline - time.monotonic()))
                except Exception:
                    break
                if reply["parent_header"].get("msg_id") == msg_id:
                    status = reply["content"].get("status", status)
                    count = reply["content"].get("execution_count", count)
                    break
        await self.send({"id": cell, "type": "done", "status": status, "count": count})
        self.busy = False
        self.last_request = time.monotonic()

    async def run_queue(self) -> None:
        await self.kernel_ready.wait()
        while True:
            request = await self.queue.get()
            await self.execute(request.get("id"), request["code"])

    async def restart(self) -> None:
        self.generation += 1
        while not self.queue.empty():
            dropped = self.queue.get_nowait()
            await self.send({"id": dropped.get("id"), "type": "done", "status": "aborted", "count": None})
        await self.manager.restart_kernel(now=True)
        await self.client.wait_for_ready(timeout=60)
        await self.send({"type": "restarted"})

    async def watch(self) -> None:
        """Close the session at the wall-clock limit, or when it sits idle."""
        while not self.closed.is_set():
            await asyncio.sleep(5)
            now = time.monotonic()
            if now - self.started > WALL_SECONDS:
                await self.close("time")
            elif self.socket is None and now - self.started > CONNECT_SECONDS:
                await self.close("unused")
            elif not self.busy and self.queue.empty() and now - self.last_request > IDLE_SECONDS:
                await self.close("idle")

    async def close(self, reason: str) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        await self.send({"type": "closing", "reason": reason})
        if self.socket is not None:
            await self.socket.close(4000, reason)

    async def handle(self, socket: ServerConnection) -> None:
        if self.socket is not None or self.closed.is_set():
            await socket.close(4009, "this container already has a session")
            return
        self.socket = socket
        self.last_request = time.monotonic()
        runner = asyncio.create_task(self.run_queue())
        try:
            await self.kernel_ready.wait()
            await self.send({"type": "ready"})
            async for raw in socket:
                self.last_request = time.monotonic()
                try:
                    request = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                op = request.get("op") if isinstance(request, dict) else None
                if op == "execute" and isinstance(request.get("code"), str):
                    if len(request["code"]) > MAX_CODE:
                        await self.send({"id": request.get("id"), "type": "error", "ename": "CellTooLarge",
                                         "evalue": f"A cell can hold at most {MAX_CODE:,} characters.", "traceback": []})
                        await self.send({"id": request.get("id"), "type": "done", "status": "error", "count": None})
                        continue
                    self.queue.put_nowait(request)
                elif op == "interrupt":
                    await self.manager.interrupt_kernel()
                elif op == "restart":
                    await self.restart()
        except ConnectionClosed:
            pass
        finally:
            runner.cancel()
            self.closed.set()


def ping(connection: ServerConnection, request: Request) -> Response | None:
    """Answer the platform's readiness check and any other plain HTTP request."""
    if request.path != "/ws" or request.headers.get("Upgrade", "").lower() != "websocket":
        return connection.respond(200, "ok\n")
    return None


async def main() -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    os.chown(WORKDIR, KERNEL_USER.pw_uid, KERNEL_USER.pw_gid)
    session = Session()
    kernel = asyncio.create_task(session.start_kernel())
    async with serve(session.handle, "0.0.0.0", 8888, process_request=ping, max_size=2**20,
                     ping_interval=20, ping_timeout=20):
        watcher = asyncio.create_task(session.watch())
        await session.closed.wait()
        watcher.cancel()
        await asyncio.sleep(0.5)
    await kernel
    await session.manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    asyncio.run(main())
