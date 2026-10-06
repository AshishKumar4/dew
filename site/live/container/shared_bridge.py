"""Relay authenticated sessions to private Kernel Gateway contexts."""

import asyncio
import hmac
import json
import os
import re
import time
import urllib.request
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from kernel_outputs import outputs_of
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

MAX_CODE = 100_000
MAX_OUTPUT = 2_000_000
PRELOAD = ("from model_client import CFG, DPMSolverMultistep, EulerAncestral, Heun, "
           "from_pretrained, text_model")


class Context:
    def __init__(self, identifier, channels):
        self.identifier, self.channels = identifier, channels
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        self.connected = False
        self.setup = 0

    async def execute(self, code, send):
        async with self.lock:
            self.last_used = time.monotonic()
            identifier = str(uuid.uuid4())
            await self.channels.send(json.dumps({
                "header": {"msg_id": identifier, "msg_type": "execute_request", "session": self.identifier,
                           "username": "visitor", "date": datetime.now(UTC).isoformat(),
                           "version": "5.3"},
                "parent_header": {}, "metadata": {}, "channel": "shell",
                "content": {"code": code, "silent": False, "store_history": False,
                            "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            }))
            reply, idle, size = None, False, 0
            while reply is None or not idle:
                message = json.loads(await asyncio.wait_for(self.channels.recv(), timeout=95))
                if message.get("parent_header", {}).get("msg_id") != identifier:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "execute_reply":
                    reply = {"status": content["status"], "count": content.get("execution_count")}
                elif kind == "status" and content["execution_state"] == "idle":
                    idle = True
                else:
                    for output in outputs_of(kind, content):
                        size += len(json.dumps(output))
                        if size > MAX_OUTPUT:
                            raise ValueError("this cell printed more than the live output allowance")
                        await send(output)
            self.last_used = time.monotonic()
            return reply


class Gateway:
    def __init__(self):
        self.token = Path("/run/dew/gateway-token").read_text()
        self.secret = os.environ["DEW_SHARED_SECRET"]
        self.contexts = {}
        self.allocating = asyncio.Lock()
        self.started = time.monotonic()
        self.idle = int(os.environ.get("DEW_LIVE_IDLE_SECONDS", "300"))
        self.wall = int(os.environ.get("DEW_LIVE_WALL_SECONDS", "1200"))

    async def api(self, path, data=None, method=None):
        def call():
            body = None if data is None else json.dumps(data).encode()
            request = urllib.request.Request("http://127.0.0.1:8890" + path, data=body, method=method,
                                             headers={"Authorization": "token " + self.token,
                                                      "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        return await asyncio.to_thread(call)

    async def create(self, session):
        async with self.allocating:
            if session in self.contexts:
                return self.contexts[session]
            if len(self.contexts) >= 8:
                raise ValueError("the shared container has no free Python contexts")
            started = time.monotonic()
            kernel = await self.api("/api/kernels", {"name": "python3"})
            channels = None
            try:
                url = f"ws://127.0.0.1:8890/api/kernels/{kernel['id']}/channels"
                channels = await connect(url, max_size=MAX_OUTPUT,
                                         additional_headers={"Authorization": "token " + self.token})
                context = Context(kernel["id"], channels)
                result = await context.execute(PRELOAD, lambda _: asyncio.sleep(0))
                if result["status"] != "ok":
                    raise RuntimeError("the isolated Python context could not load its inference client")
                context.setup = time.monotonic() - started
                self.contexts[session] = context
                return context
            except Exception:
                if channels:
                    await channels.close()
                await self.api(f"/api/kernels/{kernel['id']}", method="DELETE")
                raise

    async def close(self, session):
        context = self.contexts.pop(session, None)
        if context:
            await context.channels.close()
            await self.api(f"/api/kernels/{context.identifier}", method="DELETE")

    async def http(self, connection, request):
        if request.path == "/health":
            ready = Path("/run/dew/model/ready").exists()
            return Response(200 if ready else 503, "OK" if ready else "Warming up", Headers(), b"")
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + self.secret):
            return Response(403, "Forbidden", Headers(), b"")
        match = re.fullmatch(r"/contexts/([0-9a-f-]{36})(?:/(ws|status|close))?", request.path)
        if not match:
            return Response(404, "Not found", Headers(), b"")
        if match[2] == "ws":
            return None
        if match[2] == "status":
            known = match[1] in self.contexts
            return Response(200 if known else 404, "OK" if known else "Not found", Headers(), b"")
        # Container RPCs use a small GET endpoint; WebSocket connections use /ws.
        try:
            if match[2] == "close":
                await self.close(match[1])
            else:
                await self.create(match[1])
            return Response(200, "OK", Headers(), b"")
        except Exception as error:
            return Response(503, "Unavailable", Headers(), str(error).encode())

    async def websocket(self, socket):
        session = socket.request.path.split("/")[2]
        context = await self.create(session)
        if context.connected:
            await socket.close(1008, "this context is already connected")
            return
        context.connected = True
        await socket.send(json.dumps({"type": "ready", "uptime": time.monotonic() - self.started,
                                      "setup": context.setup,
                                      "dew": Path("/opt/live/dew-commit").read_text().strip()}))
        executing = None

        async def run(message):
            try:
                result = await context.execute(message["code"],
                    lambda output: socket.send(json.dumps({"id": message["id"], **output})))
                await socket.send(json.dumps({"id": message["id"], "type": "done", **result}))
            except Exception as error:
                with suppress(OSError):
                    await self.api(f"/api/kernels/{context.identifier}/interrupt", {}, "POST")
                await socket.send(json.dumps({"id": message["id"], "type": "error",
                                              "ename": type(error).__name__,
                                              "evalue": str(error), "traceback": []}))
                await socket.send(json.dumps({"id": message["id"], "type": "done",
                                              "status": "error", "count": None}))

        try:
            deadline = time.monotonic() + self.wall
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=min(15, deadline - time.monotonic()))
                except TimeoutError:
                    if ((not executing or executing.done()) and
                            time.monotonic() - context.last_used > self.idle):
                        break
                    continue
                message = json.loads(raw)
                context.last_used = time.monotonic()
                if message.get("op") == "interrupt":
                    await self.api(f"/api/kernels/{context.identifier}/interrupt", {}, "POST")
                elif message.get("op") == "execute":
                    if executing and not executing.done():
                        raise ValueError("a cell is already running in this context")
                    if (not isinstance(message.get("code"), str) or len(message["code"]) > MAX_CODE or
                            not isinstance(message.get("id"), str) or len(message["id"]) > 128):
                        raise ValueError("invalid or oversized cell")
                    executing = asyncio.create_task(run(message))
                else:
                    raise ValueError("unsupported kernel operation")
        finally:
            if executing and not executing.done():
                executing.cancel()
                with suppress(asyncio.CancelledError):
                    await executing
            await self.close(session)

    async def sweep(self):
        while True:
            await asyncio.sleep(15)
            for session, context in list(self.contexts.items()):
                if not context.connected and time.monotonic() - context.last_used > self.idle:
                    await self.close(session)


async def main():
    gateway = Gateway()
    async with serve(gateway.websocket, "0.0.0.0", 8888, process_request=gateway.http,
                     max_size=MAX_CODE * 4, ping_interval=20, ping_timeout=20):
        await gateway.sweep()


if __name__ == "__main__":
    asyncio.run(main())
