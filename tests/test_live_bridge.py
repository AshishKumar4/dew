"""The live bridge (site/live/container/shared_bridge.py) over real sockets: a page's WebSocket, as the
Worker forwards it, to a stand-in for the Kernel Gateway that speaks its REST and channel protocol."""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from aiohttp import web

CONTAINER = Path(__file__).resolve().parents[1] / "site/live/container"
SECRET = "bridge-secret"


@pytest.fixture(scope="module")
def bridge():
    sys.path.insert(0, str(CONTAINER))
    spec = importlib.util.spec_from_file_location("shared_bridge", CONTAINER / "shared_bridge.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    yield loaded
    sys.path.remove(str(CONTAINER))


class Kernels:
    """The Kernel Gateway's REST and channel protocol over a set of kernels. A phase in `held`
    ("create", or "preload", a context loading its client) sets its `entered` event and waits for
    its `release`; a cell whose code is `sleep N` runs N seconds, or until the kernel is interrupted.
    The next `failing` deletes fail. `live` holds the kernels created and not deleted, and `created`
    counts every one."""

    def __init__(self):
        self.held, self.failing = set(), 0
        self.entered = {phase: asyncio.Event() for phase in ("create", "preload")}
        self.release = {phase: asyncio.Event() for phase in ("create", "preload")}
        self.live, self.interrupts, self.created = set(), {}, 0

    async def phase(self, name):
        if name in self.held:
            self.entered[name].set()
            await self.release[name].wait()

    async def create(self, request):
        await self.phase("create")
        identifier = str(uuid.uuid4())
        self.live.add(identifier)
        self.created += 1
        self.interrupts[identifier] = asyncio.Event()
        return web.json_response({"id": identifier, "name": (await request.json())["name"]})

    async def delete(self, request):
        if self.failing:
            self.failing -= 1
            return web.Response(status=500)
        self.live.discard(request.match_info["kernel"])
        return web.Response(status=204)

    async def interrupt(self, request):
        self.interrupts[request.match_info["kernel"]].set()
        return web.Response(status=204)

    async def channels(self, request):
        kernel = request.match_info["kernel"]
        socket = web.WebSocketResponse(max_msg_size=0)
        await socket.prepare(request)
        async for message in socket:
            sent = json.loads(message.data)
            identifier, code = sent["header"]["msg_id"], sent["content"]["code"]

            def reply(kind, content, identifier=identifier):
                return socket.send_str(json.dumps({"msg_type": kind, "parent_header": {"msg_id": identifier},
                                                   "content": content}))

            status = "ok"
            if code.startswith(("import model_client", "import live_training")):
                await self.phase("preload")
            elif code.startswith("sleep "):
                self.interrupts[kernel].clear()
                try:
                    await asyncio.wait_for(self.interrupts[kernel].wait(), float(code.split()[1]))
                    status = "error"
                except TimeoutError:
                    pass
            else:
                await reply("stream", {"name": "stdout", "text": code})
            await reply("execute_reply", {"status": status, "execution_count": 1})
            await reply("status", {"execution_state": "idle"})
        return socket

    async def serve(self):
        app = web.Application()
        app.add_routes([web.post("/api/kernels", self.create),
                        web.delete("/api/kernels/{kernel}", self.delete),
                        web.post("/api/kernels/{kernel}/interrupt", self.interrupt),
                        web.get("/api/kernels/{kernel}/channels", self.channels)])
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, runner.addresses[0][1]


async def serving(bridge, kernels):
    """The bridge in front of `kernels`, as start-shared.sh serves it, and its port."""
    from websockets.asyncio.server import serve

    runner, port = await kernels.serve()
    gateway = bridge.Gateway("token", SECRET, "commit", kernels=f"127.0.0.1:{port}")
    server = await serve(gateway.websocket, "127.0.0.1", 0, process_request=gateway.http,
                         max_size=bridge.MAX_CODE * 4)
    return gateway, server, runner, server.sockets[0].getsockname()[1]


async def page(port, session, ends_in=60.0):
    """A page's socket to `session`, as the Worker forwards it: the bridge's secret, and the session's
    end `ends_in` seconds from now."""
    from websockets.asyncio.client import connect

    socket = await connect(f"ws://127.0.0.1:{port}/contexts/{session}/ws", additional_headers={
        "Authorization": "Bearer " + SECRET, "X-Dew-Deadline": str(int((time.time() + ends_in) * 1000))})
    assert json.loads(await socket.recv())["type"] == "ready"
    return socket


async def answer(socket, identifier):
    """What the bridge sends about request `identifier` until its done."""
    heard = []
    while not heard or heard[-1]["type"] != "done":
        message = json.loads(await asyncio.wait_for(socket.recv(), 30))
        if message.get("id") == identifier:
            heard.append(message)
    return heard


async def settled(kernels):
    """Wait for the bridge's deletes to reach the gateway, at most ten seconds."""
    for _ in range(100):
        if not kernels.live:
            return
        await asyncio.sleep(0.1)


def test_a_page_runs_or_queues_at_most_its_cells_of_either_kind(bridge, monkeypatch):
    """A page's training cells queue behind the host's one training slot, a page holding at most
    MAX_CELLS of them, running or waiting; one past it is refused at once, and the page carries
    on. Another page still runs its cells, and Stop on a queued one frees its place."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        gateway, server, runner, port = await serving(bridge, kernels)
        try:
            first = await page(port, str(uuid.uuid4()))
            for index in range(bridge.MAX_CELLS):
                await first.send(json.dumps({"op": "execute", "id": f"train-{index}", "cell": f"c{index}",
                                             "kernel": "train", "code": "sleep 30"}))
            for index in range(bridge.MAX_CELLS, bridge.MAX_CELLS + 20):
                await first.send(json.dumps({"op": "execute", "id": f"train-{index}", "cell": f"c{index}",
                                             "kernel": "train", "code": "sleep 30"}))
                refused = await answer(first, f"train-{index}")
                assert [message["type"] for message in refused] == ["error", "done"]
                assert "cells already" in refused[0]["evalue"]
            assert len(gateway.waiting) == bridge.MAX_CELLS - 1
            second = await page(port, str(uuid.uuid4()))
            await second.send(json.dumps({"op": "execute", "id": "print", "cell": "a", "code": "print(1)"}))
            assert [message["type"] for message in await answer(second, "print")] == ["stream", "done"]
            await first.send(json.dumps({"op": "interrupt", "cell": "c3"}))
            assert (await answer(first, "train-3"))[-1]["status"] == "aborted"
            await first.send(json.dumps({"op": "execute", "id": "train-next", "cell": "c99",
                                         "kernel": "train", "code": "sleep 30"}))
            await asyncio.sleep(0.5)
            assert len(gateway.waiting) == bridge.MAX_CELLS - 1, "the stopped cell's place went to the next"
            await first.close()
            await second.close()
            await settled(kernels)
            assert kernels.live == set()
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["create", "preload"])
@pytest.mark.parametrize("ending", ["disconnect", "restart"])
def test_a_context_ended_while_it_starts_leaves_no_kernel(bridge, monkeypatch, phase, ending):
    """A disconnect or a Restart that cancels a cell's context while the gateway creates its kernel,
    or while the kernel loads its client, deletes the kernel; the next cell starts as usual."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        _, server, runner, port = await serving(bridge, kernels)
        try:
            session = str(uuid.uuid4())
            socket = await page(port, session)
            # The page's first cell takes the context made ready as it connected; its second starts one.
            await socket.send(json.dumps({"op": "execute", "id": "first", "cell": "a", "code": "print(1)"}))
            await answer(socket, "first")
            kernels.held.add(phase)
            await socket.send(json.dumps({"op": "execute", "id": "fresh", "cell": "b", "code": "print(2)"}))
            await asyncio.wait_for(kernels.entered[phase].wait(), 30)
            if ending == "restart":
                await socket.send(json.dumps({"op": "restart", "id": "again", "cell": "b"}))
                # The cancelled start's kernel is deleted once the gateway has made it.
                kernels.held.clear()
                kernels.release[phase].set()
                assert (await answer(socket, "fresh"))[-1]["status"] == "aborted"
                assert (await answer(socket, "again"))[-1]["status"] == "ok"
            await socket.close()
            kernels.held.clear()
            kernels.release[phase].set()
            await settled(kernels)
            assert kernels.live == set()
            assert kernels.created == 2, "the second cell's kernel was the one being made"
            socket = await page(port, session)
            await socket.send(json.dumps({"op": "execute", "id": "next", "cell": "b", "code": "print(3)"}))
            assert [message["type"] for message in await answer(socket, "next")] == ["stream", "done"]
            await socket.close()
            await settled(kernels)
            assert kernels.live == set()
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_a_failed_cleanup_never_hides_why_a_start_ended(bridge, monkeypatch):
    """A disconnect while a context loads its client, whose kernel's delete then fails, still
    closes the page's session: its other contexts are deleted, and the session is free again."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        gateway, server, runner, port = await serving(bridge, kernels)
        try:
            session = str(uuid.uuid4())
            socket = await page(port, session)
            await socket.send(json.dumps({"op": "execute", "id": "first", "cell": "a", "code": "print(1)"}))
            await answer(socket, "first")
            kernels.held.add("preload")
            await socket.send(json.dumps({"op": "execute", "id": "fresh", "cell": "b", "code": "print(2)"}))
            await asyncio.wait_for(kernels.entered["preload"].wait(), 30)
            kernels.failing = 1
            await socket.close()
            kernels.release["preload"].set()
            for _ in range(100):
                if session not in gateway.connected and session not in gateway.contexts:
                    break
                await asyncio.sleep(0.1)
            assert session not in gateway.connected and session not in gateway.contexts
            assert len(kernels.live) == 1, "only the kernel whose delete failed is left"
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_every_connection_of_a_session_ends_at_the_session_end(bridge, monkeypatch):
    """The Worker names the session's end on each connection. A page that reconnects shortly before it
    is closed then, not a full allowance later, and one that connects after it is refused."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        _, server, runner, port = await serving(bridge, kernels)
        try:
            session, began = str(uuid.uuid4()), time.monotonic()
            first = await page(port, session, ends_in=4.0)
            await first.close()
            await asyncio.sleep(1.0)
            again = await page(port, session, ends_in=4.0 - (time.monotonic() - began))
            with pytest.raises(ConnectionClosed):
                while True:
                    await asyncio.wait_for(again.recv(), 60)
            # Closed at the session's end, not an allowance after the reconnect.
            assert 4.0 <= time.monotonic() - began < 30
            await settled(kernels)
            assert kernels.live == set()
            late = await connect(f"ws://127.0.0.1:{port}/contexts/{session}/ws", additional_headers={
                "Authorization": "Bearer " + SECRET, "X-Dew-Deadline": str(int(time.time() * 1000) - 1)})
            with pytest.raises(ConnectionClosed) as closed:
                await asyncio.wait_for(late.recv(), 10)
            assert closed.value.rcvd.code == 1008
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_the_context_smoke_loads_beside_the_bridge_it_reports_with(bridge):
    """benchmark_gateway.py, the preparation's required context smoke, imports and reports the host's
    load through the bridge's own probe."""
    spec = importlib.util.spec_from_file_location("benchmark_gateway", CONTAINER / "benchmark_gateway.py")
    smoke = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CONTAINER))
    try:
        spec.loader.exec_module(smoke)
    finally:
        sys.path.remove(str(CONTAINER))
    assert {"loadavg", "stalls", "cpu_ticks"} <= set(smoke.load())


def test_a_training_context_caps_the_batch_of_every_way_a_cell_loads_data(tmp_path):
    """In a training context (live_training.install), a bare source through dew.data.load, a recipe
    through its loader and a recipe's own load each read batches of BATCH rows, saying so first."""
    import numpy as np

    np.arange(400, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.arange(400, dtype=np.uint16).tofile(tmp_path / "val.bin")
    code = f"""
import json, sys
sys.path.insert(0, {str(CONTAINER)!r})
import live_training
live_training.install()
from dew.config.data import TextData, loader
from dew.data import Loading, Memory, load
windows = TextData(source={str(tmp_path)!r}, seq_len=4, stride=1)
options = dict(read="memory", loading=Loading(workers=0), validation_records=32)
bare = load(Memory([{{"id": index}} for index in range(320)]), batch=64, provision=False,
            loading=Loading(workers=0))
batches = [bare.batch, loader(windows, batch=64, **options).batch, windows.load(batch=64, **options).batch]
print(json.dumps(batches))
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-4000:]
    *said, batches = done.stdout.strip().splitlines()
    assert json.loads(batches) == [8, 8, 8]
    assert said == ["Live run: batch 64 -> 8, so it fits a shared 4-vCPU host."] * 3


def test_a_page_s_restarts_are_admitted_as_its_cells_are_and_end_with_it(bridge, monkeypatch):
    """A Restart starts a task as a cell does, which counts among the page's cells; a cell restarts
    once at a time; and a page that closes ends every task it started, its kernels deleted. Stopped
    cells leave nothing behind once they end."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)
    pages = []

    class Recorded(bridge.Page):
        def __init__(self, *args):
            super().__init__(*args)
            pages.append(self)
    monkeypatch.setattr(bridge, "Page", Recorded)

    async def scenario():
        kernels = Kernels()
        _, server, runner, port = await serving(bridge, kernels)
        try:
            socket = await page(port, str(uuid.uuid4()))
            for index in range(bridge.MAX_CELLS):
                await socket.send(json.dumps({"op": "execute", "id": f"run-{index}", "cell": f"c{index}",
                                              "code": "sleep 30"}))
            await asyncio.sleep(1)
            await socket.send(json.dumps({"op": "interrupt", "cell": "c0"}))
            # Each running cell restarted; their fresh contexts wait on the gateway.
            kernels.held.add("create")
            for index in range(bridge.MAX_CELLS):
                await socket.send(json.dumps({"op": "restart", "id": f"restart-{index}",
                                              "cell": f"c{index}"}))
            await asyncio.wait_for(kernels.entered["create"].wait(), 30)
            # The stopped cell ended in its error; the others, the restart aborted.
            ended = [(await answer(socket, f"run-{index}"))[-1]["status"]
                     for index in range(bridge.MAX_CELLS)]
            assert ended == ["error", *["aborted"] * (bridge.MAX_CELLS - 1)]
            # A restart of a restarting cell is refused, and the restarts count as the page's cells.
            await socket.send(json.dumps({"op": "restart", "id": "again", "cell": "c1"}))
            assert "restarting already" in (await answer(socket, "again"))[0]["evalue"]
            await socket.send(json.dumps({"op": "execute", "id": "more", "cell": "new", "code": "print(1)"}))
            assert "cells already" in (await answer(socket, "more"))[0]["evalue"]
            [held] = pages
            assert len(held.tasks) == bridge.MAX_CELLS
            assert held.restarting == {f"c{index}" for index in range(bridge.MAX_CELLS)}
            await socket.close()
            kernels.held.clear()
            kernels.release["create"].set()
            for _ in range(100):
                if not held.tasks:
                    break
                await asyncio.sleep(0.1)
            assert (held.tasks, held.stops, held.restarting) == (set(), set(), set())
            await settled(kernels)
            assert kernels.live == set()
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())
