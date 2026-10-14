"""The live bridge (site/live/container/shared_bridge.py) over real sockets: a page's WebSocket, as the
Worker forwards it, to a stand-in for the Kernel Gateway that speaks its REST and channel protocol."""
import asyncio
import importlib.util
import json
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
    """The Kernel Gateway's REST and channel protocol over a set of kernels. Creating one takes
    `creating` seconds and running its preload `preloading`; a cell whose code is `sleep N` runs N
    seconds, or until the kernel is interrupted. `live` holds the kernels created and not deleted, and
    `created` counts every one."""

    def __init__(self, creating=0.0, preloading=0.0):
        self.creating, self.preloading = creating, preloading
        self.live, self.interrupts, self.created = set(), {}, 0

    async def create(self, request):
        await asyncio.sleep(self.creating)
        identifier = str(uuid.uuid4())
        self.live.add(identifier)
        self.created += 1
        self.interrupts[identifier] = asyncio.Event()
        return web.json_response({"id": identifier, "name": (await request.json())["name"]})

    async def delete(self, request):
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
                await asyncio.sleep(self.preloading)
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


@pytest.mark.parametrize("phase", ["creating", "preloading"])
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
            kernels.creating, kernels.preloading = 1.0, 1.0
            await socket.send(json.dumps({"op": "execute", "id": "fresh", "cell": "b", "code": "print(2)"}))
            await asyncio.sleep(0.5 if phase == "creating" else 1.5)
            if ending == "restart":
                await socket.send(json.dumps({"op": "restart", "id": "again", "cell": "b"}))
                assert (await answer(socket, "fresh"))[-1]["status"] == "aborted"
                assert (await answer(socket, "again"))[-1]["status"] == "ok"
            await socket.close()
            await asyncio.sleep(1.5)
            await settled(kernels)
            assert kernels.live == set()
            assert kernels.created >= 2, "the second cell's kernel was being made"
            kernels.creating, kernels.preloading = 0.0, 0.0
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
                    await asyncio.wait_for(again.recv(), 10)
            assert 3.5 < time.monotonic() - began < 6.0
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
