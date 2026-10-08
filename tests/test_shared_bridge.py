"""One page's cells run at the same time, each in a Python context of its own."""
import asyncio
import importlib.util
import json
import sys
import time
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "site/live/container"


class Closed(Exception):
    pass


class Channels:
    async def close(self):
        pass


class Context:
    """A kernel that prints which context ran the code, and holds `wait` until released."""

    def __init__(self, name):
        self.identifier, self.channels = name, Channels()
        self.last_used, self.setup = time.monotonic(), 0
        self.release = asyncio.Event()

    async def execute(self, code, send):
        await send({"type": "stream", "name": "stdout", "text": self.identifier})
        if code == "wait":
            await self.release.wait()
        return {"status": "ok", "count": 1}


class Socket:
    def __init__(self, session):
        self.request = type("Request", (), {"path": f"/contexts/{session}/ws"})()
        self.incoming, self.sent = asyncio.Queue(), []

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        message = await self.incoming.get()
        if message is None:
            raise Closed
        return json.dumps(message)

    async def close(self, code, reason):
        self.sent.append({"type": "closed", "code": code})

    def done(self, identifier):
        return any(m.get("id") == identifier and m["type"] == "done" for m in self.sent)


async def until(condition, handler):
    """Wait for `condition`, raising what the connection's handler raised if it ends first."""
    while not condition():
        if handler.done():
            handler.result()
        await asyncio.sleep(0)


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    # The bridge's own network ends, which these tests replace, come from websockets.
    names = {"websockets": None, "websockets.asyncio": None, "websockets.asyncio.client": "connect",
             "websockets.asyncio.server": "serve", "websockets.datastructures": "Headers",
             "websockets.http11": "Response"}
    for name, attribute in names.items():
        stub = types.ModuleType(name)
        if attribute:
            setattr(stub, attribute, None)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location("shared_bridge", ROOT / "shared_bridge.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    gateway = bridge.Gateway("token", "secret", "0" * 40)
    started = []

    async def start(session, kernel="python3"):
        started.append(Context(f"{'train' if kernel == 'dew-train' else 'kernel'}-{len(started)}"))
        return started[-1]

    async def api(path, data=None, method=None):
        return None

    gateway.start, gateway.api, gateway.kernels = start, api, started
    return gateway


def test_two_cells_run_at_once_in_their_own_contexts_and_close_with_the_page(gateway):
    async def scenario():
        session = str(uuid.uuid4())
        await gateway.create(session)
        socket = Socket(session)
        handler = asyncio.create_task(gateway.websocket(socket))
        await socket.incoming.put({"op": "execute", "id": "1", "code": "wait", "cell": "text"})
        await socket.incoming.put({"op": "execute", "id": "2", "code": "now", "cell": "image"})
        await until(lambda: socket.done("2"), handler)
        assert not socket.done("1")
        outputs = {m["id"]: m["text"] for m in socket.sent if m.get("type") == "stream"}
        # The text cell takes the context made ready with the session; the image cell starts one.
        assert outputs == {"1": "kernel-0", "2": "kernel-1"}
        gateway.kernels[0].release.set()
        await until(lambda: socket.done("1"), handler)
        await socket.incoming.put(None)
        with pytest.raises(Closed):
            await handler
        assert session not in gateway.contexts and session not in gateway.connected

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_a_cell_without_a_name_is_the_page_only_cell_and_runs_one_at_a_time(gateway):
    async def scenario():
        session = str(uuid.uuid4())
        await gateway.create(session)
        socket = Socket(session)
        handler = asyncio.create_task(gateway.websocket(socket))
        await socket.incoming.put({"op": "execute", "id": "1", "code": "wait"})
        await socket.incoming.put({"op": "execute", "id": "2", "code": "now"})
        with pytest.raises(ValueError, match="already running"):
            await handler
        assert len(gateway.kernels) == 1 and session not in gateway.contexts

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_training_cells_take_turns_in_fresh_contexts(gateway):
    """Three pages train at once: the first runs, the others wait, told how many runs are ahead;
    the third, stopped while it waits, ends without running, and the second runs in a fresh
    context once the first ends."""
    def train(identifier, code):
        return {"op": "execute", "id": identifier, "code": code, "cell": "hero", "kernel": "train"}

    def ahead(socket):
        return [json.loads(m["text"])["dew-wait"]["ahead"] for m in socket.sent
                if m.get("type") == "display" and "dew-wait" in m.get("text", "")]

    def trainers():
        return [context for context in gateway.kernels if context.identifier.startswith("train")]

    async def scenario():
        pages = [str(uuid.uuid4()) for _ in range(3)]
        sockets = [Socket(page) for page in pages]
        handlers = []
        for page, socket in zip(pages, sockets, strict=True):
            await gateway.create(page)
            handlers.append(asyncio.create_task(gateway.websocket(socket)))
        first, second, third = sockets
        await first.incoming.put(train("a", "wait"))
        await until(lambda: trainers(), handlers[0])
        await second.incoming.put(train("b", "now"))
        await third.incoming.put(train("c", "now"))
        await until(lambda: ahead(second) == [1] and ahead(third) == [2], handlers[1])
        await third.incoming.put({"op": "interrupt", "cell": "hero"})
        await until(lambda: third.done("c"), handlers[2])
        assert [m["status"] for m in third.sent if m.get("type") == "done"] == ["aborted"]
        trainers()[0].release.set()
        await until(lambda: first.done("a") and second.done("b"), handlers[1])
        assert len(trainers()) == 2 and not gateway.training and not gateway.waiting
        assert [m["text"] for m in second.sent if m.get("type") == "stream"] == [trainers()[1].identifier]
        for socket in sockets:
            await socket.incoming.put(None)
        for handler in handlers:
            with pytest.raises(Closed):
                await handler

    asyncio.run(asyncio.wait_for(scenario(), 10))
