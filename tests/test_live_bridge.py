"""The live bridge (site/live/container/shared_bridge.py) over real sockets: a page's WebSocket, as the
Worker forwards it, to a stand-in for the Kernel Gateway that speaks its REST and channel protocol."""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from aiohttp import web
from tool_support import run_from

CONTAINER = Path(__file__).resolve().parents[1] / "site/live/container"
SECRET = "bridge-secret"
FRAME = 4000


@pytest.fixture(scope="module")
def bridge():
    with run_from("shared_bridge", "site/live/container") as module:
        yield module


class Kernels:
    """The Kernel Gateway's REST and channel protocol over a set of kernels. A phase in `held`
    ("create", or "preload", a context loading its client) sets its `entered` event and waits for
    its `release`; a cell whose code is `sleep N` runs N seconds, or until the kernel is interrupted;
    one whose code is `panel N` shows a display under an id, as IPython's display handle does, and
    updates it N times with frames of FRAME characters, as the notebook panel does; `display N` shows
    N displays of FRAME characters under one id, and `redraw N` clears with wait=True and shows a
    frame N times, as a cell redrawing a plot does.
    The next `failing` deletes fail. `live` holds the kernels created and not deleted, and `created`
    counts every one."""

    def __init__(self):
        self.held, self.failing = set(), 0
        self.entered = {phase: asyncio.Event() for phase in ("create", "preload")}
        self.release = {phase: asyncio.Event() for phase in ("create", "preload")}
        self.live, self.interrupts, self.created = set(), {}, 0
        # The code of every cell a kernel ran, its client's loading aside.
        self.ran: list[str] = []

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
            if not code.startswith(("import model_client", "import live_training")):
                self.ran.append(code)
            if code.startswith(("import model_client", "import live_training")):
                await self.phase("preload")
            elif code.startswith("panel "):
                transient = {"display_id": "panel"}
                await reply("display_data", {"data": {"text/plain": "frame 0"}, "transient": transient})
                for frame in range(1, int(code.split()[1]) + 1):
                    text = f"frame {frame} ".ljust(FRAME, ".")
                    await reply("update_display_data", {"data": {"text/plain": text}, "transient": transient})
            elif code.startswith("display "):
                for _ in range(int(code.split()[1])):
                    await reply("display_data", {"data": {"text/plain": "x" * FRAME},
                                                 "transient": {"display_id": "same"}})
            elif code.startswith("redraw "):
                for _ in range(int(code.split()[1])):
                    await reply("clear_output", {"wait": True})
                    await reply("display_data", {"data": {"text/plain": "x" * FRAME}})
            elif code.startswith("sleep "):
                # Its seconds, unless an interrupt ends it first, as an error.
                self.interrupts[kernel].clear()
                waiting = asyncio.ensure_future(self.interrupts[kernel].wait())
                done, _ = await asyncio.wait({waiting}, timeout=float(code.split()[1]))
                waiting.cancel()
                status = "error" if done else "ok"
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


def test_a_display_updated_in_place_is_one_output_its_frames_replace(bridge, monkeypatch):
    """A training cell's panel is one display that IPython's handle updates eight times a second: the
    bridge sends each frame under the display's id, an update marked as one, and counts against the
    cell's output allowance what the page shows, so a long run's frames, past the allowance together,
    end the cell as it ends (7fb1f1bd3's preview: the finetune cell stopped at 101 s), as do a redraw's
    cleared frames; displays that share an id each count."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)
    frames = bridge.MAX_OUTPUT // FRAME * 2

    async def scenario():
        kernels = Kernels()
        _, server, runner, port = await serving(bridge, kernels)
        try:
            socket = await page(port, str(uuid.uuid4()))
            await socket.send(json.dumps({"op": "execute", "id": "train", "cell": "finetune",
                                          "kernel": "train", "code": f"panel {frames}"}))
            heard = await answer(socket, "train")
            assert heard[-1]["status"] == "ok", heard[-1]
            shown = [message for message in heard if message["type"] == "display"]
            assert len(shown) == frames + 1
            assert {message["display"] for message in shown} == {"panel"}
            assert [message.get("update", False) for message in shown] == [False] + [True] * frames
            assert shown[-1]["text"].startswith(f"frame {frames} ")
            # A cell that redraws, clearing as it shows each frame, shows one frame at a time.
            await socket.send(json.dumps({"op": "execute", "id": "redraw", "cell": "finetune",
                                          "kernel": "train", "code": f"redraw {frames}"}))
            heard = await answer(socket, "redraw")
            assert heard[-1]["status"] == "ok", heard[-1]
            assert heard[0] == {"id": "redraw", "type": "clear", "wait": True}
            # Every display the page appends counts, whatever id it shares with the others.
            await socket.send(json.dumps({"op": "execute", "id": "same", "cell": "finetune",
                                          "kernel": "train", "code": f"display {frames}"}))
            heard = await answer(socket, "same")
            assert heard[-2]["type"] == "error" and "output allowance" in heard[-2]["evalue"], heard[-2:]
            await socket.close()
            await settled(kernels)
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


def test_a_session_closed_while_its_first_context_starts_leaves_no_kernel(bridge, monkeypatch):
    """The Worker's close of a session while the gateway still creates its first context (the container's
    create and close RPCs, `Gateway.create` and `Gateway.close`) ends that creation: the create says the
    session was closed, the kernel the gateway then makes is deleted, and the session can start again."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        gateway, server, runner, _ = await serving(bridge, kernels)
        try:
            session = str(uuid.uuid4())
            kernels.held.add("create")
            created = asyncio.ensure_future(gateway.create(session))
            await asyncio.wait_for(kernels.entered["create"].wait(), 30)
            await gateway.close(session)
            kernels.held.clear()
            kernels.release["create"].set()
            with pytest.raises(RuntimeError):
                await created
            await settled(kernels)
            assert kernels.live == set() and session not in gateway.contexts
            assert None in await gateway.create(session)
            await gateway.close(session)
            await settled(kernels)
            assert kernels.live == set()
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["python3", "train"])
def test_a_cell_stopped_while_its_context_starts_never_runs(bridge, monkeypatch, kind):
    """The page's Stop of a cell, a training one too, while its context loads its client: the cell ends
    aborted once the context is up, its code never sent to the kernel, and the next cell runs as usual."""
    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)

    async def scenario():
        kernels = Kernels()
        _, server, runner, port = await serving(bridge, kernels)
        try:
            socket = await page(port, str(uuid.uuid4()))
            await socket.send(json.dumps({"op": "execute", "id": "first", "cell": "a", "code": "print(1)"}))
            await answer(socket, "first")
            kernels.held.add("preload")
            await socket.send(json.dumps({"op": "execute", "id": "stopped", "cell": "b", "code": "print(2)",
                                          **({"kernel": "train"} if kind == "train" else {})}))
            await asyncio.wait_for(kernels.entered["preload"].wait(), 30)
            await socket.send(json.dumps({"op": "interrupt", "cell": "b"}))
            await asyncio.sleep(0.5)
            kernels.held.clear()
            kernels.release["preload"].set()
            assert (await answer(socket, "stopped"))[-1]["status"] == "aborted"
            assert "print(2)" not in kernels.ran
            await socket.send(json.dumps({"op": "execute", "id": "next", "cell": "b", "code": "print(3)"}))
            assert (await answer(socket, "next"))[-1]["status"] == "ok"
            await socket.close()
            await settled(kernels)
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
                released = session not in gateway.connected and session not in gateway.contexts
                if released and len(kernels.live) == 1:
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
    is closed then, not a full allowance later, and told why; one that connects after it is refused."""
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
            with pytest.raises(ConnectionClosed) as ended:
                while True:
                    await asyncio.wait_for(again.recv(), 60)
            # Closed at the session's end, not an allowance after the reconnect, and told why.
            assert 4.0 <= time.monotonic() - began < 30
            assert ended.value.rcvd.code == 1000 and ended.value.rcvd.reason
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
    with run_from("benchmark_gateway", "site/live/container") as smoke:
        assert {"loadavg", "stalls", "cpu_ticks"} <= set(smoke.load())


def test_a_training_context_caps_the_batch_of_every_way_a_cell_loads_data_and_its_steps():
    """In a training context (live_training.install), a cell's dataset batched itself, by any name and
    with its size passed either way, a preset called and a preset through `pipe` each read batches of
    BATCH rows, and its fit, called by another name with its steps unpacked, trains STEPS steps, each
    saying so first: IPython runs the cell with its calls capped, and Dew's own functions are untouched;
    a call that reads its caller's frame (`locals`, however named) still reads the cell's."""
    cell = """
import numpy as np
from dew import Trainer
from dew.config import OptimConfig
from dew.data import Memory, presets
from dew.data.compile import compile as compile_data
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
records = Memory([{"id": index} for index in range(320)])
tokens = Memory([{"text": np.arange(400, dtype=np.int32)}])
windows = dict(seq_len=4, pack=None, stride=1)
rows = records.batch
batches = [compile_data(records.batch(64)).batch, compile_data(rows(64)).batch,
           compile_data(rows(size=64)).batch,
           compile_data(presets.lm(tokens, None, batch=64, **windows)).batch,
           compile_data(tokens.pipe(presets.lm, None, batch=64, **windows)).batch]
model = CausalTransformer(vocab_size=400, emb_features=8, num_layers=1, num_heads=1, mlp_features=16,
                          max_seq_len=4)
run = Trainer(LMObjective(model, seq_len=4), OptimConfig(learning_rate=1e-3), key=0).fit
state = run(tokens.pipe(presets.lm, None, batch=64, **windows), **dict(steps=1000))
read, import_builtins = locals, __import__("builtins")
frames = ["records" in read(), "records" in import_builtins.locals(), "records" in locals()]
print("ran", json.dumps([batches, int(state.step), presets.lm is dew.data.presets.lm, frames]))
"""
    code = f"""
import json, sys
sys.path.insert(0, {str(CONTAINER)!r})
from IPython.core.interactiveshell import InteractiveShell
import live_training
shell = InteractiveShell.instance()
live_training.install()
shell.run_cell("import json, dew.data.presets")
shell.run_cell({cell!r}).raise_error()
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-4000:]
    lines = done.stdout.strip().splitlines()
    assert json.loads(lines[-1].removeprefix("ran ")) == [[8, 8, 8, 8, 8], 20, True, [True, True, True]]
    assert [line for line in lines if line.startswith("Live run:")] == [
        "Live run: batch 64 -> 8, so it fits a shared 4-vCPU host."] * 6 + [
        "Live run: steps 1000 -> 20, log_every 100 -> 5, so it finishes in about two minutes."]


def test_a_session_idle_past_its_allowance_ends_and_says_so(bridge, monkeypatch):
    """With no cell running and no request for the idle allowance, the bridge closes the page's socket
    with its reason, and the session's kernels."""
    from websockets.exceptions import ConnectionClosed

    monkeypatch.setattr(bridge, "available", lambda: 1 << 40)
    monkeypatch.setenv("DEW_LIVE_IDLE_SECONDS", "120")

    async def scenario():
        kernels = Kernels()
        gateway, server, runner, port = await serving(bridge, kernels)
        try:
            socket = await page(port, str(uuid.uuid4()))
            # Its contexts were last used three minutes ago.
            for cells in gateway.contexts.values():
                for context in cells.values():
                    context.last_used -= 180
            with pytest.raises(ConnectionClosed) as ended:
                while True:
                    await asyncio.wait_for(socket.recv(), 30)
            closed = ended.value.rcvd
            assert closed.code == 1000 and closed.reason
            await settled(kernels)
            assert kernels.live == set()
        finally:
            server.close()
            await runner.cleanup()

    asyncio.run(scenario())


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
            # The stopped cell ends in its error before its restart.
            assert (await answer(socket, "run-0"))[-1]["status"] == "error"
            # Each cell restarted; their fresh contexts wait on the gateway.
            kernels.held.add("create")
            for index in range(bridge.MAX_CELLS):
                await socket.send(json.dumps({"op": "restart", "id": f"restart-{index}",
                                              "cell": f"c{index}"}))
            await asyncio.wait_for(kernels.entered["create"].wait(), 30)
            # The running ones, the restart aborted, in whatever order they end.
            ended = {}
            while len(ended) < bridge.MAX_CELLS - 1:
                message = json.loads(await asyncio.wait_for(socket.recv(), 30))
                if message["type"] == "done" and message["id"].startswith("run-"):
                    ended[message["id"]] = message["status"]
            assert ended == {f"run-{index}": "aborted" for index in range(1, bridge.MAX_CELLS)}
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
