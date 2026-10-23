"""Relay authenticated sessions to private Kernel Gateway contexts.

A session is one page. Each of its cells runs in a context of its own, so the
cells of a page run at the same time; a message's "cell" names the context, and
a message without one uses the page's only cell.

A cell sent with "kernel": "train" runs Dew itself in a fresh training context
(gateway_manager.py, live_training.py), closed when the cell ends. A host runs
one at a time; the others wait in order and hear their place in line. A page
runs or queues at most MAX_CELLS cells of either kind at once.

A session's lifetime is the Worker's: it names the session's absolute end
(DEADLINE_HEADER, from the Coordinator's first start), and every connection of
the session closes then, however late it reconnected.
"""

import asyncio
import hmac
import json
import os
import re
import signal
import sys
import time
import urllib.request
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from guests import GUEST_UIDS, MAX_CONTEXTS
from kernel_outputs import outputs_of
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

MAX_CODE = 100_000
MAX_OUTPUT = 2_000_000
# Python contexts one page may hold; a host's are guests.MAX_CONTEXTS.
MAX_CELLS = 6
DEADLINE_HEADER = "X-Dew-Deadline"
"""The session's end, in milliseconds since the epoch, which the Worker sets on every connection. The
bridge reads it against its own wall clock: the host's and the Worker's clocks, both NTP-synced, may
differ by the end's accuracy, a few seconds at most, which moves the end by as much and no more."""
# Memory the model process and the bridge keep free. Below it a new context is refused and the
# largest guest is killed at once: waiting for the kernel's OOM killer lets the host thrash first.
RESERVE = 1536 << 20
# A guest cannot lower its OOM score below the 1000 it starts at; its /proc is read-only besides. The
# smoke checks (smoke-shared.py, benchmark_gateway.py) run this in a guest's context.
OOM_PROBE = """for score in ("-1000", "-999", "0"):
    try:
        with open("/proc/self/oom_score_adj", "w") as file:
            file.write(score)
    except OSError:
        pass
    else:
        raise AssertionError(f"a guest lowered its OOM score to {score}")
assert open("/proc/self/oom_score_adj").read().strip() == "1000"
"""
PRELOAD = {"python3": "import model_client; model_client.install()",
           "dew-train": "import live_training; live_training.install()"}


class Stopped(RuntimeError):
    """The context's kernel died while a cell ran: the kernel's OOM killer stops guests first."""

    def __init__(self):
        super().__init__("This cell's Python context stopped, most likely out of memory on the shared host. "
                         "Run it again to start a new one.")


class Context:
    def __init__(self, identifier, channels):
        self.identifier, self.channels = identifier, channels
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
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
            # What the page shows of the cell, as it shows it (site/src/scripts/live.ts): every output since
            # the last clear, a display under an id at its latest frame each place it shows (kernel_outputs).
            reply, idle, size, displays, clearing = None, False, 0, {}, False
            while reply is None or not idle:
                try:
                    message = json.loads(await asyncio.wait_for(self.channels.recv(), timeout=95))
                except TimeoutError:
                    raise TimeoutError("this cell's Python context sent nothing for 95 seconds, so it "
                                       "was stopped; the shared host may be busy, so try again in a "
                                       "minute") from None
                kind, content = message["msg_type"], message["content"]
                # The gateway announces a kernel that died, and restarts it, outside any request.
                if kind == "status" and content["execution_state"] in ("restarting", "dead"):
                    raise Stopped
                if message.get("parent_header", {}).get("msg_id") != identifier:
                    continue
                if kind == "execute_reply":
                    reply = {"status": content["status"], "count": content.get("execution_count")}
                elif kind == "status" and content["execution_state"] == "idle":
                    idle = True
                else:
                    for output in outputs_of(kind, content):
                        cost = len(json.dumps(output))
                        if output["type"] == "clear" and output.get("wait"):
                            clearing = True
                        elif output["type"] == "clear":
                            size, displays, clearing = 0, {}, False
                        elif output.get("update"):
                            displays[output["display"]] = [cost] * len(displays.get(output["display"], []))
                        else:
                            if clearing:
                                size, displays, clearing = 0, {}, False
                            if "display" in output:
                                displays.setdefault(output["display"], []).append(cost)
                            else:
                                size += cost
                        if size + sum(map(sum, displays.values())) > MAX_OUTPUT:
                            raise ValueError("this cell printed more than the live output allowance")
                        await send(output)
            self.last_used = time.monotonic()
            return reply


def available(meminfo=Path("/proc/meminfo")):
    """The host's available memory, in bytes."""
    line = next(line for line in meminfo.read_text().splitlines() if line.startswith("MemAvailable:"))
    return int(line.split()[1]) << 10


def processes(proc=Path("/proc")):
    """Each running process as /proc shows it: its pid, its status fields ("Uid", "VmRSS") and its
    command line. One that exits while it is read is left out, as /proc then has no files for it."""
    for process in proc.glob("[0-9]*"):
        try:
            status, command = (process / "status").read_text(), (process / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        yield (int(process.name), dict(line.split(":", 1) for line in status.splitlines() if ":" in line),
               command)


def rss_kib(fields):
    """A process's resident memory in KiB, from its status `fields`; a kernel thread holds none."""
    return int(fields.get("VmRSS", "0 kB").split()[0])


def resident(proc=Path("/proc")):
    """Resident memory by user, in MiB: the model process, the gateway and bridge, each guest."""
    held = {}
    for _, fields, _ in processes(proc):
        uid = fields["Uid"].split()[1]
        held[uid] = held.get(uid, 0) + (rss_kib(fields) >> 10)
    return held


def load(proc=Path("/proc")):
    """The host's load, its CPU and IO stalls, and the processes that used the most CPU, for a
    failure's message: a context that starts too slowly on an idle host was starved of one."""
    used = []
    for process in proc.glob("[0-9]*"):
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[-60:]
        except (OSError, IndexError):
            continue
        used.append((int(fields[11]) + int(fields[12]), process.name, command))
    stalls = {kind: (proc / "pressure" / kind).read_text().split("\n")[0]
              for kind in ("cpu", "io") if (proc / "pressure" / kind).exists()}
    return {"loadavg": (proc / "loadavg").read_text().strip(), "stalls": stalls,
            "cpu_ticks": sorted(used, reverse=True)[:5]}


def largest_guest(proc=Path("/proc")):
    """The pids of the guest context holding the most memory, summed over its processes: each
    context runs as a uid of its own (gateway_manager.py). Empty when no guest runs."""
    held, pids = {}, {}
    for pid, fields, _ in processes(proc):
        uid = int(fields["Uid"].split()[1])
        if uid in GUEST_UIDS:
            held[uid] = held.get(uid, 0) + rss_kib(fields)
            pids.setdefault(uid, []).append(pid)
    return pids[max(held, key=held.__getitem__)] if held else []


class Running(NamedTuple):
    """What runs in a cell: its task, and the request it answers."""
    task: asyncio.Task
    request: str


class Page:
    """One page's connection: its socket and contexts, what runs in each cell, every
    task it started and has not ended (a restart's predecessor included), the contexts of its
    running training cells, the cells its Stop ended and the cells restarting."""

    def __init__(self, socket, session, cells):
        self.socket, self.session, self.cells = socket, session, cells
        self.running: dict[str, Running] = {}
        self.training, self.stops = {}, set()
        self.tasks, self.restarting = set(), set()

    def admit(self, message, cell):
        """Why the bridge refuses `message` on `cell` now, or None: a page runs or queues at most
        MAX_CELLS cells, and a restart, which ends a running cell, at most as many beside them."""
        if message["op"] == "restart" and cell in self.restarting:
            return "this cell is restarting already"
        if len(self.tasks) >= (MAX_CELLS if message["op"] == "execute" else 2 * MAX_CELLS):
            return f"this page runs or queues {MAX_CELLS} cells already; stop one, or wait for one to finish"
        return None

    def busy(self):
        return any(not running.task.done() for running in self.running.values())

    async def fail(self, identifier, error):
        await self.socket.send(json.dumps({"id": identifier, "type": "error",
                                           "ename": type(error).__name__, "evalue": str(error),
                                           "traceback": []}))
        await self.socket.send(json.dumps({"id": identifier, "type": "done", "status": "error",
                                           "count": None}))


class Gateway:
    def __init__(self, token, secret, commit, kernels="127.0.0.1:8890"):
        # `kernels` is the Kernel Gateway's address (start-gateway.sh).
        self.token, self.secret, self.commit, self.kernels = token, secret, commit, kernels
        # session -> cell -> context; None holds the context made ready before the page names a cell.
        self.contexts: dict[str, dict[str | None, Context]] = {}
        # session -> the one task creating its first context, which create and close await or cancel.
        self.creating: dict[str, asyncio.Task] = {}
        self.connected = set()
        self.allocating = asyncio.Lock()
        # The training slot: whether a training cell runs, and the tickets of those waiting, in order.
        self.slot = asyncio.Condition()
        self.training = False
        self.waiting = []
        self.started = time.monotonic()
        self.idle = int(os.environ.get("DEW_LIVE_IDLE_SECONDS", "300"))
        # Cleanups running to their end after what they clean up for was cancelled (`settle`).
        self.cleanups = set()

    async def api(self, path, data=None, method=None):
        def call():
            body = None if data is None else json.dumps(data).encode()
            request = urllib.request.Request(f"http://{self.kernels}{path}", data=body, method=method,
                                             headers={"Authorization": "token " + self.token,
                                                      "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        return await asyncio.to_thread(call)

    async def start(self, session, kernel="python3"):
        """A new context for `session`: a page cell's, its inference client loaded, or a training one."""
        async with self.allocating:
            if kernel == "python3" and len(self.contexts.get(session, {})) >= MAX_CELLS:
                raise ValueError("this page has more live cells than one session runs")
            if kernel == "python3" and sum(map(len, self.contexts.values())) >= MAX_CONTEXTS:
                raise ValueError("the shared container has no free Python contexts")
            if available() < RESERVE + (512 << 20):
                raise ValueError("the shared host is short of memory right now; try again in a minute")
            started = time.monotonic()
            # The create runs to its end whatever cancels this start, so its kernel is deleted below.
            creating = asyncio.ensure_future(self.api("/api/kernels", {"name": kernel}))
            try:
                created = await asyncio.shield(creating)
            except asyncio.CancelledError:
                async def unmade():
                    await self.discard(None, (await creating)["id"])
                await self.settle(unmade())
                raise
            kernel = {**created, "preload": PRELOAD[kernel]}
            channels = None
            try:
                url = f"ws://{self.kernels}/api/kernels/{kernel['id']}/channels"
                try:
                    channels = await connect(url, max_size=MAX_OUTPUT, open_timeout=60,
                                             additional_headers={"Authorization": "token " + self.token})
                except TimeoutError:
                    raise TimeoutError("this cell's Python context did not start within a minute; the "
                                       "shared host may be busy, so try again in a minute") from None
                context = Context(kernel["id"], channels)
                result = await context.execute(kernel["preload"], lambda _: asyncio.sleep(0))
                if result["status"] != "ok":
                    raise RuntimeError("the isolated Python context could not load its client")
                context.setup = time.monotonic() - started
                return context
            except BaseException:
                # Cancellation too (a disconnect, a Restart): nothing else knows this kernel yet.
                await self.settle(self.discard(channels, kernel["id"]))
                raise

    async def discard(self, channels, identifier):
        """Close a kernel: its channels, if open, and the kernel, deleted though they fail to close."""
        try:
            if channels:
                await channels.close()
        finally:
            await self.api(f"/api/kernels/{identifier}", method="DELETE")

    async def settle(self, cleanup):
        """Run `cleanup` to its end, though the caller is cancelled meanwhile, which then goes on to
        cancel the caller; its failure is logged, never raised: why the caller ends is its own."""
        task = asyncio.ensure_future(cleanup)
        self.cleanups.add(task)

        def ended(task):
            self.cleanups.discard(task)
            if not task.cancelled() and task.exception() is not None:
                print(f"a kernel's cleanup failed: {task.exception()!r}", file=sys.stderr, flush=True)
        task.add_done_callback(ended)
        # Waits for its end without raising its failure, and leaves it running if the caller is cancelled.
        await asyncio.wait({task})

    async def first(self, session):
        """`session`'s first context, made ready before the page names a cell: the one creation the gateway
        owns for the session (`creating`), which a close cancels, `start` then deleting the kernel it made."""
        mine = asyncio.current_task()
        try:
            self.contexts[session][None] = await self.start(session)
        except BaseException:
            # A close took the session already, and a later create may have begun it again, which is
            # not this one's to end.
            if self.creating.get(session) is mine:
                self.contexts.pop(session, None)
            raise
        finally:
            if self.creating.get(session) is mine:
                del self.creating[session]

    async def create(self, session):
        """`session`'s contexts, its first made ready if it has none (`first`), which every create of the
        session waits for, and a close of it ends."""
        if session not in self.contexts:
            self.contexts[session] = {}
            self.creating[session] = asyncio.ensure_future(self.first(session))
        if task := self.creating.get(session):
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise RuntimeError("this session was closed while its context started") from None
                raise
        return self.contexts[session]

    async def context(self, session, cell):
        """The context of `cell`: the ready one when the page has not used it, else a new one."""
        cells = self.contexts[session]
        if cell not in cells:
            cells[cell] = cells.pop(None) if None in cells else await self.start(session)
        return cells[cell]

    async def stop(self, context):
        """Close `context`'s kernel, to its end though what stops it is cancelled (`settle`)."""
        await self.settle(self.discard(context.channels, context.identifier))

    async def close(self, session):
        """End `session`: its first context's creation, if one runs, cancelled, and every context it holds
        closed. The creation is not awaited: `start` deletes the kernel it made once the gateway has made it
        (`settle`), which this close need not wait for."""
        if task := self.creating.pop(session, None):
            task.cancel()
        for context in self.contexts.pop(session, {}).values():
            await self.stop(context)

    async def train(self, session, code, send, stopped, started):
        """Run a training cell in a fresh context once the host's training slot is free.

        While it waits, `send` hears how many runs are ahead of it, and `stopped()`,
        which the page's Stop sets, ends the wait. `started` hears the context once
        it exists, so Stop can interrupt it. Returns the cell's reply.
        """
        ticket, told = object(), None
        async with self.slot:
            self.waiting.append(ticket)
        try:
            while True:
                async with self.slot:
                    while True:
                        if stopped():
                            return {"status": "aborted", "count": None}
                        ahead = self.waiting.index(ticket) + self.training
                        if not ahead:
                            self.waiting.remove(ticket)
                            self.training = True
                            break
                        if ahead != told:
                            break
                        await self.slot.wait()
                if not ahead:
                    break
                await send({"type": "display", "text": json.dumps({"dew-wait": {"ahead": ahead}})})
                told = ahead
        finally:
            async with self.slot:
                if ticket in self.waiting:
                    self.waiting.remove(ticket)
                    self.slot.notify_all()
        try:
            context = await self.start(session, "dew-train")
            started(context)
            try:
                # A Stop while the context started ends the run before its code reaches the kernel.
                if stopped():
                    return {"status": "aborted", "count": None}
                return await context.execute(code, send)
            finally:
                await self.stop(context)
        finally:
            async with self.slot:
                self.training = False
                self.slot.notify_all()

    async def http(self, connection, request):
        if request.path == "/health":
            ready = Path("/run/dew/model/ready").exists()
            # Reachable only through the host's Durable Object (shared-host.ts), for the operator.
            report = {"available_mib": available() >> 20, "contexts": sum(map(len, self.contexts.values())),
                      "training": self.training, "rss_mib": resident()}
            return Response(200 if ready else 503, "OK" if ready else "Warming up", Headers(),
                            json.dumps(report).encode())
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
        # The session's end (DEADLINE_HEADER), on this host's monotonic clock.
        deadline = time.monotonic() + int(socket.request.headers[DEADLINE_HEADER]) / 1000 - time.time()
        if deadline <= time.monotonic():
            await socket.close(1008, "this session is over")
            return
        try:
            cells = await self.create(session)
        except Exception as error:
            # 1013 is "try again later"; the reason says why, within the frame's 123 bytes.
            await socket.close(1013, str(error).encode()[:123].decode(errors="ignore"))
            return
        if session in self.connected:
            await socket.close(1008, "this context is already connected")
            return
        self.connected.add(session)
        await socket.send(json.dumps({"type": "ready", "uptime": time.monotonic() - self.started,
                                      "setup": next(iter(cells.values())).setup if cells else 0,
                                      "dew": self.commit}))
        page = Page(socket, session, cells)
        # Why the session ends, which the page shows: the socket's close reason.
        ending = "it reached its time limit"
        try:
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=min(15, deadline - time.monotonic()))
                except TimeoutError:
                    used = max((context.last_used for context in cells.values()), default=0)
                    if not page.busy() and time.monotonic() - used > self.idle:
                        ending = f"it had no request for {self.idle // 60} minutes"
                        break
                    continue
                message = json.loads(raw)
                cell = message.get("cell", "")
                if not isinstance(cell, str) or len(cell) > 64:
                    raise ValueError("invalid cell name")
                for context in cells.values():
                    context.last_used = time.monotonic()
                if message.get("op") == "interrupt":
                    await self.interrupt(page, cell)
                elif message.get("op") in ("execute", "restart"):
                    running = page.running.get(cell)
                    if message["op"] == "execute" and running is not None and not running.task.done():
                        raise ValueError("a cell is already running in this context")
                    if (not isinstance(message.get("id"), str) or len(message["id"]) > 128 or
                            (message["op"] == "execute" and
                             (not isinstance(message.get("code"), str) or len(message["code"]) > MAX_CODE))):
                        raise ValueError("invalid or oversized cell")
                    for ended in [name for name, running in page.running.items() if running.task.done()]:
                        del page.running[ended]
                    if refusal := page.admit(message, cell):
                        await page.fail(message["id"], ValueError(refusal))
                        continue
                    step = self.run if message["op"] == "execute" else self.restart
                    if step == self.restart:
                        # Before the next message is read: a restart of this cell admitted now is refused.
                        page.restarting.add(cell)
                    task = asyncio.create_task(step(page, message, cell, page.running.get(cell)))
                    page.tasks.add(task)
                    task.add_done_callback(page.tasks.discard)
                    page.running[cell] = Running(task, message["id"])
                else:
                    raise ValueError("unsupported kernel operation")
        finally:
            tasks = list(page.tasks)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            self.connected.discard(session)
            await self.close(session)
        await socket.close(1000, ending)

    async def interrupt(self, page, cell):
        """Interrupt what runs in `cell`, and end a training cell's wait for its turn."""
        if cell in page.training:
            await self.api(f"/api/kernels/{page.training[cell].identifier}/interrupt", {}, "POST")
        elif cell in page.cells:
            await self.api(f"/api/kernels/{page.cells[cell].identifier}/interrupt", {}, "POST")
        if cell in page.running and not page.running[cell].task.done():
            page.stops.add(cell)
            async with self.slot:
                self.slot.notify_all()

    async def run(self, page, message, cell, _before):
        """Run one cell's code in its context, or a training cell in a fresh one, sending its
        outputs as they come and then done."""
        context, socket = None, page.socket

        def send(output):
            return socket.send(json.dumps({"id": message["id"], **output}))
        try:
            if message.get("kernel") == "train":
                page.stops.discard(cell)
                result = await self.train(page.session, message["code"], send, lambda: cell in page.stops,
                                          lambda started: page.training.__setitem__(cell, started))
            else:
                context = await self.context(page.session, cell)
                # A Stop while the context started ends the run before its code reaches the kernel.
                result = ({"status": "aborted", "count": None} if cell in page.stops
                          else await context.execute(message["code"], send))
            await socket.send(json.dumps({"id": message["id"], "type": "done", **result}))
        except Exception as error:
            if isinstance(error, Stopped) and context is not None and page.cells.get(cell) is context:
                # The next run of this cell starts a new context, with its client loaded again.
                del page.cells[cell]
                await self.stop(context)
            elif context:
                with suppress(OSError):
                    await self.api(f"/api/kernels/{context.identifier}/interrupt", {}, "POST")
            await page.fail(message["id"], error)
        finally:
            page.training.pop(cell, None)
            page.stops.discard(cell)

    async def restart(self, page, message, cell, before):
        """End what runs in `cell`, give it a fresh context, and answer done: the notebook's Restart."""
        try:
            if before is not None and not before.task.done():
                page.stops.add(cell)
                before.task.cancel()
                with suppress(asyncio.CancelledError):
                    await before.task
                await page.socket.send(json.dumps({"id": before.request, "type": "done", "status": "aborted",
                                                   "count": None}))
            # A cell that never ran has no context of its own yet; its first run starts a fresh one.
            if cell in page.cells:
                await self.stop(page.cells.pop(cell))
                page.cells[cell] = await self.start(page.session)
            await page.socket.send(json.dumps({"id": message["id"], "type": "done", "status": "ok",
                                               "count": None}))
        except Exception as error:
            await page.fail(message["id"], error)
        finally:
            page.restarting.discard(cell)

    async def guard(self):
        """Kill the largest guest while the host's available memory is under RESERVE; the bridge
        then tells its cell the context stopped (`Stopped`)."""
        while True:
            await asyncio.sleep(0.5)
            if available() < RESERVE:
                for pid in largest_guest():
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)

    async def reap(self):
        """Close each session no page holds whose contexts have been idle too long. A session
        with no context yet is starting its first (`create`), so is not idle."""
        for session, cells in list(self.contexts.items()):
            if (cells and session not in self.connected
                    and time.monotonic() - max(context.last_used for context in cells.values()) > self.idle):
                await self.close(session)

    async def sweep(self):
        while True:
            await asyncio.sleep(15)
            await self.reap()


async def main():
    gateway = Gateway(Path("/run/dew/gateway-token").read_text(), os.environ["DEW_SHARED_SECRET"],
                      Path("/opt/live/dew-commit").read_text().strip())
    async with serve(gateway.websocket, "0.0.0.0", 8888, process_request=gateway.http,
                     max_size=MAX_CODE * 4, ping_interval=20, ping_timeout=20):
        await asyncio.gather(gateway.sweep(), gateway.guard())


if __name__ == "__main__":
    asyncio.run(main())
