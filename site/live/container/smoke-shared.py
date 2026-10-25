"""Exercise the exact browser protocol through an offline shared container."""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from guests import GUEST_UIDS
from shared_bridge import DEADLINE_HEADER, OOM_PROBE, available, load, processes, resident
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

TEXT = ("from dew.interop import PretrainedDecoder\nfrom dew.sampling import Sampling\n"
        "model = PretrainedDecoder.load('HuggingFaceTB/SmolLM2-135M-Instruct', dtype='float32', "
        "max_seq_len=256)\n"
        "task = model.text_generation(sampling=Sampling(temperature=0))\n"
        "print(task('The capital of France is', 24, key=0).text[0])")
# Fill memory 128 MiB at a time until the context's own limit refuses more, having first
# tried to make itself the last choice of the OOM killer.
FILL = OOM_PROBE + """import numpy as np
blocks = []
try:
    while True:
        blocks.append(np.ones(16 << 20))
except MemoryError:
    print(f"held {len(blocks) * 128} MiB")
"""


def model_process():
    for pid, _, command in processes():
        if b"model_service.py" in command:
            return str(pid)
    raise AssertionError("the model process is not running")


def memory():
    """The host's memory and what each process group holds, in MiB, for a failure's message."""
    lines = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"total": int(lines["MemTotal"].split()[0]) >> 10, "available": available() >> 20,
            "by_uid": resident()}


def guests():
    """Each guest process's uid, state and command line."""
    return [(str(pid), int(fields["Uid"].split()[1]), fields["State"].strip(),
             command.replace(b"\0", b" ").decode(errors="replace")[-80:])
            for pid, fields, command in processes() if int(fields["Uid"].split()[1]) in GUEST_UIDS]


def gateway_log():
    return "\n".join(Path("/run/dew/gateway.log").read_text(errors="replace").splitlines()[-40:])


def report():
    return f"memory {memory()}; load {load()}; guests {guests()}; gateway log:\n{gateway_log()}"


def step(text):
    """Say what the smoke does next, so a smoke that fails or is cut off shows how far it got."""
    print(f"relay smoke: {text}", file=sys.stderr, flush=True)


async def page(headers):
    socket = await connect(f"ws://127.0.0.1:8888/contexts/{uuid.uuid4()}/ws", additional_headers=headers)
    try:
        assert json.loads(await socket.recv())["type"] == "ready"
    except ConnectionClosed as closed:
        raise AssertionError(f"the bridge refused a page: {closed.rcvd}; {report()}") from None
    return socket


async def outcomes(socket, count):
    """Each cell's final status and errors, as the page hears them, for the next `count` cells."""
    seen = {}
    while sum(1 for status, _ in seen.values() if status) < count:
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=600))
        status, errors = seen.get(message["id"], (None, []))
        if message["type"] == "error":
            errors.append(f"{message['ename']}: {message['evalue']}")
        seen[message["id"]] = (message["status"] if message["type"] == "done" else status, errors)
    return seen


async def closed():
    """Wait for the bridge to close every context, connection file included, after its page closed."""
    for _ in range(120):
        left = sorted(path.name for path in Path("/sessions/connections").glob("kernel-*.json"))
        if not left:
            return
        await asyncio.sleep(0.5)
    raise AssertionError(f"contexts left open after their pages closed: {left}; memory {memory()}")


async def pressure(headers):
    """More memory than the host holds, asked for at once: 24 page cells filling their address
    space, six a page as a page may run, and a training run on a fifth page filling its writable
    memory. The bridge refuses contexts or stops guests, never the model process, every cell that
    does not finish says why, and the model still answers afterwards."""
    model, before = model_process(), available() >> 20
    step("pressure: opening five pages")
    pages = [await page(headers) for _ in range(5)]
    step("pressure: filling memory from 24 page cells and a training cell")
    for number, socket in enumerate(pages[:4]):
        for cell in range(6):
            await socket.send(json.dumps({"op": "execute", "id": f"{number}.{cell}", "code": FILL,
                                          "cell": f"fill{cell}"}))
    await pages[4].send(json.dumps({"op": "execute", "id": "train", "code": FILL, "cell": "train",
                                    "kernel": "train"}))
    seen = {}
    for number, socket in enumerate(pages):
        seen.update(await outcomes(socket, 1 if number == 4 else 6))
    # A cell that did not finish was stopped or refused, in words; the pressure must have done one.
    reasons = ("Stopped: This cell's Python context stopped",
               "ValueError: the shared container has no free",
               "ValueError: the shared host is short of memory",
               "TimeoutError: this cell's Python context sent")
    unfinished = {name: errors for name, (status, errors) in seen.items() if status != "ok"}
    report = f"{seen}; MiB available before {before}, now {memory()}"
    assert unfinished, f"nothing was refused or stopped: {report}"
    assert all(errors and all(error.startswith(reasons) for error in errors)
               for errors in unfinished.values()), report
    assert model_process() == model, "the model process was restarted"
    for socket in pages:
        await socket.close()
    await closed()
    step("pressure: the model answers after it")
    socket = await page(headers)
    await socket.send(json.dumps({"op": "execute", "id": "after", "code": TEXT}))
    (status, errors), = (await outcomes(socket, 1)).values()
    assert status == "ok", errors
    await socket.close()
    await closed()
    print(f"Memory pressure: {len(seen) - len(unfinished)} of 25 cells ran, the others were told why; "
          f"the model process kept serving. Memory before: {before} MiB available; after: {memory()}",
          flush=True)


async def main():
    secret = sys.stdin.read().strip()
    if not secret:
        raise ValueError("the smoke check requires the private relay credential on stdin")
    session = str(uuid.uuid4())
    # As the Worker sends them: the relay's credential and the session's end, past this smoke's own.
    headers = {"Authorization": "Bearer " + secret, DEADLINE_HEADER: str(int((time.time() + 3600) * 1000))}
    async with connect(f"ws://127.0.0.1:8888/contexts/{session}/ws", additional_headers=headers) as socket:
        assert json.loads(await socket.recv())["type"] == "ready"
        # The page's cells, as Dew code the context's model_client stands in for.
        cells = [
            TEXT,
            "from dew.inference import TextToImage\n"
            "from dew.sampling import CFG, DPMSolverMultistep\n"
            "pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')\n"
            "pipe(['a lake beneath the northern lights'], key=3, steps=15, "
            "solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9))).pil()[0]",
        ]
        for index, code in enumerate(cells):
            step(f"page cell {index}")
            await socket.send(json.dumps({"op": "execute", "id": str(index), "code": code}))
            outputs = []
            while True:
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=100))
                if message.get("id") != str(index):
                    continue
                if message["type"] == "error":
                    raise RuntimeError(message["evalue"])
                if message["type"] == "done":
                    assert message["status"] == "ok", message
                    break
                outputs.append(message)
            if index == 0:
                assert any(output.get("text", "").strip() for output in outputs)
            else:
                assert any(output.get("png") for output in outputs)
        await socket.send(json.dumps({"op": "execute", "id": "stop", "code": "import time; time.sleep(60)"}))
        await asyncio.sleep(0.5)
        await socket.send(json.dumps({"op": "interrupt"}))
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
            if message.get("id") == "stop" and message["type"] == "done":
                assert message["status"] in ("error", "aborted"), message
                break
        print("Browser text/image and Stop protocol passed through the isolated shared context.", flush=True)
    # The bridge closes the first page's contexts as its socket closes.
    await asyncio.sleep(2)
    await pressure(headers)


async def bounded(seconds):
    """main(), failed in words past `seconds`: the preparation's smoke has one 15-minute alarm
    for this and the context smoke after it (site/live/src/preparer.ts)."""
    try:
        await asyncio.wait_for(main(), seconds)
    except TimeoutError:
        raise TimeoutError(f"the relay smoke did not end within {seconds} s; {report()}") from None


if __name__ == "__main__":
    asyncio.run(bounded(300))
