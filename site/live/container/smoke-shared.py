"""Exercise the exact browser protocol through an offline shared container."""

import asyncio
import json
import sys
import uuid
from pathlib import Path

from websockets.asyncio.client import connect

TEXT = ("from dew.interop import PretrainedDecoder\nfrom dew.sampling import Sampling\n"
        "model = PretrainedDecoder.load('HuggingFaceTB/SmolLM2-135M-Instruct', dtype='float32', "
        "max_seq_len=256)\n"
        "task = model.text_generation(sampling=Sampling(temperature=0))\n"
        "print(task('The capital of France is', 24, key=0).text[0])")
# A guest cannot lower its OOM score below the 1000 it starts at.
LOWER = """for score in ("-1000", "-999", "0"):
    try:
        with open("/proc/self/oom_score_adj", "w") as file:
            file.write(score)
    except PermissionError:
        pass
    else:
        raise AssertionError(f"a guest lowered its OOM score to {score}")
assert open("/proc/self/oom_score_adj").read().strip() == "1000"
"""
# Fill memory 128 MiB at a time until the context's own limit refuses more, having first
# tried to make itself the last choice of the OOM killer.
FILL = LOWER + """import numpy as np
blocks = []
try:
    while True:
        blocks.append(np.ones(16 << 20))
except MemoryError:
    print(f"held {len(blocks) * 128} MiB")
"""


def model_process():
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if b"model_service.py" in (proc / "cmdline").read_bytes():
                return proc.name
        except OSError:
            pass
    raise AssertionError("the model process is not running")


async def page(headers):
    socket = await connect(f"ws://127.0.0.1:8888/contexts/{uuid.uuid4()}/ws", additional_headers=headers)
    assert json.loads(await socket.recv())["type"] == "ready"
    return socket


async def outcomes(socket, count):
    """Each cell's final status and errors, as the page hears them, for the next `count` cells."""
    seen = {}
    while sum(1 for status, _ in seen.values() if status) < count:
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=600))
        status, errors = seen.get(message["id"], (None, []))
        if message["type"] == "error":
            errors.append(message["evalue"])
        seen[message["id"]] = (message["status"] if message["type"] == "done" else status, errors)
    return seen


async def pressure(headers):
    """More memory than the host holds, asked for at once: 24 page cells filling their address
    space and a training run filling its writable memory. The OOM killer stops guests and never
    the model process, a stopped cell is told so, and the model still answers afterwards."""
    model = model_process()
    pages = [await page(headers) for _ in range(4)]
    for number, socket in enumerate(pages):
        for cell in range(6):
            await socket.send(json.dumps({"op": "execute", "id": f"{number}.{cell}", "code": FILL,
                                          "cell": f"fill{cell}"}))
    await pages[0].send(json.dumps({"op": "execute", "id": "train", "code": FILL, "cell": "train",
                                    "kernel": "train"}))
    seen = {}
    for number, socket in enumerate(pages):
        seen.update(await outcomes(socket, 7 if number == 0 else 6))
    stopped = [name for name, (_, errors) in seen.items()
               if any(error.startswith("This cell's Python context stopped") for error in errors)]
    assert stopped, f"nothing ran out of memory: {seen}"
    # Every other cell either ran to its own limit or was refused a context in words.
    refusals = ("the shared container has no free Python contexts",)
    assert all(status == "ok" or all(error in refusals for error in errors)
               for name, (status, errors) in seen.items() if name not in stopped), seen
    assert model_process() == model, "the model process was restarted"
    for socket in pages:
        await socket.close()
    socket = await page(headers)
    await socket.send(json.dumps({"op": "execute", "id": "after", "code": TEXT}))
    (status, errors), = (await outcomes(socket, 1)).values()
    assert status == "ok", errors
    await socket.close()
    print(f"Memory pressure stopped {len(stopped)} of 25 cells; the model process kept serving.", flush=True)


async def main():
    secret = sys.stdin.read().strip()
    if not secret:
        raise ValueError("the smoke check requires the private relay credential on stdin")
    session = str(uuid.uuid4())
    headers = {"Authorization": "Bearer " + secret}
    async with connect(f"ws://127.0.0.1:8888/contexts/{session}/ws", additional_headers=headers) as socket:
        assert json.loads(await socket.recv())["type"] == "ready"
        # The page's cells, as Dew code the context's model_client stands in for.
        cells = [
            TEXT,
            "from dew.sampling import CFG, DPMSolverMultistep, TextToImage\n"
            "pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')\n"
            "pipe(['a lake beneath the northern lights'], key=3, steps=15, "
            "solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9))).pil()[0]",
        ]
        for index, code in enumerate(cells):
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


if __name__ == "__main__":
    asyncio.run(main())
