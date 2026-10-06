"""Exercise the exact browser protocol through an offline shared container."""

import asyncio
import json
import sys
import uuid

from websockets.asyncio.client import connect


async def main():
    secret = sys.stdin.read().strip()
    if not secret:
        raise ValueError("the smoke check requires the private relay credential on stdin")
    session = str(uuid.uuid4())
    headers = {"Authorization": "Bearer " + secret}
    async with connect(f"ws://127.0.0.1:8888/contexts/{session}/ws", additional_headers=headers) as socket:
        assert json.loads(await socket.recv())["type"] == "ready"
        cells = [
            "task = text_model('HuggingFaceTB/SmolLM2-135M-Instruct')\n"
            "print(task('The capital of France is', 24, key=0).text[0])",
            "pipe = from_pretrained('dewml/hybrid-dit-176m')\n"
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
        print("Browser text/image protocol passed through the isolated shared context.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
