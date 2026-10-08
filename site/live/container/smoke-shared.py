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
        # The page's cells, as Dew code the context's model_client stands in for.
        cells = [
            "from dew.interop import PretrainedDecoder\nfrom dew.sampling import Sampling\n"
            "model = PretrainedDecoder.load('HuggingFaceTB/SmolLM2-135M-Instruct', dtype='float32', "
            "max_seq_len=256)\n"
            "task = model.text_generation(sampling=Sampling(temperature=0))\n"
            "print(task('The capital of France is', 24, key=0).text[0])",
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


if __name__ == "__main__":
    asyncio.run(main())
