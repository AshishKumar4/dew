"""Prompt-only inference: resident models shared by visitors, never a code kernel.

One owner thread runs Dew's text Server, admitting queued prompts together.
Diffusion requests share a resident pipeline but run separately: batching its
per-request randomness and progress needs a separate numerical oracle first.
Only the declared JSON fields enter either model. There is no execute endpoint.
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
IMAGE_MODEL = "dewml/hybrid-dit-176m"
MAX_QUEUE = 32


@dataclass(frozen=True)
class Request:
    prompt: str
    key: int = 0
    tokens: int = 24
    steps: int = 15


def request_from(body: object, kind: str) -> Request:
    if kind not in ("text", "image"):
        raise ValueError("model kind must be text or image")
    fields = {"prompt", "key", "tokens"} if kind == "text" else {"prompt", "key", "steps"}
    if not isinstance(body, dict) or set(body) - fields:
        raise ValueError("request has unknown fields; only prompt and the declared settings are accepted")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise ValueError("prompt must contain 1 to 512 characters")
    key = body.get("key", 0)
    if type(key) is not int or not 0 <= key < 2**32:
        raise ValueError("key must be an integer from 0 to 4294967295")
    tokens, steps = body.get("tokens", 24), body.get("steps", 15)
    if kind == "text" and (type(tokens) is not int or not 1 <= tokens <= 64):
        raise ValueError("tokens must be an integer from 1 to 64")
    if kind == "image" and (type(steps) is not int or steps not in (15, 30)):
        raise ValueError("steps must be 15 or 30")
    return Request(prompt, key, tokens, steps)


class Inference:
    def __init__(self, kind: str):
        self.kind = kind
        self.jobs: queue.Queue[tuple[Request, Future, float]] = queue.Queue(MAX_QUEUE)
        self.ready = threading.Event()
        self.failed: str | None = None
        self.metrics: dict = {}
        threading.Thread(target=self.run, daemon=True).start()

    def submit(self, request: Request) -> Future:
        future = Future()
        self.jobs.put_nowait((request, future, time.perf_counter()))
        return future

    def run(self):
        try:
            started = time.perf_counter()
            if self.kind == "text":
                import jax.numpy as jnp
                from dew.interop import PretrainedDecoder
                from dew.inference import Server
                from dew.sampling import Sampling

                bundle = PretrainedDecoder.load(f"/opt/models/{TEXT_MODEL}", dtype=jnp.float32, max_seq_len=256)
                task = bundle.text_generation(sampling=Sampling(temperature=0))
                server = Server.from_task(task, slots=4, capacity=128)
                # Server accepts bare token rows. The source processor supplies ordinary
                # positions; whole, unpadded rows have exactly the same implicit positions.
                def tokens(prompt):
                    inputs = bundle.processor(prompt)
                    import numpy as np
                    ids = np.asarray(inputs.tokens)
                    positions = inputs.token_fields.get("positions")
                    if positions is not None and not np.array_equal(positions, np.arange(ids.shape[1])[None, :]):
                        raise ValueError("the demo only accepts ordinary text positions")
                    if set(inputs.token_fields) - {"positions"} or inputs.conditioning or ids.shape != (1, ids.shape[1]):
                        raise ValueError("the demo only accepts one unpadded text prompt")
                    if ids.shape[1] > 64:
                        raise ValueError("the prompt exceeds 64 model tokens")
                    return ids[0]

                warm = server.submit(tokens("The capital of France is"), 24, key=0)
                server.run()
                warm.result()
                self.metrics = {"prepare_seconds": time.perf_counter() - started}
                self.ready.set()
                self.text_loop(server, tokens)
            else:
                from dew.sampling import CFG, DPMSolverMultistep, TextToImage

                pipe = TextToImage.from_pretrained(IMAGE_MODEL)
                for steps in (15, 30):
                    pipe(["a turquoise alpine lake"], key=0, steps=steps,
                         sampler=DPMSolverMultistep(), guidance=CFG(5.0)).host()
                self.metrics = {"prepare_seconds": time.perf_counter() - started}
                self.ready.set()
                while True:
                    request, future, submitted = self.jobs.get()
                    if future.cancelled():
                        continue
                    try:
                        result = pipe([request.prompt], key=request.key, steps=request.steps,
                                      sampler=DPMSolverMultistep(), guidance=CFG(5.0))
                        buffer = io.BytesIO()
                        result.pil()[0].save(buffer, format="PNG")
                        future.set_result({"png": base64.b64encode(buffer.getvalue()).decode(),
                                           "seconds": time.perf_counter() - submitted})
                    except Exception as error:
                        future.set_exception(error)
        except Exception as error:
            self.failed = str(error)
            self.ready.set()
            while not self.jobs.empty():
                _, future, _ = self.jobs.get_nowait()
                future.set_exception(RuntimeError(self.failed))

    def text_loop(self, server, tokenize):
        active = []
        while True:
            if not active:
                job = self.jobs.get()
                waiting = [job]
                # A short admission window lets simultaneous HTTP arrivals share a step.
                time.sleep(0.01)
            else:
                waiting = []
            while len(waiting) + len(active) < 4:
                try:
                    waiting.append(self.jobs.get_nowait())
                except queue.Empty:
                    break
            for request, future, submitted in waiting:
                if future.cancelled():
                    continue
                try:
                    ticket = server.submit(tokenize(request.prompt), request.tokens, key=request.key)
                    active.append((ticket, future, submitted))
                except Exception as error:
                    future.set_exception(error)
            if active:
                server.step()
            remaining = []
            for ticket, future, submitted in active:
                if ticket.done():
                    try:
                        generation = ticket.result()
                        future.set_result({"text": generation.text[0], "seconds": time.perf_counter() - submitted,
                                           "first_token_seconds": ticket.first - ticket.submitted
                                           if ticket.first is not None else None})
                    except Exception as error:
                        future.set_exception(error)
                else:
                    remaining.append((ticket, future, submitted))
            active = remaining


def main():
    kind = os.environ["DEW_DEMO_KIND"]
    if kind not in ("text", "image"):
        raise ValueError("DEW_DEMO_KIND must be text or image")
    engine = Inference(kind)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "not-found"})
            elif engine.failed:
                self.reply(503, {"error": engine.failed})
            else:
                self.reply(200 if engine.ready.is_set() else 202,
                           {"ready": engine.ready.is_set(), "kind": kind, **engine.metrics,
                            "image": os.environ.get("DEW_LIVE_IMAGE"), "queued": engine.jobs.qsize()})

        def do_POST(self):
            if self.path != "/generate":
                self.reply(404, {"error": "not-found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("request body must contain 1 to 4096 bytes")
                request = request_from(json.loads(self.rfile.read(size)), kind)
                if not engine.ready.is_set() or engine.failed:
                    self.reply(503, {"error": engine.failed or "warming"})
                    return
                future = engine.submit(request)
                self.reply(200, future.result(timeout=120))
            except (ValueError, json.JSONDecodeError) as error:
                self.reply(400, {"error": str(error)})
            except queue.Full:
                self.reply(429, {"error": "queue full"})
            except Exception as error:
                self.reply(503, {"error": str(error)})

    ThreadingHTTPServer(("0.0.0.0", 8888), Handler).serve_forever()


if __name__ == "__main__":
    main()
