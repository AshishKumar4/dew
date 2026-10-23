"""Serve a decision model over HTTP, in TypeSafe Jev's wire format.

Dew has no HTTP server of its own. `Decide.systemone` takes a Jev request
body and returns a Jev response body, so serving it is a few lines of any
web framework; this uses Starlette, which the `serve` extra installs with
uvicorn:

    uv pip install "dewml[serve] @ git+https://github.com/AshishKumar4/dew" \
        --extra-index-url https://pypi.dewml.dev/simple
    python examples/serve_decisions.py --model convaiinnovations/laya --port 8000
    python examples/serve_decisions.py --run runs/tickets

`--model` loads a released Laya checkpoint (with `--subfolder multilingual`
for the multilingual one) or a joint-head release, and `--run` a
`DecisionObjective` run with the calibration saved into it. `--dtype`
chooses the compute dtype (bfloat16 on a GPU), and `--max-len`,
`--head-max-len` and `--option-tokens` widen the layout's budgets.
`POST /v1/systemone` takes what Jev's endpoint takes and answers as it
answers, so a client written for Jev only changes its base URL:

    curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
      "state": "I was charged twice for March, fix it today.",
      "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'

A request Jev would refuse gets a 422 with the reason, as Jev's does. With
`--api-key` (or `DEW_API_KEY`) set, a request must carry `Authorization:
Bearer <key>`, and gets a 401 without it. `--details` adds Laya's extra
answer fields (a noul's confidence, abstention, how much of the state was
read); `--strict` answers exactly as Jev's endpoint does, dropping the
request fields Jev does not know, such as `images`. `--whole`
refuses, with a 422 naming the maximum context length, a request the
layout could answer only by cutting its state, instructions or options, as
Decision Index requires of an engine:

    python examples/serve_decisions.py --run runs/encoder --whole --dtype bfloat16 \\
        --max-len 8192 --head-max-len 6144 --option-tokens 512
    python -m decision_index pipeline --edition 0.2.1 --engine http \\
        --option base_url=http://127.0.0.1:8000 --out runs/di-encoder

Concurrent requests are answered together: those that arrive within
`--batch-window` seconds of the first, up to `--max-batch`, go through one
`systemone` call, which scores their rows in one forward pass per budget's
worth of them and answers each as it would alone; a request Jev would refuse
gets its own 422 and the others their answers. `GET /health` answers once the
model is loaded.
"""

import contextlib
import hmac
import json
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace

import tyro
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dew.decision import Decide


class Batches:
    """Requests answered together: a worker thread takes the first request waiting, those that
    arrive within `window` seconds of it, up to `most`, and answers them through one `systemone`
    call, each request's own response or refusal through its own future. The worker runs between
    `start` and `stop`, the app's lifespan."""

    def __init__(self, decide: Decide, *, window: float, most: int, **answering: bool):
        if window < 0 or most < 1:
            raise ValueError(f"a batch waits a window of 0 s or more for 1 request or more, not "
                             f"{window} s and {most}")
        self.decide, self.window, self.most, self.answering = decide, window, most, answering
        self.waiting: queue.Queue[tuple[dict, Future] | None] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.runner: ThreadPoolExecutor | None = None

    def start(self) -> None:
        self.runner = ThreadPoolExecutor(1, thread_name_prefix="decision-batch")
        self.worker = threading.Thread(target=self._serve, name="decision-batches")
        self.worker.start()

    def stop(self) -> None:
        assert self.worker is not None and self.runner is not None
        self.waiting.put(None)
        self.worker.join()
        self.runner.shutdown()
        self.worker = self.runner = None

    def answer(self, body: dict) -> dict:
        """`body`'s response, from the batch it joins; its refusal raised as a ValueError."""
        if self.worker is None:
            raise RuntimeError("the app answers within its lifespan, which starts its batches")
        answered: Future = Future()
        self.waiting.put((body, answered))
        return answered.result()

    def _serve(self) -> None:
        while (first := self.waiting.get()) is not None:
            batch, closing = [first], False
            deadline = time.monotonic() + self.window
            while len(batch) < self.most and (left := deadline - time.monotonic()) > 0:
                try:
                    more = self.waiting.get(timeout=left)
                except queue.Empty:
                    break
                if more is None:
                    closing = True
                    break
                batch.append(more)
            self._answer(batch)
            if closing:
                return

    def _answer(self, batch: list[tuple[dict, Future]]) -> None:
        # The call runs on the runner, whose future carries what it raised: a batch that fails whole
        # tells every request of it why.
        assert self.runner is not None
        called = self.runner.submit(self.decide.systemone, [body for body, _ in batch], **self.answering)
        wait([called])
        failed = called.exception()
        if failed is not None:
            for _, answered in batch:
                answered.set_exception(failed)
            return
        for (_, answered), response in zip(batch, called.result(), strict=True):
            if isinstance(response, ValueError):
                answered.set_exception(response)
            else:
                answered.set_result(response)


def app(decide: Decide, *, api_key: str | None = None, details: bool = False,
        strict: bool = False, whole: bool = False, batch_window: float = 0.005,
        max_batch: int = 32) -> Starlette:
    """A Starlette app answering `POST /v1/systemone` with `decide`, concurrent requests together
    (`Batches`)."""
    batches = Batches(decide, window=batch_window, most=max_batch, details=details, strict=strict,
                      whole=whole)

    @contextlib.asynccontextmanager
    async def lifespan(_):
        batches.start()
        try:
            yield
        finally:
            batches.stop()

    async def systemone(request: Request) -> JSONResponse:
        given = request.headers.get("authorization", "")
        if api_key is not None and not hmac.compare_digest(given, f"Bearer {api_key}"):
            return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
        try:
            body = json.loads(await request.body())
            if not isinstance(body, dict):
                raise ValueError("the request body is a JSON object with state and questions")
            return JSONResponse(await run_in_threadpool(batches.answer, body))
        except ValueError as refused:
            return JSONResponse({"error": str(refused)}, status_code=422)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"model": decide.name})

    return Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]),
                             Route("/health", health, methods=["GET"])], lifespan=lifespan)


@dataclass(frozen=True)
class Options:
    model: str | None = "convaiinnovations/laya"
    """A released Laya checkpoint, a Hub repo or a directory."""
    subfolder: str | None = None
    revision: str | None = None
    run: str | None = None
    """A DecisionObjective run directory, in place of `model`."""
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = os.environ.get("DEW_API_KEY")
    details: bool = False
    strict: bool = False
    whole: bool = False
    """Refuse a request rather than cut it to fit."""
    dtype: str = "float32"
    max_len: int | None = None
    head_max_len: int | None = None
    """For a layout of one row per question: the tokens its instructions and options share."""
    option_tokens: int | None = None
    """For a layout of one row per question: the most tokens one option keeps."""
    batch_window: float = 0.005
    """Seconds a batch waits, after its first request, for more to answer with it."""
    max_batch: int = 32
    """The most requests one batch answers."""


def main(options: Options) -> None:
    import uvicorn

    decide = (Decide.from_run(options.run, dtype=options.dtype) if options.run is not None
              else Decide.from_pretrained(options.model or "convaiinnovations/laya",
                                          subfolder=options.subfolder, revision=options.revision,
                                          dtype=options.dtype))
    budgets = {"max_len": options.max_len, "head_max_len": options.head_max_len,
               "option_tokens": options.option_tokens}
    widened = {name: value for name, value in budgets.items() if value is not None}
    if widened:
        decide = replace(decide, layout=replace(decide.layout, **widened))
    served = app(decide, api_key=options.api_key, details=options.details, strict=options.strict,
                 whole=options.whole, batch_window=options.batch_window, max_batch=options.max_batch)
    uvicorn.run(served, host=options.host, port=options.port)


if __name__ == "__main__":
    main(tyro.cli(Options))
