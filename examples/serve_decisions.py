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

The task is served by Dew's serving engine (`dew.inference.engine.Engine`)
as a scoring server (`dew.inference.scoring.ScoringServer`): concurrent
requests are answered together, one forward pass a turn over the rows waiting,
up to `--batch-rows` rows and `--batch-tokens` of their padded tokens a pass
(the task's `Budget`), each as it would be alone; a request Jev would refuse
gets its own 422 and the others their answers. `GET /health` answers once the
model is loaded.
"""

import asyncio
import contextlib
import hmac
import json
import os
from dataclasses import dataclass, replace

import tyro
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dew.decision import Decide
from dew.decision.task import Budget, WireScoring
from dew.inference import Engine, ScoringServer


def served(decide: Decide, *, details: bool = False, strict: bool = False, whole: bool = False,
           tokens: int = 8192, rows: int = 64) -> ScoringServer:
    """A scoring server over `decide`'s wire answers (`Decide.scoring`), concurrent requests' rows
    answered together, at most `rows` of them and `tokens` of their padded tokens a pass."""
    task = replace(decide, budget=Budget(tokens=tokens, rows=rows))
    return ScoringServer.from_task(*task.scoring(details=details, strict=strict, whole=whole))


def app(server: ScoringServer, *, api_key: str | None = None) -> Starlette:
    """A Starlette app answering `POST /v1/systemone` through `server` (`served`), which the serving
    engine runs within the app's lifespan."""
    engine = Engine(server)

    @contextlib.asynccontextmanager
    async def lifespan(_):
        engine.start()
        try:
            yield
        finally:
            engine.close()

    async def systemone(request: Request) -> JSONResponse:
        given = request.headers.get("authorization", "")
        if api_key is not None and not hmac.compare_digest(given, f"Bearer {api_key}"):
            return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
        try:
            body = json.loads(await request.body())
            if not isinstance(body, dict):
                raise ValueError("the request body is a JSON object with state and questions")
            answered = await run_in_threadpool(server.submit, body)
            return JSONResponse(await asyncio.wrap_future(answered))
        except ValueError as refused:
            return JSONResponse({"error": str(refused)}, status_code=422)

    async def health(request: Request) -> JSONResponse:
        # The task served now, whose name a reload's task replaces.
        served = server.task
        assert isinstance(served, WireScoring)
        return JSONResponse({"model": served.decide.name})

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
    param_dtype: str | None = None
    """The weights' storage dtype; None stores them in `dtype`, the dtype they serve in, so a run's
    weights take the memory they are served in, not the float32 a run trains them in."""
    max_len: int | None = None
    head_max_len: int | None = None
    """For a layout of one row per question: the tokens its instructions and options share."""
    option_tokens: int | None = None
    """For a layout of one row per question: the most tokens one option keeps."""
    batch_tokens: int = 8192
    """The most padded tokens one pass holds."""
    batch_rows: int = 64
    """The most rows one pass holds."""


def loaded(options: Options) -> Decide:
    """The task `options` name, a run's (`run`) or a release's (`model`), its weights restored in
    `param_dtype`, else in the dtype they serve in."""
    param_dtype = options.param_dtype or options.dtype
    if options.run is not None:
        return Decide.from_run(options.run, dtype=options.dtype, param_dtype=param_dtype)
    return Decide.from_pretrained(options.model or "convaiinnovations/laya", subfolder=options.subfolder,
                                  revision=options.revision, dtype=options.dtype, param_dtype=param_dtype)


def main(options: Options) -> None:
    import uvicorn

    decide = loaded(options)
    budgets = {"max_len": options.max_len, "head_max_len": options.head_max_len,
               "option_tokens": options.option_tokens}
    widened = {name: value for name, value in budgets.items() if value is not None}
    if widened:
        decide = replace(decide, layout=replace(decide.layout, **widened))
    server = served(decide, details=options.details, strict=options.strict, whole=options.whole,
                    tokens=options.batch_tokens, rows=options.batch_rows)
    # The server holds the weights resident, so the loaded task's own reference goes.
    del decide
    uvicorn.run(app(server, api_key=options.api_key), host=options.host, port=options.port)


if __name__ == "__main__":
    main(tyro.cli(Options))
