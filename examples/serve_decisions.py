"""Serve a decision model over HTTP, in TypeSafe Jev's wire format.

Dew has no HTTP server of its own. `Decide.systemone` takes a Jev request
body and returns a Jev response body, so serving it is a few lines of any
web framework; this uses Starlette, which the `serve` extra installs with
uvicorn:

    uv pip install "dewml[serve] @ git+https://github.com/AshishKumar4/dew"
    python examples/serve_decisions.py --model convaiinnovations/laya --port 8000
    python examples/serve_decisions.py --run runs/tickets

`--model` loads a released Laya checkpoint (with `--subfolder multilingual`
for the multilingual one) or a Clef release, and `--run` a
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
request fields Jev does not know, such as Clef's `images`. `--whole`
refuses, with a 422 naming the maximum context length, a request the
layout could answer only by cutting its state, instructions or options, as
Decision Index requires of an engine:

    python examples/serve_decisions.py --run runs/encoder --whole --dtype bfloat16 \\
        --max-len 8192 --head-max-len 6144 --option-tokens 512
    python -m decision_index pipeline --edition 0.2.1 --engine http \\
        --option base_url=http://127.0.0.1:8000 --out runs/di-encoder
 Requests are
answered one at a time, each in one forward pass per budget's worth of
question rows; `GET /health` answers once the model is loaded.
"""

import hmac
import json
import os
import threading
from dataclasses import dataclass, replace

import tyro
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dew.decision import Decide


def app(decide: Decide, *, api_key: str | None = None, details: bool = False,
        strict: bool = False, whole: bool = False) -> Starlette:
    """A Starlette app answering `POST /v1/systemone` with `decide`."""
    one_at_a_time = threading.Lock()

    def answer(body: dict) -> dict:
        with one_at_a_time:
            return decide.systemone(body, details=details, strict=strict, whole=whole)

    async def systemone(request: Request) -> JSONResponse:
        given = request.headers.get("authorization", "")
        if api_key is not None and not hmac.compare_digest(given, f"Bearer {api_key}"):
            return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
        try:
            body = json.loads(await request.body())
            if not isinstance(body, dict):
                raise ValueError("the request body is a JSON object with state and questions")
            return JSONResponse(await run_in_threadpool(answer, body))
        except ValueError as refused:
            return JSONResponse({"error": str(refused)}, status_code=422)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"model": decide.name})

    return Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]),
                             Route("/health", health, methods=["GET"])])


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
                 whole=options.whole)
    uvicorn.run(served, host=options.host, port=options.port)


if __name__ == "__main__":
    main(tyro.cli(Options))
