"""Serve a decision model over HTTP, in TypeSafe Jev's wire format.

Dew has no HTTP server of its own. `Decide.systemone` takes a Jev request
body and returns a Jev response body, so serving it is a few lines of any
web framework; this uses Starlette, which the `serve` extra installs with
uvicorn:

    uv pip install "dewml[serve] @ git+https://github.com/AshishKumar4/dew"
    python examples/serve_decisions.py --model convaiinnovations/laya --port 8000
    python examples/serve_decisions.py --run runs/tickets

`--model` loads a released Laya checkpoint (with `--subfolder multilingual`
for the multilingual one), and `--run` a `DecisionObjective` run with the
calibration saved into it. `POST /v1/systemone` takes what Jev's endpoint
takes and answers as it answers, so a client written for Jev only changes
its base URL:

    curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
      "state": "I was charged twice for March, fix it today.",
      "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'

A request Jev would refuse gets a 422 with the reason, as Jev's does. With
`--api-key` (or `DEW_API_KEY`) set, a request must carry `Authorization:
Bearer <key>`, and gets a 401 without it. `--details` adds Laya's extra
answer fields (a noul's confidence, abstention, how much of the state was
read); `--strict` answers exactly as Jev's endpoint does, dropping the
request fields Jev does not know, such as Clef's `images`. Requests are
answered one at a time, each in one forward pass per budget's worth of
question rows; `GET /health` answers once the model is loaded.
"""

import hmac
import json
import os
import threading
from dataclasses import dataclass

import tyro
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dew.decision import Decide


def app(decide: Decide, *, api_key: str | None = None, details: bool = False,
        strict: bool = False) -> Starlette:
    """A Starlette app answering `POST /v1/systemone` with `decide`."""
    one_at_a_time = threading.Lock()

    def answer(body: dict) -> dict:
        with one_at_a_time:
            return decide.systemone(body, details=details, strict=strict)

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


def main(options: Options) -> None:
    import uvicorn

    decide = (Decide.from_run(options.run) if options.run is not None
              else Decide.from_pretrained(options.model or "convaiinnovations/laya",
                                          subfolder=options.subfolder, revision=options.revision))
    served = app(decide, api_key=options.api_key, details=options.details, strict=options.strict)
    uvicorn.run(served, host=options.host, port=options.port)


if __name__ == "__main__":
    main(tyro.cli(Options))
