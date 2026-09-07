"""HTTP serving over an Engine: JSON requests, JSON results, NDJSON streams.

The adapter maps protocol concerns onto the engine and nothing else. It does
not own weights, sampling science or scheduling; a client disconnect cancels
its request, and admission errors become HTTP status codes.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
import threading
from concurrent.futures import CancelledError
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import jax
import numpy as np

from dew.sampling.engine import CapacityError, Engine, GenerationJob
from dew.sampling.text import Sampling

MAX_BODY_BYTES = 64 * 1024 ** 2


class RequestError(ValueError):
    """A malformed request; reported as 400."""


def _generation(engine: Engine[Any, Any], body: dict[str, Any]) -> object | None:
    if "sampling" in body and "process" in body:
        raise RequestError("a request carries either sampling or process")
    if "sampling" in body:
        fields = body["sampling"]
        if not isinstance(fields, dict):
            raise RequestError("sampling must be an object")
        if isinstance(fields.get("eos_id"), list):
            fields = {**fields, "eos_id": tuple(fields["eos_id"])}
        try:
            return Sampling(**fields)
        except TypeError as error:
            raise RequestError(f"unknown sampling field: {error}") from error
    if "process" in body:
        fields = body["process"]
        default = getattr(engine.family, "process", None)
        if not isinstance(fields, dict) or default is None:
            raise RequestError("process applies to a canvas family and must be an object")
        try:
            return dataclasses.replace(default, **fields)
        except TypeError as error:
            raise RequestError(f"unknown process field: {error}") from error
    return None


def _inputs(server: Server, body: dict[str, Any]) -> tuple[Any, int]:
    """Model inputs and their prompt width, from text or token ids."""
    if ("prompt" in body) == ("input_ids" in body):
        raise RequestError("a request carries either prompt or input_ids")
    if "prompt" in body:
        if server.processor is None:
            raise RequestError("this server has no tokenizer; send input_ids")
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(prompt, list) or not prompt or not all(isinstance(item, str) for item in prompt):
            raise RequestError("prompt must be a string or a non-empty list of strings")
        inputs = server.processor(prompt)
        return inputs, int(inputs.tokens.shape[1])
    ids = body["input_ids"]
    if not isinstance(ids, list) or not ids or not all(
            isinstance(row, list) and row and all(type(token) is int for token in row) for row in ids):
        raise RequestError("input_ids must be a non-empty list of non-empty integer rows")
    if len({len(row) for row in ids}) != 1:
        raise RequestError("input_ids rows must share one width; left-pad shorter prompts")
    return np.asarray(ids, np.int32), len(ids[0])


def _record(value: Any) -> dict[str, Any]:
    return {field.name: np.asarray(getattr(value, field.name)).tolist()
            for field in dataclasses.fields(value)}


def _text(server: Server, result: Any, prompt_width: int) -> list[str] | None:
    if server.processor is None:
        return None
    tokens, lengths = np.asarray(result.tokens), np.asarray(result.lengths)
    return [server.processor.decode(tokens[row:row + 1, prompt_width:prompt_width + int(lengths[row])])[0]
            for row in range(tokens.shape[0])]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def _server(self) -> Server:
        return cast(Server, self.server)

    def log_message(self, format: str, *args: object) -> None:
        return

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
            return
        stats = self._server.engine.stats()
        self._json(HTTPStatus.OK, {"status": "ok", "version": self._server.engine.version.serial,
                                   **dataclasses.asdict(stats)})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length must be an integer"})
            return
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
            return
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise RequestError("the request body must be a JSON object")
            job, prompt_width = self._submit(body)
        except (RequestError, ValueError, TypeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except CapacityError as error:
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
            return
        except RuntimeError as error:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
            return
        if body.get("stream", False):
            self._stream(job, prompt_width)
        else:
            self._complete(job, prompt_width)

    def _submit(self, body: dict[str, Any]) -> tuple[GenerationJob[Any, Any], int]:
        engine = self._server.engine
        budget = body.get("max_new_tokens")
        if type(budget) is not int:
            raise RequestError("max_new_tokens must be an integer")
        seed = body.get("seed", None)
        if seed is not None and type(seed) is not int:
            raise RequestError("seed must be an integer")
        inputs, prompt_width = _inputs(self._server, body)
        key = jax.random.key(secrets.randbits(63) if seed is None else seed)
        job = engine.submit(inputs, budget, key=key, generation=_generation(engine, body))
        return job, prompt_width

    def _complete(self, job: GenerationJob[Any, Any], prompt_width: int) -> None:
        try:
            result = job.result()
        except CancelledError:
            self._json(HTTPStatus.CONFLICT, {"error": "the generation was cancelled"})
            return
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})
            return
        self._json(HTTPStatus.OK, {"result": _record(result), "text": _text(self._server, result, prompt_width),
                                   "version": job.version.serial})

    def _stream(self, job: GenerationJob[Any, Any], prompt_width: int) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for event in job.stream():
                self._chunk({"event": dataclasses.asdict(event)})
            result = job.result()
            self._chunk({"result": _record(result), "text": _text(self._server, result, prompt_width),
                         "version": job.version.serial})
        except (BrokenPipeError, ConnectionResetError):
            job.cancel()
            self.close_connection = True
            return
        except CancelledError:
            self._chunk({"error": "the generation was cancelled"})
        except Exception as error:
            self._chunk({"error": f"{type(error).__name__}: {error}"})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _chunk(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload) + "\n").encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    """An engine behind POST /generate and GET /health.

    ``processor`` maps prompt strings to model inputs and decodes results;
    without it, requests carry ``input_ids``. Streaming responses are
    newline-delimited JSON events followed by the result.
    """

    daemon_threads = True

    def __init__(self, engine: Engine[Any, Any], *, processor: Any = None,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), _Handler)
        self.engine = engine
        self.processor = processor
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server_address[:2]
        return str(host), int(port)

    def start(self) -> Server:
        if self._thread is not None:
            raise RuntimeError("the server was already started")
        self._thread = threading.Thread(target=self.serve_forever, name="dew-serve", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop accepting connections; the engine keeps its own lifecycle."""
        if self._thread is not None:
            self.shutdown()
            self._thread.join()
            self._thread = None
        self.server_close()

    def __enter__(self) -> Server:
        return self if self._thread is not None else self.start()

    def __exit__(self, kind, error, traceback) -> None:
        self.close()


def serve(engine: Engine[Any, Any], *, processor: Any = None, host: str = "127.0.0.1",
          port: int = 0) -> Server:
    """Start serving an open engine; ``port=0`` picks a free port."""
    return Server(engine, processor=processor, host=host, port=port).start()
