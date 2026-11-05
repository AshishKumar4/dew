"""Pinned Dew models, shared by bounded requests from isolated Python kernels."""

import base64
import io
import json
import math
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

import contract
from guests import GUEST_UIDS

MAX_REQUEST = 32_768
REQUEST_SECONDS = 90
CAPACITY = 8
"""Requests the service holds at once, queued or running, one per guest: a text model's serving slots,
the request queue, the guests admitted and a text batch's rows."""


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected an integer from {low} to {high}")


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"expected a finite number from {low} to {high}")


def text(value, length=4096):
    if not isinstance(value, str) or len(value) > length:
        raise ValueError(f"text must contain at most {length} characters")


def fields(value, names, optional=""):
    named = set(names.split())
    if not isinstance(value, dict) or not named <= set(value) <= named | set(optional.split()):
        raise ValueError("unsupported model request fields")


def guidance_record(value):
    """A CFG record a cell sent, as Dew's CFG takes it (its fields, `contract.record`: each required one
    given, no other), each field the service bounds within its bounds and any other at Dew's default."""
    from dew.sampling import CFG

    record = {entry["name"]: entry for entry in contract.record(CFG)}
    if not isinstance(value, dict) or not {name for name, entry in record.items() if "default" not in entry
                                           and not entry.get("omitted")} <= set(value) <= set(record):
        raise ValueError("unsupported guidance fields")
    bounds = {"scale": lambda scale: number(scale, 0, 20), "rescale": lambda rescale: number(rescale, 0, 1),
              "interval": interval}
    for name, given in value.items():
        if name in bounds:
            bounds[name](given)
        elif given != record[name].get("default"):
            raise ValueError(f"the live sampler takes CFG's {name} at its default")


def interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("guidance interval needs two bounds")
    number(value[0], 0, 1)
    number(value[1], value[0], 1)


def nullable(check, value, *bounds):
    """`check` of `value`, or None, where Dew takes a task's own default (contract.py)."""
    if value is not None:
        check(value, *bounds)


def validate(request):
    if not isinstance(request, dict):
        raise ValueError("a model request must be an object")
    op = request.get("op")
    if op == "describe":
        fields(request, "op repo revision")
    elif op == "text":
        fields(request, "op model prompt tokens key")
        text(request["model"], 256)
        text(request["prompt"])
        integer(request["tokens"], 1, contract.TOKENS)
        integer(request["key"], 0, 2**32 - 1)
        return request
    elif op == "sample":
        # A step count, a solver and guidance left out take the pipeline's own, as Dew's call does (`guidance`
        # absent, where None is the plain conditional prediction).
        fields(request, "op repo revision prompt negative prepare_key prepare_steps key steps solver",
               "guidance")
        text(request["prompt"])
        nullable(text, request["negative"])
        integer(request["key"], 0, 2**32 - 1)
        nullable(integer, request["steps"], 2, 40)
        integer(request["prepare_key"], 0, 2**32 - 1)
        nullable(integer, request["prepare_steps"], 2, 40)
        solver = request["solver"]
        if solver is not None:
            fields(solver, "name args")
            if solver["name"] not in contract.SOLVERS or solver["args"]:
                raise ValueError("the live sampler supports the displayed solvers with their default "
                                 "settings")
        guidance = request.get("guidance")
        if guidance is None or type(guidance) in (int, float):
            nullable(number, guidance, 0, 20)
        else:
            guidance_record(guidance)
    else:
        raise ValueError("unsupported model operation")
    text(request["repo"], 256)
    if request["revision"] is not None:
        text(request["revision"], 64)
    return request


class NativeModels:
    """The pinned models, served by one `dew.inference.Engine`: each text model's `Server` and the
    text-to-image pipeline's `ScoringServer` (`TextToImage.scoring`), so the engine admits, batches and
    steps every guest's request and the service only carries them."""

    def __init__(self, root=Path("/opt/live")):
        import jax.numpy as jnp
        import numpy as np
        import progress

        import dew
        from dew import sampling
        from dew.inference import Engine, ScoringServer, Server
        from dew.interop import PretrainedDecoder
        from dew.sampling import CFG, Sampling

        print("MODEL_PHASE", time.time(), "pipeline load", flush=True)
        self.np, self.progress = np, progress
        self.cfg = CFG
        self.solvers = {name: getattr(sampling, name) for name in contract.SOLVERS}
        self.repo, self.revision = (root / "text-to-image").read_text().strip().split("@")
        self.pipe = dew.pipeline(self.repo, revision=self.revision)
        # A pass of one request walks with its lone call's own programs: a cell run here draws the pixels
        # it draws run locally.
        self.images = ScoringServer.from_task(*self.pipe.scoring(images=1))
        self.text_servers = {}
        for line in (root / "text-models").read_text().split():
            name, _ = line.split("@")
            print("MODEL_PHASE", time.time(), "text load", name, flush=True)
            bundle = PretrainedDecoder.load(f"/opt/models/{name}", **{
                **contract.LOADED, "dtype": getattr(jnp, contract.LOADED["dtype"])})
            task = bundle.text_generation(sampling=Sampling(**contract.GREEDY))
            self.text_servers[name] = Server.from_task(task, slots=CAPACITY, capacity=256)
        self.engine = Engine(self.images, *self.text_servers.values(), name="live-models")
        self.engine.start()
        print("MODEL_PHASE", time.time(), "warm sample", flush=True)
        self.sample({"repo": self.repo, "revision": self.revision,
                     "prompt": "the northern lights over a frozen lake at night", "negative": "",
                     "prepare_key": 3, "prepare_steps": 15, "steps": 15, "key": 3,
                     "solver": {"name": "DPMSolverMultistep", "args": {}},
                     "guidance": {"scale": 6, "interval": [0.15, 0.9], "rescale": 0}}, lambda *_: None)
        for name in self.text_servers:
            print("MODEL_PHASE", time.time(), "warm text", name, flush=True)
            self.text({"model": name, "prompt": "The capital of France is", "tokens": 24, "key": 0},
                      lambda *_: None)

    def describe(self, request):
        if request["repo"] != self.repo:
            raise ValueError("the live sampler serves only its pinned public model")
        if request["revision"] not in (None, self.revision):
            raise self.progress.StalePage("This page was updated. Reload it to use the current model.")
        return {"repo": self.repo, "revision": self.revision}

    def sample(self, request, emit):
        """The request's image, as Dew's `pipe(pipe.prepare([prompt], key=prepare_key, steps=prepare_steps,
        unconditional=negative), key=key, steps=steps, ...)` draws it, each step's preview emitted as the
        walk runs (`progress.preview_png`), then the decode's."""
        answer, answered = self._sample(request, emit)
        return answered(answer.result())

    def text(self, request, emit):
        """The request's greedy continuation, its text streamed as its tokens arrive."""
        answer, answered = self._text(request, emit)
        return answered(answer.result())

    def submitted(self, request, emit):
        """A validated sample or text request submitted to the engine: its future, and what turns the
        future's result into the reply, run on the caller's thread, off the engine's."""
        if request["op"] == "text":
            return self._text(request, emit)
        return self._sample(request, emit)

    def _sample(self, request, emit):
        from dew.inference import ImageRequest
        from dew.objectives.base import OMITTED

        self.describe(request)
        options = request.get("guidance", OMITTED)
        guidance = (options if options is OMITTED or options is None or type(options) in (int, float) else
                    self.cfg(**{name: tuple(given) if isinstance(given, list) else given
                                for name, given in options.items()}))
        steps = self.pipe.steps if request["steps"] is None else request["steps"]
        process, times = self.pipe.prepared_process(steps)
        walked = len(process.times(steps) if times is None else times) - 1

        def stepped(step, rows):
            emit({"progress": {"step": step, "steps": steps}, "png": self.progress.preview_png(rows[0])})
            if step == walked:
                emit({"progress": {"stage": "decode"}, "png": None})

        solver = request["solver"]
        answer = self.images.submit(ImageRequest(
            request["prompt"], request["key"], steps=steps, guidance=guidance,
            solver=None if solver is None else self.solvers[solver["name"]](), negative=request["negative"],
            prepare_key=request["prepare_key"],
            prepare_steps=self.pipe.steps if request["prepare_steps"] is None else request["prepare_steps"],
            on_step=stepped))
        return answer, self._image_reply

    def _image_reply(self, images):
        result = images.host()
        if not self.np.isfinite(self.np.asarray(result.images)).all():
            raise ArithmeticError("the sampler returned non-finite pixels")
        encoded = []
        for image in result.pil():
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode())
        pixels = self.np.asarray(result.images)
        if pixels.dtype.name not in ("float16", "float32", "float64", "bfloat16"):
            raise ValueError("the live sampler cannot transport this pixel dtype exactly")
        return {"pngs": encoded, "pixels": base64.b64encode(pixels.tobytes()).decode(),
                "shape": list(pixels.shape),
                "dtype": pixels.dtype.name if pixels.dtype.name == "bfloat16" else pixels.dtype.str}

    def _text(self, request, emit):
        if request["model"] not in self.text_servers:
            raise ValueError("the live kernel serves only its pinned text models")
        server = self.text_servers[request["model"]]
        ticket = server.submit(request["prompt"], request["tokens"], key=request["key"])

        def streamed(ticket):
            decoded = server.processor.decode(self.np.asarray([ticket.tokens], dtype="int32"))[0]
            emit({"text": decoded, "first_token_seconds": ticket.first - ticket.submitted})

        ticket.add_tokens_callback(streamed)
        return ticket, lambda generation: {"text": generation.text}


class ModelService(socketserver.ThreadingUnixStreamServer):
    """Carries guests' requests to the models: one request a guest uid at a time, CAPACITY in all, each
    answered by the models' engine on its connection's thread."""

    daemon_threads = True

    def __init__(self, path, models, uids=GUEST_UIDS):
        self.path, self.models, self.uids = Path(path), models, frozenset(uids)
        self.pending = {}
        self.lock = threading.Lock()
        self.path.unlink(missing_ok=True)
        super().__init__(str(path), ModelRequest)
        os.chmod(path, 0o666)
        self.listener = threading.Thread(target=self.serve_forever, daemon=True)
        self.listener.start()

    def process_request(self, request, address):
        uid = struct.unpack("3i", request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        token = {"submitted": False}
        with self.lock:
            if uid not in self.uids:
                error = {"name": "PermissionError",
                         "message": "model requests require an isolated kernel uid"}
            elif uid in self.pending or len(self.pending) >= CAPACITY:
                error = {"name": "ValueError", "message": "the shared model request queue is full; try again"}
            else:
                self.pending[uid] = token
                error = None
        if error is not None:
            try:
                request.sendall(json.dumps({"error": error}).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError) as gone:
                print(f"a refused model request's client left before hearing why: {gone!r}", file=sys.stderr,
                      flush=True)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.release(uid, token)
            raise

    def process_request_thread(self, request, address):
        uid = struct.unpack("3i", request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        token = self.pending[uid]
        try:
            super().process_request_thread(request, address)
        finally:
            # A request the engine holds keeps its guest's admission until the engine settles it
            # (`submitted`); any other ends with its connection.
            if not token["submitted"]:
                self.release(uid, token)

    def submitted(self, uid, future):
        """`future`, the engine's work for `uid`'s request, holds the guest's admission until it settles,
        however long its connection waits for it."""
        token = self.pending[uid]
        token["submitted"] = True
        future.add_done_callback(lambda _: self.release(uid, token))

    def release(self, uid, token):
        with self.lock:
            if self.pending.get(uid) is token:
                del self.pending[uid]

    def server_close(self):
        self.shutdown()
        super().server_close()
        self.path.unlink(missing_ok=True)


class ModelRequest(socketserver.StreamRequestHandler):
    def handle(self):
        uid = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        # One frame at a time, and none after the reply: the engine's progress callbacks and the handler's
        # reply write under one lock, and the reply ends the connection's output.
        writing = threading.Lock()
        ended = False

        def emit(value, *, final=False):
            nonlocal ended
            with writing:
                if ended:
                    return
                ended = final
                try:
                    try:
                        line = json.dumps(value, allow_nan=False)
                    except ValueError as error:
                        # A non-finite number has no JSON form: the client hears so in place of the reply.
                        why = f"the model's reply holds a non-finite number: {error}"
                        line = json.dumps({"error": {"name": "ValueError", "message": why}})
                    self.wfile.write(line.encode() + b"\n")
                    self.wfile.flush()
                except OSError as gone:
                    print(f"a model request's client left before its reply: {gone!r}", file=sys.stderr,
                          flush=True)

        try:
            self.request.settimeout(10)
            raw = self.rfile.readline(MAX_REQUEST + 1)
            if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
                raise ValueError("the model request is too large")
            request = validate(json.loads(raw))
            if request["op"] == "describe":
                emit({"result": self.server.models.describe(request)}, final=True)
                return
            future, answered = self.server.models.submitted(request, emit)
            self.server.submitted(uid, future)
            try:
                result = future.result(REQUEST_SECONDS)
            except FuturesTimeout:
                # A request still queued is dropped where the engine would take it; a running one finishes,
                # and holds the guest's admission until it has.
                future.cancel()
                raise TimeoutError("the shared model did not finish within 90 seconds") from None
            emit({"result": answered(result)}, final=True)
        except Exception as error:
            emit({"error": {"name": type(error).__name__, "message": str(error)}}, final=True)
        finally:
            with writing:
                ended = True


if __name__ == "__main__":
    models = NativeModels()
    with ModelService("/run/dew/model/model.sock", models):
        Path("/run/dew/model/ready").write_text("ready\n")
        threading.Event().wait()
