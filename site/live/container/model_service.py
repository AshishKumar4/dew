"""Pinned Dew models, shared by bounded requests from isolated Python kernels."""

import base64
import io
import json
import math
import os
import queue
import socket
import socketserver
import struct
import threading
import time
from pathlib import Path

MAX_REQUEST = 32_768
REQUEST_SECONDS = 90
KERNEL_UIDS = range(6100, 6200)


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected an integer from {low} to {high}")


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"expected a finite number from {low} to {high}")


def text(value, length=4096):
    if not isinstance(value, str) or len(value) > length:
        raise ValueError(f"text must contain at most {length} characters")


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("unsupported model request fields")


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
        integer(request["tokens"], 1, 64)
        integer(request["key"], 0, 2**32 - 1)
        return request
    elif op == "sample":
        fields(request, "op repo revision prompt negative prepare_key prepare_steps "
                        "key steps solver guidance")
        text(request["prompt"])
        text(request["negative"])
        integer(request["key"], 0, 2**32 - 1)
        integer(request["steps"], 2, 40)
        integer(request["prepare_key"], 0, 2**32 - 1)
        integer(request["prepare_steps"], 2, 40)
        solver = request["solver"]
        fields(solver, "name args")
        if solver["name"] not in ("DPMSolverMultistep", "Heun", "EulerAncestral") or solver["args"]:
            raise ValueError("the live sampler supports the displayed solvers with their default settings")
        guidance = request["guidance"]
        if guidance is None:
            text(request["repo"], 256)
            if request["revision"] is not None:
                text(request["revision"], 64)
            return request
        fields(guidance, "scale interval rescale")
        number(guidance["scale"], 0, 20)
        number(guidance["rescale"], 0, 1)
        interval = guidance["interval"]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("guidance interval needs two bounds")
        number(interval[0], 0, 1)
        number(interval[1], interval[0], 1)
    else:
        raise ValueError("unsupported model operation")
    text(request["repo"], 256)
    if request["revision"] is not None:
        text(request["revision"], 64)
    return request


class NativeModels:
    def __init__(self, root=Path("/opt/live")):
        import jax.numpy as jnp
        import numpy as np
        import progress

        from dew.inference import Server
        from dew.interop import PretrainedDecoder
        from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, Sampling, TextToImage

        print("MODEL_PHASE", time.time(), "pipeline load", flush=True)
        self.np, self.progress = np, progress
        self.cfg = CFG
        self.solvers = {solver.__name__: solver for solver in (DPMSolverMultistep, EulerAncestral, Heun)}
        self.repo, self.revision = (root / "text-to-image").read_text().strip().split("@")
        self.pipe = progress.Reporting(TextToImage.from_pretrained(self.repo, revision=self.revision))
        self.text_servers = {}
        for line in (root / "text-models").read_text().split():
            name, _ = line.split("@")
            print("MODEL_PHASE", time.time(), "text load", name, flush=True)
            bundle = PretrainedDecoder.load(f"/opt/models/{name}", dtype=jnp.float32, max_seq_len=256)
            task = bundle.text_generation(sampling=Sampling(temperature=0))
            self.text_servers[name] = Server.from_task(task, slots=8, capacity=256)
        print("MODEL_PHASE", time.time(), "warm sample", flush=True)
        self.sample({"repo": self.repo, "revision": self.revision,
                     "prompt": "the northern lights over a frozen lake at night", "negative": "",
                     "prepare_key": 3, "prepare_steps": 15, "steps": 15, "key": 3,
                     "solver": {"name": "DPMSolverMultistep", "args": {}},
                     "guidance": {"scale": 6, "interval": [0.15, 0.9], "rescale": 0}}, lambda *_: None)
        for name in self.text_servers:
            print("MODEL_PHASE", time.time(), "warm text", name, flush=True)
            warmed = self.text([{"model": name, "prompt": "The capital of France is",
                                 "tokens": 24, "key": 0}])
            if isinstance(warmed[0], Exception):
                raise warmed[0]

    def describe(self, request):
        if request["repo"] != self.repo:
            raise ValueError("the live sampler serves only its pinned public model")
        if request["revision"] not in (None, self.revision):
            raise self.progress.StalePage("This page was updated. Reload it to use the current model.")
        return {"repo": self.repo, "revision": self.revision}

    def sample(self, request, emit):
        self.describe(request)
        options = request["guidance"]
        guidance = (None if options is None else
                    self.cfg(options["scale"], interval=tuple(options["interval"]),
                             rescale=options["rescale"]))
        inputs = self.pipe.prepare([request["prompt"]], key=request["prepare_key"],
                                   steps=request["prepare_steps"],
                                   unconditional=request["negative"])
        previous = self.progress._show
        self.progress._show = lambda report, png=None: emit({"progress": report, "png": png})
        try:
            result = self.pipe(inputs, key=request["key"], steps=request["steps"], guidance=guidance,
                               solver=self.solvers[request["solver"]["name"]]()).host()
        finally:
            self.progress._show = previous
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

    def text(self, requests, emitters=None):
        tickets = []
        for index, request in enumerate(requests):
            try:
                if request["model"] not in self.text_servers:
                    raise ValueError("the live kernel serves only its pinned text models")
                server = self.text_servers[request["model"]]
                ticket = server.submit(request["prompt"], request["tokens"], key=request["key"])
                if emitters is not None:
                    def streamed(ticket, server=server, emit=emitters[index]):
                        decoded = server.processor.decode(self.np.asarray([ticket.tokens], dtype="int32"))[0]
                        emit({"text": decoded, "first_token_seconds": ticket.first - ticket.submitted})
                    ticket.add_tokens_callback(streamed)
                tickets.append(ticket)
            except Exception as error:
                tickets.append(error)
        for server in self.text_servers.values():
            server.run()
        return [ticket if isinstance(ticket, Exception) else {"text": ticket.result().text}
                for ticket in tickets]


class ModelService(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, models, uids=KERNEL_UIDS):
        self.path, self.models, self.uids = Path(path), models, frozenset(uids)
        self.jobs = queue.Queue(maxsize=8)
        self.pending = {}
        self.lock = threading.Lock()
        self.path.unlink(missing_ok=True)
        super().__init__(str(path), ModelRequest)
        os.chmod(path, 0o666)
        self.worker = threading.Thread(target=self.compute, daemon=True)
        self.worker.start()
        self.listener = threading.Thread(target=self.serve_forever, daemon=True)
        self.listener.start()

    def process_request(self, request, address):
        uid = struct.unpack("3i", request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        token = {"queued": False}
        with self.lock:
            if uid not in self.uids:
                error = {"name": "PermissionError",
                         "message": "model requests require an isolated kernel uid"}
            elif uid in self.pending or len(self.pending) >= 8:
                error = {"name": "ValueError", "message": "the shared model request queue is full; try again"}
            else:
                self.pending[uid] = token
                error = None
        if error is not None:
            try:
                request.sendall(json.dumps({"error": error}).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
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
            if not token["queued"]:
                self.release(uid, token)

    def release(self, uid, token):
        with self.lock:
            if self.pending.get(uid) is token:
                del self.pending[uid]

    def compute(self):
        missing = object()
        held = missing
        while True:
            job = self.jobs.get() if held is missing else held
            held = missing
            if job is None:
                return
            group = [job]
            request, emit = job
            try:
                if request["op"] == "text":
                    while len(group) < 8:
                        try:
                            following = self.jobs.get_nowait()
                        except queue.Empty:
                            break
                        if following is None or following[0]["op"] != "text":
                            held = following
                            break
                        group.append(following)
                    results = self.models.text([request for request, _ in group], [emit for _, emit in group])
                    for result, (_, send) in zip(results, group, strict=True):
                        if isinstance(result, Exception):
                            send({"error": {"name": type(result).__name__, "message": str(result)}})
                        else:
                            send({"result": result})
                else:
                    result = (self.models.describe(request) if request["op"] == "describe" else
                              self.models.sample(request, emit))
                    emit({"result": result})
            except Exception as error:
                for _, send in group:
                    send({"error": {"name": type(error).__name__, "message": str(error)}})

    def server_close(self):
        self.shutdown()
        self.jobs.put(None)
        super().server_close()
        self.path.unlink(missing_ok=True)


class ModelRequest(socketserver.StreamRequestHandler):
    def handle(self):
        uid = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        token = self.server.pending[uid]
        completed = threading.Event()

        def emit(value, release=True):
            if release and ("result" in value or "error" in value):
                self.server.release(uid, token)
            try:
                self.wfile.write(json.dumps(value, allow_nan=False).encode() + b"\n")
                self.wfile.flush()
            except (OSError, ValueError):
                pass
            finally:
                if "result" in value or "error" in value:
                    completed.set()

        try:
            self.request.settimeout(10)
            raw = self.rfile.readline(MAX_REQUEST + 1)
            if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
                raise ValueError("the model request is too large")
            request = validate(json.loads(raw))
            token["queued"] = True
            try:
                self.server.jobs.put_nowait((request, emit))
            except queue.Full:
                token["queued"] = False
                raise
            if not completed.wait(REQUEST_SECONDS):
                raise TimeoutError("the shared model did not finish within 90 seconds")
        except Exception as error:
            emit({"error": {"name": type(error).__name__, "message": str(error)}}, release=False)


if __name__ == "__main__":
    models = NativeModels()
    with ModelService("/run/dew/model/model.sock", models):
        Path("/run/dew/model/ready").write_text("ready\n")
        threading.Event().wait()
