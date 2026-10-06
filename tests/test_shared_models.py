"""Shared model requests keep untrusted inputs away from model memory."""
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "site/live/container"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def service():
    return module("model_service")


def test_requests_are_bounded_before_native_execution(service):
    valid = {"op": "sample", "repo": "model", "revision": "pinned", "prompt": "a lake",
             "negative": "", "prepare_key": 3, "prepare_steps": 15, "key": 3, "steps": 15,
             "solver": {"name": "Heun", "args": {}},
             "guidance": {"scale": 6, "interval": [0.15, 0.9], "rescale": 0}}
    assert service.validate(valid) == valid
    for change in ({"steps": 10000}, {"key": -1}, {"guidance": {"scale": float("nan")}},
                   {"prompt": "x" * 4097}, {"__reduce__": "code"}, {"solver": {"name": "eval"}}):
        with pytest.raises(ValueError):
            service.validate({**valid, **change})
    with pytest.raises(ValueError):
        service.validate({"op": "exec", "code": "print('not allowed')"})


def test_a_foreign_uid_cannot_submit_a_request(service, tmp_path):
    with (service.ModelService(tmp_path / "model.sock", object(), uids=[]) as server,
          socket.socket(socket.AF_UNIX) as connection):
        connection.connect(str(server.path))
        request = {"op": "describe", "repo": "model", "revision": "pinned"}
        connection.sendall(json.dumps(request).encode() + b"\n")
        result = json.loads(connection.makefile("rb").readline())
        assert result["error"]["name"] == "PermissionError"


def test_image_transport_keeps_native_pixels_and_preparation_keys(service, tmp_path):
    import numpy as np
    from test_inference import make_run

    from dew.sampling import CFG, Heun, TextToImage

    progress = module("progress")
    client = module("model_client")
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress, native.cfg = np, progress, CFG
    native.repo, native.revision = "model", "pinned"
    native.pipe = progress.Reporting(pipe)
    native.solvers = {"Heun": Heun}
    request = {"repo": "model", "revision": "pinned", "prompt": "a lake", "negative": "",
               "prepare_key": 3, "prepare_steps": 4, "key": 5, "steps": 4,
               "solver": {"name": "Heun", "args": {}},
               "guidance": {"scale": 2, "interval": [0, 1], "rescale": 0}}
    prepared = pipe.prepare(["a lake"], key=3, steps=4, unconditional="")
    expected = pipe(prepared, key=5, steps=4, solver=Heun(), guidance=CFG(2)).host().images
    reports = []
    result = client.ImageResult(native.sample(request, reports.append))
    np.testing.assert_array_equal(result.host().images, expected)
    assert len(result.pil()) == 1
    assert [report["progress"] for report in reports] == [
        {"step": 1, "steps": 4}, {"step": 2, "steps": 4}, {"step": 3, "steps": 4}, {"stage": "decode"}]


def test_sequential_calls_release_the_same_kernel_uid(service, tmp_path, monkeypatch):
    import os

    class Models:
        def describe(self, request):
            return {"repo": request["repo"], "revision": request["revision"]}

    client = module("model_client")
    with service.ModelService(tmp_path / "model.sock", Models(), uids=[os.getuid()]) as server:
        monkeypatch.setattr(client, "SOCKET", str(server.path))
        for _ in range(3):
            result = client.request({"op": "describe", "repo": "model", "revision": "pinned"})
            assert result == {"repo": "model", "revision": "pinned"}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pixel_transport_does_not_reconstruct_floats_from_a_png(dtype):
    import base64

    import ml_dtypes
    import numpy as np

    client = module("model_client")
    kind = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float32
    pixels = np.array([[[[0.00001, -0.234567, 0.99999]]]], kind)
    result = client.ImageResult({"pngs": [], "pixels": base64.b64encode(pixels.tobytes()).decode(),
                                 "shape": list(pixels.shape),
                                 "dtype": dtype if dtype == "bfloat16" else pixels.dtype.str})
    np.testing.assert_array_equal(result.images, pixels)


def test_a_timed_out_handler_does_not_release_ongoing_compute(service, tmp_path, monkeypatch):
    import os
    import threading

    finished = threading.Event()

    class Models:
        def describe(self, request):
            finished.wait(1)
            return {"repo": request["repo"], "revision": request["revision"]}

    client = module("model_client")
    monkeypatch.setattr(service, "REQUEST_SECONDS", 0.05)
    with service.ModelService(tmp_path / "model.sock", Models(), uids=[os.getuid()]) as server:
        monkeypatch.setattr(client, "SOCKET", str(server.path))
        payload = {"op": "describe", "repo": "model", "revision": "pinned"}
        try:
            with pytest.raises(ValueError, match="did not finish"):
                client.request(payload)
            with pytest.raises(ValueError, match="queue is full"):
                client.request(payload)
        finally:
            finished.set()


def test_an_early_rejection_remains_readable_after_the_peer_closes(service, tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    closed = threading.Event()
    client = module("model_client")

    class DelayedSocket(socket.socket):
        def sendall(self, data, *args):
            assert closed.wait(2), "the rejection did not close its connection"
            return super().sendall(data, *args)

    with service.ModelService(tmp_path / "model.sock", object(), uids=[]) as server:
        shutdown = server.shutdown_request

        def rejected(request):
            shutdown(request)
            closed.set()

        monkeypatch.setattr(server, "shutdown_request", rejected)
        monkeypatch.setattr(client, "SOCKET", str(server.path))
        monkeypatch.setattr(client, "socket", SimpleNamespace(
            socket=DelayedSocket, AF_UNIX=socket.AF_UNIX, SOCK_STREAM=socket.SOCK_STREAM))
        with pytest.raises(ValueError, match="isolated kernel uid"):
            client.request({"op": "describe", "repo": "model", "revision": "pinned"})


def test_shared_text_delegates_prompt_preparation_to_dew_server(service):
    from types import SimpleNamespace

    calls = []

    class Server:
        def submit(self, prompt, budget, *, key):
            calls.append((prompt, budget, key))
            return SimpleNamespace(result=lambda: SimpleNamespace(text=["ok"]))

        def run(self):
            pass

    models = service.NativeModels.__new__(service.NativeModels)
    models.text_servers = {"model": Server()}
    request = {"model": "model", "prompt": "hello", "tokens": 2, "key": 0}
    assert models.text([request]) == [{"text": ["ok"]}]
    assert calls == [("hello", 2, 0)]
