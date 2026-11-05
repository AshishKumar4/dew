"""Shared model requests keep untrusted inputs away from model memory."""
import ast
import contextlib
import dataclasses
import sys
import threading
from pathlib import Path

import pytest
from tool_support import load, run_from

SITE = Path(__file__).resolve().parents[1] / "site"


@pytest.fixture(scope="module")
def service():
    with run_from("model_service", "site/live/container") as module:
        yield module


def test_image_transport_keeps_native_pixels_and_preparation_keys(service, tmp_path):
    import numpy as np

    from dew.sampling import CFG, Heun

    client, pipe, native = live_kernel(service, tmp_path)
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


@contextlib.contextmanager
def stood_in(client):
    """The live kernel's stand-ins for Dew's modules (`model_client.install`) for the block, and Dew's
    own back after it, whichever modules the stand-ins name: a later test in this process imports
    Dew's own, where restoring a list of names once missed `dew.inference`, and `dew.inference.scoring`
    found the stand-in."""
    held = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "dew"}
    try:
        client.install()
        yield held
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "dew"]:
            del sys.modules[name]
        sys.modules.update(held)


def test_the_live_sampler_cell_runs_whole_through_the_stand_ins_as_dew_runs_it(service, tmp_path,
                                                                                 monkeypatch):
    """The live editor's cell (site/src/data/sampler.py), run whole through the live kernel's stand-ins,
    asks the shared model process for the model it loaded from the image's pin
    (site/live/container/text-to-image), which serves that revision alone, and its image is the one Dew's
    own calls of the cell make, its solver and guidance included: a cell left on an earlier revision, or
    calling what the stand-ins bind otherwise than Dew, fails every unedited Run (ledger 1159, 2339). A
    tiny run stands in for the pinned model, under the pinned name, at 4 steps where the page takes 15."""
    import numpy as np


    client, pipe, native = live_kernel(service, tmp_path)
    pinned = (SITE / "live" / "container" / "text-to-image").read_text()
    native.repo, native.revision = pinned.strip().split("@")
    asked = []

    def shared(payload):
        asked.append(payload["op"])
        with as_service(real):
            request = service.validate(payload)
            if payload["op"] == "describe":
                return native.describe(request)
            return native.sample(request, lambda _: None)

    monkeypatch.setattr(client, "request", shared)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    # The cell's calls as the page writes them, at 4 steps where it takes 15: the tiny run compiles each step.
    tree = ast.parse((SITE / "src" / "data" / "sampler.py").read_text())
    for keyword in ast.walk(tree):
        if isinstance(keyword, ast.keyword) and keyword.arg == "steps":
            keyword.value = ast.copy_location(ast.Constant(4), keyword.value)
    cell = compile(tree, "sampler.py", "exec")
    # Dew's own run of the cell first: the stand-ins, once installed, refuse every other Dew module to the
    # rest of the test's process.
    import dew

    native_namespace = {}
    with monkeypatch.context() as patched:
        patched.setattr(dew, "pipeline", lambda *args, **kwargs: pipe)
        exec(cell, native_namespace)
    namespace = {}
    with stood_in(client) as real:
        exec(cell, namespace)
    assert asked == ["describe", "sample"]
    assert (namespace["pipe"].repo, namespace["pipe"].revision) == (native.repo, native.revision)
    np.testing.assert_array_equal(namespace["result"].host().images, native_namespace["result"].host().images)


@contextlib.contextmanager
def as_service(held):
    """Dew's own modules (`held`, those `stood_in` put aside) for the block, as the shared model process holds
    them, under a cell's stand-ins, which take theirs back after it: the service runs in a process of its
    own."""
    stood = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "dew"}
    finders = list(sys.meta_path)
    sys.modules.update(held)
    sys.meta_path[:] = [finder for finder in finders if type(finder).__name__ != "Elsewhere"]
    try:
        yield
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "dew"]:
            del sys.modules[name]
        sys.modules.update(stood)
        sys.meta_path[:] = finders


def live_kernel(service, tmp_path, served=None):
    """The live page's two sides over a tiny run: the guest's client (its contract read from Dew), the tiny
    run's pipeline, and the shared process's native models serving it through an engine, as "model" at
    "pinned", with the guidance record and solvers it takes from Dew. `served` makes the engine's image
    kind of the pipeline's (`TextToImage.scoring`), a double of it where a test hands one."""
    import numpy as np
    from test_inference import make_run

    from dew import sampling
    from dew.inference import Engine, ScoringServer, TextToImage

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress, native.pipe = np, progress, pipe
    scoring, variables = pipe.scoring(images=1)
    native.images = ScoringServer.from_task(scoring if served is None else served(scoring), variables)
    native.engine = Engine(native.images)
    native.engine.start()
    native.cfg = sampling.CFG
    native.solvers = {name: getattr(sampling, name)
                      for name in ("DPMSolverMultistep", "EulerAncestral", "Heun")}
    native.repo, native.revision = "model", "pinned"
    return client, pipe, native


def test_a_live_cell_takes_dews_arguments_and_is_refused_only_what_the_service_does_not_serve(
        service, tmp_path, monkeypatch):
    """The live kernel's stand-ins take what Dew's own calls take, read from Dew (site/live/container/
    contract.py): a prepare without its key fails as Dew's does; a call naming no steps, solver or guidance
    takes the pipeline's own, the image the service returns the one Dew's call makes; `CFG` holds Dew's
    defaults, and `Sampling` Dew's, so a greedy policy named in full is the greedy policy; an argument the
    service does not serve, or one of another type than Dew's default (decode=1), is refused by name, and so
    is a solver of the cell's own named as one the service runs; and the service refuses a solver outside the
    contract's (ledger 1159)."""
    import numpy as np

    from dew.sampling import CFG as CFGOfDew, DPMSolverMultistep as DPMOfDew, Heun as HeunOfDew

    client, pipe, native = live_kernel(service, tmp_path)
    monkeypatch.setattr(client, "request",
                        lambda payload: native.sample(service.validate(payload), lambda _: None))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    with pytest.raises(TypeError):
        pipe.prepare(["a lake"])
    with stood_in(client):
        from dew.sampling import CFG, DPMSolverMultistep, Heun, Sampling

        live = client.Pipeline("model", "pinned")
        with pytest.raises(TypeError):
            live.prepare(["a lake"])
        images = live(live.prepare(["a lake"], key=3), key=5).host().images
        heun = live(["a lake"], key=5, steps=4, solver=Heun(), guidance=CFG(2)).host().images
        defaults = live(["a lake"], key=5, steps=4, solver=DPMSolverMultistep(order=2)).host().images
        with pytest.raises(ValueError):
            live(["a lake"], key=5, steps=4, solver=DPMSolverMultistep(order=3))
        for decode in (False, 1):
            with pytest.raises(ValueError):
                live(["a lake"], key=5, decode=decode)

        class Mine:
            pass

        Mine.__name__ = "Heun"
        with pytest.raises(ValueError):
            live(["a lake"], key=5, solver=Mine())
        for greedy in (Sampling(temperature=0), Sampling(0), Sampling(temperature=0, top_p=1.0)):
            client.Decoder("model").text_generation(sampling=greedy)
        with pytest.raises(ValueError):
            client.Decoder("model").text_generation(sampling=Sampling(temperature=0.5))
    np.testing.assert_array_equal(images, pipe(pipe.prepare(["a lake"], key=3), key=5).host().images)
    np.testing.assert_array_equal(heun, pipe(["a lake"], key=5, steps=4, solver=HeunOfDew(),
                                             guidance=CFGOfDew(2)).host().images)
    np.testing.assert_array_equal(defaults, pipe(["a lake"], key=5, steps=4, solver=DPMOfDew()).host().images)
    request = {"op": "sample", "repo": "model", "revision": "pinned", "prompt": "a lake", "negative": None,
               "prepare_key": 3, "prepare_steps": None, "key": 5, "steps": None,
               "solver": {"name": "Euler", "args": {}}}
    with pytest.raises(ValueError):
        service.validate(request)



SLOW = {"op": "sample", "repo": "model", "revision": "pinned", "prompt": "a lake", "negative": None,
        "prepare_key": 3, "prepare_steps": 40, "key": 5, "steps": 40, "solver": None}
"""An image request whose 40-step walk compiles past a shortened wait."""
DESCRIBE = {"op": "describe", "repo": "model", "revision": "pinned"}


def connected(path, payload):
    """A guest's connection to the service at `path` carrying `payload`, and its lines."""
    import json
    import socket

    client = socket.socket(socket.AF_UNIX)
    client.connect(str(path))
    client.sendall(json.dumps(payload).encode() + b"\n")
    return client, client.makefile()


def reply(lines):
    """A connection's reply, after the progress before it."""
    import json

    return next(message for message in map(json.loads, lines) if "result" in message or "error" in message)


@contextlib.contextmanager
def serving(service, native, path):
    """The service over `native`, admitting this process's uid as a guest's."""
    import os

    shared = service.ModelService(path, native, uids=(os.getuid(),))
    try:
        yield shared
    finally:
        shared.server_close()


@dataclasses.dataclass
class Holding:
    """A double of the pipeline's image kind (`TextToImage.scoring`) whose passes wait for `release` once
    they start, so a test knows a request runs, not by a clock: `entered` is set as a pass starts, and
    `passed` holds each pass's request keys."""

    entered: threading.Event = dataclasses.field(default_factory=threading.Event)
    release: threading.Event = dataclasses.field(default_factory=threading.Event)
    passed: list = dataclasses.field(default_factory=list)

    def __call__(self, scoring):
        import jax

        from dew.inference.images import ImageScoring

        holding = self

        class Held(ImageScoring):
            def forward(self, variables, rows):
                holding.passed.append([int(jax.random.key_data(row.key)[-1]) for row in rows])
                holding.entered.set()
                assert holding.release.wait(timeout=600)
                return super().forward(variables, rows)

        return Held(**{field.name: getattr(scoring, field.name) for field in dataclasses.fields(scoring)})


def test_a_guest_whose_request_outlasts_its_connection_stays_admitted_until_the_engine_settles_it(
        service, tmp_path, monkeypatch):
    """A guest's image request the engine runs when its connection's wait runs out keeps the guest's one
    admission: its connection hears the timeout and nothing after it, its request runs on (its cancel
    fails), a second request of the guest's is refused while it runs, and the guest is admitted again once
    the engine has settled it, so a slow pass never lets one guest queue a second job past the service's
    bounds. The service's real models run it through the socket, their pass held (`Holding`)."""
    holding = Holding()
    _, _, native = live_kernel(service, tmp_path, served=holding)
    monkeypatch.setattr(service, "REQUEST_SECONDS", 5)
    path = tmp_path / "model.sock"
    with serving(service, native, path):
        try:
            first, lines = connected(path, SLOW)
            with first:
                assert reply(lines)["error"]["name"] == "TimeoutError"
                assert holding.entered.is_set()
                second, refused = connected(path, SLOW)
                with second:
                    assert reply(refused)["error"]["name"] == "ValueError"
                assert len(holding.passed) == 1
        finally:
            holding.release.set()
        # The engine answers in order: once a later request is answered, the first has been settled.
        native.sample(service.validate({**SLOW, "key": 6}), lambda _: None)
        assert lines.read() == ""
        third, lines = connected(path, DESCRIBE)
        with third:
            assert "result" in reply(lines)


def test_a_guest_whose_queued_request_times_out_frees_its_admission_and_its_request_never_runs(
        service, tmp_path, monkeypatch):
    """A guest's request still queued behind a running one when its connection's wait runs out is
    cancelled: the guest is admitted again at once, and the engine drops the request where it would take
    it, never running it, while the request ahead runs on."""
    holding = Holding()
    _, _, native = live_kernel(service, tmp_path, served=holding)
    monkeypatch.setattr(service, "REQUEST_SECONDS", 0.05)
    path = tmp_path / "model.sock"
    with serving(service, native, path):
        ahead = threading.Thread(target=native.sample, args=(service.validate({**SLOW, "key": 6}),
                                                             lambda _: None))
        ahead.start()
        try:
            assert holding.entered.wait(timeout=600)
            queued, lines = connected(path, SLOW)
            with queued:
                assert reply(lines)["error"]["name"] == "TimeoutError"
            described, lines = connected(path, DESCRIBE)
            with described:
                assert "result" in reply(lines)
        finally:
            holding.release.set()
        ahead.join()
        # The engine answers in order: once a later request is answered, every earlier one is settled.
        native.sample(service.validate({**SLOW, "key": 7}), lambda _: None)
    assert [held for passed in holding.passed for held in passed] == [6, 7]


def test_a_guest_whose_request_fails_in_the_engine_hears_why_and_is_admitted_again(service, tmp_path):
    """A request whose pass fails stops the engine: its connection hears the failure and nothing after it,
    and the guest is admitted again."""
    from dew.inference.images import ImageScoring

    class Failing(ImageScoring):
        def forward(self, variables, rows):
            raise RuntimeError("an injected pass failure")

    def failing(scoring):
        return Failing(**{field.name: getattr(scoring, field.name) for field in dataclasses.fields(scoring)})

    _, _, native = live_kernel(service, tmp_path, served=failing)
    path = tmp_path / "model.sock"
    with serving(service, native, path):
        failed, lines = connected(path, SLOW)
        with failed:
            assert reply(lines)["error"]["name"] == "RuntimeError"
            assert lines.read() == ""
        described, lines = connected(path, DESCRIBE)
        with described:
            assert "result" in reply(lines)
