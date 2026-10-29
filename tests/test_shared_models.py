"""Shared model requests keep untrusted inputs away from model memory."""
import ast
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tool_support import load, run_from

SITE = Path(__file__).resolve().parents[1] / "site"


@pytest.fixture(scope="module")
def service():
    with run_from("model_service", "site/live/container") as module:
        yield module


def test_image_transport_keeps_native_pixels_and_preparation_keys(service, tmp_path):
    import numpy as np
    from test_inference import make_run

    from dew.inference import TextToImage
    from dew.sampling import CFG, Heun

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
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
    from test_inference import make_run

    from dew.inference import TextToImage

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress = np, progress
    native.cfg, native.solvers = service_records()
    pinned = (SITE / "live" / "container" / "text-to-image").read_text()
    native.repo, native.revision = pinned.strip().split("@")
    native.pipe = progress.Reporting(pipe)
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
    native_namespace = {"dew": SimpleNamespace(pipeline=lambda *args, **kwargs: pipe)}
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


def service_records():
    """The service's guidance record and solvers, as NativeModels takes them from Dew."""
    from dew import sampling
    from dew.sampling import CFG

    return CFG, {name: getattr(sampling, name) for name in ("DPMSolverMultistep", "EulerAncestral", "Heun")}


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
    from test_inference import make_run

    from dew.inference import TextToImage
    from dew.sampling import CFG as CFGOfDew, DPMSolverMultistep as DPMOfDew, Heun as HeunOfDew

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress = np, progress
    native.cfg, native.solvers = service_records()
    native.repo, native.revision = "model", "pinned"
    native.pipe = progress.Reporting(pipe)
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

