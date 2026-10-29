"""Shared model requests keep untrusted inputs away from model memory."""
import ast
import contextlib
import sys
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
        yield
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "dew"]:
            del sys.modules[name]
        sys.modules.update(held)


def test_the_live_sampler_cell_loads_the_model_the_live_image_serves(service, monkeypatch):
    """The live editor's cell (site/src/data/sampler.py), run through the live kernel's stand-ins,
    asks the shared model process for the model it loaded from the image's pin
    (site/live/container/text-to-image), which serves that revision alone and refuses a page that
    asks for another: a cell left on an earlier revision fails every unedited Run."""
    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
    native = service.NativeModels.__new__(service.NativeModels)
    native.progress = progress
    pinned = (SITE / "live" / "container" / "text-to-image").read_text()
    native.repo, native.revision = pinned.strip().split("@")
    asked = []

    def shared(payload):
        assert payload["op"] == "describe", payload
        asked.append(payload)
        return native.describe(payload)

    monkeypatch.setattr(client, "request", shared)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    # The cell up to its pipeline: what follows samples, which the shared process does with the
    # pipeline the description names.
    cell = ast.parse((SITE / "src" / "data" / "sampler.py").read_text())
    loaded = next(index for index, statement in enumerate(cell.body)
                  if isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == "pipe")
    namespace = {}
    with stood_in(client):
        exec(compile(ast.Module(cell.body[:loaded + 1], type_ignores=[]), "sampler.py", "exec"), namespace)
    assert len(asked) == 1
    assert (namespace["pipe"].repo, namespace["pipe"].revision) == (native.repo, native.revision)


def test_a_live_cell_takes_dews_arguments_and_is_refused_only_what_the_service_does_not_serve(
        service, tmp_path, monkeypatch):
    """The live kernel's stand-ins take what Dew's own calls take, read from Dew (site/live/container/
    contract.py): a prepare without its key fails as Dew's does; a call naming no steps, solver or guidance
    takes the pipeline's own, the image the service returns the one Dew's call makes; `CFG` holds Dew's
    defaults; an argument the service does not serve is refused by name; and the service refuses a solver
    outside the contract's (ledger 1159)."""
    import numpy as np
    from test_inference import make_run

    from dew.inference import TextToImage

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    client.TERMS = load("contract", "site/live/container").written()
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress, native.cfg, native.solvers = np, progress, None, {}
    native.repo, native.revision = "model", "pinned"
    native.pipe = progress.Reporting(pipe)
    monkeypatch.setattr(client, "request",
                        lambda payload: native.sample(service.validate(payload), lambda _: None))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    with pytest.raises(TypeError, match="key"):
        pipe.prepare(["a lake"])
    with stood_in(client):
        from dew.sampling import CFG

        live = client.Pipeline("model", "pinned")
        with pytest.raises(TypeError, match="key"):
            live.prepare(["a lake"])
        images = live(live.prepare(["a lake"], key=3), key=5).host().images
        assert (CFG(2).interval, CFG(2).rescale) == ((0.0, 1.0), 0.0)
        with pytest.raises(ValueError, match="decode stays at Dew's default"):
            live(["a lake"], key=5, decode=False)
    np.testing.assert_array_equal(images, pipe(pipe.prepare(["a lake"], key=3), key=5).host().images)
    request = {"op": "sample", "repo": "model", "revision": "pinned", "prompt": "a lake", "negative": None,
               "prepare_key": 3, "prepare_steps": None, "key": 5, "steps": None,
               "solver": {"name": "Euler", "args": {}}}
    with pytest.raises(ValueError, match="displayed solvers"):
        service.validate(request)

