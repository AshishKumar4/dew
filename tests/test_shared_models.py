"""Shared model requests keep untrusted inputs away from model memory."""
import ast
import sys
from pathlib import Path

import pytest
from tool_support import load, run_from

SITE = Path(__file__).resolve().parents[1] / "site"


@pytest.fixture(scope="module")
def service():
    with run_from("model_service", "site/live/container") as module:
        yield module


SERVED_SEEDS = 16
"""Check 1's declared seeds for the served images against the pipeline's own call (to size)."""


def test_image_transport_keeps_native_pixels_and_preparation_keys(service, tmp_path):
    """The service samples what the pipeline's own call samples for the request's preparation and
    sampling keys, on par with it over drawn keys from the same call in float64 (check 1): its
    progress callback is another program, so its last bits are its own. It reports each step and
    the decode, and hands back one image."""
    import jax
    import numpy as np
    from reference_error import assert_as_exact
    from test_inference import make_run

    from dew.sampling import CFG, Heun, TextToImage

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
    make_run(tmp_path)
    pipe = TextToImage.from_run(str(tmp_path))
    with jax.enable_x64(new_val=True):
        wide = TextToImage.from_run(str(tmp_path), dtype="float64", param_dtype="float64")
    native = service.NativeModels.__new__(service.NativeModels)
    native.np, native.progress, native.cfg = np, progress, CFG
    native.repo, native.revision = "model", "pinned"
    native.pipe = progress.Reporting(pipe)
    native.solvers = {"Heun": Heun}

    def request(prepare_key, key):
        return {"repo": "model", "revision": "pinned", "prompt": "a lake", "negative": "",
                "prepare_key": prepare_key, "prepare_steps": 4, "key": key, "steps": 4,
                "solver": {"name": "Heun", "args": {}},
                "guidance": {"scale": 2, "interval": [0, 1], "rescale": 0}}

    def draw(seed):
        return 3 + 2 * seed, 5 + 2 * seed

    def called(pipe, prepare_key, key):
        prepared = pipe.prepare(["a lake"], key=prepare_key, steps=4, unconditional="")
        return np.asarray(pipe(prepared, key=key, steps=4, solver=Heun(), guidance=CFG(2)).host().images)

    def truth(prepare_key, key):
        with jax.enable_x64(new_val=True):
            return called(wide, prepare_key, key)

    def served(prepare_key, key):
        return np.asarray(client.ImageResult(native.sample(request(prepare_key, key), lambda report: None))
                          .host().images)

    assert_as_exact(served, lambda *keys: called(pipe, *keys), draw, "served images", seeds=SERVED_SEEDS,
                    truth_fn=truth)
    reports = []
    result = client.ImageResult(native.sample(request(*draw(0)), reports.append))
    assert len(result.pil()) == 1
    assert [report["progress"] for report in reports] == [
        {"step": 1, "steps": 4}, {"step": 2, "steps": 4}, {"step": 3, "steps": 4}, {"stage": "decode"}]


def test_the_live_sampler_cell_loads_the_model_the_live_image_serves(service, monkeypatch):
    """The live editor's cell (site/src/data/sampler.py), run through the live kernel's stand-ins,
    asks the shared model process for the model it loaded from the image's pin
    (site/live/container/text-to-image), which serves that revision alone and refuses a page that
    asks for another: a cell left on an earlier revision fails every unedited Run."""
    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
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
    for name in ("dew", "dew.interop", "dew.sampling"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    client.install()
    # The cell up to its pipeline: what follows samples, which the shared process does with the
    # pipeline the description names.
    cell = ast.parse((SITE / "src" / "data" / "sampler.py").read_text())
    loaded = next(index for index, statement in enumerate(cell.body)
                  if isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == "pipe")
    namespace = {}
    exec(compile(ast.Module(cell.body[:loaded + 1], type_ignores=[]), "sampler.py", "exec"), namespace)
    assert len(asked) == 1
    assert (namespace["pipe"].repo, namespace["pipe"].revision) == (native.repo, native.revision)
