"""Shared model requests keep untrusted inputs away from model memory."""
import pytest
from tool_support import load, run_from


@pytest.fixture(scope="module")
def service():
    with run_from("model_service", "site/live/container") as module:
        yield module


def test_image_transport_keeps_native_pixels_and_preparation_keys(service, tmp_path):
    import numpy as np
    from test_inference import make_run

    from dew.sampling import CFG, Heun, TextToImage

    progress = load("progress", "site/live/container")
    client = load("model_client", "site/live/container")
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
