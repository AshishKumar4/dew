"""The live sampler's progress reports (site/live/container/progress.py).

The landing page samples through `Reporting(pipe)`, which swaps the solver it
is handed for one that also sends each step's clean prediction to the host.
The page must show the images the plain pipeline, and so the Colab notebook,
makes: the same bits for the same seed, with the page's sampler and with the
task's own, and one report per solver step, then the decode.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_inference import make_run

from dew.sampling import DPMSolverMultistep, TextToImage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def progress():
    spec = importlib.util.spec_from_file_location(
        "live_progress", ROOT / "site" / "live" / "container" / "progress.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass resolves annotations through it
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run")
    make_run(directory)
    return TextToImage.from_run(str(directory))


@pytest.mark.parametrize("solver", [None, DPMSolverMultistep()], ids=["task-default", "dpm-solver"])
def test_reporting_samples_the_same_bits_and_reports_each_step(progress, pipe, monkeypatch, solver):
    reports = []
    monkeypatch.setattr(progress, "_show", lambda report, png=None: reports.append(report))
    steps = 4
    expected = pipe(["a lily"], steps=steps, key=0, solver=solver).host().images
    images = progress.Reporting(pipe)(["a lily"], steps=steps, key=0, solver=solver).host().images
    np.testing.assert_array_equal(images, expected)
    # A walk of `steps` points takes steps - 1 solver steps; the model's last call,
    # the clean prediction at the final point, runs with the decode.
    assert reports == [{"step": k, "steps": steps} for k in range(1, steps)] + [{"stage": "decode"}]
