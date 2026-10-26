"""FlaxDiff 0.2 DiTs, their names and configs, onto Dew's models.

tools/flaxdiff_reference.py ran the original SimpleUDiT (3e3497e) and
hybrid DiT (94d2d21) on CPU at fp32. The fixtures contain random weights,
fixed inputs and original outputs, with a text mask for Dew's caller.
"""

import json
from importlib import import_module
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax.traverse_util import unflatten_dict

from dew.interop.flaxdiff import (
    hybrid_dit_fields,
    hybrid_dit_variables,
    read_checkpoint,
    simple_udit_fields,
    simple_udit_variables,
)
from dew.nn.dit import TextContext
from dew.registry import models

import_module("dew.nn.backbones")  # registers the fixture kind


FIXTURE = Path(__file__).parent / "fixtures" / "flaxdiff"

# The largest difference measured against FlaxDiff's output (of order 0.1)
# is 7.5e-9 on CPU and 1.5e-8 on an RTX 4080, at fp32. Each FlaxDiff choice
# the port makes moves the output by at least 5.6e-4 when it is dropped: the
# SiLU before ada_proj, the pooling over every position, and either other
# Fourier table.
TOLERANCE = 1e-6


def read_reference(directory):
    with np.load(directory / "reference.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    config = json.loads((directory / "config.json").read_text())
    weights = unflatten_dict({name.removeprefix("params/"): value for name, value in data.items()
                              if name.startswith("params/")}, sep="/")
    return config["model"], weights, data


@pytest.fixture(scope="module")
def reference():
    return read_reference(FIXTURE)


@pytest.fixture(scope="module")
def hybrid_reference():
    return read_reference(FIXTURE / "hybrid_dit")


def dew_output(model_config, weights, data, architecture="simple_udit"):
    fields = hybrid_dit_fields if architecture == "hybrid_dit" else simple_udit_fields
    convert = hybrid_dit_variables if architecture == "hybrid_dit" else simple_udit_variables
    model = models.build(architecture, fields(model_config))
    variables = convert(weights, model_config, jax_version="0.5.3")
    with jax.default_matmul_precision("highest"):
        return np.asarray(model.apply(variables, data["x"], data["temb"],
                                      TextContext(data["text"], data["text_mask"])))


def test_simple_udit_computes_what_flaxdiff_computed(reference):
    """Names, config and Fourier table mapped, Dew's model gives FlaxDiff's output."""
    model_config, weights, data = reference
    assert np.max(np.abs(dew_output(model_config, weights, data) - data["output"])) < TOLERANCE


def test_a_hilbert_scan_simple_udit_computes_what_flaxdiff_computed():
    """FlaxDiff's `use_hilbert` SimpleUDiT, whose patches run along a Hilbert
    curve through `hilbert_projection`, is Dew's `scan_order="hilbert"`."""
    model_config, weights, data = read_reference(FIXTURE / "hilbert")
    assert model_config["use_hilbert"] and simple_udit_fields(model_config)["scan_order"] == "hilbert"
    assert np.max(np.abs(dew_output(model_config, weights, data) - data["output"])) < TOLERANCE


def test_hybrid_dit_computes_what_flaxdiff_computed(hybrid_reference):
    """S5, 2D fusion, zigzag order and attention compute the trained architecture."""
    # CPU fp32 max error is 1.12e-8. Project-then-pool in the reference and
    # pool-then-project here round their reductions differently; 1e-6 leaves
    # room for backend rounding on outputs of order 0.1. Restoring SiLU or
    # real-token pooling moves this fixture by 2.44e-4 or 5.93e-4 respectively.
    model_config, weights, data = hybrid_reference
    actual = dew_output(model_config, weights, data, "hybrid_dit")
    assert np.max(np.abs(actual - data["output"])) < TOLERANCE


def test_checkpoint_selects_last_or_best_and_live_or_averaged_weights(reference, tmp_path):
    """All four selections restore distinct saved weights through the production reader."""
    model_config, weights, data = reference
    live = jax.tree.map(lambda leaf: leaf + 0.01, weights)
    best_ema = jax.tree.map(lambda leaf: leaf + 0.02, weights)
    best_live = jax.tree.map(lambda leaf: leaf + 0.03, weights)
    tree = {"state": {"params": {"params": live}, "ema_params": {"params": weights},
                      "step": np.asarray(7)},
            "best_state": {"params": {"params": best_live}, "ema_params": {"params": best_ema},
                           "step": np.asarray(5)},
            "best_loss": np.asarray(0.3)}
    step = tmp_path / "7"
    ocp.PyTreeCheckpointer().save((step / "default").resolve(), tree)

    for options, expected in (({}, weights), ({"ema": False}, live),
                              ({"best": True}, best_ema),
                              ({"best": True, "ema": False}, best_live)):
        restored = read_checkpoint(step, **options)
        jax.tree.map(np.testing.assert_array_equal, restored, expected)
    published = dew_output(model_config, read_checkpoint(step), data)
    assert np.max(np.abs(published - data["output"])) < TOLERANCE


def test_a_run_imported_by_from_flaxdiff_samples_what_flaxdiffs_preview_did(tmp_path):
    """tools/flaxdiff_pipeline_reference.py ran FlaxDiff's own preview
    sampling (3e3497e) of a tiny SimpleUDiT run over the tiny SD pipeline's
    CLIP and VAE: `EulerAncestralSampler` on `KarrasVENoiseScheduler` with
    `KarrasPredictionTransform`, guidance 3, 200 steps of its own grid, then
    the decode and the clip, on Dew's random draws. `TextToImage.from_flaxdiff`
    reads the same checkpoint and the run config FlaxDiff logs, and its call
    holds tests/reference_error.py's rule against FlaxDiff's images: as the
    run sampled, its towers in bfloat16, and with the towers in float32 on
    both sides, where their rounding (5.2e-3 rms from float64 in FlaxDiff's
    own run, against 2.3e-6 without it) no longer hides a sampler
    difference. A failure with a large ratio after a change to how Dew
    draws its noise needs the tool rerun."""
    from dataclasses import replace

    from reference_error import assert_as_exact_as_the_reference

    from dew.inference import TextToImage
    from dew.interop.flaxdiff import _towers
    from tools.flaxdiff_pipeline_reference import KEY, PROMPTS, checkpoint, run_config, towers

    with np.load(Path(__file__).parent / "fixtures" / "flaxdiff_pipeline" / "reference.npz") as stored:
        recorded = {name: stored[name] for name in stored.files}
    meta = json.loads(recorded.pop("meta").tobytes())
    weights = unflatten_dict({name.removeprefix("params/"): value for name, value in recorded.items()
                              if name.startswith("params/")}, sep="/")
    clip, vae = towers(tmp_path)
    checkpoint(tmp_path / "1", weights)
    task = TextToImage.from_flaxdiff(tmp_path / "1", run_config(meta["input_config"], clip, vae),
                                     jax_version=meta["jax_version"])
    np.testing.assert_allclose(task.prepare(list(PROMPTS), key=KEY).noise, recorded["start"], rtol=1e-6)

    def sampled(task):
        return np.clip(np.asarray(task(list(PROMPTS), key=KEY).images), -1, 1)

    truth = recorded["fp64.images"]
    assert_as_exact_as_the_reference(sampled(task), recorded["as_run.images"], truth, "as run")
    (keyword,) = task.inputs.conditions
    condition, autoencoder = _towers(str(clip), "", str(vae), "float32")
    wide = replace(task, inputs=replace(task.inputs, conditions={keyword: condition}),
                   autoencoder=autoencoder,
                   variables={**task.variables, "encoders": {keyword: condition.encoder.params},
                              "autoencoder": autoencoder.params})
    assert_as_exact_as_the_reference(sampled(wide), recorded["fp32_towers.images"], truth, "float32 towers")
