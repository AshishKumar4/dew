"""FlaxDiff 0.2 DiTs, their names and configs, onto Dew's models.

tools/flaxdiff_reference.py ran the original SimpleUDiT (3e3497e) and
hybrid DiT (94d2d21) on CPU at fp32. The fixtures contain random weights,
fixed inputs and original outputs, with a text mask for Dew's caller.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax.traverse_util import unflatten_dict

import dew.nn.backbones  # noqa: F401  (registers the kind)
from dew.interop.flaxdiff import (
    fourier_table,
    hybrid_dit_fields,
    hybrid_dit_variables,
    read_checkpoint,
    simple_udit_fields,
    simple_udit_variables,
)
from dew.nn.dit import TextContext
from dew.registry import models

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


def test_hybrid_dit_computes_what_flaxdiff_computed(hybrid_reference):
    """S5, 2D fusion, zigzag order and attention compute the trained architecture."""
    # CPU fp32 max error is 1.12e-8. Project-then-pool in the reference and
    # pool-then-project here round their reductions differently; 1e-6 leaves
    # room for backend rounding on outputs of order 0.1. Restoring SiLU or
    # real-token pooling moves this fixture by 2.44e-4 or 5.93e-4 respectively.
    model_config, weights, data = hybrid_reference
    actual = dew_output(model_config, weights, data, "hybrid_dit")
    assert np.max(np.abs(actual - data["output"])) < TOLERANCE


def test_the_fourier_table_follows_the_jax_the_run_trained_under(reference):
    """jax 0.5.0 changed the stream FlaxDiff 0.2 drew its table from,
    jax.random.normal at PRNGKey(42) times 16. The table is that draw under
    the run's stream, on the backend that computes it: every backend draws
    the same bits, but the normal transform rounds its last bit its own way
    (a GPU's table differs from the CPU's by an ulp in one entry). So the
    table is held exactly to this backend's draw of each stream, and drawn
    on the CPU, which every lane keeps beside its accelerator, exactly to
    the table FlaxDiff's own code recorded on the CPU that wrote the
    fixture."""
    model_config, _, data = reference
    features = model_config["emb_features"]

    def drawn(partitionable):
        with jax.threefry_partitionable(partitionable):
            draw = jax.random.normal(jax.random.PRNGKey(42), (features // 2,), dtype=jnp.float32)
        return np.asarray(draw * 16)

    streams = {False: drawn(partitionable=False), True: drawn(partitionable=True)}
    for version, partitionable in (("0.4.31", False), ("0.5.0", True), ("0.5.3", True), ("0.10.1", True)):
        np.testing.assert_array_equal(fourier_table(features, version), streams[partitionable])
    with jax.default_device(jax.devices("cpu")[0]):
        np.testing.assert_array_equal(fourier_table(features, "0.5.3"), data["fourier_table"])


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


def test_config_the_port_does_not_reproduce_is_refused(reference):
    model_config, _, _ = reference
    for name in ("use_hilbert", "learn_sigma"):
        with pytest.raises(ValueError, match=name):
            simple_udit_fields({**model_config, name: True})
    with pytest.raises(ValueError, match="unknown"):
        simple_udit_fields({**model_config, "attention_bias": True})
