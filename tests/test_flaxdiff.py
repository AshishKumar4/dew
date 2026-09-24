"""dew.interop.flaxdiff: FlaxDiff 0.2's SimpleUDiT, its names and its config,
onto Dew's SimpleUDiT.

tools/flaxdiff_reference.py ran FlaxDiff's own SimpleUDiT at commit 3e3497e
(the code flaxdiff 0.2.8 shipped) on CPU at fp32 and wrote the fixture:
random weights under FlaxDiff's names, the inputs, the text mask a Dew
caller hands the model beside them, and FlaxDiff's output.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax.traverse_util import unflatten_dict

from dew import models
from dew.interop.flaxdiff import (
    flaxdiff_weights,
    fourier_table,
    read_checkpoint,
    simple_udit_fields,
    simple_udit_variables,
)
from dew.nn.dit import TextContext

FIXTURE = Path(__file__).parent / "fixtures" / "flaxdiff"

# The largest difference measured against FlaxDiff's output (of order 0.1)
# is 7.5e-9 on CPU and 1.5e-8 on an RTX 4080, at fp32. Each FlaxDiff choice
# the port makes moves the output by at least 5.6e-4 when it is dropped: the
# SiLU before ada_proj, the pooling over every position, and either other
# Fourier table.
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE / "reference.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    config = json.loads((FIXTURE / "config.json").read_text())
    weights = unflatten_dict({name.removeprefix("params/"): value for name, value in data.items()
                              if name.startswith("params/")}, sep="/")
    return config["model"], weights, data


def dew_output(model_config, weights, data):
    model = models.build("simple_udit", simple_udit_fields(model_config))
    variables = simple_udit_variables(weights, model_config, jax_version="0.5.3")
    with jax.default_matmul_precision("highest"):
        return np.asarray(model.apply(variables, data["x"], data["temb"],
                                      TextContext(data["text"], data["text_mask"])))


def test_simple_udit_computes_what_flaxdiff_computed(reference):
    """Names, config and Fourier table mapped, Dew's model gives FlaxDiff's output."""
    model_config, weights, data = reference
    assert np.max(np.abs(dew_output(model_config, weights, data) - data["output"])) < TOLERANCE


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

    streams = {False: drawn(False), True: drawn(True)}
    for version, partitionable in (("0.4.31", False), ("0.5.0", True), ("0.5.3", True), ("0.10.1", True)):
        np.testing.assert_array_equal(fourier_table(features, version), streams[partitionable])
    with jax.default_device(jax.devices("cpu")[0]):
        np.testing.assert_array_equal(fourier_table(features, "0.5.3"), data["fourier_table"])


def test_a_checkpoint_publishes_the_averaged_weights_of_its_last_state(reference, tmp_path):
    """FlaxDiff saved live and averaged weights for `state` and `best_state`;
    by default the loader takes the averaged ones of `state`."""
    model_config, weights, data = reference
    shifted = jax.tree.map(lambda leaf: leaf + 0.01, weights)
    tree = {"state": {"params": {"params": shifted}, "ema_params": {"params": weights},
                      "step": np.asarray(7)},
            "best_state": {"params": {"params": shifted}, "ema_params": {"params": shifted},
                           "step": np.asarray(5)},
            "best_loss": np.asarray(0.3)}
    step = tmp_path / "7"
    ocp.PyTreeCheckpointer().save((step / "default").resolve(), tree)

    restored = read_checkpoint(step)
    published = dew_output(model_config, flaxdiff_weights(restored), data)
    assert np.max(np.abs(published - data["output"])) < TOLERANCE
    for other in ({"ema": False}, {"best": True}):
        assert np.max(np.abs(dew_output(model_config, flaxdiff_weights(restored, **other), data)
                             - data["output"])) > 1e-3


def test_config_the_port_does_not_reproduce_is_refused(reference):
    model_config, _, _ = reference
    for name in ("use_hilbert", "learn_sigma"):
        with pytest.raises(ValueError, match=name):
            simple_udit_fields({**model_config, name: True})
    with pytest.raises(ValueError, match="unknown"):
        simple_udit_fields({**model_config, "attention_bias": True})
