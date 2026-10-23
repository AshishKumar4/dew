"""A batch's rows split over the same mesh axes where the data places them and where a model's
layout places them: `dew.mesh` owns which axes split a batch's rows, and `dew.nn.sharding`'s
rule table composes its entries, under Dew's default rules, a layout's own data axes and a table
of the user's own."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.distributed import Layout, MeshSpec, shard_batch
from dew.nn.sharding import RESIDUAL, constrain

pytestmark = pytest.mark.mesh(devices=8)

ROWS, LENGTH, WIDTH = 16, 4, 8


def _rows_per_device(array: jax.Array) -> dict[int, tuple[int, int]]:
    """The rows each device holds of `array`, by the device's id."""
    return {device.id: (index[0].start or 0, index[0].stop or array.shape[0])
            for device, index in array.sharding.devices_indices_map(array.shape).items()}


@pytest.mark.parametrize(("rules", "spec"), [
    (None, MeshSpec(fsdp=2, tensor=2)),
    (None, MeshSpec(expert=2, sequence=2, tensor=2)),
    ("data axes", MeshSpec(fsdp=2, tensor=2)),
    ("own table", MeshSpec(fsdp=2, tensor=2)),
])
def test_the_data_and_the_models_layout_split_a_batchs_rows_over_the_same_axes(rules, spec):
    """A batch the trainer places (`shard_batch`, by the data's shares, `DataPartition.of`) holds
    on each device the rows a model's activation constrained by the layout's rules
    (`nn.sharding.constrain`, `logical_spec`) holds there: under Dew's rules, under a layout that
    splits rows over the fsdp axis alone (`Layout.data_axes`), and under a table of the user's
    own that splits them over data and leaves fsdp to the weights."""
    mesh = spec.build(jax.devices()[:8])
    table = {None: Layout().axis_rules, "data axes": Layout(data_axes=("fsdp",)).axis_rules,
             "own table": (("activation_batch", "data"), ("embed", "fsdp"))}[rules]
    tokens = np.arange(ROWS * LENGTH * WIDTH, dtype=np.float32).reshape(ROWS, LENGTH, WIDTH)
    with jax.set_mesh(mesh), nn.logical_axis_rules(table):
        placed = shard_batch(mesh, {"x": tokens})["x"]
        constrained = jax.jit(lambda x: constrain(x, RESIDUAL))(jnp.asarray(tokens))
    assert _rows_per_device(placed) == _rows_per_device(constrained)
    assert len(set(_rows_per_device(placed).values())) > 1
