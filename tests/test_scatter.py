"""dew.nn.scatter: an index past its axis, its size or `DROPPED`, is dropped by a scatter with `mode='drop'`,
the CUDA lane's deterministic scatter's too."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.scatter import DROPPED


@pytest.mark.parametrize("dropped", [4, DROPPED], ids=["axis-size", "dropped"])
def test_an_element_scatter_of_rows_by_columns_drops_a_column_past_its_axis(dropped):
    """A `[rows, columns]` scatter of one value an element with `mode='drop'` drops a column at its axis
    size or at `DROPPED`: row 0's dropped write leaves row 1's 3 in its slot 0, a set's and an add's alike.
    XLA's deterministic GPU scatter (`--xla_gpu_deterministic_ops`, the CUDA lane's) wrote a column at its
    axis size onto the next row's slot 0 on jaxlib 0.11.2 (openxla/xla#49380, ledger 951); its fix
    (openxla/xla#49498) is in the XLA jaxlib 0.12.0.1 builds from (PC-951-scatter-redgreen-a100)."""
    rows, columns = jnp.arange(2)[:, None], jnp.asarray([[0, dropped]] * 2)
    values = jnp.asarray([[1, 2], [3, 4]], jnp.int32)
    buffer = jnp.zeros((2, 4), jnp.int32)
    written = jax.jit(lambda buffer: buffer.at[rows, columns].set(values, mode="drop"))(buffer)
    added = jax.jit(lambda buffer: buffer.at[rows, columns].add(values, mode="drop"))(buffer)
    for result in (written, added):
        np.testing.assert_array_equal(np.asarray(result), [[1, 0, 0, 0], [3, 0, 0, 0]])
