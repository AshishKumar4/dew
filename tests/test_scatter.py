"""dew.nn.scatter: `DROPPED` is dropped by the scatter XLA's deterministic GPU scatter mislinearizes."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.scatter import DROPPED


def test_an_element_scatter_of_rows_by_columns_drops_a_column_at_dropped():
    """A `[rows, columns]` scatter of one value an element with `mode='drop'` drops a column at
    `DROPPED`: row 0's dropped write leaves row 1's 3 in its slot 0. On the CUDA lane's deterministic
    scatter (`--xla_gpu_deterministic_ops`) a column at its axis size instead lands on the next row's
    slot 0 and writes 2 there (openxla/xla#49380, ledger 951; measured on an A100 with jaxlib 0.11.2)."""
    scatter = jax.jit(lambda buffer, rows, columns, values: buffer.at[rows, columns].set(values, mode="drop"))
    written = scatter(jnp.zeros((2, 4), jnp.int32), jnp.arange(2)[:, None], jnp.asarray([[0, DROPPED]] * 2),
                      jnp.asarray([[1, 2], [3, 4]], jnp.int32))
    np.testing.assert_array_equal(np.asarray(written), [[1, 0, 0, 0], [3, 0, 0, 0]])
