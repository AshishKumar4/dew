"""How a consumer stages an output tile through shared memory in chunks
(`dew.nn.kernels.templates.stores.output_staging`)."""

import jax.numpy as jnp
import pytest

from dew.nn.kernels.templates import stores


def test_a_stated_chunk_stores_whole_tma_rows_in_each_output_dtype():
    """A launch's stated staging (`stores.output_staging`'s `staging`) holds chunks whose rows are whole
    16-byte TMA stores in each of its outputs' dtypes, a narrower one refused before it lowers: 4 bf16 columns
    store 8 bytes a row, as 2 fp32 columns do, where 8 bf16 and 4 fp32 store 16; a call of fp32 and bf16
    outputs takes a chunk whole in both (theme gemm-output-staging-4's review: a bf16 output stated in
    4-column chunks reached its tensor map)."""
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.bfloat16,), (4, 1))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32, jnp.bfloat16), (4, 2))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32,), (2, 1))
    assert stores.output_staging(128, (jnp.bfloat16,), (8, 1)) == (8, 1)
    assert stores.output_staging(128, (jnp.float32, jnp.bfloat16), (8, 2)) == (8, 2)
