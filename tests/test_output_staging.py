"""How a consumer stages an output tile through shared memory in chunks
(`dew.nn.kernels.templates.stores.output_staging`)."""

import jax.numpy as jnp
import pytest

from dew.nn.kernels.templates import stores


def test_a_stated_chunk_stores_whole_tma_rows_in_each_output_dtype():
    """A launch's stated staging (`stores.output_staging`'s `staging`) holds chunks whose rows are whole
    16-byte TMA stores in each of its outputs' dtypes, a narrower one refused before it lowers: 4 bf16 columns
    store 8 bytes a row, as 2 fp32 columns do, where 8 bf16 and 4 fp32 store 16; a call of fp32 and bf16
    outputs takes a chunk whole in both; a packed fp4 output's 16 columns store 8 bytes, its 32 store 16
    (theme gemm-output-staging-4's review: a bf16 output stated in 4-column chunks reached its tensor map;
    this head's review: an fp4 output's chunk judged by its dtype's byte, not its half byte)."""
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.bfloat16,), (4, 1))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32, jnp.bfloat16), (4, 2))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32,), (2, 1))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float4_e2m1fn,), (16, 1))
    assert stores.output_staging(128, (jnp.bfloat16,), (8, 1)) == (8, 1)
    assert stores.output_staging(128, (jnp.float32, jnp.bfloat16), (8, 2)) == (8, 2)
    assert stores.output_staging(128, (jnp.float4_e2m1fn,), (32, 1)) == (32, 1)


@pytest.mark.parametrize("columns, dtypes, staging, piece, holds", [
    (128, (jnp.bfloat16,), (4, 1), None, False), (128, (jnp.bfloat16,), (8, 1), None, True),
    (128, (jnp.float4_e2m1fn,), (16, 1), None, False), (128, (jnp.float4_e2m1fn,), (32, 1), None, True),
    (128, (jnp.float32,), (128, 1), 64, False), (128, (jnp.float32,), (64, 0), None, False),
    (96, (jnp.float32,), None, None, True), (128, (jnp.float32,), (0, 1), None, False),
    (128, (jnp.float32,), (-8, 1), None, False),
], ids=["bf16-4", "bf16-8", "fp4-16", "fp4-32", "past-a-piece", "no-buffer", "derived", "no-column",
        "negative"])
def test_a_candidate_asks_whether_a_staging_holds_with_no_exception(columns, dtypes, staging, piece, holds):
    """A launch's candidate asks whether a staging holds before it pins it (`stores.staging_refusal`, None
    where it does, its reason where it does not, raising nothing): chunks of whole 16-byte TMA rows in each
    dtype, by its bits; a chunk within the piece its issuer emits; a buffer at least (the GEMM
    output-staging head's review: GEMM's candidates held a copy of the rule to ask it)."""
    assert (stores.staging_refusal(columns, dtypes, staging, piece=piece) is None) == holds
