"""A forward's three consumer roles finish their shared schedule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention, attention_mosaic
from generation_support import INTERPRETS


@INTERPRETS
@pytest.mark.parametrize("kept", [attention.Kept.NONE, attention.Kept.SPLIT])
def test_three_consumers_finish_a_forward_and_write_every_query(kept):
    """Three query blocks share two streamed key tiles, with and without training residuals."""
    query = jnp.ones((1, 192, 1, 64), jnp.bfloat16)
    key = jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    spans = attention.spans(192, 128, 1)
    tiles = attention_mosaic.Tiles(64, 64, 2, 3, family="wgmma")

    def forward(query, key):
        with generation.assumed("sm90"), tuning.pinned({attention_mosaic.MosaicForward: tiles}):
            return attention_mosaic.forward(query, key, key, spans, residuals=kept, interpret=True)

    result = jax.jit(forward)(query, key)
    output = result[0] if kept != attention.Kept.NONE else result
    np.testing.assert_array_equal(np.asarray(output), np.ones(query.shape))
