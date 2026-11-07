"""A forward's three consumer roles finish their shared schedule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention, attention_mosaic
from generation_support import INTERPRETS
from ring_protocol import programs


def _forward(kept, *, interpret):
    query = jnp.ones((1, 192, 1, 64), jnp.bfloat16)
    key = jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    spans = attention.spans(192, 128, 1)
    tiles = attention_mosaic.Tiles(64, 64, 2, 3, family="wgmma")

    def forward(query, key):
        with generation.assumed("sm90"), tuning.pinned({attention_mosaic.MosaicForward: tiles}):
            return attention_mosaic.forward(query, key, key, spans, residuals=kept, interpret=interpret)

    return forward, query, key, tiles.consumers


@INTERPRETS
@pytest.mark.parametrize("kept", [attention.Kept.NONE, attention.Kept.SPLIT])
def test_three_consumers_finish_a_forward_and_write_every_query(kept):
    """Three query blocks share two streamed key tiles, with and without training residuals."""
    forward, query, key, _ = _forward(kept, interpret=True)
    result = jax.jit(forward)(query, key)
    output = result[0] if kept != attention.Kept.NONE else result
    np.testing.assert_array_equal(np.asarray(output), np.ones(query.shape))


@pytest.mark.parametrize("kept", [attention.Kept.NONE, attention.Kept.SPLIT])
def test_every_query_holder_completes_the_same_schedule_phases(kept):
    """The actual forward's consumer programs finish every shared barrier's phases."""
    forward, query, key, consumers = _forward(kept, interpret=False)
    threads, arrivals = programs(forward, query, key, threads=consumers + 1)
    for barrier, expected in arrivals.items():
        counts = [sum(event.kind == "arrive" and event.at[0] == barrier for event in thread)
                  for thread in threads]
        if expected == consumers and any(counts[:-1]) and counts[-1] == 0:
            assert len(set(counts[:-1])) == 1, (barrier, counts)
