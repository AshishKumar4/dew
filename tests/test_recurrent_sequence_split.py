"""A delta-rule mixer split over the sequence axis, the gated delta net and Kimi delta attention: each shard
runs its slice from the memory it enters with, its summary's composition over the earlier shards
(`recurrence_chunk.entering_memory`), and its conv after the real tokens before it (`linear.sequence_conv`),
checked against the same layer on one device (row 1437)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sequence_split_support import assert_split_as_whole

from dew.nn.kda import KimiDeltaAttention
from dew.nn.linear import GatedDeltaNet

WIDTH = 32
"""The hidden width every mixer here reads and writes."""

MIXERS = {
    "gated delta net": GatedDeltaNet(emb_features=WIDTH, num_k_heads=2, num_v_heads=4, head_k_dim=8,
                                     head_v_dim=8, chunk_size=4, dtype=jnp.float32),
    "kimi delta attention": KimiDeltaAttention(emb_features=WIDTH, num_heads=2, head_dim=8, conv_kernel=4,
                                               lower_bound=-5.0, chunk_size=4),
}
"""Each mixer at a small geometry: the gated delta net's queries and keys on half its value heads, chunks of
four tokens so a shard's slice of eight holds two."""

SEEDS = {"all_valid": 64, "packed": 64, "right_padded": 64, "empty_shard_packed": 64}
"""Each row's declared draws."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SEEDS))
@pytest.mark.parametrize("name", sorted(MIXERS))
def test_a_delta_rule_split_over_the_sequence_runs_its_rows_as_whole_ones(name, row):
    """Under a sequence axis of 4, the mixer runs each row as it runs it on one device: real tokens, packed
    documents (one continuing across a cut, one starting on a shard's first real token), right padding and
    an interior shard of padding, which the memory crosses unchanged (`assert_split_as_whole`)."""
    module = MIXERS[name]
    hidden = jnp.zeros((1, 32, WIDTH), jnp.float32)
    variables = jax.tree.map(lambda leaf: np.asarray(leaf), module.init(jax.random.key(7), hidden))
    assert_split_as_whole(module, variables, row, SEEDS[row], f"{name} split over the sequence, {row}",
                          width=WIDTH)
