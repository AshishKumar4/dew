"""A delta-rule mixer split over the sequence axis, the gated delta net and Kimi delta attention: each shard
runs its slice from the memory it enters with, its summary's composition over the earlier shards
(`recurrence_chunk.entering_memory`), and its conv after the real tokens before it (`linear.sequence_conv`),
checked against the same layer on one device (row 1437)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, widened
from sequence_split_support import SPLIT_BATCH, SPLIT_ROWS, assert_split_as_whole

from dew.nn.kda import KimiDeltaAttention
from dew.nn.linear import GatedDeltaNet, held_conv1d, over_sequence, sequence_conv, sequence_halo
from dew.training import MeshSpec

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


CONV_SEEDS = 96
"""The split conv's declared draws, the count its bias gradient's excess was measured over (row 2169)."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", ["packed", "empty_shard_packed"])
def test_a_split_convs_bias_gradient_sums_its_shards_as_the_whole_conv_does(row):
    """A causal conv over a sequence split four ways (`linear.sequence_conv`), on packed rows: its output
    and every gradient on par with the unsplit conv's from float64 (check 1). Summed as the boundary sums a
    replicated weight's cotangent, four rounded partials added in float32, the bias gradient's error
    doubled the unsplit conv's (1.82x and 2.03x, row 2169); the shards' partials are added in one rounding
    (`linear._shards_summed`). What remains is each shard's own sum, XLA:CPU's sequential reduction over
    its rows where the unsplit conv's is one blocked reduction over all of them: the bias gradient is held
    within the larger of the unsplit conv's error and a library arm's (the gate's ruling on row 2169),
    which sums each shard's partial through XLA as the split does and adds the four in float64."""
    valid, segments = (jnp.asarray(np.broadcast_to(field, (SPLIT_BATCH, 32))) for field in SPLIT_ROWS[row])
    rng = np.random.default_rng(0x2169)
    taps, bias = (rng.standard_normal(shape).astype(np.float32) for shape in ((16, 4), (16,)))
    mesh = MeshSpec(sequence=4).build(jax.devices()[:4])

    def pulled(conv):
        def run(taps, bias, mixed, cotangent):
            out, pull = jax.vjp(conv, taps, bias, mixed)
            return out, *pull(cotangent)
        return jax.jit(run)

    def split_conv(taps, bias, mixed):
        return over_sequence(lambda mixed, valid, segments, taps, bias: sequence_conv(
            mixed, taps, bias, segments, valid), 4, (mixed, valid, segments), (taps, bias))

    def whole_conv(taps, bias, mixed):
        return held_conv1d(mixed, taps, valid, bias=bias, segments=segments)[0]

    split_run, whole, wide = pulled(split_conv), pulled(whole_conv), pulled(whole_conv)

    def shard_partial(mixed, valid, segments, cotangent, taps, bias):
        # The shard's own bias gradient as the split computes it (`sequence_conv`'s halo and conv), before
        # any sum across the shards: `[rows, 1, D]`, its rows alike, one slot a shard.
        halo = sequence_halo(mixed, valid, segments, taps.shape[1] - 1)
        _, pull = jax.vjp(lambda bias: held_conv1d(mixed, taps, valid, halo, bias=bias, segments=segments)[0],
                          bias)
        (partial,) = pull(cotangent)
        return jnp.broadcast_to(partial, (mixed.shape[0], 1, partial.shape[-1]))

    partials = jax.jit(lambda taps, bias, mixed, cotangent: over_sequence(
        shard_partial, 4, (mixed, valid, segments, cotangent), (taps, bias)))

    def draw(seed):
        generator = np.random.default_rng([0x2169, seed])
        return tuple(generator.standard_normal((SPLIT_BATCH, 32, 16)).astype(np.float32) for _ in range(2))

    @as_given
    def split(mixed, cotangent):
        with jax.set_mesh(mesh):
            return split_run(taps, bias, mixed, cotangent)

    @as_given
    def truth(mixed, cotangent):
        with jax.enable_x64(new_val=True):
            return wide(*widened((taps, bias, mixed, cotangent)))

    @as_given
    def library(mixed, cotangent):
        out, grad_taps, _, grad_mixed = whole(taps, bias, mixed, cotangent)
        with jax.set_mesh(mesh):
            shards = np.asarray(partials(taps, bias, mixed, cotangent), np.float64)[0]
        return out, grad_taps, np.sum(shards, axis=0).astype(np.float32), grad_mixed

    assert_as_exact(split, lambda mixed, cotangent: whole(taps, bias, mixed, cotangent), draw,
                    f"a conv split over the sequence, {row}", seeds=CONV_SEEDS, truth_fn=truth,
                    library_fn=library, library_row=2169, library_leaves=("[2]",))
