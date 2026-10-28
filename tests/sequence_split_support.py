"""The check of a recurrent mixer split over the sequence axis against the same mixer on one device:
the rows the check draws (real tokens, padding, packed documents, an empty shard) and the check itself,
check 1 from float64 of the output and every gradient (`assert_split_as_whole`)."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import as_given, assert_as_exact, widened

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.training import MeshSpec

SPLIT_ROWS = {
    # Every slot real, as a pool of processes fills validity where no producer padded.
    "all_valid": (np.ones(32, bool), None),
    # Right-padded: real tokens first, the tail padding inside the last two shards.
    "right_padded": (np.arange(32) < 21, None),
    # Left-padded: a prompt's padding in front, across the first shard and into the second.
    "left_padded": (np.arange(32) >= 11, None),
    # Gaps inside a document, one at a shard's cut, and a shard holding a single real token, fewer
    # than the conv's three-token history: its next shard reads the ones before it too.
    "gaps": (~np.isin(np.arange(32), [2, 7, 8, 13, 16, 17, 18, 19, 20, 21, 22, 25]), None),
    # Packed documents with gaps between and inside them, one starting on a shard's first real token.
    "packed": (~np.isin(np.arange(32), [5, 6, 14, 15, 30, 31]),
               np.repeat([1, 1, 2, 3], [5, 11, 8, 8])),
    # A whole interior shard of padding with real tokens on both sides: the history and the state
    # cross it unchanged.
    "empty_shard": (~((np.arange(32) >= 8) & (np.arange(32) < 16)), None),
    # The same with one document continuing across the empty shard and another starting after it.
    "empty_shard_packed": (~((np.arange(32) >= 8) & (np.arange(32) < 16)),
                           np.repeat([1, 2], [20, 12])),
}
"""Each row of the sequence split's check: its real tokens and its documents (None for one)."""

SPLIT_BATCH = 8
"""The rows a sequence-split draw holds: every parameter's gradient, the heads' `D` and `A_log` too,
sums over all of them."""



def assert_split_as_whole(module, variables, row: str, seeds: int, site: str, *, width: int, shards: int = 4):
    """Under a sequence axis of `shards`, `module` with `variables` runs the rows `SPLIT_ROWS[row]` as it
    runs them on one device: a padded slot neither reads nor advances the conv or the state, each real
    token's conv reading the real tokens before it in its document across gaps and shards, and padded
    slots emit zero. Output and input gradient are check 1's against the unsplit layer, from float64, over
    `seeds` draws of hidden states `width` wide, and so are the parameters' gradients."""
    valid, segments = SPLIT_ROWS[row]
    fields = {"valid": jnp.asarray(np.broadcast_to(valid, (SPLIT_BATCH, 32)))}
    if segments is not None:
        fields["segments"] = jnp.asarray(np.broadcast_to(segments, (SPLIT_BATCH, 32)), jnp.int32)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(**fields)))

    def draw(seed):
        rng = np.random.default_rng([0x5E9, seed])
        return (rng.standard_normal((SPLIT_BATCH, 32, width)).astype(np.float32),
                rng.standard_normal((SPLIT_BATCH, 32, width)).astype(np.float32))

    def run(model, params, mesh=None):
        def pulled(hidden, cotangent):
            out, pull = jax.vjp(lambda params, hidden: model.apply(params, hidden, call), params, hidden)
            return out, *pull(cotangent)
        if mesh is None:
            return jax.jit(pulled)
        @as_given
        def split(hidden, cotangent):
            with jax.set_mesh(mesh):
                return jax.jit(pulled)(hidden, cotangent)
        return split

    @as_given
    def truth(hidden, cotangent):
        with jax.enable_x64(new_val=True):
            return run(module.clone(dtype=jnp.float64), widened(variables))(*widened((hidden, cotangent)))

    mesh = MeshSpec(sequence=shards).build(jax.devices()[:shards])
    assert_as_exact(run(module, variables, mesh), run(module, variables), draw, site, seeds=seeds,
                    truth_fn=truth)
    split = run(module, variables, mesh)(*draw(0))[0]
    assert not np.any(np.asarray(split)[:, ~valid])
