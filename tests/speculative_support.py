"""A block drafter's replay seam (row 13317): a speculative block's verified stretch, each row keeping its own
count of it, replayed into the cache saved before the block (`DecodeOps.verify`) and recorded into the
drafter's windows (`DecodeOps.record`), as `strategies.block_step` settles a block; then the drafter's next
draft from it, against the same draft from a fresh prefill of each row's emitted history."""

import jax
import jax.numpy as jnp
import numpy as np
from arm_forms import dew_path
from reference_error import as_given, assert_as_exact, widened

from dew.nn.inputs import ModelInputs
from dew.sampling.text import decode_ops, prefill_state


def _draft(ops, state, first, fed):
    """The drafter's logits at each place of its next block from `state`, drafted from each row's `first`
    token, the token fed back at each place `fed`'s (`[rows, places]`), so no place's logits decide the
    next's input: each place's logits a continuous function of the state."""
    logits = []

    def choose(index, scores):
        logits.append(scores)
        return fed[:, min(index, fed.shape[1] - 1)]

    ops.draft(state, first, choose)
    return {f"draft {index}": scores for index, scores in enumerate(logits)}


def _arms(model, params, *, mask_on_host: bool):
    """The replayed draft and the fresh one (the module doc), each compiled once, over a draw's prompts
    `[rows, P]`, block stretch `[rows, B]`, each row's kept count of it, first token and fed tokens. With
    `mask_on_host`, the fresh prefill reads its rows' real tokens on the host, as a sparse attention's
    prefill does (DSpark's), and runs as given; otherwise it is compiled whole."""
    ops = decode_ops(model, params, 0, 0)

    @jax.jit
    def replayed(prompts, stretch, counts, first, fed):
        state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
        keep = jnp.arange(stretch.shape[1])[None, :] < counts[:, None]
        following, _, seen = ops.verify(state, stretch, keep)
        assert seen is not None, "a block drafter's verify returns the states its windows record"
        return _draft(ops, ops.record(following, seen, keep), first, fed)

    def history(prompts, stretch, counts):
        """Each row's history, the prompt and its kept tokens, left-padded to one width, and its real ones."""
        xp = np if mask_on_host else jnp
        joined = xp.concatenate([prompts, stretch], axis=1)
        source = xp.arange(joined.shape[1])[None, :] - (stretch.shape[1] - xp.asarray(counts)[:, None])
        valid = source >= 0
        return xp.where(valid, xp.take_along_axis(joined, xp.maximum(source, 0), axis=1), 0), valid

    def prefilled(ids, valid, first, fed):
        inputs = ModelInputs(jnp.asarray(ids, jnp.int32), {"attention_mask": jnp.asarray(valid)})
        return _draft(ops, prefill_state(model, params, inputs, ops)[0], first, fed)

    if not mask_on_host:
        return replayed, jax.jit(lambda prompts, stretch, counts, first, fed: prefilled(
            *history(prompts, stretch, counts), first, fed))
    drafted = jax.jit(lambda state, first, fed: _draft(ops, state, first, fed))

    @as_given
    def fresh(prompts, stretch, counts, first, fed):
        ids, valid = history(prompts, stretch, counts)
        inputs = ModelInputs(jnp.asarray(ids, jnp.int32), {"attention_mask": jnp.asarray(valid)})
        return drafted(prefill_state(model, params, inputs, ops)[0], jnp.asarray(first), jnp.asarray(fed))

    return replayed, fresh


def assert_replay_drafts_as_a_prefill(model, params, draw, label: str, *, seeds: int,
                                      mask_on_host: bool = False) -> None:
    """The drafter's next draft after a replayed block whose rows kept their own counts (`_arms`' replayed)
    on par with its draft after a fresh prefill of each row's history, from that prefill's float64 run
    (check 1, over `draw`'s prompts, stretches, counts and fed tokens): two of Dew's programs, held so and
    not bit for bit."""
    replayed, fresh = _arms(model, params, mask_on_host=mask_on_host)
    with jax.enable_x64(new_val=True):
        _, wide = _arms(model.clone(dtype=jnp.float64), widened(params), mask_on_host=mask_on_host)

    @as_given
    def truth(*inputs):
        with jax.enable_x64(new_val=True):
            return wide(*inputs)

    assert_as_exact(replayed, dew_path(fresh), draw, label, seeds=seeds, truth_fn=truth)


def block_draw(rows: np.ndarray, prompt: int, block: int, generator: np.random.Generator):
    """A block's draw from token `rows` `[2, >= prompt + 2 block + 1]`: the prompt, the block's stretch
    after it, the first row keeping all of it and the second fewer (from one, `generator`'s), and from each
    row's next token on, its first draft token and the tokens fed back after it."""
    counts = np.asarray([block, generator.integers(1, block)], np.int32)
    after = prompt + counts[:, None] + np.arange(block + 1)[None, :]
    following = np.take_along_axis(rows, after, axis=1).astype(np.int32)
    return (rows[:, :prompt].astype(np.int32), rows[:, prompt:prompt + block].astype(np.int32), counts,
            following[:, 0], following[:, 1:])
