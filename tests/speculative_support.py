"""A block drafter's replay seam (row 13317): one speculative block run as `strategies.block_step` runs it,
its drafted candidates controlled so its rows keep their own counts of the block, the target's accepted prefix
replayed into the cache saved before the block and recorded into the drafter's windows as the block settles
itself (`DecodeOps.verify`, `DecodeOps.record`); then the drafter's next draft from it, against the same draft
from a fresh prefill of each row's emitted history."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from arm_forms import dew_path
from jax.experimental import checkify
from reference_error import as_given, assert_as_exact, widened

from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, Speculative, decoding, generate
from dew.sampling.decoding import StepState
from dew.sampling.strategies import block_step
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


def _controlled(ops, proposals):
    """`ops` whose drafter proposes `proposals` `[rows, B - 1]` as a block's candidates after the target's
    own first (each place's logits all but the proposal's -inf, drawn as the block draws them), its states
    and caches its own."""
    def draft(state, first, choose):
        def chosen(index, scores):
            if index < proposals.shape[1]:
                picked = jnp.arange(scores.shape[-1])[None, :] == proposals[:, index, None]
                scores = jnp.where(picked, jnp.zeros_like(scores), jnp.full_like(scores, -jnp.inf))
            return choose(index, scores)

        ops.draft(state, first, chosen)

    return dataclasses.replace(ops, draft=draft)


def _arms(model, params, *, mask_on_host: bool):
    """The block's next draft and the fresh one (the module doc), each compiled once, over a draw's prompts
    `[rows, P]`, the target's greedy walk after them `[rows, B + 2]`, each row's count of the block's slots
    it keeps (the first row all `B + 1`, the others fewer) and the tokens fed back to the next draft: the
    next draft's logits at each place, and the block's emitted tokens (-1 past a row's count), which the
    fresh arm takes as the walk's. With `mask_on_host`, the fresh prefill reads its rows' real tokens on
    the host, as a sparse attention's prefill does (DSpark's), and runs as given; otherwise it is compiled
    whole. Each arm's next token its own logits' greedy choice (`next`), which opens its next draft."""
    ops = decode_ops(model, params, 0, 0)

    def settled_draft(prompts, greedy, counts, fed):
        rows, width = prompts.shape
        block = greedy.shape[1] - 2
        # The drafted candidates the walk's own, but where a row's count stops it: there a token the walk's
        # is not, which the target rejects, replacing it with its own.
        depth = jnp.arange(1, block)[None, :]
        walked = greedy[:, 1:block]
        proposals = jnp.where(depth == counts[:, None] - 1, jnp.where(walked == 0, 1, 0), walked)
        state, real = prefill_state(model, params, ModelInputs(prompts), ops)
        start = StepState(tokens=jnp.concatenate([prompts, jnp.zeros((rows, block + 1), jnp.int32)], axis=1),
                          valid=jnp.concatenate([jnp.ones(prompts.shape, bool),
                                                 jnp.zeros((rows, block + 1), bool)], axis=1),
                          step=jnp.zeros(rows, jnp.int32), active=real,
                          keys=jax.random.split(jax.random.PRNGKey(0), rows), prompt_width=width,
                          budget=jnp.full(rows, block + 1, jnp.int32))
        policy = Sampling(temperature=0.0, strategy=Speculative(block=block))
        transforms, finishing, plan = policy.components()
        settled = block_step(state, start, jnp.zeros(rows, bool), plan, _controlled(ops, proposals),
                             decoding.chain(transforms), decoding.finish(finishing), block + 1)
        # The next token the settled block's own logits choose (greedy), which opens its next draft: a block
        # that kept stale logits opens it, and chooses it, apart from the fresh prefill's.
        first = jnp.argmax(settled.state.logits, axis=-1).astype(jnp.int32)
        return {**_draft(ops, settled.state, first, fed), "next": first,
                "emitted": jnp.where(settled.kept, settled.tokens, -1)}

    # A block's device checks discharged as a decode loop's are (`text._generate`) and raised on the host: run
    # as given, the gate tracing none of it, as the loop's caller runs it.
    checked = jax.jit(checkify.checkify(settled_draft, errors=checkify.user_checks))

    @as_given
    def replayed(prompts, greedy, counts, fed):
        error, leaves = checked(prompts, greedy, counts, fed)
        error.throw()
        return leaves

    def history(prompts, greedy, counts):
        """Each row's history, the prompt and its kept tokens, left-padded to one width, its real ones, and
        its emitted tokens as the block lays them."""
        xp = np if mask_on_host else jnp
        kept = greedy[:, :-1]
        joined = xp.concatenate([prompts, kept], axis=1)
        counts = xp.asarray(counts)
        source = xp.arange(joined.shape[1])[None, :] - (kept.shape[1] - counts[:, None])
        valid = source >= 0
        emitted = xp.where(xp.arange(kept.shape[1])[None, :] < counts[:, None], kept, -1)
        return xp.where(valid, xp.take_along_axis(joined, xp.maximum(source, 0), axis=1), 0), valid, emitted

    def prefilled(ids, valid, emitted, greedy, counts, fed):
        inputs = ModelInputs(jnp.asarray(ids, jnp.int32), {"attention_mask": jnp.asarray(valid)})
        state = prefill_state(model, params, inputs, ops)[0]
        first = jnp.argmax(state.logits, axis=-1).astype(jnp.int32)
        return {**_draft(ops, state, first, jnp.asarray(fed)), "next": first,
                "emitted": jnp.asarray(emitted)}

    if not mask_on_host:
        return replayed, jax.jit(lambda prompts, greedy, counts, fed: prefilled(
            *history(prompts, greedy, counts), greedy, counts, fed))

    @as_given
    def fresh(prompts, greedy, counts, fed):
        return prefilled(*history(prompts, greedy, counts), greedy, counts, fed)

    return replayed, fresh


def assert_replay_drafts_as_a_prefill(model, params, draw, label: str, *, seeds: int,
                                      mask_on_host: bool = False) -> None:
    """The drafter's next draft after one speculative block whose rows kept their own counts (`_arms`'
    replayed) on par with its draft after a fresh prefill of each row's history, from that prefill's float64
    run, and the block's emitted tokens the greedy walk's, each row's count of them (check 1, over `draw`'s
    prompts, counts and fed tokens): two of Dew's programs, held so and not bit for bit."""
    replayed, fresh = _arms(model, params, mask_on_host=mask_on_host)
    with jax.enable_x64(new_val=True):
        _, wide = _arms(model.clone(dtype=jnp.float64), widened(params), mask_on_host=mask_on_host)

    @as_given
    def truth(*inputs):
        with jax.enable_x64(new_val=True):
            return wide(*inputs)

    evaluated = assert_as_exact(replayed, dew_path(fresh), draw, label, seeds=seeds, truth_fn=truth,
                                returns=True)
    # The next token each arm's logits choose, the same one, read off the draws check 1 evaluated: check 1
    # holds integers by their distance.
    for seed, drawn in (evaluated or {}).items():
        dew = {jax.tree_util.keystr(path): leaf for path, leaf in drawn.dew}
        reference = dict(zip(sorted(dew), drawn.reference, strict=True))
        np.testing.assert_array_equal(dew["['next']"], reference["['next']"], err_msg=f"{label}, seed {seed}")


def greedy_walk(model, params):
    """The target's greedy walk of `steps` tokens after each of a draw's prompts (`generate` at zero
    temperature, a token a step), as a block's draw reads it (`block_draw`)."""
    def walked(prompts: np.ndarray, steps: int) -> np.ndarray:
        drawn = generate(model, params, ModelInputs(jnp.asarray(prompts)), steps, key=jax.random.key(0),
                         sampling=Sampling(temperature=0.0))
        return np.asarray(drawn.tokens)[:, prompts.shape[1]:].astype(np.int32)

    return walked


def block_draw(walk, rows: np.ndarray, prompt: int, block: int, generator: np.random.Generator):
    """A block's draw from token `rows` `[2, >= prompt + block]`: the prompt, the target's greedy walk of
    `block + 2` tokens after it (`walk`, `greedy_walk`'s), the first row keeping all `block + 1` of the
    block's slots and the second fewer (from two, `generator`'s), and the tokens fed back to the next draft,
    the rows' own after the prompt."""
    prompts = rows[:, :prompt].astype(np.int32)
    counts = np.asarray([block + 1, generator.integers(2, block + 1)], np.int32)
    return prompts, walk(prompts, block + 2), counts, rows[:, prompt:prompt + block].astype(np.int32)
