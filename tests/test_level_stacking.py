"""A pretrained decoder's own feed-forward blocks stepped in context (dew.objectives.lm.stacking, Nested
Learning's ad-hoc level stacking, arXiv 2512.24695 section 7.3): the chunked pass through the decode cache
on par with a literal loop that reruns every chunk's prefix whole, each token's blocks at its own window's
version, and the blocks at rate zero the decoder's own forward."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation
from dew.objectives.lm.stacking import stacked_scores

VOCAB, FEATURES, LENGTH, ROWS = 16, 16, 12, 4

STACKED = CausalTransformer(vocab_size=VOCAB, emb_features=FEATURES, num_layers=2, num_heads=2,
                            head_dim=16, mlp_features=16, max_seq_len=32, attention_impl="xla")
"""A two-layer test decoder, its feed-forward blocks the levels: its heads 16 wide, so a query or key
norm's gradient, one sum over every token the draw holds, has 16 elements a draw for check 1 to pool."""

TRUTH = STACKED.clone(dtype=jnp.float64, attention_impl="reference")
SCANNED = STACKED.clone(scan_layers=True)
"""`STACKED` scanning its run of layers, which stores them as its own layers, one subtree a layer."""
"""The decoder in float64 on the reference kernel: the loops' truth."""

STACKING_SEEDS = {"zero": 8, ((1,), (3,)): 8, ((0, 1), (3, 6)): 8, ((1, 0), (6, 3)): 8, "gradient": 512}
"""Check 1's declared seeds for the stacked sites, by their layers and windows; the gradient's sized
paired by `seeds_for` on armada (job 20261102200705-e3d60ec1), drawn by CI's map tasks
(`check1_shards`)."""


def draw(seed: int, scale: float = 0.25):
    """A seed's variables at `scale` N(0, 1) and `[ROWS, LENGTH + 1]` tokens."""
    rng = np.random.default_rng(seed)
    shapes = jax.eval_shape(lambda: STACKED.init(jax.random.key(0), jnp.zeros((1, LENGTH), jnp.int32)))
    variables = jax.tree.map(lambda leaf: (scale * rng.standard_normal(leaf.shape)).astype(np.float32),
                             shapes)
    return variables, rng.integers(0, VOCAB, (ROWS, LENGTH + 1)).astype(np.int32)


def token_losses(logits, targets, *, truth: bool):
    """Each target's next-token loss: its log-softmax, or with `truth` the float64 form check 1 holds
    both arms to (`test_chunk_adaptation.equation_71`'s)."""
    if truth:
        picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)
        others = jnp.where(jnp.arange(logits.shape[-1]) == targets[..., None], -jnp.inf, logits - picked)
        return jnp.log1p(jnp.sum(jnp.exp(others), axis=-1))
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)[..., 0]


def chunk_loop(model, variables, tokens, adaptation: ChunkAdaptation, *, truth: bool = False):
    """Section 7.3 read literally, chunk by chunk along each row, a chunk the least window: each chunk's
    tokens read every level at the version its window opened with; the chunk's prefix runs whole again,
    each earlier token's blocks at the versions its own chunk read (an interceptor applies each block at
    each version and keeps a token's own); a level's window gradient sums its chunks' loss gradients in
    it, the earlier chunks' versions fixed; at a window's end its level steps by its rate times it."""
    layers = adaptation.layers
    windows, rates = adaptation.windows(len(layers)), adaptation.rates(len(layers))
    chunk = min(windows)
    starts = model.feedforward_levels(variables, layers)
    names = tuple(starts)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    length = targets.shape[1]

    def logits_at(history, row_tokens):
        which = jnp.arange(row_tokens.shape[0]) // chunk

        def versions(next_fun, args, kwargs, context):
            path = context.module.path
            if (context.method_name != "__call__" or len(path) < 2 or path[-1] != "mlp"
                    or path[-2] not in names):
                return next_fun(*args, **kwargs)
            alone = context.module.clone(parent=None)
            read = [alone.apply({"params": levels[path[-2]]}, *args, **kwargs) for levels in history]
            kept = read[0]
            for at, alternative in enumerate(read[1:], start=1):
                kept = jnp.where((which == at)[None, :, None], alternative, kept)
            return kept

        with nn.intercept_methods(versions):
            return model.apply(variables, row_tokens[None])[0]

    def row(row_inputs, row_targets):
        levels, history, losses = dict(starts), [], []
        partials = jax.tree.map(jnp.zeros_like, starts)
        for index in range(-(-length // chunk)):
            for at, name in enumerate(names):
                if index and index % (windows[at] // chunk) == 0:
                    levels[name] = jax.tree.map(lambda leaf, grad, rate=rates[at]: leaf - rate * grad,
                                                levels[name], partials[name])
                    partials[name] = jax.tree.map(jnp.zeros_like, partials[name])
            span = slice(index * chunk, min((index + 1) * chunk, length))

            def chunk_loss(current, span=span):
                logits = logits_at([*history, current], row_inputs[:span.stop])[span]
                chunk_losses = token_losses(logits, row_targets[span], truth=truth)
                return jnp.sum(chunk_losses), chunk_losses

            (_, chunk_losses), gradient = jax.value_and_grad(chunk_loss, has_aux=True)(dict(levels))
            partials = jax.tree.map(jnp.add, partials, gradient)
            history.append(dict(levels))
            losses.append(chunk_losses)
        return jnp.concatenate(losses)

    return jax.vmap(row)(inputs, targets)


def stacked(adaptation: ChunkAdaptation, model=STACKED):
    """Dew's arm: the stacked pass's losses (`stacked_scores`) of `model` over a draw's tokens, every
    target weighed alike."""
    def losses(variables, tokens):
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        _, adapted = stacked_scores(model, variables, inputs, targets, jnp.ones(targets.shape), {},
                                    adaptation, predict=False, logits=False)
        return adapted.losses
    return losses


def test_a_decoders_own_blocks_at_rate_zero_score_as_its_forward():
    """At rate zero the stacked pass through the decode cache, chunk by chunk, scores each target as the
    decoder's own forward over the whole row does (check 1)."""
    adaptation = ChunkAdaptation(window=3, rate=0.0, layers=(0, 1))

    def plain(model, *, truth):
        return jax.jit(lambda variables, tokens: token_losses(model.apply(variables, tokens[:, :-1]),
                                                              tokens[:, 1:], truth=truth))

    assert_as_exact(jax.jit(stacked(adaptation)), plain(STACKED, truth=False), draw,
                    "the stacked pass at rate zero", seeds=STACKING_SEEDS["zero"],
                    truth_fn=plain(TRUTH, truth=True))


@pytest.mark.parametrize("model", [STACKED, SCANNED], ids=["layers", "scanned"])
@pytest.mark.parametrize(("layers", "windows"), [((1,), (3,)), ((0, 1), (3, 6)), ((1, 0), (6, 3))],
                         ids=["one-layer", "faster-first", "slower-first"])
def test_a_decoders_own_blocks_step_as_the_literal_chunk_loop(layers, windows, model):
    """A decoder's feed-forward blocks step in context at their own windows and rates, each target's loss
    on par with the literal chunk loop's (`chunk_loop`, check 1): one layer's block, and two at windows of
    three and six in either order; a stack of its own layers and one that scans them alike."""
    adaptation = ChunkAdaptation(window=windows, rate=(0.1, 0.05)[:len(layers)], layers=layers)

    def literal(reference, *, truth):
        return jax.jit(lambda variables, tokens: chunk_loop(reference, variables, tokens, adaptation,
                                                            truth=truth))

    assert_as_exact(jax.jit(stacked(adaptation, model)), literal(STACKED, truth=False), draw,
                    f"stacked blocks of layers {layers} at windows {windows}",
                    seeds=STACKING_SEEDS[layers, windows], truth_fn=literal(TRUTH, truth=True))


@pytest.mark.check1_shards(seeds=512, per_task=64)
def test_training_differentiates_through_the_blocks_steps():
    """The outer loss's gradient flows through every chunk's steps of the blocks to their starts, the
    meta-gradient continued pre-training trains (section 7.3): the gradient of the summed losses, second
    order through the steps, on par with the literal chunk loop's (check 1)."""
    adaptation = ChunkAdaptation(window=(3, 6), rate=(0.1, 0.05), layers=(0, 1))

    def gradient_draw(seed):
        variables, tokens = draw(seed)
        return variables, tokens, np.float32(1.0)

    def summed(arm):
        return with_gradients(lambda variables, tokens: jnp.sum(arm(variables, tokens)))

    def literal(model, *, truth):
        def losses(variables, tokens):
            return chunk_loop(model, variables, tokens, adaptation, truth=truth)
        return jax.jit(lambda variables, tokens, cotangent: summed(losses)(variables, tokens, cotangent)[:2])

    def dew(variables, tokens, cotangent):
        return summed(stacked(adaptation))(variables, tokens, cotangent)[:2]

    assert_as_exact(jax.jit(dew), literal(STACKED, truth=False), gradient_draw,
                    "the gradient through stacked blocks' steps", seeds=STACKING_SEEDS["gradient"],
                    truth_fn=literal(TRUTH, truth=True))


def test_a_run_continues_pretraining_through_its_blocks_steps():
    """An objective that steps a decoder's own blocks in context trains, its loss finite and its blocks'
    starts moved by the outer step (`LMObjective(adaptation=ChunkAdaptation(layers=...))`)."""
    import optax

    adaptation = ChunkAdaptation(window=(3, 6), rate=0.05, layers=(0, 1))
    objective = LMObjective(STACKED, LENGTH, adaptation=adaptation)
    variables = objective.init(jax.random.key(0))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (ROWS, LENGTH + 1)), jnp.int32)

    def loss(params):
        return jnp.mean(objective.token_scores(params, tokens).losses)

    value, gradient = jax.value_and_grad(loss)(variables)
    assert np.isfinite(float(value))
    blocks = gradient["params"]["layers_0"]["mlp"]
    assert any(float(jnp.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(blocks))
    updates, _ = optax.sgd(0.1).update(gradient, optax.sgd(0.1).init(variables))
    assert np.isfinite(float(loss(optax.apply_updates(variables, updates))))


def test_a_training_step_continues_pretraining_through_the_blocks_steps():
    """A training step's loss (`LMObjective.loss`, its summed head and z-loss) of a decoder stepping its
    own blocks in context is finite and moves the blocks' starts, as continued pre-training trains them."""
    from dew.objectives.base import Step

    objective = LMObjective(STACKED, LENGTH, adaptation=ChunkAdaptation(window=(3, 6), rate=0.05,
                                                                        layers=(0, 1)), z_loss=1e-4)
    variables = objective.init(jax.random.key(0))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (ROWS, LENGTH + 1)), jnp.int32)

    def loss(params):
        step = Step(jnp.asarray(0), jax.random.key(0), None)
        statistics, _ = objective.loss(params, {"text": tokens}, step)
        return statistics.mean()[0]

    value, gradient = jax.value_and_grad(loss)(variables)
    assert np.isfinite(float(value))
    blocks = gradient["params"]["layers_0"]["mlp"]
    assert any(float(jnp.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(blocks))


def padded_draw(seed: int):
    """A draw (`draw`) and each row's validity, its last few slots padding."""
    variables, tokens = draw(seed)
    lengths = np.random.default_rng(seed).integers(LENGTH // 2, LENGTH + 1, ROWS)
    valid = np.arange(LENGTH + 1)[None, :] < lengths[:, None] + 1
    return variables, tokens, jnp.asarray(valid)


def test_a_padded_rows_layout_reads_as_its_coordinates_say():
    """At rate zero, a stacked pass over rows whose last slots are padding, given as `Coordinates`, scores
    each real target as the decoder's own forward over those coordinates does, each padding target
    weighing nothing (check 1)."""
    from dew.batch import Coordinates

    adaptation = ChunkAdaptation(window=3, rate=0.0, layers=(0, 1))
    objective = LMObjective(STACKED, LENGTH, adaptation=adaptation)
    plain = LMObjective(STACKED, LENGTH)

    def scored(scorer):
        def losses(variables, tokens, valid):
            scores = scorer.token_scores(variables, tokens, coordinates=Coordinates(valid=valid))
            return scores.losses * scores.weights
        return jax.jit(losses)

    def truth(variables, tokens, valid):
        logits = TRUTH.apply(variables, tokens[:, :-1], coordinates=Coordinates(valid=valid[:, :-1]))
        return token_losses(logits, tokens[:, 1:], truth=True) * valid[:, 1:]

    assert_as_exact(scored(objective), scored(plain), padded_draw, "a padded stacked pass at rate zero",
                    seeds=STACKING_SEEDS["zero"], truth_fn=jax.jit(truth))


def test_a_stacked_row_refuses_what_its_cache_cannot_read():
    """A stacked pass refuses a row of several packed documents (the decode cache keeps no document ids,
    the decoder's own refusal) and a row whose attention reads its blocks whole (a prefix run's), which a
    block crossing a chunk cannot (ValueError)."""
    adaptation = ChunkAdaptation(window=3, rate=0.1, layers=(0,))
    variables, tokens = draw(0)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    documents = jnp.asarray(np.repeat([[1] * (LENGTH // 2) + [2] * (LENGTH - LENGTH // 2)], ROWS, axis=0))
    for fields in ({"segment_ids": documents}, {"block_groups": documents}):
        with pytest.raises(ValueError):
            stacked_scores(STACKED, variables, inputs, targets, jnp.ones(targets.shape), fields, adaptation,
                           predict=False, logits=False)
