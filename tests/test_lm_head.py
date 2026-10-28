"""The LM objective's head as a program: which head a step fuses, the vocabulary contractions a
prediction and a training step trace, and their losses and gradients against the tiled head's
(check 1). The objective's losses themselves are tests/test_lm_objective.py's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.batch import TEXT_KEY
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training.memory import climb_to


def vocabulary_products(jaxpr, vocab: int) -> int:
    """The products in `jaxpr` and every program inside it that read or write a vocabulary-sized array,
    a cond counted by the branch with the most, the one a run takes at most (as test_model_flops'
    `jaxpr_products` counts): a product's platform choice is a cond of both."""
    count = 0
    for eqn in jaxpr.eqns:
        shapes = [getattr(var.aval, "shape", ()) for var in (*eqn.invars, *eqn.outvars)]
        count += eqn.primitive.name == "dot_general" and any(vocab in shape for shape in shapes)
        inner = [value if hasattr(value, "eqns") else value.jaxpr
                 for value in jax.tree.leaves(eqn.params, is_leaf=lambda leaf: hasattr(leaf, "eqns")
                                              or hasattr(getattr(leaf, "jaxpr", None), "eqns"))
                 if hasattr(value, "eqns") or hasattr(getattr(value, "jaxpr", None), "eqns")]
        counts = [vocabulary_products(program, vocab) for program in inner]
        count += max(counts, default=0) if eqn.primitive.name == "cond" else sum(counts)
    return count


def head_scored(vocab: int, **options):
    """A two-layer decoder whose untied head spans `vocab` columns, and two
    9-token rows of its ids, for the tests that count the head's products."""
    model = CausalTransformer(vocab_size=vocab, emb_features=24, num_layers=2, num_heads=2, mlp_features=32,
                              max_seq_len=8, tie_embeddings=False, qk_norm=False, attention_impl="reference",
                              **options)
    return model, jax.random.randint(jax.random.key(1), (2, 9), 0, vocab)


def test_a_head_that_is_no_matrix_is_never_fused():
    """A head past a nonlinear transform is no one matrix (`chunked.head_table`), so a training step
    tiles it unfused, its program says so once traced, and the fit ladder has no rung to unfuse."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)
    jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)
    assert not objective.program_key()[0].head_fused
    assert objective.unfuse_head() is None


def test_a_head_put_back_unfused_stays_so_whatever_decoder_was_traced_before():
    """A record's unfused rung, put back (`climb_to`) on a decoder the objective has not traced,
    holds once the decoder's head turns out to be one matrix: a head whose transform was traced, then
    the decoder replaced by one with a matrix head, as a run's substitution does."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)

    def trace():
        params = objective.init(jax.random.key(0))
        jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)

    trace()
    objective.substitute([model.clone(head_transform=None)])
    climb_to(objective, {"head_tile": [4, 16], "head_fused": False, "remat": None})
    trace()
    assert not objective.program_key()[0].head_fused and objective.unfuse_head() is None


def test_a_decoder_put_back_keeps_what_its_trace_found_of_its_head():
    """Each decoder a loss traced keeps what that trace found of its head, so a decoder put back
    after another was traced, as a half-rung probe that missed puts its decoder back, is still no
    fused head where its head is no matrix."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)
    params = objective.init(jax.random.key(0))

    def trace():
        jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)

    trace()
    objective.substitute([model.clone(remat="full")])
    trace()
    objective.substitute([model])
    assert not objective.program_key()[0].head_fused and objective.unfuse_head() is None


PREDICTION_SEEDS = 8
"""Check 1's declared seeds for a prediction against the tiled head's scores (`seeds_for`, paired)."""

HEAD_STEP_SEEDS = 16
"""Check 1's declared seeds for a training step's tiled head against the per-token tiled head's loss
and gradients (`seeds_for`, paired)."""


def test_a_prediction_applies_the_head_once_and_scores_the_logits_it_returns():
    """A teacher's or a student's prediction (`LMObjective.predict`) holds the
    whole logits, and scores its losses from them: its forward contracts the
    vocabulary once and its gradient three times, and its losses and logits
    are the tiled head's and the model's own. A padding id past the
    vocabulary as the last target picks no column, as the tiled head's, so
    every loss stays finite."""
    from dew.objectives.lm.chunked import model_logits
    vocab = 61
    model, tokens = head_scored(vocab)
    tokens = tokens.at[:, -1].set(vocab)
    objective = LMObjective(model, 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=False)

    def predicted(params):
        return objective.predict(params, {TEXT_KEY: tokens}, step, train=False)[2]

    def pulled(params):
        prediction = predicted(params)
        return jnp.sum(prediction.logits * 0.5) + jnp.sum(prediction.losses * prediction.weights)

    assert vocabulary_products(jax.make_jaxpr(predicted)(params).jaxpr, vocab) == 1
    assert vocabulary_products(jax.make_jaxpr(jax.grad(pulled))(params).jaxpr, vocab) == 3
    assert np.all(np.isfinite(predicted(params).losses))

    initialized = jax.jit(objective.init)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed + (1 << 20)), tokens.shape, 0, vocab)
        return initialized(jax.random.key(seed)), drawn.at[:, -1].set(vocab)

    def prediction_of(params, ids):
        prediction = objective.predict(params, {TEXT_KEY: ids}, step, train=False)[2]
        return prediction.losses, prediction.logits

    def scored(of):
        def tiled(params, ids):
            scores = of.token_scores(params, ids)
            return scores.losses, model_logits(of.model, params, scores.hidden)
        return tiled

    # Over drawn decoders and rows, the prediction's losses and logits on par with the tiled head's
    # losses and the model's own logits of its states (check 1).
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    with jax.default_matmul_precision("highest"):
        assert_as_exact(prediction_of, scored(objective), draw, "a prediction",
                        seeds=PREDICTION_SEEDS, truth_fn=scored(wide))


@pytest.mark.parametrize("tile_tokens", [4, 16])
def test_a_training_steps_tiled_head_contracts_the_vocabulary_three_times_and_matches_the_tiled_head(
        tile_tokens):
    """A training step whose head is tiled takes the head's gradient in the
    forward (`fused_head_loss`), a tile's tokens a chunk, all 16 in one where
    the tile holds them: beside blocks recomputed in the backward, the step
    contracts the whole vocabulary three times, the logits, the states'
    gradient and the head's, none again, and its loss and gradients are the
    per-token tiled head's, z-loss and padding included. Unfused
    (`unfuse_head`, the fit ladder's rung for a fused chunk that does not
    fit), the step takes the tiled head's gradient in its backward, a tile
    of the vocabulary at a time, to the same loss and gradients."""
    vocab = 61
    model, tokens = head_scored(vocab, remat="full")
    tokens = tokens.at[1, 6:].set(0)
    fused = LMObjective(model, 8, ema_decay=None, z_loss=1e-3, pad_id=0, head_tile=(tile_tokens, 16))
    unfused = LMObjective(model, 8, ema_decay=None, z_loss=1e-3, pad_id=0, head_tile=(tile_tokens, 16))
    assert unfused.unfuse_head() is not None and unfused.unfuse_head() is None
    params = fused.init(jax.random.key(0))

    def loss(objective, *, training):
        step = Step(jnp.asarray(0), jax.random.key(1), None, training=training)

        def scalar(params, ids):
            stats, _ = objective.loss(params, {TEXT_KEY: ids}, step)
            return objective.reduce_loss(stats)[0]
        return scalar

    def step_products(objective) -> int:
        program = jax.make_jaxpr(jax.grad(loss(objective, training=True)))(params, tokens)
        return vocabulary_products(program.jaxpr, vocab)

    assert step_products(fused) == 3
    # The tiled head contracts the vocabulary a tile of its columns at a time, so no product spans it.
    assert step_products(unfused) == 0
    initialized = jax.jit(fused.init)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed + (1 << 20)), tokens.shape, 0, vocab)
        return initialized(jax.random.key(seed)), drawn.at[1, 6:].set(0)

    def stepped(objective, *, training):
        return jax.value_and_grad(loss(objective, training=training))

    # Over drawn decoders and rows, each step's loss and gradients on par with the per-token tiled
    # head's (check 1).
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, ema_decay=None, z_loss=1e-3, pad_id=0,
                       head_tile=(tile_tokens, 16))
    with jax.default_matmul_precision("highest"):
        for objective, label in ((fused, "fused"), (unfused, "unfused")):
            assert_as_exact(stepped(objective, training=True), stepped(fused, training=False), draw,
                            f"a {label} head's step, tiles of {tile_tokens}", seeds=HEAD_STEP_SEEDS,
                            truth_fn=stepped(wide, training=False))
