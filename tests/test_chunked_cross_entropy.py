"""The chunked language-model head: same numbers, no full logits tensor.

The loss this replaces built the whole `[tokens, vocab]` logits tensor, then
read it for the logsumexp, the target logit, the top-1 prediction and again
for the softmax gradient. These tests hold the replacement to the code it
replaced: the loss to 1e-5, both gradients to 1e-4, and the top-1 prediction
exactly, including the tie that decides which of two equal logits wins.

The mutation tests are the point of the file. Dropping the target term or one
chunk of the vocabulary leaves a loss that still looks like a loss, so each
one is fed through the real chunk loop and shown to fail the parity check.
The tie rule has its own two tests, which is what a weaker comparison breaks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy, vocabulary_chunks

CHUNKS = [1, 2, 4, 8]


def reference(hidden, head, targets, softcap=None):
    """The full-vocabulary path: one big logits tensor, optax's cross entropy."""
    logits = jnp.einsum('...d,dv->...v', hidden.astype(jnp.float32), head)
    if softcap is not None:
        cap = jnp.asarray(softcap, jnp.float32)
        logits = cap * jnp.tanh(logits / cap)
    return (optax.softmax_cross_entropy_with_integer_labels(logits, targets),
            jnp.argmax(logits, axis=-1))


def inputs(vocab=97, features=24, tokens=(4, 5), dtype=jnp.float32, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (jax.random.normal(keys[0], (*tokens, features), dtype),
            jax.random.normal(keys[1], (features, vocab), jnp.float32),
            jax.random.randint(keys[2], tokens, 0, vocab))


@pytest.mark.parametrize("chunks", CHUNKS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_loss_and_prediction_match_the_full_vocabulary_pass(chunks, dtype):
    hidden, head, targets = inputs(dtype=dtype)
    expected_losses, expected_top1 = reference(hidden, head, targets)

    losses, predicted = chunked_cross_entropy(hidden, head, targets, chunks)

    assert losses.shape == targets.shape and losses.dtype == jnp.float32
    assert jnp.abs(losses - expected_losses).max() <= 1e-5 * jnp.abs(expected_losses).max()
    assert jnp.array_equal(predicted, expected_top1)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_the_softcap_is_applied_inside_the_chunk(chunks):
    """Gemma's cap is elementwise, so capping a tile has to equal capping a row."""
    hidden, head, targets = inputs()
    expected_losses, expected_top1 = reference(hidden, head, targets, softcap=2.0)

    losses, predicted = chunked_cross_entropy(hidden, head, targets, chunks,
                                              softcap=2.0)

    assert jnp.abs(losses - expected_losses).max() <= 1e-5 * jnp.abs(expected_losses).max()
    assert jnp.array_equal(predicted, expected_top1)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_both_gradients_match_the_full_vocabulary_pass(chunks):
    hidden, head, targets = inputs()

    def full(states, matrix):
        return jnp.mean(reference(states, matrix, targets)[0])

    def chunked(states, matrix):
        return jnp.mean(chunked_cross_entropy(states, matrix, targets, chunks)[0])

    expected = jax.grad(full, argnums=(0, 1))(hidden, head)
    got = jax.grad(chunked, argnums=(0, 1))(hidden, head)

    for name, want, have in zip(("hidden", "head"), expected, got):
        largest = jnp.abs(want).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        assert jnp.abs(have - want).max() <= 1e-4 * largest, name


def test_a_tie_goes_to_the_lowest_column_across_chunk_boundaries():
    """Equal logits: `jnp.argmax` takes the first, and so must the chunk loop."""
    features, vocab = 8, 24
    hidden = jnp.ones((1, 3, features), jnp.float32)
    # Column 17 is the winner, columns 5 and 21 tie with it, and the three sit
    # in different chunks of four, so the running comparison decides.
    head = jnp.zeros((features, vocab), jnp.float32)
    head = head.at[:, 5].set(0.5).at[:, 17].set(0.5).at[:, 21].set(0.5)
    targets = jnp.asarray([[5, 17, 21]], jnp.int32)

    _, expected_top1 = reference(hidden, head, targets)
    assert int(expected_top1[0, 0]) == 5, "the reference argmax did not tie-break low"

    for chunks in CHUNKS + [6]:
        _, predicted = chunked_cross_entropy(hidden, head, targets, chunks)
        assert jnp.array_equal(predicted, expected_top1), chunks


def test_a_flat_head_predicts_the_first_column():
    """Every logit equal, the degenerate tie a zero-initialised head produces."""
    hidden, head, targets = inputs()
    flat = jnp.zeros_like(head)

    _, predicted = chunked_cross_entropy(hidden, flat, targets, 4)

    assert jnp.array_equal(predicted, jnp.zeros_like(targets))
    assert jnp.array_equal(predicted, reference(hidden, flat, targets)[1])


# --- the mutations, each one a loss that would still train ------------------

def mutating_chunk_terms(monkeypatch, mutate):
    """Run the real chunk loop with `mutate` applied to each tile's terms.

    `_chunk_terms` is what the loop reads a tile's logsumexp, target logit,
    best logit and column from, so a mutation there is a mutation of the loss
    that ships, not of a copy of it.
    """
    original = chunked._chunk_terms

    def mutated(hidden, head_chunk, targets, start, stop, softcap, precision):
        terms = original(hidden, head_chunk, targets, start, stop, softcap,
                         precision)
        return mutate(terms, start, stop)

    monkeypatch.setattr(chunked, "_chunk_terms", mutated)


def test_dropping_the_target_term_fails_the_parity_check(monkeypatch):
    hidden, head, targets = inputs()
    expected = reference(hidden, head, targets)[0]
    mutating_chunk_terms(
        monkeypatch,
        lambda terms, start, stop: (terms[0], jnp.zeros_like(terms[1]), *terms[2:]))

    losses, _ = chunked_cross_entropy(hidden, head, targets, 4)

    assert jnp.abs(losses - expected).max() > 1e-5 * jnp.abs(expected).max()


@pytest.mark.parametrize("dropped", [0, 3])
def test_dropping_one_chunk_fails_the_parity_check(monkeypatch, dropped):
    hidden, head, targets = inputs()
    expected = reference(hidden, head, targets)[0]
    skipped = vocabulary_chunks(head.shape[1], 4)[dropped]

    def skip(terms, start, stop):
        if (start, stop) != skipped:
            return terms
        # What the loop starts from: a tile that contributes no column.
        chunk_lse, picked, chunk_best, chunk_column = terms
        return (jnp.full_like(chunk_lse, -jnp.inf), jnp.zeros_like(picked),
                jnp.full_like(chunk_best, -jnp.inf), chunk_column)

    mutating_chunk_terms(monkeypatch, skip)

    losses, _ = chunked_cross_entropy(hidden, head, targets, 4)

    assert jnp.abs(losses - expected).max() > 1e-5 * jnp.abs(expected).max()


# --- against the real backbone ---------------------------------------------

def small_model(**overrides):
    config = dict(vocab_size=97, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16, dtype=jnp.bfloat16)
    return CausalTransformer(**{**config, **overrides})


@pytest.mark.parametrize("tie_embeddings", [True, False])
@pytest.mark.parametrize("chunks", [4, 8])
def test_bf16_states_from_the_backbone_score_as_the_logits_did(chunks, tie_embeddings):
    """Real bf16 hidden states and a real head, tied and untied."""
    model = small_model(tie_embeddings=tie_embeddings)
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, 12), 0, 97)

    logits = model.apply(variables, ids)
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    head = model.apply(variables, variables['params'],
                       method=CausalTransformer.head_weight)
    losses, predicted = chunked_cross_entropy(hidden, head, targets, chunks)

    assert hidden.dtype == jnp.bfloat16
    assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
    assert jnp.array_equal(predicted, jnp.argmax(logits, axis=-1))


def test_the_gradient_reaches_the_backbone_through_the_states_and_the_head():
    """Both paths a tied head has: the trunk and the embedding table.

    The model computes in bf16, and the gradient into the tied embedding is a
    scatter-add of bf16 products, which a GPU runs as atomics in no fixed
    order. Two correct paths can then differ by one bf16 ulp of the largest
    entry (measured: 2**-9 on 0.44, in some processes and not others). The
    bound is one bf16 ulp, 2**-8 relative, with a margin: a real divergence of
    the two paths is orders above it."""
    model = small_model()
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, 12), 0, 97)

    def full(params):
        logits = model.apply({'params': params}, ids)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    def chunked(params):
        hidden = model.apply({'params': params}, ids,
                             method=CausalTransformer.hidden_states)
        head = model.apply({'params': params}, params,
                           method=CausalTransformer.head_weight)
        return jnp.mean(chunked_cross_entropy(hidden, head, targets, 4)[0])

    expected = jax.grad(full)(variables['params'])
    got = jax.grad(chunked)(variables['params'])

    flat_expected = jax.tree_util.tree_flatten_with_path(expected)[0]
    flat_got = jax.tree_util.tree_leaves(got)
    assert len(flat_expected) == len(flat_got)
    for (path, want), have in zip(flat_expected, flat_got):
        largest = jnp.abs(want).max()
        name = jax.tree_util.keystr(path)
        assert largest > 0, f"{name} has a zero gradient, so nothing is checked"
        assert jnp.abs(have - want).max() <= 1e-2 * largest, name


# --- the chunk arithmetic itself -------------------------------------------

@pytest.mark.parametrize("vocab, chunks, expected", [
    (50304, 4, ((0, 12576), (12576, 25152), (25152, 37728), (37728, 50304))),
    (97, 4, ((0, 25), (25, 50), (50, 75), (75, 97))),
    (8, 1, ((0, 8),)),
])
def test_chunk_bounds_cover_the_vocabulary_once(vocab, chunks, expected):
    bounds = vocabulary_chunks(vocab, chunks)
    assert bounds == expected
    covered = [column for start, stop in bounds for column in range(start, stop)]
    assert covered == list(range(vocab))


@pytest.mark.parametrize("vocab, chunks, message", [
    (100, 0, "at least one chunk"),
    (4, 8, "empty tiles"),
    (9, 4, "do not split"),
])
def test_impossible_chunk_counts_are_refused(vocab, chunks, message):
    with pytest.raises(ValueError, match=message):
        vocabulary_chunks(vocab, chunks)


def test_mismatched_shapes_are_refused():
    hidden, head, targets = inputs()
    with pytest.raises(ValueError, match="wide"):
        chunked_cross_entropy(hidden[..., :-1], head, targets, 4)
    with pytest.raises(ValueError, match="targets"):
        chunked_cross_entropy(hidden, head, targets[:, :-1], 4)
