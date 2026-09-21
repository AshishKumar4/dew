"""The chunked language-model head: same numbers, no full logits tensor.

The loss this replaces built the whole `[tokens, vocab]` logits tensor, then
read it for the logsumexp, the target logit, the top-1 prediction and again
for the softmax gradient. These tests hold the replacement to the code it
replaced: the loss to 1e-5, both gradients to 1e-4, and the top-1 prediction
exactly, including the tie that decides which of two equal logits wins.

The mutation tests are the point of the file. Dropping the target term or one
chunk of the vocabulary leaves a loss that still looks like a loss, so each
one is fed through the real chunk loop and shown to fail the parity check.
The tie rule has its own two tests; a looser comparison would pass either winner.

The backward recomputes tiles rather than taping the logits, so the last
section runs it against that same full-vocabulary pass, to 1e-5 relative on
both gradients, on shapes whose last token tile and last column tile are both
short. The two reductions run in different orders, so nothing there is exact;
the one case whose head is bf16 rounds at the end and holds to 1e-4.
"""

import math

import jax
import jax.numpy as jnp
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

    losses, predicted, _ = chunked_cross_entropy(hidden, head, targets, chunks)

    assert losses.shape == targets.shape and losses.dtype == jnp.float32
    assert jnp.abs(losses - expected_losses).max() <= 1e-5 * jnp.abs(expected_losses).max()
    assert jnp.array_equal(predicted, expected_top1)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_the_softcap_is_applied_inside_the_chunk(chunks):
    """Gemma's cap is elementwise, so capping a tile has to equal capping a row."""
    hidden, head, targets = inputs()
    expected_losses, expected_top1 = reference(hidden, head, targets, softcap=2.0)

    losses, predicted, _ = chunked_cross_entropy(hidden, head, targets, chunks,
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
        _, predicted, _ = chunked_cross_entropy(hidden, head, targets, chunks)
        assert jnp.array_equal(predicted, expected_top1), chunks


def test_a_flat_head_predicts_the_first_column():
    """Every logit equal, the degenerate tie a zero-initialised head produces."""
    hidden, head, targets = inputs()
    flat = jnp.zeros_like(head)

    _, predicted, _ = chunked_cross_entropy(hidden, flat, targets, 4)

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

    losses, _, _ = chunked_cross_entropy(hidden, head, targets, 4)

    assert jnp.abs(losses - expected).max() > 1e-5 * jnp.abs(expected).max()


@pytest.mark.parametrize("dropped", [0, 3])
def test_dropping_one_chunk_fails_the_parity_check(monkeypatch, dropped):
    hidden, head, targets = inputs()
    expected = reference(hidden, head, targets)[0]
    skipped = vocabulary_chunks(head.shape[1], 4)[dropped]

    def skip(terms, start, stop):
        # The full-width chunks are one loop's iterations, so `start` is
        # traced and the tile is dropped by a select: what the loop starts
        # from, a tile that contributes no column.
        dropped_tile = jnp.asarray(start) == skipped[0]
        chunk_lse, picked, chunk_best, chunk_column = terms
        return (jnp.where(dropped_tile, -jnp.inf, chunk_lse),
                jnp.where(dropped_tile, 0.0, picked),
                jnp.where(dropped_tile, -jnp.inf, chunk_best), chunk_column)

    mutating_chunk_terms(monkeypatch, skip)

    losses, _, _ = chunked_cross_entropy(hidden, head, targets, 4)

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
    losses, predicted, _ = chunked_cross_entropy(hidden, head, targets, chunks)

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


# --- the bounded backward --------------------------------------------------

# 2,053 columns in 2,048-column tiles and 131 tokens in 128-token tiles both
# leave a short last tile, so a dropped ragged end shows up as a missing
# gradient rather than as a shape error.
RAGGED = (128, 2048)
RAGGED_VOCAB, RAGGED_TOKENS = 2053, 131


def oracle(hidden, head, targets, softcap=None, precision=None):
    """The full-vocabulary losses and log partitions, one big logits tensor."""
    logits = chunked.head_logits(hidden, head, softcap=softcap, precision=precision)
    log_z = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return log_z - picked, log_z


@pytest.mark.parametrize("loss_weight, z_weight", [(1., 0.), (0., 1.), (1., .7)])
@pytest.mark.parametrize("softcap", [None, 30.])
def test_the_bounded_backward_matches_the_full_vocabulary_pass(
        loss_weight, z_weight, softcap):
    """Both outputs' gradients, against the pass that holds all the logits."""
    hidden, head, targets = inputs(
        vocab=RAGGED_VOCAB, features=7, tokens=(1, RAGGED_TOKENS))
    precision = jax.lax.Precision.HIGHEST

    def full(states, matrix):
        losses, log_z = oracle(states, matrix, targets, softcap, precision)
        return jnp.sum(loss_weight * losses + z_weight * jnp.square(log_z))

    def tiled(states, matrix):
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=softcap, precision=precision,
            tile=RAGGED)
        return jnp.sum(loss_weight * losses + z_weight * jnp.square(log_z))

    want_loss, want = jax.value_and_grad(full, argnums=(0, 1))(hidden, head)
    got_loss, got = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(hidden, head)

    assert jnp.abs(got_loss - want_loss) <= 1e-5 * jnp.abs(want_loss)
    for name, expected, actual in zip(("hidden", "head"), want, got):
        largest = jnp.abs(expected).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        assert jnp.abs(actual - expected).max() <= 1e-5 * largest, name
    assert jnp.abs(got[1][:, RAGGED[1]:]).max() > 0, "the short column tile was dropped"
    assert jnp.abs(got[0][:, RAGGED[0]:]).max() > 0, "the short token tile was dropped"


def test_the_softcap_gradient_survives_saturated_logits():
    """Logits well past the cap: `tanh` still has to pass a gradient back."""
    hidden, head, targets = inputs(vocab=97, features=8, tokens=(1, 9))
    hidden, head = hidden * 8, head * 4

    def full(states, matrix, cap):
        losses, log_z = oracle(states, matrix, targets, cap)
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    def tiled(states, matrix, cap):
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=cap, tile=RAGGED)
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    expected = jax.grad(full, argnums=(0, 1, 2))(hidden, head, 30.)
    got = jax.grad(tiled, argnums=(0, 1, 2))(hidden, head, 30.)
    for name, want, have in zip(("hidden", "head", "softcap"), expected, got):
        largest = jnp.abs(want).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        assert jnp.abs(have - want).max() <= 1e-5 * largest, name


def test_an_integer_softcap_gives_the_same_gradients_as_a_float_one():
    hidden, head, targets = inputs(features=7, tokens=(1, 129))

    def loss(states, matrix, cap):
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=cap, tile=RAGGED)
        return jnp.mean(losses + 0.02 * jnp.square(log_z))

    gradient = jax.grad(loss, argnums=(0, 1))
    for whole, fractional in zip(gradient(hidden, head, 30),
                                 gradient(hidden, head, 30.), strict=True):
        assert jnp.array_equal(whole, fractional)


def test_a_target_outside_the_vocabulary_scores_the_partition_alone():
    """The weighting is the caller's, so an out-of-range id still returns, and
    its gradient is the partition's rather than a wrapped column's."""
    hidden, head, targets = inputs(vocab=97, tokens=(1, 3))
    targets = targets.at[0, 0].set(-1).at[0, 1].set(97)
    outside = jnp.asarray([[1., 1., 0.]])

    def weighted(index):
        return jax.grad(lambda states: jnp.sum(outside * chunked_cross_entropy(
            states, head, targets, 4, tile=RAGGED)[index]))(hidden)

    losses, _, log_z = chunked_cross_entropy(hidden, head, targets, 4, tile=RAGGED)

    assert jnp.array_equal(losses[0, :2], log_z[0, :2])
    from_partition = weighted(2)
    assert jnp.abs(from_partition).max() > 0
    assert jnp.abs(weighted(0) - from_partition).max() <= 1e-6 * jnp.abs(
        from_partition).max()


def test_a_zero_cotangent_leaves_both_gradients_at_zero():
    hidden, head, targets = inputs(vocab=97, tokens=(1, 3))
    _, pullback = jax.vjp(
        lambda states, matrix: chunked_cross_entropy(
            states, matrix, targets, 4, tile=RAGGED)[0], hidden, head)

    gradients = pullback(jnp.zeros(targets.shape, jnp.float32))

    assert all(jnp.array_equal(g, jnp.zeros_like(g)) for g in gradients)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_head_gradient_accumulates_every_token_tile_before_it_rounds(dtype):
    """257 tokens is three tiles; rounding per tile would lose the last ones.

    A bf16 head gradient rounds once at the end, which 1e-5 would not admit.
    """
    hidden, head, targets = inputs(vocab=17, features=7, tokens=(257,), dtype=dtype)
    head = head.astype(dtype)

    def full(states, matrix):
        return jnp.mean(oracle(states, matrix, targets)[0])

    def tiled(states, matrix):
        return jnp.mean(
            chunked_cross_entropy(states, matrix, targets, 4, tile=RAGGED)[0])

    expected = jax.grad(full, argnums=(0, 1))(hidden, head)
    got = jax.grad(tiled, argnums=(0, 1))(hidden, head)

    assert got[0].dtype == hidden.dtype and got[1].dtype == head.dtype == dtype
    for name, want, have in zip(("hidden", "head"), expected, got):
        want, have = want.astype(jnp.float32), have.astype(jnp.float32)
        largest = jnp.abs(want).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        assert jnp.abs(have - want).max() <= 1e-4 * largest, name


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_the_backbone_receives_both_outputs_gradients(tie_embeddings):
    model = small_model(dtype=jnp.float32, tie_embeddings=tie_embeddings,
                        final_logit_softcap=30.)
    ids = jax.random.randint(jax.random.PRNGKey(31), (1, 7), 0, 97)
    params = model.init(jax.random.PRNGKey(32), ids)["params"]
    targets = jnp.roll(ids, -1, axis=-1)

    def full(tree):
        logits = model.apply({"params": tree}, ids)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.mean(log_z - picked + 0.02 * jnp.square(log_z))

    def tiled(tree):
        states = model.apply({"params": tree}, ids,
                             method=CausalTransformer.hidden_states)
        matrix = model.apply({"params": tree}, tree,
                             method=CausalTransformer.head_weight)
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=30.)
        return jnp.mean(losses + 0.02 * jnp.square(log_z))

    expected, got = jax.grad(full)(params), jax.grad(tiled)(params)
    for want, have in zip(jax.tree.leaves(expected), jax.tree.leaves(got),
                          strict=True):
        assert jnp.abs(have - want).max() <= 1e-5 * jnp.abs(want).max()


def residual_elements(function, hidden, head):
    """How many numbers the public VJP of `function` carries to its backward."""
    _, shapes = jax.make_jaxpr(
        lambda states, matrix: jax.vjp(function, states, matrix),
        return_shape=True)(hidden, head)
    return sum(leaf.size for leaf in jax.tree.leaves(shapes[1]))


@pytest.mark.parametrize("tokens", [257, 515])
def test_the_backward_saves_the_inputs_and_a_few_numbers_per_token(tokens):
    hidden, head, targets = inputs(vocab=4099, features=7, tokens=(tokens,))

    def tiled(states, matrix):
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=30., tile=RAGGED)
        return jnp.sum(losses + 0.03 * jnp.square(log_z))

    def full(states, matrix):
        losses, log_z = oracle(states, matrix, targets, 30.)
        return jnp.sum(losses + 0.03 * jnp.square(log_z))

    budget = hidden.size + head.size + 8 * tokens + 128
    assert residual_elements(tiled, hidden, head) <= budget
    # The same count over the pass this replaces, which does tape the logits.
    assert residual_elements(full, hidden, head) > budget


def equations(graph):
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation
            yield from equations(equation.params)
    elif hasattr(graph, "jaxpr"):
        yield from equations(graph.jaxpr)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from equations(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from equations(value)


def test_no_matmul_is_wider_than_one_tile():
    tokens, features, vocab = 257, 7, 4099
    hidden, head, targets = inputs(vocab=vocab, features=features, tokens=(tokens,))

    def loss(states, matrix):
        losses, _, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, softcap=30., tile=RAGGED)
        return jnp.sum(losses + 0.03 * jnp.square(log_z))

    graph = jax.make_jaxpr(jax.value_and_grad(loss, argnums=(0, 1)))(hidden, head)
    shapes = [tuple(variable.aval.shape)
              for equation in equations(graph)
              if equation.primitive.name == "dot_general"
              for variable in equation.outvars]

    assert shapes, "the structural check must inspect actual head matmuls"
    # The whole head gradient fits inside one tile's budget at this shape, so
    # its exact shape is the one thing the budget cannot rule out.
    assert (features, vocab) not in shapes and (vocab, features) not in shapes, \
        "a tile's VJP formed the full head gradient"
    oversized = [shape for shape in shapes if math.prod(shape) > math.prod(RAGGED)]
    assert not oversized, f"matmuls larger than one {RAGGED} tile: {oversized}"
