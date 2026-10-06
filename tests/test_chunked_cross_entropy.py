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

import contextlib
import math

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import rounded_operand, rounded_to
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy, vocabulary_chunks

CHUNKS = [1, 2, 4, 8]


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('tied', [False, True])
def test_vocabulary_bias_is_fp32_after_tied_or_untied_head_products(dtype, tied):
    """FP32 params keep their vocabulary bias outside the bf16 product."""
    model = CausalTransformer(vocab_size=97, emb_features=24, num_layers=1, num_heads=3,
                              mlp_features=32, max_seq_len=8, head_bias=True,
                              tie_embeddings=tied, dtype=dtype, qk_norm=False,
                              attention_impl='reference')
    ids = jnp.asarray([[2, 7, 3], [11, 4, 9]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    bias = jax.random.normal(jax.random.key(3), (97,), jnp.float32) / 17
    variables['params']['head_bias'] = bias
    with jax.default_matmul_precision('default'):
        logits = model.apply(variables, ids)
        states = model.apply(variables, ids, method='hidden_states')
        table = model.apply(variables, method='output_table')
        actual = chunked.head_logits(states, table.matrix, softcap=None, precision=None,
                                     vocab_major=table.vocab_major, bias=table.bias)
        np.testing.assert_array_equal(actual, logits)
        for tile in (None, (2, 11)):
            losses, predicted, _ = chunked_cross_entropy(states, table.matrix, ids, 4, tile=tile, bias=bias,
                                                         vocab_major=table.vocab_major)
            expected = optax.softmax_cross_entropy_with_integer_labels(logits, ids)
            assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
            np.testing.assert_array_equal(predicted, logits.argmax(-1))
    assert variables['params']['head_bias'].dtype == jnp.float32
    assert not np.array_equal(np.asarray(bias), np.asarray(bias.astype(jnp.bfloat16), np.float32))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('tile', [None, (3, 11)], ids=['whole', 'tiled'])
def test_vocabulary_bias_gradient_matches_full_logits_under_the_float64_rule(dtype, tile):
    from reference_error import assert_as_exact_as_the_reference

    from dew.nn.precision import head_product

    states, matrix, targets = inputs(vocab=37, features=16, tokens=(2, 5), dtype=dtype)
    bias = jax.random.normal(jax.random.key(3), (37,), jnp.float32) / 17
    with jax.default_matmul_precision('default'):
        def full(offset):
            logits = chunked.head_logits(states, matrix, softcap=None, precision=None, bias=offset)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        def bounded(offset):
            return chunked_cross_entropy(states, matrix, targets, 4, tile=tile, bias=offset)[0].mean()

        reference = jax.grad(full)(bias)
        actual = jax.grad(bounded)(bias)
        product = head_product('...d,dv->...v', states, matrix, None)
        with jax.enable_x64(new_val=True):
            # For bf16 the rounded product is exact input to the affine
            # head: this oracle isolates its fp32 bias and softmax gradient.
            if dtype == jnp.float32:
                product = jnp.einsum('...d,dv->...v', states.astype(jnp.float64),
                                     matrix.astype(jnp.float64), precision=jax.lax.Precision.HIGHEST)
            wide = product.astype(jnp.float64)

            def truth(offset):
                return optax.softmax_cross_entropy_with_integer_labels(wide + offset, targets).mean()

            exact = jax.grad(truth)(bias.astype(jnp.float64))
    assert_as_exact_as_the_reference(actual, reference, exact, f'{dtype} vocabulary bias gradient')


def test_vocabulary_bias_reference_catches_rounding_the_offset_into_the_product(monkeypatch):
    real = chunked._biased_logits
    monkeypatch.setattr(chunked, '_biased_logits', lambda logits, bias: real(
        logits, None if bias is None else bias.astype(jnp.bfloat16).astype(jnp.float32)))
    with pytest.raises(AssertionError):
        test_vocabulary_bias_is_fp32_after_tied_or_untied_head_products(jnp.bfloat16, tied=True)


@pytest.mark.mesh(devices=2)
def test_vocabulary_split_slices_the_bias_with_its_table_columns():
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import shard_batch

    model = CausalTransformer(vocab_size=128, emb_features=8, num_layers=1, num_heads=2,
                              mlp_features=16, max_seq_len=4, head_bias=True,
                              qk_norm=False, attention_impl='reference')
    tokens = jnp.arange(20, dtype=jnp.int32).reshape(4, 5)
    variables = model.init(jax.random.key(0), tokens[:, :-1])
    variables['params']['head_bias'] = variables['params']['head_bias'].at[117].set(13.027)
    objective = LMObjective(model, seq_len=4, variables=variables, head_chunks=4,
                            head_tile=(2, 11), ema_decay=None)
    expected = objective.token_scores(variables, tokens)
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    shardings = Layout(min_shard=1).shardings(mesh, variables)
    with jax.set_mesh(mesh):
        actual = jax.jit(objective.token_scores)(
            jax.device_put(variables, shardings), shard_batch(mesh, {'text': np.asarray(tokens)})['text'])
    assert jnp.abs(actual.losses - expected.losses).max() <= 1e-5 * jnp.abs(expected.losses).max()
    np.testing.assert_array_equal(actual.correct, expected.correct)


def reference(hidden, head, targets, softcap=None):
    """The full-vocabulary path: one big logits tensor, optax's cross entropy.
    bf16 states are bf16 compute, which multiplies the head rounded to bf16
    and rounds the logits to bf16, as torch autocast does."""
    bf16 = hidden.dtype == jnp.bfloat16
    if bf16:
        head = head.astype(jnp.bfloat16).astype(jnp.float32)
    logits = jnp.einsum('...d,dv->...v', hidden.astype(jnp.float32), head,
                        precision=jax.lax.Precision.HIGHEST)
    if bf16:
        logits = rounded_to(logits, jnp.bfloat16)
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


def eighths(*values):
    """`values` rounded to multiples of 1/8. Every product of two is a
    multiple of 1/64 and every sum of a few dozen is exact in fp32 in any
    order, so two products of them give the same logits, and the same bf16
    rounding of them, whatever order each sums in."""
    return tuple((jnp.round(value.astype(jnp.float32) * 8) / 8).astype(value.dtype) for value in values)


@pytest.mark.parametrize("chunks", CHUNKS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_loss_and_prediction_match_the_full_vocabulary_pass(chunks, dtype):
    hidden, head, targets = inputs(dtype=dtype)
    if dtype == jnp.bfloat16:
        hidden, head = eighths(hidden, head)
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

    for name, want, have in zip(("hidden", "head"), expected, got, strict=True):
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

    for chunks in [*CHUNKS, 6]:
        _, predicted, _ = chunked_cross_entropy(hidden, head, targets, chunks)
        assert jnp.array_equal(predicted, expected_top1), chunks


def test_a_flat_head_predicts_the_first_column():
    """Every logit equal, the degenerate tie a zero-initialised head produces."""
    hidden, head, targets = inputs()
    flat = jnp.zeros_like(head)

    _, predicted, _ = chunked_cross_entropy(hidden, flat, targets, 4)

    assert jnp.array_equal(predicted, jnp.zeros_like(targets))
    assert jnp.array_equal(predicted, reference(hidden, flat, targets)[1])


def test_the_prediction_is_the_int32_column_under_x64():
    """jax_enable_x64 widens argmax to int64, while the tile loop carries its
    best column as int32; the prediction is still the reference's argmax,
    as int32 columns."""
    with jax.enable_x64(new_val=True):
        hidden, head, targets = inputs()
        _, predicted, _ = chunked_cross_entropy(hidden, head, targets.astype(jnp.int32), 4)
        assert predicted.dtype == jnp.int32
        assert jnp.array_equal(predicted, reference(hidden, head, targets)[1])


def float32_values(jaxpr) -> int:
    """How many values of float32 `jaxpr`, and every jaxpr it holds, makes."""
    made = sum(any(getattr(getattr(var, "aval", None), "dtype", None) == jnp.float32
                   for var in eqn.outvars) for eqn in jaxpr.eqns)
    return made + sum(float32_values(inner) for eqn in jaxpr.eqns
                      for inner in jax.extend.core.jaxprs_in_params(eqn.params))


@pytest.mark.parametrize("tile", [(3, 16), None], ids=["tiled", "whole"])
def test_float64_states_and_head_compute_nothing_in_float32(tile):
    """A float64 reference's loss (tools/layout_parity.py `--anchor`, a
    float64 twin) is float64 forward and backward: the logits, log Z, the
    tile carries and both gradients, softcapped and predicting. Pinned to
    float32, the reference shared the float32 run's roundings of the head,
    so the run's distance from it understated the run's own rounding. It
    traces on the CPU backend, since a TPU has no float64."""
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        hidden, head, targets = jax.tree.map(
            lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
            inputs())

        def loss(hidden, head):
            losses, _, log_z = chunked_cross_entropy(hidden, head, targets.astype(jnp.int32), 4,
                                                     softcap=30.0, tile=tile)
            return jnp.sum(losses) + jnp.sum(log_z ** 2)

        jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, argnums=(0, 1)))(hidden, head)
    assert float32_values(jaxpr.jaxpr) == 0


# --- the mutations, each one a loss that would still train ------------------

def mutating_chunk_terms(monkeypatch, mutate):
    """Run the real chunk loop with `mutate` applied to each tile's terms.

    `_chunk_terms` is what the loop reads a tile's logsumexp, target logit,
    best logit and column from, so a mutation there is a mutation of the loss
    that ships, not of a copy of it.
    """
    original = chunked._chunk_terms

    def mutated(hidden, head_chunk, targets, start, stop, softcap, precision, predict, temperature,
                excluded, bias=None):
        terms = original(hidden, head_chunk, targets, start, stop, softcap,
                         precision, predict, temperature, excluded, bias)
        return mutate(terms, start, stop)

    monkeypatch.setattr(chunked, "_chunk_terms", mutated)


def test_dropping_the_target_term_fails_the_parity_check(monkeypatch):
    hidden, head, targets = inputs()
    expected = reference(hidden, head, targets)[0]
    mutating_chunk_terms(
        monkeypatch,
        lambda terms, start, stop: terms._replace(picked=jnp.zeros_like(terms.picked)))

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
        return terms._replace(lse=jnp.where(dropped_tile, -jnp.inf, terms.lse),
                              picked=jnp.where(dropped_tile, 0.0, terms.picked),
                              best=jnp.where(dropped_tile, -jnp.inf, terms.best))

    mutating_chunk_terms(monkeypatch, skip)

    losses, _, _ = chunked_cross_entropy(hidden, head, targets, 4)

    assert jnp.abs(losses - expected).max() > 1e-5 * jnp.abs(expected).max()


# --- against the real backbone ---------------------------------------------

def small_model(**overrides):
    config = {'vocab_size': 97, 'emb_features': 32, 'num_layers': 2, 'num_heads': 4,
                  'mlp_features': 64, 'max_seq_len': 16, 'dtype': jnp.bfloat16}
    return CausalTransformer(**{**config, **overrides})


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("tie_embeddings", [True, False])
@pytest.mark.parametrize("chunks", [4, 8])
def test_bf16_states_from_the_backbone_score_as_the_logits_did(chunks, tie_embeddings, dtype):
    """Real hidden states and a real head, tied and untied: the model's head
    multiplies as the chunked loss does under bf16 and fp32 compute."""
    model = small_model(tie_embeddings=tie_embeddings, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, 12), 0, 97)

    logits = model.apply(variables, ids)
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    table = model.apply(variables, method='output_table')
    losses, predicted, _ = chunked_cross_entropy(hidden, table.matrix, targets, chunks,
                                                 vocab_major=table.vocab_major)

    assert hidden.dtype == dtype
    assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
    assert jnp.array_equal(predicted, jnp.argmax(logits, axis=-1))


def test_the_gradient_reaches_the_backbone_through_the_states_and_the_head():
    """Both paths a tied head has: the trunk and the embedding table.

    The model computes in bf16, and the gradient into the tied embedding is a
    scatter-add of bf16 products, which a GPU runs as atomics in no fixed
    order. Two correct paths can then differ by one bf16 ulp of the largest
    entry (measured: 2**-9 on 0.44, in some processes and not others). The
    bound is one bf16 ulp, 2**-8 relative, with a margin: a real divergence of
    the two paths is orders above it.

    The reference takes the logits from the operands the head multiplies,
    rounded to bf16, at the highest precision, and rounds them to bf16 as the
    chunked head does, so its backward rounds the logits' cotangent to bf16
    once, as the chunked head's does.
    """
    model = small_model()
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, 12), 0, 97)

    def full(params):
        hidden = model.apply({'params': params}, ids,
                             method=CausalTransformer.hidden_states)
        table = model.apply({'params': params}, method='output_table')
        head = table.matrix.T if table.vocab_major else table.matrix
        operands = (rounded_operand(value.astype(jnp.float32), jnp.bfloat16)
                    for value in (hidden, head))
        logits = rounded_to(jnp.einsum('btd,dv->btv', *operands, precision=jax.lax.Precision.HIGHEST),
                            jnp.bfloat16)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    def chunked(params):
        hidden = model.apply({'params': params}, ids,
                             method=CausalTransformer.hidden_states)
        table = model.apply({'params': params}, method='output_table')
        return jnp.mean(chunked_cross_entropy(hidden, table.matrix, targets, 4,
                                              vocab_major=table.vocab_major)[0])

    expected = jax.grad(full)(variables['params'])
    got = jax.grad(chunked)(variables['params'])

    flat_expected = jax.tree_util.tree_flatten_with_path(expected)[0]
    flat_got = jax.tree_util.tree_leaves(got)
    assert len(flat_expected) == len(flat_got)
    for (path, want), have in zip(flat_expected, flat_got, strict=True):
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
    for name, expected, actual in zip(("hidden", "head"), want, got, strict=True):
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
    for name, want, have in zip(("hidden", "head", "softcap"), expected, got, strict=True):
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


@pytest.mark.parametrize("tile", [RAGGED, None], ids=["tiled", "whole"])
@pytest.mark.parametrize("excluded", [0, 41, RAGGED_VOCAB - 1], ids=["first", "inside", "last"])
def test_an_excluded_column_takes_no_mass_in_the_loss_the_prediction_or_the_gradient(tile, excluded):
    """The excluded column leaves the partition and the prediction, in
    whichever tile holds it: the losses, log partitions, predictions and both
    gradients are the full pass's with that column's logits at -inf, and its
    own head column receives no gradient."""
    hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=7, tokens=(1, RAGGED_TOKENS))
    targets = jnp.where(targets == excluded, (excluded + 1) % RAGGED_VOCAB, targets)
    precision = jax.lax.Precision.HIGHEST
    # The excluded column scores highest everywhere, so a prediction that kept it would show.
    head = head.at[:, excluded].set(jnp.abs(head).max(axis=1) * 4)

    def full(states, matrix):
        logits = chunked.head_logits(states, matrix, softcap=None, precision=precision)
        logits = logits.at[..., excluded].set(-jnp.inf)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses + 0.7 * jnp.square(log_z)), (losses, log_z, jnp.argmax(logits, axis=-1))

    def chunked_pass(states, matrix):
        losses, predicted, log_z = chunked_cross_entropy(
            states, matrix, targets, 4, precision=precision, tile=tile, excluded=excluded)
        return jnp.sum(losses + 0.7 * jnp.square(log_z)), (losses, log_z, predicted)

    (_, want), want_grads = jax.value_and_grad(full, argnums=(0, 1), has_aux=True)(hidden, head)
    (_, got), got_grads = jax.jit(jax.value_and_grad(chunked_pass, argnums=(0, 1), has_aux=True))(
        hidden, head)

    np.testing.assert_allclose(got[0], want[0], rtol=1e-5)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-5)
    np.testing.assert_array_equal(got[2], want[2])
    assert not jnp.any(got[2] == excluded)
    for expected, actual in zip(want_grads, got_grads, strict=True):
        assert jnp.abs(actual - expected).max() <= 1e-5 * jnp.abs(expected).max()
    assert jnp.all(got_grads[1][:, excluded] == 0)


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

    Held to an fp32 oracle on the same values: a gradient in `dtype` that
    rounds once, at the end, is within half its ulp of it, plus fp32 sum
    noise; rounding each tile's partial sum adds up to half an ulp per tile.
    The products run at HIGHEST, so the logits' cotangent is not rounded to
    bf16 inside them, which moves a cancelling sum by more than an ulp.
    Comparing against a gradient rounded to `dtype` would count two
    roundings, and a near-tie then differs by a whole ulp.
    """
    hidden, head, targets = inputs(vocab=17, features=7, tokens=(257,), dtype=dtype)
    head = head.astype(dtype)
    precision = jax.lax.Precision.HIGHEST

    def full(states, matrix):
        return jnp.mean(oracle(states, matrix, targets, precision=precision)[0])

    def tiled(states, matrix):
        return jnp.mean(chunked_cross_entropy(
            states, matrix, targets, 4, tile=RAGGED, precision=precision)[0])

    expected = jax.grad(full, argnums=(0, 1))(hidden.astype(jnp.float32),
                                              head.astype(jnp.float32))
    got = jax.grad(tiled, argnums=(0, 1))(hidden, head)

    assert got[0].dtype == hidden.dtype and got[1].dtype == head.dtype == dtype
    for name, want, have in zip(("hidden", "head"), expected, got, strict=True):
        largest = jnp.abs(want).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        half_ulp = jnp.spacing(jnp.abs(want).astype(dtype)).astype(jnp.float32) / 2
        noise = 1e-6 * largest
        assert bool(jnp.all(jnp.abs(have.astype(jnp.float32) - want) <= half_ulp + noise)), name


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
        table = model.apply({"params": tree}, method="output_table")
        losses, _, log_z = chunked_cross_entropy(
            states, table.matrix, targets, 4, softcap=30., vocab_major=table.vocab_major)
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


# --- a production vocabulary against float64 ---------------------------------

# Gemma's 262,144 columns. What changes with the width is the length of the
# reductions: the partition sums 262,144 exponentials and the hidden
# gradient 262,144 products. The worst-case bound for such a sum, (V - 1) u,
# is 1.6e-2 and says nothing, so the bounds here are Higham and Mary's
# probabilistic ones ("A New Approach to Probabilistic Rounding Error
# Analysis", SIAM J. Sci. Comput. 41(5), 2019, theorem 3.1): a sum of n
# terms rounded independently in fp32 is off by at most lambda sqrt(n) u
# times the sum of its absolute terms, except with probability
# 2 exp(-lambda^2 / 2), which lambda = 6 puts at 3e-8 a sum.
GEMMA_VOCAB = 262_144
LAMBDA = 6.0
U32 = 2.0 ** -24


def float64_cross_entropy(hidden, head, targets, weights, softcap):
    """Losses, log partitions and both gradients of `sum(weights * losses)`,
    in NumPy float64, with the sums of absolute terms the bounds scale."""
    hidden, head = np.asarray(hidden, np.float64), np.asarray(head, np.float64)
    weights, targets = np.asarray(weights, np.float64), np.asarray(targets)
    scores = hidden @ head
    magnitude = np.abs(hidden) @ np.abs(head)
    logits, slope = scores, np.ones_like(scores)
    if softcap is not None:
        logits = softcap * np.tanh(scores / softcap)
        slope = 1 - np.tanh(scores / softcap) ** 2
    peak = logits.max(-1, keepdims=True)
    log_z = (peak + np.log(np.exp(logits - peak).sum(-1, keepdims=True)))[:, 0]
    rows = np.arange(len(targets))
    losses = log_z - logits[rows, targets]
    probabilities = np.exp(logits - log_z[:, None])
    residual = probabilities.copy()
    residual[rows, targets] -= 1
    cotangent = weights[:, None] * residual * slope
    return {"losses": losses, "log_z": log_z, "hidden": cotangent @ head.T,
            "head": hidden.T @ cotangent, "magnitude": magnitude, "logits": logits,
            "probabilities": probabilities, "residual": residual, "slope": slope}


def float64_bounds(hidden, head, weights, exact, chunks=None):
    """The fp32 bound on every loss and gradient entry.

    A logit is a dot over F features, so it is off by at most
    lambda sqrt(F) u of its absolute terms, plus 3 u of itself through a
    softcap's divide, tanh and multiply; the largest of a row's is E. The
    partition's exponentials are each off by |dz| + 2 u relative, and their
    sum by lambda sqrt(V) u relative more, so log Z is off by at most
    E + (lambda sqrt(V) + 4) u + C u |log Z|, where C counts the roundings
    of log Z itself: one for the whole row, one per chunk for the chunked
    loop, whose logaddexp rounds the running log Z at every chunk (observed:
    the log Z error grows from 1.0e-06 unchunked to 2.9e-06 at 16 chunks, on
    log Z near 17, where u |log Z| is 1.0e-06). A loss is off by that plus its target
    logit's error. A probability exp(z - log Z) carries the same E + log Z
    error plus 2 u relative, and the gradients sum weight * (p - onehot)
    against the head over V columns and against the states over T tokens.
    """
    hidden, head = np.asarray(hidden, np.float64), np.asarray(head, np.float64)
    weights = np.abs(np.asarray(weights, np.float64))
    features, vocab = head.shape
    tokens = hidden.shape[0]
    logit_error = LAMBDA * np.sqrt(features) * U32 * exact["magnitude"] + 3 * U32 * np.abs(exact["logits"])
    largest = logit_error.max(-1)
    z_error = largest + (LAMBDA * np.sqrt(vocab) + 4) * U32 + (chunks or 1) * U32 * np.abs(exact["log_z"])
    probability_error = (largest + z_error + 2 * U32)[:, None] * exact["probabilities"]
    spread = weights[:, None] * exact["slope"]
    gradient_error = spread * probability_error
    terms = spread * np.abs(exact["residual"])
    return {
        "losses": z_error + largest + U32 * np.abs(exact["losses"]),
        "log_z": z_error,
        "hidden": (gradient_error @ np.abs(head).T
                   + LAMBDA * np.sqrt(vocab) * U32 * (terms @ np.abs(head).T)),
        "head": (np.abs(hidden).T @ gradient_error
                 + LAMBDA * np.sqrt(tokens) * U32 * (np.abs(hidden).T @ terms)),
    }


@pytest.fixture(scope="module")
def gemma_vocabulary():
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    tokens, features = 32, 64
    hidden = jax.random.normal(keys[0], (tokens, features), jnp.float32)
    # Logits of standard deviation 3, so the partition is carried by a few
    # hundred columns scattered over every chunk rather than by all of them
    # equally or by one.
    head = 3 / np.sqrt(features) * jax.random.normal(keys[1], (features, GEMMA_VOCAB), jnp.float32)
    targets = jax.random.randint(keys[2], (tokens,), 0, GEMMA_VOCAB)
    weights = jax.random.uniform(keys[3], (tokens,), jnp.float32, 0.5, 1.5)
    return hidden, head, targets, weights


def fp32_cross_entropy(hidden, head, targets, weights, softcap, chunks):
    """Losses, log partitions and both gradients from the chunked head, or
    with `chunks` None from the full-vocabulary pass it replaced."""
    def run(states, matrix):
        if chunks is None:
            losses, log_z = oracle(states, matrix, targets, softcap)
        else:
            losses, _, log_z = chunked_cross_entropy(states, matrix, targets, chunks, softcap=softcap)
        return jnp.sum(weights * losses), (losses, log_z)

    (_, (losses, log_z)), (d_hidden, d_head) = jax.jit(
        jax.value_and_grad(run, argnums=(0, 1), has_aux=True))(hidden, head)
    return {"losses": losses, "log_z": log_z, "hidden": d_hidden, "head": d_head}


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap-30"])
@pytest.mark.parametrize("chunks", [None, 4, 16], ids=["unchunked", "4-chunks", "16-chunks"])
def test_a_gemma_sized_vocabulary_is_within_fp32_rounding_of_float64(gemma_vocabulary, chunks, softcap):
    """Losses, log partitions and the gradients of both the states and the
    head, chunked and unchunked, every entry inside its bound. Observed at
    most 1.2e-2 of the bound on log Z and the losses (errors 1.0e-06 to
    3.4e-06), 5.5e-3 on the states' gradient and 0.23 on the head's."""
    hidden, head, targets, weights = gemma_vocabulary
    exact = float64_cross_entropy(hidden, head, targets, weights, softcap)
    bounds = float64_bounds(hidden, head, weights, exact, chunks)

    actual = fp32_cross_entropy(hidden, head, targets, weights, softcap, chunks)

    for name, value in actual.items():
        error = np.abs(np.asarray(value, np.float64) - exact[name])
        assert np.all(error <= bounds[name]), (name, float(np.max(error / bounds[name])))


def test_the_gemma_bound_is_tighter_than_a_dropped_chunk(gemma_vocabulary):
    """The loss bound has to reject the mutation a chunk loop invites: the
    partition without its last of 16 chunks moves every loss by more than
    the bound admits."""
    hidden, head, targets, weights = gemma_vocabulary
    exact = float64_cross_entropy(hidden, head, targets, weights, None)
    bounds = float64_bounds(hidden, head, weights, exact, 16)
    first, last = vocabulary_chunks(GEMMA_VOCAB, 16)[-1]
    kept = np.delete(exact["logits"], np.s_[first:last], axis=-1)
    peak = kept.max(-1, keepdims=True)
    partial = (peak + np.log(np.exp(kept - peak).sum(-1, keepdims=True)))[:, 0]
    assert np.all(np.abs(partial - exact["log_z"]) > bounds["log_z"])


def test_the_forward_rounds_the_table_once_not_once_per_token_tile():
    """Under bf16 the forward multiplies the table rounded to bf16. Rounding
    each vocabulary chunk inside the token loop reads the fp32 table once per
    token tile: on an A100, Qwen3-0.6B's 4 x 1024 tokens in 1024-token tiles
    spent 2.65 ms of a 162 ms step converting it, where once costs 0.65."""
    hidden, head, targets = inputs(vocab=64, features=16, tokens=(4, 32), dtype=jnp.bfloat16)
    program = jax.make_jaxpr(lambda h, w: chunked_cross_entropy(h, w, targets, 4, tile=(32, 16)))(
        hidden, head)

    def converts(jaxpr, looped):
        found = []
        for eqn in jaxpr.eqns:
            inner = [sub for param in eqn.params.values()
                     for sub in (param if isinstance(param, tuple) else (param,))
                     if isinstance(sub, jax.extend.core.ClosedJaxpr | jax.extend.core.Jaxpr)]
            for sub in inner:
                found += converts(
                    getattr(sub, "jaxpr", sub), looped or eqn.primitive.name in ("while", "scan")
                )
            if eqn.primitive.name == "convert_element_type" and eqn.outvars[0].aval.ndim == 2 \
                    and eqn.outvars[0].aval.shape[-1] == 16 and eqn.outvars[0].aval.dtype == jnp.bfloat16:
                found.append((eqn.outvars[0].aval.shape, looped))
        return found

    tables = converts(program.jaxpr, looped=False)
    assert tables == [((64, 16), False)], tables


@pytest.mark.parametrize("tile", [None, RAGGED])
def test_the_bf16_head_rounds_its_logits_and_their_cotangent_once(tile):
    """Under the bf16 algorithm the head takes torch autocast's rounding: the
    logits are bf16 values, and their cotangent is rounded to bf16 once
    before both products. The reference rounds the logits with
    `rounded_to`, whose derivative rounds the cotangent where the value
    rounds, and multiplies at the highest precision. On eighths both sides
    form the same logits; the softmax's fp32 rounding can still move a
    cotangent entry across a bf16 rounding boundary, one bf16 ulp, 2^-7 of
    it at most, so each gradient entry is within 2^-7 of its terms'
    magnitudes, plus fp32's rounding of them. Kept in fp32 (`highest`) the
    cotangent misses that bound."""
    hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=16, tokens=(RAGGED_TOKENS,))
    hidden, head = eighths(hidden, head)
    table = head.T

    def strict_or_rounded(round_logits):
        def loss(states, matrix):
            logits = jnp.einsum('td,vd->tv', states, matrix, precision=jax.lax.Precision.HIGHEST)
            if round_logits:
                logits = rounded_to(logits, jnp.bfloat16)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
        return loss

    rounded = strict_or_rounded(round_logits=True)
    want_states, want_table = jax.grad(rounded, argnums=(0, 1))(hidden, table)
    have_states, have_table = jax.grad(lambda states, matrix: jnp.mean(chunked_cross_entropy(
        states, matrix.T, targets, 4, tile=tile, precision=chunked.BF16)[0]), argnums=(0, 1))(hidden, table)
    logits = jnp.einsum('td,vd->tv', hidden, table, precision=jax.lax.Precision.HIGHEST)
    cotangent = jnp.abs(jax.grad(lambda z: jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(z, targets)))(logits))
    terms_states = jnp.einsum('tv,vd->td', cotangent, jnp.abs(table), precision=jax.lax.Precision.HIGHEST)
    terms_table = jnp.einsum('tv,td->vd', cotangent, jnp.abs(hidden), precision=jax.lax.Precision.HIGHEST)
    slack = (2 ** -7 + 64 * 2 ** -24)

    assert jnp.all(jnp.abs(have_states - want_states) <= slack * terms_states)
    assert jnp.all(jnp.abs(have_table - want_table) <= slack * terms_table)
    strict_states, _ = jax.grad(strict_or_rounded(round_logits=False), argnums=(0, 1))(hidden, table)
    assert not jnp.all(jnp.abs(strict_states - have_states) <= 2 ** -20 * terms_states)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_the_whole_logits_head_computes_what_the_tiled_one_does(dtype, softcap):
    """`tile=None` keeps the logits for the backward instead of recomputing
    them tile by tile. Same losses, predictions, log partitions and
    gradients: the logits' cotangent stays fp32 into the state product
    either way, and the head's gradient accumulates in fp32 once."""
    hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=16,
                                   tokens=(RAGGED_TOKENS,), dtype=dtype)
    # Targets outside the vocabulary pick no column, which the
    # vocabulary-split head relies on for every shard but the owner.
    targets = targets.at[:2].set(jnp.asarray([-1, RAGGED_VOCAB + 5], targets.dtype))

    def run(tile):
        def loss(states, matrix, cap):
            losses, _, log_z = chunked_cross_entropy(
                states, matrix, targets, 4, tile=tile, softcap=cap)
            return jnp.mean(losses) + 0.01 * jnp.mean(jnp.square(log_z))
        outputs = chunked_cross_entropy(hidden, head, targets, 4, tile=tile, softcap=softcap)
        return outputs, jax.grad(loss, argnums=(0, 1, 2) if softcap else (0, 1))(
            hidden, head, softcap)

    # fp32 products at full precision, so the fp32 bound below holds on a
    # backend whose default fp32 product is TF32 too.
    with (jax.default_matmul_precision("highest") if dtype == jnp.float32 else contextlib.nullcontext()):
        (tiled, tiled_grads), (whole, whole_grads) = run(RAGGED), run(None)
    assert jnp.array_equal(tiled[1], whole[1])
    # The two take log Z in a different order, so a probability can move by
    # an fp32 ulp; where the head's gradient multiplies bf16 operands, that
    # ulp can flip the cotangent's bf16 rounding: one bf16 ulp of an entry,
    # which is 2^-7 of an entry sitting at a power of two, the bottom of its
    # binade, and the flipped entry can be the largest.
    # In fp32 each path's log Z is one running reduction over its row
    # (`_row_terms`). A merge rounds the exponential that rescales a partial
    # sum, the product and the add, three roundings, and XLA's merge order is
    # not shown here, so its depth is taken as the row's V entries, the
    # conservative count: the sum of positive terms is off by at most
    # gamma(3V) relative, which is log Z's absolute error from the sum, and
    # log Z = m + log(s) rounds twice more, gamma(2) |log Z|. A probability
    # exp(x - log Z) carries that error relative and three roundings of its
    # own, and a gradient entry sums V of them, V more roundings. Two paths,
    # so twice gamma(4V + 5) + gamma(2) max |log Z|, 9.8e-4 here. Measured:
    # 4.9e-6 of the largest entry on CPU and 1.5e-6 on an RTX 4080.
    if dtype == jnp.float32:
        def gamma(steps):
            return steps * 2.0**-24 / (1 - steps * 2.0**-24)
        bound = 2 * (gamma(4 * RAGGED_VOCAB + 5) + gamma(2) * float(jnp.abs(tiled[2]).max()))
    else:
        bound = 2.0**-7
    for have, want in [
        *zip(whole[::2], tiled[::2], strict=True),
        *zip(whole_grads, tiled_grads, strict=True),
    ]:
        have, want = jnp.asarray(have, jnp.float32), jnp.asarray(want, jnp.float32)
        assert jnp.abs(have - want).max() <= bound * jnp.abs(want).max() + 1e-7


def test_a_step_that_does_not_fit_tiles_the_head_before_it_recomputes_blocks():
    """The first rung of the fit ladder tiles a head that kept its whole
    logits; only then does the model's remat climb."""
    from dew.objectives.lm import LMObjective
    from dew.training.trainer import recompute_more

    objective = LMObjective(small_model(), seq_len=12)
    assert objective.head_tile is None
    assert recompute_more(objective) and objective.head_tile == chunked.chunked_tile()
    assert objective.model.remat is None
    assert recompute_more(objective) and objective.model.remat is not None


def test_a_pass_with_no_backward_never_holds_the_logits_whole():
    """Whole logits pay only for a backward; an evaluation or a scoring pass
    runs the tiled forward, whose temporaries hold a tile, not the logits."""
    hidden, head, targets = inputs(vocab=32768, features=16, tokens=(4096,))

    def temporaries(tile):
        forward = jax.jit(lambda h, w: chunked_cross_entropy(h, w, targets, 4, tile=tile)[0])
        return forward.lower(hidden, head).compile().memory_analysis().temp_size_in_bytes

    assert temporaries(None) <= temporaries((1024, 8192))
    assert temporaries(None) < 4096 * 32768 * 4 // 4


@pytest.mark.parametrize("tile", [None, RAGGED])
def test_both_gradient_products_read_one_bf16_cotangent(tile):
    """Under the bf16 algorithm the backward hands the state product and the
    head's own one bf16 copy of the logits' cotangent. Every product into a
    gradient (the ones whose result is feature-wide) multiplies bf16
    operands; the tiled backward's logits products read fp32-stored bf16
    values under the algorithm, as its forward does."""
    hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=16, tokens=(RAGGED_TOKENS,))
    program = jax.make_jaxpr(jax.grad(lambda states, matrix: jnp.mean(chunked_cross_entropy(
        states, matrix, targets, 4, tile=tile, precision=chunked.BF16)[0]), argnums=(0, 1)))(hidden, head)

    def dots(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "dot_general":
                yield equation
            for param in equation.params.values():
                for sub in (param if isinstance(param, tuple) else (param,)):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr | jax.extend.core.Jaxpr):
                        yield from dots(getattr(sub, "jaxpr", sub))

    found = [equation for equation in dots(program.jaxpr) if equation.outvars[0].aval.shape[-1] == 16]
    assert len(found) >= 2, found
    assert all(v.aval.dtype == jnp.bfloat16 for equation in found for v in equation.invars), found
