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

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import rounded_operand, rounded_to
from dew.nn.protocols import OutputTable
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy, fused_head_loss, vocabulary_chunks

EXERCISED = {jnp.float32: 'highest', jnp.bfloat16: 'default'}
"""The matmul precision each compute dtype's head is compared under: bf16's
default, the bf16 product it exists for, and fp32's highest. At the default
an A100 runs each fp32 dot at TF32 or at fp32 as XLA picks for the program,
so a whole head and its tiles would differ by TF32's rounding."""


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
    with jax.default_matmul_precision(EXERCISED[dtype]):
        logits = model.apply(variables, ids)
        states = model.apply(variables, ids, method='hidden_states')
        table = model.apply(variables, method='output_table')
        actual = OutputTable(table.matrix, table.vocab_major, bias).logits(states)
        np.testing.assert_array_equal(actual, logits)
        for tile in (None, (2, 11)):
            losses, predicted, _ = chunked_cross_entropy(
                states, OutputTable(table.matrix, vocab_major=table.vocab_major, bias=bias), ids, 4, tile=tile
            )
            expected = optax.softmax_cross_entropy_with_integer_labels(logits, ids)
            assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
            np.testing.assert_array_equal(predicted, logits.argmax(-1))
    assert variables['params']['head_bias'].dtype == jnp.float32
    assert not np.array_equal(np.asarray(bias), np.asarray(bias.astype(jnp.bfloat16), np.float32))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('softcap', [None, 3.0], ids=['uncapped', 'capped'])
def test_the_fused_head_takes_the_whole_heads_loss_and_gradients_under_the_float64_rule(dtype, softcap):
    """The gradient-in-forward head's weighted sum of losses and z-loss, and
    its gradients for the states, the head and the bias, as close to float64
    as the whole-logits head's, over chunks whose last is short, padded
    weights and a cap; its per-token losses, log Z and top-1 columns are the
    whole head's."""
    from reference_error import assert_as_exact_as_the_reference

    states, matrix, targets = inputs(vocab=37, features=16, tokens=(2, 5), dtype=dtype)
    # Sums exact in any order, so the chunks' products are the whole head's.
    states, matrix = eighths(states, matrix)
    bias = jax.random.normal(jax.random.key(3), (37,), jnp.float32) / 17
    weights = jnp.asarray([[1, 1, 0, 1, 2], [1, 0, 1, 1, 1]], jnp.float32)
    z_loss = 1e-2

    def whole(states, matrix, bias):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap).logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(weights * (losses + z_loss * jnp.square(log_z))), (losses, log_z, logits)

    def fused(states, matrix, bias):
        return fused_head_loss(states, OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap),
                               targets, weights, z_loss=z_loss, tile=(3, 11))

    with jax.default_matmul_precision(EXERCISED[dtype]):
        (total, (losses, predicted, log_z)), gradients = jax.value_and_grad(
            fused, argnums=(0, 1, 2), has_aux=True)(states, matrix, bias)
        (reference, (expected, expected_z, logits)), expected_gradients = jax.value_and_grad(
            whole, argnums=(0, 1, 2), has_aux=True)(states, matrix, bias)
    np.testing.assert_array_equal(predicted, logits.argmax(-1))
    np.testing.assert_allclose(losses, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(log_z, expected_z, rtol=1e-6, atol=1e-6)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(value, jnp.float64) for value in (states, matrix, bias)]
        exact, exact_gradients = jax.value_and_grad(
            lambda *values: whole(*values)[0], argnums=(0, 1, 2))(*wide)
    for label, mine, theirs, true in zip(('loss', 'states', 'head', 'bias'), (total, *gradients),
                                         (reference, *expected_gradients), (exact, *exact_gradients),
                                         strict=True):
        assert_as_exact_as_the_reference(mine, theirs, true, f'{dtype} {softcap} fused {label}')


@pytest.mark.parametrize('kind', ['nvfp4', 'fp8'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_a_checkpoints_head_input_quantization_reaches_every_head_that_scores_it(kind, dtype):
    """A model computing under a checkpoint's input quantization of its
    lm_head (ModelOpt's, as `Pretrained` reads it) has its head's input
    rounded by the quantizer in its forward; the table a loss, a scorer and
    a sampler read carries the same quantizer, so the table's logits are the
    forward's and the tiled loss is the forward's logits' cross entropy."""
    from dew.training.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

    model = InputQuantization(format='modelopt', sites=(('lm_head', kind),)).apply(CausalTransformer(
        vocab_size=97, emb_features=32, num_layers=1, num_heads=2, mlp_features=32, max_seq_len=8,
        head_bias=True, tie_embeddings=False, dtype=dtype, qk_norm=False, attention_impl='reference'))
    ids = jnp.asarray([[2, 7, 3, 5], [11, 4, 9, 1]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    variables[QUANTIZATION]['lm_head'][INPUT_SCALES[kind]] = jnp.asarray(0.37, jnp.float32)
    with jax.default_matmul_precision(EXERCISED[dtype]):
        logits = model.apply(variables, ids)
        states = model.apply(variables, ids, method='hidden_states')
        table = chunked.head_table(model, variables)
        assert table is not None and table.input_quantizer is not None
        np.testing.assert_array_equal(chunked.model_logits(model, variables, states), logits)
        unquantized = OutputTable(table.matrix, table.vocab_major, table.bias, compute=table.compute)
        assert not np.array_equal(unquantized.logits(states), logits)
        losses, predicted, _ = chunked.head_cross_entropy(model, variables, states, ids, 4, tile=(3, 11))
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, ids)
    assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
    np.testing.assert_array_equal(predicted, logits.argmax(-1))


@pytest.mark.parametrize('tied', [False, True])
def test_a_bf16_models_table_scores_states_handed_over_in_fp32_as_its_head_does(tied):
    """A table carries its model's compute dtype, so states that reach a
    scorer widened to fp32 (a fallback's or a tool's) multiply the head as the
    model's own bf16 head multiplies them: bitwise its logits."""
    model = CausalTransformer(vocab_size=97, emb_features=24, num_layers=1, num_heads=3,
                              mlp_features=32, max_seq_len=8, tie_embeddings=tied, dtype=jnp.bfloat16,
                              qk_norm=False, attention_impl='reference')
    ids = jnp.asarray([[2, 7, 3], [11, 4, 9]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    with jax.default_matmul_precision('default'):
        logits = model.apply(variables, ids)
        states = model.apply(variables, ids, method='hidden_states')
        table = model.apply(variables, method='output_table')
        widened = states.astype(jnp.float32) + jnp.float32(2.0 ** -12)
        rounded = OutputTable(table.matrix, table.vocab_major, compute=jnp.bfloat16).logits(widened)
        np.testing.assert_array_equal(rounded, OutputTable(table.matrix, table.vocab_major).logits(
            widened.astype(jnp.bfloat16)))
        np.testing.assert_array_equal(OutputTable(table.matrix, table.vocab_major).logits(states), logits)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('tile', [None, (3, 11)], ids=['whole', 'tiled'])
def test_vocabulary_bias_gradient_matches_full_logits_under_the_float64_rule(dtype, tile):
    from reference_error import assert_as_exact_as_the_reference

    from dew.nn.precision import head_product

    # Sums exact in any order: a tile's product is then the whole product on
    # any device, as the oracle takes it, where an fp32 sum's last bit, which
    # a GPU's tile shape picks, can move a bf16 logit a whole bf16 step.
    states, matrix, targets = inputs(vocab=37, features=16, tokens=(2, 5), dtype=dtype)
    states, matrix = eighths(states, matrix)
    bias = jax.random.normal(jax.random.key(3), (37,), jnp.float32) / 17
    with jax.default_matmul_precision('default'):
        def full(offset):
            logits = OutputTable(matrix, vocab_major=False, bias=offset).logits(states)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        def bounded(offset):
            return chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, bias=offset), targets, 4, tile=tile
            )[0].mean()

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


@pytest.mark.mesh(devices=2)
def test_a_quantized_split_head_scores_as_one_device_and_moves_no_head_sized_array():
    """A head whose input a checkpoint quantizes, split over the vocabulary on
    a mesh, scores each token as one device does, and no collective of the
    step carries an array the head's size: tokens and per-token terms cross
    devices, the head stays in its shards."""
    import re

    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.objectives.lm import LMObjective
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import shard_batch
    from dew.training.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

    model = InputQuantization(format='modelopt', sites=(('lm_head', 'fp8'),)).apply(CausalTransformer(
        vocab_size=512, emb_features=16, num_layers=1, num_heads=2, mlp_features=16, max_seq_len=4,
        tie_embeddings=False, qk_norm=False, attention_impl='reference'))
    tokens = jnp.arange(20, dtype=jnp.int32).reshape(4, 5)
    variables = model.init(jax.random.key(0), tokens[:, :-1])
    variables[QUANTIZATION]['lm_head'][INPUT_SCALES['fp8']] = jnp.asarray(0.02, jnp.float32)
    objective = LMObjective(model, seq_len=4, variables=variables, head_chunks=4,
                            head_tile=(2, 64), ema_decay=None)
    expected = objective.token_scores(variables, tokens)
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    shardings = Layout(min_shard=1).shardings(mesh, variables)
    with jax.set_mesh(mesh):
        placed = jax.device_put(variables, shardings)
        batch = shard_batch(mesh, {'text': np.asarray(tokens)})['text']
        actual = jax.jit(objective.token_scores)(placed, batch)
        states = jax.device_put(model.apply(variables, tokens[:, :-1], method='hidden_states'),
                                NamedSharding(mesh, P('fsdp')))
        targets = jax.device_put(tokens[:, 1:], NamedSharding(mesh, P('fsdp')))

        def loss(params, states):
            return chunked.head_cross_entropy(model, {**placed, 'params': params}, states, targets, 4,
                                              tile=(2, 64))[0].sum()

        # The head's forward and backward alone: the embedding's own gather is not the head's.
        program = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(placed['params'], states).compile().as_text()
    assert jnp.abs(actual.losses - expected.losses).max() <= 1e-5 * jnp.abs(expected.losses).max()
    np.testing.assert_array_equal(actual.correct, expected.correct)
    head = 512 * 16
    collective = re.compile(r'=\s*(.*?)\s(?:all-gather|all-reduce|reduce-scatter|collective-permute|'
                            r'all-to-all)(?:-start)?\(')
    moved = [int(np.prod([int(size) for size in dims.split(',') if size], dtype=np.int64))
             for line in program.splitlines() if (match := collective.search(line))
             for dims in re.findall(r'\w+\[([0-9,]*)\]', match.group(1))]
    assert moved and max(moved) < head // 2, moved


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


def float32_values(jaxpr) -> int:
    """How many values of float32 `jaxpr`, and every jaxpr it holds, makes."""
    made = sum(any(getattr(getattr(var, "aval", None), "dtype", None) == jnp.float32
                   for var in eqn.outvars) for eqn in jaxpr.eqns)
    return made + sum(float32_values(inner) for eqn in jaxpr.eqns
                      for inner in jax.extend.core.jaxprs_in_params(eqn.params))


# --- the mutations, each one a loss that would still train ------------------

# --- against the real backbone ---------------------------------------------

def head_matrix(model, variables):
    """The `[features, vocab]` matrix `model`'s head contracts (`output_table`)."""
    table = model.apply(variables, method="output_table")
    return table.matrix.T if table.vocab_major else table.matrix


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
    losses, predicted, _ = chunked_cross_entropy(
        hidden, OutputTable(head_matrix(model, variables), vocab_major=False), targets, chunks
    )

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
        head = head_matrix(model, {'params': params})
        operands = (rounded_operand(value.astype(jnp.float32), jnp.bfloat16)
                    for value in (hidden, head))
        logits = rounded_to(jnp.einsum('btd,dv->btv', *operands, precision=jax.lax.Precision.HIGHEST),
                            jnp.bfloat16)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    def chunked(params):
        hidden = model.apply({'params': params}, ids,
                             method=CausalTransformer.hidden_states)
        return jnp.mean(
            chunked_cross_entropy(
                hidden, OutputTable(head_matrix(model, {'params': params}), vocab_major=False), targets, 4
            )[0]
        )

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

@pytest.mark.parametrize("vocab, chunks, message", [
    (100, 0, "at least one chunk"),
    (4, 8, "empty tiles"),
    (9, 4, "do not split"),
])
def test_impossible_chunk_counts_are_refused(vocab, chunks, message):
    with pytest.raises(ValueError):
        vocabulary_chunks(vocab, chunks)


def test_mismatched_shapes_are_refused():
    hidden, head, targets = inputs()
    with pytest.raises(ValueError):
        chunked_cross_entropy(hidden[..., :-1], OutputTable(head, vocab_major=False), targets, 4)
    with pytest.raises(ValueError):
        chunked_cross_entropy(hidden, OutputTable(head, vocab_major=False), targets[:, :-1], 4)


# --- the bounded backward --------------------------------------------------

# 2,053 columns in 2,048-column tiles and 131 tokens in 128-token tiles both
# leave a short last tile, so a dropped ragged end shows up as a missing
# gradient rather than as a shape error.
RAGGED = (128, 2048)
RAGGED_VOCAB, RAGGED_TOKENS = 2053, 131


def oracle(hidden, head, targets, softcap=None, precision=None):
    """The full-vocabulary losses and log partitions, one big logits tensor."""
    logits = OutputTable(head, vocab_major=False, softcap=softcap, precision=precision).logits(hidden)
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
            states,
            OutputTable(matrix, vocab_major=False, softcap=softcap, precision=precision),
            targets,
            4,
            tile=RAGGED,
        )
        return jnp.sum(loss_weight * losses + z_weight * jnp.square(log_z))

    want_loss, want = jax.value_and_grad(full, argnums=(0, 1))(hidden, head)
    got_loss, got = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(hidden, head)

    assert jnp.abs(got_loss - want_loss) <= 1e-5 * jnp.abs(want_loss)
    _, predicted, _ = chunked_cross_entropy(
        hidden,
        OutputTable(head, vocab_major=False, softcap=softcap, precision=precision),
        targets,
        4,
        tile=RAGGED,
    )
    _, expected_top1 = reference(hidden, head, targets, softcap=softcap)
    np.testing.assert_array_equal(predicted, expected_top1)
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
            states, OutputTable(matrix, vocab_major=False, softcap=cap), targets, 4, tile=RAGGED
        )
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    expected = jax.grad(full, argnums=(0, 1, 2))(hidden, head, 30.)
    got = jax.grad(tiled, argnums=(0, 1, 2))(hidden, head, 30.)
    _, predicted, _ = chunked_cross_entropy(
        hidden, OutputTable(head, vocab_major=False, softcap=30.0), targets, 4, tile=RAGGED
    )
    _, expected_top1 = reference(hidden, head, targets, softcap=30.)
    np.testing.assert_array_equal(predicted, expected_top1)
    for name, want, have in zip(("hidden", "head", "softcap"), expected, got, strict=True):
        largest = jnp.abs(want).max()
        assert largest > 0, f"the {name} gradient is zero, so nothing is checked"
        assert jnp.abs(have - want).max() <= 1e-5 * largest, name


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
        logits = OutputTable(matrix, vocab_major=False, precision=precision).logits(states)
        logits = logits.at[..., excluded].set(-jnp.inf)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses + 0.7 * jnp.square(log_z)), (losses, log_z, jnp.argmax(logits, axis=-1))

    def chunked_pass(states, matrix):
        losses, predicted, log_z = chunked_cross_entropy(
            states,
            OutputTable(matrix, vocab_major=False, precision=precision),
            targets,
            4,
            tile=tile,
            excluded=excluded,
        )
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
        return jnp.mean(
            chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, precision=precision), targets, 4, tile=RAGGED
            )[0]
        )

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
        losses, _, log_z = chunked_cross_entropy(
            states,
            OutputTable(head_matrix(model, {"params": tree}), vocab_major=False, softcap=30.0),
            targets,
            4,
        )
        return jnp.mean(losses + 0.02 * jnp.square(log_z))

    expected, got = jax.grad(full)(params), jax.grad(tiled)(params)
    for want, have in zip(jax.tree.leaves(expected), jax.tree.leaves(got),
                          strict=True):
        assert jnp.abs(have - want).max() <= 1e-5 * jnp.abs(want).max()


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


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap-30"])
def test_a_262144_vocabulary_matches_optax_in_float64_on_cpu(softcap):
    """Seven targets spanning chunk boundaries, losses, log Z and both gradients
    against Optax's full-vocabulary CE; 1e-12 allows float64 reduction order.
    """
    rng = np.random.default_rng(713)
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        hidden = jnp.asarray(rng.normal(size=(7, 5)), jnp.float64)
        head = jnp.asarray(rng.normal(size=(5, GEMMA_VOCAB)), jnp.float64)
        targets = jnp.asarray([0, 65535, 65536, 131071, 131072, 196608, 262143], jnp.int32)
        weights = jnp.asarray(rng.uniform(.5, 1.5, 7), jnp.float64)

        def upstream(states, matrix):
            logits = states @ matrix
            if softcap is not None:
                logits = softcap * jnp.tanh(logits / softcap)
            losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return jnp.sum(weights * losses), (
                losses, jax.nn.logsumexp(logits, -1), logits.argmax(-1).astype(jnp.int32))

        def tiled(states, matrix):
            losses, predictions, log_z = chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, softcap=softcap), targets, 16, tile=(3, 8192)
            )
            return jnp.sum(weights * losses), (losses, log_z, predictions)

        expected = jax.jit(jax.value_and_grad(upstream, (0, 1), has_aux=True))(hidden, head)
        actual = jax.jit(jax.value_and_grad(tiled, (0, 1), has_aux=True))(hidden, head)
        for have, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert have.dtype == want.dtype
            if jnp.issubdtype(want.dtype, jnp.integer):
                np.testing.assert_array_equal(have, want)
            else:
                np.testing.assert_allclose(have, want, rtol=1e-12, atol=1e-12)


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
            losses, _, log_z = chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, softcap=softcap), targets, chunks
            )
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
    have_states, have_table = jax.grad(
        lambda states, matrix: jnp.mean(
            chunked_cross_entropy(
                states,
                OutputTable(matrix.T, vocab_major=False, precision=chunked.BF16),
                targets,
                4,
                tile=tile,
            )[0]
        ),
        argnums=(0, 1),
    )(hidden, table)
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
                states, OutputTable(matrix, vocab_major=False, softcap=cap), targets, 4, tile=tile
            )
            return jnp.mean(losses) + 0.01 * jnp.mean(jnp.square(log_z))
        outputs = chunked_cross_entropy(
            hidden, OutputTable(head, vocab_major=False, softcap=softcap), targets, 4, tile=tile
        )
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
