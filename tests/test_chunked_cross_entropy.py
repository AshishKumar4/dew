"""The chunked language-model head: the same numbers, no full logits tensor.

The loss this replaces built the whole `[tokens, vocab]` logits tensor, then
read it for the logsumexp, the target logit, the top-1 prediction and again
for the softmax gradient. These tests hold the replacement to the code it
replaced: its losses, log partitions and gradients on par with the whole
logits' from float64 over each site's declared seeds (check 1,
tests/reference_error.py), and its top-1 prediction exactly.

The backward recomputes tiles rather than taping the logits, so the later
sections run it against that same full-vocabulary pass on shapes whose last
token tile and last column tile are both short, and on a Gemma-sized
vocabulary. The two reductions run in different orders, so nothing there is
exact but the predictions, an excluded column's zero gradient, and a float64
head's float64 arithmetic.
"""

import contextlib
import math

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact
from residual_orders import order
from sharded import collectives

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import capped_logits, head_product, rounded_operand, rounded_to, scaled_logits
from dew.nn.protocols import OutputTable
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy, fused_head_loss

EXERCISED = {jnp.float32: 'highest', jnp.bfloat16: 'default'}
"""The matmul precision each compute dtype's head is compared under: bf16's
default, the bf16 product it exists for, and fp32's highest. At the default
an A100 runs each fp32 dot at TF32 or at fp32 as XLA picks for the program,
so a whole head and its tiles would differ by TF32's rounding."""


def true_logits(table, states, dtype):
    """`table`'s logits of `states` (`OutputTable.logits`) for a float64 truth: the states through its
    input quantizer and, under bf16 compute at its default precision, its product as the head rounds
    it, each taken as exact input to the rest, which runs in float64. Called where x64 is enabled."""
    with jax.enable_x64(new_val=False), jax.default_matmul_precision(EXERCISED[dtype]):
        read = table.states(jnp.asarray(states, dtype))
        matrix = jnp.asarray(table.matrix, jnp.float32)
        subscripts = '...d,vd->...v' if table.vocab_major else '...d,dv->...v'
        rounded = (head_product(subscripts, read, matrix, table.precision, compute=table.compute)
                   if dtype == jnp.bfloat16 else None)
    if rounded is None:
        product = jnp.einsum(subscripts, np.asarray(read, np.float64), np.asarray(matrix, np.float64))
    else:
        product = jnp.asarray(np.asarray(rounded), jnp.float64)
    wide = [None if value is None else jnp.asarray(value, jnp.float64) for value in (table.scale, table.bias)]
    return capped_logits(scaled_logits(product, *wide), table.softcap)


def named_gradients(loss, names, argnums=(0, 1)):
    """`loss`'s gradients for its arguments `argnums`, by `names`: one tree for check 1."""
    return lambda *terms: dict(zip(names, jax.grad(loss, argnums=argnums)(*terms), strict=True))


def true_losses(table, states, targets, dtype):
    """The float64 cross entropy of `table`'s logits of `states` (`true_logits`) against `targets`."""
    return optax.softmax_cross_entropy_with_integer_labels(true_logits(table, states, dtype), targets)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('tied', [False, True])
def test_vocabulary_bias_is_fp32_after_tied_or_untied_head_products(dtype, tied):
    """FP32 params keep their vocabulary bias outside the bf16 product, and the chunked head's losses
    over it are on par with the whole logits' over drawn states, biases and targets (check 1)."""
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
            _, predicted, _ = chunked_cross_entropy(
                states, OutputTable(table.matrix, vocab_major=table.vocab_major, bias=bias), ids, 4, tile=tile
            )
            np.testing.assert_array_equal(predicted, logits.argmax(-1))
    assert variables['params']['head_bias'].dtype == jnp.float32
    assert not np.array_equal(np.asarray(bias), np.asarray(bias.astype(jnp.bfloat16), np.float32))

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(5), seed), 3)
        return (jax.random.normal(keys[0], states.shape, dtype), jax.random.normal(keys[1], (97,)) / 17,
                jax.random.randint(keys[2], ids.shape, 0, 97))

    def head(bias):
        return OutputTable(table.matrix, vocab_major=table.vocab_major, bias=bias)

    def tiled(states, bias, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            return {f'{tile}': chunked_cross_entropy(states, head(bias), targets, 4, tile=tile)[0]
                    for tile in (None, (2, 11))}

    def whole(states, bias, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            losses = optax.softmax_cross_entropy_with_integer_labels(head(bias).logits(states), targets)
        return {f'{tile}': losses for tile in (None, (2, 11))}

    def truth(states, bias, targets):
        losses = true_losses(head(bias), states, targets, dtype)
        return {f'{tile}': losses for tile in (None, (2, 11))}

    assert_as_exact(jax.jit(tiled), jax.jit(whole), draw, f'{dtype} {tied} biased head', seeds=SEEDS['bias'],
                    truth_fn=truth)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('softcap', [None, 3.0], ids=['uncapped', 'capped'])
def test_the_fused_head_takes_the_whole_heads_loss_and_gradients_under_the_float64_rule(dtype, softcap):
    """The gradient-in-forward head's weighted sum of losses and z-loss, and
    its gradients for the states, the head, the per-token scale and the bias,
    and its per-token losses and log Z, on par with the whole-logits head's over
    drawn heads and columns (check 1), over chunks whose last is short, padded
    weights and a cap; its top-1 columns are the whole head's."""
    weights = jnp.asarray([[1, 1, 0, 1, 2], [1, 0, 1, 1, 1]], jnp.float32)
    z_loss = 1e-2

    def draw(seed):
        states, matrix, targets = inputs(vocab=37, features=16, tokens=(2, 5), dtype=dtype, seed=seed)
        # Sums exact in any order, so the chunks' products are the whole head's.
        return (*eighths(states, matrix), *columns(37, seed), targets)

    def whole(states, matrix, scale, bias, targets):
        head = OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap, scale=scale)
        logits = head.logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(weights * (losses + z_loss * jnp.square(log_z))), (losses, logits.argmax(-1), log_z)

    def fused(states, matrix, scale, bias, targets):
        return fused_head_loss(states, OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap,
                                                   scale=scale),
                               targets, weights, z_loss=z_loss, tile=(3, 11))

    def held(loss):
        def run(*terms):
            with jax.default_matmul_precision(EXERCISED[dtype]):
                (total, (losses, _, log_z)), gradients = jax.value_and_grad(loss, argnums=(0, 1, 2, 3),
                                                                            has_aux=True)(*terms)
            return dict(zip(('loss', 'losses', 'log Z', 'states', 'head', 'scale', 'bias'),
                            (total, losses, log_z, *gradients), strict=True))

        return run

    assert_as_exact(jax.jit(held(fused)), jax.jit(held(whole)), draw, f'{dtype} {softcap} fused head',
                    seeds=SEEDS['fused'])
    terms = draw(0)
    with jax.default_matmul_precision(EXERCISED[dtype]):
        np.testing.assert_array_equal(fused(*terms)[1][1], whole(*terms)[1][1])


@pytest.mark.parametrize('kind', ['nvfp4', 'fp8'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_a_checkpoints_head_input_quantization_reaches_every_head_that_scores_it(kind, dtype):
    """A model computing under a checkpoint's input quantization of its
    lm_head (ModelOpt's, as `Pretrained` reads it) has its head's input
    rounded by the quantizer in its forward; the table a loss, a scorer and
    a sampler read carries the same quantizer, so the table's logits are the
    forward's and the tiled loss is on par with the forward's logits' cross
    entropy over drawn states and targets (check 1)."""
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

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
        _, predicted, _ = chunked.head_cross_entropy(model, variables, states, ids, 4, tile=(3, 11))
    np.testing.assert_array_equal(predicted, logits.argmax(-1))

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(6), seed))
        return jax.random.normal(keys[0], states.shape, dtype), jax.random.randint(keys[1], ids.shape, 0, 97)

    def tiled(states, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            return chunked.head_cross_entropy(model, variables, states, targets, 4, tile=(3, 11))[0]

    def whole(states, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            return optax.softmax_cross_entropy_with_integer_labels(
                chunked.model_logits(model, variables, states), targets)

    def truth(states, targets):
        return true_losses(table, states, targets, dtype)

    assert_as_exact(jax.jit(tiled), jax.jit(whole), draw, f'{dtype} {kind} quantized head',
                    seeds=SEEDS['quantized'], truth_fn=truth)


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


def columns(vocab: int, seed: int = 0):
    """A per-token scale about 1 and a bias about 0, fp32, as a head's columns hold them past its product."""
    scale, bias = jax.random.split(jax.random.fold_in(jax.random.key(4), seed))
    return (jnp.exp(jax.random.normal(scale, (vocab,), jnp.float32) / 3),
            jax.random.normal(bias, (vocab,), jnp.float32) / 17)


SEEDS = {'bias': 8, 'fused': 16, 'quantized': 8, 'scale and bias': 64, 'vocabulary split': 256,
         'column orders': 8, 'split bias': 8, 'quantized split': 8, 'backbone states': 8, 'backbone': 8,
         'small vocabulary': 8, 'small vocabulary training': 8, 'bounded': 8, 'saturated': 8, 'excluded': 8,
         'token tiles': 8, 'both outputs': 8, 'gemma': 8, 'bf16 cotangent': 8, 'whole logits': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('tile', [None, (3, 11)], ids=['whole', 'tiled'])
def test_the_scale_and_bias_gradients_match_full_logits_under_the_float64_rule(dtype, tile):
    """The per-token scale's and the bias's gradients of the tiled head and of
    the whole-logits head, each taken past the product's rounding, on par with
    the full logits' own over drawn heads and columns (check 1)."""
    # Sums exact in any order: a tile's product is then the whole product on
    # any device, as the oracle takes it, where an fp32 sum's last bit, which
    # a GPU's tile shape picks, can move a bf16 logit a whole bf16 step.
    def draw(seed):
        states, matrix, targets = inputs(vocab=37, features=16, tokens=(2, 5), dtype=dtype, seed=seed)
        return (*eighths(states, matrix), targets, *columns(37, seed))

    def full(states, matrix, targets, scale, bias):
        def loss(scale, bias):
            logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        with jax.default_matmul_precision('default'):
            return dict(zip(('scale', 'bias'), jax.grad(loss, argnums=(0, 1))(scale, bias), strict=True))

    def bounded(states, matrix, targets, scale, bias):
        def loss(scale, bias):
            return chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, bias=bias, scale=scale), targets, 4, tile=tile
            )[0].mean()

        with jax.default_matmul_precision('default'):
            return dict(zip(('scale', 'bias'), jax.grad(loss, argnums=(0, 1))(scale, bias), strict=True))

    def truth(states, matrix, targets, scale, bias):
        # For bf16 the rounded product is exact input to the affine head: the
        # truth isolates its fp32 scale, bias and softmax gradient (`true_logits`).
        def loss(scale, bias):
            head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
            return true_losses(head, states, targets, dtype).mean()

        return dict(zip(('scale', 'bias'), jax.grad(loss, argnums=(0, 1))(scale, bias), strict=True))

    assert_as_exact(jax.jit(bounded), jax.jit(full), draw, f'{dtype} vocabulary gradient',
                    seeds=SEEDS['scale and bias'], truth_fn=truth)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("tile", [None, (2, 2)], ids=["whole", "tiled"])
def test_a_vocabulary_split_head_rounds_a_confident_targets_gradient_as_one_device_does(tile):
    """Under bf16 compute a split head's gradient rounds the target's whole cotangent, its softmax
    share and its one-hot together, once, as one device's head does: a target whose share rounds to
    one in bf16 keeps a gradient, where rounding the share apart from the one-hot would cancel it."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.training import MeshSpec

    states = jnp.ones((2, 1), jnp.float32)
    matrix = jnp.asarray([[8.0, 0.0, 0.0, 0.0]], jnp.float32)
    targets = jnp.zeros((2,), jnp.int32)

    def loss(states, matrix):
        head = OutputTable(matrix, vocab_major=False, compute=jnp.bfloat16)
        losses, _, _ = chunked_cross_entropy(states, head, targets, 1, tile=tile)
        return jnp.sum(losses)

    alone = jax.jit(jax.grad(loss, argnums=1))(states, matrix)
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    with jax.set_mesh(mesh):
        placed = (jax.device_put(states, NamedSharding(mesh, P("fsdp"))),
                  jax.device_put(matrix, NamedSharding(mesh, P(None, "fsdp"))))
        split = jax.jit(jax.grad(loss, argnums=1))(*placed)
    assert float(alone[0, 0]) != 0
    np.testing.assert_array_equal(np.asarray(split), np.asarray(alone))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize('tile', [None, (2, 64)], ids=['whole', 'tiled'])
def test_a_vocabulary_split_head_takes_every_gradient_under_the_float64_rule(tile):
    """Split over the vocabulary on a mesh, the head's loss and its gradients
    for the states, the head, the per-token scale and the bias are as close to
    float64 as one device's whole logits over drawn heads and columns (check 1),
    and no collective carries an array the head's size: each device's columns,
    scale and bias stay in its shard."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.training import MeshSpec

    def draw(seed):
        states, matrix, targets = inputs(vocab=512, features=16, tokens=(4, 5), seed=seed)
        return (*eighths(states, matrix), *columns(512, seed), targets)

    def loss(states, matrix, scale, bias, targets):
        head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
        losses, _, log_z = chunked_cross_entropy(states, head, targets, 4, tile=tile)
        return jnp.sum(losses + 1e-2 * jnp.square(log_z))

    def whole(states, matrix, scale, bias, targets):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses + 1e-2 * jnp.square(log_z))

    gradient = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    specs = (P('fsdp'), P(None, 'fsdp'), P('fsdp'), P('fsdp'), P())

    def split(*terms):
        with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
            placed = [jax.device_put(term, NamedSharding(mesh, spec)) for term, spec in
                      zip(terms, specs, strict=True)]
            return jax.tree.map(np.asarray, gradient(*placed))

    def alone(*terms):
        with jax.default_matmul_precision('highest'):
            return jax.value_and_grad(whole, argnums=(0, 1, 2, 3))(*terms)

    assert_as_exact(split, alone, draw, 'vocabulary-split head', seeds=SEEDS['vocabulary split'])
    terms = draw(0)
    matrix = terms[1]
    with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
        placed = [jax.device_put(term, NamedSharding(mesh, spec))
                  for term, spec in zip(terms, specs, strict=True)]
        program = gradient.lower(*placed).compile().as_text()
    moved = [math.prod(shape) for _, shapes in collectives(program) for shape in shapes]
    assert moved and max(moved) < matrix.size // 2, moved


@pytest.mark.parametrize('tile', [None, (2, 64)], ids=['whole', 'tiled'])
def test_a_heads_terms_and_gradients_are_as_exact_as_logsumexps_over_its_column_orders(tile):
    """One device's head over 512 columns: each token's loss and log Z, and the gradients of a loss
    of both for the states, the head, the scale and the bias, are on par with `jax.nn.logsumexp`'s
    whole logits' from float64 over the orders of its columns (check 1 over the orbit: the targets
    relabelled, which reorders every row's sum of exponentials and moves no value). A row summed in
    one loop of XLA's CPU reduction with a combine of its own, and a share divided out and then
    scaled, held the whole head's state gradient 3.9 times as far."""
    states, matrix, targets = inputs(vocab=512, features=16, tokens=(4, 5))
    states, matrix = eighths(states, matrix)
    scale, bias = columns(512)

    def draw(seed):
        columns_in = order(512, seed)
        relabel = np.argsort(columns_in)
        return states, matrix[:, columns_in], scale[columns_in], bias[columns_in], relabel[targets]

    def terms(states, matrix, scale, bias, targets):
        head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
        losses, _, log_z = chunked_cross_entropy(states, head, targets, 4, tile=tile)
        return losses, log_z

    def whole(states, matrix, scale, bias, targets):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        return log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0], log_z

    def scored(head):
        def loss(*arguments):
            losses, log_z = head(*arguments)
            return jnp.sum(losses + 1e-2 * jnp.square(log_z)), (losses, log_z)

        def run(*arguments):
            with jax.default_matmul_precision('highest'):
                gradients, (losses, log_z) = jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(*arguments)
            return dict(zip(('losses', 'log Z', 'states', 'head', 'scale', 'bias'),
                            (losses, log_z, *gradients), strict=True))

        return run

    assert_as_exact(jax.jit(scored(terms)), jax.jit(scored(whole)), draw, f'{tile} column orders',
                    seeds=SEEDS['column orders'])


def split_scores(model, wide_model, shape, vocab: int, objective_of, adjusted, label: str, seeds: int):
    """A vocabulary-split objective's (`objective_of(model, variables)`) token losses on a two-device
    mesh on par with one device's over drawn variables and `shape` tokens of `vocab` (check 1), the
    truth the objective of
    `wide_model`, `model` in float64, on them widened; its top-1 hits at the first draw one device's.
    A draw's variables are `model`'s initial ones as `adjusted` sets them. Returns the mesh, and the
    first draw's variables and batch placed on it."""
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import shard_batch

    def draw(seed):
        tokens = jax.random.randint(jax.random.fold_in(jax.random.key(1), seed), shape, 0, vocab)
        return adjusted(model.init(jax.random.fold_in(jax.random.key(0), seed), tokens[:, :-1])), tokens

    variables, _ = draw(0)
    objective, wide = objective_of(model, variables), objective_of(wide_model, variables)
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    shardings = Layout(min_shard=1).shardings(mesh, variables)
    scored = jax.jit(objective.token_scores)

    def placed(variables, tokens):
        return jax.device_put(variables, shardings), shard_batch(mesh, {'text': np.asarray(tokens)})['text']

    def split(variables, tokens):
        with jax.set_mesh(mesh):
            return scored(*placed(variables, tokens))

    def alone(variables, tokens):
        wider = jax.tree.leaves(variables)[0].dtype == np.float64
        return (wide if wider else objective).token_scores(variables, tokens)

    assert_as_exact(lambda *drawn: split(*drawn).losses, jax.jit(lambda *drawn: alone(*drawn).losses), draw,
                    label, seeds=seeds)
    np.testing.assert_array_equal(split(*draw(0)).correct, alone(*draw(0)).correct)
    return mesh, *placed(*draw(0))


@pytest.mark.mesh(devices=2)
def test_vocabulary_split_slices_the_bias_with_its_table_columns():
    from dew.objectives.lm import LMObjective

    def model(dtype):
        return CausalTransformer(vocab_size=128, emb_features=8, num_layers=1, num_heads=2,
                                 mlp_features=16, max_seq_len=4, head_bias=True,
                                 qk_norm=False, attention_impl='reference', dtype=dtype)

    def objective(model, variables):
        return LMObjective(model, seq_len=4, variables=variables, head_chunks=4, head_tile=(2, 11),
                           ema_decay=None)

    def spiked(variables):
        variables['params']['head_bias'] = variables['params']['head_bias'].at[117].set(13.027)
        return variables

    split_scores(model(jnp.float32), model(jnp.float64), (4, 5), 128, objective, spiked,
                 'vocabulary-split bias', SEEDS['split bias'])


@pytest.mark.mesh(devices=2)
def test_a_quantized_split_head_scores_as_one_device_and_moves_no_head_sized_array():
    """A head whose input a checkpoint quantizes, split over the vocabulary on
    a mesh, scores each token on par with one device (`split_scores`), and no
    collective of the step carries an array the head's size: tokens and
    per-token terms cross devices, the head stays in its shards."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.objectives.lm import LMObjective
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

    def quantized(dtype):
        return InputQuantization(format='modelopt', sites=(('lm_head', 'fp8'),)).apply(CausalTransformer(
            vocab_size=512, emb_features=16, num_layers=1, num_heads=2, mlp_features=16, max_seq_len=4,
            tie_embeddings=False, qk_norm=False, attention_impl='reference', dtype=dtype))

    def objective(model, variables):
        return LMObjective(model, seq_len=4, variables=variables, head_chunks=4, head_tile=(2, 64),
                           ema_decay=None)

    def scaled(variables):
        variables[QUANTIZATION]['lm_head'][INPUT_SCALES['fp8']] = jnp.asarray(0.02, jnp.float32)
        return variables

    model = quantized(jnp.float32)
    tokens = jnp.arange(20, dtype=jnp.int32).reshape(4, 5)
    mesh, placed, _ = split_scores(model, quantized(jnp.float64), (4, 5), 512, objective, scaled,
                                   'quantized vocabulary split', SEEDS['quantized split'])
    variables = jax.device_get(placed)
    with jax.set_mesh(mesh):
        states = jax.device_put(model.apply(variables, tokens[:, :-1], method='hidden_states'),
                                NamedSharding(mesh, P('fsdp')))
        targets = jax.device_put(tokens[:, 1:], NamedSharding(mesh, P('fsdp')))

        def loss(params, states):
            return chunked.head_cross_entropy(model, {**placed, 'params': params}, states, targets, 4,
                                              tile=(2, 64))[0].sum()

        # The head's forward and backward alone: the embedding's own gather is not the head's.
        program = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(placed['params'], states).compile().as_text()
    head = 512 * 16
    moved = [math.prod(shape) for _, shapes in collectives(program) for shape in shapes]
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
    """Real hidden states and a real head, tied and untied: the chunked loss of the backbone's states
    is on par with the cross entropy of the model's own head's logits of them over drawn tokens and
    targets (check 1), under bf16 and fp32 compute, and picks the model's top-1."""
    model = small_model(tie_embeddings=tie_embeddings, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    table = OutputTable(head_matrix(model, variables), vocab_major=False)

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(1), seed))
        drawn = jax.random.randint(keys[0], (2, 12), 0, 97)
        return (model.apply(variables, drawn, method=CausalTransformer.hidden_states),
                jax.random.randint(keys[1], (2, 12), 0, 97))

    def tiled(hidden, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            return chunked_cross_entropy(hidden, table, targets, chunks)[0]

    def whole(hidden, targets):
        with jax.default_matmul_precision(EXERCISED[dtype]):
            return optax.softmax_cross_entropy_with_integer_labels(
                chunked.model_logits(model, variables, hidden), targets)

    assert_as_exact(jax.jit(tiled), jax.jit(whole), draw,
                    f'{dtype} {tie_embeddings} {chunks} backbone states', seeds=SEEDS['backbone states'],
                    truth_fn=lambda hidden, targets: true_losses(table, hidden, targets, dtype))
    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    _, predicted, _ = chunked_cross_entropy(hidden, table, ids, chunks)
    assert hidden.dtype == dtype
    assert jnp.array_equal(predicted, jnp.argmax(model.apply(variables, ids), axis=-1))


def test_the_gradient_reaches_the_backbone_through_the_states_and_the_head():
    """Both paths a tied head has, the trunk and the embedding table: every parameter's gradient
    through the chunked head is on par with the whole logits' over drawn models, tokens and targets
    (check 1), the truth the model in float64, and none is zero, so each is checked.

    The reference takes the logits from the operands the head multiplies,
    rounded to bf16, at the highest precision, and rounds them to bf16 as the
    chunked head does, so its backward rounds the logits' cotangent to bf16
    once, as the chunked head's does.
    """
    model, wide = small_model(), small_model(dtype=jnp.float64)

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(0), seed), 3)
        ids = jax.random.randint(keys[0], (2, 12), 0, 97)
        return model.init(keys[1], ids)['params'], ids, jax.random.randint(keys[2], (2, 12), 0, 97)

    def full(params, ids, targets):
        hidden = model.apply({'params': params}, ids, method=CausalTransformer.hidden_states)
        head = head_matrix(model, {'params': params})
        operands = (rounded_operand(value.astype(jnp.float32), jnp.bfloat16) for value in (hidden, head))
        logits = rounded_to(jnp.einsum('btd,dv->btv', *operands, precision=jax.lax.Precision.HIGHEST),
                            jnp.bfloat16)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    def tiled(params, ids, targets):
        hidden = model.apply({'params': params}, ids, method=CausalTransformer.hidden_states)
        table = OutputTable(head_matrix(model, {'params': params}), vocab_major=False)
        return jnp.mean(chunked_cross_entropy(hidden, table, targets, 4)[0])

    def true(params, ids, targets):
        logits = wide.apply({'params': params}, ids)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    assert_as_exact(jax.jit(jax.grad(tiled)), jax.jit(jax.grad(full)), draw, 'backbone gradient',
                    seeds=SEEDS['backbone'], truth_fn=jax.grad(true))
    for path, leaf in jax.tree_util.tree_leaves_with_path(jax.grad(full)(*draw(0))):
        name = jax.tree_util.keystr(path)
        assert jnp.abs(leaf).max() > 0, f"{name} has a zero gradient, so nothing is checked"


# --- the chunk arithmetic itself -------------------------------------------

@pytest.mark.parametrize("chunks", [0, -1])
def test_nonpositive_chunk_counts_are_refused(chunks):
    hidden, head, targets = inputs()
    with pytest.raises(ValueError):
        chunked_cross_entropy(hidden, OutputTable(head, vocab_major=False), targets, chunks)


@pytest.mark.parametrize("chunks", [4, 8])
@pytest.mark.parametrize("tile", [None, (3, 2)])
def test_small_vocabularies_take_any_positive_chunk_count(chunks, tile):
    """A five-column head's short tail and one-column tiles score every column and gradient: on par
    with the whole logits over drawn heads (check 1), and the whole logits' top-1."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=5, features=8, tokens=(2, 4), seed=seed)
        return hidden, head, jax.random.normal(jax.random.fold_in(jax.random.key(3), seed), (5,)) / 7, targets

    def tiled(hidden, head, bias, targets):
        losses, predicted, log_z = chunked_cross_entropy(
            hidden, OutputTable(head, vocab_major=False, bias=bias, softcap=3.),
            targets, chunks, tile=tile)
        return jnp.sum(losses + .01 * jnp.square(log_z)), (losses, predicted, log_z)

    def whole(hidden, head, bias, targets):
        logits = OutputTable(head, vocab_major=False, bias=bias, softcap=3.).logits(hidden)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(losses + .01 * jnp.square(log_z)), (losses, logits.argmax(-1), log_z)

    def held(loss):
        def run(*terms):
            with jax.default_matmul_precision("highest"):
                (total, (losses, _, log_z)), gradients = jax.value_and_grad(loss, argnums=(0, 1, 2),
                                                                            has_aux=True)(*terms)
            return dict(zip(("loss", "losses", "log Z", "hidden", "head", "bias"), (total, losses, log_z,
                                                                                   *gradients), strict=True))
        return run

    assert_as_exact(jax.jit(held(tiled)), jax.jit(held(whole)), draw, f"{chunks} {tile} small vocabulary",
                    seeds=SEEDS["small vocabulary"])
    with jax.default_matmul_precision("highest"):
        np.testing.assert_array_equal(tiled(*draw(0))[1][1], whole(*draw(0))[1][1])


@pytest.mark.parametrize("chunks", [4, 8])
def test_lm_training_with_small_vocabularies_matches_the_unchunked_head(chunks):
    """The objective's chunk count reaches the tiled head in both the loss and optimizer update: each
    on par with the unchunked head's over drawn batches (check 1), the truth the model in float64."""
    from reference_error import widened

    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=5, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8, head_bias=True, attention_impl="xla")
    objective = LMObjective(model, seq_len=4, head_chunks=chunks, head_tile=(3, 2), ema_decay=None)
    trainer = Trainer(objective, optax.sgd(.01), key=jax.random.key(0))
    initial = trainer.initial_state()
    step = Step(jnp.array(0), jax.random.key(1), None)

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(2), seed)
        return (jax.random.randint(key, (jax.device_count(), 5), 0, 5),)

    compiled = trainer.compile(initial, {"text": draw(0)[0]})
    loss = jax.jit(lambda ids: objective.scalar_loss(initial.variables, {"text": ids}, step)[0])

    def trained(ids):
        state, _, _, _, accepted, _ = compiled(jax.tree.map(jnp.copy, initial), {"text": ids})
        assert bool(accepted) and int(state.updates) == 1
        return {"loss": loss(ids), "params": state.variables["params"]}

    def stepped(model, variables):
        def whole(params, ids):
            logits = model.apply({**variables, "params": params}, ids[:, :-1])
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]))

        def run(ids):
            want, gradient = jax.value_and_grad(whole)(variables["params"], ids)
            updated = jax.tree.map(lambda p, g: p - .01 * g, variables["params"], gradient)
            return {"loss": want, "params": updated}
        return run

    assert_as_exact(trained, jax.jit(stepped(model, initial.variables)), draw,
                    f"{chunks} small vocabulary step", seeds=SEEDS["small vocabulary training"],
                    truth_fn=stepped(model.clone(dtype=jnp.float64), widened(initial.variables)))


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
    """Both outputs' gradients, on par with the pass that holds all the logits over drawn heads
    (check 1); its top-1 the full pass's, and neither short tile's gradient dropped."""
    precision = jax.lax.Precision.HIGHEST

    def draw(seed):
        return inputs(vocab=RAGGED_VOCAB, features=7, tokens=(1, RAGGED_TOKENS), seed=seed)

    def full(states, matrix, targets):
        losses, log_z = oracle(states, matrix, targets, softcap, precision)
        return jnp.sum(loss_weight * losses + z_weight * jnp.square(log_z))

    def tiled(states, matrix, targets):
        losses, _, log_z = chunked_cross_entropy(
            states,
            OutputTable(matrix, vocab_major=False, softcap=softcap, precision=precision),
            targets,
            4,
            tile=RAGGED,
        )
        return jnp.sum(loss_weight * losses + z_weight * jnp.square(log_z))

    def held(loss):
        def run(*terms):
            total, gradients = jax.value_and_grad(loss, argnums=(0, 1))(*terms)
            return dict(zip(("loss", "hidden", "head"), (total, *gradients), strict=True))
        return run

    assert_as_exact(jax.jit(held(tiled)), jax.jit(held(full)), draw,
                    f"{loss_weight} {z_weight} {softcap} bounded", seeds=SEEDS["bounded"])
    hidden, head, targets = draw(0)
    _, predicted, _ = chunked_cross_entropy(
        hidden,
        OutputTable(head, vocab_major=False, softcap=softcap, precision=precision),
        targets,
        4,
        tile=RAGGED,
    )
    _, expected_top1 = reference(hidden, head, targets, softcap=softcap)
    np.testing.assert_array_equal(predicted, expected_top1)
    got = held(tiled)(hidden, head, targets)
    assert jnp.abs(got["head"][:, RAGGED[1]:]).max() > 0, "the short column tile was dropped"
    assert jnp.abs(got["hidden"][:, RAGGED[0]:]).max() > 0, "the short token tile was dropped"


def test_the_softcap_gradient_survives_saturated_logits():
    """Logits well past the cap: `tanh` still has to pass a gradient back, on par with the full pass's
    for the states, the head and the cap over drawn heads (check 1), and the top-1 the full pass's."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=97, features=8, tokens=(1, 9), seed=seed)
        return hidden * 8, head * 4, targets, jnp.float32(30.)

    def full(states, matrix, targets, cap):
        losses, log_z = oracle(states, matrix, targets, cap)
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    def tiled(states, matrix, targets, cap):
        losses, _, log_z = chunked_cross_entropy(
            states, OutputTable(matrix, vocab_major=False, softcap=cap), targets, 4, tile=RAGGED
        )
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    def held(loss):
        return named_gradients(loss, ("hidden", "head", "softcap"), (0, 1, 3))

    assert_as_exact(jax.jit(held(tiled)), jax.jit(held(full)), draw, "saturated softcap",
                    seeds=SEEDS["saturated"])
    hidden, head, targets, _ = draw(0)
    _, predicted, _ = chunked_cross_entropy(
        hidden, OutputTable(head, vocab_major=False, softcap=30.0), targets, 4, tile=RAGGED
    )
    _, expected_top1 = reference(hidden, head, targets, softcap=30.)
    np.testing.assert_array_equal(predicted, expected_top1)


@pytest.mark.parametrize("tile", [RAGGED, None], ids=["tiled", "whole"])
@pytest.mark.parametrize("excluded", [0, 41, RAGGED_VOCAB - 1], ids=["first", "inside", "last"])
def test_an_excluded_column_takes_no_mass_in_the_loss_the_prediction_or_the_gradient(tile, excluded):
    """The excluded column leaves the partition and the prediction, in
    whichever tile holds it: the losses, log partitions and both gradients
    are on par with the full pass's with that column's logits at -inf over
    drawn heads (check 1), the predictions are its, and the excluded head
    column receives no gradient."""
    precision = jax.lax.Precision.HIGHEST

    def draw(seed):
        hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=7, tokens=(1, RAGGED_TOKENS), seed=seed)
        # The excluded column scores highest everywhere, so a prediction that kept it would show.
        return (hidden, head.at[:, excluded].set(jnp.abs(head).max(axis=1) * 4),
                jnp.where(targets == excluded, (excluded + 1) % RAGGED_VOCAB, targets))

    def full(states, matrix, targets):
        logits = OutputTable(matrix, vocab_major=False, precision=precision).logits(states)
        logits = logits.at[..., excluded].set(-jnp.inf)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses + 0.7 * jnp.square(log_z)), (losses, log_z, jnp.argmax(logits, axis=-1))

    def chunked_pass(states, matrix, targets):
        losses, predicted, log_z = chunked_cross_entropy(
            states,
            OutputTable(matrix, vocab_major=False, precision=precision),
            targets,
            4,
            tile=tile,
            excluded=excluded,
        )
        return jnp.sum(losses + 0.7 * jnp.square(log_z)), (losses, log_z, predicted)

    def held(loss):
        def run(*terms):
            (_, (losses, log_z, _)), gradients = jax.value_and_grad(loss, argnums=(0, 1),
                                                                    has_aux=True)(*terms)
            return dict(zip(("losses", "log Z", "hidden", "head"), (losses, log_z, *gradients), strict=True))
        return run

    assert_as_exact(jax.jit(held(chunked_pass)), jax.jit(held(full)), draw, f"{tile} {excluded} excluded",
                    seeds=SEEDS["excluded"])
    terms = draw(0)
    (_, got), got_grads = jax.jit(jax.value_and_grad(chunked_pass, argnums=(0, 1), has_aux=True))(*terms)
    np.testing.assert_array_equal(got[2], full(*terms)[1][2])
    assert not jnp.any(got[2] == excluded)
    assert jnp.all(got_grads[1][:, excluded] == 0)


@pytest.mark.parametrize("path", ["tiled", "whole", "fused"])
def test_a_target_at_the_excluded_column_scores_infinity_and_moves_no_logit(path):
    """A target at the excluded column scores +inf, a constant, in every head: its token's loss
    moves no logit, so the head's excluded column takes no gradient from it, and the other
    columns' gradient is the partition's alone."""
    excluded = 41
    hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=7, tokens=(1, RAGGED_TOKENS))
    targets = targets.at[0, 0].set(excluded)
    precision = jax.lax.Precision.HIGHEST

    def scored(states, matrix):
        table = OutputTable(matrix, vocab_major=False, precision=precision)
        if path == "fused":
            total, (losses, _, _) = fused_head_loss(states, table, targets, jnp.ones(targets.shape),
                                                    z_loss=1.0, tile=RAGGED, excluded=excluded)
            return total, (losses, total)
        losses, _, log_z = chunked_cross_entropy(states, table, targets, 4,
                                                 tile=RAGGED if path == "tiled" else None, excluded=excluded)
        # The sum is +inf; each token's loss, the excluded target's included, takes a cotangent of 1.
        total = jnp.sum(losses + log_z)
        return total, (losses, total)

    (_, (losses, total)), (_, d_matrix) = jax.value_and_grad(scored, argnums=(0, 1), has_aux=True)(
        hidden, head)
    assert jnp.isinf(losses[0, 0]) and losses[0, 0] > 0
    assert jnp.isinf(total) and total > 0
    assert jnp.all(jnp.isfinite(d_matrix))
    assert jnp.all(d_matrix[:, excluded] == 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_head_gradient_accumulates_every_token_tile_before_it_rounds(dtype):
    """257 tokens is three tiles; rounding per tile would lose the last ones.

    The gradients, in `dtype`, are on par over drawn heads (check 1) with an
    fp32 oracle's on the same values rounded to `dtype` once, at the end,
    which rounding each tile's partial sum adds to. The products run at
    HIGHEST, so the logits' cotangent is not rounded to bf16 inside them.
    """
    precision = jax.lax.Precision.HIGHEST

    def draw(seed):
        hidden, head, targets = inputs(vocab=17, features=7, tokens=(257,), dtype=dtype, seed=seed)
        return hidden, head.astype(dtype), targets

    def full(states, matrix, targets):
        return jnp.mean(oracle(states, matrix, targets, precision=precision)[0])

    def tiled(states, matrix, targets):
        return jnp.mean(
            chunked_cross_entropy(
                states, OutputTable(matrix, vocab_major=False, precision=precision), targets, 4, tile=RAGGED
            )[0]
        )

    def once(states, matrix, targets):
        wide = jax.grad(full, argnums=(0, 1))(states.astype(jnp.float32), matrix.astype(jnp.float32), targets)
        return dict(zip(("hidden", "head"), (gradient.astype(dtype) for gradient in wide), strict=True))

    def held(loss):
        return named_gradients(loss, ("hidden", "head"))

    assert_as_exact(jax.jit(held(tiled)), jax.jit(once), draw, f"{dtype} token tiles",
                    seeds=SEEDS["token tiles"], truth_fn=held(full))
    got = held(tiled)(*draw(0))
    assert got["hidden"].dtype == got["head"].dtype == dtype


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_the_backbone_receives_both_outputs_gradients(tie_embeddings):
    """Every parameter's gradient of a loss of both outputs through the tiled head, on par with the
    whole logits' over drawn models and tokens (check 1), the truth the model in float64."""
    model, wide = (small_model(dtype=dtype, tie_embeddings=tie_embeddings, final_logit_softcap=30.)
                   for dtype in (jnp.float32, jnp.float64))

    def draw(seed):
        ids = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(31), seed), (1, 7), 0, 97)
        return model.init(jax.random.fold_in(jax.random.PRNGKey(32), seed), ids)["params"], ids

    def full(tree, ids):
        targets = jnp.roll(ids, -1, axis=-1)
        wider = jax.tree.leaves(tree)[0].dtype == jnp.float64
        logits = (wide if wider else model).apply({"params": tree}, ids)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.mean(log_z - picked + 0.02 * jnp.square(log_z))

    def tiled(tree, ids):
        states = model.apply({"params": tree}, ids,
                             method=CausalTransformer.hidden_states)
        losses, _, log_z = chunked_cross_entropy(
            states,
            OutputTable(head_matrix(model, {"params": tree}), vocab_major=False, softcap=30.0),
            jnp.roll(ids, -1, axis=-1),
            4,
        )
        return jnp.mean(losses + 0.02 * jnp.square(log_z))

    assert_as_exact(jax.jit(jax.grad(tiled)), jax.jit(jax.grad(full)), draw, f"{tie_embeddings} both outputs",
                    seeds=SEEDS["both outputs"])


# --- a production vocabulary --------------------------------------------------

# Gemma's 262,144 columns. What changes with the width is the length of the
# reductions: the partition sums 262,144 exponentials and the hidden
# gradient 262,144 products.
GEMMA_VOCAB = 262_144
BOUNDARIES = jnp.asarray([0, 65535, 65536, 131071, 131072, 196608, 262143], jnp.int32)
"""Targets on each side of the 16-chunk head's chunk boundaries, and at its first and last columns."""


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap-30"])
def test_a_262144_vocabulary_computes_in_float64_and_picks_optaxs_top1_on_cpu(softcap):
    """Seven targets spanning chunk boundaries: a float64 head makes no float32 value in its
    forward or its backward, so it is the float64 truth the gate's other heads are measured from,
    and its top-1 columns are Optax's full-vocabulary logits'."""
    rng = np.random.default_rng(713)
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        hidden = jnp.asarray(rng.normal(size=(7, 5)), jnp.float64)
        head = jnp.asarray(rng.normal(size=(5, GEMMA_VOCAB)), jnp.float64)
        weights = jnp.asarray(rng.uniform(.5, 1.5, 7), jnp.float64)

        def tiled(states, matrix):
            table = OutputTable(matrix, vocab_major=False, softcap=softcap)
            losses, predictions, log_z = chunked_cross_entropy(states, table, BOUNDARIES, 16, tile=(3, 8192))
            return jnp.sum(weights * losses), (losses, log_z, predictions)

        graph = jax.make_jaxpr(jax.value_and_grad(tiled, (0, 1), has_aux=True))(hidden, head)
        assert float32_values(graph.jaxpr) == 0
        logits = hidden @ head
        if softcap is not None:
            logits = softcap * jnp.tanh(logits / softcap)
        np.testing.assert_array_equal(jax.jit(tiled)(hidden, head)[1][2], logits.argmax(-1).astype(jnp.int32))


TILE = (1024, 8192)
"""The tiled head's tile on the Gemma-sized vocabulary: 1024 tokens by 8192 columns."""


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap-30"])
@pytest.mark.parametrize("chunks", [4, 16], ids=["4-chunks", "16-chunks"])
def test_a_gemma_sized_vocabulary_is_as_exact_as_the_whole_logits(chunks, softcap):
    """Losses, log partitions and the gradients of both the states and the head of a weighted loss,
    chunked, on par with the full-vocabulary pass's from float64 over drawn heads and targets, the
    first seven on the chunks' boundaries (check 1). The head's logits have a standard deviation of
    3, so the partition is carried by a few hundred columns scattered over every chunk rather than
    by all of them equally or by one."""
    tokens, features = 32, 64

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(7), seed), 4)
        targets = jax.random.randint(keys[2], (tokens,), 0, GEMMA_VOCAB).at[:len(BOUNDARIES)].set(BOUNDARIES)
        return (jax.random.normal(keys[0], (tokens, features), jnp.float32),
                3 / np.sqrt(features) * jax.random.normal(keys[1], (features, GEMMA_VOCAB), jnp.float32),
                targets, jax.random.uniform(keys[3], (tokens,), jnp.float32, 0.5, 1.5))

    def scored(chunks):
        def run(hidden, head, targets, weights):
            def loss(states, matrix):
                if chunks is None:
                    losses, log_z = oracle(states, matrix, targets, softcap)
                else:
                    losses, _, log_z = chunked_cross_entropy(
                        states, OutputTable(matrix, vocab_major=False, softcap=softcap), targets, chunks,
                        tile=TILE)
                return jnp.sum(weights * losses), (losses, log_z)

            (_, (losses, log_z)), (d_hidden, d_head) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
                hidden, head)
            return {"losses": losses, "log Z": log_z, "hidden": d_hidden, "head": d_head}
        return jax.jit(run)

    assert_as_exact(scored(chunks), scored(None), draw, f"{chunks} {softcap} gemma vocabulary",
                    seeds=SEEDS["gemma"])


@pytest.mark.parametrize("tile", [None, RAGGED])
def test_the_bf16_head_rounds_its_logits_and_their_cotangent_once(tile):
    """Under the bf16 algorithm the head takes torch autocast's rounding: the
    logits are bf16 values, and their cotangent is rounded to bf16 once
    before both products. Its gradients are on par over drawn heads (check 1)
    with a reference that rounds the logits with `rounded_to`, whose
    derivative rounds the cotangent where the value rounds, and multiplies at
    the highest precision; on eighths both form the same logits, and the
    truth takes their bf16 values as exact, its cotangent unrounded. Kept in
    fp32 (`highest`) the cotangent is another computation."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=16, tokens=(RAGGED_TOKENS,), seed=seed)
        hidden, head = eighths(hidden, head)
        return hidden, head.T, targets

    def strict_or_rounded(round_logits):
        def loss(states, matrix, targets):
            logits = jnp.einsum('td,vd->tv', states, matrix, precision=jax.lax.Precision.HIGHEST)
            if round_logits:
                logits = rounded_to(logits, jnp.bfloat16)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
        return loss

    def chunked_loss(states, matrix, targets):
        table = OutputTable(matrix.T, vocab_major=False, precision=chunked.BF16)
        return jnp.mean(chunked_cross_entropy(states, table, targets, 4, tile=tile)[0])

    def true(states, matrix, targets):
        logits = jnp.einsum('td,vd->tv', states, matrix)
        # The logits' bf16 values, exact on eighths, their cotangent passed through unrounded.
        rounded = jnp.asarray(logits, jnp.bfloat16).astype(logits.dtype)
        logits = logits + jax.lax.stop_gradient(rounded - logits)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    def held(loss):
        return named_gradients(loss, ("states", "table"))

    assert_as_exact(jax.jit(held(chunked_loss)), jax.jit(held(strict_or_rounded(round_logits=True))), draw,
                    f"{tile} bf16 cotangent", seeds=SEEDS["bf16 cotangent"], truth_fn=held(true))
    terms = draw(0)
    strict = held(strict_or_rounded(round_logits=False))(*terms)
    assert not np.array_equal(strict["states"], held(chunked_loss)(*terms)["states"])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_the_whole_logits_head_computes_what_the_tiled_one_does(dtype, softcap):
    """`tile=None` keeps the logits for the backward instead of recomputing
    them tile by tile. The same predictions, and losses, log partitions and
    gradients on par with the tiled head's over drawn heads (check 1): the
    logits' cotangent stays fp32 into the state product either way, and the
    head's gradient accumulates in fp32 once."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=RAGGED_VOCAB, features=16, tokens=(RAGGED_TOKENS,), dtype=dtype,
                                       seed=seed)
        # Targets outside the vocabulary pick no column, which the
        # vocabulary-split head relies on for every shard but the owner.
        outside = targets.at[:2].set(jnp.asarray([-1, RAGGED_VOCAB + 5], targets.dtype))
        return hidden, head, outside, jnp.float32(30.)

    def run(tile):
        def scored(hidden, head, targets, cap):
            def loss(states, matrix, cap):
                table = OutputTable(matrix, vocab_major=False, softcap=softcap and cap)
                losses, _, log_z = chunked_cross_entropy(states, table, targets, 4, tile=tile)
                return jnp.mean(losses) + 0.01 * jnp.mean(jnp.square(log_z)), (losses, log_z)

            # fp32 products at full precision, so a backend whose default fp32 product is TF32 runs
            # both heads alike.
            highest = dtype == jnp.float32
            with jax.default_matmul_precision("highest") if highest else contextlib.nullcontext():
                gradients, (losses, log_z) = jax.grad(loss, argnums=(0, 1, 2) if softcap else (0, 1),
                                                      has_aux=True)(hidden, head, cap)
            names = ("losses", "log Z", "hidden", "head", "softcap")[:2 + len(gradients)]
            return dict(zip(names, (losses, log_z, *gradients), strict=True))
        return scored

    assert_as_exact(jax.jit(run(RAGGED)), jax.jit(run(None)), draw, f"{dtype} {softcap} whole logits",
                    seeds=SEEDS["whole logits"])
    hidden, head, targets, _ = draw(0)
    table = OutputTable(head, vocab_major=False, softcap=softcap)
    tiled, whole = (chunked_cross_entropy(hidden, table, targets, 4, tile=tile) for tile in (RAGGED, None))
    assert jnp.array_equal(tiled[1], whole[1])
