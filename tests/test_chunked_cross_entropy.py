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

import functools
import math

import jax
import jax.extend
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest
from confident_support import confident, label_derivative, log_partition, log_softmax
from reference_error import as_given, assert_as_exact, seeds_for, with_gradients
from residual_orders import order
from sharded import collectives

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import capped_logits, rounded_operand, rounded_to, scaled_logits
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
    input quantizer, taken as exact, and the rest in float64; under bf16 compute, which multiplies bf16
    operands and rounds the logits to bf16 (`head_product`), the exact product of the bf16 operands
    rounded to bf16 once, as each head's one product (`bf16_product`) is, its own order aside. Called
    where x64 is enabled."""
    with jax.enable_x64(new_val=False):
        read = np.asarray(table.states(jnp.asarray(states, dtype)))
    operands = [np.asarray(value, np.float64) for value in (read, table.matrix)]
    if dtype == jnp.bfloat16:
        operands = [np.asarray(value.astype(ml_dtypes.bfloat16), np.float64) for value in operands]
    product = jnp.einsum('...d,vd->...v' if table.vocab_major else '...d,dv->...v', *operands)
    if dtype == jnp.bfloat16:
        product = jnp.asarray(np.asarray(product).astype(ml_dtypes.bfloat16), jnp.float64)
    wide = [None if value is None else jnp.asarray(value, jnp.float64) for value in (table.scale, table.bias)]
    return capped_logits(scaled_logits(product, *wide), table.softcap)


def pulled(fn, count: int = 2, *, has_aux: bool = False):
    """`fn` as `run(*arguments, cotangent)`: its output and the gradients of its first `count`
    arguments under the cotangent the draw passes last (`with_gradients`), the rest (targets, weights)
    held as given. Where `has_aux`, `fn` returns its differentiated output and values beside it that
    carry no gradient (a fused head's per-token terms), which `run` returns after."""
    def run(*arguments):
        *arguments, cotangent = arguments
        held = arguments[count:]
        pulled_back = with_gradients(lambda *taken: fn(*taken, *held), has_aux=has_aux)(
            *arguments[:count], cotangent)
        return (pulled_back[:-1], pulled_back[-1]) if has_aux else pulled_back
    return run


def with_cotangent(draw, fn, *, has_aux: bool = False):
    """`draw` with a cotangent for `fn`'s differentiated output on the drawn terms appended (`pulled`):
    standard normal in each output's dtype, from the seed."""
    def drawn(seed):
        terms = draw(seed)
        shapes = jax.eval_shape(fn, *terms)
        rng = np.random.default_rng([0xC07, seed])
        cotangent = jax.tree.map(lambda leaf: jnp.asarray(rng.standard_normal(leaf.shape), leaf.dtype),
                                 shapes[0] if has_aux else shapes)
        return (*terms, cotangent)
    return drawn


def under(precision: str, run):
    """`run` under the matmul precision `precision`."""
    def held(*arguments):
        with jax.default_matmul_precision(precision):
            return run(*arguments)
    return held


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

    @as_given
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
    z_loss = 1e-2

    def draw(seed):
        states, matrix, targets = inputs(vocab=37, features=16, tokens=(8, 16), dtype=dtype, seed=seed)
        # Padded tokens weigh nothing, others one or two.
        weights = jax.random.randint(jax.random.fold_in(jax.random.key(8), seed), targets.shape, 0, 3)
        # Sums exact in any order, so the chunks' products are the whole head's.
        return (*eighths(states, matrix), *columns(37, seed), targets, weights.astype(jnp.float32))

    def whole(states, matrix, scale, bias, targets, weights):
        head = OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap, scale=scale)
        logits = head.logits(states)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(weights * (losses + z_loss * jnp.square(log_z))), (losses, logits.argmax(-1), log_z)

    def fused(states, matrix, scale, bias, targets, weights):
        return fused_head_loss(states, OutputTable(matrix, vocab_major=False, bias=bias, softcap=softcap,
                                                   scale=scale),
                               targets, weights, z_loss=z_loss, tile=(3, 11))

    def held(loss):
        # Only the sum carries a gradient (`fused_head_loss`); each token's loss and log Z beside it.
        def summed(*terms):
            total, (losses, _, log_z) = loss(*terms)
            return total, (losses, log_z)
        return under(EXERCISED[dtype], pulled(summed, 4, has_aux=True))

    assert_as_exact(jax.jit(held(fused)), jax.jit(held(whole)),
                    with_cotangent(draw, lambda *terms: whole(*terms)[0]), f'{dtype} {softcap} fused head',
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

    @as_given
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


SEEDS = {'projected head': 32,
         'bias': 128, 'fused': 512, 'quantized': 64, 'scale and bias': 128, 'vocabulary split': 512,
         'column orders': 32, 'split bias': 32, 'quantized split': 32, 'backbone states': 16, 'backbone': 128,
         'small vocabulary': 128, 'small vocabulary training': 128, 'bounded': 32, 'saturated': 256,
         'excluded': 32, 'token tiles': 16, 'both outputs': 64, 'gemma': 256, 'bf16 cotangent': 256,
         'whole logits': 512}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on each platform the site
runs on, the most any of its cases took: on armada, and paired (Dew's arm as `dew_fn`) on an A100 under
JAX 0.12 for 'bf16 cotangent', 256 at the first tile, and for 'gemma', which runs on a GPU alone, 256 at
softcap 30."""


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
        return (*columns(37, seed), *eighths(states, matrix), targets)

    def full(scale, bias, states, matrix, targets):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    def bounded(scale, bias, states, matrix, targets):
        return chunked_cross_entropy(
            states, OutputTable(matrix, vocab_major=False, bias=bias, scale=scale), targets, 4, tile=tile
        )[0].mean()

    def truth(scale, bias, states, matrix, targets):
        # For bf16 the rounded product is exact input to the affine head: the
        # truth isolates its fp32 scale, bias and softmax gradient (`true_logits`).
        head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
        return true_losses(head, states, targets, dtype).mean()

    assert_as_exact(jax.jit(under('default', pulled(bounded))), jax.jit(under('default', pulled(full))),
                    with_cotangent(draw, full), f'{dtype} vocabulary gradient', seeds=SEEDS['scale and bias'],
                    truth_fn=as_given(pulled(truth)))


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

    gradient = jax.jit(pulled(loss, 4))
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    # The cotangent, a scalar's, is every device's.
    specs = (P('fsdp'), P(None, 'fsdp'), P('fsdp'), P('fsdp'), P(), P())
    drawn = with_cotangent(draw, whole)

    @as_given
    def split(*terms):
        with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
            placed = [jax.device_put(term, NamedSharding(mesh, spec)) for term, spec in
                      zip(terms, specs, strict=True)]
            return jax.tree.map(np.asarray, gradient(*placed))

    assert_as_exact(split, under('highest', pulled(whole, 4)), drawn, 'vocabulary-split head',
                    seeds=SEEDS['vocabulary split'])
    terms = drawn(0)
    matrix = terms[1]
    with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
        placed = [jax.device_put(term, NamedSharding(mesh, spec))
                  for term, spec in zip(terms, specs, strict=True)]
        program = gradient.lower(*placed).compile().as_text()
    moved = [math.prod(shape) for _, shapes in collectives(program) for shape in shapes]
    assert moved and max(moved) < matrix.size // 2, moved


def column_order_case():
    """A head over 512 columns ordered by its columns, as its check-1 site
    (`test_a_heads_terms_and_gradients_are_as_exact_as_logsumexps_over_its_column_orders`) and that site's
    null runs (tools/gate_null_runs.py) take it: the draw of an order, its columns permuted and its
    targets relabelled, with one cotangent for every order (so the orbit moves only the order of the
    sums) and the inverse relabelling last; the scoring of a head's terms through it, each token's loss and
    log Z and the gradients for the states, the head, the scale and the bias under the cotangent, at the
    highest matmul precision; and the reference so scored, the whole logits' (`scaled_whole_head`)."""
    states, matrix, targets = inputs(vocab=512, features=16, tokens=(4, 5))
    states, matrix = eighths(states, matrix)
    scale, bias = columns(512)
    rng = np.random.default_rng(29)
    cotangent = tuple(jnp.asarray(rng.standard_normal(targets.shape), jnp.float32) for _ in range(2))

    def draw(seed):
        columns_in = order(512, seed)
        relabel = np.argsort(columns_in)
        return (states, matrix[:, columns_in], scale[columns_in], bias[columns_in], relabel[targets],
                cotangent, relabel)

    def scored(head):
        return under('highest', pulled(head, 4))

    return draw, scored, scored(scaled_whole_head)


def scaled_whole_head(states, matrix, scale, bias, targets):
    """Each token's loss and log Z from the whole logits of a head with a per-token scale and bias
    (`jax.nn.logsumexp`), the reference a tiled head's terms are held to."""
    logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
    log_z = jax.nn.logsumexp(logits, axis=-1)
    return log_z - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0], log_z


@pytest.mark.parametrize('tile', [None, (2, 64)], ids=['whole', 'tiled'])
def test_a_heads_terms_and_gradients_are_as_exact_as_logsumexps_over_its_column_orders(tile):
    """One device's head over 512 columns: each token's loss and log Z, and the gradients of a loss
    of both for the states, the head, the scale and the bias, are on par with `jax.nn.logsumexp`'s
    whole logits' from float64 over the orders of its columns (check 1 over the orbit: the targets
    relabelled, which reorders every row's sum of exponentials and moves no value). A row summed in
    one loop of XLA's CPU reduction with a combine of its own, and a share divided out and then
    scaled, held the whole head's state gradient 3.9 times as far."""
    ordered, scored, reference = column_order_case()

    def draw(seed):
        return ordered(seed)[:-1]

    def terms(states, matrix, scale, bias, targets):
        head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
        losses, _, log_z = chunked_cross_entropy(states, head, targets, 4, tile=tile)
        return losses, log_z

    assert_as_exact(jax.jit(scored(terms)), jax.jit(reference), draw, f'{tile} column orders',
                    seeds=SEEDS['column orders'])


def split_scores(model, wide_model, shape, vocab: int, objective_of, adjusted, label: str, seeds: int):
    """A vocabulary-split objective's (`objective_of(model, variables)`) token losses on a two-device
    mesh on par with one device's over drawn variables and `shape` tokens of `vocab` (check 1), the
    truth the objective of
    `wide_model`, `model` in float64, on them widened; its top-1 hits at the first draw one device's.
    A draw's variables are `model`'s initial ones as `adjusted` sets them. Returns the mesh, and the
    first draw's variables and batch placed on it."""
    from dew.distributed.batches import shard_batch
    from dew.training import Layout, MeshSpec

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

    @as_given
    def split(variables, tokens):
        with jax.set_mesh(mesh):
            return scored(*placed(variables, tokens))

    def alone(variables, tokens):
        wider = jax.tree.leaves(variables)[0].dtype == np.float64
        return (wide if wider else objective).token_scores(variables, tokens)

    assert_as_exact(as_given(lambda *drawn: split(*drawn).losses), lambda *drawn: alone(*drawn).losses, draw,
                    label, seeds=seeds)
    np.testing.assert_array_equal(split(*draw(0)).correct, alone(*draw(0)).correct)
    return mesh, *placed(*draw(0))


@pytest.mark.mesh(devices=2)
def test_a_vocabulary_split_heads_loss_is_taken_against_the_rows_maximum():
    """Split over the vocabulary on a mesh, a head whose logits lie near 4096 (a bias of 2**12) gives
    each token's loss on par with the whole logits' loss taken against the row's maximum and sum, log
    of the mass less the target's distance below the peak (`dew.nn.categorical.target_loss`), over drawn
    heads (check 1): log Z less the target rounded at the logits' magnitude before the subtraction that
    cancels it."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.nn.categorical import target_loss
    from dew.training import MeshSpec

    def draw(seed):
        states, matrix, targets = inputs(vocab=512, features=16, tokens=(4, 5), seed=seed)
        scale, bias = columns(512, seed)
        return (*eighths(states, matrix), scale, bias + 2.0**12, targets)

    def losses(states, matrix, scale, bias, targets):
        head = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale)
        return chunked_cross_entropy(states, head, targets, 4, tile=RAGGED)[0]

    def whole(states, matrix, scale, bias, targets):
        logits = OutputTable(matrix, vocab_major=False, bias=bias, scale=scale).logits(states)
        peak = jnp.max(logits, axis=-1)
        mass = jnp.sum(jnp.exp(logits - peak[..., None]), axis=-1)
        return target_loss(peak, mass, jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0])

    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    specs = (P('fsdp'), P(None, 'fsdp'), P('fsdp'), P('fsdp'), P())
    run = jax.jit(losses)

    @as_given
    def split(*terms):
        with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
            placed = [jax.device_put(term, NamedSharding(mesh, spec)) for term, spec in
                      zip(terms, specs, strict=True)]
            return np.asarray(run(*placed))

    assert_as_exact(split, under('highest', whole), draw, 'vocabulary-split loss near 4096',
                    seeds=SEEDS['vocabulary split'])


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
    # The model's own table, whose product the whole logits round as the chunks do.
    table = chunked.head_table(model, variables)

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
                    truth_fn=as_given(lambda hidden, targets: true_losses(table, hidden, targets, dtype)))
    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    matrix = OutputTable(head_matrix(model, variables), vocab_major=False)
    _, predicted, _ = chunked_cross_entropy(hidden, matrix, ids, chunks)
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
    init = jax.jit(model.init)

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(0), seed), 3)
        ids = jax.random.randint(keys[0], (2, 12), 0, 97)
        return init(keys[1], ids)['params'], ids, jax.random.randint(keys[2], (2, 12), 0, 97)

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

    drawn = with_cotangent(draw, full)
    assert_as_exact(jax.jit(pulled(tiled, 1)), jax.jit(pulled(full, 1)), drawn, 'backbone gradient',
                    seeds=SEEDS['backbone'], truth_fn=jax.jit(pulled(true, 1)))
    _, gradients = pulled(full, 1)(*drawn(0))
    for path, leaf in jax.tree_util.tree_leaves_with_path(gradients):
        name = jax.tree_util.keystr(path)
        assert jnp.abs(leaf).max() > 0, f"{name} has a zero gradient, so nothing is checked"


# --- the chunk arithmetic itself -------------------------------------------

@pytest.mark.parametrize("chunks", [0, -1])
def test_an_lm_run_of_a_nonpositive_chunk_count_is_refused_where_its_step_compiles(chunks):
    """An LM objective asked to score its head in no vocabulary slices, or fewer, is refused
    (ValueError) where a run's step compiles, through the trainer, before it takes any step."""
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=5, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8, attention_impl="xla")
    trainer = Trainer(LMObjective(model, seq_len=4, head_chunks=chunks, ema_decay=None), optax.sgd(.01),
                      key=jax.random.key(0))
    initial = trainer.initial_state()
    with pytest.raises(ValueError):
        trainer.compile(initial, {"text": jnp.zeros((jax.device_count(), 5), jnp.int32)})


@pytest.mark.parametrize("chunks", [4, 8])
@pytest.mark.parametrize("tile", [None, (3, 2)])
def test_small_vocabularies_take_any_positive_chunk_count(chunks, tile):
    """A five-column head's short tail and one-column tiles score every column and gradient: on par
    with the whole logits over drawn heads (check 1), and the whole logits' top-1."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=5, features=8, tokens=(16, 32), seed=seed)
        return hidden, head, jax.random.normal(jax.random.fold_in(jax.random.key(3), seed), (5,)) / 7, targets

    def tiled(hidden, head, bias, targets):
        losses, predicted, log_z = chunked_cross_entropy(
            hidden, OutputTable(head, vocab_major=False, bias=bias, softcap=3.),
            targets, chunks, tile=tile)
        return (losses, log_z), predicted

    def whole(hidden, head, bias, targets):
        logits = OutputTable(head, vocab_major=False, bias=bias, softcap=3.).logits(hidden)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (losses, log_z), logits.argmax(-1)

    def held(loss):
        # Each token's loss and log Z, and the gradients of a loss of both; the top-1 apart.
        return under("highest", pulled(lambda *terms: loss(*terms)[0], 3))

    assert_as_exact(jax.jit(held(tiled)), jax.jit(held(whole)),
                    with_cotangent(draw, lambda *terms: whole(*terms)[0]),
                    f"{chunks} {tile} small vocabulary", seeds=SEEDS["small vocabulary"])
    with jax.default_matmul_precision("highest"):
        np.testing.assert_array_equal(tiled(*draw(0))[1], whole(*draw(0))[1])


@pytest.mark.parametrize("chunks", [4, 8])
def test_lm_training_with_small_vocabularies_matches_the_unchunked_head(chunks):
    """The objective's chunk count reaches the tiled head in both its token losses and the optimizer's
    update: each on par with the unchunked head's over drawn batches (check 1), the truth the model in
    float64. The update's loss is the objective's (`scalar_loss`), its token losses' mean."""
    from reference_error import widened

    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=5, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8, head_bias=True, attention_impl="xla")
    objective = LMObjective(model, seq_len=4, head_chunks=chunks, head_tile=(3, 2), ema_decay=None)
    trainer = Trainer(objective, optax.sgd(.01), key=jax.random.key(0))
    initial = trainer.initial_state()

    def draw(seed):
        key = jax.random.fold_in(jax.random.key(2), seed)
        # A step pulls back its loss's cotangent of one, which the trainer takes as given.
        return jax.random.randint(key, (jax.device_count(), 5), 0, 5), np.float32(1)

    compiled = trainer.compile(initial, {"text": draw(0)[0]})
    scores = jax.jit(lambda ids: objective.token_scores(initial.variables, ids).losses)

    @as_given
    def trained(ids, cotangent):
        del cotangent  # One, the trainer's own.
        state, _, _, _, accepted, _ = compiled(jax.tree.map(jnp.copy, initial), {"text": ids})
        assert bool(accepted) and int(state.updates) == 1
        return {"losses": scores(ids), "params": state.variables["params"]}

    def stepped(model, variables):
        def losses(params, ids):
            logits = model.apply({**variables, "params": params}, ids[:, :-1])
            return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:])

        def run(ids, cotangent):
            def mean(params, ids):
                # The per-token losses ride the same evaluation as the mean they pull back through.
                each = losses(params, ids)
                return jnp.mean(each), each

            (_, gradient), each = pulled(mean, 1, has_aux=True)(variables["params"], ids, cotangent)
            updated = jax.tree.map(lambda p, g: p - .01 * g, variables["params"], gradient)
            return {"losses": each, "params": updated}
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
BOUNDED_FEATURES = 64
"""The bounded backward's features, the terms each of its logits sums."""
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
    """Both outputs, and their gradients, on par with the pass that holds all the logits over drawn
    heads (check 1); its top-1 the full pass's, and neither short tile's gradient dropped."""
    precision = jax.lax.Precision.HIGHEST

    def draw(seed):
        # 64 features, a head's row of terms each logit sums: each draw's error averages over them, where
        # 7 left the z-loss's head gradient heavy-tailed across draws on an H100 (row 1725).
        return inputs(vocab=RAGGED_VOCAB, features=BOUNDED_FEATURES, tokens=(1, RAGGED_TOKENS), seed=seed)

    def full(states, matrix, targets):
        losses, log_z = oracle(states, matrix, targets, softcap, precision)
        return loss_weight * losses + z_weight * jnp.square(log_z), (losses, log_z)

    def tiled(states, matrix, targets):
        losses, _, log_z = chunked_cross_entropy(
            states,
            OutputTable(matrix, vocab_major=False, softcap=softcap, precision=precision),
            targets,
            4,
            tile=RAGGED,
        )
        return loss_weight * losses + z_weight * jnp.square(log_z), (losses, log_z)

    drawn = with_cotangent(draw, full, has_aux=True)
    assert_as_exact(jax.jit(pulled(tiled, has_aux=True)), jax.jit(pulled(full, has_aux=True)), drawn,
                    f"{loss_weight} {z_weight} {softcap} bounded", seeds=SEEDS["bounded"])
    hidden, head, targets, cotangent = drawn(0)
    _, predicted, _ = chunked_cross_entropy(
        hidden,
        OutputTable(head, vocab_major=False, softcap=softcap, precision=precision),
        targets,
        4,
        tile=RAGGED,
    )
    _, expected_top1 = reference(hidden, head, targets, softcap=softcap)
    np.testing.assert_array_equal(predicted, expected_top1)
    (_, d_hidden, d_head), _ = pulled(tiled, has_aux=True)(hidden, head, targets, cotangent)
    assert jnp.abs(d_head[:, RAGGED[1]:]).max() > 0, "the short column tile was dropped"
    assert jnp.abs(d_hidden[:, RAGGED[0]:]).max() > 0, "the short token tile was dropped"


def test_the_softcap_gradient_survives_saturated_logits():
    """Logits well past the cap: `tanh` still has to pass a gradient back, on par with the full pass's
    for the states, the head and the cap over drawn heads (check 1), and the top-1 the full pass's."""
    def draw(seed):
        hidden, head, targets = inputs(vocab=97, features=8, tokens=(1, 9), seed=seed)
        return hidden * 8, head * 4, jnp.float32(30.), targets

    def full(states, matrix, cap, targets):
        losses, log_z = oracle(states, matrix, targets, cap)
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    def tiled(states, matrix, cap, targets):
        losses, _, log_z = chunked_cross_entropy(
            states, OutputTable(matrix, vocab_major=False, softcap=cap), targets, 4, tile=RAGGED
        )
        return jnp.mean(losses + 0.03 * jnp.square(log_z))

    assert_as_exact(jax.jit(pulled(tiled, 3)), jax.jit(pulled(full, 3)), with_cotangent(draw, full),
                    "saturated softcap", seeds=SEEDS["saturated"])
    hidden, head, _, targets = draw(0)
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
        return (losses, log_z), jnp.argmax(logits, axis=-1)

    def chunked_pass(states, matrix, targets):
        losses, predicted, log_z = chunked_cross_entropy(
            states,
            OutputTable(matrix, vocab_major=False, precision=precision),
            targets,
            4,
            tile=tile,
            excluded=excluded,
        )
        return (losses, log_z), predicted

    def held(loss):
        # Each token's loss and log Z, and the gradients of a loss of both; the top-1 apart.
        return pulled(lambda *terms: loss(*terms)[0])

    drawn = with_cotangent(draw, lambda *terms: full(*terms)[0])
    assert_as_exact(jax.jit(held(chunked_pass)), jax.jit(held(full)), drawn, f"{tile} {excluded} excluded",
                    seeds=SEEDS["excluded"])
    terms = drawn(0)
    _, _, d_head = jax.jit(held(chunked_pass))(*terms)
    predicted = chunked_pass(*terms[:-1])[1]
    np.testing.assert_array_equal(predicted, full(*terms[:-1])[1])
    assert not jnp.any(predicted == excluded)
    assert jnp.all(d_head[:, excluded] == 0)


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

    def once(states, matrix, targets, cotangent):
        loss, *wide = pulled(full)(states.astype(jnp.float32), matrix.astype(jnp.float32), targets, cotangent)
        return loss, *(gradient.astype(dtype) for gradient in wide)

    @jax.jit
    def logits_cotangent(states, matrix, targets, cotangent):
        """The fp32 logits' cotangent the site's loss gives under the drawn cotangent."""
        logits = OutputTable(matrix, vocab_major=False, precision=precision).logits(states)

        def loss(logits):
            picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
            return jnp.mean(jax.nn.logsumexp(logits, axis=-1) - picked)

        return jax.vjp(loss, logits)[1](cotangent)[0]

    tile_product = jax.jit(lambda states, logits: jnp.einsum("tf,tv->fv", states, logits,
                                                             precision=precision))
    reference = jax.jit(once)
    # The tiled head's pieces, from the site's own constants: 4 chunks of 17 columns, so blocks of
    # ceil(17 / 4) = 5 columns, each over token tiles of RAGGED's 128.
    columns_wide, tokens_tall = -(-17 // 4), RAGGED[0]

    @as_given
    def library(states, matrix, targets, cotangent):
        """The library arm (BIBLE §7's exception, row 1543): the oracle's loss and state gradient, and
        its head gradient as the tiled head takes it, one fp32 product a block of its columns by a token
        tile, through the library, on the site's own logits' cotangent, the pieces summed exactly in
        float64 on the host and rounded to `dtype` once; the gate holds it to the head gradient alone
        (`library_leaves`), the loss and the state gradient the oracle's. On an H100 PCIe a 128-token
        tile's product rounds 1.85 times as far from float64 as the 257-token whole product (per element
        3.8-7.3e-9 RMS against 1.9-4.5e-9, both cuBLASLt at beta 0 with a 32 MiB workspace;
        TL-h100-tokentiles-1725),
        which no combining of the tiles repairs."""
        loss, d_hidden, _ = reference(states, matrix, targets, cotangent)
        wide_states = states.astype(jnp.float32)
        d_logits = logits_cotangent(wide_states, matrix.astype(jnp.float32), targets, cotangent)
        d_head = np.zeros(matrix.shape, np.float64)
        for first in range(0, matrix.shape[1], columns_wide):
            for start in range(0, states.shape[0], tokens_tall):
                d_head[:, first:first + columns_wide] += np.asarray(tile_product(
                    wide_states[start:start + tokens_tall],
                    d_logits[start:start + tokens_tall, first:first + columns_wide]), np.float64)
        return loss, d_hidden, d_head.astype(dtype)

    drawn = with_cotangent(draw, full)
    assert_as_exact(jax.jit(pulled(tiled)), reference, drawn, f"{dtype} token tiles",
                    seeds=SEEDS["token tiles"], truth_fn=pulled(full), library_fn=library,
                    library_row=1543, library_leaves=("[2]",))
    _, d_hidden, d_head = pulled(tiled)(*drawn(0))
    assert d_hidden.dtype == d_head.dtype == dtype


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_the_backbone_receives_both_outputs_gradients(tie_embeddings):
    """Every parameter's gradient of a loss of both outputs through the tiled head, on par with the
    whole logits' over drawn models and tokens (check 1), the truth the model in float64."""
    model, wide = (small_model(dtype=dtype, tie_embeddings=tie_embeddings, final_logit_softcap=30.)
                   for dtype in (jnp.float32, jnp.float64))

    init = jax.jit(model.init)

    def draw(seed):
        ids = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(31), seed), (4, 16), 0, 97)
        return init(jax.random.fold_in(jax.random.PRNGKey(32), seed), ids)["params"], ids

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

    assert_as_exact(jax.jit(pulled(tiled, 1)), jax.jit(pulled(full, 1)), with_cotangent(draw, full),
                    f"{tie_embeddings} both outputs", seeds=SEEDS["both outputs"])


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
    and its top-1 columns are Optax's full-vocabulary logits'. The losses and gradients of the same
    boundary targets are held to the whole logits' in fp32 on a Gemma-sized head (check 1,
    `test_a_gemma_sized_vocabulary_is_as_exact_as_the_whole_logits`)."""
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


# A Gemma-sized vocabulary's float64 truth over the seeds its softcapped cases need is numeric work at
# model size, a GPU's: on a CPU each case takes about eleven minutes.
@pytest.mark.gpu(reason="a Gemma-sized vocabulary's float64 truth at the seeds it needs")
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
        def loss(states, matrix, targets, weights):
            if chunks is None:
                losses, log_z = oracle(states, matrix, targets, softcap)
            else:
                losses, _, log_z = chunked_cross_entropy(
                    states, OutputTable(matrix, vocab_major=False, softcap=softcap), targets, chunks,
                    tile=TILE)
            return jnp.sum(weights * losses), (losses, log_z)
        return loss

    # The weights are the loss's; its cotangent, a scalar's, the draw's.
    assert_as_exact(jax.jit(pulled(scored(chunks), has_aux=True)),
                    jax.jit(pulled(scored(None), has_aux=True)),
                    with_cotangent(draw, scored(None), has_aux=True), f"{chunks} {softcap} gemma vocabulary",
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

    drawn = with_cotangent(draw, true)
    assert_as_exact(jax.jit(pulled(chunked_loss)), jax.jit(pulled(strict_or_rounded(round_logits=True))),
                    drawn, f"{tile} bf16 cotangent", seeds=SEEDS["bf16 cotangent"], truth_fn=pulled(true))
    terms = drawn(0)
    strict = pulled(strict_or_rounded(round_logits=False))(*terms)
    assert not np.array_equal(strict[1], pulled(chunked_loss)(*terms)[1])


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
        return hidden, head, jnp.float32(30.), outside

    def scored(tile):
        def loss(states, matrix, cap, targets):
            table = OutputTable(matrix, vocab_major=False, softcap=softcap and cap)
            losses, _, log_z = chunked_cross_entropy(states, table, targets, 4, tile=tile)
            return jnp.mean(losses) + 0.01 * jnp.mean(jnp.square(log_z)), (losses, log_z)
        return loss

    def run(tile):
        # fp32 products at full precision, so a backend whose default fp32 product is TF32 runs
        # both heads alike; the cap's gradient where the head caps.
        held = pulled(scored(tile), 3 if softcap else 2, has_aux=True)
        return under("highest", held) if dtype == jnp.float32 else held

    assert_as_exact(jax.jit(run(None)), jax.jit(run(RAGGED)),
                    with_cotangent(draw, scored(RAGGED), has_aux=True), f"{dtype} {softcap} whole logits",
                    seeds=SEEDS["whole logits"])
    hidden, head, _, targets = draw(0)
    table = OutputTable(head, vocab_major=False, softcap=softcap)
    tiled, whole = (chunked_cross_entropy(hidden, table, targets, 4, tile=tile) for tile in (RAGGED, None))
    assert jnp.array_equal(tiled[1], whole[1])


def test_an_fp32_whole_heads_overflowed_state_gradient_stays_infinite():
    """A whole fp32 head's state gradient overflows as one product's does: infinite, never inf - inf.
    One zero state, 2048 columns, the target's at -3e38 and the rest at +3e38: a finite loss, log
    2048, whose state gradient overflows."""
    columns = 2048
    head = jnp.full((1, columns), 3e38, jnp.float32).at[0, 0].set(-3e38)
    hidden, targets = jnp.zeros((1, 1, 1), jnp.float32), jnp.zeros((1, 1), jnp.int32)

    def loss(states):
        table = OutputTable(head, vocab_major=False)
        losses, _, _ = chunked_cross_entropy(states, table, targets, 1, tile=None)
        return jnp.sum(losses)

    with jax.default_matmul_precision("highest"):
        value, gradient = jax.value_and_grad(loss)(hidden)
    assert np.isfinite(float(value))
    assert np.all(np.isinf(np.asarray(gradient))) and not np.any(np.isnan(np.asarray(gradient)))


def test_a_fused_heads_overflowed_column_gradient_stays_infinite():
    """The fused head sums each column term's gradient over its token tiles as a double-float pair: a
    term that overflows leaves the sum infinite, as a plain sum's is, never inf - inf's NaN. One state
    of 1, columns at -3e38, 3e38 and 3e38 scaled by 1e-37: logits -30, 30 and 30, a finite loss, and the
    target column's scale gradient 2 * 3e38, past float32."""
    matrix = jnp.asarray([[-3e38, 3e38, 3e38]], jnp.float32)
    hidden, targets, weights = jnp.ones((1, 1), jnp.float32), jnp.zeros((1,), jnp.int32), jnp.full((1,), 2.0)

    def total(scale):
        head = OutputTable(matrix, vocab_major=False, scale=scale)
        return fused_head_loss(hidden, head, targets, weights)[0]

    with jax.default_matmul_precision("highest"):
        value, gradient = jax.value_and_grad(total)(jnp.full((3,), 1e-37, jnp.float32))
    gradient = np.asarray(gradient)
    assert np.isfinite(float(value))
    assert np.isposinf(gradient[0]) and np.all(np.isfinite(gradient[1:]))


CONFIDENT_SEEDS = 8
"""Check 1's declared seeds for a confident target's gradient, as `test_categorical`'s are."""


def identity_head() -> OutputTable:
    """A head whose logits are its states, so the gradient the states take is the logits' own."""
    return OutputTable(jnp.eye(7), vocab_major=False, precision=jax.lax.Precision.HIGHEST)


@pytest.mark.parametrize("label", [0, 5], ids=["column 0", "column 5"])
@pytest.mark.parametrize("margin", [20.0, 40.0, 80.0])
@pytest.mark.parametrize("path", ["tiled", "whole", "fused", "logits"])
def test_a_confident_targets_logit_gradient_is_the_rest_of_the_rows_eq_7(path, margin, label):
    """Every head's gradient of a confident target's loss with respect to the logits is arXiv:2610.31642's
    eq. 7 from float64 (check 1): each other class's probability, and the target's entry the rest of the
    row's mass, summed directly. A head that takes the target's entry as its share less one collapses it
    to 0 once the share rounds to 1, and then the row stops summing to zero. The target's entry is
    negative."""
    targets = np.full((3,), label, np.int32)
    gradient = jax.jit(jax.grad(functools.partial(path_loss, path, targets)))
    assert_as_exact(gradient, as_given(lambda logits: label_derivative(logits, label)),
                    lambda seed: confident(margin, seed, label),
                    f"{path} at margin {margin}, column {label}", seeds=CONFIDENT_SEEDS)
    (rows,) = confident(margin, 0, label)
    assert np.all(np.asarray(gradient(rows))[:, label] < 0)


@pytest.mark.parametrize("path", ["tiled", "whole", "fused", "logits"])
@pytest.mark.parametrize("margin", [20.0, 40.0, 80.0])
def test_a_confident_targets_rest_of_the_row_leaves_an_excluded_column_out(path, margin):
    """A masked diffusion model's excluded column (MDLM's mask token) gives no mass, so a confident target's
    rest of the row leaves it out as the partition does, in every head: the gradient is eq. 7 over the other
    columns from float64 (check 1), and the excluded column takes none. The excluded column scores highest in
    every row, so a rest of the row that kept it would show."""
    excluded, label = 6, 2
    targets = np.full((3,), label, np.int32)
    head = identity_head()

    def draw(seed):
        (rows,) = confident(margin, seed, label)
        rows[:, excluded] = 40.0
        return (rows,)

    def loss(logits):
        if path == "logits":
            return chunked.logits_cross_entropy(logits, targets, excluded=excluded)[0].sum()
        if path == "fused":
            return fused_head_loss(logits, head, targets, jnp.ones(targets.shape), excluded=excluded)[0]
        tile = (3, 3) if path == "tiled" else None
        return chunked_cross_entropy(logits, head, targets, 1, tile=tile, excluded=excluded)[0].sum()

    @as_given
    def reference(logits):
        kept = label_derivative(np.delete(logits, excluded, axis=-1), label)
        return np.concatenate([kept[:, :excluded], np.zeros((len(kept), 1), kept.dtype), kept[:, excluded:]],
                              axis=-1)

    gradient = jax.jit(jax.grad(loss))
    assert_as_exact(gradient, reference, draw, f"{path} excluding column {excluded} at margin {margin}",
                    seeds=CONFIDENT_SEEDS)
    (rows,) = draw(0)
    assert np.all(np.asarray(gradient(rows))[:, excluded] == 0)


def vocabulary_split(gradient):
    """`gradient(logits, matrix)` over two devices' vocabulary halves: rows `[B, 8]` split by their
    columns through an identity head split the same way, at the highest matmul precision."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])

    @as_given
    def split(logits):
        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
            placed = (jax.device_put(logits, NamedSharding(mesh, P("fsdp"))),
                      jax.device_put(jnp.eye(8), NamedSharding(mesh, P(None, "fsdp"))))
            return np.asarray(gradient(*placed))
    return split


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("excluded", [None, 6], ids=["no column", "column 6 excluded"])
@pytest.mark.parametrize("tile", [None, (2, 2)], ids=["whole", "tiled"])
@pytest.mark.parametrize("margin", [20.0, 40.0, 80.0])
def test_a_vocabulary_split_heads_confident_target_gradient_is_eq_7(tile, margin, excluded):
    """Split over the vocabulary on a mesh, the head's gradient of a confident target's loss with respect to
    the logits is eq. 7 from float64 (check 1), as one device's is, with a column excluded or none. The
    target's shard takes its entry from the rest of the row summed over every shard, not from its share less
    one, and an excluded column, in whichever shard it sits, joins neither the rest nor the mass."""
    label = 5
    targets = np.full((4,), label, np.int32)

    def draw(seed):
        (rows,) = confident(margin, seed, label, width=8, count=4)
        if excluded is not None:
            rows[:, excluded] = 40.0
        return (rows,)

    def loss(logits, matrix):
        head = OutputTable(matrix, vocab_major=False, precision=jax.lax.Precision.HIGHEST)
        return chunked_cross_entropy(logits, head, targets, 1, tile=tile, excluded=excluded)[0].sum()

    gradient = jax.jit(jax.grad(loss))

    @as_given
    def reference(logits):
        if excluded is None:
            return label_derivative(logits, label)
        kept = label_derivative(np.delete(logits, excluded, axis=-1), label)
        return np.concatenate([kept[:, :excluded], np.zeros((len(kept), 1), kept.dtype), kept[:, excluded:]],
                              axis=-1)

    split = vocabulary_split(gradient)
    assert_as_exact(split, reference, draw, f"vocabulary-split at margin {margin}, {tile}, {excluded}",
                    seeds=CONFIDENT_SEEDS)
    if excluded is not None:
        (rows,) = draw(0)
        assert np.all(split(rows)[:, excluded] == 0)


@as_given
def softmax(rows: np.ndarray) -> np.ndarray:
    """NumPy's softmax over the last axis, in the rows' own dtype (`log_softmax`)."""
    return np.exp(log_softmax(rows))


@pytest.mark.parametrize("margin", [20.0, 40.0, 80.0])
@pytest.mark.parametrize("path", ["tiled", "whole", "logits"])
def test_an_unlikely_targets_log_partition_gradient_is_its_own_probability(path, margin):
    """The log partition's gradient is each column's probability, the target's included, whatever the
    target: at a target `margin` below the row's lead, its probability, about exp(-margin), on par with
    NumPy's softmax from float64 (check 1), where 1 less the rest of the row's share cancels it to 0."""
    lead, target = 0, 5
    targets = np.full((3,), target, np.int32)
    head = identity_head()

    def partition(logits):
        if path == "logits":
            return chunked.logits_cross_entropy(logits, targets)[2].sum()
        tile = (3, 3) if path == "tiled" else None
        return chunked_cross_entropy(logits, head, targets, 1, tile=tile)[2].sum()

    gradient = jax.jit(jax.grad(partition))
    assert_as_exact(gradient, softmax, lambda seed: confident(margin, seed, lead),
                    f"{path} at margin {margin}", seeds=CONFIDENT_SEEDS)
    (rows,) = confident(margin, 0, lead)
    assert np.all(np.asarray(gradient(rows))[:, target] > 0)


@pytest.mark.parametrize("margin", [20.0, 40.0])
@pytest.mark.parametrize("path", ["tiled", "whole", "fused", "logits"])
def test_a_dominant_z_loss_keeps_an_unlikely_targets_share(path, margin):
    """A z-loss whose cotangent outweighs the loss's (PaLM's, at a large coefficient) moves an unlikely
    target by its probability times that cotangent, less the rest of its row: the target's entry on par
    with eq. 7 plus the z-loss's `2 z log Z p` in NumPy from float64 (check 1), which a target entry of
    `d_partition (1 - rest / mass)` rounds to 0. The entry alone is measured: the lead column's, near the
    cotangent itself, rounds by more than the target's whole value."""
    lead, target, weight = 0, 5, 1e8
    targets = np.full((3,), target, np.int32)
    head = identity_head()

    def loss(logits):
        if path == "fused":
            return fused_head_loss(logits, head, targets, jnp.ones(targets.shape), z_loss=weight)[0]
        if path == "logits":
            losses, _, log_z = chunked.logits_cross_entropy(logits, targets)
        else:
            losses, _, log_z = chunked_cross_entropy(logits, head, targets, 1,
                                                     tile=(3, 3) if path == "tiled" else None)
        return (losses + weight * jnp.square(log_z)).sum()

    @as_given
    def reference(rows):
        return (label_derivative(rows, target) + 2 * weight * log_partition(rows) * softmax(rows))[:, target]

    gradient = jax.jit(jax.grad(loss))

    def entry(rows):
        return gradient(rows)[:, target]

    assert_as_exact(entry, reference, lambda seed: confident(margin, seed, lead),
                    f"{path} at margin {margin}", seeds=CONFIDENT_SEEDS)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("tile", [None, (2, 2)], ids=["whole", "tiled"])
@pytest.mark.parametrize("margin", [20.0, 40.0])
def test_a_vocabulary_split_heads_unlikely_target_keeps_its_log_partition_share(tile, margin):
    """Split over the vocabulary on a mesh, the log partition's gradient at an unlikely target is on par
    with NumPy's softmax from float64 (check 1), as one device's is: the target's shard, which picks its
    logit, takes the partition's share of it directly, not 1 less the rest of the row."""
    lead, target = 0, 5
    targets = np.full((4,), target, np.int32)

    gradient = jax.jit(jax.grad(functools.partial(split_head_term, 2, targets, tile)))

    assert_as_exact(vocabulary_split(gradient), softmax,
                    lambda seed: confident(margin, seed, lead, width=8, count=4),
                    f"vocabulary-split at margin {margin}, {tile}", seeds=CONFIDENT_SEEDS)


def path_loss(path: str, targets: np.ndarray, logits: jax.Array) -> jax.Array:
    """The summed cross entropy of `logits` (each row a token's) at `targets` through one of the heads:
    the whole-logits head, the fused one, or the chunked one tiled or whole over an identity table."""
    head = identity_head()
    if path == "logits":
        return chunked.logits_cross_entropy(logits, targets)[0].sum()
    if path == "fused":
        return fused_head_loss(logits, head, targets, jnp.ones(targets.shape))[0]
    tile = (3, 3) if path == "tiled" else None
    return chunked_cross_entropy(logits, head, targets, 1, tile=tile)[0].sum()


def split_head_term(term: int, targets: np.ndarray, tile, logits: jax.Array, matrix: jax.Array) -> jax.Array:
    """The chunked head's `term` (0 the loss, 2 the log partition) of `logits` against `matrix`, a
    hidden-major table, summed: the vocabulary-split tests' function, `vocabulary_split` its mesh."""
    head = OutputTable(matrix, vocab_major=False, precision=jax.lax.Precision.HIGHEST)
    return chunked_cross_entropy(logits, head, targets, 1, tile=tile)[term].sum()


def unconfident(seed: int, width: int = 7, count: int = 3) -> tuple[np.ndarray]:
    """`count` rows of `width` float32 logits, each a standard normal draw: no column stands out, so a
    target is seldom likelier than a half."""
    return (np.random.default_rng([width, seed]).standard_normal((count, width)).astype(np.float32),)


def written_gradient(targets: np.ndarray):
    """Autodiff's gradient of the cross entropy written out from `jax.nn.log_softmax`, the reference a
    target no likelier than a half takes: its entry `p - 1`, rounded once (`test_categorical`)."""
    return jax.jit(jax.grad(
        lambda logits: -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=-1).sum()))


@pytest.mark.parametrize("path", ["tiled", "whole", "fused", "logits"])
def test_an_unconfident_targets_logit_gradient_is_on_par_with_autodiffs(path):
    """Every head's gradient of an unconfident target's loss with respect to the logits is on par from
    float64 with autodiff's of the cross entropy written out (check 1): below a half a target's entry is
    its share less one, rounded once, where the rest of the row summed carries each class's rounding of
    the partition (`dew.nn.categorical.label_entry`, the rule every head reads)."""
    targets = np.arange(3, dtype=np.int32)
    gradient = jax.jit(jax.grad(functools.partial(path_loss, path, targets)))
    reference = written_gradient(targets)
    label = f"{path} at an unconfident target"
    with jax.default_matmul_precision("highest"):
        seeds = seeds_for(reference, unconfident, label, dew_fn=gradient)
        assert_as_exact(gradient, reference, unconfident, label, seeds=seeds)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("tile", [None, (2, 2)], ids=["whole", "tiled"])
def test_a_vocabulary_split_heads_unconfident_target_gradient_is_on_par_with_autodiffs(tile):
    """Split over the vocabulary on a mesh, the head's gradient of an unconfident target's loss with respect
    to the logits is on par from float64 with autodiff's of the cross entropy written out (check 1), as one
    device's is: the target's shard takes its share less one, its share of the partition summed over every
    shard."""
    targets = np.arange(4, dtype=np.int32) * 2

    split = vocabulary_split(jax.jit(jax.grad(functools.partial(split_head_term, 0, targets, tile))))
    reference = written_gradient(targets)
    label = f"vocabulary-split at an unconfident target, {tile}"

    def draw(seed):
        return unconfident(seed, width=8, count=4)

    with jax.default_matmul_precision("highest"):
        seeds = seeds_for(reference, draw, label, dew_fn=split)
        assert_as_exact(split, reference, draw, label, seeds=seeds)


def test_a_projected_head_is_scored_in_tiles_as_its_whole_logits_are():
    """A decoder whose embedding has a width of its own (`embedding_features`, OPT-350m's) is scored in
    tiles, its final states projected once (`head_input`) and met by the vocabulary matrix
    (`projected_output_table`): its per-token cross entropy and the gradient of their sum in the head's
    parameters are on par with the whole logits' (check 1, both products in float64 the truth)."""
    model = CausalTransformer(vocab_size=97, emb_features=32, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=8, embedding_features=16, qk_norm=False, attention_impl='reference')
    ids = jnp.asarray([[2, 7, 3, 5], [11, 4, 9, 1]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    states = model.apply(variables, ids, method='hidden_states')

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(8), seed))
        return (variables['params'], jax.random.normal(keys[0], states.shape),
                jax.random.randint(keys[1], ids.shape, 0, 97))

    def scored(loss):
        def run(params, states, targets):
            def total(params):
                return jnp.sum(loss({**variables, 'params': params}, states, targets))
            return loss({**variables, 'params': params}, states, targets), jax.grad(total)(params)
        return run

    def tiled(variables, states, targets):
        with jax.default_matmul_precision('highest'):
            return chunked.head_cross_entropy(model, variables, states, targets, 4, tile=(3, 11))[0]

    def whole(variables, states, targets):
        with jax.default_matmul_precision('highest'):
            return optax.softmax_cross_entropy_with_integer_labels(
                chunked.model_logits(model, variables, states), targets)

    wide = model.clone(dtype=jnp.float64)

    @as_given
    def truth(params, states, targets):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), {**variables, 'params': params})

            def total(params):
                read = jnp.asarray(states, jnp.float64)
                logits = chunked.model_logits(wide, {**held, 'params': params}, read)
                return optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return jax.device_get((total(held['params']),
                                   jax.grad(lambda p: jnp.sum(total(p)))(held['params'])))

    assert chunked.head_table(model, variables) is None
    assert_as_exact(jax.jit(scored(tiled)), jax.jit(scored(whole)), draw, 'projected head',
                    seeds=SEEDS['projected head'], truth_fn=truth)
