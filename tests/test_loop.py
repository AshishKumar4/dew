"""A looped stack (`Loop`): passes that share the layers' parameters, each
with its own entries in the decode cache. A loop computes the stack it
unrolls to as that stack does, to the bit; decoding, scanned runs and
checkpointed passes, which compute it another way, are on par with it from
float64 over drawn models and tokens (check 1)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER, decode_steps, model_draw, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop

VOCAB = 37


def tiny(**overrides):
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "attention_impl": "reference",
                                "tie_embeddings": False, "loop": Loop(3, exit_gate=True), **overrides})


def tokens(rng, length=12, rows=2):
    return jax.random.randint(rng, (rows, length), 0, VOCAB)


SEEDS = {'decode': 64, 'scanned': 64, 'checkpointed': 512, 'block decode': 32}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada):
the most any of a site's cases took."""


def drawn(model, length=12, rows=2):
    """Check 1's draw for `model`: a seed's parameters and `rows` rows of `length` tokens."""
    return model_draw(model, (rows, length), VOCAB)


def wide(model):
    """`model` in float64, a function of its parameters and tokens: the truth of a site that compares
    two ways of computing it."""
    return jax.jit(model.clone(dtype=jnp.float64).apply)


def decoded(model, params, ids, prompt=4):
    """Prefill `prompt` tokens, then decode the rest one at a time: each
    position's logits and the cache after the last."""
    made, step = decode_steps(model)
    logits, cache = step(params, made(params, ids.shape[0]), ids[:, :prompt])
    steps = [logits]
    for position in range(prompt, ids.shape[1]):
        logits, cache = step(params, cache, ids[:, position:position + 1])
        steps.append(logits)
    return jnp.concatenate(steps, axis=1), cache


def unrolled(params, steps):
    """The parameters of a stack `steps` times as deep whose layer i is layer i mod depth."""
    depth = TINY_DECODER["num_layers"]
    tree = dict(params['params'])
    for index in range(depth, steps * depth):
        tree[f'layers_{index}'] = tree[f'layers_{index % depth}']
    return {'params': tree}


def test_a_loop_without_its_norm_is_the_stack_repeated_cache_and_all(rng):
    """Two passes of two layers compute four layers whose last two repeat the
    first two, and their decode cache holds the same entries under the same names."""
    looped = tiny(loop=Loop(2, step_norm=False))
    deep = tiny(num_layers=4, loop=None)
    ids = tokens(rng)
    params = looped.init(rng, ids)
    np.testing.assert_array_equal(looped.apply(params, ids), deep.apply(unrolled(params, 2), ids))
    ours, cache = decoded(looped, params, ids)
    theirs, deep_cache = decoded(deep, unrolled(params, 2), ids)
    np.testing.assert_array_equal(ours, theirs)
    assert jax.tree.structure(cache) == jax.tree.structure(deep_cache)
    jax.tree.map(np.testing.assert_array_equal, cache, deep_cache)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_decoding_a_loop_matches_its_prefill(rng, scan_layers):
    """A loop decoded a token at a time is on par with its prefill (check 1), and its cache holds
    each pass's layers."""
    model = tiny(scan_layers=scan_layers)
    assert_as_exact(lambda params, ids: decoded(model, params, ids)[0], jax.jit(model.apply),
                    drawn(model), f'{scan_layers} loop decode', seeds=SEEDS['decode'], truth_fn=wide(model))
    ids = tokens(rng)
    _, cache = decoded(model, model.init(rng, ids), ids)
    assert sorted(cache) == [f'layers_{index}' for index in range(6)]


def test_scanned_runs_loop_as_the_plain_layers_do():
    """A loop's layers scanned are on par with them run one by one (check 1)."""
    assert_as_exact(jax.jit(tiny(scan_layers=True).apply), jax.jit(tiny().apply), drawn(tiny()),
                    'scanned loop', seeds=SEEDS['scanned'], truth_fn=wide(tiny()))


def test_backprop_steps_train_through_the_last_passes_alone(rng):
    ids = tokens(rng)
    params = tiny().init(rng, ids)

    def gradient(backprop_steps):
        model = tiny(loop=Loop(3, backprop_steps=backprop_steps))

        def loss(tree):
            return jnp.mean(model.apply(tree, ids) ** 2)

        return jax.value_and_grad(loss)({'params': {name: value for name, value in params['params'].items()
                                                    if name != 'early_exit_gate'}})

    (whole, every), (cut, last) = gradient(None), gradient(1)
    assert whole == cut
    assert np.any(every['params']['embed_tokens']['embedding'])
    assert not np.any(last['params']['embed_tokens']['embedding'])
    assert np.any(last['params']['layers_0']['mlp']['down_proj']['kernel'])


@pytest.mark.parametrize("backprop_steps", [None, 1])
@pytest.mark.parametrize("scan_layers", [False, True])
def test_checkpointed_passes_preserve_logits_and_gradients(backprop_steps, scan_layers):
    """Three passes over layers [1, 3) of four, checkpointed, keep the fp32
    logits and gradients of the plain passes on par (check 1), including the
    cut before the last pass: a pass recomputed instead of retained may fuse
    its backward another way.
    """
    loop = Loop(3, layers=(1, 3), backprop_steps=backprop_steps, remat=False)
    plain = tiny(num_layers=4, loop=loop, scan_layers=scan_layers, dtype="float32")
    checkpointed = plain.clone(loop=dataclasses.replace(loop, remat=True))

    def result(model):
        def run(tree, ids, cotangent):
            pulled = with_gradients(lambda tree: model.apply(tree, ids, train=True))
            logits, gradients = pulled(tree, cotangent)
            return {'logits': logits, 'gradients': gradients}
        return jax.jit(run)

    # Eight rows of sixteen tokens: a norm's scale sums its gradient over enough of them to resolve.
    draw = with_logit_cotangent(drawn(plain, 16, 8), VOCAB, 0x100)
    assert_as_exact(result(checkpointed), result(plain), draw,
                    f'{scan_layers} {backprop_steps} checkpointed', seeds=SEEDS['checkpointed'],
                    truth_fn=result(plain.clone(dtype=jnp.float64)))
    gradients = result(checkpointed)(*draw(0))['gradients']['params']
    assert bool(jnp.any(gradients['embed_tokens']['embedding'])) == (backprop_steps is None)
    assert bool(jnp.any(gradients['layers_1']['mlp']['down_proj']['kernel']))


@pytest.mark.parametrize("scan_layers", [False, True])
def test_checkpointed_passes_leave_decode_and_its_collections_unchanged(rng, scan_layers):
    """Decode bypasses the pass checkpoints and retains each pass's cache
    names and values, under both the plain stack and its scanned runs.
    """
    loop = Loop(3, layers=(1, 3), remat=False)
    plain = tiny(num_layers=4, loop=loop, scan_layers=scan_layers)
    checkpointed = plain.clone(loop=dataclasses.replace(loop, remat=True))
    ids = tokens(rng)
    params = plain.init(rng, ids)
    expected, actual = decoded(plain, params, ids), decoded(checkpointed, params, ids)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    assert sorted(actual[1]) == [f'layers_{index}' for index in range(8)]
    boundaries = []
    for model in (plain, checkpointed):
        program = jax.make_jaxpr(lambda tree, model=model: model.apply(
            tree, ids[:, :1], decode=True, mutable=['cache']))(params).jaxpr
        boundaries.append(sum(equation.primitive.name == "remat2" for equation in program.eqns))
    assert boundaries[0] == boundaries[1]


def test_a_loop_over_a_block_is_the_stack_with_the_block_repeated_cache_and_all(rng):
    """Layers 1 and 2 of four, run twice with no norm between, compute six
    layers whose fourth and fifth repeat the second and third, to the bit,
    and keep the second pass's entries after the first's; decoded, and
    scanned, they are on par with their prefill (check 1)."""
    looped = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)))
    ids = tokens(rng)
    params = looped.init(rng, ids)
    tree = dict(params['params'])
    shared = {name: value for name, value in tree.items() if not name.startswith('layers_')}
    order = [0, 1, 2, 1, 2, 3]
    repeated = {f'layers_{index}': tree[f'layers_{source}'] for index, source in enumerate(order)}
    deep_params = {'params': {**shared, **repeated}}
    deep = tiny(num_layers=6, loop=None)
    np.testing.assert_array_equal(looped.apply(params, ids), deep.apply(deep_params, ids))
    ours, cache = decoded(looped, params, ids)
    theirs, deep_cache = decoded(deep, deep_params, ids)
    np.testing.assert_array_equal(ours, theirs)
    # The second pass's block sits after the decoder's own layers: deep layers 3, 4 at stored 4, 5.
    for stored, deep_index in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4), (3, 5)):
        jax.tree.map(np.testing.assert_array_equal, cache[f'layers_{stored}'],
                     deep_cache[f'layers_{deep_index}'])
    assert_as_exact(lambda params, ids: decoded(looped, params, ids)[0], jax.jit(looped.apply), drawn(looped),
                    'block loop decode', seeds=SEEDS['block decode'], truth_fn=wide(looped))
    scanned = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)), scan_layers=True)
    assert_as_exact(jax.jit(scanned.apply), jax.jit(looped.apply), drawn(looped), 'scanned block loop',
                    seeds=SEEDS['scanned'], truth_fn=wide(looped))


def test_a_decoder_gains_a_loop_over_its_own_block_that_one_pass_leaves_as_it_was(rng):
    plain = tiny(num_layers=4, loop=None)
    ids = tokens(rng)
    params = plain.init(rng, ids)
    once, twice = plain.with_passes(1, (1, 3)), plain.with_passes(2, (1, 3))
    assert twice.loop == Loop(2, step_norm=False, layers=(1, 3))
    np.testing.assert_array_equal(once.apply(params, ids), plain.apply(params, ids))
    assert not np.array_equal(twice.apply(params, ids), plain.apply(params, ids))
    assert plain.with_passes(2) is None and plain.with_passes(None, (1, 3)) is None
    assert twice.with_passes(4).loop == Loop(4, step_norm=False, layers=(1, 3))


DELTA_NETS = {'linear_attention': {'mixer': {'class': 'gated_delta_net', 'fields': {
    'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_key_head_dim': 6,
    'linear_value_head_dim': 8, 'linear_conv_kernel_dim': 4}}}}
"""A stack of gated delta nets, whose decode state is a recurrent memory and a convolution tail."""


def test_each_pass_of_a_shared_cache_step_starts_from_the_cache_the_step_found(rng):
    """Every pass of a `Loop(shared_cache=True)` step starts from the cache as the step found it
    (`SharedPassView`), its recurrent memory and convolution tail with its cursors: allocating the
    cache leaves every entry zero, and a stack of gated delta nets prefills, bit for bit, what the same
    loop with a cache of its own per pass prefills, where the second pass took the prompt again over
    the state the first pass left (audit NN-A1)."""
    models = [tiny(loop=Loop(2, shared_cache=shared), layer_types=('linear_attention',) * 2, kinds=DELTA_NETS)
              for shared in (True, False)]
    ids = tokens(rng)
    params = models[0].init(rng, ids)
    prefilled = []
    for model in models:
        made, step = decode_steps(model)
        cache = made(params, ids.shape[0])
        assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
        prefilled.append(step(params, cache, ids)[0])
    np.testing.assert_array_equal(*prefilled)
