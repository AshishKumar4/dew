"""A looped stack (`Loop`): passes that share the layers' parameters, each
with its own entries in the decode cache."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop

VOCAB = 37


def tiny(**overrides):
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "attention_impl": "reference",
                                "tie_embeddings": False, "loop": Loop(3, exit_gate=True), **overrides})


def tokens(rng, length=12):
    return jax.random.randint(rng, (2, length), 0, VOCAB)


def decoded(model, params, ids, prompt=4):
    """Prefill `prompt` tokens, then decode the rest one at a time: each
    position's logits and the cache after the last."""
    _, cache = model.apply(params, ids.shape[0], method=CausalTransformer.init_cache, mutable=['cache'])
    logits, cache = model.apply({**params, **cache}, ids[:, :prompt], decode=True, mutable=['cache'])
    steps = [logits]
    for position in range(prompt, ids.shape[1]):
        logits, cache = model.apply({**params, **cache}, ids[:, position:position + 1], decode=True,
                                    mutable=['cache'])
        steps.append(logits)
    return jnp.concatenate(steps, axis=1), cache['cache']


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
    np.testing.assert_allclose(looped.apply(params, ids), deep.apply(unrolled(params, 2), ids), atol=1e-5)
    ours, cache = decoded(looped, params, ids)
    theirs, deep_cache = decoded(deep, unrolled(params, 2), ids)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
    assert jax.tree.structure(cache) == jax.tree.structure(deep_cache)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5), cache, deep_cache)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_decoding_a_loop_matches_its_prefill(rng, scan_layers):
    model = tiny(scan_layers=scan_layers)
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    incremental, cache = decoded(model, params, ids)
    np.testing.assert_allclose(incremental, full, atol=1e-5)
    assert sorted(cache) == [f'layers_{index}' for index in range(6)]


def test_scanned_runs_loop_as_the_plain_layers_do(rng):
    ids = tokens(rng)
    params = tiny().init(rng, ids)
    scanned = tiny(scan_layers=True).apply(params, ids)
    np.testing.assert_allclose(scanned, tiny().apply(params, ids), atol=1e-5)


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
def test_checkpointed_passes_preserve_loss_and_gradients(rng, backprop_steps, scan_layers):
    """Three passes over layers [1, 3) of four share fp32 loss and gradients,
    including the cut before the last pass. The tolerance allows different
    backward fusions when residuals are recomputed instead of retained.
    """
    loop = Loop(3, layers=(1, 3), backprop_steps=backprop_steps, remat=False)
    plain = tiny(num_layers=4, loop=loop, scan_layers=scan_layers, dtype="float32")
    checkpointed = plain.clone(loop=dataclasses.replace(loop, remat=True))
    ids = tokens(rng)
    params = plain.init(rng, ids)

    def result(model):
        return jax.jit(jax.value_and_grad(lambda tree: jnp.mean(
            model.apply(tree, ids, train=True) ** 2)))(params)

    expected, actual = result(plain), result(checkpointed)
    gradient_error = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                         for a, b in zip(jax.tree.leaves(actual[1]), jax.tree.leaves(expected[1]),
                                         strict=True))
    print(f"loop fp32: scan={scan_layers}, backprop_steps={backprop_steps}, "
          f"loss_error={float(jnp.abs(actual[0] - expected[0]))}, gradient_error={gradient_error}")
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6), actual, expected)
    assert bool(jnp.any(actual[1]['params']['embed_tokens']['embedding'])) == (backprop_steps is None)
    assert bool(jnp.any(actual[1]['params']['layers_1']['mlp']['down_proj']['kernel']))


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
    layers whose fourth and fifth repeat the second and third, decode as they
    prefill, and keep the second pass's entries after the first's."""
    looped = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)))
    ids = tokens(rng)
    params = looped.init(rng, ids)
    tree = dict(params['params'])
    shared = {name: value for name, value in tree.items() if not name.startswith('layers_')}
    order = [0, 1, 2, 1, 2, 3]
    repeated = {f'layers_{index}': tree[f'layers_{source}'] for index, source in enumerate(order)}
    deep_params = {'params': {**shared, **repeated}}
    deep = tiny(num_layers=6, loop=None)
    np.testing.assert_allclose(looped.apply(params, ids), deep.apply(deep_params, ids), atol=1e-5)
    ours, cache = decoded(looped, params, ids)
    np.testing.assert_allclose(ours, looped.apply(params, ids), atol=1e-5)
    theirs, deep_cache = decoded(deep, deep_params, ids)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
    # The second pass's block sits after the decoder's own layers: deep layers 3, 4 at stored 4, 5.
    for stored, deep_index in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4), (3, 5)):
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5),
                     cache[f'layers_{stored}'], deep_cache[f'layers_{deep_index}'])
    scanned = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)), scan_layers=True)
    np.testing.assert_allclose(scanned.apply(params, ids), looped.apply(params, ids), atol=1e-5)


def test_a_decoder_gains_a_loop_over_its_own_block_that_one_pass_leaves_as_it_was(rng):
    plain = tiny(num_layers=4, loop=None)
    ids = tokens(rng)
    params = plain.init(rng, ids)
    once, twice = plain.with_passes(1, (1, 3)), plain.with_passes(2, (1, 3))
    assert twice.loop == Loop(2, step_norm=False, layers=(1, 3))
    np.testing.assert_array_equal(once.apply(params, ids), plain.apply(params, ids))
    assert not np.allclose(twice.apply(params, ids), plain.apply(params, ids))
    assert plain.with_passes(2) is None and plain.with_passes(None, (1, 3)) is None
    assert twice.with_passes(4).loop == Loop(4, step_norm=False, layers=(1, 3))
