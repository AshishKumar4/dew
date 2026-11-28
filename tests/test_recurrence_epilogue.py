"""The layers' gates and gated norm in the chunked op (row 1967): a gated delta net's and Kimi delta
attention's chunked call on the hooks their decode tokens take, on par with the recurrence a token at a time,
and the recurrence template's output launch taking the norm (`templates.recurrence.output.Normed`), on par
with the norm after the launches."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_recurrence_chunk import CHUNK, KEY_WIDTH, PACKED, chunk_on, draw, hosted, token_by_token

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk
from dew.nn.linear import GatedDeltaNet

LAYER_HOOKS = {
    'gated delta net': recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), scale=KEY_WIDTH ** -0.5,
                                        output=recurrence.OutputNorm(1e-6), write_bound=1.0),
    'kimi delta attention': recurrence.Hooks(decay='vector', gates=recurrence.Gates(bound=-5.0),
                                             scale=KEY_WIDTH ** -0.5,
                                             output=recurrence.OutputNorm(1e-5, 'sigmoid'), write_bound=1.0,
                                             decay_bound=-5.0),
    'a key dimension softplus step': recurrence.Hooks(decay='vector', gates=recurrence.Gates(), l2norm=True),
}
"""The layers' chunked calls (`GatedDeltaNet._chunk_hooks`, `KimiDeltaAttention._chunk_hooks`): the gated
delta rule's gates and gated norm; Kimi's decay a key dimension's bounded below and its sigmoid-gated norm;
and a decay a key dimension's softplus step, unbounded. Their norms read the output as it comes (a read-back
dtype narrows the float64 truth)."""


def keyed_draw(hooks: recurrence.Hooks, packed: bool, seed: int):
    """`test_recurrence_chunk.draw`'s operands, a decay a key dimension's gate and bias each key
    dimension's."""
    (laid,) = draw(hooks, packed, seed)
    if hooks.decay != 'vector':
        return (laid,)
    rng = np.random.default_rng([1967, seed])
    width = laid.key.shape[-1]
    gate, bias = (rng.standard_normal((*x.shape, width)).astype(np.float32) for x in (laid.gate, laid.bias))
    return (dataclasses.replace(laid, gate=hosted(gate), bias=hosted(bias)),)


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', LAYER_HOOKS.values(), ids=LAYER_HOOKS)
def test_a_layers_chunked_call_is_as_exact_as_its_tokens(hooks, packed):
    """The chunked op's portable math on a layer's hooks, its gates formed from their pre-activations and its
    output through the gated norm, against the decode step's reference a token at a time, output and final
    memory, from float64 (check 1)."""
    dew = jax.jit(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(keyed_draw, hooks, packed), f'{hooks} packed={packed}',
                    seeds=16)


def gated_delta_net_epilogue(training: bool, seeds: int):
    """A bf16 gated delta net naming the template, its heads' 256 values over several of the output launch's
    value blocks and its 160 tokens over three chunks: its output (or in `training` its gradients by its
    input and every parameter) on par from float64 with the portable rule's (check 1, the layer's own float64
    step the truth), the output launch taking the norm, interpreted, over `seeds` draws."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=256,
                          dtype=jnp.bfloat16)
    weights = jax.random.normal(jax.random.key(31), (1, 160, 64), jnp.float32)

    def sample(seed):
        drawn, initial = jax.random.split(jax.random.key(1967 + seed))
        x = jax.random.normal(drawn, (1, 160, 64)).astype(jnp.bfloat16)
        with generation.chosen(gated_delta_rule='xla'):
            return layer.init(initial, x), x

    def stepped(layer, params, x):
        if not training:
            return layer.apply(params, x)

        def loss(params, x):
            out = layer.apply(params, x)
            # The loss at least fp32, float64 in the truth's own; its gradients judged, the outputs the
            # forward's (one scalar a draw resolved no factor over 40).
            return jnp.sum(out.astype(jnp.promote_types(out.dtype, jnp.float32)) * weights)
        return jax.grad(loss, argnums=(0, 1))(params, x)

    def applied(kernel):
        def run(params, x):
            with generation.chosen(gated_delta_rule=kernel):
                return stepped(layer, params, x)
        return run

    assert_as_exact(applied('mosaic_gpu'), applied('xla'), sample,
                    f'the norm in the output launch, {training=}', seeds=seeds,
                    truth_fn=functools.partial(stepped, layer.clone(dtype=jnp.float64)))


def test_a_gated_delta_nets_norm_in_the_output_launch_is_on_par_with_the_portable_rule():
    """`gated_delta_net_epilogue`'s forward: the output launch's epilogue, a forward without a gradient's."""
    gated_delta_net_epilogue(training=False, seeds=8)


@pytest.mark.check1_shards(seeds=80, per_task=8)
def test_a_gated_delta_nets_training_through_its_epilogue_is_on_par_with_the_portable_rule():
    """`gated_delta_net_epilogue`'s training: the template's training forward keeps the output its norm read,
    whose gradient the norm's own forms before the fast backward's (`template.mosaic_rule`). 80 draws, 8 a
    task: at 40 its `dt_bias` gradient was inconclusive (interval [0.899, 1.507] across 1.5)."""
    gated_delta_net_epilogue(training=True, seeds=80)


def test_a_norm_reading_its_output_in_fp32_reads_the_kernels_unrounded_output():
    """A bf16 token through fla's port, its norm reading the output back in fp32 (`OutputNorm.read_in`): the
    norm of the port's fp32 output, where the output landed in bf16 first moved the norm across a bf16
    midpoint (a head review: 0.251953125 for 0.25)."""
    hooks = recurrence.Hooks(decay='scalar', scale=0.1,
                             output=recurrence.OutputNorm(0.02972, 'sigmoid', read_in='float32'))
    unit = jnp.zeros((1, 1, 1, 8), jnp.bfloat16).at[..., 0].set(1.0)
    operands = Chunk(unit, unit, jnp.ones((1, 1, 1, 8), jnp.bfloat16), log_decay=jnp.zeros((1, 1, 1)),
                     beta=jnp.ones((1, 1, 1)), z=jnp.zeros((1, 1, 1, 8)), weight=jnp.ones((8,)))
    raw, _ = chunk_on('fla', dataclasses.replace(hooks, output=None), operands, dtype=jnp.float32)
    found, _ = chunk_on('fla', hooks, operands, dtype=jnp.bfloat16)
    expected = recurrence.output_normed(hooks.output, raw, operands.z, operands.weight, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


def test_a_padded_token_keeps_the_memory_its_value_decay_would_halve():
    """A padded token (`Chunk.valid`) under a value decay of `-log 2` (`Hooks.value_decay`), a zero query, key
    and value and a memory of ones entering: the memory leaves as it came, its value decay none, where the
    value decay alone left unmasked halved it (a head review)."""
    hooks = recurrence.Hooks(decay='scalar', delta=False, value_decay=True)
    zeros = jnp.zeros((1, 1, 1, 4), jnp.float32)
    operands = Chunk(zeros, zeros, zeros, log_decay=jnp.zeros((1, 1, 1)), state=jnp.ones((1, 1, 4, 4)),
                     value_decay=jnp.full((1, 1, 1, 4), -np.log(2.0), jnp.float32),
                     valid=jnp.zeros((1, 1), bool))
    _, final = chunk_on('xla', hooks, operands)
    np.testing.assert_array_equal(np.asarray(final), np.ones((1, 1, 4, 4), np.float32))

