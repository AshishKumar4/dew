"""The layers' gates and gated norm in the chunked op (row 1967): a gated delta net's chunked call on the
hooks its decode token takes, and Kimi delta attention's decay's gate law, on par with the recurrence a token
at a time, and the recurrence template's output launch taking the norm (`templates.recurrence.output.Normed`),
on par with the norm after the launches."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_recurrence_chunk import CHUNK, KEY_WIDTH, PACKED, chunk_on, draw, hosted, token_by_token

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.recurrence import mainloop, template
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
"""The gated delta rule's gates and gated norm (`GatedDeltaNet._chunk_hooks`); Kimi's gate law, its decay a
key dimension's bounded below (`KimiDeltaAttention._decay`), with its sigmoid-gated norm; and a decay a key
dimension's softplus step, unbounded. Their norms read the output as it comes (a read-back dtype narrows the
float64 truth)."""


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
    expected = recurrence.output_normed(hooks.output, raw, operands.z, operands.weight, jnp.bfloat16,
                                        jnp.float32)
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



def test_a_padded_tokens_overflowing_key_leaves_the_live_memory_as_without_it():
    """A delta rule's chunk of four tokens, the third padded (`Chunk.valid`) with its key and value `2^70`:
    the final memory and the live tokens' outputs those of the same call with the padded token's operands
    zero, bit for bit, every token operand of a padded token zeroed before any product, where its key's
    self-product overflowed and `inf * 0` crossed the chunk's inverse into the live memory, NaN (train 1461's
    review)."""
    hooks = recurrence.Hooks(decay='scalar')
    rng = np.random.default_rng(1967)
    q, k, v = (jnp.asarray(rng.standard_normal((1, 4, 1, 8)) / 4, jnp.float32) for _ in range(3))
    valid = jnp.asarray([[True, True, False, True]])

    def call(k, v):
        operands = Chunk(q, k, v, log_decay=jnp.full((1, 4, 1), -0.25), beta=jnp.full((1, 4, 1), 0.5),
                         valid=valid)
        return chunk_on('xla', hooks, operands)

    found = call(k.at[0, 2].set(2.0 ** 70), v.at[0, 2].set(2.0 ** 70))
    expected = call(k.at[0, 2].set(0.0), v.at[0, 2].set(0.0))
    live = np.asarray([0, 1, 3])
    np.testing.assert_array_equal(np.asarray(found[0])[0, live], np.asarray(expected[0])[0, live])
    np.testing.assert_array_equal(np.asarray(found[1]), np.asarray(expected[1]))


def test_a_float64_recurrences_norm_read_back_in_fp32_keeps_its_gates_precision():
    """A float64 delta rule's one token writing ones under `e0` with unit query, its sigmoid-gated norm (`eps`
    1e-6) reading the output back in fp32 (`OutputNorm.read_in`) under a gate of -100: `sigmoid(-100) / sqrt(1
    + 1e-6)` each (its binade), the read-back widened to the recurrence's float64 before the norm, where the
    norm worked in the read-back's fp32 and its gate's `exp(100)` overflowed to a zero output (train 1461's
    review)."""
    hooks = recurrence.Hooks(decay='scalar', output=recurrence.OutputNorm(1e-6, 'sigmoid', read_in='float32'))
    with jax.enable_x64(True):
        unit = jnp.zeros((1, 1, 1, 8), jnp.float64).at[..., 0].set(1.0)
        operands = Chunk(unit, unit, jnp.ones((1, 1, 1, 8), jnp.float64), log_decay=jnp.zeros((1, 1, 1)),
                         beta=jnp.ones((1, 1, 1), jnp.float64), z=jnp.full((1, 1, 1, 8), -100.0, jnp.float64),
                         weight=jnp.ones((8,), jnp.float64))
        found, _ = chunk_on('xla', hooks, operands)
        # `sigmoid(-100)`'s binade (a zero where the gate overflowed).
        assert (np.frexp(np.asarray(found))[1] == np.frexp(np.exp(-100.0))[1]).all(), found


def test_a_float64_recurrences_rate_is_exponentiated_in_its_work_dtype():
    """A float64 delta rule's gates from their pre-activations (`Gates`) under an fp32 `log_a` of 100, its
    gate -100 and bias 0 (a decay of `-exp(100) softplus(-100)`, -1), a zero key and a memory of ones
    entering: the memory leaves `exp(-1)` of itself (its binade), the rate exponentiated in the recurrence's
    float64, where in its own fp32 `exp(100)` overflowed and cleared the memory (train 1461's review)."""
    hooks = recurrence.Hooks(decay='scalar', gates=recurrence.Gates())
    with jax.enable_x64(True):
        zeros = jnp.zeros((1, 1, 1, 8), jnp.float64)
        operands = Chunk(zeros, zeros, zeros, gate=jnp.full((1, 1, 1), -100.0, jnp.float64),
                         log_a=jnp.full((1,), 100.0, jnp.float32), bias=jnp.zeros((1,), jnp.float64),
                         beta=jnp.zeros((1, 1, 1), jnp.float64), state=jnp.ones((1, 1, 8, 8), jnp.float64))
        _, final = chunk_on('xla', hooks, operands)
        # `exp(-1)`'s binade (a zero where the rate overflowed).
        assert (np.frexp(np.asarray(final))[1] == np.frexp(np.exp(-1.0))[1]).all(), final


EPILOGUE_BLOCKS = {'one value block': 64, 'two value blocks': 32}
"""The output launch's value blocks over a head's 64 values: the row its one block, normalized in registers
past the walk, and two, its blocks staged in the launch's own tile across the walk (`output._normed_read`)."""


def epilogue_draw(seed: int):
    """A bf16 delta rule over 128 tokens (two chunks) of two value heads, 64 keys and values each, its
    decays, write gates, sigmoid output gate and norm weight."""
    rng = np.random.default_rng([1967, 7, seed])

    def normal(*shape, scale=1.0, dtype=jnp.float32):
        return jnp.asarray(rng.standard_normal(shape) * scale, dtype)

    q, k = (normal(1, 128, 1, 64, scale=0.125, dtype=jnp.bfloat16) for _ in range(2))
    return (q, k, normal(1, 128, 2, 64, dtype=jnp.bfloat16), -jnp.abs(normal(1, 128, 2, scale=0.1)),
            jax.nn.sigmoid(normal(1, 128, 2)), normal(1, 128, 2, 64), 1 + normal(64, scale=0.1))


def epilogue_run(block: int, fused: bool, training: bool):
    """The template's rule at the output launch's value `block` (`template.mosaic_rule`, interpreted off a
    GPU) with the sigmoid-gated norm in its output launch (`fused`), or after its launches
    (`recurrence.output_normed`): its output, or in `training` the gradients of its output's weighted sum by
    every operand."""
    norm = recurrence.OutputNorm(1e-6, 'sigmoid', read_in='bfloat16')
    tiles = mainloop.Tiles(32, 1, block_o=block, chunk=64)

    def output(q, k, v, g, beta, z, weight):
        if fused:
            out, _ = template.mosaic_rule(q, k, v, g, beta, tiles=tiles, scale=0.125, normed=norm, gate=z,
                                          weight=weight)
            return out
        out, _ = template.mosaic_rule(q, k, v, g, beta, tiles=tiles, scale=0.125)
        return recurrence.output_normed(norm, out, z, weight, jnp.bfloat16, jnp.float32)

    if not training:
        return output
    weights = np.random.default_rng(1967).standard_normal((1, 128, 2, 64)).astype(np.float32)
    return jax.grad(lambda *x: jnp.sum(output(*x).astype(jnp.float32) * weights), argnums=tuple(range(7)))


@pytest.mark.parametrize('training', [False, True], ids=['forward', 'training'])
@pytest.mark.parametrize('block', EPILOGUE_BLOCKS.values(), ids=EPILOGUE_BLOCKS)
def test_the_output_launchs_norm_is_on_par_with_the_norm_after_it(block, training):
    """The template's rule with the norm in its output launch (`epilogue_run`'s fused), at one value block
    and at two, its forward (the launch's epilogue) and its training (the norm's read-back kept and its
    gradient the norm's own before the fast backward's): on par from float64 with the same launches' norm
    after them (check 1, the float64 composition the truth), the launch taking the epilogue itself, no route
    deciding it (train 1461's review)."""
    # The truth's norm reads the exact output (a read-back dtype narrows the float64 truth).
    hooks = recurrence.Hooks(decay='scalar', scale=0.125, output=recurrence.OutputNorm(1e-6, 'sigmoid'))

    def truth(*operands):
        with jax.enable_x64(True):
            wide = [jnp.asarray(x, jnp.float64) for x in operands]

            def composed(q, k, v, g, beta, z, weight):
                with generation.chosen(gated_delta_rule='xla'):
                    operands = Chunk(q, k, v, log_decay=g, beta=beta, z=z, weight=weight)
                    return chunk(hooks, operands, dtype=jnp.float64)[0]

            if not training:
                return composed(*wide)
            weights = np.random.default_rng(1967).standard_normal((1, 128, 2, 64))
            return jax.grad(lambda *x: jnp.sum(composed(*x) * weights), argnums=tuple(range(7)))(*wide)

    fused, apart = (jax.jit(epilogue_run(block, kept, training)) for kept in (True, False))
    assert_as_exact(fused, apart, epilogue_draw, f'the output launch norm at {block} {training=}', seeds=8,
                    truth_fn=truth)


def test_a_skip_beside_the_norm_declines_the_epilogue_on_par_with_the_portable_rule():
    """A bf16 chunked call on the template without the delta rule, with a skip and the gated norm: the
    epilogue declined (the output launch adds no skip before a norm, `_normed_runs`' `fuses`), the norm
    after the recurrence, on par from float64 with the portable rule's (check 1)."""
    hooks = recurrence.Hooks(decay='scalar', delta=False, skip=True, scale=0.125,
                             output=recurrence.OutputNorm(1e-6, read_in='bfloat16'))

    def sample(seed):
        q, k, v, g, beta, z, weight = epilogue_draw(seed)
        return (Chunk(q, k, v, log_decay=g, beta=beta, skip=jnp.asarray([0.5, -0.25]), z=z, weight=weight),)

    template_rule, portable = (jax.jit(functools.partial(chunk_on, kernel, hooks))
                               for kernel in ('mosaic_gpu', 'xla'))
    assert_as_exact(template_rule, portable, sample, 'a skip declining the epilogue', seeds=8)


def test_a_padded_tokens_infinite_gate_leaves_the_norm_weights_gradient_finite():
    """A width-one SiLU-gated norm (`eps` 1) over two tokens, the second padded (`Chunk.valid`) with an
    infinite output gate, the first's unit query, key, value and write: the gradient of the live output by the
    shared norm weight the one-token call's, the padded token's gate zeroed before the norm, where its
    infinity met its zero cotangent as NaN (a head review)."""
    hooks = recurrence.Hooks(decay='scalar', output=recurrence.OutputNorm(1.0, 'silu'))
    ones = jnp.ones((1, 2, 1, 1), jnp.float32)

    def live(weight, z, valid):
        operands = Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 2, 1)), beta=jnp.ones((1, 2, 1)), z=z,
                         weight=weight, valid=valid)
        return chunk_on('xla', hooks, operands)[0][0, 0, 0, 0]

    weight = jnp.ones((1,), jnp.float32)
    valid = jnp.asarray([[True, False]])
    found = jax.grad(live)(weight, jnp.asarray([1.0, np.inf]).reshape(1, 2, 1, 1), valid)
    expected = jax.grad(live)(weight, jnp.ones((1, 2, 1, 1)), valid)
    np.testing.assert_array_equal(np.asarray(found), np.asarray(expected))
