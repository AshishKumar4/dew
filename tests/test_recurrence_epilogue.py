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
from generation_support import COMPILES_ON_CPU, lowers
from reference_error import assert_as_exact
from test_recurrence_chunk import CHUNK, KEY_WIDTH, PACKED, chunk_on, draw, hosted, token_by_token

from dew.nn import norms
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation, norm_swiglu, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.hooks import RowHook, Side
from dew.nn.kernels.templates.recurrence import mainloop, template
from dew.nn.linear import GatedDeltaNet
from dew.nn.mixers.mamba2 import MambaRMSNormGated

LAYER_HOOKS = {
    'gated delta net': recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), scale=KEY_WIDTH ** -0.5,
                                        output=norms.OutputNorm(1e-6), write_bound=1.0),
    'kimi delta attention': recurrence.Hooks(decay='vector', gates=recurrence.Gates(bound=-5.0),
                                             scale=KEY_WIDTH ** -0.5,
                                             output=norms.OutputNorm(1e-5, 'sigmoid'), write_bound=1.0,
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
                             output=norms.OutputNorm(0.02972, 'sigmoid', read_in='float32'))
    unit = jnp.zeros((1, 1, 1, 8), jnp.bfloat16).at[..., 0].set(1.0)
    operands = Chunk(unit, unit, jnp.ones((1, 1, 1, 8), jnp.bfloat16), log_decay=jnp.zeros((1, 1, 1)),
                     beta=jnp.ones((1, 1, 1)), z=jnp.zeros((1, 1, 1, 8)), weight=jnp.ones((8,)))
    raw, _ = chunk_on('fla', dataclasses.replace(hooks, output=None), operands, dtype=jnp.float32)
    found, _ = chunk_on('fla', hooks, operands, dtype=jnp.bfloat16)
    expected = norms.output_normed(hooks.output, raw, operands.z, operands.weight, jnp.bfloat16,
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



@pytest.mark.parametrize('kernel', ['fla', 'xla'])
def test_a_padded_tokens_overflowing_key_leaves_the_live_memory_as_without_it(kernel):
    """A bf16 delta rule's two tokens on `kernel`, the second padded (`Chunk.valid`) with its key `2^70 e0`:
    the final memory and the live token's output those of the same call with the padded token's operands
    zero, bit for bit, every token operand of a padded token zeroed before any product, where its key's
    self-product overflowed fp32 and neither the pairs' zero diagonal nor the zero write gate erased the
    infinity, NaN through the chunk's inverse into the live memory (train 1461's review)."""
    hooks = recurrence.Hooks(decay='scalar')
    rng = np.random.default_rng(1967)
    q, k, v = (jnp.asarray(rng.standard_normal((1, 2, 1, 64)) / 8, jnp.bfloat16) for _ in range(3))
    valid = jnp.asarray([[True, False]])

    def call(k):
        operands = Chunk(q, k, v, log_decay=jnp.full((1, 2, 1), -0.25), beta=jnp.full((1, 2, 1), 0.5),
                         valid=valid)
        return chunk_on(kernel, hooks, operands)

    found = call(k.at[0, 1].set(0.0).at[0, 1, 0, 0].set(2.0 ** 70))
    expected = call(k.at[0, 1].set(0.0))
    live = (np.asarray(out, np.float32)[0, 0] for out in (found[0], expected[0]))
    np.testing.assert_array_equal(*live)
    np.testing.assert_array_equal(np.asarray(found[1]), np.asarray(expected[1]))


def test_a_float64_recurrences_norm_read_back_in_fp32_keeps_its_gates_precision():
    """A float64 delta rule's one token writing ones under `e0` with unit query, its sigmoid-gated norm (`eps`
    1e-6) reading the output back in fp32 (`OutputNorm.read_in`) under a gate of -100: `sigmoid(-100) / sqrt(1
    + 1e-6)` each (its binade), the read-back widened to the recurrence's float64 before the norm, where the
    norm worked in the read-back's fp32 and its gate's `exp(100)` overflowed to a zero output (train 1461's
    review)."""
    hooks = recurrence.Hooks(decay='scalar', output=norms.OutputNorm(1e-6, 'sigmoid', read_in='float32'))
    with jax.enable_x64(new_val=True):
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
    with jax.enable_x64(new_val=True):
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


NORM_FORMS = {
    'norm before gate': {},
    'gate before norm, weight rounded': {'gate_first': True, 'rounded': True},
}
"""The epilogue's norm forms (`norms.OutputNorm`): Qwen3-Next's and Kimi's, and Mamba-2's
`MambaRMSNormGated`'s."""


def epilogue_run(block: int, fused: bool, training: bool, form: dict | None = None):
    """The template's rule at the output launch's value `block` (`template.mosaic_rule`, interpreted off a
    GPU) with the sigmoid-gated norm of `form` (`NORM_FORMS`) in its output launch (`fused`), or after its
    launches (`norms.output_normed`): its output, or in `training` the gradients of its output's
    weighted sum by every operand."""
    norm = norms.OutputNorm(1e-6, 'sigmoid', read_in='bfloat16', **(form or {}))
    tiles = mainloop.Tiles(32, 1, block_o=block, chunk=64)

    def output(q, k, v, g, beta, z, weight):
        if fused:
            out, _ = template.mosaic_rule(q, k, v, g, beta, tiles=tiles, scale=0.125, normed=norm, gate=z,
                                          weight=weight)
            return out
        out, _ = template.mosaic_rule(q, k, v, g, beta, tiles=tiles, scale=0.125)
        return norms.output_normed(norm, out, z, weight, jnp.bfloat16, jnp.float32)

    if not training:
        return output
    weights = np.random.default_rng(1967).standard_normal((1, 128, 2, 64)).astype(np.float32)
    return jax.grad(lambda *x: jnp.sum(output(*x).astype(jnp.float32) * weights), argnums=tuple(range(7)))


@pytest.mark.parametrize('form', NORM_FORMS.values(), ids=NORM_FORMS)
@pytest.mark.parametrize('training', [False, True], ids=['forward', 'training'])
@pytest.mark.parametrize('block', EPILOGUE_BLOCKS.values(), ids=EPILOGUE_BLOCKS)
def test_the_output_launchs_norm_is_on_par_with_the_norm_after_it(block, training, form):
    """The template's rule with the norm in its output launch (`epilogue_run`'s fused), at one value block
    and at two, its forward (the launch's epilogue) and its training (the norm's read-back kept and its
    gradient the norm's own before the fast backward's): on par from float64 with the same launches' norm
    after them (check 1, the float64 composition the truth), the launch taking the epilogue itself, no route
    deciding it (train 1461's review), each norm form (`NORM_FORMS`) the launch's row hook (train 1473's)."""
    # The truth's norm reads the exact output (a read-back dtype narrows the float64 truth).
    norm = norms.OutputNorm(1e-6, 'sigmoid', **form)
    hooks = recurrence.Hooks(decay='scalar', scale=0.125, output=norm)

    def truth(*operands):
        with jax.enable_x64(new_val=True):
            wide = [jnp.asarray(x, jnp.float64) for x in operands]

            def composed(q, k, v, g, beta, z, weight):
                with generation.chosen(gated_delta_rule='xla'):
                    operands = Chunk(q, k, v, log_decay=g, beta=beta, z=z, weight=weight)
                    return chunk(hooks, operands, dtype=jnp.float64)[0]

            if not training:
                return composed(*wide)
            weights = np.random.default_rng(1967).standard_normal((1, 128, 2, 64))
            return jax.grad(lambda *x: jnp.sum(composed(*x) * weights), argnums=tuple(range(7)))(*wide)

    fused, apart = (jax.jit(epilogue_run(block, kept, training, form)) for kept in (True, False))
    assert_as_exact(fused, apart, epilogue_draw, f'the output launch norm at {block} {training=} {form}',
                    seeds=8, truth_fn=truth)


@COMPILES_ON_CPU
@lowers('sm90')
@pytest.mark.parametrize('training', [False, True], ids=['forward', 'training'])
@pytest.mark.parametrize('block', EPILOGUE_BLOCKS.values(), ids=EPILOGUE_BLOCKS)
def test_the_output_launchs_norm_lowers_on_a_hopper(block, training):
    """`epilogue_run`'s fused rule at each value block, its forward and its training, compiled ahead for an
    H100 (`kernel_compile.compile_mosaic_kernels`, compile-only): every launch lowers within its shared
    memory, where the epilogue's rows staged in a plain tile had no Hopper transfer plan for a product
    layout's store (M-JS-324c2's GH200 run)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, failures

    run = epilogue_run(block, fused=True, training=training)
    operands = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in epilogue_draw(0))
    case = KernelCase(f'the epilogue at {block} {training=}', lambda: (run, operands))
    found = compile_mosaic_kernels([case], 'sm90')
    assert not failures(found), failures(found)


def test_a_skip_beside_the_norm_declines_the_epilogue_on_par_with_the_portable_rule():
    """A bf16 chunked call on the template without the delta rule, with a skip and the gated norm: the
    epilogue declined (the output launch adds no skip before a norm, `_normed_runs`' `fuses`), the norm
    after the recurrence, on par from float64 with the portable rule's (check 1)."""
    hooks = recurrence.Hooks(decay='scalar', delta=False, skip=True, scale=0.125,
                             output=norms.OutputNorm(1e-6, read_in='bfloat16'))

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
    hooks = recurrence.Hooks(decay='scalar', output=norms.OutputNorm(1.0, 'silu'))
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


def test_a_named_norm_kernel_keeps_the_norm_after_the_launches():
    """A bf16 gated delta rule on the template whose norm reads its output back in bf16, the output launch's
    epilogue taking it (no norm's route reported), and under `dew.kernels.use(rms_norm='xla')` that kernel
    after the launches, its route reported (train 1473's review)."""
    from dew import kernels

    hooks = recurrence.Hooks(decay='scalar', scale=0.125,
                             output=norms.OutputNorm(1e-6, 'sigmoid', read_in='bfloat16'))
    q, k, v, g, beta, z, weight = epilogue_draw(0)
    operands = Chunk(q, k, v, log_decay=g, beta=beta, z=z, weight=weight)

    def norm_routes(*named):
        with kernels.use(**dict(named)), generation.routes() as found:
            jax.eval_shape(functools.partial(chunk_on, 'mosaic_gpu', hooks), operands)
        return [route.kernel for route in found if route.choice == generation.Choice.RMS_NORM]

    assert norm_routes() == []
    assert norm_routes(('rms_norm', 'xla')) == ['xla']


def test_an_idle_chunks_rate_gradient_is_zero_whatever_its_rate():
    """A gated delta rule's two tokens on the portable path, both padded (`Chunk.valid`, a runtime mask), its
    head's rate `log_a` 100: its gradient by the rate exactly zero, a padded token's gates formed from zeros,
    its rate among them (`recurrence.log_gates`), where `exp(100)` overflowed fp32 before the padding masked
    its decay and its zero cotangent times it was NaN (train 1473's review)."""
    hooks = recurrence.Hooks(decay='scalar', gates=recurrence.Gates())
    rng = np.random.default_rng(1473)
    q, k, v = (jnp.asarray(rng.standard_normal((1, 2, 1, 16)) / 4, jnp.float32) for _ in range(3))

    def total(log_a, valid):
        operands = Chunk(q, k, v, gate=jnp.zeros((1, 2, 1)), beta=jnp.zeros((1, 2, 1)), log_a=log_a,
                         bias=jnp.zeros((1,)), valid=valid)
        return jnp.sum(chunk_on('xla', hooks, operands)[0])

    found = jax.jit(jax.grad(total))(jnp.full((1,), 100.0), jnp.asarray([[False, False]]))
    np.testing.assert_array_equal(np.asarray(found), np.zeros((1,), np.float32))


MAMBA_FORMS = {'gate after wide': (False, False), 'gate after rounded': (False, True),
               'gate first wide': (True, False), 'gate first rounded': (True, True)}
"""`OutputNorm`'s `(gate_first, rounded)` forms, `MambaRMSNormGated`'s four."""


@pytest.mark.parametrize('form', MAMBA_FORMS.values(), ids=MAMBA_FORMS)
def test_an_output_norms_form_is_mamba2s_gated_norm_over_a_head(form):
    """`norms.output_normed` over bf16 rows of two heads of 64 values, each `OutputNorm` form (the gate
    first or after, the weight wide or the normalized rows rounded first): bit for bit `MambaRMSNormGated`
    with a group a head (`groups` 2) in the matching form (`gate_after`, `scale_wide`), one arithmetic
    (`norms.gated_norm_rows`) for the layers' gated norms and the launch's (train 1473's review)."""
    gate_first, rounded = form
    rng = np.random.default_rng(1473)
    out, z = (jnp.asarray(rng.standard_normal((2, 3, 2, 64)), jnp.bfloat16) for _ in range(2))
    weight = jnp.asarray(1 + rng.standard_normal(128) / 8, jnp.float32)
    norm = norms.OutputNorm(1e-5, 'silu', gate_first=gate_first, rounded=rounded)
    found = norms.output_normed(norm, out, z, weight.reshape(2, 64), jnp.bfloat16, jnp.float32)
    layer = MambaRMSNormGated(1e-5, groups=2, gate_after=not gate_first, scale_wide=not rounded)
    expected = layer.apply({'params': {'weight': weight}}, out.reshape(2, 3, 128), z.reshape(2, 3, 128))
    np.testing.assert_array_equal(np.asarray(found.reshape(2, 3, 128), np.float32),
                                  np.asarray(expected, np.float32))


def test_a_norm_form_the_fused_kernel_lacks_is_xlas_alone():
    """A gate-first norm (`OutputNorm.gate_first`, Mamba-2's form) after a chunked call: XLA's, its route
    reported with the form's refusal, and refused where a call names the fused kernel for it
    (`dew.kernels.use(rms_norm='pallas')`), never run as another kernel's arithmetic (train 1473's head
    review)."""
    from dew import kernels
    from dew.nn.kernels.selection import KernelRefused

    norm = norms.OutputNorm(1e-6, 'silu', gate_first=True)
    out, z = (jnp.ones((1, 2, 1, 8), jnp.bfloat16) for _ in range(2))

    def normed():
        return norms.output_normed(norm, out, z, jnp.ones((8,), jnp.float32), jnp.bfloat16, jnp.float32)

    with generation.routes() as found:
        jax.eval_shape(normed)
    assert [route.kernel for route in found if route.choice == generation.Choice.RMS_NORM] == ['xla'], found
    # On an H100's facts, so the form is the kernel's first refusal (off a GPU its capability's is).
    with generation.assumed('sm90'), kernels.use(rms_norm='pallas'), pytest.raises(KernelRefused) as refused:
        jax.eval_shape(normed)
    assert refused.value.first.code == 'form'


@pytest.mark.parametrize('form', norm_swiglu.GATED_FORMS)
def test_the_norm_selector_takes_its_kernel_for_its_own_gated_form_alone(form):
    """The norm selector's fused kernel (`norm_swiglu.NORM`) over a bf16 gated norm on an H100's facts: no
    'form' refusal of the gate-after form, Qwen3-Next's, and that refusal of every other
    (`norm_swiglu.GATED_FORMS`), which XLA computes; a form not among them raises."""
    rows = jax.ShapeDtypeStruct((4, 128), jnp.bfloat16)
    case = norm_swiglu.norm_case(rows, jax.ShapeDtypeStruct((128,), jnp.float32), jnp.bfloat16, None,
                                 gate=rows, activation='silu', form=form)
    with generation.assumed('sm90'):
        codes = [refusal.code for refusal in norm_swiglu.NORM.kernels[0].refusals(case)]
    assert ('form' in codes) == (form != 'gate after'), codes
    with pytest.raises(ValueError):
        dataclasses.replace(case, form='gate between')


def test_a_rounding_norm_forms_decode_token_is_its_reference_bit_for_bit():
    """A bf16 decode token through the step launch (interpreted off a GPU) with Mamba-2's gate-first rounding
    norm (`OutputNorm(gate_first=True, rounded=True)`, `eps` 0.39) over a head of 128 values reading ones,
    its sigmoid gate's logits zero and its weight `1 + 7/2048`: bit for bit `recurrence.step_reference`,
    both rounding the normalized rows to the token's bf16 output before the weight, where the reference
    rounded them in its fp32 work (a head review)."""
    hooks = recurrence.Hooks(decay='scalar', output=norms.OutputNorm(0.39, 'sigmoid', gate_first=True,
                                                                          rounded=True))
    unit = jnp.zeros((1, 1, 128), jnp.bfloat16).at[..., 0].set(1.0)
    given = recurrence.Step(state=jnp.zeros((1, 1, 128, 128), jnp.float32), query=unit, key=unit,
                            value=jnp.ones((1, 1, 128), jnp.bfloat16), log_decay=jnp.zeros((1, 1)),
                            beta=jnp.ones((1, 1)), z=jnp.zeros((1, 1, 128)),
                            weight=jnp.full((128,), 1 + 7 / 2048, jnp.float32))
    found = jax.jit(functools.partial(recurrence.step, hooks))(given)[0]
    expected = recurrence.step_reference(hooks, given)[0]
    assert found.dtype == expected.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


ROW_HOOK_SIDES = {'columns first': ('columns', 'rows'), 'rows twice': ('rows', 'rows', 'columns'),
                  'no columns': ('rows',), 'another axis': ('tokens', 'columns')}
"""Axes no row hook's side lies on (`RowHook.SIDE_AXES`)."""


@pytest.mark.parametrize('axes', ROW_HOOK_SIDES.values(), ids=ROW_HOOK_SIDES)
def test_a_row_hooks_side_lies_on_its_axes_in_the_launchs_order(axes):
    """A row hook's side on axes a launch does not load as they are given (`RowHook.SIDE_AXES`' order, each
    once, its columns among them) is refused where the hook is built: a side on its columns before its rows
    loaded transposed into a Hopper block's layout (a head review)."""
    def hook(axes):
        return RowHook(lambda block: block, lambda block, total, side: block,
                       (Side(jnp.zeros((4, 4), jnp.float32), axes),))

    hook(('rows', 'columns'))
    with pytest.raises(ValueError):
        hook(axes)


@pytest.mark.parametrize("name", ["gated delta net", "kimi delta attention"])
def test_a_captured_norm_runs_as_its_module(name):
    """A caller capturing a delta-rule mixer's calls (`capture_intermediates`) keeps its gated norm's: the
    chunked op leaves the norm to its module where a caller watches or captures it (`dense.watched_or_owned`),
    which an epilogue standing in for the call would bypass, the capture holding no norm (train 1484's K-F3);
    the layer's output the same either way."""
    module, norm = {
        "gated delta net": (GatedDeltaNet(emb_features=32, num_k_heads=1, num_v_heads=2, head_k_dim=8,
                                          head_v_dim=8, chunk_size=4, dtype=jnp.float32), 'norm'),
        "kimi delta attention": (KimiDeltaAttention(emb_features=32, num_heads=2, head_dim=8, conv_kernel=4,
                                                    lower_bound=-5.0, chunk_size=4), 'o_norm')}[name]
    x = jnp.asarray(np.random.default_rng(1484).standard_normal((2, 12, 32)), jnp.float32)
    variables = module.init(jax.random.key(3), x)
    out, captured = module.apply(variables, x, capture_intermediates=True, mutable=['intermediates'])
    (normed,) = captured['intermediates'][norm]['__call__']
    assert normed.shape[:2] == x.shape[:2]
    np.testing.assert_allclose(np.asarray(out), np.asarray(module.apply(variables, x)), rtol=1e-5, atol=1e-6)


def test_a_captured_norms_padded_gate_leaves_its_weights_gradient_finite():
    """A gated delta net whose caller captures its calls, so its gated norm runs as its module
    (`dense.watched_or_owned`), over a row whose last two tokens are padded and infinite: the gradient of
    its live tokens' output by the norm's weight finite, the padded tokens' output gate zeroed before the
    norm as the chunked op's epilogue zeroes it, where their infinite or undefined gate met their zero
    cotangent as NaN (head review of train 1484's K-F3)."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata, MixerCall

    module = GatedDeltaNet(emb_features=32, num_k_heads=1, num_v_heads=2, head_k_dim=8, head_v_dim=8,
                           chunk_size=4, dtype=jnp.float32)
    x = np.random.default_rng(1484).standard_normal((1, 12, 32)).astype(np.float32)
    x[:, 10:] = np.inf
    valid = jnp.asarray(np.arange(12) < 10)[None]
    variables = module.init(jax.random.key(3), jnp.zeros((1, 12, 32), jnp.float32))
    call = MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid)))

    def live(weight):
        params = {**variables['params'], 'norm': {**variables['params']['norm'], 'weight': weight}}
        out, _ = module.apply({'params': params}, jnp.asarray(x), call, capture_intermediates=True,
                              mutable=['intermediates'])
        return jnp.sum(out[:, :10])

    found = jax.jit(jax.grad(live))(variables['params']['norm']['weight'])
    assert np.all(np.isfinite(np.asarray(found)))
