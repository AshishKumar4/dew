"""The recurrences on the GPU (G14, the owner's rule: no unit tests on GPUs): each native recurrence route's
cases the CPU does not interpret, run in one process (`gpu_flow`), each cell its own file's property:

- The Mosaic mainloop's forward and gradient on par from float64 with their references at the cases the
  CPU leaves to a GPU (test_recurrence_mosaic's `native`, test_recurrence_mosaic_gradient), every hook
  it takes token by token, and a two-role walk's training on an H100.
- The template finite and on par where chunked forms break, every form, regime and product
  (test_recurrence_template_stability).
- The portable backward as a GPU compiles it, and the SSD port's forward and gradient, finite and on par
  under every regime (test_recurrence_stability).
- The short conv's kernels as exact as XLA's conv at the training shapes (test_short_conv).
- Mamba's selective scan on the template under the fast forms as exact as recorded mamba_ssm in each
  recorded case, and a Mamba layer under the template's choice, packed and from a held state, as exact as
  under the portable scan, on an H100 (test_selective_template; an A100's after post42).
- Each recurrence layer trained on its port's kernels, natively (`PORT_KERNELS`): a gated delta net on the
  FLA GDN kernels and a Kimi delta attention layer on FLA KDA's chunk_kda, packed rows, forward and every
  gradient on par from float64 with the portable rule's; their training step the same bit for bit each
  run, on the port and on the template; a gated delta net's decode tokens on its decode kernels (the fused
  step and the step alone), its state held in fp32 and in bf16, on par with the portable step's; and the
  port on par with fla's own in fp32 and mixed dtypes (test_fla_gdn), fla run in the reference interpreter
  DEW_FLA_PYTHON names, which a GPU job provides to a file naming it.
- The decode launches as a server runs them (test_linear_attention): the step kernel leaving an idle row's
  state and output alone, bit for bit; the fused step's bits the same however its launches are timed; and
  a layer choosing the fused step decoding over two GPUs, its rows or its heads split, as on one device.
"""

import functools

import gpu_flow
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recurrence_matrix as matrix
import selective_support
import test_fla_gdn as fla_gdn
import test_mamba_ssd as mamba_ssd
import test_recurrence_mosaic as forward
import test_recurrence_mosaic_gradient as gradient
import test_recurrence_stability as stability
import test_recurrence_template_stability as template_stability
import test_selective_template as selective_template
import test_short_conv as short_conv
from delta_rule_support import fused_step
from gpu_flow import Cell, cells_of, run_cells, when
from jax.sharding import NamedSharding, PartitionSpec
from model_support import decode_through
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, with_gradients
from test_linear_attention import decode_operands, fused_case, fused_inputs

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.kernels.templates.recurrence import mainloop
from dew.nn.linear import GatedDeltaNet, decode_gated_delta_rule
from dew.training import MeshSpec

pytestmark = pytest.mark.gpu(reason="each recurrence route's native kernel: the Mosaic mainloop, the ports")


def on_its_family(check):
    """`check` where this GPU is of a family the mainloop's bundles claim (an A100's, an H100's), else
    skipped (`when`)."""
    return when(mainloop.family_of() is not None,
                "the mainloop's bundles run compiled on a GPU of the family each claims", check)


PORT_KERNELS = {
    "fla": ("fla_gdn: kkt_solve, recompute_w_u, chunk_fwd_h, chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dhu, "
            "chunk_bwd_dqkwg, prepare_wy_repr_bwd"),
    "kda": ("fla_kda: intra_rows, intra_solve, recompute_w_u, chunk_fwd_o, bwd_dav, bwd_wy_dqkg, bwd_intra; "
            "fla_gdn: chunk_fwd_h, chunk_bwd_dhu"),
}
"""The port kernels each layer's training cell runs natively, by the layer's port (`trains_on_its_port`)."""

ROWS, TOKENS, WIDTH = 2, 256, 256
"""A layer cell's rows, their tokens (four 64-token chunks) and the layers' model width."""


def gated_delta_net():
    return GatedDeltaNet(emb_features=WIDTH, num_k_heads=2, num_v_heads=4, head_k_dim=128, head_v_dim=128,
                         dtype=jnp.bfloat16)


def kimi_delta_attention():
    return KimiDeltaAttention(emb_features=WIDTH, num_heads=2, head_dim=128, dtype=jnp.bfloat16)


def named(fn, **kernels: str):
    """`fn` with the recurrence's selectors named to `kernels` (`generation.chosen`)."""
    def run(*args):
        with generation.chosen(**kernels):
            return fn(*args)
    return run


def trains_on_its_port(make):
    """A layer over two rows of two packed documents each, its forward and the gradients of its input and
    every parameter under a drawn cotangent on its port (`PORT_KERNELS`, named), on par from float64 with
    the portable rule's (check 1, the layer in float64 the truth)."""
    layer = make()
    packed = MixerCall(AttentionMetadata(coordinates=coordinates(
        segments=jnp.asarray([[1] * 150 + [2] * 106, [1] * 90 + [2] * 166]), max_segments=2)))

    def apply(params, x):
        return layer.apply(params, x, packed)

    def draw(seed):
        x = jax.random.normal(jax.random.key(seed), (ROWS, TOKENS, WIDTH)).astype(jnp.bfloat16)
        params = named(layer.init, gated_delta_rule="xla")(jax.random.key(1000 + seed), x)
        cotangent = jax.random.normal(jax.random.key(2000 + seed), x.shape).astype(jnp.bfloat16)
        return params, x, cotangent

    def truth(params, x, cotangent):
        wide = layer.clone(dtype=jnp.float64)
        return with_gradients(lambda params, x: wide.apply(params, x, packed))(params, x, cotangent)

    assert_as_exact(named(with_gradients(apply), gated_delta_rule="fla"),
                    named(with_gradients(apply), gated_delta_rule="xla"), draw,
                    f"{type(layer).__name__} trained on its port", seeds=FEWEST_SEEDS, truth_fn=truth)


def repeats_bit_for_bit(make, kernel: str):
    """A layer's training step on `kernel`, the same program on the same inputs, run twice, compiled afresh
    the second time: every gradient the same bit for bit (the kernels sum in a fixed order)."""
    layer = make()
    x = jax.random.normal(jax.random.key(5), (ROWS, TOKENS, WIDTH)).astype(jnp.bfloat16)
    params = named(layer.init, gated_delta_rule="xla")(jax.random.key(6), x)
    step = jax.jit(named(jax.grad(lambda params, x: jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2),
                                  argnums=(0, 1)), gated_delta_rule=kernel))
    gpu_flow.repeats_bit_for_bit(step, params, x)


DECODED = 4
"""The tokens a decode cell decodes in turn after its prefill."""


def decodes_on_its_kernel(kernel: str, held):
    """A gated delta net's decode tokens on `kernel` (`linear.DELTA_DECODE`: 'fused' or 'step'), `DECODED`
    in turn after a prefill, its recurrent state held in `held`: each token's output, and the conv history
    and recurrent state it leaves, on par with the portable step's (check 1) from the layer in float64 (the
    truth: a bf16 layer on widened arguments fixes its projections and norm in bf16 and casts its state
    back, so only the stored state's rounding is the contract's)."""
    layer = fused_case(4, 128, 3, jnp.bfloat16, held)[0]

    def decoded(module):
        def decode(variables, cache, tokens):
            outs = []
            for at in range(DECODED):
                out, after = module.apply({**variables, "cache": cache}, tokens[:, at:at + 1],
                                          MixerCall(decode=True), mutable=["cache"])
                outs.append(out)
                cache = after["cache"]
            return jnp.concatenate(outs, axis=1), cache["conv_state"], cache["recurrent_state"]
        return decode

    def draw(seed):
        _, variables, cache, token = fused_case(4, 128, 3, jnp.bfloat16, held, seed)
        later = jax.random.normal(jax.random.key(3000 + seed), (token.shape[0], DECODED - 1, token.shape[-1]))
        return variables, cache, jnp.concatenate([token, later.astype(token.dtype)], axis=1)

    assert_as_exact(named(decoded(layer), delta_decode=kernel), named(decoded(layer), delta_decode="xla"),
                    draw,
                    f"a gated delta net decoding {DECODED} tokens on its {kernel} kernel, its state held in "
                    f"{jnp.dtype(held).name}", seeds=FEWEST_SEEDS,
                    truth_fn=named(decoded(layer.clone(dtype=jnp.float64)), delta_decode="xla"))


def the_decode_kernel_leaves_an_idle_row_alone():
    """A serving step holds every slot, drawing or not: a row the step kernel (`decode_gated_delta_rule`)
    marks idle, at Qwen3.5-0.8B's 16 heads 128 wide, keeps its state bit for bit, unread and unwritten, and
    outputs zeros; the drawing rows step to the bits they take with every row active."""
    query, key, value, g, beta, state = decode_operands(np.random.default_rng(1), 8, 16, 128)
    held = np.asarray(state)
    active = jnp.asarray([True, False, True, True, False, False, True, False])
    every, every_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state)
    got, got_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state, active)
    drawing = np.asarray(active)
    np.testing.assert_array_equal(np.asarray(got)[drawing], np.asarray(every)[drawing])
    np.testing.assert_array_equal(np.asarray(got_state)[drawing], np.asarray(every_state)[drawing])
    np.testing.assert_array_equal(np.asarray(got)[~drawing], 0)
    np.testing.assert_array_equal(np.asarray(got_state)[~drawing], held[~drawing])


def the_fused_decode_step_gives_the_same_bits_however_its_launches_are_timed():
    """The fused decode step (`fused_step`) at Qwen3.5-0.8B's widths, a bf16 state, run 20 times, each run
    beside a large matmul that holds most of the GPU or alone, gives the same output, history and state bits
    every run: no value head reads a history or state another program is writing, whatever order they run
    in."""
    layer, variables, cache, token = fused_case(16, 128, 4, jnp.bfloat16, jnp.bfloat16)
    operands = fused_inputs(layer, variables, cache, token)
    load = jnp.ones((8192, 8192), jnp.bfloat16)
    busy = jax.jit(lambda a: a @ a)
    run = jax.jit(functools.partial(fused_step, epsilon=layer.norm_eps, activation=layer.gate_activation,
                                    dtype=jnp.bfloat16))
    first = jax.tree.map(np.asarray, run(*operands))
    for repeat in range(20):
        if repeat % 2:
            busy(load)
        for mine, theirs in zip(jax.tree.leaves(run(*operands)), jax.tree.leaves(first), strict=True):
            np.testing.assert_array_equal(np.asarray(mine), theirs, err_msg=f"run {repeat}")


def a_layer_choosing_the_fused_step_decodes_on_a_mesh_as_on_one_device(axis: str):
    """A gated delta net choosing the fused step (`generation.chosen`) over two GPUs, its rows (`axis`
    'data') or its heads ('tensor') split two ways (`_fused_token`: each shard's channels its heads'
    queries, keys and values): 9 decode tokens after a 3-token prefill on par (check 1) with the same
    layer's on one device, from the float64 parallel forward, over drawn hidden states and weights."""
    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
                           dtype=jnp.float32)
    # The data axis takes the devices the others leave.
    mesh = MeshSpec(**({} if axis == "data" else {axis: 2})).build(jax.devices()[:2])

    def draw(seed):
        hidden = jax.random.normal(jax.random.key(13 + 2 * seed), (2, 12, 64), jnp.float32)
        return module.init(jax.random.key(14 + 2 * seed), hidden[:, :1]), hidden

    def decoded(variables, hidden):
        return decode_through(module, variables, hidden, prefill=3)[0]

    split_run = jax.jit(named(decoded, delta_decode="fused"))

    @as_given
    def split(variables, hidden):
        with jax.set_mesh(mesh):
            return split_run(variables, jax.device_put(hidden, NamedSharding(mesh, PartitionSpec())))

    assert_as_exact(split, named(decoded, delta_decode="fused"), draw,
                    f"a layer choosing the fused step, decoding split over {axis}", seeds=FEWEST_SEEDS,
                    truth_fn=module.clone(dtype=jnp.float64).apply)


def mainloop_forward_cells() -> list[Cell]:
    """The Mosaic mainloop's forward at the cases the CPU leaves to a GPU, and every hook it takes."""
    return [
        *(Cell(f"the Mosaic forward is as exact as its reference: {name}",
               functools.partial(forward.the_mosaic_forward_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(forward.FORWARD_GRID, forward.INTERPRETED_FORWARD)),
        *(Cell(f"every hook the Mosaic mainloop takes is as exact as the recurrence: {hooks} {packed}",
               on_its_family(functools.partial(
                   forward.every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token,
                   hooks, packed)))
          for hooks in forward.TAKEN for packed in (False, True)),
    ]


def mainloop_gradient_cells() -> list[Cell]:
    """The Mosaic mainloop's gradient at the cases the CPU leaves to a GPU, and a two-role walk's."""
    return [
        *(Cell(f"the Mosaic gradient is as exact as its reference: {name}",
               functools.partial(gradient.the_mosaic_gradient_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(gradient.GRADIENT_GRID, gradient.INTERPRETED_GRADIENT)),
        *(Cell(f"a walk of two roles is as exact as its reference: {name}",
               functools.partial(gradient.test_a_walk_of_two_roles_is_as_exact_as_its_reference, *values),
               "sm90")
          for name, values in forward.native(gradient.WALK_GRID, gradient.INTERPRETED_WALKS)),
    ]


def template_stability_cells() -> list[Cell]:
    """The template finite and on par where chunked forms break, every form, regime and product."""
    finite = template_stability.the_template_stays_finite_and_on_par_where_chunked_forms_break
    return [Cell(f"the template stays finite and on par where chunked forms break: {name} {regime} {product}"
                 f"{'' if forms is None else f' ({forms} forms)'}",
                 functools.partial(finite, name, regime, product, forms))
            for name, regime, product, forms in template_stability.CASES]


def portable_and_ssd_port_cells() -> list[Cell]:
    """The portable backward as a GPU compiles it, and the SSD port's forward and gradient under every
    regime, and the SSD on the template against mamba_ssm."""
    return [
        *(Cell(f"the portable backward as a GPU compiles it is as exact as the recurrence's: {name} {regime}",
               functools.partial(stability.test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences,
                                 name, regime))
          for name, regime in stability.ACCELERATED),
        *(Cell(f"the SSD port stays finite and exact where chunked forms break: {regime}",
               functools.partial(stability.the_ssd_port_stays_finite_and_exact_where_chunked_forms_break,
                                 regime))
          for regime in stability.REGIMES),
        Cell("the SSD on the template under the fast forms is as exact as mamba_ssm",
             mamba_ssd.the_template_is_as_exact_as_mamba_ssm),
        *(Cell(f"the SSD port's gradient is as exact as the recurrence's: {regime}",
               functools.partial(stability.the_ssd_ports_gradient_is_as_exact_as_the_recurrences, regime))
          for regime in stability.REGIMES),
    ]


def selective_cells() -> list[Cell]:
    """Mamba's selective scan on the template against recorded mamba_ssm, and a Mamba layer under it."""
    return [
        *(Cell(f"the selective scan on the template under the fast forms is as exact as mamba_ssm: {case}",
               functools.partial(selective_template.the_template_is_as_exact_as_mamba_ssm, case), "sm90")
          for case in selective_support.SEEDED_CASES),
        *(Cell(f"a Mamba layer is as exact under the template: packed {packed}"
               f"{f', a held state {held}' if held else ''}",
               functools.partial(selective_template.the_layer_is_as_exact_under_the_template, packed, held),
               "sm90")
          for packed, held in ((False, None), (True, None), (False, "through the template"))),
    ]


def short_conv_cells() -> list[Cell]:
    """The short conv's kernels against XLA's conv at the training shapes."""
    return [Cell(f"the short conv's kernels are as exact as XLA's at the training shapes: {model} "
                 f"{dtype.__name__}{' packed' if packed else ''}",
                 functools.partial(short_conv.the_kernels_are_as_exact_as_xla_at_the_training_shapes,
                                   model, dtype, packed))
            for model in sorted(short_conv.CONV_SHAPES) for dtype in (jnp.bfloat16, jnp.float32)
            for packed in (False, True)]


def port_training_cells() -> list[Cell]:
    """Each recurrence layer trained on its port's kernels, its step repeating bit for bit, and the ports
    against fla's own."""
    return [
        Cell(f"a gated delta net trains on its port on par with the portable rule ({PORT_KERNELS['fla']})",
             functools.partial(trains_on_its_port, gated_delta_net)),
        Cell(f"a Kimi delta attention layer trains on its port on par with the portable rule "
             f"({PORT_KERNELS['kda']})", functools.partial(trains_on_its_port, kimi_delta_attention)),
        # Kimi delta attention trains on its port alone: the template's fast backward takes no decay a key
        # dimension.
        *(Cell(f"a {make.__name__} training step on {kernel} repeats bit for bit",
               functools.partial(repeats_bit_for_bit, make, kernel))
          for make, kernels in ((gated_delta_net, ("fla", "mosaic_gpu")), (kimi_delta_attention, ("fla",)))
          for kernel in kernels),
        *(Cell(f"the port is as exact as fla's own: {'kda' if per_key else 'gdn'} {dtypes}",
               functools.partial(fla_gdn.the_port_is_as_exact_as_fla, per_key, dtypes))
          for per_key in (False, True) for dtypes in fla_gdn.FLA_DTYPES),
    ]


def decode_cells() -> list[Cell]:
    """The decode launches as a server runs them."""
    return [
        *(Cell(f"a gated delta net decodes on its {kernel} kernel on par with the portable step, its state "
               f"held in {jnp.dtype(held).name}", functools.partial(decodes_on_its_kernel, kernel, held))
          for kernel in ("fused", "step") for held in (jnp.float32, jnp.bfloat16)),
        Cell("the decode kernel leaves an idle row alone", the_decode_kernel_leaves_an_idle_row_alone),
        Cell("the fused decode step gives the same bits however its launches are timed",
             the_fused_decode_step_gives_the_same_bits_however_its_launches_are_timed),
        *(Cell(f"a layer choosing the fused step decodes on a mesh as on one device: {axis}",
               when(jax.device_count() >= 2, "needs two GPUs", functools.partial(
                   a_layer_choosing_the_fused_step_decodes_on_a_mesh_as_on_one_device, axis)))
          for axis in ("data", "tensor")),
    ]


FAMILIES = {"mainloop forward": mainloop_forward_cells, "mainloop gradient": mainloop_gradient_cells,
            "template stability": template_stability_cells,
            "portable and SSD port": portable_and_ssd_port_cells, "selective scan": selective_cells,
            "short conv": short_conv_cells, "port training": port_training_cells, "decode": decode_cells}
"""The recurrence cells by family, each family a test of its own: the planner times and packs each, and one
family's failure hides no other's (a single flow of every family ran past a GPU job's 50 minutes)."""


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_recurrences_run_on_this_gpu(family):
    """A family's recurrence cells on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together; every direct mainloop call keeps its own bound."""
    run_cells(FAMILIES[family]())


def test_the_recurrence_matrix_holds_on_this_gpu():
    """The recurrence's check-1 matrix (tests/kernel_matrix.py): the chunked op's training step, its kernel
    named, on par with the recurrence a token at a time at every pair of axis values the route serves, each
    cell on its own GPU."""
    run_cells(cells_of(matrix.a_recurrences_training_step_is_on_par))
