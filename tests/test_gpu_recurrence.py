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
- Each recurrence layer trained on its port's kernels, natively (`PORT_KERNELS`): a gated delta net on the
  FLA GDN kernels and a Kimi delta attention layer on FLA KDA's chunk_kda, packed rows, forward and every
  gradient on par from float64 with the portable rule's; their training step the same bit for bit each
  run, on the port and on the template; a gated delta net's decode tokens on its decode kernels (the fused
  step and the step alone), its state held in fp32 and in bf16, on par with the portable step's; and the
  port on par with fla's own in fp32 and mixed dtypes (test_fla_gdn), fla run in the reference interpreter
  DEW_FLA_PYTHON names, which a GPU job provides to a file naming it.
"""

import functools

import gpu_flow
import jax
import jax.numpy as jnp
import pytest
import test_fla_gdn as fla_gdn
import test_mamba_ssd as mamba_ssd
import test_recurrence_mosaic as forward
import test_recurrence_mosaic_gradient as gradient
import test_recurrence_stability as stability
import test_recurrence_template_stability as template_stability
import test_short_conv as short_conv
from gpu_flow import Cell, run_cells
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients
from test_linear_attention import fused_case

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.kernels.templates.recurrence import mainloop
from dew.nn.linear import GatedDeltaNet

pytestmark = pytest.mark.gpu(reason="each recurrence route's native kernel: the Mosaic mainloop, the ports")


def on_its_family(check):
    """`check` where this GPU is of a family the mainloop's bundles claim (an A100's, an H100's)."""
    def run():
        if mainloop.family_of() is None:
            pytest.skip("the mainloop's bundles run compiled on a GPU of the family each claims")
        check()
    return run


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


def test_the_recurrences_run_on_this_gpu():
    """Every recurrence cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together; every direct mainloop call keeps its own bound."""
    templates_finite = template_stability.the_template_stays_finite_and_on_par_where_chunked_forms_break
    run_cells([
        *(Cell(f"the Mosaic forward is as exact as its reference: {name}",
               functools.partial(forward.test_the_mosaic_forward_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(forward.FORWARD_GRID, forward.INTERPRETED_FORWARD)),
        *(Cell(f"every hook the Mosaic mainloop takes is as exact as the recurrence: {hooks} {packed}",
               on_its_family(functools.partial(
                   forward.every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token,
                   hooks, packed)))
          for hooks in forward.TAKEN for packed in (False, True)),
        *(Cell(f"the Mosaic gradient is as exact as its reference: {name}",
               functools.partial(gradient.test_the_mosaic_gradient_is_as_exact_as_its_reference, *values))
          for name, values in forward.native(gradient.GRADIENT_GRID, gradient.INTERPRETED_GRADIENT)),
        *(Cell(f"a walk of two roles is as exact as its reference: {name}",
               functools.partial(gradient.test_a_walk_of_two_roles_is_as_exact_as_its_reference, *values),
               "sm90")
          for name, values in forward.native(gradient.WALK_GRID, gradient.INTERPRETED_WALKS)),
        *(Cell(f"the template stays finite and on par where chunked forms break: {name} {regime} {product}",
               functools.partial(templates_finite, name, regime, product))
          for name, regime, product in template_stability.CASES),
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
        *(Cell(f"the short conv's kernels are as exact as XLA's at the training shapes: {model} "
               f"{dtype.__name__}{' packed' if packed else ''}",
               functools.partial(short_conv.the_kernels_are_as_exact_as_xla_at_the_training_shapes,
                                 model, dtype, packed))
          for model in sorted(short_conv.CONV_SHAPES) for dtype in (jnp.bfloat16, jnp.float32)
          for packed in (False, True)),
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
        *(Cell(f"a gated delta net decodes on its {kernel} kernel on par with the portable step, its state "
               f"held in {jnp.dtype(held).name}", functools.partial(decodes_on_its_kernel, kernel, held))
          for kernel in ("fused", "step") for held in (jnp.float32, jnp.bfloat16)),
        *(Cell(f"the port is as exact as fla's own: {'kda' if per_key else 'gdn'} {dtypes}",
               functools.partial(fla_gdn.the_port_is_as_exact_as_fla, per_key, dtypes))
          for per_key in (False, True) for dtypes in fla_gdn.FLA_DTYPES),
    ])
