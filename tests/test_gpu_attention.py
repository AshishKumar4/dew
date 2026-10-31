"""Attention on the GPU (G14, the owner's rule: no unit tests on GPUs): each attention route whose kernel
runs on a GPU alone, run in one process (`gpu_flow`), each cell its own file's property:

- The frozen Triton template at every config of its forward's and its backward's spaces, bf16, every hook,
  over causal documents under a window and the irregular calls (test_attention_triton); its default form
  finite where the reference is at fp32's edges for the families too slow to interpret
  (test_attention_stability).
- The Mosaic mainloop's default form, and its backward in two launches, finite where the reference is at
  fp32's edges over every family it takes, on an H100 (test_attention_stability).
- A dropping route call on a GPU runs finite (test_attention_dropout).
- The template's training step with a bias every row shares, run again under the lane's deterministic ops,
  gives every gradient bit for bit the same (`a_shared_bias_step_repeats_bit_for_bit`, ledger 2620).
- A decode step's sinks take their gradient and tangent through the dispatch, plain and softcapped; the
  decode kernel's attention differentiates as the folded attention (test_decode_attention).
- Fused routes, without XLA's deterministic ops, which Dew refuses them under: flash reads a cache call's
  documents as exactly as jax.nn reads its mask (test_decode_attention), and packed windowed bf16
  attention runs its band on par (test_local_attention); flash, cuDNN and the Triton kernel train, take
  packed and odd-length calls, sinks, windows and empty contexts on par with XLA from float64, and each
  kernel's requirements are what it takes on this GPU (test_kernels); a row that sees no key reads
  nothing on the fused kernels (test_attention_masked_rows).
- The Mosaic template on an H100, each test its own cells: forward, training forward, natural forward
  and backward over every call form, head width, hook and run (test_attention_mosaic); an fp16 call
  through the route, and a long sequence on par with the reference path (test_attention_route); the fast
  form's default route not on par with IEEE fp32, the accurate form's control (test_attention_accurate_form).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_attention_accurate_form as accurate_form
import test_attention_dropout as dropout
import test_attention_masked_rows as masked_rows
import test_attention_mosaic as mosaic
import test_attention_route as route
import test_attention_stability as stability
import test_attention_triton as triton
import test_decode_attention as decode
import test_kernels as kernels
import test_local_attention as local
from gpu_flow import Cell, cells_of, repeats_bit_for_bit, run_cells, when

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.generation import bf16_dot_runs

pytestmark = pytest.mark.gpu(reason="each route's GPU kernel: Triton, Mosaic, flash and the band")

BF16_PRODUCTS = "needs a GPU whose bf16 products run, sm80 or later"
"""Why a library kernel's cell skips: flash, cuDNN, tokamax's Triton and the band run wherever bf16 products
do (`bf16_dot_runs`), an A10 and an L4 among them, whatever Dew's own launches' sizes."""

CASES = ("causal-documents-window", *triton.IRREGULAR)
"""The Triton sweeps' calls (`triton.every_config_on_a_gpu`)."""


def sweep(check, space) -> Cell:
    """`check` at every config of `space` over every call of CASES, as one cell: each config its own
    inner cell, every failure kept (`run_cells`)."""
    return Cell(f"{check.__name__.replace('_', ' ')}: {len(space)} configs by {len(CASES)} calls",
                lambda: run_cells([Cell(f"{case} {config}", functools.partial(check, config, case))
                                   for case in CASES for config in space]))


def a_shared_bias_step_repeats_bit_for_bit(bias_shape: tuple) -> None:
    """A bf16 causal training step through the template at a sequence of 2048, its bias of `bias_shape`
    shared by every batch row (a tile of each head's logits, or a key bias), the same program on the same
    inputs run six times under the lane's deterministic ops (tests/lane_environment.py), the second compiled
    afresh: every
    gradient, the bias's own included, the same bits as the first run's, as kernel-templates.md promises
    a schedule's reproducibility (ledger 2620). Only a GPU's atomics reorder, so only a GPU can fail it."""
    rng = np.random.default_rng(2620)
    batch, length, heads, width = 4, 2048, 8, 64

    def normal(*shape, scale=1.0):
        return jnp.asarray(scale * rng.standard_normal(shape, dtype=np.float32), jnp.bfloat16)

    shape = (batch, length, heads, width)
    query, key, value, cotangent = normal(*shape), normal(*shape, scale=0.3), normal(*shape), normal(*shape)
    bias = jnp.asarray(0.5 * rng.standard_normal((1, heads, *bias_shape), dtype=np.float32))

    @jax.jit
    def step(query, key, value, bias, cotangent):
        _, pull = jax.vjp(lambda q, k, v, b: scaled_dot_product_attention(
            q, k, v, bias=b, causal=True, implementation="template"), query, key, value, bias)
        return pull(cotangent)

    repeats_bit_for_bit(step, query, key, value, bias, cotangent, runs=6,
                        names=[f"the {name}'s gradient" for name in ("query", "key", "value", "bias")])


def test_the_attention_kernels_run_on_this_gpu():
    """Every kernel cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together; the Mosaic cells on an H100 alone."""
    run_cells([
        sweep(triton.every_config_on_a_gpu, triton.attention_triton.SPACE),
        sweep(triton.every_backward_config_on_a_gpu, triton.attention_triton.BACKWARD_SPACE),
        *(Cell(f"the frozen Triton mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "triton"))
          for family in stability.GPU_TRITON_FAMILIES),
        *(Cell(f"the Mosaic mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic"), "sm90")
          for family in stability.MOSAIC_FAMILIES),
        *(Cell(f"the Mosaic backward in two launches is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic two launches"), "sm90")
          for family in stability.HOOKLESS_FAMILIES),
        *(Cell(f"a step with a bias every row shares repeats bit for bit: {name}",
               functools.partial(a_shared_bias_step_repeats_bit_for_bit, shape))
          for name, shape in (("a tile of the logits", (2048, 2048)), ("a key bias", (1, 2048)))),
        Cell("a dropping route call on a GPU runs finite",
             dropout.a_dropping_route_call_on_a_gpu_runs_finite),
        *(Cell(f"a decode step's sinks take their gradient and tangent through the dispatch: {name}",
               functools.partial(
                   decode.a_decode_steps_sinks_take_their_gradient_and_tangent_through_the_dispatch, softcap))
          for name, softcap in (("plain", None), ("softcap", decode.SOFTCAP))),
        Cell("the decode kernel's attention differentiates as the folded attention",
             decode.attention_the_decode_kernel_takes_differentiates_as_the_folded_attention),
    ])


def test_the_fused_attention_routes_run_on_this_gpu(without_deterministic_ops):
    """Every fused route's cell above on this GPU, without XLA's deterministic ops
    (`without_deterministic_ops`); the band on a bf16 GPU, sm80 or later."""
    run_cells([
        *(Cell(f"flash reads a cache call's documents as exactly as jax.nn reads its mask: window {window}",
               functools.partial(decode.flash_reads_a_cache_calls_documents_as_exactly_as_jax_nn_reads_its_mask,
                                 without_deterministic_ops, window))
          for window in (None, 4)),
        Cell("packed windowed bf16 attention runs its band on par",
             when(bf16_dot_runs(), BF16_PRODUCTS, functools.partial(
                 local.packed_windowed_bf16_attention_runs_its_band_on_par, without_deterministic_ops))),
    ])


MOSAIC_CELLS = (mosaic.the_forward_is_on_par_on_an_h100, mosaic.the_training_forward_is_on_par_on_an_h100,
                mosaic.the_natural_forward_is_on_par_on_an_h100,
                mosaic.the_forward_is_on_par_over_every_call_form_on_an_h100,
                mosaic.the_forward_is_on_par_at_head_widths_on_an_h100,
                mosaic.the_forward_is_on_par_under_every_hook_on_an_h100,
                mosaic.the_backward_is_on_par_on_an_h100,
                mosaic.the_backward_is_on_par_at_head_widths_on_an_h100,
                mosaic.the_backward_is_on_par_over_every_call_form_on_an_h100,
                mosaic.the_backward_is_on_par_under_every_score_hook_on_an_h100,
                mosaic.a_call_over_runs_is_on_par_on_an_h100,
                mosaic.a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100)
"""The Mosaic template's H100 checks, each test its own cells, so a red names its form."""

KERNEL_CELLS = (kernels.flash_through_dews_attention_refuses_rows_it_would_misread,
                kernels.a_fused_kernel_takes_forward_mode_through_the_reference,
                kernels.cudnn_trains_odd_lengths_and_agrees_with_xla,
                kernels.triton_trains_and_agrees_with_xla,
                kernels.triton_trains_at_large_logits_on_par_with_xla,
                kernels.triton_at_saturated_large_logits_is_finite_where_xla_is,
                kernels.flash_trains_and_agrees_with_xla,
                kernels.packed_sequences_attend_within_themselves_as_far_from_float64_as_one_at_a_time,
                kernels.queries_packed_against_keys_of_their_own_lengths_attend_their_own_documents,
                kernels.fused_kernels_are_as_exact_as_the_xla_kernel_from_float64,
                kernels.flash_with_a_window_and_sinks_is_as_exact_as_the_xla_kernel_from_float64,
                kernels.a_window_over_more_keys_than_queries_keeps_the_keys_each_row_centres,
                kernels.packed_rows_on_flash_are_as_exact_as_the_document_mask_from_float64,
                kernels.queries_reading_another_sequences_documents_on_flash_are_as_exact_as_the_document_mask,
                kernels.queries_over_a_context_of_no_tokens_read_nothing_on_flash,
                kernels.cudnn_takes_key_lengths_and_agrees_with_xla,
                kernels.single_query_cudnn_decode_matches_float64_attention_and_gradients,
                kernels.a_kernels_requirements_are_what_the_kernel_takes_on_this_gpu)
"""test_kernels' fused and Triton kernels on a GPU of cuDNN's bf16 floor (sm80) or later; its mesh cell
is the two-GPU flow's (tests/test_gpu_mesh.py)."""


def test_the_fused_kernels_hold_on_this_gpu(without_deterministic_ops):
    """Every test_kernels and masked-rows cell above on this GPU (`gpu_flow.cells_of`), without XLA's
    deterministic ops, each one's failure kept and all raised together."""
    run_cells([
        *(cell for check in KERNEL_CELLS
          for cell in cells_of(check, without_deterministic_ops=without_deterministic_ops)
          for cell in [dataclasses.replace(cell, run=when(bf16_dot_runs(), BF16_PRODUCTS, cell.run))]),
        *cells_of(masked_rows.a_row_that_sees_no_key_reads_nothing_on_the_fused_gpu_kernels,
                  without_deterministic_ops=without_deterministic_ops),
    ])


def test_the_mosaic_template_holds_on_an_h100():
    """Every Mosaic template cell above on an H100 (`gpu_flow.cells_of`), and the route's fp16 call and
    long sequence, each one's failure kept and all raised together."""
    run_cells([
        *(cell for check in MOSAIC_CELLS for cell in cells_of(check, "sm90")),
        *cells_of(route.an_fp16_call_on_an_h100_is_on_par, "sm90"),
        *cells_of(route.a_long_sequence_is_on_par_with_the_reference_path),
        *cells_of(accurate_form.the_fast_form_on_a_gpus_default_route_is_not_on_par_with_ieee_fp32, "sm90"),
    ])

