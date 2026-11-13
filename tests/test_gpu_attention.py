"""Attention on the GPU (G14, the owner's rule: no unit tests on GPUs): each attention route whose kernel
runs on a GPU alone and no model flow reaches (tests/test_gpu_flows.py, row 5461's covering set), run in one
process (`gpu_flow`), each cell its own file's property; a route's shapes and hooks past these are the CPU's,
interpreted and through the compile gate:

- The Pallas-Triton template's default form finite where the reference is at fp32's edges for the families
  too slow to interpret (test_attention_stability), and its prescaled backward on an H100.
- The Mosaic mainloop's default form, and its backward in two launches, finite where the reference is at
  fp32's edges over every family it takes, on an H100 (test_attention_stability).
- A dropping route call on a GPU runs finite (test_attention_dropout).
- The template's training step with a bias every row shares, run again under the lane's deterministic ops,
  gives every gradient bit for bit the same (`a_shared_bias_step_repeats_bit_for_bit`, ledger 2620).
- A decode step's sinks take their gradient and tangent through the dispatch, plain and softcapped; the
  decode kernel's attention differentiates as the folded attention (test_decode_attention).
- Fused routes, without XLA's deterministic ops, which Dew refuses them under: flash reads a cache call's
  documents as exactly as jax.nn reads its mask (test_decode_attention), and packed windowed bf16
  attention runs its band on par (test_local_attention); cuDNN and the Triton kernel train, flash and the
  others take packed and odd-length calls, sinks, windows and empty contexts on par with XLA from float64,
  and each kernel's requirements are what it takes on this GPU (test_kernels); a row that sees no key
  reads nothing on the fused kernels (test_attention_masked_rows). Flash's plain training step is the
  dense flow's, a user's.
- The Mosaic template on an H100, each test its own cells: its forward and backward over every call form,
  head width and hook it declares, its natural forward, a call over runs, a chunked prefill's backward, its
  one launch backward and a span call's (test_attention_mosaic, attention_inplace_operands); an fp16 call
  through the route, and a long sequence on par with the reference path (test_attention_route). Its causal
  forward and backward at a flow's widths are the flows'.
- The check-1 matrices of attention, its kernels and decode (tests/kernel_matrix.py), each at the cases a
  red found on a GPU (`kernel_matrix.seeded_cells`); their other pairs are the CPU's, interpreted.
"""

import dataclasses
import functools

import attention_inplace_operands as inplace
import attention_triton_stages as triton_stages
import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
import pytest
import test_attention_decode_template as decode_template
import test_attention_dropout as dropout
import test_attention_masked_rows as masked_rows
import test_attention_mosaic as mosaic
import test_attention_route as route
import test_attention_stability as stability
import test_contracts_attention as contracts
import test_decode_attention as decode
import test_kernels as kernels
import test_local_attention as local
from gpu_flow import (
    BF16_PRODUCTS,
    Cell,
    cells_of,
    covering,
    in_parts,
    parted,
    repeats_bit_for_bit,
    run_cells,
    when,
)

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.generation import bf16_dot_runs

pytestmark = pytest.mark.gpu(reason="each route's GPU kernel: Triton, Mosaic, flash and the band")

def a_shared_bias_step_repeats_bit_for_bit(bias_shape: tuple, kv_heads: int = 8) -> None:
    """A bf16 causal training step through the template at a sequence of 2048, its bias of `bias_shape`
    shared by every batch row (a tile of each head's logits, or a key bias), its 8 query heads over
    `kv_heads` key heads, the same program on the same inputs run six times under the lane's deterministic
    ops (tests/lane_environment.py), the second compiled afresh: every gradient, the bias's own included,
    the same bits as the first run's, as kernel-templates.md promises a schedule's reproducibility (ledger
    2620; a key head's dK/dV programs write each grouped query head's bias gradient, row 2645). Only a
    GPU's atomics reorder, so only a GPU can fail it."""
    rng = np.random.default_rng(2620)
    batch, length, heads, width = 4, 2048, 8, 64

    def normal(*shape, scale=1.0):
        return jnp.asarray(scale * rng.standard_normal(shape, dtype=np.float32), jnp.bfloat16)

    shape, kv_shape = (batch, length, heads, width), (batch, length, kv_heads, width)
    query, key, value, cotangent = (normal(*shape), normal(*kv_shape, scale=0.3), normal(*kv_shape),
                                    normal(*shape))
    bias = jnp.asarray(0.5 * rng.standard_normal((1, heads, *bias_shape), dtype=np.float32))

    @jax.jit
    def step(query, key, value, bias, cotangent):
        _, pull = jax.vjp(lambda q, k, v, b: scaled_dot_product_attention(
            q, k, v, bias=b, causal=True, implementation="template"), query, key, value, bias)
        return pull(cotangent)

    repeats_bit_for_bit(step, query, key, value, bias, cotangent, runs=6,
                        names=[f"the {name}'s gradient" for name in ("query", "key", "value", "bias")])


KERNEL_PARTS = 4
"""The kernels flow's parts (`in_parts`): 310 s whole on a GH200 (MAINGPU-709ddea7a)."""


@in_parts(KERNEL_PARTS)
def test_the_attention_kernels_run_on_this_gpu(part):
    """Every kernel cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together, a part's share of them (`in_parts`); the Mosaic cells on an H100 alone."""
    run_cells(parted([
        # The Pallas-Triton dK/dV launch's prescaled query on an H100: right and repeating (the pipeliner's
        # race, attention_triton_stages).
        *cells_of(triton_stages.a_prescaled_tritons_backward_is_on_par_and_repeats_on_an_h100, "sm90"),
        *(Cell(f"the Pallas-Triton mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "triton"))
          for family in stability.GPU_TRITON_FAMILIES),
        # The stability cases whose kernel interpreted passes the CPU suite's target (`GPU_ONLY`), compiled.
        *(Cell(f"the accurate form is finite and as exact as the reference at fp32's edges: {family} {name}",
               functools.partial(
                   stability.test_the_accurate_form_is_finite_and_as_exact_as_the_reference_at_fp32s_edges,
                   family, name))
          for family, name in sorted(stability.GPU_ONLY["accurate"])),
        *(Cell(f"the Pallas-Triton backward reads a scaled forward's rows whatever its pass: {family}",
               functools.partial(
                   stability.test_the_frozen_triton_backward_reads_a_scaled_forwards_rows_whatever_its_pass,
                   family))
          for family in sorted(stability.GPU_ONLY["frozen rows"])),
        *(Cell(f"decode is finite and as exact as the reference at fp32's edges: {family} {form}",
               functools.partial(stability.test_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges,
                                 family, form))
          for family, form in sorted(stability.GPU_ONLY["decode"])),
        *(Cell(f"the template's paged decode holds every order: {edge}",
               functools.partial(contracts.test_the_paged_decode_kernel_holds_every_order, edge,
                                 "template_paged"), "sm90")
          for edge in contracts.EDGES),
        *(Cell(f"the Mosaic mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic"), "sm90")
          for family in stability.mosaic_families()),
        *(Cell(f"the Mosaic backward in two launches is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic two launches"), "sm90")
          for family in stability.hookless_families()),
        *(Cell(f"a step with a bias every row shares repeats bit for bit: {name}",
               functools.partial(a_shared_bias_step_repeats_bit_for_bit, shape, kv_heads))
          for name, shape, kv_heads in (("a tile of the logits", (2048, 2048), 8),
                                        ("a key bias", (1, 2048), 8),
                                        ("a tile of the logits over grouped heads", (2048, 2048), 2))),
        Cell("a dropping route call on a GPU runs finite",
             dropout.a_dropping_route_call_on_a_gpu_runs_finite),
        *(Cell(f"a decode step's sinks take their gradient and tangent through the dispatch: {name}",
               functools.partial(
                   decode.a_decode_steps_sinks_take_their_gradient_and_tangent_through_the_dispatch, softcap))
          for name, softcap in (("plain", None), ("softcap", decode.SOFTCAP))),
        Cell("the decode kernel's attention differentiates as the folded attention",
             decode.attention_the_decode_kernel_takes_differentiates_as_the_folded_attention),
        Cell("a decode call naming cuDNN differentiates under deterministic ops",
             decode.a_decode_call_naming_cudnn_differentiates_under_deterministic_ops),
        *(Cell(f"a verify block of {drafts} through the route is on par",
               functools.partial(decode_template.a_verify_block_through_the_route_is_on_par, drafts))
          for drafts in (4, 8, 16)),
        *(Cell(f"a paged decode step of {page}-key pages is on par",
               functools.partial(decode_template.a_paged_decode_step_of_small_pages_is_on_par, page))
          for page in decode_template.served_small_pages()),
    ], part, KERNEL_PARTS))


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


UNEVEN_WALKS = ("lengths-starts", "cross-causal-lengths-starts", "causal-chunk")
"""The call forms whose programs in one launch walk very different key-tile counts, none among them: an
mbarrier phase or a ring slot a producer and its consumers disagree on shows on a GPU alone, whose roles run
asynchronously, where the interpreter runs them in sequence (the attention lane, 11-11)."""

TRANSCENDENTAL = "softcap"
"""The score hook whose transcendental (tanh) a GPU computes on its approximate instruction (MUFU), which the
interpreter computes exactly (the attention lane, 11-11)."""


def _dtype(dtype) -> str:
    return jnp.dtype(dtype).name


def mosaic_cells() -> list[Cell]:
    """The Mosaic template's H100 checks, each with its native axes (`Cell.native`: the pass and the
    backward's launch form, the dtype, fold or natural logits) and the GPU-only faults it is kept for by
    name (`Cell.fault`): a walk uneven across one launch's programs (`UNEVEN_WALKS`), the softcap's
    approximate tanh (`TRANSCENDENTAL`), the one-launch backward's red at 475913063 and row 3171's span calls.
    A long sequence's length is native: its D, refinement and denominator are sums its mainloop runs in
    sequence over the key tiles, a ratio that grows with the length. Its causal forward,
    training forward and backward at a flow's widths are the model flows' (row 5461); the covering of the
    rest runs here (`gpu_flow.covering`), every other cell interpreted on the CPU and through the compile
    gate."""
    forward = {"pass": "forward", "launch": "forward"}
    return [
        *cells_of(mosaic.the_natural_forward_is_on_par_on_an_h100, "sm90",
                  native=lambda call, dtype: {**forward, "dtype": _dtype(dtype), "logits": "natural"}),
        *cells_of(mosaic.the_forward_is_on_par_over_every_call_form_on_an_h100, "sm90",
                  native=lambda form: {**forward, "dtype": "bfloat16", "logits": "fold"},
                  fault=lambda form: "uneven walks" if form in UNEVEN_WALKS else None),
        *cells_of(mosaic.the_forward_is_on_par_at_head_widths_on_an_h100, "sm90",
                  native=lambda width, value_width: {**forward, "dtype": "bfloat16", "logits": "fold"}),
        *cells_of(mosaic.the_forward_is_on_par_under_every_hook_on_an_h100, "sm90",
                  native=lambda hooks, dtype, natural: {**forward, "dtype": _dtype(dtype),
                                                        "logits": "natural" if natural else "fold"},
                  fault=lambda hooks, dtype, natural: (
                      "approximate tanh" if hooks == TRANSCENDENTAL and _dtype(dtype) == "bfloat16"
                      and not natural else None)),
        *cells_of(mosaic.the_backward_is_on_par_at_head_widths_on_an_h100, "sm90",
                  native=lambda width, value_width: {"pass": "backward", "launch": "default",
                                                     "dtype": "bfloat16", "logits": "fold"}),
        *cells_of(mosaic.the_backward_is_on_par_over_every_call_form_on_an_h100, "sm90",
                  native=lambda case, form: {"pass": "backward", "launch": form, "dtype": "bfloat16",
                                             "logits": "natural" if form == "accurate" else "fold"},
                  fault=lambda case, form: "uneven walks" if case in UNEVEN_WALKS else None),
        *cells_of(mosaic.the_backward_is_on_par_under_every_score_hook_on_an_h100, "sm90",
                  native=lambda hooks, form: {"pass": "backward", "launch": form, "dtype": "bfloat16",
                                              "logits": "natural" if form == "accurate" else "fold"},
                  fault=lambda hooks, form: "approximate tanh" if hooks == TRANSCENDENTAL else None),
        *cells_of(mosaic.a_call_over_runs_is_on_par_on_an_h100, "sm90",
                  native=lambda case, form: {"pass": "forward" if form == "forward" else "backward",
                                             "launch": form, "dtype": "bfloat16",
                                             "logits": "natural" if form == "accurate" else "fold"}),
        *cells_of(mosaic.a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100, "sm90",
                  native=lambda: {"pass": "backward", "launch": "default", "dtype": "bfloat16",
                                  "logits": "fold"}),
        *cells_of(route.an_fp16_call_on_an_h100_is_on_par, "sm90",
                  native=lambda case: {"pass": "backward", "launch": "route", "dtype": "float16",
                                       "logits": "fold", "hooks": case}),
        *cells_of(route.a_long_sequence_is_on_par_with_the_reference_path,
                  native=lambda length, dtype: {"pass": "backward", "launch": "route", "dtype": _dtype(dtype),
                                                "logits": "fold", "length": str(length)}),
        *cells_of(inplace.a_routes_one_launch_backward_is_on_par_on_an_h100, "sm90",
                  fault=lambda name, causal: "red at 475913063 (its float64 truth's float32 product)"),
        *cells_of(inplace.a_span_calls_one_launch_is_on_par_on_an_h100, "sm90",
                  fault=lambda name: "row 3171's packed windowed call, its dK and dV wrong and unrepeatable "
                                     "on an H100 before row 3180"),
    ]


KERNEL_CELLS = (kernels.flash_through_dews_attention_refuses_rows_it_would_misread,
                kernels.a_fused_kernel_takes_forward_mode_through_the_reference,
                kernels.cudnn_trains_odd_lengths_and_agrees_with_xla,
                kernels.cudnn_runs_a_calls_packed_documents_on_par,
                kernels.triton_trains_and_agrees_with_xla,
                kernels.triton_trains_at_large_logits_on_par_with_xla,
                kernels.triton_at_saturated_large_logits_is_finite_where_xla_is,
                kernels.a_mapped_flash_call_is_each_rows_own,
                kernels.a_fold_holds_the_kernels_grid_and_its_offsets,
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

LIBRARY_AXES = ("implementation", "kernel", "causal", "sunk", "window", "sinks", "as_mask", "call", "case")
"""A library kernel's native axes, the parameters of its checks a GPU alone shows (Main, 11-11): a library's
kernel (FlashAttention's, cuDNN's, tokamax's Triton) has no CPU form, so the kernel and every axis of how it
takes Dew's call (its mask, packing, offsets, window and sinks, and the call a requirement names) enter its
class's covering, beside each check itself; its shapes do not."""


def library_cells(check, **fixtures) -> list[Cell]:
    """`check`'s cells (`gpu_flow.cells_of`), each with its native axes: the check and its parameters a GPU
    alone shows (`LIBRARY_AXES`)."""
    return cells_of(check, native=lambda **given: {"check": check.__name__, **{
        name: repr(value) for name, value in given.items() if name in LIBRARY_AXES}}, **fixtures)


FUSED_PARTS = 2
"""The fused kernels' flow's parts (`in_parts`): 155 s whole on a GH200 (MAINGPU-709ddea7a)."""


@in_parts(FUSED_PARTS)
def test_the_fused_kernels_hold_on_this_gpu(without_deterministic_ops, part):
    """The test_kernels and masked-rows cells above a GPU runs (`gpu_flow.covering` over each library
    kernel's native axes, `library_cells`), without XLA's deterministic ops, each one's failure kept and all
    raised together, a part's share of them."""
    run_cells(parted(covering([
        *(cell for check in KERNEL_CELLS
          for cell in library_cells(check, without_deterministic_ops=without_deterministic_ops)
          for cell in [dataclasses.replace(cell, run=when(bf16_dot_runs(), BF16_PRODUCTS, cell.run))]),
        *library_cells(masked_rows.a_row_that_sees_no_key_reads_nothing_on_the_fused_gpu_kernels,
                       without_deterministic_ops=without_deterministic_ops),
    ]), part, FUSED_PARTS))


@pytest.mark.usefixtures("without_deterministic_ops")
def test_the_attention_matrix_holds_on_this_gpu():
    """Attention's check-1 matrix (tests/kernel_matrix.py): the route's training step on par with the
    reference path at each case a GPU-only red seeded (`kernel_matrix.seeded_cells`), each cell on its own
    GPU, its other pairs the CPU's, without XLA's deterministic ops (a deterministic run never derives the
    one launch its cells pin)."""
    run_cells(kernel_matrix.seeded_cells("attention"))


def test_the_decode_matrix_holds_on_this_gpu():
    """Decode's check-1 matrix (tests/kernel_matrix.py): a serving step's decode, dense and paged, on par with
    the reference path at each case a GPU-only red seeded (`kernel_matrix.seeded_cells`), each cell on its own
    GPU, its other pairs the CPU's, under the lane's deterministic ops."""
    run_cells(kernel_matrix.seeded_cells("decode"))


MOSAIC_PARTS = 8
"""The Mosaic flow's parts (`in_parts`): past the 3600 s deadline whole on a GH200 (MAINGPU-df3e6ac66), its
cells' kernels' compiles most of it."""


@pytest.mark.usefixtures("without_deterministic_ops")
@in_parts(MOSAIC_PARTS)
def test_the_mosaic_template_holds_on_an_h100(part):
    """The Mosaic template's cells on an H100 (`mosaic_cells`) a GPU runs (`gpu_flow.covering`: each cell
    kept for a GPU-only fault by name, and the fewest others covering its native axes), each one's failure
    kept and all raised together, a part's share of them, without XLA's
    deterministic ops: the fused backward forms its cells pin add dQ atomically, which a deterministic run
    never derives (`MosaicBackward.derived`), so held to a bitwise repeat there (check 1's
    `_repeat_promised`) they failed as unpaired draws (row 3171)."""
    run_cells(parted(covering(mosaic_cells()), part, MOSAIC_PARTS))
