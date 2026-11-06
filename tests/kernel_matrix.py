"""The template kernel families' check-1 matrices (Main, 2026-11-02, after the attention dK escape on main).

Each family declares its axes (`AXES`) and one matrix cell function (`CELLS`): a route-level training flow,
the forward and its backward in one program as a user calls them, held to check 1 with the float64 terms,
its launch form pinned. Its cells cover every pair of axis values the family's route serves
(docs/design/testing.md's covering scenarios): what it serves is read from the route's own analyses
(`supported`, the cell module's), never a second list. Main's GPU tier runs `FLOWS` on an H100 and an A100
after a publish whose diff reaches the kernels, and tests/test_kernel_matrix.py holds each family's cells to
its supported pairs (`covered`). Nothing here needs a GPU.
"""

import functools
import importlib
import itertools

import gpu_flow

AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "attention": {
        "generation": ("sm90", "sm80"),
        "width": ("64", "128", "256"),
        "heads": ("mha", "gqa-8-2", "gqa-8-4"),
        "mask": ("causal", "full", "documents"),
        "hook": ("plain", "key-bias", "tile-bias", "softcap"),
        "launch": ("one", "two", "shared-keys"),
        "dtype": ("bfloat16", "float16", "float32"),
        "form": ("fast", "accurate"),
    },
    "attention_kernels": {
        "generation": ("sm90", "sm80", "v5e"),
        "kernel": ("flash", "triton", "cudnn", "template", "tpu"),
        "width": ("64", "128", "256"),
        "heads": ("mha", "gqa-8-2"),
        "mask": ("causal", "full", "documents"),
        "dtype": ("bfloat16", "float16", "float32"),
    },
    "decode": {
        "generation": ("sm90", "sm80"),
        "layout": ("dense", "paged"),
        "heads": ("gqa-16-8", "mha-8"),
        "hook": ("plain", "sinks", "softcap"),
        "dtype": ("bfloat16", "float16", "float32"),
        "width": ("64", "128"),
        "kernel": ("decode", "cudnn", "triton", "template"),
    },
    "gemm": {
        "generation": ("sm90", "sm80", "v5e"),
        "kernel": ("mosaic_gpu", "triton", "cutlass", "mosaic_tpu"),
        "form": ("dense", "grouped"),
        "matrix": ("n-major", "k-major"),
        "hook": ("plain", "bias", "row-scale", "paired"),
        "dtype": ("bfloat16", "float16"),
    },
    "fused_gemm": {
        "generation": ("sm90", "sm80"),
        "stage": ("epilogue", "prologue", "both"),
        "consumer": ("bias-cast", "silu-residual"),
        "dtype": ("bfloat16", "float16"),
    },
    "low_rank": {
        "generation": ("sm90", "sm80"),
        "rank": ("8", "16"),
        "bias": ("none", "bias"),
        "dtype": ("bfloat16", "float16", "float32"),
    },
    "join": {
        "generation": ("sm90", "sm80"),
        "join": ("gate_up", "delta_gates"),
        "activation": ("swiglu", "geglu", "none"),
        "bias": ("none", "bias"),
        "gate": ("plain", "scaled"),
        "dtype": ("bfloat16", "float16", "float32"),
    },
    "recurrence": {
        "generation": ("sm90", "sm80"),
        "form": ("gated delta net", "kimi delta attention", "write gate", "rated scan"),
        "kernel": ("mosaic_gpu", "xla", "derived"),
        "dtype": ("bfloat16", "float32"),
        "documents": ("one", "packed"),
    },
    "short_conv": {
        "generation": ("sm90", "sm80"),
        "model": ("0.8b", "9b"),
        "dtype": ("bfloat16", "float32"),
        "documents": ("one", "packed"),
    },
    "selective_scan": {
        "generation": ("sm90", "sm80"),
        "case": ("bf16", "fp32", "decays-near-one", "decays-near-zero", "saturated-steps", "documents",
                 "long"),
    },
    "delta_decode": {
        "generation": ("sm90", "sm80"),
        "kernel": ("fused", "step"),
        "state": ("float32", "bfloat16"),
    },
    "norm_swiglu": {
        "generation": ("sm90", "sm80"),
        "op": ("rms_norm", "swiglu"),
        "form": ("plain", "stream", "gated", "apart", "packed"),
        "dtype": ("bfloat16", "float16", "float32"),
        "width": ("narrow", "ragged", "wide", "beyond"),
    },
}
"""Each family's axes and their values. Attention's: the GPU; the head width; 8 query heads over 8 key heads,
2, or 4; a causal, a full or a causal call over three documents and padding; no hook, two key or two tile
parameters of one shape added to the logits (two in-place gradients of one shape, row 3180's), or a softcap;
the backward in one launch, two, or one whose consumers share a key block; the operands' dtype; the fast or
the accurate form. Decode's (the choices `decode` and `paged_decode`): the GPU; a dense cache's step or a
paged pool's; 16 query heads over 8 key heads, or 8 over 8; no hook, attention sinks, or a softcap; the
operands' dtype; the head width. GEMM's: the GPU or TPU; the grouped matmul's kernel; a dense product or one
over three groups (one of them empty); the matrix stored N-major (`[depth, width]`) or K-major
(`transpose_rhs`, `[width, depth]`); no hook, a bias (a group's under a grouped call), a scale a row, or a
paired gate and up product under SwiGLU; the operands' dtype. The fused GEMM's (the choices `gemm_epilogue`
and `gemm_prologue`): the GPU; the stage a region fuses, its consumers into the epilogue, its producer (an RMS
norm) into the prologue (its consumers
after the product in XLA), or both; the consumers, a bias then a cast, or a SiLU of twice the product plus a
residual; the operands' dtype. The joined projections' (the choices `gate_up` and `delta_gates`): the GPU; a
gated MLP's gate and up projections joined, or a gated delta net's b and a; the MLP's activation, SwiGLU or
GeGLU (`none` for the delta net, which has no gated product); projections with no bias or one; a gate times
its multiplier or not (`plain` for the delta net); the operands' dtype. The adapted product's (the choice
`low_rank`): the GPU; the adapter's rank; a projection with no bias or one; the operands' dtype. The
recurrence's (the choices `gated_delta_rule` and `ssd`): the GPU; the form (tests/recurrence_matrix.py's
`FORMS`); the kernel the call names (its form's port, fla's or mamba_ssm's, the reference); the operands'
dtype; one sequence or packed
documents. The short conv's: the GPU; a gated delta net's training shapes (Qwen3.5-0.8B's or 9B's); the
operands' dtype; whole rows or packed documents. The selective scan's: the GPU; the recorded mamba_ssm case
(tests/selective_support.py's `SEEDED_CASES`). The recurrent decode's: the GPU; the fused step or the step
alone; the dtype its recurrent state is held in. The fused norms' and SiLU products' (the choices `rms_norm`
and `swiglu`): the GPU; the RMS norm or the SwiGLU product; the norm's plain, stream (a residual stream and
its norm) or gated form, or the product of a gate and an up projection apart or of one packed projection; the
operands' dtype; the feature width, narrower than a warp, ragged, a whole row of a program, or past the norm's
register tile (4096 and 16384 features: the norm refuses the last)."""

SEEDED: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    "attention": (
        (3522, ("sm90", "256", "gqa-8-4", "causal", "plain", "two", "float32", "fast")),
    ),
}
"""Each family's cases a ledger row names, by the row, its axis values in `AXES`' order: its cells take them
whatever the covering takes besides (`covering`), so the row's own case proves it closed."""

CELLS: dict[str, tuple[str, str]] = {
    "attention": ("attention_matrix", "a_routes_training_step_is_on_par"),
    "attention_kernels": ("attention_kernels_matrix", "a_kernels_training_step_is_on_par"),
    "decode": ("decode_matrix", "a_decode_step_is_on_par"),
    "gemm": ("gemm_matrix", "a_gemm_training_step_is_on_par"),
    "fused_gemm": ("fused_gemm_matrix", "a_fused_gemm_training_step_is_on_par"),
    "join": ("join_matrix", "a_joined_layers_training_step_is_on_par"),
    "low_rank": ("low_rank_matrix", "an_adapted_projections_training_step_is_on_par"),
    "recurrence": ("recurrence_matrix", "a_recurrences_training_step_is_on_par"),
    "short_conv": ("short_conv_matrix", "a_short_convs_training_step_is_on_par"),
    "selective_scan": ("selective_matrix", "a_selective_scan_is_as_exact_as_mamba_ssm"),
    "delta_decode": ("delta_decode_matrix", "a_decode_token_is_on_par"),
    "norm_swiglu": ("norm_swiglu_matrix", "a_fused_entrys_training_step_is_on_par"),
}
"""Each family's matrix cell function, by its module and name (`gpu_flow.cell`)."""

FLOWS: tuple[str, ...] = (
    "tests/test_gpu_attention.py::test_the_attention_matrix_holds_on_this_gpu",
    "tests/test_gpu_attention.py::test_the_attention_kernels_matrix_holds_on_this_gpu",
    "tests/test_gpu_attention.py::test_the_decode_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_gemm_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_fused_gemm_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_join_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_low_rank_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_recurrence_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_short_conv_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_selective_scan_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_recurrent_decode_matrix_holds_on_this_gpu",
    "tests/test_gpu_numerics.py::test_the_norm_swiglu_matrix_holds_on_this_gpu",
)
"""The flows that run the matrices' cells on the GPU tier, by pytest node id."""

TPU_FLOWS: tuple[str, ...] = (
    "tests/test_attention_splash.py::test_the_attention_kernels_matrix_holds_on_this_tpu",
)
"""The flows that run the matrices' TPU cells on the TPU tier (`-m tpu`), by pytest node id."""


def check(family: str):
    """`family`'s matrix cell function (`CELLS`)."""
    module, name = CELLS[family]
    return getattr(importlib.import_module(module), name)


def combinations(family: str) -> list[dict[str, str]]:
    """Every combination of `family`'s axis values, in `AXES`' order."""
    axes = AXES[family]
    return [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]


@functools.cache
def supported(family: str, combination: tuple[tuple[str, str], ...]) -> bool:
    """Whether `family`'s route serves `combination` (its axis-value pairs), read from the route's own
    analyses by the family's cell module (`supported` there)."""
    return importlib.import_module(CELLS[family][0]).supported(dict(combination))


def supported_pairs(family: str) -> set[tuple[tuple[str, str], tuple[str, str]]]:
    """Every pair of axis values (each an axis and its value) that some combination the route serves holds."""
    pairs = set()
    for combination in combinations(family):
        if supported(family, tuple(combination.items())):
            pairs |= set(itertools.combinations(combination.items(), 2))
    return pairs


def covered(family: str) -> list[tuple[str, dict[str, str]]]:
    """Each registered cell of `family` by its label beside the combination it covers (`gpu_flow.covered`)."""
    return gpu_flow.covered(check(family))


def covering(family: str) -> list[tuple[str, ...]]:
    """Combinations the route serves in which every supported pair occurs (`supported_pairs`), each as its
    values in `AXES`' order: the family's seeded cases (`SEEDED`) first, then the combination holding the most
    pairs not yet covered, a tie to the first in `combinations`' order. A family's cell module takes these as
    its cases, regenerated where its axes or its route's support change (tests/test_kernel_matrix.py fails
    until they are)."""
    left = supported_pairs(family)
    served = [combination for combination in combinations(family)
              if supported(family, tuple(combination.items()))]
    chosen = []
    for _, case in SEEDED.get(family, ()):
        combination = dict(zip(AXES[family], case, strict=True))
        chosen.append(case)
        left -= set(itertools.combinations(combination.items(), 2))
    while left:
        best = max(served, key=lambda combination: len(
            left & set(itertools.combinations(combination.items(), 2))))
        chosen.append(tuple(best.values()))
        left -= set(itertools.combinations(best.items(), 2))
    return chosen
