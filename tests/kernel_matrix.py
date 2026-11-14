"""The template kernel families' check-1 matrices (Main, 2026-11-02, after the attention dK escape on main).

Each family declares its axes (`AXES`) and one matrix cell function (`CELLS`): a route-level training flow,
the forward and its backward in one program as a user calls them, held to check 1 with the float64 terms,
its launch form pinned. Its cells cover every pair of axis values the family's route serves
(docs/design/testing.md's covering scenarios): what it serves is read from the route's own analyses
(`supported`, the cell module's), never a second list. The CPU runs every cell, interpreted. Main's GPU
tier runs `FLOWS` on an H100 and an A100, each over the cases a GPU-only red seeded (`seeded_cells`: the
owner's ruling of 2026-11-11, row 5461's covering set), and tests/test_kernel_matrix.py holds each family's
cells to its supported pairs (`covered`). A TPU tier runs `TPU_FLOWS`. Nothing here needs a GPU.
"""

import functools
import importlib
import itertools

import gpu_flow
import jax.numpy as jnp

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
        "kernel": ("flash", "triton", "cudnn", "template", "inference", "tpu"),
        "width": ("64", "128", "256"),
        "heads": ("mha", "gqa-8-2"),
        "mask": ("causal", "full", "documents"),
        "dtype": ("bfloat16", "float16", "float32"),
    },
    "decode": {
        "generation": ("sm90", "sm80", "v5e"),
        "layout": ("dense", "paged"),
        "heads": ("gqa-16-8", "mha-8"),
        "hook": ("plain", "sinks", "softcap"),
        "dtype": ("bfloat16", "float16", "float32"),
        "width": ("64", "128"),
        "kernel": ("decode", "cudnn", "triton", "template", "tpu"),
    },
    "gemm": {
        "generation": ("sm90", "sm80", "v5e"),
        "kernel": ("mosaic", "triton", "cutlass"),
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
    "blockwise_fp8": {
        "generation": ("sm90",),
        "columns": ("128", "1"),
        "groups": ("1", "4"),
        "out_dtype": ("float32", "bfloat16"),
    },
    "mxfp8": {
        "generation": ("sm100",),
        "format": ("mxfp8", "nvfp4"),
        "width": ("128", "256"),
        "groups": ("1", "4"),
        "out_dtype": ("float32", "bfloat16"),
    },
    "nvfp4": {
        "generation": ("sm90", "sm80"),
        "width": ("128", "256"),
        "depth": ("256", "512"),
        "out_dtype": ("float32", "bfloat16"),
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
    "gated_mlp": {
        "generation": ("sm90", "sm100"),
        "activation": ("swiglu", "geglu"),
        "limit": ("none", "clamped"),
        "dtype": ("bfloat16", "float16"),
    },
    "recurrence": {
        "generation": ("sm90", "sm80"),
        "form": ("gated delta net", "kimi delta attention", "write gate", "rated scan"),
        "kernel": ("mosaic_gpu", "xla", "fla", "pallas"),
        "dtype": ("bfloat16", "float32"),
        "documents": ("one", "packed"),
    },
    "recurrence_tpu": {
        "generation": ("v6e", "v5e"),
        "form": ("rated scan",),
        "kernel": ("tpu",),
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
        "layer": ("gated delta net", "kimi delta attention", "mamba2"),
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
    "narrow_copies": {
        "generation": ("sm90", "sm80"),
        "dtype": ("bfloat16", "float16"),
        "form": ("plain", "scanned"),
        "window": ("one", "accumulated", "split"),
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
norm) into the prologue (its consumers after the product in XLA), or both; the consumers, a bias then a cast,
or a SiLU of twice the product plus a residual; the operands' dtype. The joined projections' (the choices
`gate_up` and `delta_gates`): the GPU; a gated MLP's gate and up projections joined, or a gated delta net's b
and a; the MLP's activation, SwiGLU or GeGLU (`none` for the delta net, which has no gated product);
projections with no bias or one; a gate times its multiplier or not (`plain` for the delta net); the operands'
dtype. The fused gated MLP's (the choice `gated_mlp`): the GPU; its gated product, SwiGLU or GeGLU; its
projections unclamped or clamped (`swiglu_limit`); the operands' dtype. The adapted product's (the choice
`low_rank`): the GPU; the adapter's rank; a projection with no bias
or one; the operands' dtype. The blockwise-scaled FP8 product's (the choice `blockwise_fp8`): the GPU (an
H100's FP8 tensor cores); the columns a column scale covers (a weight block's 128, or one); the groups;
the output's dtype. The block-scaled product's (the choices `mxfp8` and `nvfp4_block`): the GPU (a B200's
block-scaled tensor cores); the block format (MXFP8's E4M3 under E8M0 every 32, NVFP4's E2M1 under E4M3
every 16); the matrix's width; the groups; the output's dtype. The NVFP4 product's (the choice `nvfp4`): the
GPU; the matrix's width and depth;
the output's dtype. The recurrence's (the choices `gated_delta_rule` and `ssd`): the GPU; the form
(tests/recurrence_matrix.py's `FORMS`); the kernel under test (the template and the portable math, its
plain-JAX mainloop, each held to its form's port; each port held to the portable math); the operands' dtype;
one sequence or packed documents; on the TPU tier (`recurrence_tpu`) the TPU, the rated scan on its TPU
kernel. The short conv's: the GPU; a gated delta net's training shapes (Qwen3.5-0.8B's or 9B's); the operands'
dtype; whole rows or packed documents. The selective scan's: the GPU; the recorded mamba_ssm case
(tests/selective_support.py's `SEEDED_CASES`). The recurrent decode's: the GPU; the layer (a gated delta net,
Kimi delta attention, Mamba-2); the fused step or the step alone; the dtype its recurrent state is held in.
The fused norms' and SiLU products' (the choices `rms_norm` and `swiglu`): the GPU; the RMS norm or the SwiGLU
product; the norm's plain, stream (a residual stream and its norm) or gated form, or the product of a gate and
an up projection apart or of one packed projection; the operands' dtype; the feature width, narrower than a
warp, ragged, a whole row of a program, or past the norm's register tile (4096 and 16384 features: the norm
refuses the last). The narrow copies' (the choice `narrow_copies`): the GPU; the dtype the decoder computes
in; its layers plain or scanned; one microbatch an update, two accumulated, or one batch read in two parts."""

SEEDED: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    "attention": (
        (3522, ("sm90", "256", "gqa-8-4", "causal", "plain", "two", "float32", "fast")),
    ),
    "decode": (
        (5268, ("sm90", "dense", "mha-8", "plain", "float32", "64", "decode")),
        (5404, ("sm80", "dense", "gqa-16-8", "softcap", "float32", "128", "template")),
    ),
    "gemm": (
        (5406, ("sm80", "triton", "grouped", "n-major", "row-scale", "float16")),
    ),
    "fused_gemm": (
        (5416, ("sm90", "both", "silu-residual", "bfloat16")),
    ),
    "low_rank": (
        (5407, ("sm80", "8", "bias", "float32")),
        (5418, ("sm90", "8", "none", "bfloat16")),
    ),
    "nvfp4": (
        (5408, ("sm80", "128", "256", "bfloat16")),
        (5419, ("sm90", "128", "256", "float32")),
    ),
    "blockwise_fp8": (
        (5415, ("sm90", "128", "1", "float32")),
    ),
    "recurrence": (
        (5330, ("sm90", "gated delta net", "mosaic_gpu", "bfloat16", "one")),
        (5330, ("sm90", "rated scan", "pallas", "bfloat16", "packed")),
        (5409, ("sm80", "gated delta net", "fla", "bfloat16", "packed")),
        (5409, ("sm80", "rated scan", "pallas", "bfloat16", "one")),
        (5409, ("sm80", "rated scan", "mosaic_gpu", "bfloat16", "one")),
    ),
    "short_conv": (
        (5411, ("sm80", "0.8b", "bfloat16", "packed")),
        (5441, ("sm90", "0.8b", "bfloat16", "one")),
        (5441, ("sm90", "0.8b", "float32", "one")),
        (5441, ("sm90", "9b", "bfloat16", "one")),
        (5441, ("sm80", "9b", "float32", "one")),
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
    "gated_mlp": ("gated_mlp_matrix", "a_fused_mlps_training_step_is_on_par"),
    "low_rank": ("low_rank_matrix", "an_adapted_projections_training_step_is_on_par"),
    "nvfp4": ("nvfp4_matrix", "an_nvfp4_products_training_step_is_on_par"),
    "blockwise_fp8": ("blockwise_matrix", "a_blockwise_fp8_product_is_on_par"),
    "mxfp8": ("mx_matrix", "an_mxfp8_product_is_on_par"),
    "recurrence": ("recurrence_matrix", "a_recurrences_training_step_is_on_par"),
    "recurrence_tpu": ("recurrence_matrix", "a_recurrences_training_step_on_a_tpu_is_on_par"),
    "short_conv": ("short_conv_matrix", "a_short_convs_training_step_is_on_par"),
    "selective_scan": ("selective_matrix", "a_selective_scan_is_as_exact_as_mamba_ssm"),
    "delta_decode": ("delta_decode_matrix", "a_decode_token_is_on_par"),
    "norm_swiglu": ("norm_swiglu_matrix", "a_fused_entrys_training_step_is_on_par"),
    "narrow_copies": ("narrow_copies_matrix", "a_narrow_copy_training_run_is_on_par"),
}
"""Each family's matrix cell function, by its module and name (`gpu_flow.cell`)."""

def seeded_cells(family: str) -> list:
    """The cells of `family`'s matrix a GPU runs (`FLOWS`): each case a ledger row names (`SEEDED`), a red a
    GPU alone showed, whose own case proves the row closed where it was found (row 5461's covering set); every
    other pair the CPU's interpreted cells cover, and the compile gate, and a GPU's default routes the model
    flows' (tests/test_gpu_flows.py)."""
    seeded = {case for _, case in SEEDED.get(family, ())}
    cells = zip(gpu_flow.cells_of(check(family)), covered(family), strict=True)
    return [cell for cell, (_, combination) in cells if tuple(combination.values()) in seeded]


FLOWS: tuple[str, ...] = (
    "tests/test_gpu_attention.py::test_the_attention_matrix_holds_on_this_gpu",
    "tests/test_gpu_attention.py::test_the_decode_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_gemm_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_fused_gemm_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_low_rank_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_nvfp4_matrix_holds_on_this_gpu",
    "tests/test_gpu_gemm.py::test_the_blockwise_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_recurrence_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_short_conv_matrix_holds_on_this_gpu",
)
"""The flows that run the matrices' seeded cases on the GPU tier (`seeded_cells`), by pytest node id: a
family no GPU-only red seeded runs on the CPU alone."""

TPU_FLOWS: tuple[str, ...] = (
    "tests/test_attention_splash.py::test_the_attention_kernels_matrix_holds_on_this_tpu",
    "tests/test_attention_splash.py::test_the_decode_matrix_holds_on_this_tpu",
)
"""The flows that run the matrices' TPU cells on the TPU tier (`-m tpu`), by pytest node id."""


def ops() -> dict[str, list]:
    """Every kernel op Dew's dispatches serve (`selection.Op`), by its choice, read from the production
    modules under `dew.nn` as they define them, a module's own or held in a module's mapping or sequence (a
    layer's selectors by choice, `joined.JOINS`): none is listed here."""
    import pkgutil

    import dew.nn
    from dew.nn.kernels import selection

    def held(value):
        if isinstance(value, selection.Op):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from held(item)
        elif isinstance(value, list | tuple):
            for item in value:
                yield from held(item)

    found: dict[str, list] = {}
    for module in pkgutil.walk_packages(dew.nn.__path__, "dew.nn."):
        for value in vars(importlib.import_module(module.name)).values():
            for op in held(value):
                if all(op is not seen for seen in found.get(op.choice, [])):
                    found.setdefault(op.choice, []).append(op)
    return found


def head_shapes(case: dict[str, str], heads: dict[str, tuple[int, int]]) -> tuple:
    """An attention family's case's query heads and key heads (by its `heads` axis, `heads`' table), head
    width and dtype, as its cells draw them."""
    query_heads, kv_heads = heads[case["heads"]]
    return query_heads, kv_heads, int(case["width"]), jnp.dtype(case["dtype"])


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
            pairs |= gpu_flow.held(combination, values=False)
    return pairs


def covered(family: str) -> list[tuple[str, dict[str, str]]]:
    """Each registered cell of `family` by its label beside the combination it covers (`gpu_flow.covered`)."""
    return gpu_flow.covered(check(family))


def covering(family: str) -> list[tuple[str, ...]]:
    """Combinations the route serves in which every supported pair occurs (`pairwise`), each as its values in
    `AXES`' order, the family's seeded cases (`SEEDED`) first. A family's cell module takes these as its
    cases, regenerated where its axes or its route's support change (tests/test_kernel_matrix.py fails until
    they are)."""
    served = [combination for combination in combinations(family)
              if supported(family, tuple(combination.items()))]
    seeded = [dict(zip(AXES[family], case, strict=True)) for _, case in SEEDED.get(family, ())]
    return [tuple(combination.values()) for combination in pairwise(served, seeded)]


def pairwise(served: list[dict], seeded: list[dict]) -> list[dict]:
    """Combinations of `served` (each an axis's value by its name) in which every pair of axis values some
    served combination holds occurs: `seeded` first, then the fewest others holding the pairs left, chosen
    as `gpu_flow.greedy` chooses."""
    def pairs(group: list[dict]) -> set:
        return set().union(*(gpu_flow.held(combination, values=False) for combination in group))

    left = pairs(served) - pairs(seeded)
    return [*seeded, *(served[index] for index in gpu_flow.greedy(served, left, values=False))]
