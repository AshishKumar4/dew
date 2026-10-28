"""The tools' names for a traced kernel: its category (training's, decode's), summed kernel
durations by name or category, and XProf's summed self time by program. A capture's intervals,
its busy time, idle and window, are Dew's one reading of them, `dew.telemetry.trace`. The
standard library only at import, so a torch venv imports it too.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

# The decode report groups movement and reductions separately from training's
# kernel categories, and keeps random-number generation visible.
DECODE_CATEGORIES = {
    "attention": r"fmha|flash|cudnn|attention",
    "gemm": r"gemm|cutlass|cublas|xmma|matmul|_dot",
    "movement": r"memcpy|copy|scatter|gather|transpose|concatenate",
    "reduction": r"reduce|argmax|topk|sort",
    "random": r"random|threefry|philox",
}

# A kernel's category from the tokens of its name, first match wins: XLA
# names its fusions after the ops they hold (`loop_convert_fusion`,
# `input_add_reduce_fusion`, `gemm_fusion_dot`), cuDNN and cuBLAS after the
# kernel family, torch's ATen after the operator
# (`vectorized_elementwise_kernel`, `nll_loss_forward_reduce_cuda_kernel_2d`,
# the fused AdamW's `multi_tensor_apply_kernel`). A needle matches whole
# tokens, not substrings, so `convert` is not `conv`; a needle of several
# tokens (`nll_loss`) matches them in a row; `*gemm` matches a token ending in
# `gemm`, since cuBLAS writes the family into one token (`s16816gemm`).
KERNEL_CATEGORIES = (
    ("attention", ("sdpa", "fmha", "flash", "attention")),
    ("conv", ("conv", "conv2d", "convolution", "fprop", "dgrad", "wgrad", "wgrad2d", "implicit")),
    ("gemm", ("*gemm", "cublas", "cutlass", "nvjet", "xmma", "matmul", "dot", "splitk")),
    ("optimizer", ("multi_tensor_apply",)),
    ("loss", ("nll_loss", "cross_entropy", "softmaxforward", "softmaxbackward", "logsumexp")),
    ("norm", ("layer_norm", "rms_norm", "group_norm", "batch_norm")),
    ("reduce", ("reduce",)),
    ("convert", ("convert",)),
    ("copy", ("memcpy", "memset", "copy", "clone", "transpose", "concatenate", "cat",
              "gather", "scatter", "slice",
              "pad", "broadcast", "select", "dynamic", "index", "embedding", "catarraybatchedcopy")),
    ("elementwise", ("fusion", "elementwise", "triton")),
)


def kernel_category(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "collective"
    if "cudnn::fusion" in lowered:
        # cuDNN's helpers around its flash kernel (dO.O, dQ rearrangement).
        return "attention"
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    run = f"_{'_'.join(tokens)}_"
    for category, needles in KERNEL_CATEGORIES:
        for needle in needles:
            if (any(token.endswith(needle[1:]) for token in tokens) if needle.startswith("*")
                    else f"_{needle}_" in run):
                return category
    return "other"


def decode_category(name: str) -> str:
    return next((kind for kind, pattern in DECODE_CATEGORIES.items()
                 if re.search(pattern, name, re.I)), "other")


def kernel_totals[Time: (int, float)](
    events: Iterable[tuple[str, Time, Time]], classify: Callable[[str], str] = str,
) -> tuple[dict[str, Time], dict[str, int]]:
    """Summed kernel durations and calls by name or category, including overlapping streams."""
    durations: dict[str, Time] = {}
    counts: dict[str, int] = {}
    for name, start, end in events:
        kind = classify(name)
        durations[kind] = durations.get(kind, 0) + (end - start)
        counts[kind] = counts.get(kind, 0) + 1
    return durations, counts


def program_times(rows: Sequence[dict]) -> dict[str, float]:
    """XProf's summed self time in microseconds, by program id."""
    programs: dict[str, float] = {}
    for row in rows:
        program = str(row.get("program_id"))
        programs[program] = programs.get(program, 0.0) + float(row["total_self_time"])
    return programs


