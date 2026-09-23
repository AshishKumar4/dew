"""Where one device's traced time went, read the same way by every tool.

`tools/benchmark_step.py` reads JAX traces and `tools/reference_runs` reads
both JAX and torch.profiler traces; each reduces its own trace format to
`(name, start_ns, end_ns)` kernel records per device, and this module splits
and names them. The standard library only, so a torch venv imports it too.
"""

import re
from collections.abc import Sequence

# NCCL names its kernels after the collective it runs:
# `ncclDevKernel_AllGather_RING_LL`, `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`,
# `ncclDevKernel_SendRecv` (collective-permute and all-to-all), and the same
# with `ncclKernel_` before NCCL 2.19.
COLLECTIVES = ("AllReduce", "AllGather", "ReduceScatter", "SendRecv", "Broadcast", "Reduce")

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
    ("conv", ("conv", "convolution", "fprop", "dgrad", "wgrad", "implicit")),
    ("gemm", ("*gemm", "cublas", "cutlass", "nvjet", "xmma", "matmul", "dot", "splitk")),
    ("optimizer", ("multi_tensor_apply",)),
    ("loss", ("nll_loss", "cross_entropy", "softmaxforward", "softmaxbackward", "logsumexp")),
    ("norm", ("layer_norm", "rms_norm", "group_norm", "batch_norm")),
    ("reduce", ("reduce",)),
    ("convert", ("convert",)),
    ("copy", ("memcpy", "memset", "copy", "transpose", "concatenate", "gather", "scatter", "slice",
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


def union(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    """Nanoseconds two disjoint, sorted interval lists share."""
    shared, i, j = 0, 0, 0
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        shared += max(0, end - start)
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return shared


def length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _host_to_device(name: str) -> bool:
    """JAX's `MemcpyH2D`, torch's `Memcpy HtoD (Pinned -> Device)`."""
    lowered = name.lower()
    return "memcpyh2d" in lowered or "htod" in lowered


def window_split(events: Sequence[tuple[str, int, int]]) -> dict[str, int]:
    """One device's traced window, in nanoseconds, from its `(name, start_ns,
    end_ns)` kernels.

    `window` runs from the first kernel's start to the last one's end and
    `busy` is the union of every kernel. A kernel is a collective when NCCL
    runs it and compute otherwise: `compute` and `communication` are their
    unions, and `exposed_communication` the collective time no compute kernel
    ran beside, what overlap did not hide. Each of `COLLECTIVES` gets the union
    of its own kernels. The idle rest is split by what ends each gap: a
    host-to-device copy (`idle_input`, the device waited on a batch) or
    anything else (`idle_host`, it waited on dispatch or host work). So the
    window is compute + exposed_communication + idle_input + idle_host.

    A collective's kernel spans its wait for the slowest peer as well as the
    transfer, so communication includes the skew between devices."""
    if not events:
        raise ValueError("no device kernels to split")
    ordered = sorted(events, key=lambda event: event[1])
    figures = {"window": 0, "busy": 0, "compute": 0, "communication": 0, "exposed_communication": 0,
               "idle_input": 0, "idle_host": 0}
    high = ordered[0][2]
    for name, start, end in ordered[1:]:
        if start > high:
            figures["idle_input" if _host_to_device(name) else "idle_host"] += start - high
        high = max(high, end)
    figures["window"] = high - ordered[0][1]
    figures["busy"] = length(union([(start, end) for _, start, end in ordered]))
    collectives = [(name, start, end) for name, start, end in ordered if "nccl" in name.lower()]
    compute = union([(start, end) for name, start, end in ordered if "nccl" not in name.lower()])
    every = union([(start, end) for _, start, end in collectives])
    figures["compute"] = length(compute)
    figures["communication"] = length(every)
    figures["exposed_communication"] = length(every) - overlap(every, compute)
    for kind in COLLECTIVES:
        figures[kind] = length(union([
            (start, end) for name, start, end in collectives
            if f"_{kind.lower()}_" in f"_{name.lower()}_".replace("(", "_")]))
    return figures
