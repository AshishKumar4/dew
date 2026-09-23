"""What both sides of a reference run share: the record, the schedule, the
FLOP count and the kernel categories.

The torch generators run in a venv without Dew, the Dew generators in one
without torch, so this module imports numpy and the standard library only.
Every number the comparison reads is produced by one definition here, so a
difference between two records is a difference between the runs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

# Dense bf16 tensor throughput with fp32 accumulation, the figure MFU is
# taken against. GA102 whitepaper, appendix table: RTX 3090 71 TFLOPS dense
# (142 with sparsity). DistTrain's instrumentation table carries the same.
PEAK_BF16 = {"NVIDIA GeForce RTX 3090": 71e12, "NVIDIA GeForce RTX 4080": 97.5e12,
             "NVIDIA A100": 312e12, "NVIDIA L4": 121e12}

EVIDENCE = Path(os.environ.get(
    "REFERENCE_RUNS_DIR", Path.home() / ".cache/dew/verification-evidence/reference-runs"))


def peak_flops(device_name: str) -> float | None:
    keys = [key for key in PEAK_BF16 if device_name.startswith(key)]
    return PEAK_BF16[max(keys, key=len)] if keys else None


def warmup_cosine(step: int, *, init: float, peak: float, warmup: int, decay_steps: int,
                  end: float) -> float:
    """`optax.warmup_cosine_decay_schedule` at `step`, the count of updates
    already applied: linear from `init` to `peak` over `warmup` updates, then
    a half cosine to `end` at `decay_steps`."""
    if step < warmup:
        return init + (peak - init) * step / warmup
    span = decay_steps - warmup
    progress = min(step - warmup, span) / span
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * progress))


def train_flops_per_token(config: Mapping, seq: int) -> float:
    """Model FLOPs of one trained token, forward and backward, from an HF
    config.json: 6 per multiply-add-pair of every matmul weight a token
    passes through, plus the attention scores and values at 12 L H Q T (the
    causal half is not taken off, as torchtitan and the PaLM appendix count).

    A tied embedding counts once, as the head's matmul; the lookup is not a
    matmul. A routed layer counts its router and `num_experts_per_tok`
    experts, the work a token does, not the parameters it owns."""
    kind = config["model_type"]
    hidden, vocab, layers = config["hidden_size"], config["vocab_size"], config["num_hidden_layers"]
    head = hidden * vocab
    if kind == "mamba2":
        inner = config["expand"] * hidden
        heads, groups, state = config["num_heads"], config["n_groups"], config["state_size"]
        in_proj = hidden * (2 * inner + 2 * groups * state + heads)
        out_proj = inner * hidden
        # The SSD recurrence's state update and readout: two multiply-adds
        # per (channel, state) pair per token, counted like a matmul weight.
        scan = 2 * inner * state
        return 6.0 * (layers * (in_proj + out_proj + scan) + head)
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config.get("head_dim") or hidden // heads
    attention = hidden * head_dim * (2 * heads + 2 * kv_heads)
    if kind in ("qwen3_moe", "mixtral"):
        experts = config.get("num_experts") or config["num_local_experts"]
        active = config["num_experts_per_tok"]
        width = config.get("moe_intermediate_size") or config["intermediate_size"]
        mlp = hidden * experts + active * 3 * hidden * width
    else:
        mlp = 3 * hidden * config["intermediate_size"]
    matmul = layers * (attention + mlp) + head
    return 6.0 * matmul + 12.0 * layers * heads * head_dim * seq


# Kernel categories, first match wins, read off lower-cased kernel names of
# both vocabularies: XLA's fusions and cuDNN/cuBLAS custom calls, and torch's
# ATen, cuBLAS, flash and NCCL kernels. A needle matches anywhere in the name,
# except one written "=word", which matches a whole alphanumeric token, so
# XLA's `loop_convert_fusion` is not a convolution and cuBLAS's
# `s16816gemm` still is a GEMM.
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("collective", ("nccl",)),
    ("attention", ("flash", "fmha", "sdpa", "cudnn::fusion", "attention")),
    ("conv", ("=conv", "convolution", "fprop", "dgrad", "wgrad", "implicit")),
    ("gemm", ("gemm", "cublas", "cutlass", "nvjet", "xmma", "matmul", "splitk", "=dot")),
    ("optimizer", ("multi_tensor_apply", "adam")),
    ("loss", ("softmax", "nll_loss", "cross_entropy", "logsumexp")),
    ("norm", ("layer_norm", "group_norm", "batch_norm", "rms_norm", "layernorm", "groupnorm")),
    ("reduce", ("reduce",)),
    ("convert", ("convert",)),
    ("copy", ("memcpy", "memset", "=copy", "transpose", "concatenate", "catarray", "gather",
              "scatter", "=index", "=slice", "=pad", "=broadcast", "embedding", "dynamic")),
    ("elementwise", ("elementwise", "fusion", "triton")),
)


def kernel_category(name: str) -> str:
    lowered = name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    for category, needles in CATEGORIES:
        if any(needle[1:] in tokens if needle.startswith("=") else needle in lowered
               for needle in needles):
            return category
    return "other"


def kernel_summary(kernels: Sequence[tuple[str, int, int, int]], steps: int) -> dict:
    """Per-step device time from `(name, start_ns, end_ns, device)` kernel
    records of `steps` traced steps.

    Busy is the union of one device's kernel intervals, averaged over the
    devices; the window runs from a device's first kernel start to its last
    end. Category and kernel times are per device and per step, summed over
    streams, so overlapping streams can add past busy."""
    if not kernels:
        raise ValueError("the trace holds no device kernels")
    devices = sorted({device for *_, device in kernels})
    busy = window = 0.0
    for device in devices:
        spans = sorted((start, end) for _, start, end, owner in kernels if owner == device)
        total, (low, high) = 0, spans[0]
        for start, end in spans[1:]:
            if start > high:
                total += high - low
                low, high = start, end
            else:
                high = max(high, end)
        total += high - low
        busy += total
        window += high - spans[0][0]
    scale = 1e-6 / steps / len(devices)
    by_category: dict[str, float] = {}
    by_name: dict[str, list[float]] = {}
    for name, start, end, _ in kernels:
        category = kernel_category(name)
        by_category[category] = by_category.get(category, 0.0) + (end - start)
        entry = by_name.setdefault(name, [0.0, 0])
        entry[0] += end - start
        entry[1] += 1
    return {
        "profiled_steps": steps,
        "devices": len(devices),
        "kernels_per_step": len(kernels) / steps / len(devices),
        "device_busy_ms_per_step": busy * scale,
        "device_window_ms_per_step": window * scale,
        "device_busy_percent": 100.0 * busy / window,
        "kernel_ms_by_category": {k: v * scale for k, v in
                                  sorted(by_category.items(), key=lambda item: -item[1])},
        "top_kernels": [
            {"name": name[:160], "category": kernel_category(name), "ms_per_step": total * scale,
             "calls_per_step": count / steps / len(devices)}
            for name, (total, count) in sorted(by_name.items(), key=lambda item: -item[1][0])[:25]],
    }


def chrome_trace_kernels(path: str | Path) -> list[tuple[str, int, int, int]]:
    """Kernel and memcpy/memset records from a torch.profiler chrome trace."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    kernels = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start = int(float(event["ts"]) * 1e3)
        kernels.append((event["name"], start, start + int(float(event["dur"]) * 1e3),
                        int(event.get("args", {}).get("device", 0))))
    return kernels


def xplane_kernels(directory: str | Path) -> tuple[list[tuple[str, int, int, int]], list[str]]:
    """Device kernel records from the newest JAX trace under `directory`,
    with the names of the device lines they came from.

    A GPU plane holds one line per CUDA stream; the per-op and per-module
    lines are derived from the same kernels, so only stream lines count."""
    from jax.profiler import ProfileData

    traces = sorted(Path(directory).rglob("*.xplane.pb"), key=os.path.getmtime)
    if not traces:
        raise FileNotFoundError(f"no xplane.pb under {directory}")
    kernels, lines = [], set()
    for plane in ProfileData.from_file(str(traces[-1])).planes:
        match = re.match(r"/device:GPU:(\d+)", plane.name)
        if match is None:
            continue
        for line in plane.lines:
            lines.add(f"{plane.name}:{line.name}")
            if not line.name.lower().startswith("stream"):
                continue
            device = int(match.group(1))
            kernels.extend((event.name, event.start_ns, event.end_ns, device) for event in line.events)
    return kernels, sorted(lines)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trimmed_kernels(path: str | Path) -> list[tuple[str, int, int, int]]:
    """Read back the kernel rows `trim_trace` wrote."""
    with gzip.open(path, "rt") as handle:
        return [(name, start, start + duration, device) for name, start, duration, device in json.load(handle)]


def trim_trace(kernels: Sequence[tuple[str, int, int, int]], path: str | Path) -> Path:
    """Keep a traced window as its device kernels alone, gzipped JSON rows of
    `[name, start_ns, duration_ns, device]`, so the evidence holds the
    timeline without the host events that make a raw trace large."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = min(start for _, start, _, _ in kernels)
    rows = [[name, start - origin, end - start, device] for name, start, end, device in kernels]
    with gzip.open(path, "wt") as handle:
        json.dump(rows, handle)
    return path


def git_head(path: str | Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def host() -> dict:
    return {"hostname": platform.node(), "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def window_rows(order: np.ndarray, step: int, batch: int) -> np.ndarray:
    """The window indices of global step `step`: epoch after epoch, each in
    its own recorded order, `batch` windows a step, a partial tail dropped."""
    per_epoch = order.shape[1] // batch
    epoch, index = divmod(step, per_epoch)
    return order[epoch, index * batch:(index + 1) * batch]


def steps_for(order: np.ndarray, batch: int) -> int:
    return order.shape[0] * (order.shape[1] // batch)


def write_record(path: str | Path, record: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=_plain) + "\n")
    print(f"wrote {path}")


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value)} is not JSON")


def throughput(window_seconds: float, steps: int, tokens_per_step: int,
               step_seconds: Iterable[float] | None = None) -> dict:
    """The mean rate over a timed window of `steps` steps, synchronized at
    both ends and nowhere inside, and the spread of per-step device times
    where the side records them."""
    numbers = {"timed_steps": steps, "window_seconds": window_seconds,
               "tokens_per_s": tokens_per_step * steps / window_seconds,
               "step_ms_mean": 1e3 * window_seconds / steps}
    if step_seconds is not None:
        seconds = np.asarray(list(step_seconds), np.float64)
        numbers.update({f"step_ms_p{q}": float(1e3 * np.percentile(seconds, q)) for q in (10, 50, 90)})
    return numbers
