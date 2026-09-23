"""What both sides of a reference run share: the record, the schedule and
the FLOP count.

The torch generators run in a venv without Dew, the Dew generators in one
without torch, so this module imports numpy and the standard library only,
and `tools/trace_window.py`, which names and splits a profile's kernels the
way `tools/benchmark_step.py` does. Every number the comparison reads is
produced by one definition, so a difference between two records is a
difference between the runs.
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
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_window import kernel_category, window_split

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


def dit_train_flops_per_image(model: Mapping, height: int, width: int, channels: int = 3) -> float:
    """Model FLOPs of one trained image through the SimpleDiT: 6 per
    multiply-add of every per-token matmul (patch in and out, q/k/v/o, the
    MLP) and of the attention scores and values over the image's patches.
    The per-image conditioning path (time embedding, adaLN modulation) is
    one row per image and is left out."""
    patch, width_features = model["patch_size"], model["emb_features"]
    tokens = (height // patch) * (width // patch)
    per_token = model["num_layers"] * (
        4 * width_features ** 2 + 2 * model["mlp_ratio"] * width_features ** 2
        + 2 * tokens * width_features)
    per_token += 2 * patch * patch * channels * width_features
    return 6.0 * per_token * tokens


def kernel_summary(kernels: Sequence[tuple[str, int, int, int]], steps: int) -> dict:
    """Per-step device time from `(name, start_ns, end_ns, device)` kernel
    records of `steps` traced steps.

    Each device's window is split by `trace_window.window_split`, the split
    `tools/benchmark_step.py` reports, and the figures are averaged over the
    devices: busy is the union of a device's kernels, the window runs from its
    first kernel's start to its last one's end, and the window is compute +
    exposed communication + idle ended by an input copy + idle ended by
    anything else. Category and kernel times are per device and per step,
    summed over streams, so overlapping streams can add past busy."""
    if not kernels:
        raise ValueError("the trace holds no device kernels")
    devices = sorted({device for *_, device in kernels})
    figures: dict[str, float] = {}
    for device in devices:
        for key, value in window_split([(name, start, end) for name, start, end, owner in kernels
                                        if owner == device]).items():
            figures[key] = figures.get(key, 0.0) + value
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
        "device_busy_ms_per_step": figures["busy"] * scale,
        "device_window_ms_per_step": figures["window"] * scale,
        "device_busy_percent": 100.0 * figures["busy"] / figures["window"],
        "compute_busy_percent": 100.0 * figures["compute"] / figures["window"],
        **{f"{key}_ms_per_step": figures[key] * scale
           for key in ("compute", "communication", "exposed_communication", "idle_input", "idle_host")},
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


def git_head(path: str | Path) -> str:
    """The commit a record was measured at; a record without one cannot be
    traced, so a checkout git cannot read fails the run."""
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


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
