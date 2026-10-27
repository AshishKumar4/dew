"""What both sides of a reference run share: the records and the schedule.

The torch generators run in a venv without Dew, the Dew generators in one
without torch, so this module imports numpy and the standard library only,
`tools/reference_source.py`, which loads Dew's host-only modules here, and
`tools/trace_window.py`, which names and splits a profile's kernels the
way `tools/benchmark_step.py` does. Every number the comparison reads is
produced by one definition, so a difference between two records is a
difference between the runs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_source import loaded
from trace_window import kernel_category, kernel_totals, library

SOURCE = Path(__file__).resolve().parents[2] / "src" / "dew"
EVIDENCE = Path(os.environ.get(
    "REFERENCE_RUNS_DIR", Path.home() / ".cache/dew/verification-evidence/reference-runs"))


# Dew's host-only records and target table, read without its backend import policy.
gpu_target = loaded("dew_gpu_targets", SOURCE / "nn/kernels/generation.py").gpu_target
WorkloadWork = loaded("dew_workload", SOURCE / "telemetry/workload.py").WorkloadWork


def peak_flops(device_kind: str) -> float | None:
    return gpu_target(device_kind).bf16_peak


def register_plain_causal_sdpa() -> str:
    """HF's SDPA for complete, unpadded, uncached causal training rows.

    HF 5.3's default mask builder materializes a dense mask during Dynamo
    tracing, even with attention_mask=None. Flash SDPA rejects that mask.
    Register the row contract through HF's public attention interfaces so
    every backbone can use the same no-mask path, with explicit guards.
    """
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface

    def mask(*, attention_mask=None, cache_position, kv_length, kv_offset, **kwargs):
        if (attention_mask is not None or kv_offset != 0 or cache_position.shape[0] != kv_length
                or not kwargs.get("allow_is_causal_skip", False)):
            raise ValueError("plain causal SDPA requires complete unpadded rows and no cache")

    def attention(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None or query.shape[-2] != key.shape[-2] or query.shape[-2] <= 1:
            raise ValueError("plain causal SDPA requires no mask and equal training query/key lengths > 1")
        kwargs["is_causal"] = True
        return sdpa_attention_forward(module, query, key, value, None, **kwargs)

    name = "plain_causal_sdpa"
    AttentionInterface.register(name, attention)
    AttentionMaskInterface.register(name, mask)
    return name


def run_arguments() -> argparse.ArgumentParser:
    """The arguments both sides of an LM run take, with their defaults: what
    they train on, the schedule and solver, the router probes and the timed
    window. Each side adds its own precision, attention and parallelism."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True,
                        help="the workload's count, which Dew's side writes and the reference reads")
    parser.add_argument("--batch", type=int, required=True, help="global rows per step")
    parser.add_argument("--documents", default=None,
                        help="LOW:HIGH packs every row with documents of these lengths (`documents`)")
    parser.add_argument("--steps", type=int, default=None, help="stop early; unset runs every epoch")
    parser.add_argument("--lr-peak", type=float, default=2e-5)
    parser.add_argument("--lr-init", type=float, default=2e-6)
    parser.add_argument("--lr-end", type=float, default=2e-6)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--probe-every", type=int, default=None,
                        help="MoE: router loads and entropy on a fixed batch every this many steps")
    parser.add_argument("--probe-rows", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default=None)
    return parser


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


def kernel_summary(kernels: Sequence[tuple[str, int, int, int]], steps: int) -> dict:
    """Per-step device time from `(name, start_ns, end_ns, device)` kernel
    records of `steps` traced steps.

    Each device's window is split by `dew.telemetry.timeline.window_split`, the split
    `tools/benchmark_step.py` reports, and the figures are averaged over the
    devices: busy is the union of a device's kernels, the window runs from its
    first kernel's start to its last one's end, and the window is compute +
    exposed communication + idle ended by an input copy + idle ended by
    anything else. Category and kernel times are per device and per step,
    summed over streams, so overlapping streams can add past busy."""
    if not kernels:
        raise ValueError("the trace holds no device kernels")
    devices = sorted({device for *_, device in kernels})
    figures = library("telemetry/timeline.py").window_totals(
        [[(name, start, end) for name, start, end, owner in kernels if owner == device]
         for device in devices])
    scale = 1e-6 / steps / len(devices)
    records = [(name, start, end) for name, start, end, _ in kernels]
    by_category, _ = kernel_totals(records, kernel_category)
    by_name, counts = kernel_totals(records)
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
             "calls_per_step": counts[name] / steps / len(devices)}
            for name, total in sorted(by_name.items(), key=lambda item: -item[1])[:25]],
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


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trimmed_kernels(path: str | Path) -> list[tuple[str, int, int, int]]:
    """Read back the kernel rows `trim_trace` wrote."""
    with gzip.open(path, "rt") as handle:
        return [(name, start, start + duration, device)
                for name, start, duration, device in json.load(handle)]


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


def source_version(path: str | Path) -> str:
    """Where a reference package's source came from: its checkout's commit,
    or for a copied tree the sha256 over its Python files' paths and bytes."""
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0:
        return head.stdout.strip()
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*.py")):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return f"sha256:{digest.hexdigest()}"


class TracedWindow:
    """The timed window and the traced tail of a JAX run of `total` steps.

    The steps from `warmup` up to the last `profiled` ones are timed,
    synchronized at both ends and nowhere inside; the last `profiled` are
    traced by jax.profiler into `trace_dir`, with its Python tracer off as
    in a Dew capture (`dew.telemetry.profile.capture_options`, which this
    module can't import): the tracer records every Python and C call and
    slows a host-bound run's steps. Call `before(step, pending)` ahead of
    each step with the newest output still in flight, add any in-window
    seconds that are not training to `excluded`, and end with
    `close(pending)`, which returns the window's training seconds."""

    def __init__(self, total: int, warmup: int, profiled: int, trace_dir: str | Path):
        self.warmup, self.profiled, self.trace_dir = warmup, profiled, Path(trace_dir)
        self.profile_from = total - profiled if profiled else total
        self.timed_steps = self.profile_from - warmup
        self.start: float | None = None
        self.end: float | None = None
        self.excluded = 0.0

    def times(self, step: int) -> bool:
        return self.warmup <= step < self.profile_from

    def before(self, step: int, pending) -> None:
        import jax

        if step not in (self.warmup, self.profile_from):
            return
        jax.block_until_ready(pending)
        now = time.perf_counter()
        if step == self.warmup:
            self.start = now
        if step == self.profile_from and self.profiled:
            self.end = now
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = 0
            jax.profiler.start_trace(str(self.trace_dir), profiler_options=options)

    def close(self, pending) -> float:
        import jax

        jax.block_until_ready(pending)
        end = time.perf_counter()
        if self.end is None:
            self.end = end
        else:
            jax.profiler.stop_trace()
        if self.start is None:
            raise ValueError(f"the timed window never opened: warm-up {self.warmup} is past the run")
        return self.end - self.start - self.excluded


class TorchTracedWindow:
    """The torch loop's synchronized timing window followed by a profiled tail.

    CUDA events measure individual steady steps. The wall clock spans the
    same steps, synchronized only at its boundaries. The last timing event
    is recorded before profiler startup, so startup cannot inflate its last
    sample. CPU timing exists for tiny remote runner checks, never for MFU.
    reset_peak opts into a steady-window memory peak; the default preserves
    global whole-run peak statistics for torch_lm's existing record.
    """

    def __init__(self, device, total: int, warmup: int, profiled: int, *, reset_peak: bool = False):
        import torch

        self.device = device
        self.reset_peak = reset_peak
        self.total, self.warmup, self.profile_from = total, warmup, total - profiled
        if not 0 <= warmup < self.profile_from <= total:
            raise ValueError("need at least one timed step before the profiled tail")
        self.timed_steps = self.profile_from - warmup
        self.events = ([torch.cuda.Event(enable_timing=True) for _ in range(self.timed_steps + 1)]
                       if device.type == "cuda" else [])
        self.cpu_times: list[float] = []
        self.start = self.end = None
        self.profiler = None
        self.excluded = 0.0
        self.excluded_steps: set[int] = set()
        self.peak_allocated = self.peak_reserved = None
        self.whole_peak_allocated = None

    def synchronize(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _boundary(self, step: int) -> None:
        if self.warmup <= step <= self.profile_from:
            if self.events:
                self.events[step - self.warmup].record()
            else:
                self.cpu_times.append(time.perf_counter())

    def _end(self) -> None:
        import torch

        self.synchronize()
        self.end = time.perf_counter()
        if self.device.type == "cuda":
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
            self.peak_reserved = torch.cuda.max_memory_reserved(self.device)
            self.whole_peak_allocated = max(self.whole_peak_allocated or 0, self.peak_allocated)

    def before(self, step: int) -> None:
        import torch

        if step == self.warmup:
            self.synchronize()
            if self.device.type == "cuda" and self.reset_peak:
                self.whole_peak_allocated = torch.cuda.max_memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self.start = time.perf_counter()
        self._boundary(step)
        if step == self.profile_from:
            self._end()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=True)
            self.profiler.__enter__()

    def close(self, trace_path: str | Path) -> None:
        if self.end is None:
            self._boundary(self.total)
            self._end()
        self.synchronize()
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.profiler.export_chrome_trace(str(path))

    @property
    def seconds(self) -> float:
        if self.start is None or self.end is None:
            raise ValueError("the timing window has not closed")
        return self.end - self.start - self.excluded

    @property
    def step_seconds(self) -> list[float]:
        return [(self.events[i].elapsed_time(self.events[i + 1]) / 1e3 if self.events
                 else self.cpu_times[i + 1] - self.cpu_times[i])
                for i in range(self.timed_steps) if i + self.warmup not in self.excluded_steps]


def attach_xplane_profile(record: dict, trace_dir: str | Path, steps: int,
                          rows: str | Path | None = None) -> None:
    """The traced tail's kernel summary as `record["profile"]`, beside its
    trimmed kernel rows (`rows`, by default `trace_dir/kernels.json.gz`),
    with the raw xplane files deleted. A trace the summary cannot read is
    recorded as such: the run's curves stand without it."""
    trace_dir = Path(trace_dir)
    try:
        planes, lines = library("telemetry/timeline.py").device_events(trace_dir)
        kernels = [(event.name, event.start_ns, event.end_ns, int(plane.rsplit(":", 1)[1]))
                   for plane, events in planes.items() for event in events]
        record["profile"] = kernel_summary(kernels, steps)
        record["profile"]["kernels"] = str(trim_trace(kernels, rows or trace_dir / "kernels.json.gz"))
        record["profile"]["device_lines"] = lines
    except (ValueError, FileNotFoundError) as error:
        record["profile"] = {"error": str(error)}
    for raw in trace_dir.rglob("*.xplane.pb"):
        raw.unlink()


def host() -> dict:
    return {"hostname": platform.node(), "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def documents(shape: tuple[int, int], lengths: str, seed: int = 0) -> np.ndarray:
    """Packed rows' segment ids, `shape` `[windows, seq + 1]`, numbered from 1:
    one row of documents whose lengths a PCG64 generator seeded with `seed`
    draws uniformly from `lengths`' LOW:HIGH, both ends included, the last
    cut at the row's end, and every window packed the same. So every step
    trains the same work, and Dew's count of one batch is each step's. Each
    side of a run reads the same layout from the same string."""
    low, high = (int(part) for part in lengths.split(":"))
    if not 1 <= low <= high:
        raise ValueError(f"documents are LOW:HIGH lengths, 1 <= LOW <= HIGH, got {lengths}")
    rng = np.random.Generator(np.random.PCG64(seed))
    row = np.empty(shape[1], np.int32)
    filled, segment = 0, 1
    while filled < row.size:
        length = int(rng.integers(low, high, endpoint=True))
        row[filled:filled + length] = segment
        filled, segment = filled + length, segment + 1
    return np.broadcast_to(row, shape)


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
