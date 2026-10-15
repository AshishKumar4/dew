"""The torch side's run, shared by `torch_lm.py` and `megatron_lm.py`: where
a process runs, how a step's recorded windows split over data ranks and
micro-batches, the timed loop over every step, and the record's accounting
(memory peaks, throughput, MFU from Dew's count of the same step, the traced
kernels). Each backend builds its model and optimizer and runs one step."""

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from common import (
    TorchTracedWindow,
    WorkloadWork,
    chrome_trace_kernels,
    host,
    kernel_summary,
    peak_flops,
    sha256,
    steps_for,
    throughput,
    trim_trace,
    warmup_cosine,
    window_rows,
    write_record,
)


@dataclasses.dataclass
class Ranks:
    """Where this process runs: its rank among `world` and its GPU."""

    rank: int
    world: int
    device: torch.device
    distributed: bool


@dataclasses.dataclass
class Split:
    """The recorded windows and how a step splits them over data ranks and micro-batches."""

    windows: np.ndarray
    order: np.ndarray
    seq: int
    total: int
    schedule_steps: int
    share: int
    micro: int
    accumulation: int

    def rows(self, step: int, batch: int, data_rank: int) -> np.ndarray:
        """The windows data rank `data_rank` trains at step `step`."""
        return window_rows(self.order, step, batch)[data_rank * self.share:(data_rank + 1) * self.share]


def split(args: argparse.Namespace, data_ranks: int) -> Split:
    """The windows of `args.data` over `data_ranks` ranks, `args.micro_batch`
    rows a forward. The schedule decays over every epoch; `--steps` only stops
    the run early, so a short run steps exactly as the first steps of the full
    one."""
    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)
    if args.batch % data_ranks:
        raise ValueError(f"{args.batch} rows do not split over {data_ranks} data ranks")
    share = args.batch // data_ranks
    micro = args.micro_batch or share
    if share % micro:
        raise ValueError(f"a rank's {share} rows do not split into micro-batches of {micro}")
    return Split(windows, order, windows.shape[1] - 1, total, schedule_steps, share, micro, share // micro)


@dataclasses.dataclass
class Curves:
    """What every step left, the timed window and whether the tail was traced."""

    losses: list
    norms: list
    rates: list
    auxes: list
    window_seconds: float
    timed_steps: int
    step_seconds: list
    traced: bool


def trace_directory(args: argparse.Namespace) -> Path:
    return Path(args.trace_dir or Path(args.out).with_suffix("")).resolve()


def train(args: argparse.Namespace, place: Ranks, run: Split, optimizer,
          step: Callable[[int], tuple], probe: Callable[[int], float] | None = None) -> Curves:
    """Every step in the recorded order, `step(index)` returning its loss,
    gradient norm and auxiliary loss: timed from `timing_warmup` to the
    profiled tail, which is traced. `probe(index)`, every `probe_every` steps
    and once after the last, returns the seconds it took, which the window
    leaves out."""
    losses, norms, rates, auxes = [], [], [], []
    window = TorchTracedWindow(place.device, run.total, args.timing_warmup, args.profile_steps)
    probed, probe_seconds = set(), 0.0
    probing = probe is not None and args.probe_every
    torch.cuda.reset_peak_memory_stats(place.device)
    torch.cuda.synchronize(place.device)
    for index in range(run.total):
        window.before(index)
        if probing and index % args.probe_every == 0:
            spent = probe(index)
            if args.timing_warmup <= index < window.profile_from:
                probe_seconds += spent
                probed.add(index)
        lr = warmup_cosine(index, init=args.lr_init, peak=args.lr_peak, warmup=args.warmup,
                           decay_steps=run.schedule_steps, end=args.lr_end)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, norm, aux = step(index)
        losses.append(loss)
        norms.append(norm)
        rates.append(lr)
        auxes.append(aux)
    window.close(trace_directory(args) / f"rank{place.rank}.json")
    if window.profiler is not None and place.distributed:
        dist.barrier()
    if probing:
        probe(run.total)
    window.excluded, window.excluded_steps = probe_seconds, probed
    return Curves(losses, norms, rates, auxes, window.seconds, window.timed_steps, window.step_seconds,
                  window.profiler is not None)


def record(args: argparse.Namespace, place: Ranks, run: Split, curves: Curves, work: WorkloadWork,
           loss: torch.Tensor, *, framework: str, fields: dict, config: dict, versions: dict) -> dict | None:
    """Rank 0's record of the run, None on the others: `loss`, each step's
    loss as the ranks agree on it, every rank's memory peak, gathered by all
    of them, the timed throughput, and MFU from Dew's count of the same step
    (`work`), which must train the run's tokens. `fields` and `config` are
    the backend's own."""
    peak = torch.tensor([torch.cuda.max_memory_allocated(place.device),
                         torch.cuda.max_memory_reserved(place.device)],
                        device=place.device, dtype=torch.float64)
    peaks = [peak]
    if place.distributed:
        peaks = [torch.zeros_like(peak) for _ in range(place.world)]
        dist.all_gather(peaks, peak)
    if place.rank != 0:
        return None
    tokens_per_step = args.batch * run.seq
    if work.tokens_per_step != tokens_per_step:
        raise ValueError(f"{args.work} counts {work.tokens_per_step} tokens a step, this run trains "
                         f"{tokens_per_step}")
    device_name = torch.cuda.get_device_name(place.device)
    peak_rate = peak_flops(device_name)
    return {
        "framework": framework,
        **fields,
        "world": place.world,
        "config": {**vars(args), "total_steps": run.total, "schedule_steps": run.schedule_steps,
                   "seq": run.seq, "accumulation": run.accumulation, "micro_batch": run.micro,
                   "data_sha256": sha256(args.data),
                   "schedule": "optax warmup_cosine_decay_schedule (common.warmup_cosine)", **config},
        "versions": {**versions, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()},
        "device": device_name,
        "host": host(),
        "loss": loss.cpu().tolist(),
        "grad_norm": [float(norm) for norm in curves.norms],
        "lr": curves.rates,
        "throughput": throughput(curves.window_seconds, curves.timed_steps, tokens_per_step,
                                 curves.step_seconds),
        "flops_per_token": work.training_flops_per_step / work.tokens_per_step,
        "work": dataclasses.asdict(work),
        "mfu": None if peak_rate is None else work.training_flops_per_step * curves.timed_steps
            / curves.window_seconds / (peak_rate * place.world),
        "memory": {"peak_allocated_bytes": [int(p[0]) for p in peaks],
                   "peak_reserved_bytes": [int(p[1]) for p in peaks]},
    }


def attach_profile(args: argparse.Namespace, place: Ranks, result: dict) -> None:
    """Every rank's traced kernels summarised, the raw traces replaced by
    their trimmed kernel rows."""
    trace_dir = trace_directory(args)
    kernels = []
    for index in range(place.world):
        raw = trace_dir / f"rank{index}.json"
        kernels.extend((name, start, end, index) for name, start, end, _ in chrome_trace_kernels(raw))
        raw.unlink()
    try:
        result["profile"] = kernel_summary(kernels, args.profile_steps)
        result["profile"]["kernels"] = str(trim_trace(kernels, trace_dir / "kernels.json.gz"))
    except ValueError as error:
        result["profile"] = {"error": str(error)}


def finish(args: argparse.Namespace, place: Ranks, result: dict | None, traced: bool) -> None:
    """Rank 0 writes the record, then again with its traced kernels, and
    prints its summary."""
    if result is None:
        return
    if traced:
        write_record(args.out, result)
        attach_profile(args, place, result)
    write_record(args.out, result)
    print(json.dumps({"first_loss": result["loss"][0], "last_loss": result["loss"][-1],
                      "tokens_per_s": result["throughput"]["tokens_per_s"],
                      "step_ms": result["throughput"]["step_ms_mean"], "mfu": result["mfu"],
                      "peak_gib": max(result["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))
