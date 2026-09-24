#!/usr/bin/env python3
"""The torch + transformers side of a language-model reference run.

Fine-tunes a Hugging Face causal LM on the fixed windows `token_windows.py`
wrote, in their recorded order, and writes every step's loss, gradient norm
and learning rate, the timed throughput, the allocator peak and a
torch.profiler summary of the last `--profile-steps` steps.

The loop is the plain one a transformers user writes: the model's own
forward, the logits upcast to fp32 and cross entropy over every shifted
target (`ForCausalLMLoss` with the shift done here, so a `seq + 1` row gives
`seq` targets as Dew's LM objective takes them), `clip_grad_norm_`, and
fused `torch.optim.AdamW` on the schedule `common.warmup_cosine`, which is
optax's `warmup_cosine_decay_schedule`. `--compile` compiles what
torchtitan's "model" and "loss" components do: every decoder layer before
DDP or fully_shard wraps it, then the final norm and the head, and the loss,
whose compiled upcast and cross entropy never hold the logits in fp32.

Precision policies:
  autocast   fp32 parameters, forward and loss under torch.autocast(bf16);
             the residual stream and the norms stay fp32 (a Linear's bf16
             output meets an fp32 residual), and each rounds once, at the
             next matmul's input
  autocast-bf16-residual
             autocast with Dew's rounding points: the residual stream in bf16
             from the embedding on, so every residual add rounds and each
             RMSNorm rounds its normalized activations to bf16 before the fp32
             weight scales them (transformers' `.to(input_dtype)`), and every
             norm returns bf16; the rotary table stays fp32, as Dew's does
             (`bf16_residual`)
  fp32       fp32 everywhere, TF32 off: the run the bound is measured from
  fsdp-bf16  FSDP2 MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)

    python tools/reference_runs/torch_lm.py --model <hf dir> --data windows.npz \\
        --batch 4 --precision autocast --out torch-autocast.json
    torchrun --nproc-per-node 4 tools/reference_runs/torch_lm.py --parallel fsdp2 \\
        --precision fsdp-bf16 --batch 16 ...
"""

import argparse
import contextlib
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from common import (
    chrome_trace_kernels,
    host,
    kernel_summary,
    peak_flops,
    sha256,
    steps_for,
    throughput,
    train_flops_per_token,
    trim_trace,
    warmup_cosine,
    window_rows,
    write_record,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, required=True, help="global rows per step")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="rows per forward on one rank; unset is the rank's whole share")
    parser.add_argument("--precision", choices=("autocast", "autocast-bf16-residual", "fp32", "fsdp-bf16"),
                        required=True)
    parser.add_argument("--parallel", choices=("single", "ddp", "fsdp2"), default="single")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--experts", default=None,
                        help="MoE: transformers' experts_implementation. Its default on sm80, "
                             "grouped_mm, is outside autocast and runs the experts at the fp32 "
                             "weights' dtype; 'eager' loops F.linear, which autocast runs in bf16")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile each decoder layer, the final norm and head, and the loss")
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
    parser.add_argument("--router-aux", type=float, default=None,
                        help="MoE: weight of the per-layer Switch balance loss (moe_aux)")
    parser.add_argument("--probe-every", type=int, default=None,
                        help="MoE: router loads and entropy on a fixed batch every this many steps")
    parser.add_argument("--probe-rows", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default=None)
    return parser.parse_args()


def moe_aux(router_logits, active: int, distributed: bool) -> torch.Tensor:
    """Switch Transformer's balance loss per routed layer, summed over the
    layers: E * sum_i f_i P_i, f_i the share of top-k slots expert i took and
    P_i its mean router probability, both over the global batch.

    transformers' `load_balancing_loss_func` pools every layer's tokens into
    one f and one P before the product, which is not a sum of per-layer
    losses; this is the per-layer form Dew's `global_router_loss` takes.
    Across ranks the probabilities are summed with an autograd all-reduce,
    so every rank holds the global loss, and DDP's gradient average turns
    each rank's copy of its gradient into the global loss's gradient."""
    from torch.distributed.nn.functional import all_reduce

    total = 0.0
    for logits in router_logits:
        probs = torch.softmax(logits.float(), dim=-1)
        experts = probs.shape[-1]
        chosen = torch.topk(probs, active, dim=-1).indices
        counts = F.one_hot(chosen, experts).sum(dim=(0, 1)).float()
        summed = probs.sum(dim=0)
        tokens = torch.tensor(float(probs.shape[0]), device=probs.device)
        if distributed:
            dist.all_reduce(counts)
            dist.all_reduce(tokens)
            summed = all_reduce(summed)
        total = total + experts * torch.sum(counts / (tokens * active) * summed / tokens)
    return total


def bf16_residual(model) -> None:
    """Carry the residual stream in bf16, Dew's bf16 compute policy (the JAX
    convention, as MaxText and Megatron's fp32_residual_connection=False).

    The input embedding's output is cast to bf16, and the residual stream
    follows from the dtypes: a bf16 residual plus a bf16 sublayer output is a
    bf16 add. An RMSNorm's `.to(input_dtype)` then rounds its normalized
    activations to bf16 before the fp32 weight scales them, and the product
    is rounded to bf16 too, as Dew's norm returns its compute dtype. The
    rotary embedding reads its input only for the dtype, so it is handed an
    fp32 tensor and returns fp32 cos/sin, which Dew's rotary table also is."""
    def to_bf16(module, inputs, output):
        return output.to(torch.bfloat16)

    model.get_input_embeddings().register_forward_hook(to_bf16)
    # Every norm returns bf16, as Dew's does. The decoder norms' products
    # would be rounded by the next Linear anyway; the per-head q/k norms'
    # feed the rotary embedding, which autocast leaves in fp32.
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.register_forward_hook(to_bf16)
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise ValueError(f"{type(model).__name__} has no model.rotary_emb to keep in fp32")
    rotary.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))


def autocast(precision: str):
    """The forward's context: bf16 autocast for the autocast policies."""
    return (torch.autocast("cuda", dtype=torch.bfloat16) if precision.startswith("autocast")
            else contextlib.nullcontext())


def router_probe(model, tokens: torch.Tensor, active: int, precision: str) -> tuple[list, list]:
    """Each routed layer's share of top-k slots per expert, and its mean
    router entropy, over one fixed batch at the current weights."""
    with torch.no_grad(), autocast(precision):
        outputs = model(input_ids=tokens[:, :-1], output_router_logits=True)
    loads, entropies = [], []
    for logits in outputs.router_logits:
        probs = torch.softmax(logits.float(), dim=-1)
        chosen = torch.topk(probs, active, dim=-1).indices
        counts = F.one_hot(chosen, probs.shape[-1]).sum(dim=(0, 1)).float()
        loads.append((counts / chosen.numel()).cpu().tolist())
        entropies.append(float(-(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean()))
    return loads, entropies


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy over every target, the logits upcast to fp32 as
    transformers' ForCausalLMLoss does."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


@dataclasses.dataclass
class Ranks:
    """Where this process runs: its rank among `world` and its GPU."""

    rank: int
    world: int
    device: torch.device
    distributed: bool


def ranks(args: argparse.Namespace) -> Ranks:
    """The process group, this process's GPU, and the precision flags."""
    distributed = args.parallel != "single"
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.precision == "fsdp-bf16" and args.parallel != "fsdp2":
        raise ValueError("fsdp-bf16 is FSDP2's mixed-precision policy; run it with --parallel fsdp2")
    return Ranks(rank, world, torch.device("cuda", local), distributed)


@dataclasses.dataclass
class Run:
    """The model, wrapped for its parallelism, its optimizer, the recorded
    windows and how a step splits them over ranks and micro-batches."""

    model: Any
    network: Any
    optimizer: torch.optim.Optimizer
    windows: np.ndarray
    order: np.ndarray
    seq: int
    total: int
    schedule_steps: int
    share: int
    micro: int
    accumulation: int
    active: int  # experts a token takes; 0 without the balance loss
    loss: Any  # token_loss, compiled under --compile


def trunk(model) -> torch.nn.Module:
    """The stack under a transformers causal LM's head: a decoder's `model`,
    a Mamba's `backbone`."""
    stack = getattr(model, "model", None)
    if stack is None:
        stack = getattr(model, "backbone", None)
    if stack is None or not hasattr(stack, "layers"):
        raise ValueError(f"{type(model).__name__} has no model.layers or backbone.layers")
    return stack


def final_norm(stack) -> torch.nn.Module:
    """The norm a decoder's `model.norm` or a Mamba's `backbone.norm_f` is."""
    return stack.norm if hasattr(stack, "norm") else stack.norm_f


def build(args: argparse.Namespace, place: Ranks) -> Run:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model)
    extra = {} if config.model_type == "mamba2" else {"attn_implementation": args.attention}
    if args.experts is not None:
        extra["experts_implementation"] = args.experts
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **extra).to(place.device)
    model.config.use_cache = False
    model.train()
    if args.precision == "autocast-bf16-residual":
        bf16_residual(model)
    if args.compile:
        for layer in trunk(model).layers:
            layer.compile()
        final_norm(trunk(model)).compile()
        model.lm_head.compile()
    routed = args.router_aux is not None
    if routed:
        model.config.output_router_logits = False

    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    # The schedule decays over every epoch; --steps only stops the run early,
    # so a short run steps exactly as the first steps of the full one.
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)
    if args.batch % place.world:
        raise ValueError(f"{args.batch} rows do not split over {place.world} ranks")
    share = args.batch // place.world
    micro = args.micro_batch or share
    if share % micro:
        raise ValueError(f"a rank's {share} rows do not split into micro-batches of {micro}")

    network = model
    if args.parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel
        network = DistributedDataParallel(model, device_ids=[place.device.index], gradient_as_bucket_view=True)
    elif args.parallel == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        policy = (MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
                  if args.precision == "fsdp-bf16" else MixedPrecisionPolicy())
        for layer in trunk(model).layers:
            fully_shard(layer, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(args.b1, args.b2),
                                  eps=args.eps, weight_decay=args.weight_decay, fused=True)
    return Run(model, network, optimizer, windows, order, windows.shape[1] - 1, total, schedule_steps,
               share, micro, share // micro, model.config.num_experts_per_tok if routed else 0,
               torch.compile(token_loss) if args.compile else token_loss)


def train_step(args: argparse.Namespace, place: Ranks, run: Run, step: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One optimizer step on this rank's rows of step `step`, over its
    micro-batches: the mean cross entropy, the pre-clip gradient norm and the
    weighted balance loss."""
    rows = window_rows(run.order, step, args.batch)[place.rank * run.share:(place.rank + 1) * run.share]
    tokens = torch.from_numpy(run.windows[rows].astype(np.int64)).pin_memory().to(place.device, non_blocking=True)
    step_loss = torch.zeros((), device=place.device)
    step_aux = torch.zeros((), device=place.device)
    routed = run.active > 0
    for index in range(run.accumulation):
        chunk = tokens[index * run.micro:(index + 1) * run.micro]
        last = index == run.accumulation - 1
        sync = (run.network.no_sync() if args.parallel == "ddp" and not last
                else contextlib.nullcontext())
        if args.parallel == "fsdp2":
            run.model.set_requires_gradient_sync(last)
        with sync:
            with autocast(args.precision):
                outputs = run.network(input_ids=chunk[:, :-1], output_router_logits=routed) if routed \
                    else run.network(input_ids=chunk[:, :-1])
                loss = run.loss(outputs.logits, chunk[:, 1:])
                objective = loss
                if routed:
                    aux = args.router_aux * moe_aux(outputs.router_logits, run.active, place.distributed)
                    objective = loss + aux
                    step_aux += aux.detach() / run.accumulation
            (objective / run.accumulation).backward()
        step_loss += loss.detach() / run.accumulation
    norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), args.clip)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    run.optimizer.step()
    run.optimizer.zero_grad(set_to_none=True)
    return step_loss, norm.detach(), step_aux


@dataclasses.dataclass
class Curves:
    """What every step left, the timed window and whether the tail was traced."""

    losses: list
    norms: list
    rates: list
    auxes: list
    probes: list
    window_seconds: float
    timed_steps: int
    step_seconds: list
    traced: bool


def trace_directory(args: argparse.Namespace) -> Path:
    return Path(args.trace_dir or Path(args.out).with_suffix("")).resolve()


def train(args: argparse.Namespace, place: Ranks, run: Run) -> Curves:
    """Every step in the recorded order: timed from `timing_warmup` to the
    profiled tail, which is traced, with the router probes' time taken out."""
    losses, norms, rates, auxes = [], [], [], []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(run.total + 1)]
    profile_from = run.total - args.profile_steps if args.profile_steps else run.total
    profiler = None
    probe_rows = torch.from_numpy(run.windows[:args.probe_rows].astype(np.int64)).to(place.device)
    probes, probed_steps, probe_seconds = [], set(), 0.0

    def probe(step: int) -> float:
        torch.cuda.synchronize(place.device)
        start = time.perf_counter()
        loads, entropies = router_probe(run.model, probe_rows, run.active, args.precision)
        probes.append({"step": step, "load": loads, "entropy": entropies})
        torch.cuda.synchronize(place.device)
        return time.perf_counter() - start

    torch.cuda.reset_peak_memory_stats(place.device)
    torch.cuda.synchronize(place.device)
    window_start = window_end = None
    for step in range(run.total):
        if step == profile_from:
            torch.cuda.synchronize(place.device)
            window_end = time.perf_counter()
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        if step == args.timing_warmup:
            torch.cuda.synchronize(place.device)
            window_start = time.perf_counter()
        if args.probe_every and step % args.probe_every == 0:
            spent = probe(step)
            if args.timing_warmup <= step < profile_from:
                probe_seconds += spent
                probed_steps.add(step - 1)
        events[step].record()
        lr = warmup_cosine(step, init=args.lr_init, peak=args.lr_peak, warmup=args.warmup,
                           decay_steps=run.schedule_steps, end=args.lr_end)
        for group in run.optimizer.param_groups:
            group["lr"] = lr
        loss, norm, aux = train_step(args, place, run, step)
        losses.append(loss)
        norms.append(norm)
        rates.append(lr)
        auxes.append(aux)
    events[run.total].record()
    torch.cuda.synchronize(place.device)
    end = time.perf_counter()
    if profiler is None:
        window_end = end
    else:
        profiler.__exit__(None, None, None)
        trace_dir = trace_directory(args)
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_dir / f"rank{place.rank}.json"))
        if place.distributed:
            dist.barrier()
    if args.probe_every:
        probe(run.total)
    step_seconds = [events[i].elapsed_time(events[i + 1]) / 1e3
                    for i in range(args.timing_warmup, profile_from) if i not in probed_steps]
    return Curves(losses, norms, rates, auxes, probes, window_end - window_start - probe_seconds,
                  profile_from - args.timing_warmup, step_seconds, profiler is not None)


def record(args: argparse.Namespace, place: Ranks, run: Run, curves: Curves) -> dict | None:
    """The ranks' mean losses and every rank's memory peak, gathered by all
    of them; rank 0's record of the run, None on the others."""
    loss_vector = torch.stack(curves.losses)
    if place.distributed:
        dist.all_reduce(loss_vector, op=dist.ReduceOp.AVG)
    peak = torch.tensor([torch.cuda.max_memory_allocated(place.device), torch.cuda.max_memory_reserved(place.device)],
                        device=place.device, dtype=torch.float64)
    peaks = [peak]
    if place.distributed:
        peaks = [torch.zeros_like(peak) for _ in range(place.world)]
        dist.all_gather(peaks, peak)
    if place.rank != 0:
        return None

    timed = throughput(curves.window_seconds, curves.timed_steps, args.batch * run.seq, curves.step_seconds)
    hf_config = json.loads(Path(args.model, "config.json").read_text())
    flops_token = train_flops_per_token(hf_config, run.seq)
    device_name = torch.cuda.get_device_name(place.device)
    peak_rate = peak_flops(device_name)
    routed = run.active > 0
    return {
        "framework": "torch",
        "precision": args.precision,
        "parallel": args.parallel,
        "world": place.world,
        "config": {**vars(args), "total_steps": run.total, "schedule_steps": run.schedule_steps, "seq": run.seq,
                   "accumulation": run.accumulation, "micro_batch": run.micro,
                   "data_sha256": sha256(args.data), "model_type": hf_config["model_type"],
                   "optimizer": "torch.optim.AdamW(fused=True)",
                   "schedule": "optax warmup_cosine_decay_schedule (common.warmup_cosine)",
                   "attention_implementation": run.model.config._attn_implementation,
                   "experts_implementation": getattr(run.model.config, "_experts_implementation", None)},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "device": device_name,
        "host": host(),
        "loss": loss_vector.cpu().tolist(),
        "grad_norm": torch.stack(curves.norms).cpu().tolist(),
        "lr": curves.rates,
        **({"aux_loss": torch.stack(curves.auxes).cpu().tolist(), "router_probe": curves.probes}
           if routed else {}),
        "throughput": timed,
        "flops_per_token": flops_token,
        "mfu": None if peak_rate is None else flops_token * timed["tokens_per_s"] / (peak_rate * place.world),
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


def main() -> None:
    args = arguments()
    place = ranks(args)
    run = build(args, place)
    curves = train(args, place, run)
    result = record(args, place, run, curves)
    if result is not None:
        if curves.traced:
            write_record(args.out, result)
            attach_profile(args, place, result)
        write_record(args.out, result)
        print(json.dumps({"first_loss": result["loss"][0], "last_loss": result["loss"][-1],
                          "tokens_per_s": result["throughput"]["tokens_per_s"], "mfu": result["mfu"],
                          "peak_gib": max(result["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))
    if place.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
