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
optax's `warmup_cosine_decay_schedule`.

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
import json
import os
import time
from pathlib import Path

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


def router_probe(model, tokens: torch.Tensor, active: int, autocast) -> tuple[list, list]:
    """Each routed layer's share of top-k slots per expert, and its mean
    router entropy, over one fixed batch at the current weights."""
    with torch.no_grad(), autocast():
        outputs = model(input_ids=tokens[:, :-1], output_router_logits=True)
    loads, entropies = [], []
    for logits in outputs.router_logits:
        probs = torch.softmax(logits.float(), dim=-1)
        chosen = torch.topk(probs, active, dim=-1).indices
        counts = F.one_hot(chosen, probs.shape[-1]).sum(dim=(0, 1)).float()
        loads.append((counts / chosen.numel()).cpu().tolist())
        entropies.append(float(-(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean()))
    return loads, entropies


def main() -> None:
    args = arguments()
    distributed = args.parallel != "single"
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.precision == "fsdp-bf16" and args.parallel != "fsdp2":
        raise ValueError("fsdp-bf16 is FSDP2's mixed-precision policy; run it with --parallel fsdp2")

    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(args.model)
    extra = {} if config.model_type == "mamba2" else {"attn_implementation": args.attention}
    if args.experts is not None:
        extra["experts_implementation"] = args.experts
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **extra).to(device)
    model.config.use_cache = False
    model.train()
    if args.precision == "autocast-bf16-residual":
        bf16_residual(model)
    routed = args.router_aux is not None
    if routed:
        model.config.output_router_logits = False

    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    seq = windows.shape[1] - 1
    # The schedule decays over every epoch; --steps only stops the run early,
    # so a short run steps exactly as the first steps of the full one.
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)
    if args.batch % world:
        raise ValueError(f"{args.batch} rows do not split over {world} ranks")
    share = args.batch // world
    micro = args.micro_batch or share
    if share % micro:
        raise ValueError(f"a rank's {share} rows do not split into micro-batches of {micro}")
    accumulation = share // micro

    network = model
    if args.parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel
        network = DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True)
    elif args.parallel == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        policy = (MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
                  if args.precision == "fsdp-bf16" else MixedPrecisionPolicy())
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(args.b1, args.b2),
                                  eps=args.eps, weight_decay=args.weight_decay, fused=True)

    def autocast():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision.startswith("autocast")
                else contextlib.nullcontext())

    def rate(step: int) -> float:
        return warmup_cosine(step, init=args.lr_init, peak=args.lr_peak, warmup=args.warmup,
                             decay_steps=schedule_steps, end=args.lr_end)

    losses, norms, rates, auxes = [], [], [], []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(total + 1)]
    profile_from = total - args.profile_steps if args.profile_steps else total
    trace_dir = Path(args.trace_dir or Path(args.out).with_suffix("")).resolve()
    profiler = None
    active = model.config.num_experts_per_tok if routed else 0
    probe_rows = torch.from_numpy(windows[:args.probe_rows].astype(np.int64)).to(device)
    probes, probed_steps, probe_seconds = [], set(), 0.0

    def probe(step: int) -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loads, entropies = router_probe(model, probe_rows, active, autocast)
        probes.append({"step": step, "load": loads, "entropy": entropies})
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    for step in range(total):
        if step == profile_from:
            torch.cuda.synchronize(device)
            window_end = time.perf_counter()
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        if step == args.timing_warmup:
            torch.cuda.synchronize(device)
            window_start = time.perf_counter()
        if args.probe_every and step % args.probe_every == 0:
            spent = probe(step)
            if args.timing_warmup <= step < profile_from:
                probe_seconds += spent
                probed_steps.add(step - 1)
        events[step].record()
        lr = rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = window_rows(order, step, args.batch)[rank * share:(rank + 1) * share]
        tokens = torch.from_numpy(windows[rows].astype(np.int64)).pin_memory().to(device, non_blocking=True)
        step_loss = torch.zeros((), device=device)
        step_aux = torch.zeros((), device=device)
        for index in range(accumulation):
            chunk = tokens[index * micro:(index + 1) * micro]
            last = index == accumulation - 1
            sync = (network.no_sync() if args.parallel == "ddp" and not last
                    else contextlib.nullcontext())
            if args.parallel == "fsdp2":
                model.set_requires_gradient_sync(last)
            with sync:
                with autocast():
                    outputs = network(input_ids=chunk[:, :-1], output_router_logits=routed) if routed \
                        else network(input_ids=chunk[:, :-1])
                    logits = outputs.logits.float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1))
                    objective = loss
                    if routed:
                        aux = args.router_aux * moe_aux(outputs.router_logits, active, distributed)
                        objective = loss + aux
                        step_aux += aux.detach() / accumulation
                (objective / accumulation).backward()
            step_loss += loss.detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        norms.append(norm.detach())
        rates.append(lr)
        auxes.append(step_aux)
    events[total].record()
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    if profiler is None:
        window_end = end
    else:
        profiler.__exit__(None, None, None)
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_dir / f"rank{rank}.json"))
        if distributed:
            dist.barrier()
    if args.probe_every:
        probe(total)

    loss_vector = torch.stack(losses)
    if distributed:
        dist.all_reduce(loss_vector, op=dist.ReduceOp.AVG)
    peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                        device=device, dtype=torch.float64)
    peaks = [peak]
    if distributed:
        peaks = [torch.zeros_like(peak) for _ in range(world)]
        dist.all_gather(peaks, peak)
    if rank != 0:
        dist.destroy_process_group()
        return

    step_seconds = [events[i].elapsed_time(events[i + 1]) / 1e3
                    for i in range(args.timing_warmup, profile_from) if i not in probed_steps]
    tokens_per_step = args.batch * seq
    timed = throughput(window_end - window_start - probe_seconds, profile_from - args.timing_warmup,
                       tokens_per_step, step_seconds)
    hf_config = json.loads(Path(args.model, "config.json").read_text())
    flops_token = train_flops_per_token(hf_config, seq)
    device_name = torch.cuda.get_device_name(device)
    peak_rate = peak_flops(device_name)
    record = {
        "framework": "torch",
        "precision": args.precision,
        "parallel": args.parallel,
        "world": world,
        "config": {**vars(args), "total_steps": total, "schedule_steps": schedule_steps, "seq": seq,
                   "accumulation": accumulation, "micro_batch": micro,
                   "data_sha256": sha256(args.data), "model_type": hf_config["model_type"],
                   "optimizer": "torch.optim.AdamW(fused=True)",
                   "schedule": "optax warmup_cosine_decay_schedule (common.warmup_cosine)",
                   "attention_implementation": model.config._attn_implementation,
                   "experts_implementation": getattr(model.config, "_experts_implementation", None)},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "device": device_name,
        "host": host(),
        "loss": loss_vector.cpu().tolist(),
        "grad_norm": torch.stack(norms).cpu().tolist(),
        "lr": rates,
        **({"aux_loss": torch.stack(auxes).cpu().tolist(), "router_probe": probes} if routed else {}),
        "throughput": timed,
        "flops_per_token": flops_token,
        "mfu": None if peak_rate is None else flops_token * timed["tokens_per_s"] / (peak_rate * world),
        "memory": {"peak_allocated_bytes": [int(p[0]) for p in peaks],
                   "peak_reserved_bytes": [int(p[1]) for p in peaks]},
    }
    if profiler is not None:
        write_record(args.out, record)
        kernels = []
        for index in range(world):
            raw = trace_dir / f"rank{index}.json"
            kernels.extend((name, start, end, index) for name, start, end, _ in chrome_trace_kernels(raw))
            raw.unlink()
        try:
            record["profile"] = kernel_summary(kernels, args.profile_steps)
            record["profile"]["kernels"] = str(trim_trace(kernels, trace_dir / "kernels.json.gz"))
        except ValueError as error:
            record["profile"] = {"error": str(error)}
    write_record(args.out, record)
    print(json.dumps({"first_loss": record["loss"][0], "last_loss": record["loss"][-1],
                      "tokens_per_s": timed["tokens_per_s"], "mfu": record["mfu"],
                      "peak_gib": max(record["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
