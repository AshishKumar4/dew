#!/usr/bin/env python3
"""Megatron-Core's side of a language-model throughput row: the model shape,
global batch, sequence and precision of a Dew run (bf16 compute over fp32
master weights and Adam moments) on Megatron-Core's own model, distributed
optimizer and schedule, on the windows `token_windows.py` wrote, in their
recorded order.

The model is Megatron-Core's `GPTModel` on Transformer Engine layers, built
from the checkpoint's `config.json` alone: a Llama decoder (`llama`) or Qwen
3.5's hybrid of gated delta nets and gated full attention (`qwen3_5`), the
layout Megatron-Core trains Qwen3-Next with. It trains from Megatron's own
random init, so the record carries throughput, memory and a profile but no
curve to compare: the row measures Megatron-Core on the same work, not Dew's
numerics. The step is what Megatron-LM's `pretrain_gpt.py` runs, with its
fusions on: TE's fused norm-linear layers and attention backend choice, the
fused SwiGLU, rope and vocabulary cross entropy, and weight gradients
accumulated in the fused GEMM. Weight decay takes Megatron's default
grouping, which leaves the norms and biases undecayed; nothing in the step's
time depends on it.

`--documents LOW:HIGH` packs each row with documents of `common.documents`
lengths: Dew's side reads the same segment ids as `text_coordinates`, and
this side trains the rows as one THD sequence with each document's
`cu_seqlens`, the transition into the next document out of the loss. MFU is
Dew's count of the same step (`--work`, the file `dew_lm.py` writes for the
same shape and documents) over the timed window.

The container holds the environment (`megatron.Containerfile`):

    podman run --rm --device nvidia.com/gpu=0 --ipc=host -v $DEW:$DEW -v $RUNS:$RUNS \\
        -w $DEW/tools/reference_runs dew-megatron python megatron_lm.py --model <hf dir> \\
        --data windows.npz --batch 8 --work work.json --out megatron.json
    torchrun --nproc-per-node 2 megatron_lm.py --tp 2 ...
"""

import argparse
import dataclasses
import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from common import (
    TorchTracedWindow,
    WorkloadWork,
    documents,
    host,
    peak_flops,
    run_arguments,
    sha256,
    steps_for,
    throughput,
    warmup_cosine,
    window_rows,
    write_record,
)
from torch_lm import Ranks, attach_profile, trace_directory


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[run_arguments()])
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="rows per forward on one data rank; unset is the rank's whole share")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    parser.add_argument("--fsdp", action="store_true",
                        help="Megatron-FSDP over the data ranks: parameters, gradients and optimizer state "
                             "sharded (optim_grads_params); otherwise DDP with the distributed optimizer")
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--recompute", choices=("none", "selective", "full"), default="none")
    parser.add_argument("--cross-entropy", choices=("native", "te"), default="native",
                        help="the fused vocabulary cross entropy: Megatron-Core's, or Transformer Engine's, "
                             "which keeps no fp32 copy of the logits")
    parser.add_argument("--precision-aware-optimizer", action="store_true",
                        help="Transformer Engine's Adam updates the fp32 master weights and writes the bf16 "
                             "ones in one kernel")
    parser.add_argument("--moments", choices=("float32", "bfloat16"), default="float32",
                        help="Adam's moment dtype; bfloat16 takes the precision-aware optimizer")
    parser.add_argument("--seed", type=int, default=0, help="Megatron's model init")
    return parser.parse_args(argv)


def ranks(args: argparse.Namespace) -> Ranks:
    """The process group Megatron-Core needs, one process or a torchrun pool,
    and its tensor- and data-parallel groups."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if "RANK" not in os.environ:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MASTER_PORT", "29511"),
                          RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp)
    model_parallel_cuda_manual_seed(args.seed)
    world = dist.get_world_size()
    return Ranks(dist.get_rank(), world, torch.device("cuda", local), world > 1)


def text_config(model: str) -> dict:
    config = json.loads(Path(model, "config.json").read_text())
    return config.get("text_config", config)


def model_parts(hf: dict, args: argparse.Namespace):
    """Megatron-Core's TransformerConfig, layer spec and GPTModel arguments for the checkpoint's shape."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    rope = hf.get("rope_parameters") or {"rope_theta": hf.get("rope_theta", 10000.0)}
    common = dict(
        num_layers=hf["num_hidden_layers"], hidden_size=hf["hidden_size"],
        ffn_hidden_size=hf["intermediate_size"], num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        kv_channels=hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"],
        normalization="RMSNorm", layernorm_epsilon=hf["rms_norm_eps"], gated_linear_unit=True,
        activation_func=F.silu, add_bias_linear=False, add_qkv_bias=hf.get("attention_bias", False),
        hidden_dropout=0.0, attention_dropout=0.0, init_method_std=hf.get("initializer_range", 0.02),
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=args.tp, sequence_parallel=args.sequence_parallel,
        bias_activation_fusion=True, bias_dropout_fusion=True, masked_softmax_fusion=True,
        persist_layer_norm=True, apply_rope_fusion=True, gradient_accumulation_fusion=True,
        cross_entropy_loss_fusion=True, cross_entropy_fusion_impl=args.cross_entropy,
        attention_backend=AttnBackend.auto,
        recompute_granularity=None if args.recompute == "none" else args.recompute,
        recompute_method="uniform" if args.recompute == "full" else None,
        recompute_num_layers=1 if args.recompute == "full" else None,
    )
    model_type = hf["model_type"]
    if model_type == "llama":
        config = TransformerConfig(**common)
        spec = get_gpt_layer_with_transformer_engine_spec()
        rotary_percent = 1.0
    elif model_type == "qwen3_5_text":
        kinds = hf["layer_types"]
        config = TransformerConfig(
            **common, qk_layernorm=True, attention_output_gate=hf.get("attn_output_gate", False),
            layernorm_zero_centered_gamma=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[int(kind == "linear_attention") for kind in kinds],
            linear_conv_kernel_dim=hf["linear_conv_kernel_dim"],
            linear_key_head_dim=hf["linear_key_head_dim"], linear_value_head_dim=hf["linear_value_head_dim"],
            linear_num_key_heads=hf["linear_num_key_heads"],
            linear_num_value_heads=hf["linear_num_value_heads"])
        # One pipeline stage holds every layer.
        spec = get_transformer_block_with_experimental_attention_variant_spec(config, pp_rank=0)
        rotary_percent = rope.get("partial_rotary_factor", 1.0)
    else:
        raise SystemExit(f"{model_type} has no Megatron-Core mapping here; llama and qwen3_5 do")
    # Llama 3.1's frequency scaling is Megatron-Core's rope_scaling.
    scaling = {"rope_scaling": True, "rope_scaling_factor": rope["factor"]} \
        if rope.get("rope_type") == "llama3" else {}
    return config, spec, dict(
        vocab_size=hf["vocab_size"], share_embeddings_and_output_weights=hf.get("tie_word_embeddings", False),
        position_embedding_type="rope", rotary_percent=rotary_percent, rotary_base=int(rope["rope_theta"]),
        **scaling)


@dataclasses.dataclass
class Run:
    """The wrapped model, its optimizer and schedule, and the windows a step reads."""

    model: torch.nn.Module
    optimizer: object
    forward_backward: object
    config: object
    windows: np.ndarray
    order: np.ndarray
    segments: np.ndarray | None
    seq: int
    total: int
    schedule_steps: int
    share: int
    micro: int
    accumulation: int


def build(args: argparse.Namespace, place: Ranks) -> Run:
    from megatron.core import parallel_state
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
        FullyShardedDataParallel,
    )
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.gpt import GPTModel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module

    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    seq = windows.shape[1] - 1
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)
    data_ranks = parallel_state.get_data_parallel_world_size()
    if args.batch % data_ranks:
        raise ValueError(f"{args.batch} rows do not split over {data_ranks} data ranks")
    share = args.batch // data_ranks
    micro = args.micro_batch or share
    if share % micro:
        raise ValueError(f"a rank's {share} rows do not split into micro-batches of {micro}")
    segments = None if args.documents is None else documents(windows.shape, args.documents)

    config, spec, shape = model_parts(text_config(args.model), args)
    model = GPTModel(config, spec, max_sequence_length=seq, **shape).cuda()
    model = Float16Module(config, model)
    if args.fsdp:
        model = FullyShardedDataParallel(config, DistributedDataParallelConfig(
            use_distributed_optimizer=True, use_megatron_fsdp=True,
            data_parallel_sharding_strategy="optim_grads_params"), model)
    else:
        model = DistributedDataParallel(config, DistributedDataParallelConfig(use_distributed_optimizer=True),
                                        model)
    moments = getattr(torch, args.moments)
    optimizer = get_megatron_optimizer(OptimizerConfig(
        optimizer="adam", lr=args.lr_peak, min_lr=args.lr_end, weight_decay=args.weight_decay,
        adam_beta1=args.b1, adam_beta2=args.b2, adam_eps=args.eps, clip_grad=args.clip, bf16=True,
        params_dtype=torch.bfloat16, use_distributed_optimizer=True,
        use_precision_aware_optimizer=args.precision_aware_optimizer or args.moments != "float32",
        exp_avg_dtype=moments, exp_avg_sq_dtype=moments), [model])
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    config.no_sync_func = model.no_sync
    return Run(model, optimizer, get_forward_backward_func(), config, windows, order, segments, seq, total,
               schedule_steps, share, micro, share // micro)


def micro_batch(run: Run, rows: np.ndarray, device: torch.device) -> dict:
    """One forward's tensors: whole causal rows, or with documents the rows as
    one THD sequence whose `cu_seqlens` close each document, its positions
    counted from each document's start."""
    from megatron.core.packed_seq_params import PackedSeqParams

    ids = run.windows[rows].astype(np.int64)
    tokens, labels = ids[:, :-1], ids[:, 1:]
    if run.segments is None:
        positions = np.broadcast_to(np.arange(run.seq), tokens.shape)
        mask, packed = np.ones(tokens.shape, np.float32), None
    else:
        segments = run.segments[rows]
        inputs = segments[:, :-1].reshape(-1)
        # A row's end closes its last document as a new segment id does.
        starts = np.flatnonzero(np.concatenate([[True], (inputs[1:] != inputs[:-1])
                                                | (np.arange(1, inputs.size) % run.seq == 0)]))
        lengths = np.diff(np.append(starts, inputs.size))
        positions = (np.arange(inputs.size) - np.repeat(starts, lengths)).reshape(1, -1)
        mask = (segments[:, :-1] == segments[:, 1:]).astype(np.float32).reshape(1, -1)
        tokens, labels = tokens.reshape(1, -1), labels.reshape(1, -1)
        cu = torch.tensor(np.append(starts, inputs.size), dtype=torch.int32, device=device)
        longest = int(lengths.max())
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=longest, max_seqlen_kv=longest)

    def placed(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True)

    return {"input_ids": placed(tokens), "position_ids": placed(positions), "labels": placed(labels),
            "loss_mask": placed(mask), "packed_seq_params": packed}


def forward_step(data_iterator, model):
    """pretrain_gpt's step: the model's per-token losses and the loss function
    over the counted ones."""
    batch = next(data_iterator)
    losses = model(batch["input_ids"], batch["position_ids"], None, labels=batch["labels"],
                   packed_seq_params=batch["packed_seq_params"])
    return losses, partial(token_mean, batch["loss_mask"])


def token_mean(mask: torch.Tensor, losses: torch.Tensor):
    mask = mask.view(-1)
    total = torch.sum(losses.float().view(-1) * mask)
    count = mask.sum().to(torch.int)
    return total, count, {"lm loss": torch.stack([total.detach(), count.detach().float()])}


def train_step(args: argparse.Namespace, place: Ranks, run: Run, step: int) -> tuple[float, float]:
    """One optimizer step on this data rank's rows of step `step`: the token
    mean of the loss and the pre-clip gradient norm."""
    from megatron.core import parallel_state

    rank = parallel_state.get_data_parallel_rank()
    rows = window_rows(run.order, step, args.batch)[rank * run.share:(rank + 1) * run.share]
    batches = [micro_batch(run, rows[index * run.micro:(index + 1) * run.micro], place.device)
               for index in range(run.accumulation)]
    run.model.zero_grad_buffer()
    run.optimizer.zero_grad()
    reported = run.forward_backward(
        forward_step_func=forward_step, data_iterator=iter(batches), model=[run.model],
        num_microbatches=run.accumulation, seq_length=run.seq, micro_batch_size=run.micro,
        forward_only=False)
    _, norm, _ = run.optimizer.step()
    summed = torch.stack([entry["lm loss"] for entry in reported]).sum(dim=0)
    return summed, norm


def train(args: argparse.Namespace, place: Ranks, run: Run):
    """Every step in the recorded order, timed from `timing_warmup` to the profiled tail."""
    losses, norms, rates = [], [], []
    window = TorchTracedWindow(place.device, run.total, args.timing_warmup, args.profile_steps)
    torch.cuda.reset_peak_memory_stats(place.device)
    torch.cuda.synchronize(place.device)
    for step in range(run.total):
        window.before(step)
        lr = warmup_cosine(step, init=args.lr_init, peak=args.lr_peak, warmup=args.warmup,
                           decay_steps=run.schedule_steps, end=args.lr_end)
        for group in run.optimizer.param_groups:
            group["lr"] = lr
        loss, norm = train_step(args, place, run, step)
        losses.append(loss)
        norms.append(norm)
        rates.append(lr)
    window.close(trace_directory(args) / f"rank{place.rank}.json")
    return losses, norms, rates, window


def record(args: argparse.Namespace, place: Ranks, run: Run, losses, norms, rates, window,
           work: WorkloadWork) -> dict | None:
    """Rank 0's record: the token-mean losses, the timed throughput, every rank's memory peak and MFU
    from Dew's count of the same step."""
    summed = torch.stack(losses)
    dist.all_reduce(summed)
    peak = torch.tensor([torch.cuda.max_memory_allocated(place.device),
                         torch.cuda.max_memory_reserved(place.device)],
                        device=place.device, dtype=torch.float64)
    peaks = [torch.zeros_like(peak) for _ in range(place.world)]
    dist.all_gather(peaks, peak)
    if place.rank != 0:
        return None
    tokens_per_step = args.batch * run.seq
    if work.tokens_per_step != tokens_per_step:
        raise ValueError(f"{args.work} counts {work.tokens_per_step} tokens a step, this run trains "
                         f"{tokens_per_step}")
    peak_rate = peak_flops(torch.cuda.get_device_name(place.device))
    return {
        "framework": "megatron-core",
        "world": place.world,
        "config": {**vars(args), "total_steps": run.total, "schedule_steps": run.schedule_steps,
                   "seq": run.seq, "accumulation": run.accumulation, "micro_batch": run.micro,
                   "data_sha256": sha256(args.data), "model_type": text_config(args.model)["model_type"],
                   "transformer_config": {name: str(value) for name, value in vars(run.config).items()
                                          if not name.startswith("_") and not callable(value)},
                   "optimizer": "megatron.core DistributedOptimizer (Adam, decoupled weight decay)",
                   "schedule": "optax warmup_cosine_decay_schedule (common.warmup_cosine)"},
        "versions": {name: version(name) for name in ("megatron-core", "transformer_engine", "torch",
                                                      "flash-linear-attention")}
                    | {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "container": os.environ.get("NVIDIA_PYTORCH_VERSION")},
        "device": torch.cuda.get_device_name(place.device),
        "host": host(),
        "loss": (summed[:, 0] / summed[:, 1]).cpu().tolist(),
        "grad_norm": [float(norm) for norm in norms],
        "lr": rates,
        "throughput": throughput(window.seconds, window.timed_steps, tokens_per_step, window.step_seconds),
        "flops_per_token": work.training_flops_per_step / work.tokens_per_step,
        "work": dataclasses.asdict(work),
        "mfu": None if peak_rate is None else work.training_flops_per_step * window.timed_steps
            / window.seconds / (peak_rate * place.world),
        "memory": {"peak_allocated_bytes": [int(p[0]) for p in peaks],
                   "peak_reserved_bytes": [int(p[1]) for p in peaks]},
    }


def main() -> None:
    args = arguments()
    work = WorkloadWork.read(args.work)
    place = ranks(args)
    run = build(args, place)
    losses, norms, rates, window = train(args, place, run)
    result = record(args, place, run, losses, norms, rates, window, work)
    if result is not None:
        if window.profiler is not None:
            attach_profile(args, place, result)
        write_record(args.out, result)
        print(json.dumps({"first_loss": result["loss"][0], "last_loss": result["loss"][-1],
                          "tokens_per_s": result["throughput"]["tokens_per_s"],
                          "step_ms": result["throughput"]["step_ms_mean"], "mfu": result["mfu"],
                          "peak_gib": max(result["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
