#!/usr/bin/env python3
"""MaxText's side of a language-model throughput row: the model shape, global
batch, sequence and precision policy (fp32 weights, bf16 compute) of a Dew
run, on MaxText's own trainer, in process so the record can read the devices'
peak memory after it.

MaxText trains from its own random init on its synthetic tokens, so the
record carries throughput, memory and a profile but no curve to compare: the
row measures the strongest JAX trainer on the same work, not Dew's numerics.
The optimizer takes the reference runs' AdamW settings and a schedule length
fixed apart from `--steps`, so a short warm-up run compiles the program a
measured run executes.

    PYTHONPATH=<maxtext site> python maxtext_run.py --model <hf checkpoint> \
        --model-name qwen3-0.6b --mesh fsdp --batch 8 --seq 1024 --out record.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("DECOUPLE_GCLOUD", "TRUE")

import jax
from common import attach_xplane_profile, host, peak_flops, throughput, train_flops_per_token, write_record


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="the HF checkpoint whose config.json counts the FLOPs")
    parser.add_argument("--model-name", required=True, help="MaxText's config for the same shape")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mesh", choices=("data", "fsdp"), required=True)
    parser.add_argument("--batch", type=int, required=True, help="global rows per step")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--schedule-steps", type=int, default=256)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--attention", default="cudnn_flash_jax")
    parser.add_argument("--remat", default="full")
    parser.add_argument("--lr-peak", type=float, default=2e-5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("overrides", nargs="*", help="further MaxText key=value settings")
    return parser.parse_args()


def maxtext_argv(args: argparse.Namespace, workdir: Path) -> list[str]:
    import maxtext

    devices = jax.device_count()
    if args.batch % devices:
        raise ValueError(f"batch {args.batch} does not split over {devices} devices")
    base = Path(maxtext.__file__).parent / "configs" / "base.yml"
    settings = {
        "model_name": args.model_name, "hardware": "gpu", "skip_jax_distributed_system": True,
        "run_name": "reference", "base_output_directory": str(workdir),
        "metrics_file": str(workdir / "metrics.jsonl"), "enable_tensorboard": False,
        "enable_checkpointing": False, "dataset_type": "synthetic",
        "per_device_batch_size": args.batch // devices, "max_target_length": args.seq,
        "steps": args.steps, "learning_rate_schedule_steps": args.schedule_steps,
        "attention": args.attention, "remat_policy": args.remat,
        "dtype": "bfloat16", "weight_dtype": "float32",
        "ici_data_parallelism": devices if args.mesh == "data" else 1,
        "ici_fsdp_parallelism": devices if args.mesh == "fsdp" else 1,
        "opt_type": "adamw", "learning_rate": args.lr_peak, "adam_b1": args.b1, "adam_b2": args.b2,
        "adam_eps": args.eps, "adam_weight_decay": args.weight_decay,
        "gradient_clipping_threshold": args.clip,
        "profiler": "xplane" if args.profile_steps else "",
        "skip_first_n_steps_for_profiler": args.steps - args.profile_steps,
        "profiler_steps": args.profile_steps,
    }
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        settings["jax_cache_dir"] = os.environ["JAX_COMPILATION_CACHE_DIR"]
    return ["train.py", str(base), *(f"{key}={value}" for key, value in settings.items()), *args.overrides]


def main() -> None:
    args = arguments()
    from absl import logging
    from maxtext.trainers.pre_train import train

    logging.set_verbosity(logging.INFO)  # MaxText logs its steps at INFO, absl's default hides them
    workdir = Path(tempfile.mkdtemp(prefix="maxtext-"))
    argv = maxtext_argv(args, workdir)
    train.main(argv)
    steps = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines()]
    timed = [row for row in steps if args.timing_warmup <= row["step"] < args.steps - args.profile_steps]
    seconds = [row["perf/step_time_seconds"] for row in timed]
    hf_config = json.loads(Path(args.model, "config.json").read_text())
    flops_token = train_flops_per_token(hf_config, args.seq)
    device_kind = jax.devices()[0].device_kind
    devices = jax.device_count()
    rate = throughput(sum(seconds), len(seconds), args.batch * args.seq, seconds)
    peak_rate = peak_flops(device_kind)
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    record = {
        "framework": "maxtext",
        "precision": "bf16 compute, fp32 weights",
        "parallel": f"{args.mesh}={devices}",
        "world": devices,
        "config": {**vars(args), "maxtext_argv": argv[2:], "data": "MaxText synthetic tokens",
                   "init": "MaxText random init", "model_type": hf_config["model_type"]},
        "versions": {"jax": jax.__version__, "maxtext": version("maxtext")},
        "device": device_kind,
        "host": host(),
        "loss": [row["learning/loss"] for row in steps],
        "throughput": rate,
        "flops_per_token": flops_token,
        "maxtext_tflops_per_device": [row.get("perf/per_device_tflops_per_sec") for row in timed],
        "mfu": None if peak_rate is None else flops_token * rate["tokens_per_s"] / (peak_rate * devices),
        "memory": {"peak_allocated_bytes": [int(s.get("peak_bytes_in_use", 0)) for s in stats],
                   "bytes_limit": [int(s.get("bytes_limit", 0)) for s in stats]},
    }
    if args.profile_steps:
        attach_xplane_profile(record, workdir, args.profile_steps,
                              rows=Path(args.out).resolve().with_suffix("") / "kernels.json.gz")
    shutil.rmtree(workdir)
    write_record(args.out, record)
    print(json.dumps({"tokens_per_s": rate["tokens_per_s"], "step_ms": rate["step_ms_mean"], "mfu": record["mfu"],
                      "peak_gib": max(record["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
