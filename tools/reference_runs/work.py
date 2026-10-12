#!/usr/bin/env python3
"""Count the Dew workload before running a reference in another environment.

The native config translator and objectives supply every model dimension and
the training rule. No weights, compiled program or accelerator are needed.
For example, write an LM numerator with

    PYTHONPATH=src JAX_PLATFORMS=cpu python tools/reference_runs/work.py lm \\
        --config checkpoint/config.json --batch 4 --seq 1024 --out work.json

Pass that file as --work to the reference runner at the same global shape.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from flax import linen as nn

from dew.objectives.base import Batch, Objective
from dew.registry import to_record
from dew.telemetry.workload import WorkloadWork


def measure(objective: Objective, batch: Batch) -> WorkloadWork:
    """Use the trainer's objective assembly on the complete host batch."""
    model = objective.program_key()[0].module
    return WorkloadWork(
        training_flops_per_step=objective.training_flops(None, batch),
        tokens_per_step=int(np.sum(objective.real_lengths(batch))),
        basis="analytic, real tokens, causal half where applicable, objective training rule",
        model_identity=json.dumps(to_record(model, nn.Module), sort_keys=True),
        batch_shape=np.shape(batch[objective.inputs.sample.key]))


def lm(args) -> WorkloadWork:
    """The checkpoint translator builds the same native decoder as Pretrained.load."""
    from dew.interop import translate_config
    from dew.objectives.lm import TEXT_KEY, LMObjective

    fields = translate_config(json.loads(Path(args.config).read_text()))
    fields.update(dtype=args.dtype, attention_impl=args.attention)
    objective = LMObjective(fields.value, args.seq, ema_decay=None)
    rows = np.broadcast_to(np.zeros((args.seq + 1,), np.int32), (args.batch, args.seq + 1))
    return measure(objective, {TEXT_KEY: rows})


def dit(args) -> WorkloadWork:
    """Reuse the Dew runner's denoiser and EDM objective construction."""
    from diffusion import counting_objective

    shape = (args.height, args.width, args.channels)
    objective = counting_objective(shape, args.dtype, args.attention)
    return measure(objective, {"image": np.broadcast_to(np.zeros(shape, np.uint8), (args.batch, *shape))})


def qwen35(args) -> WorkloadWork:
    """The backbone runner's hidden-state objective over the backbone it trains: the checkpoint
    through its own loader (`dew_qwen35_step.backbone`), or with --smoke the tiny config the torch
    side exports as its fixture."""
    from dew_qwen35_step import BackboneLoss, backbone
    from qwen35 import MODEL, REVISION, config, inputs

    from dew.interop import translate_config

    batch, sequence = (2, 16) if args.smoke else (8, 3072)
    if args.smoke:
        fields = translate_config(config(smoke=True))
        fields.update(dtype=args.dtype, attention_impl=args.attention)
        model = fields.value.clone(remat=None)
    else:
        model, _ = backbone(MODEL, REVISION, args.dtype)
    rows, projection = inputs(17, model.vocab_size, model.emb_features, batch, sequence)
    return measure(BackboneLoss(model, None, projection, sequence), {"tokens": rows})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    workloads = parser.add_subparsers(required=True)
    language = workloads.add_parser("lm")
    language.add_argument("--config", required=True, help="the checkpoint's config.json")
    language.add_argument("--batch", type=int, required=True, help="global rows per step")
    language.add_argument("--seq", type=int, required=True)
    language.set_defaults(count=lm)
    image = workloads.add_parser("dit")
    image.add_argument("--batch", type=int, required=True, help="global images per step")
    image.add_argument("--height", type=int, required=True)
    image.add_argument("--width", type=int, required=True)
    image.add_argument("--channels", type=int, default=3)
    image.set_defaults(count=dit)
    backbone = workloads.add_parser("qwen35")
    backbone.add_argument("--smoke", action="store_true")
    backbone.set_defaults(count=qwen35)
    for workload in (language, image, backbone):
        workload.add_argument("--out", required=True)
        workload.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
        workload.add_argument("--attention", default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = arguments()
    counted = args.count(args)
    counted.write(args.out)
    print(json.dumps({"training_flops_per_step": counted.training_flops_per_step,
                      "tokens_per_step": counted.tokens_per_step, "basis": counted.basis}))
