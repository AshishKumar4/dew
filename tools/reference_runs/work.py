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
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from flax import linen as nn

from dew.batch import Batch
from dew.nn.flops import MissingFlopsCount
from dew.objectives.base import Objective
from dew.registry import to_record
from dew.telemetry.workload import WorkloadWork

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(objective: Objective, batch: Batch) -> WorkloadWork:
    """Use the trainer's objective assembly on the complete host batch.

    A model with no token geometry, a U-Net's convolutions, counts no tokens.
    """
    model = objective.program_key()[0].module
    try:
        tokens = int(np.sum(objective.real_lengths(batch)))
    except MissingFlopsCount:
        tokens = None
    return WorkloadWork(
        training_flops_per_step=objective.training_flops(None, batch),
        tokens_per_step=tokens,
        basis="analytic, real tokens, causal half where applicable, objective training rule",
        model_identity=json.dumps(to_record(model, nn.Module), sort_keys=True),
        batch_shape=np.shape(batch[objective.inputs.sample.key]))


def lm(args) -> WorkloadWork:
    """The checkpoint translator builds the same native decoder as Pretrained.load."""
    from dew.batch import TEXT_KEY
    from dew.interop import translate_config
    from dew.objectives.lm import LMObjective

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


def case(args) -> WorkloadWork:
    """A framework parity case (`benchmark_cases.parity_case`), built and fed as tools/benchmark_step.py
    builds and feeds it, with the torch side's --batch-size and --vocab-size."""
    from benchmark_cases import parity_case
    from benchmark_models import build_objective, global_batch

    chosen = dataclasses.replace(parity_case(args.model, args.size, batch_size=args.batch_size,
                                             vocab_size=args.vocab_size), dtype=args.dtype)
    return measure(build_objective(chosen, args.attention), global_batch(chosen))


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
    parity = workloads.add_parser("case")
    parity.add_argument("--model", required=True, choices=("causal_transformer", "simple_dit", "unet"))
    parity.add_argument("--size", default="small", choices=("small", "large"))
    parity.add_argument("--batch-size", type=int, default=None)
    parity.add_argument("--vocab-size", type=int, default=None)
    parity.set_defaults(count=case)
    for workload in (language, image, backbone, parity):
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
