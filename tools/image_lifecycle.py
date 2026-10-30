#!/usr/bin/env python3
"""Qualify an image model's lifecycle at 256 pixels: trained straight, and trained, stopped and resumed from
its checkpoint, the two runs alike bit for bit.

    python tools/image_lifecycle.py compare --output lifecycle

examples/train_flowers.py's diffusion model and objective on Oxford Flowers 102 (the Hub's copy at a pinned
commit) at 256 pixels, batch 16, under XLA's deterministic GPU ops, each run in a fresh `output`: one trains
`steps` straight; the other trains half of them and ends, and a new process resumes it from its checkpoint to
`steps` (`train`, a process each). Each saves 8 samples (samples.png). It prints whether every loss the two
logged, and their samples, are equal.
"""

import dataclasses
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import tyro

FLOWERS, REVISION = "dpdl-benchmark/oxford_flowers102", "544b55c9af69a05742ead7571cd7ae23cfcd758b"


@dataclasses.dataclass(frozen=True)
class Train:
    """One process's training of the model in `output` to `steps`, from its checkpoint if it has one."""

    output: Path
    steps: int


@dataclasses.dataclass(frozen=True)
class Compare:
    """The straight run against the stopped and resumed one, each in a new directory under `output`."""

    output: Path
    steps: int = 400


def train(args: Train) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import train_flowers

    from dew.data import Hub, presets

    config = train_flowers.Config(data=Path("."), output=args.output, steps=args.steps, image_size=256)
    data = (Hub(FLOWERS, split="train", revision=REVISION, columns=("image",)).shuffle(seed=0).repeat()
            .pipe(presets.images, config.image_size, batch=config.batch, augment=True))
    train_flowers.main(config, data)


def trained(output: Path, steps: int) -> list[tuple[str, str]]:
    """Each step and loss a fresh process logs training the model in `output` to `steps` (`train`)."""
    env = {**os.environ, "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true"}
    log = subprocess.run([sys.executable, __file__, "train", "--output", str(output), "--steps", str(steps)],
                         env=env, capture_output=True, text=True, check=True).stdout
    return re.findall(r"^step (\d+)/\d+\s+loss (\S+)", log, re.MULTILINE)


def compare(args: Compare) -> None:
    # Each run from no checkpoint: one left in its directory would resume it.
    for run in ("straight", "resumed"):
        (args.output / run).mkdir(parents=True)
    straight = trained(args.output / "straight", args.steps)
    resumed = trained(args.output / "resumed", args.steps // 2) + trained(args.output / "resumed", args.steps)
    samples = {run: hashlib.sha256((args.output / run / "samples.png").read_bytes()).hexdigest()
               for run in ("straight", "resumed")}
    print(f"losses: {len(straight)} logged, {'equal' if straight == resumed else 'NOT equal'}")
    same = samples["straight"] == samples["resumed"]
    print(f"samples: {'equal' if same else 'NOT equal'} ({samples['straight']})")


if __name__ == "__main__":
    command = tyro.extras.subcommand_cli_from_dict({"train": Train, "compare": Compare})
    {Train: train, Compare: compare}[type(command)](command)
