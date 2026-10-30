#!/usr/bin/env python3
"""Qualify an image model's lifecycle at 256 pixels: trained straight, and trained, stopped and resumed from
its checkpoint, the two runs alike bit for bit.

    python tools/image_lifecycle.py --output lifecycle

examples/train_flowers.py's diffusion model and objective on Oxford Flowers 102 (the Hub's copy at a pinned
commit) at 256 pixels, batch 16, under XLA's deterministic GPU ops: one run trains `steps` straight; the other
trains half of them and ends, and a new process resumes it from its checkpoint to `steps`. Each saves 8
samples (samples.png). It prints whether every loss the two logged, and their samples, are equal.
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
RUN = """
import sys
from pathlib import Path

sys.path.insert(0, "examples")
import train_flowers

from dew.data import Hub, presets

data = (Hub({flowers!r}, split="train", revision={revision!r}, columns=("image",)).shuffle(seed=0).repeat()
        .pipe(presets.images, 256, batch=16, augment=True))
config = train_flowers.Config(data=Path("."), output=Path({output!r}), steps={steps}, image_size=256)
train_flowers.main(config, data)
"""


@dataclasses.dataclass(frozen=True)
class Args:
    output: Path
    """Where each run's checkpoints and samples go."""
    steps: int = 400


def trained(output: Path, steps: int) -> list[tuple[str, str]]:
    """Each step and loss a fresh process logs training the model in `output` to `steps`."""
    env = {**os.environ, "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true"}
    code = RUN.format(flowers=FLOWERS, revision=REVISION, output=str(output), steps=steps)
    log = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout
    return re.findall(r"^step (\d+)/\d+\s+loss (\S+)", log, re.MULTILINE)


def main(args: Args) -> None:
    straight = trained(args.output / "straight", args.steps)
    resumed = trained(args.output / "resumed", args.steps // 2) + trained(args.output / "resumed", args.steps)
    samples = {run: hashlib.sha256((args.output / run / "samples.png").read_bytes()).hexdigest()
               for run in ("straight", "resumed")}
    print(f"losses: {len(straight)} logged, {'equal' if straight == resumed else 'NOT equal'}")
    same = samples["straight"] == samples["resumed"]
    print(f"samples: {'equal' if same else 'NOT equal'} ({samples['straight']})")


if __name__ == "__main__":
    main(tyro.cli(Args))
