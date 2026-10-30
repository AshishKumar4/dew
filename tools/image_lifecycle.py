#!/usr/bin/env python3
"""Qualify an image model's lifecycle at 256 pixels: trained straight, and trained, stopped and resumed from
its checkpoint, the two runs alike bit for bit.

    python tools/image_lifecycle.py compare --output lifecycle

examples/train_flowers.py's diffusion model and objective on Oxford Flowers 102 (the Hub's copy at a pinned
commit) at 256 pixels, batch 16, under XLA's deterministic GPU ops (`--no-deterministic` without them), each
run in a fresh `output`: one trains `steps` straight; the other trains half of them and ends, and a new
process resumes it from its checkpoint to `steps` (`train`, a process each). Each saves 8 samples
(samples.png). It prints whether the two runs' states at `steps` (their checkpoints, every leaf bit for bit),
their samples (samples.png's bytes) and the losses they printed are equal.
"""

import dataclasses
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    deterministic: bool = True
    """Under XLA's deterministic GPU ops; without them a GPU's reductions may take another order each run."""


def train(args: Train) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import train_flowers

    from dew.data import Hub, presets

    config = train_flowers.Config(data=Path("."), output=args.output, steps=args.steps, image_size=256)
    data = (Hub(FLOWERS, split="train", revision=REVISION, columns=("image",)).shuffle(seed=0).repeat()
            .pipe(presets.images, config.image_size, batch=config.batch, augment=True))
    train_flowers.main(config, data)


def trained(output: Path, steps: int, deterministic: bool) -> list[tuple[str, str]]:
    """Each step and loss a fresh process logs training the model in `output` to `steps` (`train`), under
    XLA's deterministic GPU ops where `deterministic`."""
    flags = f"{os.environ.get('XLA_FLAGS', '')}{' --xla_gpu_deterministic_ops=true' if deterministic else ''}"
    env = {**os.environ, "XLA_FLAGS": flags.strip()}
    log = subprocess.run([sys.executable, __file__, "train", "--output", str(output), "--steps", str(steps)],
                         env=env, capture_output=True, text=True, check=True).stdout
    return re.findall(r"^step (\d+)/\d+\s+loss (\S+)", log, re.MULTILINE)


def weights(output: Path, step: int) -> dict[str, np.ndarray]:
    """What the run in `output` checkpointed at `step`: every leaf of its state, a host array by its path."""
    import jax

    from dew import Checkpoints

    state, _ = Checkpoints(str(output / "checkpoints")).restore(step=step)
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state)}


def compare(args: Compare) -> None:
    # Each run from no checkpoint: one left in its directory would resume it.
    runs = [args.output / run for run in ("straight", "resumed")]
    if taken := [str(run) for run in runs if run.exists()]:
        raise SystemExit(f"{', '.join(taken)} exists: compare trains each run afresh; name a new --output")
    for run in runs:
        run.mkdir(parents=True)
    straight = trained(runs[0], args.steps, args.deterministic)
    resumed = [*trained(runs[1], args.steps // 2, args.deterministic),
               *trained(runs[1], args.steps, args.deterministic)]
    first, second = (weights(run, args.steps) for run in runs)
    same = first.keys() == second.keys() and all(
        first[name].dtype == second[name].dtype and np.array_equal(first[name], second[name])
        for name in first)
    print(f"states at step {args.steps}: {'equal' if same else 'NOT equal'} ({len(first)} leaves)")
    samples = [hashlib.sha256((run / "samples.png").read_bytes()).hexdigest() for run in runs]
    print(f"samples: {'equal' if samples[0] == samples[1] else 'NOT equal'} ({samples[0]})")
    alike = "equal" if straight == resumed else "NOT equal"
    print(f"printed losses: {len(straight)}, {alike} to the digits shown")


if __name__ == "__main__":
    command = tyro.extras.subcommand_cli_from_dict({"train": Train, "compare": Compare})
    {Train: train, Compare: compare}[type(command)](command)
