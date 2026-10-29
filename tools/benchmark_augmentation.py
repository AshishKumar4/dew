#!/usr/bin/env python3
"""Compare image augmentation policies on fixed data: the same classifier, steps and seeds, one policy apart.

    python tools/benchmark_augmentation.py --output augmentation.json
    python tools/benchmark_augmentation.py --policies none flip --seeds 0 --steps 50 --output smoke.json

Oxford Flowers 102 (the Hub's copy at a pinned commit, its standard splits): a small residual network trains on
the 1,020 training images at 64 pixels from fresh weights, under each policy in turn, and is scored on the 6,149
test images, resized and never augmented. The policies are the device augmentation `presets.images` runs
(`images.DeviceAugment`): none, a horizontal flip, the flip with a random crop of 35-100% of the image's area
(staged at 80 pixels), and those with colour jitter (`augment=True`'s three). Each policy trains under every
seed, which draws the weights and the order and augmentation of the data alike, so the spread across seeds is
the noise a policy's difference is read against.
"""

import dataclasses
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import linen as nn

from dew import Trainer
from dew.data import Hub, Loading, images, presets
from dew.data.compile import compile as compile_data
from dew.objectives.supervised import Accuracy, CrossEntropy, Supervised
from dew.optim import Cosine, OptimConfig

FLOWERS, REVISION = "dpdl-benchmark/oxford_flowers102", "544b55c9af69a05742ead7571cd7ae23cfcd758b"
SIZE, STAGED, BATCH, CLASSES = 64, 80, 64, 102
TEST_BATCH = 143
"""The test split's 6,149 images in 43 whole batches."""
CROP = (0.35, 1.0)
Policy = Literal["none", "flip", "crop", "jitter"]
POLICIES: dict[str, images.DeviceAugment | None] = {
    "none": None,
    "flip": images.DeviceAugment(SIZE, flip=True, jitter=False),
    "crop": images.DeviceAugment(SIZE, crop_scale=CROP, flip=True, jitter=False),
    "jitter": images.DeviceAugment(SIZE, crop_scale=CROP, flip=True, jitter=True),
}


class Block(nn.Module):
    """Two 3x3 convolutions and a skip, group-normed (no batch statistics to carry)."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        y = nn.relu(nn.GroupNorm(8)(nn.Conv(self.features, (3, 3), self.stride)(x)))
        y = nn.GroupNorm(8)(nn.Conv(self.features, (3, 3))(y))
        if self.stride != 1 or x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1), self.stride)(x)
        return nn.relu(x + y)


class Classifier(nn.Module):
    """A residual network of four stages, 32 to 256 channels, its logits from the pooled last."""

    @nn.compact
    def __call__(self, image):
        x = image.astype(jnp.float32) / 127.5 - 1.0
        x = nn.relu(nn.GroupNorm(8)(nn.Conv(32, (3, 3))(x)))
        for index, features in enumerate((32, 64, 128, 256)):
            x = Block(features, stride=1 if index == 0 else 2)(x)
            x = Block(features)(x)
        return nn.Dense(CLASSES)(x.mean(axis=(1, 2)))


def split(name: str) -> Hub:
    return Hub(FLOWERS, split=name, revision=REVISION, columns=("image", "label"))


def trained(policy: str, seed: int, steps: int) -> tuple[float, float]:
    """Test top-1 accuracy of the classifier trained `steps` under `policy` and `seed`, and the run's seconds."""
    augment = POLICIES[policy]
    train = split("train").shuffle(seed=seed).repeat().pipe(
        presets.images, SIZE, batch=BATCH, augment=augment or False,
        staging_size=STAGED if augment is not None and augment.crop_scale != (1.0, 1.0) else None)
    data = compile_data(train, read="memory", provision=True, seed=seed,
                        loading=Loading(workers=0, threads=16, read_buffer=64))
    model = Classifier()
    optimizer = OptimConfig(learning_rate=Cosine(peak_value=2e-3, warmup_steps=steps // 20, end_value=2e-5),
                            weight_decay=0.05, clip_grads=1.0)
    trainer = Trainer(Supervised(model, CrossEntropy(), (Accuracy(),), inputs="image"), optimizer, key=seed)
    start = time.perf_counter()
    state = trainer.fit(data, steps=steps, log_every=max(1, steps // 10))
    seconds = time.perf_counter() - start
    test = split("test").pipe(presets.images, SIZE, batch=TEST_BATCH)
    predict = jax.jit(lambda variables, image: jnp.argmax(model.apply(variables, image), axis=-1))
    right = total = 0
    for batch in test:
        right += int(np.sum(np.asarray(predict(state.variables, batch["image"])) == np.asarray(batch["label"])))
        total += len(batch["label"])
    return right / total, seconds


@dataclasses.dataclass(frozen=True)
class Args:
    output: Path
    """Where the JSON record goes."""
    policies: tuple[Policy, ...] = ("none", "flip", "crop", "jitter")
    seeds: tuple[int, ...] = (0, 1, 2)
    steps: int = 3000
    """About 190 passes over the training split at batch 64."""


def main(args: Args) -> None:
    runs = []
    for policy in args.policies:
        for seed in args.seeds:
            accuracy, seconds = trained(policy, seed, args.steps)
            runs.append({"policy": policy, "seed": seed, "accuracy": accuracy, "seconds": round(seconds, 1)})
            print(json.dumps(runs[-1]), flush=True)
    summary = {policy: {"mean": float(np.mean(found)), "std": float(np.std(found, ddof=1)) if len(found) > 1
                        else None}
               for policy in args.policies
               for found in [[run["accuracy"] for run in runs if run["policy"] == policy]]}
    record = {"dataset": FLOWERS, "revision": REVISION, "size": SIZE, "staged": STAGED, "batch": BATCH,
              "steps": args.steps, "crop_scale": CROP, "device": jax.devices()[0].device_kind,
              "jax": version("jax"), "python": platform.python_version(), "runs": runs, "summary": summary}
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    for policy, found in summary.items():
        print(f"{policy}: {found['mean']:.4f}" + (f" ± {found['std']:.4f}" if found["std"] is not None else ""))


if __name__ == "__main__":
    main(tyro.cli(Args))
