#!/usr/bin/env python3
"""Serve the pretrained text-to-image model with quantized weights, and measure it.

`TextToImage.quantized` stores the denoiser's kernels as int8 or fp8 values
with their scales (`dew.training.quantization.quantize_for_serving`). This
tool loads a model once in a compute dtype, quantizes it as one precision
says, and prints one JSON line:

- `weights_mib`: the denoiser's parameter bytes, in MiB.
- `forward_ms`: the median of 5 warm denoiser forwards over the guided batch
  (every prompt twice, conditional and unconditional), as one sampling step
  runs it; `arguments_mib` and `temporaries_mib` are that compiled forward's
  memory as XLA reports it.
- `sample_s`: the warm wall time of sampling every prompt at once, 20
  DPM-Solver++(2M) steps with guidance 5, encoding the prompts and decoding
  the latents included.
- `clip`: the mean CLIP ViT-L/14 image-text cosine over the prompts and seeds
  0 and 1.

A precision is `none`, `int8` or `fp8` (weights and activations), or `int8w`
or `fp8w` (weights only). `--float` keeps the modules whose paths contain one
of its names unquantized. A case that fails to compile prints its error in
place of the measurements.

Usage:
    python tools/benchmark_quantized_serving.py float32 none
    python tools/benchmark_quantized_serving.py bfloat16 int8 --float spatial_fusion
    JAX_PLATFORMS=cpu python tools/benchmark_quantized_serving.py float32 int8w --no-clip
"""

import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew.artifacts import uint8_pixels
from dew.sampling import CFG, DPMSolverMultistep, TextToImage
from dew.training.quantization import Quantization

PROMPTS = (
    "a watercolor painting of a mountain lake at sunrise",
    "the northern lights over a frozen lake at night",
    "a waterfall in a lush green forest",
    "an oil painting of a forest in autumn with golden leaves",
    "the milky way over snowy mountains at night",
    "a tropical beach with palm trees and turquoise water",
    "a cherry blossom tree in full bloom",
    "a colorful hot air balloon over a green valley",
    "a lighthouse at sunset, oil painting",
    "a red fox in a snowy forest",
    "a bowl of ramen with an egg and green onions",
    "a stained glass window with geometric patterns",
)
STEPS, GUIDANCE, SEEDS = 20, CFG(5.0), (0, 1)


@dataclass
class Config:
    dtype: tyro.conf.Positional[Literal["float32", "bfloat16"]]
    precision: tyro.conf.Positional[Literal["none", "int8", "int8w", "fp8", "fp8w"]]
    float: tuple[str, ...] = ()
    """Names of modules to leave unquantized, matched anywhere in their path."""
    model: str = "dewml/hybrid-dit-176m"
    """A Hugging Face Hub repository, or a local Dew run directory."""
    clip: bool = True
    """Sample every prompt at both seeds and score the images with CLIP."""
    decode_batch: int = 4
    """Latents decoded per autoencoder call; bounds the decode's memory on CPU."""


def spec(config: Config) -> Quantization:
    patterns = (".*",)
    if config.float:
        patterns = (f"^(?!.*(?:{'|'.join(map(re.escape, config.float))})).*",)
    return Quantization(dtype=config.precision.removesuffix("w"), weight_only=config.precision.endswith("w"),
                        patterns=patterns)


def nbytes(tree) -> int:
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(tree))


def forward(pipe: TextToImage) -> dict:
    """Time the denoiser over one guided step's batch, the way sampling calls
    it (`Denoiser.raw_both`), and read its compiled memory."""
    prepared = pipe.prepare(list(PROMPTS), seed=0, steps=STEPS)
    x, t = prepared.noise, jnp.full(prepared.noise.shape[:1], 0.5)
    variables = {name: value for name, value in pipe.params.items() if name not in ("encoders", "autoencoder")}

    def step(variables, x, t):
        return pipe.process.denoiser(pipe.model, variables, prepared.conditions,
                                     prepared.unconditional).raw_both(x, t)

    compiled = jax.jit(step).lower(variables, x, t).compile()
    memory = compiled.memory_analysis()
    jax.block_until_ready(compiled(variables, x, t))
    times = []
    for _ in range(5):
        started = time.perf_counter()
        jax.block_until_ready(compiled(variables, x, t))
        times.append(time.perf_counter() - started)
    return {"weights_mib": round(nbytes(variables["params"]) / 2**20, 1),
            "forward_ms": round(1e3 * statistics.median(times), 2),
            "arguments_mib": round(memory.argument_size_in_bytes / 2**20, 1),
            "temporaries_mib": round(memory.temp_size_in_bytes / 2**20, 1)}


def sample(pipe: TextToImage, seed: int, decode_batch: int) -> np.ndarray:
    latents = pipe(list(PROMPTS), seed=seed, steps=STEPS, sampler=DPMSolverMultistep(), guidance=GUIDANCE,
                   decode=False).latents
    decode = jax.jit(lambda params, z: jnp.clip(pipe.autoencoder.decode(params, z), -1.0, 1.0))
    images = [decode(pipe.params["autoencoder"], latents[i:i + decode_batch])
              for i in range(0, len(latents), decode_batch)]
    return uint8_pixels(np.concatenate([np.asarray(image, np.float32) for image in images]))


def quality(pipe: TextToImage, decode_batch: int) -> dict:
    from dew.data.processors import AutoTextTokenizer
    from dew.eval.images import DEFAULT_MODEL, clip_image_text_cosine

    tokens = AutoTextTokenizer(tensor_type="np", modelname=DEFAULT_MODEL)(list(PROMPTS))
    scores, seconds = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        pixels = sample(pipe, seed, decode_batch)
        seconds.append(time.perf_counter() - started)
        scores += list(np.asarray(clip_image_text_cosine(pixels, tokens["input_ids"], tokens["attention_mask"])))
    # The first seed compiles the sampler; the second is the warm time.
    return {"sample_s": round(seconds[-1], 2), "clip": round(float(np.mean(scores)), 4)}


def main(config: Config) -> None:
    load = TextToImage.from_run if Path(config.model).is_dir() else TextToImage.from_pretrained
    pipe = load(config.model, dtype=config.dtype)
    if config.precision != "none":
        pipe = pipe.quantized(spec(config))
    row = {"device": jax.devices()[0].device_kind, "dtype": config.dtype, "precision": config.precision,
           "float": list(config.float)}
    try:
        row |= forward(pipe)
        if config.clip:
            row |= quality(pipe, config.decode_batch)
    except jax.errors.JaxRuntimeError as error:
        row["error"] = str(error).splitlines()[0][:300]
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
