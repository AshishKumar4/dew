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
- `nonfinite`, only when there are any: how many NaN or infinite values the
  sampled latents and the decoded pixels held, over both seeds. The pixels
  are clipped and cast to uint8 before CLIP scores them, which would turn
  those values into ordinary pixels, so a row that has this field is a
  failed row whatever its `clip`.

A precision is `none`, `int8` or `fp8` (weights and activations), or `int8w`
or `fp8w` (weights only). `--float` keeps the modules whose paths contain one
of its names unquantized. A case Dew refuses to quantize, or one that fails
to compile, prints its error in place of the measurements.

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
from jax.typing import ArrayLike

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
    revision: str | None = "7187bb75a425dfb9fa0055951b8f0f7520185b87"
    """The Hub commit or tag to load; local run directories do not use it."""
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
    prepared = pipe.prepare(list(PROMPTS), key=0, steps=STEPS)
    x, t = prepared.noise, jnp.full(prepared.noise.shape[:1], 0.5)
    variables = {name: value for name, value in pipe.variables.items() if name not in ("encoders", "autoencoder")}

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


def nonfinite(array: ArrayLike) -> int:
    """How many entries of `array` are NaN or infinite."""
    return int(np.size(array) - np.count_nonzero(np.isfinite(np.asarray(array, np.float32))))


def sample(pipe: TextToImage, key: int | jax.Array, decode_batch: int) -> tuple[np.ndarray, dict[str, int]]:
    """Every prompt's image as uint8 pixels, and how many non-finite values
    the latents and the decoded pixels held before the pixels were clipped
    and cast, which would hide them."""
    latents = pipe(list(PROMPTS), key=key, steps=STEPS, solver=DPMSolverMultistep(), guidance=GUIDANCE,
                   decode=False).latents
    decode = jax.jit(pipe.autoencoder.decode)
    decoded = np.concatenate([np.asarray(decode(pipe.variables["autoencoder"], latents[i:i + decode_batch]), np.float32)
                              for i in range(0, len(latents), decode_batch)])
    return (uint8_pixels(np.clip(decoded, -1.0, 1.0)),
            {"latents": nonfinite(latents), "pixels": nonfinite(decoded)})


def quality(pipe: TextToImage, decode_batch: int) -> dict:
    from dew.data.text import load_tokenizer
    from dew.eval.images import DEFAULT_MODEL, clip_image_text_cosine

    tokenizer = load_tokenizer(DEFAULT_MODEL)
    tokens = tokenizer(list(PROMPTS), padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="np")
    scores, seconds, counts = [], [], {"latents": 0, "pixels": 0}
    for seed in SEEDS:
        started = time.perf_counter()
        pixels, found = sample(pipe, seed, decode_batch)
        seconds.append(time.perf_counter() - started)
        counts = {stage: counts[stage] + found[stage] for stage in counts}
        scores += list(np.asarray(clip_image_text_cosine(pixels, tokens["input_ids"], tokens["attention_mask"])))
    # The first seed compiles the sampler; the second is the warm time.
    row = {"sample_s": round(seconds[-1], 2), "clip": round(float(np.mean(scores)), 4)}
    return row | ({"nonfinite": counts} if any(counts.values()) else {})


def main(config: Config) -> None:
    if Path(config.model).is_dir():
        pipe = TextToImage.from_run(config.model, dtype=config.dtype)
    else:
        pipe = TextToImage.from_pretrained(config.model, revision=config.revision, dtype=config.dtype)
    row = {"device": jax.devices()[0].device_kind, "dtype": config.dtype, "precision": config.precision,
           "float": list(config.float)}
    try:
        if config.precision != "none":
            pipe = pipe.quantized(spec(config))
        row |= forward(pipe)
        if config.clip:
            row |= quality(pipe, config.decode_batch)
    except (ValueError, jax.errors.JaxRuntimeError) as error:
        row["error"] = str(error).splitlines()[0][:300]
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
