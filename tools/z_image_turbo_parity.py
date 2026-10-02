"""Published Z-Image-Turbo against Diffusers, in float32 at the highest precision.

The landing page shows Turbo samples drawn through Dew, so this holds Dew to
`ZImagePipeline` on Turbo's own weights, not only the base model's. Each gap is
divided by the larger of 1 and the reference's largest value:

- the prompt states of the prompts (Qwen3's second-to-last layer, real tokens);
- the transformer's prediction at three noise levels on fixed noise, from the
  source's prompt states: the native flow against the negated source output;
- one 9-step sample, the model card's call (`num_inference_steps=9`,
  `guidance_scale=0.0`; Dew's `steps=9, guidance=None`), from the same initial
  latents and prompt states on both sides: the final latent, the decoded image,
  and that image's 8-bit pixels.

Diffusers runs first and keeps only host arrays, then Dew, so the two never share
the card: the peak is one float32 copy of the 6B transformer (about 25 GB).
Dew's transformer loads through `frontier_samples.streamed`, which puts each
translated leaf on the device as it is inserted and leaves the text encoder out
of the sampling load; the conditioner loads on its own first.

    python tools/z_image_turbo_parity.py --out out/z_image_turbo > out/z_image_turbo/parity.json
    python tools/z_image_turbo_parity.py --checkpoint tests-fixture-dir --size 16x24 --out DIR  # wiring check
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Both sides in true float32: an A100 otherwise multiplies float32 in TF32.
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
PROMPTS = [
    'a neon sign that reads "dew" glowing in a rainy Tokyo alley at night, reflections on wet pavement',
    "a red fox sitting in fresh snow in a pine forest, soft morning light, wildlife photograph",
]
SIGMAS = (0.9, 0.5, 0.1)
STEPS = 9


def gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def pixels(unit):
    """[0, 1] images as the 8-bit pixels a PNG of them holds."""
    return np.clip(np.rint(np.asarray(unit, np.float64) * 255.0), 0, 255).astype(np.uint8)


def free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source(directory: Path, height: int, width: int, noise: np.ndarray) -> dict:
    """Everything Diffusers computes, as host arrays."""
    import torch
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )
    from transformers import AutoModel, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result: dict = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu"}

    text_encoder = AutoModel.from_pretrained(directory / "text_encoder", dtype=torch.float32).to(device)
    shell = SimpleNamespace(
        text_encoder=text_encoder,
        tokenizer=AutoTokenizer.from_pretrained(directory / "tokenizer"),
        _execution_device=device,
    )
    with torch.no_grad():
        result["states"] = [
            state.cpu().numpy()
            for state in ZImagePipeline._encode_prompt(shell, prompt=list(PROMPTS), device=device)
        ]
    del shell, text_encoder
    free()

    transformer = ZImageTransformer2DModel.from_pretrained(
        directory / "transformer", torch_dtype=torch.float32
    )
    transformer = transformer.to(device)
    state = torch.from_numpy(result["states"][0]).to(device)
    result["predictions"] = {}
    with torch.no_grad():
        for sigma in SIGMAS:
            (prediction,) = transformer(
                [torch.from_numpy(noise[0, :, None]).to(device)],
                torch.full((1,), 1 - sigma, device=device),
                [state],
                return_dict=False,
            )[0]
            result["predictions"][sigma] = prediction[:, 0].cpu().numpy()[None].transpose(0, 2, 3, 1)
    vae = AutoencoderKL.from_pretrained(directory / "vae", torch_dtype=torch.float32).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory / "scheduler")
    pipeline = ZImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer
    )
    tick = time.monotonic()
    with torch.no_grad():
        latents = pipeline(
            prompt_embeds=[state],
            height=height,
            width=width,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            latents=torch.from_numpy(noise).to(device),
            output_type="latent",
        ).images
        result["latents"] = latents.cpu().numpy().transpose(0, 2, 3, 1)
        # The pipeline's own decode after the walk.
        raw = latents.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
        decoded = vae.decode(raw, return_dict=False)[0]
        result["images"] = (decoded / 2 + 0.5).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
    result["walk_seconds"] = round(time.monotonic() - tick, 1)
    result["timesteps"] = [float(value) for value in pipeline.scheduler.timesteps]
    del pipeline, transformer, vae, state
    free()
    return result


def native(checkpoint: str, revision: str | None, height: int, width: int, noise: np.ndarray, states) -> dict:
    """The same quantities through Dew."""
    import jax
    import jax.numpy as jnp
    from frontier_samples import streamed

    from dew.diffusion.process import DenoisingCondition
    from dew.inputs.diffusion import HiddenStatesConditioner
    from dew.interop.pretrained import load_diffusion_source
    from dew.sampling.pipelines import DenoisingInputs

    result: dict = {"device": jax.devices()[0].device_kind}
    with streamed():
        encoder = HiddenStatesConditioner.from_pretrained(
            checkpoint, revision=revision, dtype="float32", param_dtype="float32", attention_impl="xla"
        )
    condition = jax.jit(encoder.encode)(encoder.params, encoder.tokenize(PROMPTS))
    result["states"] = [np.asarray(condition.context[row, : len(state)]) for row, state in enumerate(states)]
    result["mask_lengths"] = [int(length) for length in np.asarray(condition.mask).sum(axis=1)]
    budget = condition.context.shape[1]
    del encoder, condition
    gc.collect()

    with streamed(skip=("text_encoder",)):
        pipeline = load_diffusion_source(
            checkpoint,
            revision=revision,
            dtype="float32",
            param_dtype="float32",
            attention_impl="xla",
            size=(height, width),
        )
    task = pipeline.text_to_image()
    padded = np.zeros((1, budget, states[0].shape[-1]), np.float32)
    padded[0, : len(states[0])] = states[0]
    mask = np.arange(budget)[None] < len(states[0])
    given = DenoisingCondition(jnp.asarray(padded), mask=jnp.asarray(mask))
    variables = {
        name: value for name, value in task.params.items() if name not in ("encoders", "autoencoder")
    }
    initial = jnp.asarray(noise.transpose(0, 2, 3, 1))
    apply = jax.jit(lambda params, x, t: task.model.apply(params, x, t, given))
    result["predictions"] = {
        sigma: -np.asarray(apply(variables, initial, jnp.full((1,), sigma * 1000.0))) for sigma in SIGMAS
    }
    tick = time.monotonic()
    prepared = DenoisingInputs(initial, {"conditioning": given}, {"conditioning": given}, rows=1)
    walked = task(prepared, steps=STEPS, guidance=None, key=0).host()
    result["walk_seconds"] = round(time.monotonic() - tick, 1)
    result["latents"] = np.asarray(walked.latents)
    result["images"] = np.clip(np.asarray(walked.images, np.float64) / 2 + 0.5, 0.0, 1.0)
    _, times = task.prepared_process(STEPS)
    result["times"] = None if times is None else [float(value) for value in times]
    result["solver"] = type(task.solver).__name__
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--size", default="1024x1024", help="HEIGHTxWIDTH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image

    from dew.interop import sources

    local = Path(args.checkpoint).is_dir()
    revision = None if local else args.revision
    directory = (
        Path(args.checkpoint)
        if local
        else sources.snapshot(args.checkpoint, revision, weights=("text_encoder", "transformer", "vae"))
    )
    height, width = (int(part) for part in args.size.lower().split("x"))
    vae_config = json.loads((directory / "vae" / "config.json").read_text())
    scale = 2 ** (len(vae_config["block_out_channels"]) - 1)
    shape = (1, vae_config["latent_channels"], height // scale, width // scale)
    noise = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    reference = source(directory, height, width, noise)
    ours = native(
        str(directory) if local else args.checkpoint, revision, height, width, noise, reference["states"]
    )

    gaps = {
        f"context.{row}": gap(ours["states"][row], state) for row, state in enumerate(reference["states"])
    }
    gaps.update(
        {
            f"prediction.sigma{sigma}": gap(ours["predictions"][sigma], reference["predictions"][sigma])
            for sigma in SIGMAS
        }
    )
    gaps["sample.latents"] = gap(ours["latents"], reference["latents"])
    gaps["sample.image"] = gap(ours["images"], reference["images"])
    difference = np.abs(
        pixels(ours["images"]).astype(np.int16) - pixels(reference["images"]).astype(np.int16)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels(reference["images"])[0]).save(args.out / "diffusers.png")
    Image.fromarray(pixels(ours["images"])[0]).save(args.out / "dew.png")
    import jax
    import torch

    record = {
        "repo": args.checkpoint,
        "revision": revision,
        "size": [height, width],
        "prompts": PROMPTS,
        "sampled_prompt": PROMPTS[0],
        "steps": STEPS,
        "source_call": (
            "ZImagePipeline(..., text_encoder=None)(prompt_embeds=[states], num_inference_steps=9, "
            "guidance_scale=0.0, latents=noise, output_type='latent'), then the pipeline's VAE decode"
        ),
        "dew_call": (
            "load_diffusion_source(..., dtype='float32', param_dtype='float32', attention_impl='xla')"
            ".text_to_image()(DenoisingInputs(noise, states), steps=9, guidance=None, key=0)"
        ),
        "dew_solver": ours["solver"],
        "source_timesteps": reference["timesteps"],
        "dew_times": ours["times"],
        "mask_lengths": ours["mask_lengths"],
        "gaps": gaps,
        "pixels": {"max_abs": int(difference.max()), "mean_abs": float(difference.mean())},
        "walk_seconds": {"diffusers": reference["walk_seconds"], "dew": ours["walk_seconds"]},
        "precision": {
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "devices": {"diffusers": reference["device"], "dew": ours["device"]},
        "versions": {
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "transformers": __import__("transformers").__version__,
            "jax": jax.__version__,
        },
    }
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
