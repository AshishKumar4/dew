"""Published Z-Image against Diffusers, in float32 on one accelerator.

Loads `Tongyi-MAI/Z-Image` (or the repo and revision given) with Diffusers'
`ZImagePipeline` and with `dew.interop.load_pretrained`, and compares, each
gap divided by the larger of 1 and the reference's largest value:

- the prompt states of two prompts (the Qwen3 encoder's second-to-last
  layer, real tokens only);
- one transformer call at sigma 0.7 on fixed noise at 512x512, the native
  flow against the negated source output;
- the VAE decode of that noise;
- a four-step walk from the same noise, guided at the pipeline's 5.0, as the
  latent it ends on.

Needs about 50 GB of accelerator memory (the 6B transformer and the 4B
encoder in float32); the diffusers side runs on CUDA when there is one and is
freed before the native side loads.

    python tools/z_image_published_parity.py [REPO] [REVISION] > z_image_parity.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

REPO = "Tongyi-MAI/Z-Image"
REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"
PROMPTS = ["a red fox sitting in fresh snow, morning light", "a bowl of ramen"]
SIZE, STEPS, SIGMA, GUIDANCE = 512, 4, 0.7, 5.0


def gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def source(repo: str, revision: str) -> dict[str, np.ndarray]:
    import torch
    from diffusers import ZImagePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = ZImagePipeline.from_pretrained(repo, revision=revision, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    rows = SIZE // 8
    noise = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 16, rows, rows)).astype(np.float32))
    out: dict[str, np.ndarray] = {"noise": noise.numpy()}
    with torch.no_grad():
        states = pipe._encode_prompt(prompt=list(PROMPTS), device=device)
        for row, state in enumerate(states):
            out[f"context.{row}"] = state.cpu().numpy()
        (prediction,) = pipe.transformer([noise[0, :, None].to(device)], torch.full((1,), 1 - SIGMA, device=device),
                                         [states[0]], return_dict=False)[0]
        out["prediction"] = prediction[:, 0].cpu().numpy()[None]
        raw = noise.to(device) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        out["decoded"] = pipe.vae.decode(raw).sample.cpu().numpy()
        out["walked"] = pipe(prompt=PROMPTS[0], height=SIZE, width=SIZE, num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE, latents=noise.to(device), output_type="latent"
                             ).images.cpu().numpy()
    del pipe
    return out


def native(repo: str, revision: str, reference: dict[str, np.ndarray]) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.pretrained import load_pretrained

    loaded = load_pretrained(repo, revision=revision, dtype="float32", param_dtype="float32")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    condition = encoder.encode(params, encoder.tokenize(PROMPTS))
    gaps = {}
    for row in range(len(PROMPTS)):
        expected = reference[f"context.{row}"]
        gaps[f"context.{row}"] = gap(condition.context[row, :len(expected)], expected)
    noise = reference["noise"].transpose(0, 2, 3, 1)
    denoiser = {name: value for name, value in loaded.variables.items() if name not in ("encoders", "autoencoder")}
    flow = loaded.model.apply(denoiser, jnp.asarray(noise), jnp.full((1,), SIGMA * 1000.0),
                              DenoisingCondition(condition.context[:1], mask=condition.mask[:1]))
    gaps["prediction"] = gap(-np.asarray(flow), reference["prediction"].transpose(0, 2, 3, 1))
    decoded = loaded.autoencoder.decode(loaded.variables["autoencoder"], jnp.asarray(noise))
    gaps["decoded"] = gap(decoded, reference["decoded"].transpose(0, 2, 3, 1))
    task = loaded.text_to_image()
    walked = task(task.prepare([PROMPTS[0]], initial=noise, seed=0, steps=STEPS), key=jax.random.PRNGKey(0)).host()
    gaps["walked"] = gap(walked.latents, reference["walked"].transpose(0, 2, 3, 1))
    return gaps


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else REPO
    revision = sys.argv[2] if len(sys.argv) > 2 else REVISION
    gaps = native(repo, revision, source(repo, revision))
    print(json.dumps({"repo": repo, "revision": revision, "gaps": gaps}, indent=1))
