"""Published FLUX.2 [klein] against Diffusers, in float32 on one accelerator.

Loads `black-forest-labs/FLUX.2-klein-4B` (or the repo and revision given)
with Diffusers' `Flux2KleinPipeline` and with `dew.interop.load_pretrained`,
and compares, each gap divided by the larger of 1 and the reference's
largest value:

- the prompt states of two prompts (the Qwen3 encoder's layers 9, 18, 27);
- one transformer call at sigma 0.7 on fixed noise at 512x512;
- the VAE decode of that noise;
- a four-step walk from the same noise, as the latent it ends on.

Needs about 40 GB of accelerator memory (both models in float32); the
diffusers side runs on CUDA when there is one.

    python tools/flux2_published_parity.py [REPO] [REVISION] > flux2_parity.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PROMPTS = ["a red fox sitting in fresh snow, morning light", "a bowl of ramen"]
SIZE, STEPS, SIGMA = 512, 4, 0.7


def gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def source(repo: str, revision: str) -> dict[str, np.ndarray]:
    import torch
    from diffusers import Flux2KleinPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Flux2KleinPipeline.from_pretrained(repo, revision=revision, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    rows = SIZE // 16
    noise = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 128, rows, rows)).astype(np.float32))
    out: dict[str, np.ndarray] = {"noise": noise.numpy()}
    with torch.no_grad():
        embeds, text_ids = pipe.encode_prompt(prompt=PROMPTS, device=device)
        out["context"] = embeds.cpu().numpy()
        latent_ids = pipe._prepare_latent_ids(noise).to(device)
        prediction = pipe.transformer(
            hidden_states=pipe._pack_latents(noise.to(device)), encoder_hidden_states=embeds[:1],
            timestep=torch.full((1,), SIGMA, device=device), img_ids=latent_ids, txt_ids=text_ids[:1],
            guidance=None, return_dict=False)[0]
        out["prediction"] = prediction.cpu().numpy()
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(device)
        raw = pipe._unpatchify_latents(noise.to(device) * std + mean)
        out["decoded"] = pipe.vae.decode(raw).sample.cpu().numpy()
        out["walked"] = pipe(prompt=PROMPTS[0], height=SIZE, width=SIZE, num_inference_steps=STEPS,
                             latents=noise.to(device), output_type="latent").images.cpu().numpy()
    del pipe
    return out


def native(repo: str, revision: str, reference: dict[str, np.ndarray]) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.pretrained import load_pretrained
    from dew.nn.autoencoders.flux2 import unfold

    loaded = load_pretrained(repo, revision=revision, dtype="float32", param_dtype="float32")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    context = encoder.encode(params, encoder.tokenize(PROMPTS)).context
    noise = reference["noise"].transpose(0, 2, 3, 1)
    denoiser = {name: value for name, value in loaded.variables.items() if name not in ("encoders", "autoencoder")}
    prediction = loaded.model.apply(denoiser, jnp.asarray(noise), jnp.full((1,), SIGMA * 1000.0),
                                    DenoisingCondition(context[:1]))
    autoencoder = loaded.autoencoder
    decoded = autoencoder.decode(loaded.variables["autoencoder"], jnp.asarray(noise))
    task = loaded.text_to_image()
    walked = task(task.prepare([PROMPTS[0]], initial=noise, seed=0, steps=STEPS), key=jax.random.PRNGKey(0)).host()
    raw = unfold(np.asarray(walked.latents) / autoencoder.latent_scale + autoencoder.latent_shift)
    rows = noise.shape[1]
    return {"context": gap(context, reference["context"]),
            "prediction": gap(np.asarray(prediction).reshape(1, rows * rows, -1), reference["prediction"]),
            "decoded": gap(decoded, reference["decoded"].transpose(0, 2, 3, 1)),
            "walked": gap(raw, reference["walked"].transpose(0, 2, 3, 1))}


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else REPO
    revision = sys.argv[2] if len(sys.argv) > 2 else REVISION
    gaps = native(repo, revision, source(repo, revision))
    print(json.dumps({"repo": repo, "revision": revision, "gaps": gaps}, indent=1))
