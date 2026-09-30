"""Published Z-Image against Diffusers, in float32, one component at a time.

Loads each component of `Tongyi-MAI/Z-Image` (or the repo and revision
given) with Diffusers and with Dew in turn - the Qwen3 encoder, the
transformer, the VAE - freeing each before the next, so the peak is the
6B transformer in float32 twice over at most (about 30 GB on the
accelerator, the source's copy freed before Dew's loads). It compares, each
gap divided by the larger of 1 and the reference's largest value:

- the prompt states of two prompts (the encoder's second-to-last layer,
  real tokens only);
- one transformer call at sigma 0.7 on fixed noise at 512x512, the native
  flow against the negated source output;
- the VAE decode of that noise.

The pipeline around them (template, guidance, grid, walk) is held to the
source by the tiny pipeline in `tests/test_z_image_source.py`.

    python tools/z_image_published_parity.py [REPO] [REVISION] > z_image_parity.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = "Tongyi-MAI/Z-Image"
REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"
PROMPTS = ["a red fox sitting in fresh snow, morning light", "a bowl of ramen"]
SIZE, SIGMA = 512, 0.7


def gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(repo: str, revision: str) -> dict[str, float]:
    import jax.numpy as jnp
    import torch
    from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel
    from transformers import AutoModel, AutoTokenizer

    from dew.diffusion.process import DenoisingCondition
    from dew.interop import sources
    from dew.interop.pretrained import _denoiser, _diffusion_vae, load_hidden_states_conditioner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    directory = sources.snapshot(repo, revision, weights=("text_encoder", "transformer", "vae"))
    gaps: dict[str, float] = {}

    # The encoder.
    text_encoder = AutoModel.from_pretrained(directory / "text_encoder", dtype=torch.float32).to(device)
    # The pipeline's own prompt encoding, over the two components it reads.
    shell = SimpleNamespace(text_encoder=text_encoder, tokenizer=AutoTokenizer.from_pretrained(directory / "tokenizer"),
                            _execution_device=device)
    with torch.no_grad():
        expected = [state.cpu().numpy()
                    for state in ZImagePipeline._encode_prompt(shell, prompt=list(PROMPTS), device=device)]
    del shell, text_encoder
    free()
    encoder = load_hidden_states_conditioner(str(directory), dtype="float32", param_dtype="float32")
    condition = encoder.encode(encoder.params, encoder.tokenize(PROMPTS))
    for row, state in enumerate(expected):
        gaps[f"context.{row}"] = gap(condition.context[row, :len(state)], state)
    context, mask = np.asarray(condition.context[:1]), np.asarray(condition.mask[:1])
    del encoder, condition
    gc.collect()

    # The transformer, on the first prompt's source states.
    vae_config = json.loads((directory / "vae" / "config.json").read_text())
    rows = SIZE // 2 ** (len(vae_config["block_out_channels"]) - 1)
    channels = vae_config["latent_channels"]
    noise = np.random.default_rng(0).standard_normal((1, channels, rows, rows)).astype(np.float32)
    transformer = ZImageTransformer2DModel.from_pretrained(directory / "transformer",
                                                           torch_dtype=torch.float32).to(device)
    with torch.no_grad():
        (prediction,) = transformer([torch.from_numpy(noise[0, :, None]).to(device)],
                                    torch.full((1,), 1 - SIGMA, device=device),
                                    [torch.from_numpy(expected[0]).to(device)], return_dict=False)[0]
        prediction = prediction[:, 0].cpu().numpy()[None].transpose(0, 2, 3, 1)
    del transformer
    free()
    denoiser = _denoiser(directory, dtype="float32", attention_impl="auto")
    variables, _ = denoiser.weights("float32")
    padded = np.zeros_like(context)
    padded[0, :len(expected[0])] = expected[0]
    flow = denoiser.model.apply(variables, jnp.asarray(noise.transpose(0, 2, 3, 1)), jnp.full((1,), SIGMA * 1000.0),
                                DenoisingCondition(jnp.asarray(padded), mask=jnp.asarray(mask)))
    gaps["prediction"] = gap(-np.asarray(flow), prediction)
    del denoiser, variables, flow
    gc.collect()

    # The VAE, decoding the noise as the pipeline decodes a latent.
    vae = AutoencoderKL.from_pretrained(directory / "vae", torch_dtype=torch.float32).to(device)
    with torch.no_grad():
        raw = torch.from_numpy(noise).to(device) / vae.config.scaling_factor + vae.config.shift_factor
        decoded = vae.decode(raw).sample.cpu().numpy().transpose(0, 2, 3, 1)
    del vae
    free()
    autoencoder, params, _, _ = _diffusion_vae(directory, jnp.float32)
    gaps["decoded"] = gap(autoencoder.decode(params, jnp.asarray(noise.transpose(0, 2, 3, 1))), decoded)
    return gaps


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else REPO
    revision = sys.argv[2] if len(sys.argv) > 2 else REVISION
    print(json.dumps({"repo": repo, "revision": revision, "gaps": main(repo, revision)}, indent=1))
