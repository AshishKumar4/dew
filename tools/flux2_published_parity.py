"""Published FLUX.2 [klein] against Diffusers, in float32, one component at a time.

Loads each component of `black-forest-labs/FLUX.2-klein-4B` (or the repo and
revision given) with Diffusers and with Dew in turn - the Qwen3 encoder, the
transformer, the VAE - freeing each before the next, so the peak is one 4B
component in float32 (about 16 GB on the accelerator). It compares, each gap
divided by the larger of 1 and the reference's largest value:

- the prompt states of two prompts (the encoder's layers 9, 18 and 27,
  stacked, padded to 512 tokens);
- one transformer call at sigma 0.7 on fixed noise at 512x512;
- the VAE decode of that noise as the pipeline decodes its latent (the
  batch-norm statistics undone, the 2x2 fold unfolded).

The pipeline around them (template, guidance, grid, walk) is held to the
source by the tiny pipeline in `tests/test_flux2_source.py`.

    python tools/flux2_published_parity.py [REPO] [REVISION] > flux2_parity.json
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Both sides in true float32: an A100 otherwise multiplies float32 in TF32,
# torch's convolutions and JAX's matmuls alike, which moves a 4B model's
# output by 1e-3.
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
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
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dew.diffusion.process import DenoisingCondition
    from dew.inputs.diffusion import HiddenStatesConditioner
    from dew.interop import sources
    from dew.interop.diffusion_components import _denoiser, _diffusion_vae

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    directory = sources.snapshot(repo, revision, weights=("text_encoder", "transformer", "vae"))
    gaps: dict[str, float] = {}

    # The encoder, through the pipeline's own prompt encoding.
    text_encoder = AutoModelForCausalLM.from_pretrained(directory / "text_encoder", dtype=torch.float32).to(device)
    with torch.no_grad():
        expected = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder, AutoTokenizer.from_pretrained(directory / "tokenizer"), list(PROMPTS),
            device=device).cpu().numpy()
    del text_encoder
    free()
    encoder = HiddenStatesConditioner.from_pretrained(str(directory), dtype="float32", param_dtype="float32")
    gaps["context"] = gap(encoder.encode(encoder.params, encoder.tokenize(PROMPTS)).context, expected)
    del encoder
    gc.collect()

    # The transformer, on the first prompt's source states.
    vae_config = json.loads((directory / "vae" / "config.json").read_text())
    rows = SIZE // 2 ** len(vae_config["block_out_channels"])
    channels = 4 * vae_config["latent_channels"]
    noise = np.random.default_rng(0).standard_normal((1, channels, rows, rows)).astype(np.float32)
    transformer = Flux2Transformer2DModel.from_pretrained(directory / "transformer",
                                                          torch_dtype=torch.float32).to(device)
    latent = torch.from_numpy(noise).to(device)
    with torch.no_grad():
        context = torch.from_numpy(expected[:1]).to(device)
        prediction = transformer(
            hidden_states=Flux2KleinPipeline._pack_latents(latent), encoder_hidden_states=context,
            timestep=torch.full((1,), SIGMA, device=device), guidance=None,
            img_ids=Flux2KleinPipeline._prepare_latent_ids(latent)[0].to(device),
            txt_ids=Flux2KleinPipeline._prepare_text_ids(context)[0].to(device), return_dict=False)[0]
        prediction = prediction.cpu().numpy().reshape(1, rows, rows, -1)
    del transformer
    free()
    denoiser = _denoiser(directory, dtype="float32", attention_impl="auto")
    variables, _ = denoiser.weights("float32")
    native = denoiser.model.apply(variables, jnp.asarray(noise.transpose(0, 2, 3, 1)),
                                  jnp.full((1,), SIGMA * 1000.0), DenoisingCondition(jnp.asarray(expected[:1])))
    gaps["prediction"] = gap(native, prediction)
    del denoiser, variables, native
    gc.collect()

    # The VAE, decoding the noise as the pipeline decodes its latent.
    vae = AutoencoderKLFlux2.from_pretrained(directory / "vae", torch_dtype=torch.float32).to(device)
    with torch.no_grad():
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        raw = Flux2KleinPipeline._unpatchify_latents(latent * std + mean)
        decoded = vae.decode(raw).sample.cpu().numpy().transpose(0, 2, 3, 1)
    del vae
    free()
    autoencoder, params, _, _ = _diffusion_vae(directory, jnp.float32)
    gaps["decoded"] = gap(autoencoder.decode(params, jnp.asarray(noise.transpose(0, 2, 3, 1))), decoded)
    return gaps


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else REPO
    revision = sys.argv[2] if len(sys.argv) > 2 else REVISION
    gaps = main(repo, revision)
    import jax
    import torch

    precision = {"jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
                 "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                 "devices": [str(device) for device in jax.devices()]}
    print(json.dumps({"repo": repo, "revision": revision, "precision": precision, "gaps": gaps}, indent=1))
