"""Vendored Stable Diffusion VAE tests, plus the latent normalization seam.

Everything touching the pretrained weights is network-marked: they download
from the HuggingFace Hub on first run. Excluded in CI (-m "not network").
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.autoencoders import AutoEncoder


class IdentityAutoEncoder(AutoEncoder):
    """Latents are the input, so only the normalization seam is under test."""
    downscale_factor = 1
    latent_channels = 4
    name = "identity"

    def __init__(self, latent_shift=0.0, latent_scale=1.0):
        self.latent_shift = latent_shift
        self.latent_scale = latent_scale
        self.params = {}

    def encode_batch(self, params, x, key=None):
        return x

    def decode_batch(self, params, z):
        return z


def test_latent_normalization_defaults_to_the_identity(rng):
    autoencoder = IdentityAutoEncoder()
    x = jax.random.normal(rng, (2, 8, 8, 4))
    assert jnp.allclose(autoencoder.encode(autoencoder.params, x), x)
    assert jnp.allclose(autoencoder.decode(autoencoder.params, x), x)


@pytest.mark.parametrize("remote", [False, True])
def test_vae_reconstructs_metadata_without_reloading_supplied_weights(tmp_path, monkeypatch, remote):
    import json
    from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
    import dew.nn.autoencoders.vae as loader

    config = dict(block_out_channels=[8, 16], latent_channels=4, in_channels=3,
                  layers_per_block=1, norm_num_groups=4, use_quant_conv=False,
                  use_post_quant_conv=False, shift_factor=0.25, scaling_factor=0.5)
    (tmp_path / "config.json").write_text(json.dumps(config))
    name = str(tmp_path)
    if remote:
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.file_download import DryRunFileInfo

        wrong = tmp_path / "wrong.json"
        wrong.write_text(json.dumps({**config, "shift_factor": 99.0}))

        def download(repo_id, filename, *, revision=None, subfolder=None, dry_run=False):
            if filename == "config.json":
                return str(wrong if revision == "bf16" else tmp_path / "config.json")
            if not dry_run:
                raise AssertionError("supplied VAE parameters triggered a weight download")
            if revision == "bf16":
                raise EntryNotFoundError("this revision has config but no matching weights")
            return DryRunFileInfo(commit_hash="a" * 40, file_size=1, filename=filename,
                                 is_cached=False, local_path=str(tmp_path / filename), will_download=True)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        name = "fixture/vae"
    model = AutoencoderKL(channels=(8, 16), blocks_per_level=1, norm_groups=4,
                          quantize=False, post_quantize=False, dtype=jnp.float32)
    image = jnp.linspace(-0.5, 0.5, 8 * 8 * 3).reshape(1, 8, 8, 3)
    saved = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16),
                         model.init(jax.random.key(4), image)["params"])

    def forbid_weights(*args, **kwargs):
        raise AssertionError("supplied VAE params must not read source weights")

    monkeypatch.setattr(loader, "_read_vae_weights", forbid_weights)
    restored = StableDiffusionVAE(name, params=saved, dtype=jnp.bfloat16)
    expected = StableDiffusionVAE(model=model.clone(dtype=jnp.bfloat16), params=saved,
                                 dtype=jnp.bfloat16, latent_shift=0.25, latent_scale=0.5)
    actual_latent = restored.encode(restored.params, image)
    expected_latent = expected.encode(saved, image)
    np.testing.assert_array_equal(actual_latent, expected_latent)
    np.testing.assert_array_equal(restored.decode(restored.params, actual_latent),
                                  expected.decode(saved, expected_latent))
    for actual, original in zip(jax.tree.leaves(restored.params), jax.tree.leaves(saved), strict=True):
        assert actual.dtype == original.dtype
        np.testing.assert_array_equal(actual, original)



@pytest.mark.parametrize("shape", [(2, 8, 8, 4), (2, 3, 8, 8, 4)])
def test_latent_normalization_shifts_and_scales_roundtrip(rng, shape):
    """SD3-style shift+scale: latents come out centred and rescaled, and
    decoding inverts it exactly, for images and for video."""
    autoencoder = IdentityAutoEncoder(latent_shift=0.25, latent_scale=4.0)
    x = jax.random.normal(rng, shape)
    latent = autoencoder.encode(autoencoder.params, x)
    assert jnp.allclose(latent, (x - 0.25) * 4.0, atol=1e-6)
    assert jnp.allclose(autoencoder.decode(autoencoder.params, latent), x, atol=1e-5)


@pytest.fixture(scope="module")
def vae():
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
    return StableDiffusionVAE(dtype=jnp.float32)


@pytest.mark.network
def test_vae_uses_the_latent_normalization_seam(vae):
    """The SD scaling factor rides on the shared seam, so there is one
    normalization path a caller can override with dataset statistics."""
    assert vae.latent_scale == pytest.approx(0.18215)
    assert vae.latent_shift == 0.0


@pytest.mark.network
def test_vae_roundtrip_reconstructs(vae, rng):
    # A smooth image should survive the encode/decode roundtrip well
    ramp = jnp.linspace(-0.8, 0.8, 64)
    x = jnp.broadcast_to(ramp[None, :, None, None], (1, 64, 64, 3)).transpose(0, 2, 1, 3)
    rec = vae.decode(vae.params, vae.encode(vae.params, x))
    mse = float(jnp.mean((rec - x) ** 2))
    psnr = 10 * np.log10(4.0 / mse)
    assert psnr > 20, f"reconstruction too poor: {psnr:.1f}dB"
