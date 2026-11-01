"""Native Wan 2.1 video VAE against the actual `AutoencoderKLWan`.

`tools/diffusers_wan_vae_reference.py` builds a tiny instance with the
published architecture's controls (four levels, time halved at the second
and third downsampling, the mid-block attention), saves it with
`save_pretrained`, and runs it as `WanPipeline` does: the encoder over a
nine-frame video in the source's causal chunks, the decoder one latent frame
at a time. It records in float32 the posterior of a rectangular batch, the
decode of a fixed three-frame latent, the vector-Jacobian products of both
against fixed probes for the input and every parameter, and the posterior
of the first frame alone.

Arrays are recorded `[B, C, T, H, W]`; the tests move channels last.
Parameter gradients are written back into the source layout through the
same `WeightLayout`s an export uses. Every gap is scaled by max(1, |reference|).
"""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import (
    assert_coder_vjps,
    assert_layout_tensors,
    channels_last,
    extract_fixture,
    fixture_arrays,
    posterior_of,
    scaled_gap,
)
from reference_error import assert_as_exact

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.wan import chunked_decode, chunked_moments, load_wan_vae

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/wan_vae.tar.xz", tmp_path_factory.mktemp("wan-vae"))


@pytest.fixture(scope="module")
def reference(source):
    return fixture_arrays(source / "wan_vae.npz")


@pytest.fixture(scope="module")
def loaded(source):
    return load_wan_vae(source)


def posterior(model, params, video):
    return posterior_of(model.apply({"params": params}, video, method=model.moments))


def test_a_configs_training_dropout_loads_the_same_autoencoder(source, loaded, tmp_path):
    """`dropout` acts in a training forward alone, which a frozen autoencoder
    never runs: a config that states one loads the same model and weights."""
    copy = shutil.copytree(source, tmp_path / "source")
    config = json.loads((copy / "vae" / "config.json").read_text())
    (copy / "vae" / "config.json").write_text(json.dumps({**config, "dropout": 0.1}))
    autoencoder, params, _, _ = load_wan_vae(copy)
    assert autoencoder.model == loaded[0].model
    for got, want in zip(jax.tree.leaves(params), jax.tree.leaves(loaded[1]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_every_published_tensor_is_mapped_and_exports_bit_identical(source, loaded):
    _, params, layouts, _ = loaded
    tensors = component_tensors(source, "vae")
    assert_layout_tensors(layouts, {"autoencoder": params}, tensors, "vae/")


def test_the_posterior_of_a_video_matches_the_source(loaded, reference):
    """Nine frames encode to three latent frames: the first alone, then four
    at a time, each reading the frames before it."""
    autoencoder, params, _, _ = loaded
    mean, std = posterior(autoencoder.model, params, channels_last(reference["video"]))
    assert mean.shape == (2, 3, 4, 6, 4)
    gaps = {"mean": scaled_gap(mean, channels_last(reference["mean"])),
            "std": scaled_gap(std, channels_last(reference["std"]))}
    print(f"posterior gaps {gaps}")
    assert max(gaps.values()) < FORWARD, gaps


def test_an_image_is_the_one_frame_video(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    first = channels_last(reference["video"])[:, :1]
    latent = model.apply({"params": params}, first, method=model.encode)
    assert scaled_gap(latent, channels_last(reference["image_mean"])) < FORWARD
    # The batch path runs the model jitted and the reference path op by op;
    # XLA fuses them differently, a float32 rounding apart.
    np.testing.assert_allclose(
        np.asarray(autoencoder.encode_batch(params, first[:, 0])), np.asarray(latent[:, 0]), rtol=0, atol=1e-6
    )


def test_the_decode_matches_the_source(loaded, reference):
    """Three latent frames decode to nine: the first to one, each later one
    to four."""
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    pixels = model.apply({"params": params}, channels_last(reference["latent"]), method=model.decode)
    assert pixels.shape == (2, 9, 32, 48, 3)
    gap = scaled_gap(pixels, channels_last(reference["pixels"]))
    print(f"decode gap {gap:.3g}")
    assert gap < FORWARD
    assert float(jnp.abs(pixels).max()) <= 1.0


def widened(model, params):
    """The model and its parameters in float64, for the truth the rule
    measures from (call under x64)."""
    return model.clone(dtype=jnp.float64), jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), params)


CHUNKED_SEEDS = 8
"""The chunked walk's declared draws (check 1), sized by `seeds_for` paired with the chunked walk on
armada."""


@pytest.mark.parametrize("frames", [3, 5])
def test_a_chunked_walk_is_the_whole_clip_as_exactly_as_float32_allows(loaded, frames):
    """The decode one latent frame at a time and the encode the first frame then four at a time, each
    convolution's last frames carried in the cache, against the one-pass walk: on par with it over drawn
    latents and clips (check 1), truth the float64 one-pass walk."""
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    variables = {"params": params}

    def draw(seed):
        rng = np.random.default_rng([frames, seed])
        return (rng.standard_normal((1, frames, 4, 6, model.latent)).astype(np.float32),
                rng.uniform(-1, 1, (1, 4 * frames - 3, 32, 48, 3)).astype(np.float32))

    def chunked(latents, video):
        return {"decode": chunked_decode(model, variables, latents),
                "moments": chunked_moments(model, variables, video)}

    def whole(latents, video):
        return {"decode": model.apply(variables, latents, method=model.decode),
                "moments": model.apply(variables, video, method=model.moments)}

    def truth(latents, video):
        wide, wide_params = widened(model, params)
        return {"decode": wide.apply({"params": wide_params}, latents, method=wide.decode),
                "moments": wide.apply({"params": wide_params}, video, method=wide.moments)}

    assert_as_exact(chunked, whole, draw, f"chunked walk, {frames} latent frames", seeds=CHUNKED_SEEDS,
                    truth_fn=truth)


def test_the_autoencoder_walks_in_chunks_and_holds_one_chunk(loaded):
    """`WanAutoencoder` decodes in chunks, and what a chunked decode holds
    beyond the pixels it returns does not grow with the clip: from 3 latent
    frames to 17, XLA's temporaries grow by less than twice the added
    pixels (the scan's stacked frames and their join), where the one-pass
    decode's grow by its activations: at least the full-resolution
    activation its last convolution reads over the added frames, which the
    one pass holds whole and a chunk never does. That much and no more is
    certain on every backend; how many more activations XLA keeps live
    depends on the convolution algorithms autotuning picks (an A100 under
    load kept 8 such, CPU over 10)."""
    autoencoder, params, _, _ = loaded
    model = autoencoder.model

    def temporaries(decode, frames: int) -> int:
        latents = jax.ShapeDtypeStruct((1, frames, 16, 24, model.latent), jnp.float32)
        compiled = jax.jit(decode).lower(params, latents).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    def whole(params, latents):
        return model.apply({"params": params}, latents, method=model.decode)

    chunked = {frames: temporaries(autoencoder._decode, frames) for frames in (3, 17)}
    one_pass = {frames: temporaries(whole, frames) for frames in (3, 17)}
    added_frames = 4 * (17 - 3) * 128 * 192
    added_pixels = added_frames * 3 * 4
    activation = added_frames * model.base * model.multipliers[0] * 4
    assert 2 * added_pixels < activation, "the full-resolution width must exceed the bound it is set against"
    assert chunked[17] - chunked[3] < 2 * added_pixels, (chunked, one_pass)
    assert one_pass[17] - one_pass[3] > activation, (chunked, one_pass)
    latents = np.random.default_rng(0).standard_normal((1, 3, 4, 6, model.latent)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(autoencoder.decode_video(params, latents)),
                                  np.asarray(jax.jit(lambda p, z: chunked_decode(model, {"params": p}, z))(
                                      params, latents)))


@pytest.mark.network
def test_the_published_vae_matches_the_source():
    """`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`'s VAE, downloaded, encodes a smooth
    nine-frame 128x192 clip and decodes its posterior mean as diffusers'
    `AutoencoderKLWan` does."""
    import diffusers
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", allow_patterns=["vae/*"])
    t, y, x = np.mgrid[0:9, 0:128, 0:192] / np.array([4.0, 32.0, 32.0])[:, None, None, None]
    video = np.stack([np.sin(x + y + t), np.cos(2 * x - y - 0.5 * t), np.sin(3 * y + t) * np.cos(x)])[None]
    video = (0.8 * video + 0.05 * np.random.default_rng(0).standard_normal(video.shape)).astype(np.float32)
    with torch.no_grad():
        source = diffusers.AutoencoderKLWan.from_pretrained(path, subfolder="vae").eval()
        expected_mean = source.encode(torch.from_numpy(video)).latent_dist.mean
        expected_pixels = source.decode(expected_mean).sample.numpy()
        expected_mean = expected_mean.numpy()
    del source

    autoencoder, params, _, _ = load_wan_vae(path, jnp.float32)
    model = autoencoder.model
    mean = model.apply({"params": params}, channels_last(video), method=model.encode)
    pixels = model.apply({"params": params}, channels_last(expected_mean), method=model.decode)
    gaps = {"mean": scaled_gap(mean, channels_last(expected_mean)),
            "pixels": scaled_gap(pixels, channels_last(expected_pixels))}
    print(f"published VAE gaps {gaps}")
    assert mean.shape == (1, 3, 16, 24, 16)
    assert max(gaps.values()) < FORWARD, gaps


def test_the_coder_gradients_match_the_source(loaded, reference):
    assert_coder_vjps(loaded, reference, "video", posterior, GRADIENT)


def test_a_bfloat16_walk_matches_the_sources_bfloat16_walk(source):
    """The fixture VAE in bfloat16 against diffusers 0.39.0's own bfloat16
    walk of it (`wan_vae_bf16.npz`, the reference tool's `bfloat16` mode):
    the posterior and decode of one bfloat16 video and latent, and their
    input gradients. Each differs from the source by less than one bfloat16
    rounding of its scale on average, and by at most 2^-3 of it anywhere.

    The norm computes `F.normalize` in float32 and rounds once. Against this
    record its mean gaps are mean 1.87e-3, std 8.8e-4, pixels 3.28e-3, video
    gradient 7.6e-5 and latent gradient 1.28e-2, where the earlier bfloat16
    reduction gave 1.99e-3, 9.8e-4, 3.40e-3, 8.0e-5 and 1.73e-2. Averaged
    over four more seeds it is closer to both 0.34.0 and 0.40.0 (which
    normalizes in float32 too) on every output and on the encoder's
    parameter gradients.
    """
    record = dict(np.load(ROOT / "tests/fixtures/wan_vae_bf16.npz"))
    autoencoder, params, _, _ = load_wan_vae(source, jnp.bfloat16)
    model = autoencoder.model

    def walked(name):
        return jnp.asarray(channels_last(record[name]), jnp.bfloat16)

    def objective(video):
        mean, std = posterior(model, params, video)
        return (jnp.sum(mean.astype(jnp.float32) * channels_last(record["probe_mean"]))
                + jnp.sum(std.astype(jnp.float32) * channels_last(record["probe_std"])))

    def decoded(latent):
        return model.apply({"params": params}, latent, method=model.decode)

    def decode_objective(latent):
        return jnp.sum(decoded(latent).astype(jnp.float32) * channels_last(record["probe"]))

    mean, std = jax.jit(lambda video: posterior(model, params, video))(walked("video"))
    ours = {"mean": mean, "std": std, "pixels": jax.jit(decoded)(walked("latent")),
            "encode.grad_video": jax.jit(jax.grad(objective))(walked("video")),
            "decode.grad_latent": jax.jit(jax.grad(decode_objective))(walked("latent"))}
    for name, value in ours.items():
        expected = channels_last(record[name]).astype(np.float64)
        difference = np.abs(np.asarray(value, np.float64) - expected)
        scale = max(1.0, float(np.abs(expected).max()))
        print(f"{name}: mean gap {difference.mean():.3g}, scaled max {difference.max() / scale:.3g}")
        # 2^-8 is bfloat16's unit roundoff: within one rounding of the
        # output's scale on average. 2^-3 is twice the largest scaled gap any
        # output reached over four other seeds (6.1e-2, the latent gradient),
        # rounded up to a power of two. Normalizing over another axis, or
        # dropping the sqrt(C) scale, fails the first at every output.
        assert difference.mean() / scale < 2.0 ** -8, name
        assert difference.max() / scale < 2.0 ** -3, name


def test_the_autoencoder_normalizes_as_the_pipeline_does(loaded, reference):
    """`WanPipeline` stores `(z - latents_mean) * (1 / latents_std)` and
    decodes `z / (1 / latents_std) + latents_mean`, per channel, and a video
    keeps its time axis through both."""
    autoencoder, params, _, config = loaded
    mean = np.asarray(config["latents_mean"], np.float32)
    scale = 1.0 / np.asarray(config["latents_std"], np.float32)
    video = channels_last(reference["video"])
    assert autoencoder.latent_shape(video.shape[1:]) == (3, 4, 6, 4)
    assert autoencoder.latent_shape(video.shape[2:]) == (4, 6, 4)
    raw = np.asarray(autoencoder.encode_video(params, video))
    np.testing.assert_allclose(
        np.asarray(autoencoder.encode(params, video)), (raw - mean) * scale, rtol=1e-6, atol=1e-6
    )

    latent = channels_last(reference["latent"])
    decoded = np.asarray(autoencoder.decode(params, latent))
    assert decoded.shape == (2, 9, 32, 48, 3)
    # XLA may divide by a broadcast operand through its reciprocal, so the
    # latent the decoder reads is a float32 rounding from numpy's.
    np.testing.assert_allclose(decoded, np.asarray(autoencoder.decode_video(params, latent / scale + mean)),
                               rtol=0, atol=1e-5)
