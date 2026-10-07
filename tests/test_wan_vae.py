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
import math
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import save_file

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.wan import (
    WanRMSNorm,
    chunked_decode,
    chunked_moments,
    load_wan_vae,
    wan_vae_fields,
    wan_vae_path,
)

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wan-vae")
    with tarfile.open(ROOT / "tests/fixtures/wan_vae.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module")
def reference(source):
    return dict(np.load(source / "wan_vae.npz"))


@pytest.fixture(scope="module")
def loaded(source):
    return load_wan_vae(source)


def channels_last(array):
    return np.asarray(array).transpose(0, 2, 3, 4, 1)


def scaled_gap(actual, expected) -> float:
    expected = np.asarray(expected, np.float64)
    difference = np.abs(np.asarray(actual, np.float64) - expected).max()
    return float(difference / max(1.0, float(np.abs(expected).max())))


def parameter_gaps(layouts, gradients, reference, walk: str) -> dict[str, float]:
    return {layout.name: scaled_gap(layout.export({"autoencoder": gradients}),
                                    reference[f"{walk}.grad.{layout.name.removeprefix('vae/')}"])
            for layout in layouts}


def posterior(model, params, video):
    mean, log_variance = jnp.split(model.apply({"params": params}, video, method=model.moments), 2, axis=-1)
    return mean, jnp.exp(0.5 * jnp.clip(log_variance, -30.0, 20.0))


def test_every_published_tensor_is_mapped_and_exports_bit_identical(source, loaded):
    _, params, layouts, _ = loaded
    tensors = component_tensors(source, "vae")
    assert {layout.name for layout in layouts} == {f"vae/{name}" for name in tensors}
    for layout in layouts:
        written = layout.export({"autoencoder": params})
        published = tensors[layout.name.removeprefix("vae/")]
        assert written.dtype == published.dtype and written.shape == published.shape, layout.name
        assert written.tobytes() == published.tobytes(), layout.name


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


@pytest.mark.parametrize("frames", [3, 5])
def test_a_chunked_walk_is_the_whole_clip_as_exactly_as_float32_allows(loaded, reference, frames):
    """The decode one latent frame at a time and the encode the first frame
    then four at a time, each convolution's last frames carried in the cache,
    against the one-pass walk: no further from the float64 one-pass walk
    than the float32 one-pass walk is. Observed RMS ratios at 3 and 5 latent
    frames: decode 1.017 and 0.978, encode 0.976 and 0.825; the chunked and
    one-pass walks lie at most 1.5e-6 (decode) and 4.9e-7 (encode) apart."""
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    rng = np.random.default_rng(frames)
    latents = (channels_last(reference["latent"]) if frames == 3
               else rng.standard_normal((1, frames, 4, 6, model.latent)).astype(np.float32))
    video = (channels_last(reference["video"]) if frames == 3
             else rng.uniform(-1, 1, (1, 4 * frames - 3, 32, 48, 3)).astype(np.float32))
    whole = {"decode": model.apply({"params": params}, latents, method=model.decode),
             "moments": model.apply({"params": params}, video, method=model.moments)}
    chunked = {"decode": chunked_decode(model, {"params": params}, latents),
               "moments": chunked_moments(model, {"params": params}, video)}
    with jax.enable_x64(new_val=True):
        wide, wide_params = widened(model, params)
        variables = {"params": wide_params}
        truth = {"decode": wide.apply(variables, latents.astype(np.float64), method=wide.decode),
                 "moments": wide.apply(variables, video.astype(np.float64), method=wide.moments)}
        truth = {name: np.asarray(value) for name, value in truth.items()}
    for name in whole:
        assert chunked[name].shape == whole[name].shape
        assert truth[name].dtype == np.float64
        assert_as_exact_as_the_reference(np.asarray(chunked[name]), np.asarray(whole[name]), truth[name],
                                         f"chunked {name}, {frames} latent frames")


def test_the_autoencoder_walks_in_chunks_and_holds_one_chunk(loaded):
    """`WanAutoencoder` decodes in chunks, and what a chunked decode holds
    beyond the pixels it returns does not grow with the clip: from 3 latent
    frames to 17, XLA's temporaries grow by less than twice the added
    pixels (the scan's stacked frames and their join), where the one-pass
    decode's grow by its activations, many times that."""
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
    added_pixels = 4 * (17 - 3) * 128 * 192 * 3 * 4
    assert chunked[17] - chunked[3] < 2 * added_pixels, (chunked, one_pass)
    assert one_pass[17] - one_pass[3] > 10 * added_pixels, (chunked, one_pass)
    latents = np.random.default_rng(0).standard_normal((1, 3, 4, 6, model.latent)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(autoencoder.decode_video(params, latents)),
                                  np.asarray(jax.jit(lambda p, z: chunked_decode(model, {"params": p}, z))(
                                      params, latents)))


@pytest.mark.network
def test_the_published_vae_matches_the_source():
    """`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`'s VAE, downloaded, encodes a smooth
    nine-frame 128x192 clip and decodes its posterior mean as diffusers'
    `AutoencoderKLWan` does."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
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


def test_encoder_gradients_match_the_source(loaded, reference):
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model
    probe_mean = channels_last(reference["probe_mean"])
    probe_std = channels_last(reference["probe_std"])

    def objective(params, video):
        mean, std = posterior(model, params, video)
        return jnp.sum(mean * probe_mean) + jnp.sum(std * probe_std)

    grad_params, grad_video = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["video"]))
    video_gap = scaled_gap(grad_video, channels_last(reference["encode.grad_video"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "encode")
    worst = max(gaps, key=gaps.get)
    print(f"encode: video gradient gap {video_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert video_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_decoder_gradients_match_the_source(loaded, reference):
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model
    probe = channels_last(reference["probe"])

    def objective(params, latent):
        return jnp.sum(model.apply({"params": params}, latent, method=model.decode) * probe)

    grad_params, grad_latent = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["latent"]))
    latent_gap = scaled_gap(grad_latent, channels_last(reference["decode.grad_latent"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "decode")
    worst = max(gaps, key=gaps.get)
    print(f"decode: latent gradient gap {latent_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert latent_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_a_bfloat16_walk_matches_the_sources_bfloat16_walk(source):
    """The fixture VAE in bfloat16 against diffusers 0.34.0's own bfloat16
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


def test_a_zero_row_normalizes_to_zero_with_a_finite_gradient():
    """`F.normalize` clamps the norm at 1e-12, so a zero row and one far
    below the clamp both divide by 1e-12: the zero row stays zero, and either
    row's gradient is the probe times sqrt(C) * gamma / 1e-12, finite, as
    torch's is. Clamping the norm after its square root would leave sqrt's
    derivative at zero, NaN through the clamp, in both."""
    features = 16
    rows = jnp.stack([jnp.zeros(features), jnp.full(features, 1e-20)])
    gamma = jnp.linspace(0.5, 1.5, features)
    probe = jnp.linspace(-1.0, 1.0, 2 * features).reshape(2, features)
    output, pullback = jax.vjp(lambda x: WanRMSNorm(features, 1).apply({"params": {"gamma": gamma}}, x), rows)
    (gradient,) = pullback(probe)
    clamped = math.sqrt(features) * np.asarray(gamma) / 1e-12
    np.testing.assert_array_equal(np.asarray(output[0]), 0.0)
    np.testing.assert_allclose(np.asarray(output[1]), np.asarray(rows[1]) * clamped, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(probe) * clamped, rtol=1e-6)


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


def test_a_video_of_the_wrong_length_is_refused(loaded):
    autoencoder, params, _, _ = loaded
    with pytest.raises(ValueError, match="1 \\+ 4k frames"):
        autoencoder.encode(params, np.zeros((1, 4, 32, 48, 3), np.float32))


def test_a_missing_tensor_is_refused(source, tmp_path):
    shutil.copytree(source / "vae", tmp_path / "vae")
    tensors = component_tensors(source, "vae")
    del tensors["decoder.up_blocks.0.upsamplers.0.time_conv.weight"]
    save_file(tensors, tmp_path / "vae" / "diffusion_pytorch_model.safetensors")
    with pytest.raises(ValueError, match="time_conv"):
        load_wan_vae(tmp_path)


@pytest.mark.parametrize("name, ndim", [
    ("encoder.down_blocks.0.norm1.gamma", 5),
    ("decoder.conv_in.weight", 3),
    ("vae.encoder.conv_in.weight", 5),
    ("encoder.conv_in.scale", 1),
])
def test_an_unknown_tensor_is_refused(name, ndim):
    with pytest.raises(ValueError):
        wan_vae_path(name, ndim)


@pytest.mark.parametrize("change", [
    {"attn_scales": [1.0]}, {"dropout": 0.1}, {"is_residual": True}, {"patch_size": 2},
    {"temperal_downsample": [False, True]},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "vae" / "config.json").read_text())
    wan_vae_fields(config)
    with pytest.raises(ValueError):
        wan_vae_fields({**config, **change})


def test_a_video_run_denoises_wan_latents_and_samples_whole_clips(source):
    """A `VideoDataset` run behind the Wan VAE denoises 1 + k latent frames
    for clips of 1 + 4k, trains, and samples clips of the length it read."""
    import optax
    from diffusion_stubs import STUB_TEXT

    from dew.config import ModelConfig, TrainerConfig
    from dew.data import Dataset, VideoDataset
    from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder, TextCondition
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "video_dit",
            {"patch_size": 1, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1,
             "dtype": "float32", "attention_impl": "reference"},
        ),
        data=VideoDataset(frame_size=32, frames=9),
        trainer=TrainerConfig(batch_size=8, steps=1),
        solver=Euler(),
        sampling_steps=2,
        val_metrics=(),
        text=TextCondition(encoder=STUB_TEXT, checkpoint="stub-clip"),
        autoencoder=PretrainedAutoencoder(modelname=str(source), revision="main", dtype="float32"),
    )
    objective = config.build()
    assert objective.latent_shape == (3, 4, 4, 4)
    clips = (np.random.default_rng(0).random((8, 9, 32, 32, 3)) * 255).astype(np.uint8)
    batch = {
        "video": clips,
        "text": objective.inputs.conditions["textcontext"].encoder.tokenize(list("abcdefgh")),
    }
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0)).fit(
        Dataset(train=lambda partition: iter(lambda: batch, None), val=None, records=None, batch=8),
        steps=1, log_every=100)
    sampled = objective.pipeline(state)(["a", "b"], key=0).host()
    assert sampled.images.shape == (2, 9, 32, 32, 3)
