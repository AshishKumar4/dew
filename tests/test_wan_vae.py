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
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.wan import load_wan_vae, wan_vae_fields, wan_vae_path

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
    np.testing.assert_allclose(np.asarray(autoencoder.encode_batch(params, first[:, 0])), np.asarray(latent[:, 0]),
                               rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(np.asarray(autoencoder.encode(params, video)), (raw - mean) * scale, rtol=1e-6, atol=1e-6)

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
    from test_diffusion_objective import StubText  # noqa: F401  registers "stub_text"

    from dew.config import ModelConfig, TrainerConfig
    from dew.data import Dataset, VideoDataset
    from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder, TextCondition
    from dew.registry import samplers
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig("video_dit", dict(patch_size=1, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
                          dtype="float32", attention_impl="reference"),
        data=VideoDataset(frame_size=32, frames=9), trainer=TrainerConfig(batch_size=8, steps=1),
        sampler=samplers.Euler(), sampling_steps=2, val_metrics=(),
        text=TextCondition(encoder="stub_text", checkpoint="stub-clip"),
        autoencoder=PretrainedAutoencoder(modelname=str(source), revision="main", dtype="float32"))
    objective = config.build()
    assert objective.latent_shape == (3, 4, 4, 4)
    clips = (np.random.default_rng(0).random((8, 9, 32, 32, 3)) * 255).astype(np.uint8)
    batch = {"video": clips, "text": objective.inputs.conditions["textcontext"].encoder.tokenize(list("abcdefgh"))}
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0)).fit(
        Dataset(train=lambda partition: iter(lambda: batch, None), val=None, records=None, batch=8),
        steps=1, log_every=100)
    sampled = objective.pipeline(state)(["a", "b"], seed=0).host()
    assert sampled.images.shape == (2, 9, 32, 32, 3)
