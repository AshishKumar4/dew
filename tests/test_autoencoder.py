"""The AutoEncoder contract, over a tiny randomly initialized AutoencoderKL.

Nothing here asserts reconstruction quality: what matters is the contract the
samplers and input config depend on (the advertised latent geometry, video
flattening, and the latent normalization seam). The Diffusers tests hold that
contract to Diffusers' own VAE, posterior and SD3 pipeline normalization, run
here over drawn clips against its float64 twin (check 1).
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.nn.autoencoders import AutoEncoder, AutoencoderKL, LatentNormalization, StableDiffusionVAE
from dew.nn.autoencoders.api import ModuleAutoEncoder
from dew.nn.autoencoders.pretrained import REGISTRATIONS
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Recorder, Step
from dew.registry import to_record

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vae"

IMAGE_SIZE = 8


def tiny_vae(channels=(8, 16, 16), latent_shift=0.0, latent_scale=1.0, params=None):
    model = AutoencoderKL(channels=channels, blocks_per_level=1, norm_groups=4, dtype=jnp.float32)
    if params is None:
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)))["params"]
    return StableDiffusionVAE(model=model, params=params, dtype=jnp.float32,
                              latent_shift=latent_shift, latent_scale=latent_scale)


@pytest.fixture(scope="module")
def autoencoder():
    return tiny_vae()


@pytest.fixture(scope="module")
def image():
    return jax.random.uniform(
        jax.random.PRNGKey(1), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )


@pytest.mark.parametrize("channels", [(8,), (8, 16), (8, 16, 16)])
def test_the_advertised_geometry_is_the_encoders(channels):
    """`downscale_factor` and `latent_channels` are what the samplers and the
    input config size latents by, so they must be what the encoder produces
    and the decoder takes back to the image."""
    autoencoder = tiny_vae(channels)
    size = 2 * autoencoder.downscale_factor
    image = jnp.zeros((1, size, size, 3))
    latent = autoencoder.encode(autoencoder.params, image)
    assert latent.shape == (1, 2, 2, autoencoder.latent_channels)
    assert jnp.all(jnp.isfinite(latent))
    other = autoencoder.encode(autoencoder.params, jnp.ones_like(image))
    assert not jnp.allclose(latent, other, atol=1e-6)
    decoded = autoencoder.decode(autoencoder.params, latent)
    assert decoded.shape == image.shape
    assert jnp.all(jnp.isfinite(decoded))
    # A decoder that dropped its latent, for a bias or a lost residual, would
    # keep every shape in this file and reconstruct the same image regardless.
    assert not jnp.allclose(
        decoded, autoencoder.decode(autoencoder.params, jnp.ones_like(latent)), atol=1e-6)


def test_latent_normalization_is_inverted_by_decode(autoencoder, image):
    """encode applies (z - shift) * scale and decode must undo exactly it, so
    decode(encode(x)) is the raw decoder applied to the raw latent."""
    normalized = tiny_vae(latent_shift=0.3, latent_scale=2.5, params=autoencoder.params)
    raw_latent = autoencoder.encode(autoencoder.params, image)  # identity normalization
    latent = normalized.encode(normalized.params, image)
    assert jnp.allclose(latent, (raw_latent - 0.3) * 2.5, atol=1e-5)
    assert jnp.allclose(normalized.decode(normalized.params, latent),
                        autoencoder.decode(autoencoder.params, raw_latent), atol=1e-5)


@pytest.fixture(scope="module")
def sd3_tiny():
    return StableDiffusionVAE(str(FIXTURES / "sd3-tiny"), dtype=jnp.float32)


CLIP = (2, 3, 16, 16)
"""The clips the Diffusers comparisons draw: batch, frames, height, width."""

DIFFUSERS_SEEDS = 8
"""Each Diffusers comparison's declared draws (check 1), sized by `seeds_for` paired with Dew's VAE on the
CPU."""


@pytest.fixture(scope="module")
def diffusers_sd3_tiny():
    """Diffusers' own `AutoencoderKL` of the sd3-tiny fixture in float32, and its float64 twin."""
    from diffusers import AutoencoderKL

    from tools.torch_precision import float64_twin

    narrow = AutoencoderKL.from_pretrained(FIXTURES / "sd3-tiny").eval()
    return narrow, float64_twin(narrow)


def _channels_first(value: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(value, -1, 1)))


def _channels_last(value) -> np.ndarray:
    return np.moveaxis(value.detach().numpy(), 1, -1)


def test_a_clip_samples_and_normalizes_as_diffusers_does_frame_by_frame(sd3_tiny, diffusers_sd3_tiny):
    """A clip's frames encoded as one `[B, T]` batch draw the latent SD3's img2img pipeline draws for each
    frame as an image (`retrieve_latents`: Diffusers' posterior's `sample`, its mean plus its deviation
    times a standard normal draw, then `(z - shift) * scale`), from the standard normal draw Dew's encoder
    takes of the same key, and their normalized means are its: on par (check 1) with Diffusers' own VAE
    and posterior over drawn clips and keys, from its float64 twin's."""
    import torch

    from tools.torch_precision import float64

    batch, frames, height, width = CLIP

    def draw(seed):
        clip = np.random.default_rng(seed).uniform(-1, 1, (batch, frames, height, width, 3))
        return clip.astype(np.float32), np.int32(seed)

    def ours(clip, seed):
        sampled = sd3_tiny.encode(sd3_tiny.params, clip, key=jax.random.key(seed))
        mean = sd3_tiny.encode(sd3_tiny.params, clip)
        return sampled.reshape(-1, *sampled.shape[2:]), mean.reshape(-1, *mean.shape[2:])

    # A clip's latents keep its batch and frame axes.
    clip, seed = draw(0)
    assert sd3_tiny.encode(sd3_tiny.params, clip, key=jax.random.key(seed)).shape[:2] == (batch, frames)

    def theirs(model, precision, dtype):
        config = model.config

        @as_given
        def run(clip, seed):
            with torch.no_grad(), precision():
                pixels = _channels_first(clip.reshape(-1, height, width, 3)).to(dtype)
                posterior = model.encode(pixels).latent_dist
                # Dew's draw for the key, channels last over the clip's frames, as its encoder takes it.
                epsilon = jax.random.normal(jax.random.key(seed), _channels_last(posterior.mean).shape)
                sampled = posterior.mean + posterior.std * _channels_first(np.asarray(epsilon)).to(dtype)
                return (_channels_last((sampled - config.shift_factor) * config.scaling_factor),
                        _channels_last((posterior.mean - config.shift_factor) * config.scaling_factor))

        return run

    narrow, wide = diffusers_sd3_tiny
    assert_as_exact(ours, theirs(narrow, contextlib.nullcontext, torch.float32), draw,
                    "a clip's sampled and mean latents", seeds=DIFFUSERS_SEEDS,
                    truth_fn=theirs(wide, float64, torch.float64))


def test_a_clip_decodes_as_diffusers_decodes_its_frames(sd3_tiny, diffusers_sd3_tiny):
    """Normalized latents of a clip decode to what Diffusers' VAE decodes of each frame's latent taken
    back SD3's way (`z / scale + shift`): on par (check 1) with it over drawn latents, from its float64
    twin's."""
    import torch

    from tools.torch_precision import float64

    batch, frames, height, width = CLIP
    latent = sd3_tiny.encode(sd3_tiny.params, jnp.zeros((1, 1, height, width, 3), jnp.float32)).shape[2:]

    def draw(seed):
        return (np.random.default_rng(seed).standard_normal((batch, frames, *latent)).astype(np.float32),)

    def ours(latents):
        decoded = sd3_tiny.decode(sd3_tiny.params, latents)
        return decoded.reshape(-1, *decoded.shape[2:])

    def theirs(model, precision, dtype):
        config = model.config

        @as_given
        def run(latents):
            with torch.no_grad(), precision():
                taken = _channels_first(latents.reshape(-1, *latent)).to(dtype)
                taken = taken / config.scaling_factor + config.shift_factor
                return _channels_last(model.decode(taken).sample)

        return run

    narrow, wide = diffusers_sd3_tiny
    assert_as_exact(ours, theirs(narrow, contextlib.nullcontext, torch.float32), draw,
                    "a clip's decoded frames", seeds=DIFFUSERS_SEEDS,
                    truth_fn=theirs(wide, float64, torch.float64))


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"

ARCHIVES = {
    "dc_ae": ("dc_ae.tar.xz", "conv"),
    "wan": ("wan_vae.tar.xz", ""),
    "rae": ("rae.tar.xz", "dinov2"),
    "flux2": ("flux2_source.tar.xz", "vae"),
    "qwen_image": ("qwen_image_vae.tar.xz", ""),
}
"""Each registered autoencoder's fixture but the SD VAE's: its archive and the folder in it."""


def _published(kind: str, directory: Path):
    """A built-in autoencoder and its parameters, built from its fixture at `directory` as a run's
    `PretrainedAutoencoder` builds it: by the class the fixture's config names."""
    from interop_support import extract_fixture

    from dew.nn.autoencoders.pretrained import PretrainedAutoencoder

    archive, inside = ARCHIVES[kind]
    source = extract_fixture(FIXTURE_ROOT / archive, directory) / inside
    # A scalar normalization takes a run's own statistics, which its record must keep.
    overrides = {"latent_shift": 0.25, "latent_scale": 0.5} if kind == "dc_ae" else {}
    autoencoder = PretrainedAutoencoder(modelname=str(source), dtype="float32", **overrides).build()
    return autoencoder, autoencoder.params


RECORDS = ("sd_vae", *ARCHIVES)
"""Every built-in autoencoder, by the fixture it is built from: the SD VAE's `tiny_vae` and the others'
archives (`_published`)."""


def test_every_registration_is_one_autoencoder_with_a_fixture():
    """No wrapper or published class registers twice, and every registered autoencoder has its fixture
    here."""
    published = [name for entry in REGISTRATIONS for name in entry.published]
    assert len(set(published)) == len(published)
    assert len({entry.wrapper for entry in REGISTRATIONS}) == len(REGISTRATIONS) == len(RECORDS)


@pytest.mark.parametrize("kind", RECORDS)
def test_every_built_in_normalizes_by_its_records_shift_and_factor_bit_for_bit(kind, tmp_path):
    """Each built-in autoencoder, the KL VAE and a published RAE among them, normalizes its encoder's
    latent by its record's shift and factor, `(z - shift) * factor` or, one that divides, over the
    factor, and decode restores the inverse before its decoder: each bit for bit the formula written out
    on the same draw, as the normalization applied it before its traced map (`LatentAffine`) held it."""
    if kind == "sd_vae":
        autoencoder = tiny_vae(latent_shift=0.3, latent_scale=2.5)
        params = autoencoder.params
    else:
        autoencoder, params = _published(kind, tmp_path)
    side = 2 * autoencoder.downscale_factor
    frames = (5,) if kind == "wan" else ()
    shape = (*frames, side, side, getattr(autoencoder.model, "image_channels", 3))
    image = jax.random.uniform(jax.random.key(0), (2, *shape), minval=-1.0, maxval=1.0)
    raw = autoencoder.encode_video(params, image) if frames else autoencoder.encode_batch(params, image)
    normalization = autoencoder.normalization
    shift, factor = (jnp.asarray(value, raw.dtype) for value in (normalization.shift, normalization.factor))
    np.testing.assert_array_equal(autoencoder.encode(params, image),
                                  (raw - shift) / factor if normalization.divides else (raw - shift) * factor)
    z = jax.random.normal(jax.random.key(1), raw.shape, raw.dtype)
    restored = z * factor + shift if normalization.divides else z / factor + shift
    decode = autoencoder.decode_video if frames else autoencoder.decode_batch
    np.testing.assert_array_equal(autoencoder.decode(params, z), decode(params, restored))


@pytest.mark.parametrize("kind", RECORDS)
def test_a_stochastic_encoder_samples_at_its_given_noise_and_a_deterministic_one_refuses_it(kind, tmp_path):
    """A training step hands each example's posterior draw to the encoder as `noise`, in the latent's
    own layout: a stochastic encoder's latent is its posterior's sample at it, its mean at zero noise
    whatever the key and another latent at other noise; a deterministic one (DC-AE, RAE) has no posterior
    to draw and refuses noise, and a latent objective over it draws none (`AutoEncoder.stochastic`)."""
    from dew.objectives.diffusion import DiffusionObjective

    if kind == "sd_vae":
        autoencoder = tiny_vae()
        params = autoencoder.params
    else:
        autoencoder, params = _published(kind, tmp_path)
    side = 2 * autoencoder.downscale_factor
    frames = (5,) if kind == "wan" else ()
    shape = (*frames, side, side, getattr(autoencoder.model, "image_channels", 3))
    image = jax.random.uniform(jax.random.key(0), (2, *shape), minval=-1.0, maxval=1.0)
    noise = jax.random.normal(jax.random.key(1), (2, *autoencoder.latent_shape(shape)), jnp.float32)
    assert autoencoder.stochastic is (kind not in ("dc_ae", "rae"))
    if not autoencoder.stochastic:
        with pytest.raises(TypeError):
            autoencoder.encode(params, image, noise=noise)
        recorder = Recorder()
        objective = DiffusionObjective(SimpleDiT(output_channels=autoencoder.latent_channels, patch_size=1,
                                                 emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
                                       presets.EDM(regime="latent")(),
                                       InputSpec(Field("image", shape)), autoencoder=autoencoder)
        pixels = np.asarray(jax.random.randint(jax.random.key(2), image.shape, 0, 256), np.uint8)
        step = Step(jnp.asarray(0), jax.random.key(3), None, example_key=jax.random.key(4),
                    example_index=jnp.arange(2, dtype=jnp.int32), example_count=2, recorder=recorder)
        objective.loss(objective.init(jax.random.key(5)), {"image": pixels}, step)
        assert "times" in recorder.values and "posterior" not in recorder.values
        return
    mean = autoencoder.encode(params, image)
    np.testing.assert_array_equal(autoencoder.encode(params, image, jax.random.key(6), jnp.zeros_like(noise)),
                                  mean)
    sampled = autoencoder.encode(params, image, jax.random.key(6), noise)
    np.testing.assert_array_equal(autoencoder.encode(params, image, jax.random.key(7), noise), sampled)
    assert not np.array_equal(sampled, mean)


class Plain(ModuleAutoEncoder[AutoencoderKL]):
    """A user's autoencoder with nothing of its own to record: it inherits the generic constructor."""

    def record_arguments(self) -> dict:
        return {}


class Halved(ModuleAutoEncoder[AutoencoderKL]):
    """A user's autoencoder, no built-in: a native AutoencoderKL whose latents it scales by `factor`,
    recorded with that factor (`record_arguments`)."""

    def __init__(self, model: AutoencoderKL, params, factor: float):
        super().__init__(model, params)
        self.factor = factor
        self.normalization = LatentNormalization({"factor": factor}, 0.0, factor)

    def record_arguments(self) -> dict:
        return {"factor": self.factor}


@pytest.mark.parametrize("own", ["inherited constructor", "own constructor"])
def test_a_users_autoencoder_rebuilds_from_its_record_as_a_built_in_does(own):
    """An autoencoder of the user's own records itself as the registry records any component, its class
    by import path (`registry.to_record`), and rebuilds from that record around the run's parameters
    (`AutoEncoder.bound`), no registration needed: the rebuilt one encodes and decodes bitwise as the one it
    recorded, its normalization included. Its class and its module's survive JSON as import paths."""
    vae = tiny_vae()
    original = (Plain(vae.model, vae.params) if own == "inherited constructor"
                else Halved(vae.model, vae.params, factor=0.5))
    record = json.loads(json.dumps(to_record(original, AutoEncoder)))
    assert record["class"].endswith(f":{type(original).__name__}")
    assert record["fields"]["model"]["class"].endswith(":AutoencoderKL")
    rebuilt = AutoEncoder.bound(record, params=original.params)
    assert type(rebuilt) is type(original)
    assert dict(rebuilt.normalization.record) == dict(original.normalization.record)
    assert (rebuilt.latent_shift, rebuilt.latent_scale) == (original.latent_shift, original.latent_scale)
    image = jax.random.uniform(jax.random.key(0), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0)
    latent = original.encode(original.params, image)
    np.testing.assert_array_equal(rebuilt.encode(rebuilt.params, image), latent)
    np.testing.assert_array_equal(rebuilt.decode(rebuilt.params, latent),
                                  original.decode(original.params, latent))


def test_an_earlier_runs_autoencoder_record_reads_as_its_class_record():
    """An earlier run recorded a built-in autoencoder by its registration's name with its module's bare
    fields; the run's loader (`SavedRun.pinned`, `earlier_shapes`) reads it as the class record Dew
    writes now, which rebuilds the same autoencoder."""
    from dew.inference.tasks import earlier_shapes

    original = tiny_vae(latent_shift=0.25, latent_scale=0.5)
    record = to_record(original, AutoEncoder)
    earlier = {"name": "sd_vae", "fields": {**record["fields"], "model": record["fields"]["model"]["fields"]}}
    assert earlier_shapes({"autoencoder": earlier})["autoencoder"] == record


@pytest.mark.parametrize("kind", RECORDS)
def test_every_built_in_autoencoder_rebuilds_from_its_saved_record_alone(kind, tmp_path):
    """A saved diffusion task records its autoencoder (`registry.to_record`) and a run rebuilds it from
    that record around the checkpoint's parameters (`AutoEncoder.bound`), the source it was loaded from
    gone: the record survives JSON, and the rebuilt autoencoder declares the same latent shape and
    encodes and decodes bitwise as the one it recorded, its normalization included."""
    if kind == "sd_vae":
        original = tiny_vae(latent_shift=0.25, latent_scale=0.5)
        params = original.params
    else:
        original, params = _published(kind, tmp_path / "source")
        shutil.rmtree(tmp_path / "source")
    record = json.loads(json.dumps(to_record(original, AutoEncoder)))
    rebuilt = AutoEncoder.bound(record, params=params)
    assert type(rebuilt) is type(original)
    side = 2 * original.downscale_factor
    frames = (5,) if kind == "wan" else ()
    # Qwen-Image's VAE takes four image channels.
    shape = (*frames, side, side, getattr(original.model, "image_channels", 3))
    assert rebuilt.latent_shape(shape) == original.latent_shape(shape)
    image = jax.random.uniform(jax.random.key(0), (1, *shape), minval=-1.0, maxval=1.0)
    latent = original.encode(params, image)
    np.testing.assert_array_equal(np.asarray(rebuilt.encode(params, image)), np.asarray(latent))
    np.testing.assert_array_equal(np.asarray(rebuilt.decode(params, latent)),
                                  np.asarray(original.decode(params, latent)))


@pytest.mark.parametrize("family", ["wan_vae", "qwen_image_vae"])
def test_a_kl_loader_refuses_another_autoencoders_checkpoint(tmp_path, family):
    """A Wan or Qwen-Image VAE's directory given to the AutoencoderKL loaders, a pipeline's and a
    standalone one's, is refused by the class its config names, before its tensors meet the KL tree."""
    from interop_support import extract_fixture

    from dew.nn.autoencoders.pretrained import load_kl_vae

    source = extract_fixture(Path(__file__).parent / "fixtures" / f"{family}.tar.xz", tmp_path)
    folder = next(path.parent for path in source.rglob("config.json"))
    with pytest.raises(ValueError):
        load_kl_vae(folder, subfolder="")
    with pytest.raises(ValueError):
        StableDiffusionVAE(str(folder), dtype=jnp.float32)


def test_a_flax_layout_vae_binds_as_its_torch_layout_does(tmp_path):
    """One KL autoencoder's parameters as diffusers' Flax msgpack (the SD1-era layout) and as its torch
    safetensors (the committed sd3-tiny) bind into one tree through the one loader (`load_kl_vae`, which a
    standalone load and `StableDiffusionVAE` reach): the same parameters, the msgpack's in the dtype it
    stores, and the same source layouts, each written back as the torch tensor it reads from, so either
    exports as the torch checkpoint."""
    from flax.serialization import msgpack_serialize

    from dew.nn.autoencoders.pretrained import load_kl_vae

    _, torch_params, torch_layouts, _ = load_kl_vae(FIXTURES / "sd3-tiny", jnp.float32, subfolder="")
    shutil.copy(FIXTURES / "sd3-tiny" / "config.json", tmp_path / "config.json")
    stored = jax.tree.map(lambda leaf: np.asarray(leaf, jnp.bfloat16), torch_params)
    (tmp_path / "diffusion_flax_model.msgpack").write_bytes(msgpack_serialize(stored))
    vae, flax_params, flax_layouts, _ = load_kl_vae(tmp_path, jnp.float32, subfolder=None)
    assert jax.tree.map(lambda leaf: leaf.dtype, flax_params) == jax.tree.map(lambda leaf: leaf.dtype, stored)
    jax.tree.map(np.testing.assert_array_equal, flax_params, stored)
    def written(layouts):
        return sorted((layout.name, layout.paths, layout.shape, layout.transpose) for layout in layouts)

    assert written(flax_layouts) == written(torch_layouts)
    latent = jax.random.normal(jax.random.key(0), (1, 2, 2, 16))
    standalone = StableDiffusionVAE(str(tmp_path), dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(standalone.decode(standalone.params, latent)),
                                  np.asarray(vae.decode(flax_params, latent)))


_PINNED_LOCATION = """
import json, sys
from dew.nn.autoencoders.vae import located_vae
directory, config, weights = located_vae(sys.argv[1], sys.argv[2])
print(json.dumps({"directory": str(directory), "weights": weights}))
"""


def test_a_pinned_vae_loads_its_own_revisions_weights_not_the_flax_branchs(tmp_path):
    """A standalone VAE loaded at a pinned commit whose weights are torch's alone takes them, from that
    commit, though the repo's `flax` branch holds a flax VAE beside it; the `flax` layout name still finds
    the flax branch's."""
    from interop_support import hub_cache

    from dew.nn.autoencoders.vae import FLAX_WEIGHTS, TORCH_WEIGHTS

    config = json.dumps({"_class_name": "AutoencoderKL", "latent_channels": 4})
    pinned, flax = tmp_path / "pinned", tmp_path / "flax" / "vae"
    pinned.mkdir()
    flax.mkdir(parents=True)
    for directory, weights in ((pinned, TORCH_WEIGHTS), (flax, FLAX_WEIGHTS)):
        (directory / "config.json").write_text(config)
        (directory / weights).write_bytes(b"weights")
    commit = hub_cache(tmp_path / "hub", "dew/tiny-vae", pinned, "1" * 40)
    hub_cache(tmp_path / "hub", "dew/tiny-vae", flax.parent, "2" * 40, branch="flax")
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    def located(revision: str) -> dict:
        done = subprocess.run([sys.executable, "-c", _PINNED_LOCATION, "dew/tiny-vae", revision], env=env,
                              check=False, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-3000:]
        return json.loads(done.stdout.strip().splitlines()[-1])

    snapshots = tmp_path / "hub" / "models--dew--tiny-vae" / "snapshots"
    assert located(commit) == {"directory": str(snapshots / commit), "weights": TORCH_WEIGHTS}
    assert located("flax") == {"directory": str(snapshots / ("2" * 40) / "vae"), "weights": FLAX_WEIGHTS}


def test_a_kl_config_naming_no_class_is_an_autoencoder_kls():
    """An older AutoencoderKL's config names no class; the KL loaders take it as one."""
    from dew.nn.autoencoders.api import check_class
    from dew.nn.autoencoders.vae import PUBLISHED

    check_class({"block_out_channels": [64]}, PUBLISHED, "an old checkpoint", nameless=True)
    with pytest.raises(ValueError):
        check_class({"_class_name": "AutoencoderKLWan"}, PUBLISHED, "a Wan VAE", nameless=True)


def test_a_local_pipelines_autoencoder_loads_from_its_vae_folder(tmp_path):
    """`load_autoencoder` given a local pipeline's root reads the AutoencoderKL in its `vae/` folder."""
    from interop_support import extract_fixture

    from dew.nn.autoencoders.pretrained import load_autoencoder

    root = extract_fixture(FIXTURE_ROOT / "sd3_source.tar.xz", tmp_path)
    pipeline = next(path.parent.parent for path in root.rglob("vae/config.json"))
    assert isinstance(load_autoencoder(str(pipeline), dtype=jnp.float32), StableDiffusionVAE)


def test_autoencoders_of_one_module_share_its_programs():
    """An autoencoder's encode and decode are the process's programs over its module
    (`dew.nn.kernels.options.program`): a second load of one module encodes and decodes with no compile
    of its own, and its frames come back bit for bit as the first's."""
    from dew.telemetry.compiles import Compiles

    first, second = tiny_vae(), tiny_vae()
    frames = jax.random.uniform(jax.random.key(1), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0)
    key = jax.random.key(2)
    latents = first.encode_batch(first.params, frames, key)
    decoded = first.decode_batch(first.params, latents)
    # Every program, by any name: a load's own jit of its module compiles one.
    with Compiles(None) as compiles:
        again = second.encode_batch(second.params, frames, key)
        redecoded = second.decode_batch(second.params, again)
    assert compiles.count == 0, [program.name for program in compiles.programs]
    np.testing.assert_array_equal(np.asarray(again), np.asarray(latents))
    np.testing.assert_array_equal(np.asarray(redecoded), np.asarray(decoded))
