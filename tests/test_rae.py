"""Native RAE against the actual `AutoencoderRAE`.

`tools/diffusers_rae_reference.py` builds three tiny instances, one per
frozen encoder (DINOv2 with registers, SigLIP, ViT-MAE), saves each with
`save_pretrained`, and records in float32 the latent of a batch with the
gradient of a probe against the pixels, and the decode of a fixed latent
with the gradients of a probe against the latent and every decoder
parameter. Each resizes its image and its position table, as the published
checkpoints do; DINOv2's shrinks both, SigLIP's and MAE's enlarge the image.

It also records what the published checkpoints compute in float64 on
inputs numpy builds (`tests/fixtures/rae_published.npz`), which the
`network` test holds the port to after downloading the weights.

Arrays are recorded `[B, C, H, W]` with pixels in [0, 1]; the tests move
channels last and hand the autoencoder pixels in [-1, 1]. Every gap is
scaled by max(1, |reference|).
"""

import importlib.util
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
from dew.nn.autoencoders.rae import UNREAD, load_dinov2, load_rae, rae_fields, rae_path

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4
VARIANTS = ("dinov2", "siglip2", "mae")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rae")
    with tarfile.open(ROOT / "tests/fixtures/rae.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request, source):
    directory = source / request.param
    return directory, load_rae(directory), dict(np.load(directory / "reference.npz"))


def channels_last(array):
    return np.asarray(array).transpose(0, 2, 3, 1)


def scaled_gap(actual, expected) -> float:
    expected = np.asarray(expected, np.float64)
    difference = np.abs(np.asarray(actual, np.float64) - expected).max()
    return float(difference / max(1.0, float(np.abs(expected).max())))


def test_every_computed_tensor_is_mapped_and_exports_bit_identical(variant):
    directory, (_, params, layouts, _), _ = variant
    tensors = component_tensors(directory, "")
    read = {name for name in tensors if not name.startswith(UNREAD)}
    assert {layout.name for layout in layouts} == {f"vae/{name}" for name in read}
    for layout in layouts:
        written = layout.export({"autoencoder": params})
        published = tensors[layout.name.removeprefix("vae/")]
        assert written.dtype == published.dtype and written.shape == published.shape, layout.name
        assert written.tobytes() == published.tobytes(), layout.name


def test_the_latent_matches_the_source(variant):
    """The normalized latent: the encoder's patch tokens as a grid, less the
    checkpoint's mean and over its deviation, per position where it gives
    one per position."""
    _, (autoencoder, params, _, config), reference = variant
    image = channels_last(reference["image"])
    latent = autoencoder.encode(params, 2 * image - 1)
    expected = channels_last(reference["latent"])
    assert latent.shape == expected.shape == (image.shape[0], *autoencoder.latent_shape(image.shape[1:]))
    gap = scaled_gap(latent, expected)
    print(f"{config['encoder_type']} latent gap {gap:.3g}")
    assert gap < FORWARD


def test_the_decode_matches_the_source(variant):
    _, (autoencoder, params, _, config), reference = variant
    decoded = autoencoder.decode(params, channels_last(reference["code"]))
    gap = scaled_gap((decoded + 1) / 2, channels_last(reference["pixels"]))
    print(f"{config['encoder_type']} decode gap {gap:.3g}")
    assert gap < FORWARD


def test_the_pixel_gradient_through_the_frozen_encoder_matches_the_source(variant):
    _, (autoencoder, params, _, config), reference = variant
    probe = channels_last(reference["probe_latent"])
    grad = jax.grad(lambda image: jnp.sum(autoencoder.encode(params, 2 * image - 1) * probe))(
        channels_last(reference["image"]))
    gap = scaled_gap(grad, channels_last(reference["encode.grad_image"]))
    print(f"{config['encoder_type']} pixel gradient gap {gap:.3g}")
    assert gap < GRADIENT


def test_decoder_gradients_match_the_source(variant):
    _, (autoencoder, params, layouts, config), reference = variant
    probe = channels_last(reference["probe"])

    def objective(params, code):
        return jnp.sum((autoencoder.decode(params, code) + 1) / 2 * probe)

    grad_params, grad_code = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["code"]))
    code_gap = scaled_gap(grad_code, channels_last(reference["decode.grad_code"]))
    gaps = {layout.name: scaled_gap(layout.export({"autoencoder": grad_params}),
                                    reference[f"decode.grad.{layout.name.removeprefix('vae/')}"])
            for layout in layouts if layout.name.startswith("vae/decoder.")}
    worst = max(gaps, key=gaps.get)
    print(
        f"{config['encoder_type']}: latent gradient gap {code_gap:.3g}; "
        f"worst parameter {worst} {gaps[worst]:.3g}"
    )
    assert code_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_any_image_size_encodes_to_the_grid(variant):
    _, (autoencoder, params, _, config), _ = variant
    grid = config["encoder_input_size"] // config["encoder_patch_size"]
    for size in (48, 96):
        assert autoencoder.latent_shape((size, size, 3)) == (grid, grid, config["encoder_hidden_size"])
        assert autoencoder.encode(params, jnp.zeros((1, size, size, 3))).shape == (1, grid, grid,
                                                                                    config["encoder_hidden_size"])
    with pytest.raises(ValueError, match="decodes a"):
        autoencoder.decode_batch(params, jnp.zeros((1, grid + 1, grid + 1, config["encoder_hidden_size"])))


def test_a_missing_tensor_is_refused(source, tmp_path):
    shutil.copytree(source / "dinov2", tmp_path / "dinov2")
    tensors = component_tensors(source / "dinov2", "")
    del tensors["encoder.encoder.layer.1.layer_scale2.lambda1"]
    save_file(tensors, tmp_path / "dinov2" / "diffusion_pytorch_model.safetensors")
    with pytest.raises(ValueError, match="layer_scale2"):
        load_rae(tmp_path / "dinov2")


@pytest.mark.parametrize("name, ndim", [
    ("encoder.encoder.layer.0.attention.attention.rotary.weight", 2),
    ("encoder.vision_model.encoder.layers.0.self_attn.qkv.weight", 2),
    ("decoder.decoder_layers.x.attention.to_q.weight", 2),
    ("encoder.embeddings.distillation_token", 3),
    ("decoder.decoder_pred.scale", 1),
])
def test_an_unknown_tensor_is_refused(name, ndim):
    with pytest.raises(ValueError):
        rae_path(name, ndim)


@pytest.mark.parametrize("change", [
    {"encoder_type": "clip"}, {"reshape_to_2d": False}, {"num_channels": 4}, {"encoder_input_size": 120},
    {"image_size": 100},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "dinov2" / "config.json").read_text())
    rae_fields(config)
    with pytest.raises(ValueError):
        rae_fields({**config, **change})


@pytest.mark.network
@pytest.mark.parametrize("name", VARIANTS)
def test_the_published_rae_matches_the_source(name):
    """Each published RAE, downloaded, encodes a smooth 256x256 image and
    decodes a standard normal latent as diffusers' `AutoencoderRAE` does in
    float64, on the channels and pixels `rae_published.npz` keeps, to within
    twice the source's own float32 distance from that result: two float32
    walks rounding in different orders, and SigLIP's residual stream reaches
    several hundred."""
    spec = importlib.util.spec_from_file_location(
        "diffusers_rae_reference", ROOT / "tools/diffusers_rae_reference.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    repo, revision = tool.PUBLISHED[name]
    reference = np.load(ROOT / "tests/fixtures/rae_published.npz")
    autoencoder, params, _, config = load_rae(repo, revision=revision)
    grid = config["encoder_input_size"] // config["encoder_patch_size"]

    latent = autoencoder.encode(params, 2 * channels_last(tool.smooth_image()) - 1)[
        ..., : tool.LATENT_CHANNELS
    ]
    decoded = autoencoder.decode(
        params, channels_last(tool.normal_latent(config["encoder_hidden_size"], grid))
    )
    gaps = {"latent": scaled_gap(latent, channels_last(reference[f"{name}.latent"])),
            "pixels": scaled_gap((decoded[:, :tool.CROP, :tool.CROP] + 1) / 2,
                                 channels_last(reference[f"{name}.pixels"]))}
    bounds = {key: max(FORWARD, 2 * float(reference[f"{name}.{key}.float32_gap"])) for key in gaps}
    print(f"published {name} gaps {gaps}, bounds {bounds}")
    assert all(gaps[key] < bounds[key] for key in gaps), (gaps, bounds)


def test_the_plain_dinov2_matches_transformers(source):
    """transformers' `Dinov2Model` (no registers, the final norm kept):
    patch tokens of a normalized 112-pixel batch, the 37x37 table resized to
    the grid without antialiasing, and the gradient of a probe against the
    pixels."""
    module, params, layouts = load_dinov2(source / "dinov2_plain")
    reference = dict(np.load(source / "dinov2_plain" / "reference.npz"))
    assert {layout.name for layout in layouts} == {
        f"dinov2/{name}"
        for name in component_tensors(source / "dinov2_plain", "")
        if name != "embeddings.mask_token"
    }
    module = module.clone(input_size=112)
    pixels = channels_last(reference["pixels"])
    tokens = module.apply({"params": params}, pixels)
    assert tokens.shape == (1, 8, 8, 64)
    probe = reference["probe"].reshape(1, 8, 8, 64)
    grad = jax.grad(lambda pixels: jnp.sum(module.apply({"params": params}, pixels) * probe))(pixels)
    gaps = {"tokens": scaled_gap(tokens.reshape(1, 64, 64), reference["tokens"]),
            "gradient": scaled_gap(grad, channels_last(reference["grad_pixels"]))}
    print(f"plain DINOv2 gaps {gaps}")
    assert gaps["tokens"] < FORWARD and gaps["gradient"] < GRADIENT, gaps


@pytest.mark.network
def test_the_published_dinov2_matches_transformers():
    """`facebook/dinov2-base`, downloaded, on the smooth image at 224 pixels,
    against `Dinov2Model` in float64, as the published RAEs are held."""
    spec = importlib.util.spec_from_file_location(
        "diffusers_rae_reference", ROOT / "tools/diffusers_rae_reference.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    reference = np.load(ROOT / "tests/fixtures/rae_published.npz")
    module, params, _ = load_dinov2(tool.DINOV2[0], revision=tool.DINOV2[1])
    mean, std = (np.asarray(value, np.float32) for value in tool.IMAGENET)
    pixels = (channels_last(tool.smooth_image(224)) - mean) / std
    tokens = module.clone(input_size=224).apply({"params": params}, pixels)[..., :tool.LATENT_CHANNELS]
    gap = scaled_gap(tokens.reshape(1, 256, -1), reference["dinov2_plain.tokens"])
    bound = max(FORWARD, 2 * float(reference["dinov2_plain.tokens.float32_gap"]))
    print(f"published DINOv2 gap {gap:.3g}, bound {bound:.3g}")
    assert gap < bound


def test_a_latent_run_trains_behind_the_rae_and_samples_its_image_size(source):
    """A diffusion run denoises the RAE's `[grid, grid, width]` latent with
    its per-position normalization, trains the model alone, and samples
    images of the size the decoder paints."""
    import optax
    from diffusion_stubs import STUB_TEXT

    from dew.config import ModelConfig, ObjectiveConfig, TrainerConfig
    from dew.data import Dataset, TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder, TextCondition
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 1, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1,
             "dtype": "float32", "attention_impl": "reference"},
        ),
        data=TFDSImages(image_size=64),
        trainer=TrainerConfig(batch_size=8, steps=2),
        text=TextCondition(encoder=STUB_TEXT, checkpoint="stub-clip"),
        autoencoder=PretrainedAutoencoder(
            modelname=str(source / "siglip2"), revision="main", dtype="float32"
        ), objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}),
    )
    objective = config.build()
    assert objective.latent_shape == (8, 8, 64)
    images = (np.random.default_rng(0).random((8, 64, 64, 3)) * 255).astype(np.uint8)
    batch = {
        "image": images,
        "text": objective.inputs.conditions["textcontext"].encoder.tokenize(list("abcdefgh")),
    }
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    initial = trainer.initial_state()
    state = trainer.fit(
        Dataset(train=lambda partition: iter(lambda: batch, None), val=None, records=None, batch=8),
        steps=2,
        log_every=100,
    )

    for before, after in zip(jax.tree.leaves(initial.variables["autoencoder"]),
                             jax.tree.leaves(state.variables["autoencoder"]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    sampled = objective.pipeline(state)(["a", "b"], key=0).host()
    assert sampled.images.shape == (2, 64, 64, 3)
