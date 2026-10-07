"""Native DC-AE against the actual `AutoencoderDC`.

`tools/diffusers_dc_ae_reference.py` builds two tiny instances with the
published checkpoints' controls - `conv` as SANA 1.1's (strided convolutions,
interpolation, multiscale attention) and `shuffle` as the `in-1.0` family's
(pixel shuffles, resampling input and output blocks, batch norms, ReLU) -
saves them with `save_pretrained`, and records in float32 the latent of a
rectangular batch, the decode of a fixed latent, their vector-Jacobian
products against fixed probes for the input and every parameter, and the
encode and decode of a batch small enough for the quadratic attention path.

Arrays are recorded channel-first; the tests move channels last. Parameter
gradients are written back into the source layout through the same
`WeightLayout`s an export uses. Every gap is scaled by max(1, |reference|).
"""

import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from dew.nn.autoencoders.dc_ae import dc_ae_fields, dc_ae_path, load_dc_ae

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4
VARIANTS = ("conv", "shuffle")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dc-ae")
    with tarfile.open(ROOT / "tests/fixtures/dc_ae.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request, source):
    directory = source / request.param
    return directory, load_dc_ae(directory), dict(np.load(directory / "reference.npz"))


def channels_last(array):
    return np.asarray(array).transpose(0, 2, 3, 1)


def scaled_gap(actual, expected) -> float:
    expected = np.asarray(expected, np.float64)
    difference = np.abs(np.asarray(actual, np.float64) - expected).max()
    return float(difference / max(1.0, float(np.abs(expected).max())))


def parameter_gaps(layouts, gradients, reference, walk: str) -> dict[str, float]:
    """Each native gradient of a source parameter, exported to its tensor's
    layout, against the source's gradient of that tensor. The batch norms'
    running statistics are buffers in the source, with no gradient."""
    return {layout.name: scaled_gap(layout.export({"autoencoder": gradients}), reference[key])
            for layout in layouts
            if (key := f"{walk}.grad.{layout.name.removeprefix('vae/')}") in reference}


def test_every_published_tensor_is_mapped_and_exports_bit_identical(variant):
    directory, (_, params, layouts, _), _ = variant
    tensors = load_file(directory / "diffusion_pytorch_model.safetensors")
    counted = {name for name in tensors if not name.endswith("num_batches_tracked")}
    assert {layout.name for layout in layouts} == {f"vae/{name}" for name in counted}
    for layout in layouts:
        written = layout.export({"autoencoder": params})
        published = tensors[layout.name.removeprefix("vae/")]
        assert written.dtype == published.dtype and written.shape == published.shape, layout.name
        assert written.tobytes() == published.tobytes(), layout.name


def test_the_latent_matches_the_source(variant):
    _, (autoencoder, params, _, _), reference = variant
    model = autoencoder.model
    latent = model.apply({"params": params}, channels_last(reference["image"]), method=model.encode)
    assert latent.shape == (2, 8, 12, 8)
    gap = scaled_gap(latent, channels_last(reference["latent"]))
    print(f"latent gap {gap:.3g}")
    assert gap < FORWARD


def test_the_decode_matches_the_source(variant):
    _, (autoencoder, params, _, _), reference = variant
    model = autoencoder.model
    pixels = model.apply({"params": params}, channels_last(reference["code"]), method=model.decode)
    assert pixels.shape == (2, 32, 48, 3)
    gap = scaled_gap(pixels, channels_last(reference["pixels"]))
    print(f"decode gap {gap:.3g}")
    assert gap < FORWARD


def test_the_quadratic_attention_matches_the_source(variant):
    """At 8x16 pixels the deepest level holds 2x4 = attention_head_dim
    positions, where the source switches to quadratic attention."""
    _, (autoencoder, params, _, _), reference = variant
    model = autoencoder.model
    latent = model.apply({"params": params}, channels_last(reference["small"]), method=model.encode)
    pixels = model.apply({"params": params}, channels_last(reference["small_code"]), method=model.decode)
    gaps = {"encode": scaled_gap(latent, channels_last(reference["small_latent"])),
            "decode": scaled_gap(pixels, channels_last(reference["small_pixels"]))}
    print(f"quadratic gaps {gaps}")
    assert max(gaps.values()) < FORWARD, gaps


def test_encoder_gradients_match_the_source(variant):
    _, (autoencoder, params, layouts, _), reference = variant
    model = autoencoder.model
    probe = channels_last(reference["probe_latent"])

    def objective(params, image):
        return jnp.sum(model.apply({"params": params}, image, method=model.encode) * probe)

    grad_params, grad_image = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["image"]))
    image_gap = scaled_gap(grad_image, channels_last(reference["encode.grad_image"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "encode")
    worst = max(gaps, key=gaps.get)
    print(f"encode: image gradient gap {image_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert image_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_decoder_gradients_match_the_source(variant):
    _, (autoencoder, params, layouts, _), reference = variant
    model = autoencoder.model
    probe = channels_last(reference["probe"])

    def objective(params, latent):
        return jnp.sum(model.apply({"params": params}, latent, method=model.decode) * probe)

    grad_params, grad_code = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["code"]))
    code_gap = scaled_gap(grad_code, channels_last(reference["decode.grad_code"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "decode")
    worst = max(gaps, key=gaps.get)
    print(f"decode: latent gradient gap {code_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert code_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_the_autoencoder_scales_as_the_pipeline_does(variant):
    """SANA's pipeline trains on `z * scaling_factor` and decodes
    `z / scaling_factor`; a video is encoded frame by frame."""
    _, (autoencoder, params, _, config), reference = variant
    scale = config["scaling_factor"]
    assert autoencoder.downscale_factor == 4 and autoencoder.latent_channels == 8
    image = channels_last(reference["image"])
    raw = np.asarray(autoencoder.encode_batch(params, image))
    normalized = np.asarray(autoencoder.encode(params, image))
    np.testing.assert_allclose(normalized, raw * scale, rtol=1e-6)
    # The encoder is deterministic: a key draws nothing.
    np.testing.assert_array_equal(np.asarray(autoencoder.encode(params, image, key=jax.random.key(0))),
                                  normalized)
    video = np.asarray(autoencoder.encode(params, image[None]))
    np.testing.assert_array_equal(video[0], normalized)
    decoded = np.asarray(autoencoder.decode(params, normalized))
    np.testing.assert_allclose(
        decoded, np.asarray(autoencoder.decode_batch(params, raw)), rtol=1e-5, atol=1e-5
    )


def test_a_missing_tensor_is_refused(source, tmp_path):
    shutil.copytree(source / "conv", tmp_path / "vae")
    weights = tmp_path / "vae" / "diffusion_pytorch_model.safetensors"
    tensors = load_file(weights)
    del tensors["encoder.down_blocks.2.1.attn.to_qkv_multiscale.0.proj_out.weight"]
    save_file(tensors, weights)
    with pytest.raises(ValueError, match=r"down_blocks_2_1.attn.to_qkv_multiscale_0.per_head"):
        load_dc_ae(tmp_path / "vae")


@pytest.mark.parametrize("name, ndim", [
    ("encoder.down_blocks.0.0.norm.weight", 4),
    ("decoder.up_blocks.2.0.attn.to_q.bias", 2),
    ("vae.encoder.conv_in.weight", 4),
    ("encoder.conv_in.scale", 1),
])
def test_an_unknown_tensor_is_refused(name, ndim):
    with pytest.raises(ValueError):
        dc_ae_path(name, ndim)


@pytest.mark.parametrize("change", [
    {"encoder_block_types": "ViTBlock"}, {"decoder_norm_types": "layer_norm"},
    {"decoder_act_fns": "gelu"}, {"downsample_block_type": "avg_pool"},
    {"upsample_block_type": "bilinear"}, {"encoder_layers_per_block": [1, 1]},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "conv" / "config.json").read_text())
    dc_ae_fields(config)
    with pytest.raises(ValueError):
        dc_ae_fields({**config, **change})


def test_a_run_trains_behind_the_dc_ae_its_checkpoint_names(source, tmp_path):
    """`PretrainedAutoencoder` reads the checkpoint's own config: a DC-AE
    builds as one, and a run's saved params bind without the weights."""
    from dew.nn.autoencoders.dc_ae import DCAutoencoder
    from dew.objectives.diffusion import PretrainedAutoencoder

    shutil.copytree(source / "conv", tmp_path / "dc-ae")
    spec = PretrainedAutoencoder(modelname=str(tmp_path / "dc-ae"), dtype="float32", latent_scale=0.5)
    built = spec.build()
    assert isinstance(built, DCAutoencoder) and built.latent_scale == 0.5
    image = channels_last(dict(np.load(source / "conv" / "reference.npz"))["image"])
    expected = np.asarray(built.encode(built.params, image))

    (tmp_path / "dc-ae" / "diffusion_pytorch_model.safetensors").unlink()
    rebound = spec.build(params=built.params)
    np.testing.assert_array_equal(np.asarray(rebound.encode(rebound.params, image)), expected)


def test_a_latent_run_trains_behind_the_dc_ae_and_leaves_it_frozen(source):
    """A diffusion run places the DC-AE's variables beside the model's and
    trains the model alone. Placement reads a leaf's axes from its module's
    name, and a DC-AE module named like another's declaration of fewer axes
    (a bare `proj_out`) left the run unable to place its variables."""
    import optax
    from diffusion_stubs import STUB_TEXT

    from dew.config import ModelConfig, TrainerConfig
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
        data=TFDSImages(image_size=16),
        trainer=TrainerConfig(batch_size=8, steps=2),
        solver=Euler(),
        sampling_steps=2,
        text=TextCondition(encoder=STUB_TEXT, checkpoint="stub-clip"),
        autoencoder=PretrainedAutoencoder(modelname=str(source / "conv"), dtype="float32"),
    )
    objective = config.build()
    images = (np.random.default_rng(0).random((8, 16, 16, 3)) * 255).astype(np.uint8)
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
    moved = [
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(initial.variables["params"]),
            jax.tree.leaves(state.variables["params"]), strict=True
        )
    ]
    assert any(moved)
