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

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import (
    assert_autoencoder_vjp,
    assert_layout_tensors,
    channels_last,
    extract_fixture,
    fixture_arrays,
    scaled_gap,
)
from safetensors.numpy import load_file, save_file

from dew.nn.autoencoders.dc_ae import DCAE, load_dc_ae

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4
VARIANTS = ("conv", "shuffle")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/dc_ae.tar.xz", tmp_path_factory.mktemp("dc-ae"))


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request, source):
    directory = source / request.param
    return directory, load_dc_ae(directory), fixture_arrays(directory / "reference.npz")


def parameters(layouts):
    """The layouts of the source's parameters: the batch norms' running
    statistics are buffers in the source, with no gradient."""
    return [layout for layout in layouts if not layout.name.endswith(("running_mean", "running_var"))]


def test_every_published_tensor_is_mapped_and_exports_bit_identical(variant):
    directory, (_, params, layouts, _), _ = variant
    tensors = load_file(directory / "diffusion_pytorch_model.safetensors")
    counted = {name: tensor for name, tensor in tensors.items() if not name.endswith("num_batches_tracked")}
    assert_layout_tensors(layouts, {"autoencoder": params}, counted, "vae/")


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


WALKS = {"encode": ("image", "probe_latent", "latent"), "decode": ("code", "probe", "pixels")}
"""Each walk's operand, output probe and output, by their names in the source's reference."""


@pytest.mark.parametrize("method", WALKS)
def test_gradients_match_the_source(variant, method):
    """The encoder's or decoder's output and its parameter and input gradients against the source's."""
    given, probe, output = WALKS[method]
    _, (autoencoder, params, layouts, _), reference = variant
    model = autoencoder.model
    result = assert_autoencoder_vjp(
        method, lambda params, x: model.apply({"params": params}, x, method=getattr(model, method)), params,
        channels_last(reference[given]), [channels_last(reference[probe])], parameters(layouts), reference,
        channels_last(reference[f"{method}.grad_{given}"]), GRADIENT)
    assert scaled_gap(result, channels_last(reference[output])) < FORWARD


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
    with pytest.raises(ValueError):
        load_dc_ae(tmp_path / "vae")


def test_a_run_trains_behind_the_dc_ae_its_checkpoint_names(source, tmp_path):
    """`PretrainedAutoencoder` reads the checkpoint's own config: a DC-AE
    builds as one, and a run's saved params bind without the weights."""
    from dew.nn.autoencoders.dc_ae import DCAutoencoder
    from dew.nn.autoencoders.pretrained import PretrainedAutoencoder

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
    from diffusion_support import images_of

    from dew.config import ObjectiveConfig, TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.data.dataset import Loader
    from dew.inputs import TextCondition
    from dew.nn.autoencoders.pretrained import PretrainedAutoencoder
    from dew.registry import ModelConfig
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 1, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1,
             "dtype": "float32", "attention_impl": "reference"},
        ),
        data=images_of("tfds://oxford_flowers102", 16, batch=8),
        trainer=TrainerConfig(steps=2),
        text=TextCondition(encoder=STUB_TEXT, checkpoint="stub-clip"),
        autoencoder=PretrainedAutoencoder(modelname=str(source / "conv"), dtype="float32"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}),
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
        Loader(train=lambda partition: iter(lambda: batch, None), val=None, records=None, batch=8),
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


def test_a_level_without_blocks_resamples_nothing_and_the_factor_counts_it():
    """DC-AE resamples between two levels only after one with blocks (and at the image where the first
    has none): with three equal-width levels of 1, 0 and 1 blocks a 16 x 16 image encodes to an
    8 x 8 latent, the factor reports that one resampling, and the latent decodes back to 16 x 16."""
    model = DCAE(image_channels=3, latent_channels=4, head_dim=8, encoder_blocks=("ResBlock",) * 3,
                 decoder_blocks=("ResBlock",) * 3, encoder_channels=(8, 8, 8), decoder_channels=(8, 8, 8),
                 encoder_layers=(1, 0, 1), decoder_layers=(1, 0, 1), encoder_scales=((),) * 3,
                 decoder_scales=((),) * 3, decoder_norms=("rms_norm",) * 3, decoder_activations=("silu",) * 3)
    image = jnp.zeros((1, 16, 16, 3))
    variables = model.init(jax.random.key(0), image)
    latent = model.apply(variables, image, method=DCAE.encode)
    assert model.downscale_factor == 2
    assert latent.shape[1:3] == (16 // model.downscale_factor,) * 2
    assert model.apply(variables, latent, method=DCAE.decode).shape == image.shape
