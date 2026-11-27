"""Native DC-AE against the actual `AutoencoderDC`.

`tools/diffusers_dc_ae_reference.py` builds three tiny instances - `conv` with
SANA 1.1's controls (strided convolutions, interpolation, multiscale attention),
`shuffle` with the `in-1.0` family's (pixel shuffles, resampling input and
output blocks, batch norms, ReLU), and `plain`, `conv` without its channel
shortcuts and with a SiLU before the decoder's last convolution -
saves them with `save_pretrained`, and records in float32 the latent of a
rectangular batch, the decode of a fixed latent, their vector-Jacobian
products against fixed probes for the input and every parameter, and the
encode and decode of a batch small enough for the quadratic attention path.

Arrays are recorded channel-first; the tests move channels last. Parameter
gradients are written back into the source layout through the same
`WeightLayout`s an export uses. Each walk's float64 run and its gradients are
held to the source's recorded float64 truths over drawn operands and probes
(`dc_ae_<variant>_seeds.npz`, the tool's `seeds`), with no float32 parity
claimed: no count of draws resolves the walks' float32 errors (row 705).
"""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from autoencoder_walks import dc_ae_call, trained
from interop_support import (
    assert_layout_tensors,
    assert_walk_truths_as_recorded,
    channels_last,
    extract_fixture,
    fixture_arrays,
)
from model_support import attending
from safetensors.numpy import load_file, save_file

from dew.interop.autoencoders.dc_ae import load_dc_ae
from dew.nn.autoencoders.dc_ae import DCAE

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("conv", "shuffle", "plain")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/dc_ae.tar.xz", tmp_path_factory.mktemp("dc-ae"))


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request, source):
    directory = source / request.param
    return directory, load_dc_ae(directory), fixture_arrays(directory / "reference.npz")


def test_every_published_tensor_is_mapped_and_exports_bit_identical(variant):
    directory, (_, params, layouts, _), _ = variant
    tensors = load_file(directory / "diffusion_pytorch_model.safetensors")
    counted = {name: tensor for name, tensor in tensors.items() if not name.endswith("num_batches_tracked")}
    assert_layout_tensors(layouts, {"autoencoder": params}, counted, "vae/")


@attending("reference")
@pytest.mark.parametrize("walk", ["encode", "decode", "small encode", "small decode"])
def test_every_walk_and_gradient_computes_the_sources_function(variant, walk):
    """Each variant's encode of an image and decode of a latent, and against drawn probes the gradients of
    their probed sum in the operand and in every parameter (its batch norms' running statistics, buffers
    in the source, take none), each written back into its source tensor's layout as an export writes it:
    Dew's float64 walk `AutoencoderDC`'s float64 truth per leaf on each recorded draw (the truths
    recording of tools/diffusers_dc_ae_reference.py `seeds`): a rectangular batch's, and at 8x16 pixels a
    small one's, whose deepest level holds 2x4 = attention_head_dim positions, where the source switches to
    quadratic attention. No count of draws resolves the walks' float32 errors (their ReLU attention's tiny
    normalizers, row 705), so each holds its function."""
    directory, (autoencoder, params, layouts, _), _ = variant
    name, model = directory.name, autoencoder.model
    assert_walk_truths_as_recorded(ROOT / f"tests/fixtures/dc_ae_{name}_seeds.npz", f"{name} {walk}",
                                   dc_ae_call(model.clone(dtype=jnp.float64), walk), params, trained(layouts))


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


def test_a_standalone_load_reads_its_config_once(source, monkeypatch):
    """A published DC-AE loaded on its own (`load_autoencoder`) reads its config once: its dispatch's read,
    which picks the loader, is the one the loader checks its class by and builds from, where the loader
    resolved the release and parsed its config twice again (audit3 families F4). The reads are counted at
    the file."""
    from pathlib import Path as PathType

    from dew.interop.autoencoders.pretrained import load_autoencoder

    read_text, reads = PathType.read_text, []

    def counted(path, *args, **kwargs):
        if path.name == "config.json":
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(PathType, "read_text", counted)
    loaded = load_autoencoder(str(source / "plain"), dtype=jnp.float32)
    assert isinstance(loaded.model, DCAE) and len(reads) == 1


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
    from dew.interop.autoencoders.pretrained import PretrainedAutoencoder
    from dew.nn.autoencoders.dc_ae import DCAutoencoder

    shutil.copytree(source / "conv", tmp_path / "dc-ae")
    spec = PretrainedAutoencoder(modelname=str(tmp_path / "dc-ae"), dtype="float32", latent_scale=0.5)
    built = spec.build()
    assert isinstance(built, DCAutoencoder) and built.latent_scale == 0.5
    image = channels_last(dict(np.load(source / "conv" / "reference.npz"))["image"])
    expected = np.asarray(built.encode(built.params, image))

    (tmp_path / "dc-ae" / "diffusion_pytorch_model.safetensors").unlink()
    rebound = spec.build(params=built.params)
    np.testing.assert_array_equal(np.asarray(rebound.encode(rebound.params, image)), expected)


@attending("reference")
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
    from dew.interop.autoencoders.pretrained import PretrainedAutoencoder
    from dew.registry import ModelConfig
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 1, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1,
             "dtype": "float32"},
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


def test_the_linear_attention_and_its_gradients_are_on_par_with_diffusers(monkeypatch):
    """DC-AE's ReLU linear attention (`linear_attention`) over 1024 positions of four 32-wide heads, a
    released DC-AE's head width, and its queries', keys' and values' gradients, check 1 against Diffusers'
    `apply_linear_attention` in float32, the truth its float64 run. The queries and keys keep two channels
    positive, so every normalizer sums its positions, the rest drawn on both sides of the ReLU off its
    kink (row 13524: one float32 sum over the positions left the gradients 1.5-2x Diffusers' error). Its
    CPU arithmetic a library process's, float32 products summed in float32, not the suite's wide sums."""
    import torch
    from arm_forms import as_given
    from diffusers.models.attention_processor import SanaMultiscaleLinearAttention
    from reference_error import assert_as_exact

    from _dew_torch.precision import float64
    from dew.nn.autoencoders.dc_ae import linear_attention
    from dew.nn.precision import WIDE_CPU_ARITHMETIC

    monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)

    positions, heads, width = 1024, 4, 32
    attention = SanaMultiscaleLinearAttention(heads * width, heads * width, attention_head_dim=width).eval()

    def draw(seed):
        rng = np.random.default_rng([13524, seed])
        shape = (2, positions, heads, width)

        def gated():
            magnitudes = rng.uniform(0.25, 1.5, size=shape)
            magnitudes[..., 2:] *= rng.integers(0, 2, size=(*shape[:-1], width - 2)) * 2 - 1
            return magnitudes.astype(np.float32)
        return (gated(), gated(), rng.uniform(-1, 1, size=shape).astype(np.float32),
                rng.standard_normal(shape).astype(np.float32))

    @jax.jit
    def dew(query, key, value, cotangent):
        attended, backward = jax.vjp(lambda q, k, v: linear_attention(jax.nn.relu(q), jax.nn.relu(k), v),
                                     query, key, value)
        return (attended, *backward(cotangent))

    def diffusers_run(query, key, value, cotangent, dtype):
        # Diffusers' heads lie `[batch, heads, width, positions]`.
        query, key, value, cotangent = (torch.as_tensor(array, dtype=dtype).permute(0, 2, 3, 1).contiguous()
                                        for array in (query, key, value, cotangent))
        for leaf in (query, key, value):
            leaf.requires_grad_()
        attended = attention.apply_linear_attention(query.relu(), key.relu(), value)
        gradients = torch.autograd.grad(attended, (query, key, value), cotangent)
        return tuple(array.detach().permute(0, 3, 1, 2).numpy() for array in (attended, *gradients))

    def truth(*inputs):
        with float64():
            return diffusers_run(*inputs, dtype=torch.float64)

    reference = as_given(lambda *inputs: diffusers_run(*inputs, dtype=torch.float32))
    # `seeds_for`'s count at this geometry (probe IG-P13524U, 567ad6bed: green at 8 seeds, the one dot red).
    assert_as_exact(dew, reference, draw, "DC-AE linear attention", truth_fn=as_given(truth), seeds=8)


def test_the_linear_attentions_sums_cancel_past_float32s_range_under_wide_cpu_arithmetic():
    """Under the suite's wide CPU arithmetic, where a CPU's float32 products sum in float64 and round once
    (`precision.sums_in_float64`), DC-AE's linear attention on the CPU takes each sum over its positions
    whole: values
    `2^27, 2^27, -2^27, -2^26` under keys of `2^100` leave `2^26 * 2^100` over the keys' sum `2^106`,
    `2^20` exactly, where a first group's `2^128` rounded to float32 alone overflowed to infinity (train
    1387's review)."""
    from dew.nn.autoencoders.dc_ae import linear_attention
    from dew.nn.precision import sums_in_float64

    query = jnp.ones((1, 64, 1, 1), jnp.float32)
    key = jnp.full((1, 64, 1, 1), 2.0 ** 100, jnp.float32)
    value = np.zeros((1, 64, 1, 1), np.float32)
    value[0, [0, 1, 4, 5], 0, 0] = (2.0 ** 27, 2.0 ** 27, -2.0 ** 27, -2.0 ** 26)
    assert sums_in_float64(key, key), "the suite runs wide CPU arithmetic"
    # On the CPU, whose lowering alone sums wide.
    with jax.default_device(jax.devices("cpu")[0]):
        attended = jax.jit(linear_attention)(query, key, jnp.asarray(value))
    np.testing.assert_array_equal(np.asarray(attended), np.full((1, 64, 1, 1), 2.0 ** 20, np.float32))


def linear_attention_sums_where_its_operands_lie(default: str) -> None:
    """Called eagerly (no jit) on operands committed to the CPU, DC-AE's linear attention takes the CPU's
    whole sums under wide arithmetic whatever the process's default device (`default`, the backend the
    process runs), as its jitted call does there (train 1393's review: an accelerator-default process took
    the grouped sums): the CPU lane's test below, and the GPU tier's (tests/test_gpu_dc_ae.py)."""
    assert jax.default_backend() == default
    from dew.nn.autoencoders.dc_ae import linear_attention

    cpu = jax.devices("cpu")[0]
    query = jax.device_put(jnp.ones((1, 64, 1, 1), jnp.float32), cpu)
    key = jax.device_put(jnp.full((1, 64, 1, 1), 2.0 ** 100, jnp.float32), cpu)
    value = np.zeros((1, 64, 1, 1), np.float32)
    value[0, [0, 1, 4, 5], 0, 0] = (2.0 ** 27, 2.0 ** 27, -2.0 ** 27, -2.0 ** 26)
    attended = linear_attention(query, key, jax.device_put(jnp.asarray(value), cpu))
    np.testing.assert_array_equal(np.asarray(attended), np.full((1, 64, 1, 1), 2.0 ** 20, np.float32))


@pytest.mark.cpu
def test_the_linear_attention_called_eagerly_on_cpu_operands_sums_where_they_lie():
    """On the CPU lane, DC-AE's eager linear attention sums where its operands lie
    (`linear_attention_sums_where_its_operands_lie`)."""
    linear_attention_sums_where_its_operands_lie("cpu")
