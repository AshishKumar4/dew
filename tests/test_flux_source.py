"""Native Flux against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_flux_reference.py` builds tiny `FluxTransformer2DModel`
instances, saves each with its real config and safetensors, and runs each the
way its pipeline runs it - packed latents, the ids the pipeline lays out, the
timestep it has divided by the training count and the distilled guidance a
guidance-embedded checkpoint takes - recording the forward, the
vector-Jacobian products of the latent, the text tokens and the pooled
vector, and the gradient of every parameter.

The native model owns its pipeline's 2x2 packing, so it works in latents.
These tests convert between the two layouts with the source's own permutation
written out here, rather than with the functions under test.
"""

import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.diffusion import component_tensors, flux_fields, translate_flux_weights
from dew.nn.backbones.flux import FluxTransformer
from dew.nn.backbones.unet_condition import DenoisingCondition

ROOT = Path(__file__).resolve().parents[1]
CASES = ("schnell", "dev", "rect", "deep")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flux-source")
    with tarfile.open(ROOT / "tests/fixtures/flux_source.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def unpacked(packed: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """`FluxPipeline._unpack_latents`, read into NHWC: a position's channels
    run channel-major over its own two rows and columns."""
    batch, _, width = packed.shape
    channels = width // 4
    grouped = packed.reshape(batch, rows, columns, channels, 2, 2)
    return grouped.transpose(0, 1, 4, 2, 5, 3).reshape(batch, rows * 2, columns * 2, channels)


def packed(latent: np.ndarray) -> np.ndarray:
    """`FluxPipeline._pack_latents` from NHWC."""
    batch, height, width, channels = latent.shape
    grouped = latent.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return grouped.transpose(0, 1, 3, 5, 2, 4).reshape(batch, height * width // 4, channels * 4)


def transformer_case(directory: Path, arrays, name: str, grid: tuple[int, int]):
    """The native model, its loaded parameters and the case's own inputs."""
    config = json.loads(str(arrays[f"{name}.config"]))
    model = FluxTransformer(**flux_fields(config, attention_impl="xla"))
    params, layouts = translate_flux_weights(component_tensors(directory / name, "transformer"))
    latent = jnp.asarray(unpacked(arrays[f"{name}.packed"], *grid))
    context = jnp.asarray(arrays[f"{name}.context"])
    pooled = jnp.asarray(arrays[f"{name}.pooled"])
    times = jnp.asarray(arrays[f"{name}.times"])
    recorded = arrays[f"{name}.guidance"]
    guidance = None if recorded.size == 0 else jnp.asarray(recorded)

    def forward(params, latent, context, pooled):
        return model.apply({"params": params}, latent, times,
                           DenoisingCondition(context, pooled, guidance=guidance))

    return forward, params, layouts, latent, context, pooled


@pytest.mark.parametrize("name", CASES)
def test_native_flux_matches_the_source_forward_and_every_gradient(name, source):
    """One published transformer's own tensors, read through the native model.

    The variants are the ones whose wiring differs: the schnell-style model
    with no guidance embedder, the distilled one with it, a rectangular packed
    grid whose row and column rotations differ, and a stack with a different
    split between double and single blocks.

    The forward and the latent, token and pooled gradients land within 1e-5 of
    the source's scale, and every parameter gradient within 1e-4 - except the
    two embedders' first weights, which are the sinusoidal features
    themselves. Those features are the source's own `sin` and `cos` of an
    argument the model scales by a thousand, and float32 already disagrees
    there by 6.1e-5 at this case's timestep and 1.8e-4 at its guidance,
    measured against `get_timestep_embedding` directly; the gradients of
    those two weights are those features transposed, so they inherit it.
    """
    # The measured feature difference at an argument of 3.5 * 1000.
    features = ("time_text_embed.guidance_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_1.weight")
    record = json.loads((source / "flux_transformer.json").read_text())
    grid = tuple(record["cases"][name]["grid"])
    with np.load(source / "flux_transformer.npz") as arrays:
        forward, params, layouts, latent, context, pooled = transformer_case(
            source, arrays, name, grid)
        probe = jnp.asarray(unpacked(arrays[f"{name}.probe"], *grid))
        output = jax.jit(forward)(params, latent, context, pooled)
        assert relative_gap(packed(np.asarray(output)), arrays[f"{name}.output"]) < 1e-5
        gradients = jax.jit(jax.grad(
            lambda p, l, c, q: jnp.sum(forward(p, l, c, q) * probe), argnums=(0, 1, 2, 3)))(
                params, latent, context, pooled)
        assert relative_gap(packed(np.asarray(gradients[1])), arrays[f"{name}.grad_packed"]) < 1e-5
        assert relative_gap(gradients[2], arrays[f"{name}.grad_context"]) < 1e-5
        assert relative_gap(gradients[3], arrays[f"{name}.grad_pooled"]) < 1e-5
        layout = {entry.name: entry for entry in layouts}
        prefix = f"{name}.grad_param."
        names = [key for key in arrays.files if key.startswith(prefix)]
        assert names, name
        for key in names:
            entry = layout["transformer/" + key[len(prefix):]]
            node = gradients[0]
            for step in entry.paths[0][1:]:
                node = node[step]
            value = np.asarray(node)
            if entry.transpose is not None:
                value = value.transpose(entry.transpose)
            assert value.shape == tuple(entry.shape), key
            bound = 2e-4 if key[len(prefix):] in features else 1e-4
            assert relative_gap(value, arrays[key]) < bound, key


def test_every_declared_flux_tensor_is_mapped(source):
    """Every tensor the source stores lands in the native tree, and a name
    this translation does not know raises with that name rather than loading a
    checkpoint that means something else."""
    from dew.interop.diffusion import _flux_path

    tensors = component_tensors(source / "dev", "transformer")
    params, layouts = translate_flux_weights(tensors)
    assert len(layouts) == len(tensors)
    leaves = {"/".join(entry.paths[0]) for entry in layouts}
    assert len(leaves) == len(tensors)
    with pytest.raises(ValueError, match="unknown tensor name"):
        _flux_path("transformer_blocks.0.attn.to_out.1.weight")
    with pytest.raises(ValueError, match="unknown tensor name"):
        _flux_path("single_transformer_blocks.0.attn.to_add_out.weight")


def test_the_guidance_input_follows_the_checkpoints_own_embedder(source):
    """A distilled checkpoint reads a guidance value and a schnell-style one
    holds no embedder for it, so each refuses the other's call rather than
    ignoring an input the source would have used."""
    record = json.loads((source / "flux_transformer.json").read_text())
    with np.load(source / "flux_transformer.npz") as arrays:
        for name, wanted in (("dev", True), ("schnell", False)):
            grid = tuple(record["cases"][name]["grid"])
            config = json.loads(str(arrays[f"{name}.config"]))
            assert flux_fields(config)["guidance_embeds"] is wanted
            model = FluxTransformer(**flux_fields(config, attention_impl="xla"))
            params, _ = translate_flux_weights(
                component_tensors(source / name, "transformer"))
            latent = jnp.asarray(unpacked(arrays[f"{name}.packed"], *grid))
            condition = DenoisingCondition(
                jnp.asarray(arrays[f"{name}.context"]), jnp.asarray(arrays[f"{name}.pooled"]),
                guidance=None if wanted else jnp.full((latent.shape[0],), 3.5, jnp.float32))
            match = "needs it" if wanted else "no guidance embedder"
            with pytest.raises(ValueError, match=match):
                model.apply({"params": params}, latent, jnp.asarray(arrays[f"{name}.times"]),
                            condition)


def test_the_rotary_table_is_the_sources_own_interleaved_pairs(source):
    """The table Flux rotates with: one angle per adjacent channel pair, laid
    out per axis, over the ids its pipeline writes."""
    from dew.nn.backbones.flux import apply_rotary, flux_positions, rotary_table

    positions = flux_positions(3, 2, 4)
    # The text sits at the origin and each patch carries its row and column.
    assert positions.shape == (10, 3)
    np.testing.assert_array_equal(positions[:4], np.zeros((4, 3), np.float32))
    np.testing.assert_array_equal(positions[4:, 1], np.repeat(np.arange(3), 2))
    np.testing.assert_array_equal(positions[4:, 2], np.tile(np.arange(2), 3))
    cos, sin = rotary_table(positions, (4, 4, 4))
    assert cos.shape == sin.shape == (10, 12)
    # Adjacent channels share an angle, and the text row rotates by nothing.
    np.testing.assert_allclose(cos[:, 0::2], cos[:, 1::2], atol=0)
    np.testing.assert_array_equal(cos[0], np.ones(12, np.float32))
    np.testing.assert_array_equal(sin[0], np.zeros(12, np.float32))
    # A quarter turn on one pair takes (x0, x1) to (-x1, x0).
    quarter = np.zeros((1, 1, 1, 2), np.float32)
    rotated = apply_rotary(jnp.asarray([[[[1.0, 2.0]]]]), jnp.zeros((1, 1, 1, 2)) + 0.0,
                           jnp.ones((1, 1, 1, 2)) + quarter)
    np.testing.assert_allclose(np.asarray(rotated).ravel(), [-2.0, 1.0], atol=0)
