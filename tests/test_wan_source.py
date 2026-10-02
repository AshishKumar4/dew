"""Native Wan 2.1 against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_wan_reference.py` builds tiny `WanTransformer3DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and calls each as `WanPipeline` calls it,
recording the output and, against a fixed cotangent, the gradients of the
latents, the prompt states and every parameter: once in float32 and once in
float64, the truth both float32 runs are measured from.

The native transformer takes channels-last latents and Dew's model time,
which is the timestep the source's pipeline passes, and returns the flow the
source returns. Dew in float32 is held to tests/reference_error.py's rule,
within twice the reference's own RMS distance from float64. Observed on CPU
(Dew's RMS error over the reference's): published forward 1.01, gradients
1.01; variant forward 0.96, gradients 0.99. The truth is float64 throughout:
Dew run in float64 with a float64 rotary table lands 3e-16 (RMS) from it.
"""

import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, translate_wan_weights, wan_fields
from dew.nn.backbones.wan import WanTransformer

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "tests/fixtures/hf/wan-source"
CASES = ("published", "variant")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wan-transformer")
    with tarfile.open(ROOT / "tests/fixtures/wan_transformer.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module")
def arrays(source):
    with np.load(source / "wan_transformer.npz") as loaded:
        return dict(loaded)


def channels_last(value):
    """`[B, C, F, H, W]` to `[B, F, H, W, C]`."""
    return np.moveaxis(np.asarray(value), 1, -1)


def load(source, name):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = WanTransformer(**wan_fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = translate_wan_weights(component_tensors(source / name, "transformer"))
    return model, params, layouts


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert {layout.name for layout in layouts} == {f"transformer/{key}" for key in tensors}
    for layout in layouts:
        written = layout.export({"params": params})
        assert written.tobytes() == tensors[layout.name.removeprefix("transformer/")].tobytes(), layout.name


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_are_as_exact_as_the_source(source, arrays, name):
    """The flow, and the gradients of `sum(flow * probe)` with respect to the
    latents, the prompt states and every parameter, each within twice the
    float32 source's RMS distance from its float64 run."""
    model, params, layouts = load(source, name)
    latents = jnp.asarray(channels_last(arrays[f"{name}.latents"]))
    times = jnp.asarray(arrays[f"{name}.times"])
    context = jnp.asarray(arrays[f"{name}.context"])
    probe = channels_last(arrays[f"{name}.probe"])

    def flow(params, latents, context):
        return model.apply({"params": params}, latents, times, DenoisingCondition(context))

    output = flow(params, latents, context)
    grad_params, grad_latents, grad_context = jax.grad(
        lambda *args: jnp.sum(flow(*args) * probe), argnums=(0, 1, 2))(params, latents, context)
    assert_as_exact_as_the_reference(np.moveaxis(np.asarray(output), -1, 1), arrays[f"{name}.fp32.output"],
                                     arrays[f"{name}.fp64.output"], f"{name} flow")

    def source_order(precision: str) -> np.ndarray:
        leaves = [arrays[f"{name}.{precision}.grad_latents"], arrays[f"{name}.{precision}.grad_context"]]
        leaves += [arrays[f"{name}.{precision}.grad_param.{layout.name.removeprefix('transformer/')}"]
                   for layout in layouts]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    native = [np.moveaxis(np.asarray(grad_latents), -1, 1), np.asarray(grad_context)]
    native += [layout.export({"params": grad_params}) for layout in layouts]
    native_order = np.concatenate([np.ravel(leaf) for leaf in native])
    assert_as_exact_as_the_reference(native_order, source_order("fp32"), source_order("fp64"),
                                     f"{name} gradients")


@pytest.mark.parametrize("change", [
    {"image_dim": 1280}, {"added_kv_proj_dim": 5120}, {"pos_embed_seq_len": 514}, {"qk_norm": "rms_norm"},
    {"patch_size": [2, 2]}, {"attention_head_dim": 13},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "published" / "transformer" / "config.json").read_text())
    wan_fields(config)
    with pytest.raises(ValueError):
        wan_fields({**config, **change})


def test_every_tensor_of_the_published_checkpoint_has_a_native_place():
    """Wan2.1-T2V-1.3B's own config builds the native tree, and every tensor
    its shard index names lands on a leaf of it."""
    config = json.loads((PUBLISHED / "transformer" / "config.json").read_text())
    names = json.loads((PUBLISHED / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
                       .read_text())["weight_map"]
    model = WanTransformer(**wan_fields(config))
    shapes = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 1, 2, 2, config["in_channels"])), jnp.zeros((1,)),
        DenoisingCondition(jnp.zeros((1, 1, config["text_dim"])))))
    leaves = {tuple(key.key for key in path) for path, _ in jax.tree_util.tree_leaves_with_path(shapes)}
    tree, _ = translate_wan_weights({name: np.zeros((1, 1) if name.endswith("weight") else (1,), np.float32)
                                     for name in names})
    placed = {tuple(key.key for key in path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)}
    assert {("params", *path) for path in placed} == leaves
