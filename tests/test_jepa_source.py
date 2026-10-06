"""JEPA's encoder and predictor against V-JEPA's own (facebookresearch/jepa).

tools/vjepa_reference.py builds tiny image-mode V-JEPA encoders and
mask-token predictors at a pinned commit, moves every trainable parameter
off its initialization, and records the encoder's embeddings of each
image's context patches, the predictor's predictions for its target patches
and, against fixed cotangents on both, the gradients of the images and every
parameter: in float32 and in float64, the truth both float32 runs are
measured from (tests/fixtures/jepa/vjepa.npz). Its MLPs run V-JEPA's own
exact GELU, as Dew's JEPA does. Dew's JepaEncoder and JepaPredictor on the same weights are
held to tests/reference_error.py's rule: no further from float64 than twice
the float32 reference.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn.backbones.jepa import JepaEncoder, JepaPredictor

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jepa" / "vjepa.npz"
CASES = ("published", "variant")
TOP = {"patch_embed": ("embed", "patch_embed", "Conv_0"), "norm": ("norm",), "predictor_embed": ("proj_in",),
       "predictor_norm": ("norm",), "predictor_proj": ("proj_out",)}
NORM = {"weight": "scale", "bias": "bias"}
DENSE = {"weight": "kernel", "bias": "bias"}


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())


def dew_path(name: str) -> tuple[str, ...]:
    """The Dew parameter a V-JEPA name (`encoder.` or `predictor.` first)
    lands on, under that model's own tree."""
    model, first, *rest = name.split(".")
    if first == "mask_tokens":
        return (model, "mask_token")
    if first in ("blocks", "predictor_blocks"):
        index, layer, *rest = rest
        block = (model, "stack", f"block_{index}")
        if layer.startswith("norm"):
            return (*block, layer, NORM[rest[-1]])
        if layer == "mlp":
            return (*block, "mlp", {"fc1": "layers_0", "fc2": "layers_2"}[rest[0]], DENSE[rest[-1]])
        if rest[0] == "proj":
            return (*block, "attention", "to_out_0", DENSE[rest[-1]])
        return (*block, "attention", "qkv", rest[-1])
    return (model, *TOP[first], (NORM if first.endswith("norm") else DENSE)[rest[-1]])


def to_dew(name: str, value: np.ndarray, heads: int) -> dict[tuple[str, ...], np.ndarray]:
    """One V-JEPA tensor in Dew's layout: kernels transposed to [in, out],
    q, k and v split per head, the output projection read per head."""
    path = dew_path(name)
    if path[-2] == "qkv":
        width = value.shape[-1] if value.ndim == 2 else value.shape[0] // 3
        parts = value.reshape(3, heads, width // heads, *value.shape[1:])
        if value.ndim == 2:
            parts = parts.transpose(0, 3, 1, 2)
        return {(*path[:-2], part, "kernel" if value.ndim == 2 else "bias"): parts[index]
                for index, part in enumerate(("to_q", "to_k", "to_v"))}
    if path[-2:] == ("to_out_0", "kernel"):
        return {path: value.T.reshape(heads, -1, value.shape[0])}
    if path[-1] == "kernel":
        return {path: value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T}
    return {path: value}


def from_dew(name: str, tree, heads: int) -> np.ndarray:
    """The inverse of `to_dew`: a Dew gradient back in V-JEPA's layout."""
    path = dew_path(name)

    def leaf(path):
        node = tree
        for key in path:
            node = node[key]
        return np.asarray(node)

    if path[-2] == "qkv":
        if path[-1] == "weight":
            parts = [leaf((*path[:-2], part, "kernel")) for part in ("to_q", "to_k", "to_v")]
            return np.stack(parts).transpose(0, 2, 3, 1).reshape(-1, parts[0].shape[0])
        return np.concatenate([leaf((*path[:-2], part, "bias")).ravel() for part in ("to_q", "to_k", "to_v")])
    value = leaf(path)
    if path[-2:] == ("to_out_0", "kernel"):
        return value.reshape(-1, value.shape[-1]).T
    if path[-1] == "kernel":
        return value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
    return value


@pytest.mark.parametrize("case", CASES)
def test_the_embeddings_predictions_and_every_gradient_are_as_exact_as_v_jepas(reference, case):
    arrays, meta = reference
    config = meta["cases"][case]

    def part(name):
        return arrays[f"{case}/{name}"]

    prefix = f"{case}/param."
    names = [name.removeprefix(prefix) for name in arrays if name.startswith(prefix)]
    heads = {"encoder": config["num_heads"], "predictor": config["pred_num_heads"]}
    params: dict = {"encoder": {}, "predictor": {}}
    for name in names:
        value = part(f"param.{name}").view(ml_dtypes.bfloat16).astype(np.float32)
        for path, array in to_dew(name, value, heads[name.split(".")[0]]).items():
            node = params
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = jnp.asarray(array)
    grid = config["img_size"] // config["patch_size"]
    encoder = JepaEncoder(patch_size=config["patch_size"], emb_features=config["embed_dim"],
                          num_layers=config["depth"], num_heads=config["num_heads"],
                          mlp_ratio=config["mlp_ratio"], norm_epsilon=config["eps"], attention_impl="xla")
    predictor = JepaPredictor(grid=(grid, grid), emb_features=config["embed_dim"],
                              predictor_features=config["predictor_embed_dim"],
                              num_layers=config["pred_depth"], num_heads=config["pred_num_heads"],
                              mlp_ratio=config["mlp_ratio"],
                              norm_epsilon=config["eps"], attention_impl="xla")
    images = jnp.asarray(part("images").transpose(0, 2, 3, 1))
    context_idx, target_idx = jnp.asarray(part("context_idx")), jnp.asarray(part("target_idx"))
    probes = part("probe_context"), part("probe_predictions")

    def forward(params, images):
        context = encoder.apply({"params": params["encoder"]}, images, context_idx)
        return context, predictor.apply({"params": params["predictor"]}, context, context_idx, target_idx)

    context, predictions = forward(params, images)
    def objective(params, images):
        return sum(jnp.sum(out * probe) for out, probe in zip(forward(params, images), probes, strict=True))

    grads = jax.grad(objective, argnums=(0, 1))(params, images)
    assert_as_exact_as_the_reference(np.asarray(context), part("fp32.context"), part("fp64.context"),
                                     f"{case} context embeddings")
    assert_as_exact_as_the_reference(np.asarray(predictions), part("fp32.predictions"),
                                     part("fp64.predictions"), f"{case} predictions")

    def source_order(precision: str) -> np.ndarray:
        leaves = [part(f"{precision}.grad_images")]
        leaves += [part(f"{precision}.grad_param.{name}") for name in names]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    native = [np.asarray(grads[1]).transpose(0, 3, 1, 2)]
    native += [from_dew(name, grads[0], heads[name.split(".")[0]]) for name in names]
    native_order = np.concatenate([np.ravel(leaf) for leaf in native])
    assert_as_exact_as_the_reference(native_order, source_order("fp32"), source_order("fp64"),
                                     f"{case} gradients")
