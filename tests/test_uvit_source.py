"""UViT against Bao et al.'s own U-ViT (baofff/U-ViT, libs/uvit_t2i.py).

tools/uvit_reference.py builds tiny text-conditioned U-ViTs at a pinned
commit, moves every parameter off its initialization, and records each
model's output on images, times and text states and, against a fixed
cotangent, the gradients of the image, the text and every parameter: in
float32 and in float64, the truth both float32 runs are measured from
(tests/fixtures/uvit/bao.npz). Dew's UViT on the same weights is held to
tests/reference_error.py's rule: no further from float64 than twice the
float32 reference.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn.backbones.uvit import UViT
from dew.nn.dit import TextContext

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "uvit" / "bao.npz"
CASES = ("published", "variant")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())


def dew_path(name: str) -> tuple[str, ...]:
    """The Dew parameter a U-ViT state-dict name lands on."""
    parts = name.split(".")
    if parts[0] in ("in_blocks", "out_blocks"):
        block, rest = f"{parts[0]}_{parts[1]}", parts[2:]
        if rest[0] == "skip_linear":
            return (f"skip_linear_{parts[1]}", "kernel" if rest[1] == "weight" else "bias")
        return (block, *_block_path(rest))
    if parts[0] == "mid_block":
        return ("mid_block", *_block_path(parts[1:]))
    leaf = parts[-1]
    if parts[0] == "patch_embed":
        return ("patch_embed", "Conv_0", "kernel" if leaf == "weight" else "bias")
    if parts[0] == "time_embed":
        return (f"time_embed_{parts[1]}", "kernel" if leaf == "weight" else "bias")
    if parts[0] == "norm":
        return ("norm", "scale" if leaf == "weight" else "bias")
    if parts[0] == "pos_embed":
        return ("pos_embed",)
    return (parts[0], "kernel" if leaf == "weight" else "bias")


def _block_path(rest: list[str]) -> tuple[str, ...]:
    leaf = rest[-1]
    if rest[0] in ("norm1", "norm2"):
        return (rest[0], "scale" if leaf == "weight" else "bias")
    if rest[0] == "mlp":
        layer = {"fc1": "layers_0", "fc2": "layers_2"}[rest[1]]
        return ("mlp", layer, "kernel" if leaf == "weight" else "bias")
    if rest[1] == "proj":
        return ("attention", "to_out_0", "kernel" if leaf == "weight" else "bias")
    return ("attention", "qkv")


def to_dew(name: str, value: np.ndarray, config: dict) -> dict[tuple[str, ...], np.ndarray]:
    """One U-ViT tensor in Dew's layout: kernels transposed to [in, out],
    q, k and v split per head, the output projection read per head."""
    path = dew_path(name)
    heads = config["num_heads"]
    width = config["embed_dim"]
    if path[-1] == "qkv":
        split = value.reshape(3, heads, width // heads, width).transpose(0, 3, 1, 2)
        return {(*path[:-1], part, "kernel"): split[index]
                for index, part in enumerate(("to_q", "to_k", "to_v"))}
    if path[-2:] == ("to_out_0", "kernel"):
        return {path: value.T.reshape(heads, width // heads, width)}
    if path[-1] == "kernel":
        return {path: value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T}
    return {path: value}


def from_dew(name: str, tree, config: dict) -> np.ndarray:
    """The inverse of `to_dew`: a Dew gradient back in U-ViT's layout."""
    path = dew_path(name)

    def leaf(path):
        node = tree
        for key in path:
            node = node[key]
        return np.asarray(node)

    width = config["embed_dim"]
    if path[-1] == "qkv":
        parts = [leaf((*path[:-1], part, "kernel")) for part in ("to_q", "to_k", "to_v")]
        return np.stack(parts).transpose(0, 2, 3, 1).reshape(3 * width, width)
    value = leaf(path)
    if path[-2:] == ("to_out_0", "kernel"):
        return value.reshape(width, width).T
    if path[-1] == "kernel":
        return value.transpose(3, 2, 0, 1) if value.ndim == 4 else value.T
    return value


@pytest.mark.parametrize("case", CASES)
def test_the_output_and_every_gradient_are_as_exact_as_u_vits(reference, case):
    arrays, meta = reference
    config = meta["cases"][case]

    def part(name):
        return arrays[f"{case}/{name}"]

    prefix = f"{case}/param."
    names = sorted(name.removeprefix(prefix) for name in arrays if name.startswith(prefix))
    params: dict = {}
    for name in names:
        value = part(f"param.{name}").view(ml_dtypes.bfloat16).astype(np.float32)
        for path, array in to_dew(name, value, config).items():
            node = params
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = jnp.asarray(array)
    model = UViT(output_channels=config["in_chans"], patch_size=config["patch_size"],
                 emb_features=config["embed_dim"], num_layers=config["depth"], num_heads=config["num_heads"],
                 mlp_ratio=config["mlp_ratio"], mlp_time_embed=config["mlp_time_embed"], conv=config["conv"],
                 text_tokens=config["num_clip_token"], image_size=config["img_size"], attention_impl="xla")
    image = jnp.asarray(part("image").transpose(0, 2, 3, 1))
    context = jnp.asarray(part("context"))
    times = jnp.asarray(part("times"))
    probe = part("probe").transpose(0, 2, 3, 1)

    def forward(params, image, context):
        text = TextContext(context, jnp.ones(context.shape[:2]))
        return model.apply({"params": params}, image, times, text)

    output = forward(params, image, context)
    grads = jax.grad(lambda *args: jnp.sum(forward(*args) * probe), argnums=(0, 1, 2))(params, image, context)
    assert_as_exact_as_the_reference(np.asarray(output).transpose(0, 3, 1, 2), part("fp32.output"),
                                     part("fp64.output"), f"{case} output")

    def source_order(precision: str) -> np.ndarray:
        leaves = [part(f"{precision}.grad_image"), part(f"{precision}.grad_context")]
        leaves += [part(f"{precision}.grad_param.{name}") for name in names]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    native = [np.asarray(grads[1]).transpose(0, 3, 1, 2), np.asarray(grads[2])]
    native += [from_dew(name, grads[0], config) for name in names]
    native_order = np.concatenate([np.ravel(leaf) for leaf in native])
    assert_as_exact_as_the_reference(native_order, source_order("fp32"), source_order("fp64"),
                                     f"{case} gradients")
