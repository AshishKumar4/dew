"""UViT against Bao et al.'s own U-ViT (baofff/U-ViT, libs/uvit_t2i.py).

Over drawn weights, images, text states, times and a cotangent, Dew's UViT's output and the gradients of
the cotangent's product with respect to the image, the text and every parameter are as exact as U-ViT's
float32 run against its float64 run (check 1), each configuration of tools/uvit_worker.py's CASES its own
recording (tools/venv_reference.py's `uvit-<case>` sites), Dew's float64 run held to the recorded sketch.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_runs import assert_as_exact_as_recorded
from torch_layout import QKV, dew_leaf, join_qkv, node, split_qkv, torch_leaf

from dew.nn.backbones.uvit import UViT
from dew.nn.dit import TextContext

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "uvit"
UVIT_SEEDS = {"published": 8, "variant": 8}
"""Each configuration's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""
OUTPUTS = ("output", "gradients")
"""The leaves `dew` gives, each site's recorded leaves."""
CASES = {
    "published": {"img_size": 8, "patch_size": 2, "in_chans": 4, "embed_dim": 32, "depth": 4, "num_heads": 4,
                  "mlp_ratio": 4, "qkv_bias": False, "mlp_time_embed": False, "clip_dim": 12,
                  "num_clip_token": 5, "conv": True},
    "variant": {"img_size": 8, "patch_size": 4, "in_chans": 3, "embed_dim": 30, "depth": 2, "num_heads": 3,
                "mlp_ratio": 2, "qkv_bias": False, "mlp_time_embed": True, "clip_dim": 7,
                "num_clip_token": 3, "conv": False},
}
"""U-ViT's configurations (its `UViT` constructor's arguments), each drawn to its worker: `published`, its
mscoco configuration narrowed (mlp_time_embed off, the 3x3 output convolution on, as its constructor
defaults); `variant`, the time MLP on, no output convolution, an odd number of heads."""


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
        layer = {"fc1": "up_proj", "fc2": "down_proj"}[rest[1]]
        return ("mlp", layer, "kernel" if leaf == "weight" else "bias")
    if rest[1] == "proj":
        return ("attention", "to_out_0", "kernel" if leaf == "weight" else "bias")
    return ("attention", "qkv")


def to_dew(name: str, value: np.ndarray, config: dict) -> dict[tuple[str, ...], np.ndarray]:
    """One U-ViT tensor in Dew's layout: kernels transposed to [in, out],
    q, k and v split per head, the output projection read per head."""
    path = dew_path(name)
    heads = config["num_heads"]
    if path[-1] == "qkv":
        split = split_qkv(value, heads)
        return {(*path[:-1], part, "kernel"): split[index] for index, part in enumerate(QKV)}
    return {path: dew_leaf(path, value, heads)}


def from_dew(name: str, tree) -> np.ndarray:
    """The inverse of `to_dew`: a Dew gradient back in U-ViT's layout."""
    path = dew_path(name)
    if path[-1] == "qkv":
        return join_qkv([node(tree, (*path[:-1], part, "kernel")) for part in QKV])
    return torch_leaf(path, node(tree, path))


def state_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """U-ViT's state-dict names and shapes under `config` (libs/uvit_t2i.py's `UViT`)."""
    width, patch, channels = config["embed_dim"], config["patch_size"], config["in_chans"]
    hidden = width * config["mlp_ratio"]
    tokens = 1 + config["num_clip_token"] + (config["img_size"] // patch) ** 2
    shapes = {"patch_embed.proj.weight": (width, channels, patch, patch), "patch_embed.proj.bias": (width,),
              "context_embed.weight": (width, config["clip_dim"]), "context_embed.bias": (width,),
              "pos_embed": (1, tokens, width), "norm.weight": (width,), "norm.bias": (width,),
              "decoder_pred.weight": (patch * patch * channels, width),
              "decoder_pred.bias": (patch * patch * channels,)}
    if config["mlp_time_embed"]:
        shapes |= {"time_embed.0.weight": (4 * width, width), "time_embed.0.bias": (4 * width,),
                   "time_embed.2.weight": (width, 4 * width), "time_embed.2.bias": (width,)}
    if config["conv"]:
        shapes |= {"final_layer.weight": (channels, channels, 3, 3), "final_layer.bias": (channels,)}
    blocks = [f"in_blocks.{index}" for index in range(config["depth"] // 2)] + ["mid_block"]
    blocks += [f"out_blocks.{index}" for index in range(config["depth"] // 2)]
    for block in blocks:
        shapes |= {f"{block}.{name}": shape for name, shape in (
            ("norm1.weight", (width,)), ("norm1.bias", (width,)), ("attn.qkv.weight", (3 * width, width)),
            ("attn.proj.weight", (width, width)), ("attn.proj.bias", (width,)), ("norm2.weight", (width,)),
            ("norm2.bias", (width,)), ("mlp.fc1.weight", (hidden, width)), ("mlp.fc1.bias", (hidden,)),
            ("mlp.fc2.weight", (width, hidden)), ("mlp.fc2.bias", (width,)))}
        if block.startswith("out_blocks"):
            shapes |= {f"{block}.skip_linear.weight": (width, 2 * width),
                       f"{block}.skip_linear.bias": (width,)}
    return shapes


def drawing(case: str):
    """`draw(seed)` of `case`: weights at their initializer's scale, the norms' about one and the biases
    and positions off zero, two images, text states, times in (0, 1) and a cotangent of the output's
    shape, under U-ViT's names and layouts."""
    config = CASES[case]

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([118, seed])
        drawn = {}
        for name, shape in state_shapes(config).items():
            if name.endswith(("norm1.weight", "norm2.weight", "norm.weight")):
                value = rng.normal(1.0, 0.3, shape)
            elif name.endswith("bias") or name == "pos_embed":
                value = rng.normal(0.0, 0.3, shape)
            else:
                value = rng.normal(0.0, 1.0, shape) / np.sqrt(np.prod(shape[1:]))
            drawn[f"param.{name}"] = value.astype(np.float32)
        drawn |= {f"config.{key}": np.asarray(value) for key, value in config.items()}
        image = (2, config["in_chans"], config["img_size"], config["img_size"])
        return {**drawn, "image": rng.standard_normal(image).astype(np.float32),
                "context": rng.standard_normal((2, config["num_clip_token"], config["clip_dim"]))
                .astype(np.float32),
                "times": rng.uniform(0.0, 1.0, 2).astype(np.float32),
                "probe": rng.standard_normal(image).astype(np.float32)}
    return draw


def computing(case: str):
    """`dew(dtype, inputs)` of `case`: Dew's UViT's output `[B, C, H, W]` at `dtype` under a draw's
    weights, and the gradients `walked` gives, in its order."""
    config = CASES[case]
    names = sorted(state_shapes(config))
    model = UViT(output_channels=config["in_chans"], patch_size=config["patch_size"],
                 emb_features=config["embed_dim"], num_layers=config["depth"], num_heads=config["num_heads"],
                 mlp_ratio=config["mlp_ratio"], mlp_time_embed=config["mlp_time_embed"], conv=config["conv"],
                 text_tokens=config["num_clip_token"], image_size=config["img_size"], attention_impl="xla")

    def dew(dtype, inputs) -> dict:
        params: dict = {}
        for name in names:
            for path, array in to_dew(name, inputs[f"param.{name}"], config).items():
                node = params
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = jnp.asarray(array, dtype)
        image = jnp.asarray(inputs["image"], dtype).transpose(0, 2, 3, 1)
        context = jnp.asarray(inputs["context"], dtype)
        times, probe = jnp.asarray(inputs["times"], dtype), jnp.asarray(inputs["probe"], dtype)

        def forward(params, image, context):
            text = TextContext(context, jnp.ones(context.shape[:2], dtype))
            output = model.clone(dtype=dtype).apply({"params": params}, image, times, text)
            return output.transpose(0, 3, 1, 2)

        output = forward(params, image, context)
        grads = jax.grad(lambda *args: jnp.sum(forward(*args) * probe), argnums=(0, 1, 2))(
            params, image, context)
        native = [grads[1].transpose(0, 3, 1, 2), grads[2], *(from_dew(name, grads[0]) for name in names)]
        return {"output": output, "gradients": jnp.concatenate([jnp.ravel(part) for part in native])}
    return dew


def pipeline_arms(case: str):
    """tools/venv_reference.py's `uvit-<case>` site: its recording, draw, Dew run and leaves."""
    return FIXTURES / f"{case}.npz", drawing(case), computing(case), OUTPUTS


@pytest.mark.parametrize("case", list(CASES))
def test_the_output_and_every_gradient_are_as_exact_as_u_vits(case):
    """Dew's output and gradients as exact as U-ViT's float32 against float64 (check 1)."""
    fixture, draw, dew, _ = pipeline_arms(case)
    assert_as_exact_as_recorded(fixture, f"uvit-{case}", UVIT_SEEDS[case], draw, dew, f"U-ViT {case}")
