"""The DiT family and the UNets, image and video, against FlaxDiff's own models.

tools/flaxdiff_family_reference.py builds tiny SimpleDiT, SimpleUDiT,
SimpleMMDiT, HierarchicalMMDiT, Unet, VideoDiT and UNet3D models from
FlaxDiff's model modules at a pinned commit, moves every parameter off its
initialization, and records each model's output on images (or clips), times
and text states and, against a fixed cotangent, the gradients of the image,
the text and every parameter: in float32 and in float64, the truth both
float32 runs are measured from (tests/fixtures/flaxdiff_family/family.npz). Dew's
models on the same weights and Fourier table are held to
tests/reference_error.py's rule: no further from float64 than twice the
float32 reference. The UNet variant uses the same rule over 52 spatial
channel orders, since a single output or gradient draw has too few
independent rounding directions. Each order's float64 reference computes
the original output and gradient within 1e-12 relative error.

The DiTs average the text over every position, as FlaxDiff's do: Dew builds
them with `text_pooling="all"`, and is handed a padded mask the pooling
must ignore. The MM-DiTs and the VideoDiT, which pool the real tokens, get
no padding.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_as_exact_over_orders, distance

from dew.nn.attention import Stage
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.mmdit import HierarchicalMMDiT, SimpleMMDiT
from dew.nn.backbones.unet import Unet
from dew.nn.backbones.unet3d import UNet3D
from dew.nn.backbones.uvit import SimpleUDiT
from dew.nn.backbones.video_dit import VideoDiT
from dew.nn.dit import TextContext

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flaxdiff_family" / "family.npz"
CASES = ("simple_dit/published", "simple_dit/variant", "simple_udit/published", "simple_udit/variant",
         "simple_mmdit/published", "simple_mmdit/variant", "hierarchical_mmdit/published",
         "hierarchical_mmdit/variant", "unet/published", "unet/variant", "video_dit/published",
         "video_dit/variant", "unet_3d/published", "unet_3d/variant")
MODELS = {"SimpleDiT": SimpleDiT, "SimpleUDiT": SimpleUDiT, "SimpleMMDiT": SimpleMMDiT,
          "HierarchicalMMDiT": HierarchicalMMDiT, "Unet": Unet, "VideoDiT": VideoDiT, "UNet3D": UNet3D}
UNETS = ("Unet", "UNet3D")
# FlaxDiff's SimpleUDiT holds these at its root; Dew nests them under the
# conditioning embed and the output head.
UDIT_MOVED = {"time_embed": ("conditioning", "time_embed"),
              "text_proj": ("conditioning", "text_context_proj"),
              "final_norm": ("output", "final_norm"), "final_proj": ("output", "final_proj")}
DIT_POOLING = {"text_pooling": "all"}
PROJECTIONS = {"project_in_conv": "project_in", "project_out_conv": "project_out"}
MMDITS = ("SimpleMMDiT", "HierarchicalMMDiT")
# FlaxDiff's MM-DiT block module by module, as Dew's `DoubleStreamBlock`
# names it: one joint attention holds both streams' projections and norms.
MMDIT_MODULES = {"img_ada": ("norm1",), "txt_ada": ("norm1_context",),
                 "img_to_q": ("attn", "to_q"), "img_to_k": ("attn", "to_k"), "img_to_v": ("attn", "to_v"),
                 "txt_to_q": ("attn", "add_q_proj"), "txt_to_k": ("attn", "add_k_proj"),
                 "txt_to_v": ("attn", "add_v_proj"), "img_out": ("attn", "to_out_0"),
                 "txt_out": ("attn", "to_add_out"), "img_q_norm": ("attn", "norm_q"),
                 "img_k_norm": ("attn", "norm_k"), "txt_q_norm": ("attn", "norm_added_q"),
                 "txt_k_norm": ("attn", "norm_added_k"), "img_mlp": ("ff",), "txt_mlp": ("ff_context",)}
MMDIT_LAYERS = {"ada_proj": "linear", "layers_0": "up_proj", "layers_2": "down_proj"}
# FlaxDiff's six modulation pieces (scale_mlp, shift_mlp, gate_mlp, scale,
# shift, gate) in SD3's order (shift, scale, gate, shift_mlp, scale_mlp, gate_mlp).
MODULATION_ORDER = (4, 3, 5, 1, 0, 2)


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())


def dew_model(cls: str, config: dict):
    """Dew's model for FlaxDiff's class and config."""
    fields = {key: value for key, value in config.items()
              if key not in ("attention_impl", "force_fp32_for_softmax", "use_hilbert", "use_zigzag")}
    if config.get("use_hilbert") or config.get("use_zigzag"):
        fields["scan_order"] = "hilbert" if config.get("use_hilbert") else "zigzag"
    if cls in ("SimpleDiT", "SimpleUDiT"):
        fields.update(DIT_POOLING)
    if cls == "HierarchicalMMDiT":
        fields.update({key: tuple(fields[key]) for key in ("emb_features", "num_layers", "num_heads")})
    if cls in UNETS:
        fields["feature_depths"] = tuple(fields["feature_depths"])
        fields["attention_configs"] = tuple(None if stage is None else Stage(**stage)
                                            for stage in fields["attention_configs"])
    return MODELS[cls](**fields, attention_impl="reference")


def mmdit_block(cls: str, name: str) -> bool:
    """Whether a FlaxDiff parameter is one of an MM-DiT block's."""
    return cls in MMDITS and "_block_" in name.split("/")[0]


def dew_path(cls: str, name: str) -> tuple[str, ...]:
    """The Dew parameter a FlaxDiff parameter path lands on."""
    path = tuple(name.split("/"))
    if cls == "SimpleUDiT" and path[0] in UDIT_MOVED:
        return (*UDIT_MOVED[path[0]], *path[1:])
    if mmdit_block(cls, name):
        block, module, *rest = path
        return (block, *MMDIT_MODULES[module], *(MMDIT_LAYERS.get(part, part) for part in rest))
    if cls in UNETS:
        # FlaxDiff's ConvLayer wraps the convolution Dew's Conv is, and its
        # middle stage projects with a 1x1 convolution where Dew's is dense.
        return tuple(PROJECTIONS.get(part, part.replace("ConvLayer_", "Conv_")) for part in path
                     if part != "conv")
    return path


def to_dew(cls: str, name: str, value: np.ndarray) -> np.ndarray:
    """A FlaxDiff parameter in Dew's layout: a 1x1 convolution's kernel as
    the dense kernel it is, an MM-DiT's per-head projections flat, and its
    modulation pieces in SD3's order."""
    if PROJECTIONS.keys() & set(name.split("/")):
        return value[0, 0]
    if not mmdit_block(cls, name):
        return value
    module, leaf = name.split("/")[1], name.split("/")[-1]
    if module.endswith("_ada"):
        pieces = np.split(value, 6, axis=-1)
        return np.concatenate([pieces[index] for index in MODULATION_ORDER], axis=-1)
    if module[4:] in ("to_q", "to_k", "to_v"):
        return value.reshape(*value.shape[:-2], -1)
    if module[4:] == "out" and leaf == "kernel":
        return value.reshape(-1, value.shape[-1])
    return value


def from_dew(cls: str, name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The inverse of `to_dew` onto FlaxDiff's `shape`, for a Dew gradient."""
    if PROJECTIONS.keys() & set(name.split("/")):
        return value[None, None]
    if mmdit_block(cls, name) and name.split("/")[1].endswith("_ada"):
        pieces = np.split(value, 6, axis=-1)
        return np.concatenate([pieces[MODULATION_ORDER.index(index)] for index in range(6)], axis=-1)
    return value.reshape(shape)


def fourier_path(cls: str) -> tuple[str, ...]:
    return ("FourierEmbedding_0",) if cls in UNETS else ("conditioning", "time_embed", "layers_0")


def nest(flat: dict[tuple[str, ...], np.ndarray]) -> dict:
    tree: dict = {}
    for path, value in flat.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = jnp.asarray(value)
    return tree


def unet_channels(name: str, value: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Reorder the variant UNet's eight spatial channels, keeping each
    concatenated skip stream whole. Its norms are RMS norms. Every attention
    stage projects in and out, so only the projections' spatial sides move;
    the attention heads, time embedding and text keep their own order."""
    parts = name.split("/")
    if "Attention" in parts or parts[0] == "TimeProjection_0":
        return value
    if any(part.startswith("project_in") for part in parts):
        return np.take(value, order, axis=-2)
    if any(part.startswith("project_out") for part in parts):
        return np.take(value, order, axis=-1)
    for axis in range(value.ndim):
        size = value.shape[axis]
        if size in (len(order), 2 * len(order)):
            channels = np.concatenate([order + start for start in range(0, size, len(order))])
            value = np.take(value, channels, axis=axis)
    return value


@pytest.mark.parametrize("case", CASES)
def test_the_output_and_every_gradient_are_as_exact_as_flaxdiffs(reference, case):
    arrays, meta = reference
    cls, config = meta["cases"][case]["class"], meta["cases"][case]["config"]

    def part(name):
        return arrays[f"{case}/{name}"]

    prefix = f"{case}/param."
    names = sorted(name.removeprefix(prefix) for name in arrays if name.startswith(prefix))
    params = nest({dew_path(cls, name): to_dew(cls, name, part(f"param.{name}").view(ml_dtypes.bfloat16)
                                                    .astype(np.float32))
                   for name in names})
    constants = nest({(*fourier_path(cls), "frequencies"): part("fourier_table")})
    model = dew_model(cls, config)
    image, text, times = (jnp.asarray(part(name)) for name in ("image", "text", "times"))
    mask = np.ones(text.shape[:2], np.float32)
    if cls in ("SimpleDiT", "SimpleUDiT"):
        mask[0, -2:] = 0

    def forward(params, image, text):
        return model.apply({"params": params, "constants": constants}, image, times,
                           TextContext(text, jnp.asarray(mask)))

    def evaluated(params):
        output, pullback = jax.vjp(forward, params, image, text)
        return output, pullback(jnp.asarray(part("probe")))

    def leaf(tree, path):
        for key in path:
            tree = tree[key]
        return np.asarray(tree)

    def source_order(precision: str) -> np.ndarray:
        leaves = [part(f"{precision}.grad_image"), part(f"{precision}.grad_text")]
        leaves += [part(f"{precision}.grad_param.{name}") for name in names]
        return np.concatenate([np.ravel(value) for value in leaves])

    def gradient_order(gradients, back=None):
        grad_params, grad_image, grad_text = gradients
        native = [np.asarray(grad_image), np.asarray(grad_text)]
        for name in names:
            value = from_dew(cls, name, leaf(grad_params, dew_path(cls, name)), part(f"param.{name}").shape)
            native.append(value if back is None else unet_channels(name, value, back))
        return np.concatenate([np.ravel(value) for value in native])

    if case == "unet/variant":
        # The variant concentrates its rounding in a few directions. Spatial
        # channel orders move those roundings without moving the function;
        # the reference tool checks every order's output and gradient in float64.
        with np.load(FIXTURE.with_name("orders.npz")) as drawn:
            orders, reference_output, reference_gradients = (drawn[key] for key in
                                                            ("orders", "output", "gradients"))
        run = jax.jit(evaluated)
        output_distances, gradient_distances = [], []
        truth_gradients = source_order("fp64")
        for order in orders:
            moved = nest({dew_path(cls, name): to_dew(cls, name, unet_channels(
                name, part(f"param.{name}").view(ml_dtypes.bfloat16).astype(np.float32), order))
                          for name in names})
            output, gradients = run(moved)
            output_distances.append(distance(output, part("fp64.output")))
            gradient_distances.append(distance(gradient_order(gradients, np.argsort(order)), truth_gradients))
        assert_as_exact_over_orders(output_distances, reference_output, f"{case} output")
        assert_as_exact_over_orders(gradient_distances, reference_gradients, f"{case} gradients")
    else:
        output, gradients = evaluated(params)
        assert_as_exact_as_the_reference(output, part("fp32.output"), part("fp64.output"), f"{case} output")
        assert_as_exact_as_the_reference(
            gradient_order(gradients), source_order("fp32"), source_order("fp64"), f"{case} gradients")
