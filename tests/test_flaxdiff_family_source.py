"""The DiT family and the UNets, image and video, against FlaxDiff's own models.

tools/flaxdiff_family_worker.py builds tiny SimpleDiT, SimpleUDiT, SimpleMMDiT, HierarchicalMMDiT, Unet,
VideoDiT and UNet3D models from FlaxDiff's model modules at a pinned commit (tools/flaxdiff_family_cases.py's
CASES). Over drawn weights, images (or clips), times, text states and cotangents, Dew's models' outputs and
the gradients of the cotangent's product with respect to the image, the text and every parameter are as
exact as FlaxDiff's float32 run against its float64 run (check 1), each case its own recording
(tools/venv_reference.py's `flaxdiff-<model>-<config>` sites), Dew's float64 run held to the recorded sketch,
with the Fourier table FlaxDiff draws as Dew's constant.

The DiTs average the text over every position, as FlaxDiff's do: Dew builds them with
`text_pooling="all"`, and is handed a padded mask the pooling must ignore. The MM-DiTs and the VideoDiT,
which pool the real tokens, get no padding.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_runs import assert_as_exact_as_recorded
from torch_layout import node

from dew.interop.flaxdiff import SCAN_PROJECTION, scan_projection
from dew.nn.attention_blocks import Stage
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.mmdit import HierarchicalMMDiT, SimpleMMDiT
from dew.nn.backbones.unet import Unet
from dew.nn.backbones.unet3d import UNet3D
from dew.nn.backbones.uvit import SimpleUDiT
from dew.nn.backbones.video_dit import VideoDiT
from dew.nn.dit import TextContext
from tools.flaxdiff_family_cases import CASES

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "flaxdiff_family"
LAYOUT = json.loads((FIXTURES / "layout.json").read_text())
"""Each case's FlaxDiff parameters' shapes by '/'-joined path, and the Fourier table its FourierEmbedding
draws, the model's constant (tools/flaxdiff_family_worker.py --layout)."""
FAMILY_SEEDS = dict.fromkeys(CASES, 8)
"""Each case's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""
OUTPUTS = ("output", "gradients")
"""The leaves `dew` gives, each site's recorded leaves."""
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
FEED_FORWARD = {("net_0", "proj"): ("gate_up_proj",), ("net_2",): ("down_proj",)}
"""FlaxDiff's GEGLU feed-forward, diffusers', under `GatedMLP`'s names (`geglu_feed_forward`)."""
# A scan order's patch projection lands on Dew's patch convolution (`scan_projection`).
SCAN_PROJECTED = ("patch_embed", "Conv_0")
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
    if SCAN_PROJECTION in path:
        at = path.index(SCAN_PROJECTION)
        path = (*path[:at], *SCAN_PROJECTED, *path[at + 1:])
    if cls == "SimpleUDiT" and path[0] in UDIT_MOVED:
        return (*UDIT_MOVED[path[0]], *path[1:])
    if mmdit_block(cls, name):
        block, module, *rest = path
        return (block, *MMDIT_MODULES[module], *(MMDIT_LAYERS.get(part, part) for part in rest))
    if "mlp" in path[:-1]:
        # A DiT block's MLP, FlaxDiff's `nn.Sequential`, under `GatedMLP`'s names.
        at = path.index("mlp") + 1
        path = (*path[:at], MMDIT_LAYERS.get(path[at], path[at]), *path[at + 1:])
    if cls in UNETS:
        if "ff" in path:
            at = path.index("ff") + 1
            moved = next(names for names in FEED_FORWARD if path[at:at + len(names)] == names)
            path = (*path[:at], *FEED_FORWARD[moved], *path[at + len(moved):])
        # FlaxDiff's ConvLayer wraps the convolution Dew's Conv is, and its
        # middle stage projects with a 1x1 convolution where Dew's is dense.
        return tuple(PROJECTIONS.get(part, part.replace("ConvLayer_", "Conv_")) for part in path
                     if part != "conv")
    return path


def to_dew(cls: str, name: str, value: np.ndarray, patch_size: int | None = None) -> np.ndarray:
    """A FlaxDiff parameter in Dew's layout: a 1x1 convolution's kernel as
    the dense kernel it is, a scan order's patch projection as the patch
    convolution's kernel, an MM-DiT's per-head projections flat, and its
    modulation pieces in SD3's order."""
    if PROJECTIONS.keys() & set(name.split("/")):
        return value[0, 0]
    if name.endswith(f"{SCAN_PROJECTION}/kernel"):
        assert patch_size is not None
        return scan_projection(value, patch_size)
    if not mmdit_block(cls, name):
        return value
    module, leaf = name.split("/")[1], name.split("/")[-1]
    if module.endswith("_ada"):
        pieces = jnp.split(value, 6, axis=-1)
        return jnp.concatenate([pieces[index] for index in MODULATION_ORDER], axis=-1)
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
        pieces = jnp.split(value, 6, axis=-1)
        return jnp.concatenate([pieces[MODULATION_ORDER.index(index)] for index in range(6)], axis=-1)
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


def drawing(case: str):
    """`draw(seed)` of `case`: weights at their initializer's scale, the norms' scales about one and the
    biases off zero, two images (or clips), times, text states and a cotangent of the output, under FlaxDiff's
    paths."""
    _, _, config, image, text = CASES[case]

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([269, seed])
        drawn = {}
        for path, shape in LAYOUT[case]["params"].items():
            leaf = path.split("/")[-1]
            if leaf == "scale":
                value = rng.normal(1.0, 0.3, shape)
            elif leaf == "kernel":
                fan_in = shape[0] if len(shape) == 3 else np.prod(shape[:-1])
                value = rng.normal(0.0, 1.0, shape) / np.sqrt(fan_in)
            else:
                value = rng.normal(0.0, 0.3, shape)
            drawn[f"param.{path}"] = value.astype(np.float32)
        return {**drawn, "image": rng.standard_normal((2, *image)).astype(np.float32),
                "times": rng.uniform(-1.5, 1.5, 2).astype(np.float32),
                "text": rng.standard_normal((2, *text)).astype(np.float32),
                "probe": rng.standard_normal((2, *image[:-1], config["output_channels"])).astype(np.float32)}
    return draw


def computing(case: str):
    """`dew(dtype, inputs)` of `case`: Dew's model's output at `dtype` under a draw's weights, and the
    gradients tools/flaxdiff_family_worker.py gives, in its order."""
    _, cls, config, _, _ = CASES[case]
    names = sorted(LAYOUT[case]["params"])
    model = dew_model(cls, config)
    table = np.asarray(LAYOUT[case]["fourier"], np.float32)

    def dew(dtype, inputs) -> dict:
        params = nest({dew_path(cls, name): to_dew(cls, name, jnp.asarray(inputs[f"param.{name}"], dtype),
                                                   config.get("patch_size")) for name in names})
        constants = nest({(*fourier_path(cls), "frequencies"): jnp.asarray(table, dtype)})
        image, text, times = (jnp.asarray(inputs[name], dtype) for name in ("image", "text", "times"))
        mask = np.ones(text.shape[:2], np.float32)
        if cls in ("SimpleDiT", "SimpleUDiT"):
            mask[0, -2:] = 0

        def forward(params, image, text):
            return model.apply({"params": params, "constants": constants}, image, times,
                               TextContext(text, jnp.asarray(mask, dtype)))

        output, pullback = jax.vjp(forward, params, image, text)
        grad_params, grad_image, grad_text = pullback(jnp.asarray(inputs["probe"], dtype))
        native = [grad_image, grad_text, *(from_dew(cls, name, node(grad_params, dew_path(cls, name)),
                                                    LAYOUT[case]["params"][name]) for name in names)]
        return {"output": output, "gradients": jnp.concatenate([jnp.ravel(value) for value in native])}
    return dew


def site(case: str) -> str:
    """tools/venv_reference.py's site of `case`: `flaxdiff-<model>-<config>`."""
    return "flaxdiff-" + case.replace("/", "-")


def pipeline_arms(case: str):
    """tools/venv_reference.py's site of `case`: its recording, draw, Dew run and leaves."""
    return FIXTURES / f"{case.replace('/', '-')}.npz", drawing(case), computing(case), OUTPUTS


@pytest.mark.parametrize("case", list(CASES))
def test_the_output_and_every_gradient_are_as_exact_as_flaxdiffs(case):
    """Dew's output and gradients as exact as FlaxDiff's float32 against float64 (check 1)."""
    fixture, draw, dew, _ = pipeline_arms(case)
    assert_as_exact_as_recorded(fixture, site(case), FAMILY_SEEDS[case], draw, dew, f"FlaxDiff {case}")
