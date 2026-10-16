"""What every vision family shares: the tower and projector bases, the
geometry a conditioner reads off a tower, the vision-config readers and the
projector tensor table.
"""

import dataclasses
from collections.abc import Collection, Mapping

from flax import linen as nn

from dew import records
from dew.interop.decoder_parts import read_rope_theta

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


class ProjectorBase:
    """One projector kind's value: its fields, and how it builds its module.

    Each kind is a frozen dataclass of the reference's field names, with an
    alias in `dew.registry.projectors` (`"gemma"`). `build` turns the value
    into the Flax module. A record that names no known kind raises.
    """

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a projector kind but builds no module")


def head_width(hidden_size: int, num_heads: int) -> int:
    """Each head's width where `num_heads` split a tower's `hidden_size`
    evenly, and a refusal where they leave a remainder."""
    if hidden_size % num_heads:
        raise ValueError(f"hidden_size ({hidden_size}) must split over num_heads ({num_heads})")
    return hidden_size // num_heads


@dataclasses.dataclass(frozen=True)
class TowerGeometry:
    """The shapes a conditioner has to invent to create a tower's media leaves.

    A token-only init has no batch, so `VisionConditioner` and
    `AudioConditioner` build one smallest input the tower accepts. What that
    is differs by kind: a fixed-resolution tower states its `image_size`, a
    patch-and-pool tower states the patch and block that make one, and an
    audio tower states how many mel bins a frame carries. None means this
    tower has no such field, and the conditioner uses its own default.
    """

    image_size: int | None = None
    patch_size: int | None = None
    block_size: int | None = None
    channels: int | None = None
    mel_features: int | None = None


class TowerBase:
    """One tower kind's value: its fields, and how it builds its module."""

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a tower kind but builds no module")

    def geometry(self) -> TowerGeometry:
        """What an initialising input has to look like for this tower.

        Nothing by default: a tower states the fields it has, and a
        conditioner reads them here instead of asking the value at runtime
        whether it carries each one.
        """
        return TowerGeometry()


_PROJECTOR_PATHS: dict[str, dict[str, tuple[str, ...]]] = {
    "gemma": {
        "mm_soft_emb_norm.weight": ("mm_soft_emb_norm", "scale"),
        "mm_input_projection_weight": ("mm_input_projection", "kernel"),
    },
    "llama4": {"linear_1.weight": ("linear", "kernel")},
    "gemma4": {"embedding_projection.weight": ("projection", "kernel")},
    "qwen3_5": {
        "norm.weight": ("norm", "scale"), "norm.bias": ("norm", "bias"),
        "linear_fc1.weight": ("fc1", "kernel"), "linear_fc1.bias": ("fc1", "bias"),
        "linear_fc2.weight": ("fc2", "kernel"), "linear_fc2.bias": ("fc2", "bias"),
    },
    # MiMo's merger stores no biases (MiMoVisionPatchMerger's released tensors).
    "mimo": {"ln_q.weight": ("norm", "scale"), "mlp.0.weight": ("fc1", "kernel"),
             "mlp.2.weight": ("fc2", "kernel")},
    # Muse Glimmer's adapter and projection, each name its prefix off (`PROJECTOR_PREFIX`).
    "muse_glimmer": {"fc1.weight": ("adapter", "fc1", "kernel"), "fc2.weight": ("adapter", "fc2", "kernel"),
                     "weight": ("projection", "kernel")},
    "gemma3n": {
        "embedding.weight": ("embedding", "embedding"),
        "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
        "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
        "embedding_projection.weight": ("embedding_projection", "kernel"),
    },
    # The aligner's maps under its `aligner.` prefix, and the span's learned
    # vectors, which the release keeps at its top level (model.py:1215-1222).
    "deepseek_v41": {
        "w1.weight": ("w1", "kernel"), "w1.bias": ("w1", "bias"),
        "w2.weight": ("w2", "kernel"), "w2.bias": ("w2", "bias"),
        "image_start": ("image_start",), "image_newline": ("image_newline",),
        "image_end": ("image_end",),
    },
}


_BLOCK_NORMS = {"norm1": "input_layernorm", "norm2": "post_attention_layernorm"}
"""A ViT block's two norms as `clip_layer`'s decoder block names them."""


def block_tensor_path(parts: list[str], maps: Mapping[tuple[str, str], tuple[str, str]], *,
                      prefix: str = "blocks", norm_bias: bool = True) -> tuple[str, ...] | None:
    """A shared-trunk ViT's block tensor, `{prefix}.N.norm1|norm2.weight|bias` or
    `{prefix}.N.<module>.<linear>.weight|bias`, its name split on dots, into its path under
    `blocks_N`: `maps` places each release's (module, linear) pair, and a release whose norms carry
    no bias says so (`norm_bias`). None for any other name, which the family reads or refuses."""
    if (len(parts) not in (4, 5) or parts[0] != prefix or not parts[1].isdigit()
            or parts[-1] not in ("weight", "bias")):
        return None
    block, leaf = f"blocks_{parts[1]}", parts[-1]
    if len(parts) == 4 and parts[2] in _BLOCK_NORMS and (norm_bias or leaf == "weight"):
        return (block, _BLOCK_NORMS[parts[2]], "scale" if leaf == "weight" else "bias")
    if len(parts) == 5 and (parts[2], parts[3]) in maps:
        return (block, *maps[parts[2], parts[3]], "kernel" if leaf == "weight" else "bias")
    return None


def projector_weight_path(kind: str, name: str) -> tuple[str, ...]:
    """The projector's canonical tensor path, shared by load and source export."""
    if kind in ("qwen3_5", "mimo"):
        name = name.removeprefix("merger.")
    if kind not in _PROJECTOR_PATHS or name not in _PROJECTOR_PATHS[kind]:
        raise ValueError(f"unknown {kind} projector tensor {name!r}")
    return _PROJECTOR_PATHS[kind][name]


def _vision_section(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The vision half of a wrapper config, or a bare vision config."""
    return records.record(hf_config.get("vision_config", hf_config), "vision_config")


def _side(value: object, field: str) -> int:
    """A square image or patch size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    return records.integer(value, field)


def trunk_geometry(vision: Mapping[str, object]) -> dict[str, int]:
    """The width, MLP width, depth and head count a ViT config states, under
    the trunk fields of the families that share the reference's names."""
    return {
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": records.integer(vision["num_attention_heads"], "num_attention_heads"),
    }


def activation_name(config: Mapping[str, object], runs: Collection[str], default: str,
                    fields: tuple[str, ...] = ("hidden_act",)) -> str:
    """The activation the first of `fields` that `config` states names, else
    the family's `default`; a name outside `runs`, the activations its trunk
    computes, refuses."""
    name = str(next((config[field] for field in fields if field in config), default))
    if name not in runs:
        raise ValueError(f"{fields[0]} {name!r} is not expressible: this trunk runs {', '.join(runs)}")
    return name


def refuse_dropout(vision: Mapping[str, object], *fields: str) -> None:
    """Refuse a nonzero dropout among `fields`: the trunks run eval."""
    for field in fields:
        if records.number(vision.get(field, 0.0), field):
            raise ValueError(f"{field} is training-time; this trunk runs eval")


def rope_base(vision: Mapping[str, object], default: float) -> float:
    """A trunk's plain rotary base: `rope_parameters`' rope_theta, else the
    flat field, else the family's `default`. A scaled rope refuses, as any
    rope does where only plain rotary has a place (`read_rope_theta`), and so
    does the decoders' 'none' alias: the towers dispatch any type but
    'default' to Transformers' rope initializers, which name no 'none'
    (modeling_gemma4.py:716-719)."""
    entry = records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    if entry.get("rope_type", entry.get("type")) == "none":
        raise ValueError("rope_parameters: a vision tower's rotary type is 'default', not 'none'")
    theta = read_rope_theta(entry, "rope_parameters")
    return records.number(vision.get("rope_theta", default), "rope_theta") if theta is None else theta
