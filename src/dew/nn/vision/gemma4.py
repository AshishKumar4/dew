"""The Gemma 4 vision trunk and its projector, with their checkpoint maps
(transformers 5.16.1 `models/gemma4/modeling_gemma4.py`).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer, PrecisionLike

from dew import _records
from dew._model_types import SANDWICH_NORMS
from dew.batch import Variables, coordinates
from dew.interop.unread import Stored
from dew.interop.weights import native_layouts, placed
from dew.nn.attention import RMSNorm
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock
from dew.nn.blocks import table_rows
from dew.nn.dense import Dense
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mlp import GatedMLP
from dew.nn.precision import float32_dot_general
from dew.nn.scan_orders import patchify
from dew.registry import Record

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    activation_name,
    head_width,
    rope_base,
    trunk_geometry,
)

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


def clipped_linear(module: nn.Module, linear: Callable[[jax.Array], jax.Array], inputs: jax.Array,
                   clipped: bool) -> jax.Array:
    """`linear(inputs)`, its input and output clamped where `clipped` to the
    nontrainable bounds `module` keeps: the reference's one clippable linear
    for both Gemma 4 towers (modeling_gemma4.py:168-194). The bounds stay
    FP32 and clamp in each operand's dtype, as torch.clamp against a 0-dim
    buffer does, so a bf16 tower stays bf16 through the map."""
    if not clipped:
        return linear(inputs)

    def clamp(value, side):
        low = module.variable("constants", f"{side}_min", lambda: jnp.array(-jnp.inf, jnp.float32)).value
        high = module.variable("constants", f"{side}_max", lambda: jnp.array(jnp.inf, jnp.float32)).value
        return jnp.clip(value, low.astype(value.dtype), high.astype(value.dtype))

    return clamp(linear(clamp(inputs, "input")), "output")


class Gemma4ClippableLinear(Dense):
    """The reference linear map with optional nontrainable activation bounds.

    Subclassing Dense keeps the existing kernel path; HF's extra linear
    name belongs to checkpoint conversion, not another numerical operation.
    """

    use_clipped_linears: bool = False

    @nn.nowrap
    def _envelope(self, linear, inputs):
        return clipped_linear(self, linear, inputs, self.use_clipped_linears)


class AxialPatchEmbedding(nn.Module):
    """Positioned HWC patches projected once, with a learned table for each spatial axis."""

    features: int
    patch_size: int
    position_size: int
    block_size: int = 1
    use_bias: bool = False
    table_axis: Literal[0, 1] = 0
    position_validity: Literal["token", "axis"] = "token"
    norm: Callable[..., nn.Module] | None = None
    initializer: Initializer = nn.initializers.normal(0.02)
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def __call__(self, pixels, pixel_position_ids=None):
        return self.positioned(pixels, pixel_position_ids)[0]

    @nn.compact
    def positioned(self, pixels, positions=None):
        pixels = jnp.asarray(pixels, self.dtype)
        if pixels.ndim == 4:
            if positions is not None:
                raise ValueError("position IDs accompany patch pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            stride = self.patch_size * self.block_size
            if channels != 3 or height % stride or width % stride:
                raise ValueError(f"images must have three channels and sides divisible by {stride}")
            rows, columns = height // self.patch_size, width // self.patch_size
            pixels = patchify(pixels.transpose(0, 2, 3, 1), self.patch_size)
            x = jnp.tile(jnp.arange(columns), rows)
            y = jnp.repeat(jnp.arange(rows), columns)
            positions = jnp.broadcast_to(jnp.stack([x, y], axis=-1), (batch, rows * columns, 2))
        if pixels.ndim != 3 or pixels.shape[-1] != 3 * self.patch_size ** 2:
            raise ValueError("patch pixels must be [B, patches, 3 * patch_size**2]")
        if positions is None or positions.shape != (*pixels.shape[:2], 2):
            raise ValueError("patch pixels require aligned [B, patches, 2] position IDs")
        if not jnp.issubdtype(positions.dtype, jnp.integer):
            raise ValueError("pixel_position_ids must be integers")
        if pixels.shape[1] % self.block_size ** 2:
            raise ValueError("patch count must divide into whole pooling blocks")
        if self.norm is not None:
            pixels = self.norm(dtype=self.dtype, name="patch_norm1")(pixels)
        hidden = Dense(self.features, use_bias=self.use_bias, dtype=self.dtype,
                       precision=self.precision, name="patch_embed",
                       dot_general=float32_dot_general)(pixels)
        if self.norm is not None:
            hidden = self.norm(dtype=self.dtype, name="patch_norm2")(hidden)
        shape = ((2, self.position_size, self.features) if self.table_axis == 0
                 else (self.position_size, 2, self.features))
        table = jnp.moveaxis(self.param("position_table", self.initializer, shape), self.table_axis, 0)
        valid = positions >= 0
        mask = (valid if self.position_validity == "axis" else
                jnp.broadcast_to(valid.all(axis=-1, keepdims=True), valid.shape))
        safe = jnp.maximum(positions, 0)
        embedded = [jnp.where(mask[..., axis, None],
                              table_rows(table[axis], safe[..., axis], hidden.dtype), 0)
                    for axis in range(2)]
        hidden = hidden + (embedded[0] + embedded[1])
        if self.norm is not None:
            hidden = self.norm(dtype=self.dtype, name="position_norm")(hidden)
        return hidden, positions


class Gemma4VisionTransformer(nn.Module):
    """The Gemma 4 vision trunk, param layout of `Gemma4VisionConfig`.

    The processor supplies [B, patches, patch_pixels] and (x, y) position IDs
    with (-1, -1) for padding. Fixed NCHW images are patchified in the same
    HWC order. Outputs retain padded soft-token slots so their shape remains
    static under JIT; callers select valid features using processor lengths.

    Each layer is the decoder block with RMS norms before and after both
    halves (modeling_gemma4.py, Gemma4VisionEncoderLayer). Its attention has
    bias-free maps over grouped key heads, the queries and keys normed with a
    scale and the values without, its scores unscaled, and each spatial axis
    turning its own half of the head the NeoX way (Gemma4VisionAttention,
    apply_multidimensional_rope). The MLP gates the checkpoint's GELU. With
    `use_clipped_linears` every map clips its input and output to the bounds
    the checkpoint stores (`Gemma4ClippableLinear`).
    """

    config: Gemma4Vision
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        head_dim = head_width(cfg.hidden_size, cfg.num_heads) if cfg.head_dim is None else cfg.head_dim
        if head_dim <= 0 or head_dim % 4:
            raise ValueError(
                f"the head width ({head_dim}) must split "
                "over the two rotary dims and their halves")
        if cfg.num_heads % cfg.num_key_value_heads:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must repeat num_key_value_heads "
                f"({cfg.num_key_value_heads})")
        self.embedding = AxialPatchEmbedding(cfg.hidden_size, cfg.patch_size, cfg.position_embedding_size,
                                             block_size=cfg.pooling_kernel_size, dtype=self.dtype,
                                             precision=self.precision)
        nn.share_scope(self, self.embedding)
        linear = functools.partial(Gemma4ClippableLinear, use_clipped_linears=cfg.use_clipped_linears,
                                   dot_general=float32_dot_general)
        attention = functools.partial(
            CausalSelfAttention, ctx=MixerContext(
                emb_features=cfg.hidden_size, num_heads=cfg.num_heads, num_kv_heads=cfg.num_key_value_heads,
                head_dim=head_dim, max_seq_len=cfg.position_embedding_size, causal=False, v_norm=True,
                norm_eps=cfg.rms_norm_eps, attention_scale=1.0, rope_theta=cfg.rope_theta, dtype=self.dtype,
                precision=self.precision),
            rotary_axes=(head_dim // 2, head_dim // 2), rotary_per_axis=True, linear=linear)
        mlp = functools.partial(
            GatedMLP, hidden_features=cfg.intermediate_size, out_features=cfg.hidden_size,
            activation="geglu" if cfg.hidden_act == "gelu_pytorch_tanh" else "geglu_exact", linear=linear,
            dtype=self.dtype, precision=self.precision)
        self.layers = [
            DecoderBlock(attention, mlp, cfg.hidden_size, BlockWiring(output_norms=True),
                         norm_eps=cfg.rms_norm_eps, dtype=self.dtype, precision=self.precision,
                         name=f"layers_{index}")
            for index in range(cfg.num_layers)]
        if cfg.standardize:
            self.std_bias = self.variable(
                "constants", "std_bias", jnp.zeros, (cfg.hidden_size,), jnp.float32
            )
            self.std_scale = self.variable(
                "constants", "std_scale", jnp.ones, (cfg.hidden_size,), jnp.float32
            )

    def __call__(self, pixel_values, pixel_position_ids=None) -> jax.Array:
        cfg = self.config
        pixels = jnp.asarray(pixel_values)
        kernel = cfg.pooling_kernel_size
        hidden_states, pixel_position_ids = self.embedding.positioned(2 * (pixels - 0.5), pixel_position_ids)
        valid = (pixel_position_ids >= 0).all(axis=-1)
        safe = jnp.maximum(pixel_position_ids, 0)
        metadata = AttentionMetadata(coordinates=coordinates(valid=valid),
                                     rotary_positions=pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_metadata=metadata)
        output_length = hidden_states.shape[1] // kernel ** 2
        width = safe[..., 0].max(axis=-1, keepdims=True) + 1
        indices = safe[..., 0] // kernel + (width // kernel) * (safe[..., 1] // kernel)
        values = jnp.where(valid[..., None], hidden_states, 0).astype(jnp.float32) / kernel ** 2
        # Segment sums avoid the reference pooler's [patches, soft_tokens]
        # one-hot matrix while preserving its position-indexed block average.
        pooled = jax.vmap(lambda value, index: jax.ops.segment_sum(value, index, output_length))(
            values, indices
        )
        pooled = pooled * cfg.hidden_size ** 0.5
        if cfg.standardize:
            pooled = (pooled - self.std_bias.value) * self.std_scale.value
        return pooled.astype(hidden_states.dtype)


@dataclasses.dataclass(frozen=True)
class Gemma4Vision(TowerBase):
    """A Gemma 4 trunk's geometry, under the reference's field names."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_layers: int = 27
    num_heads: int = 16
    num_key_value_heads: int = 16
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    standardize: bool = False
    use_clipped_linears: bool = False
    head_dim: int | None = None
    """None retains hidden_size // num_heads; checkpoints may project a wider head."""

    def build(self) -> nn.Module:
        return Gemma4VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.pooling_kernel_size,
                             position_size=self.position_embedding_size)


class Gemma4Projector(ProjectorBase):
    """Soft tokens to text width: a scale-free RMS norm, then the map.

    The reference keeps this beside the tower (modeling_gemma4.py,
    Gemma4MultimodalEmbedder): the pre-projection norm carries no weight and
    the projection no bias.
    """

    text_width: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    input_features: int | None = None

    @property
    def input_width(self) -> int | None:
        return self.input_features

    def setup(self):
        self.pre_norm = RMSNorm(epsilon=self.norm_eps, with_scale=False,
                                dtype=self.dtype, name="pre_norm")
        self.projection = Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="projection", dot_general=float32_dot_general)

    def __call__(self, image_features) -> jax.Array:
        features = jnp.asarray(image_features, self.dtype)
        if self.input_features is not None and features.shape[-1] != self.input_features:
            raise ValueError(f"media features have width {features.shape[-1]}, "
                             f"expected {self.input_features}")
        return self.projection(self.pre_norm(features))


_GEMMA4_VISION_TENSORS = {
    "patch_embedder.input_proj.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table": ("position_table",),
    "std_bias": ("std_bias",),
    "std_scale": ("std_scale",),
}
_GEMMA4_VISION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEMMA4_VISION_MLP = ("gate_proj", "up_proj", "down_proj")


def _gemma4_vision_layer_path(parts) -> tuple[str, ...] | None:
    """`encoder.layers.N...` into the layer's path."""
    if len(parts) < 5 or parts[:2] != ["encoder", "layers"] or not parts[2].isdigit():
        return None
    layer = f"layers_{parts[2]}"
    if len(parts) == 5 and parts[3] in SANDWICH_NORMS and parts[4] == "weight":
        return (layer, SANDWICH_NORMS[parts[3]], "scale")
    if len(parts) == 7 and parts[5] == "linear" and parts[6] == "weight":
        if parts[3] == "self_attn" and parts[4] in _GEMMA4_VISION_PROJECTIONS:
            return (layer, "self_attn", parts[4], "kernel")
        if parts[3] == "mlp" and parts[4] in _GEMMA4_VISION_MLP:
            return (layer, "mlp", parts[4], "kernel")
    if (len(parts) == 6 and parts[5] in ("input_min", "input_max", "output_min", "output_max")
            and ((parts[3] == "self_attn" and parts[4] in _GEMMA4_VISION_PROJECTIONS)
                 or (parts[3] == "mlp" and parts[4] in _GEMMA4_VISION_MLP))):
        return (layer, parts[3], parts[4], parts[5])
    if (len(parts) == 6 and parts[3] == "self_attn" and parts[5] == "weight"
            and parts[4] in ("q_norm", "k_norm")):
        # The value norm carries no scale (modeling_gemma4.py,
        # Gemma4VisionAttention), so a weight under its name is unknown.
        return (layer, "self_attn", parts[4], "scale")
    return None


def gemma4_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Gemma 4 vision tensor name into its collection and trunk path.

    The rotary tables are buffers the checkpoint leaves out, recomputed from
    the grid at call time. Anything else unknown raises ValueError.
    """
    path = _GEMMA4_VISION_TENSORS.get(hf_name) or _gemma4_vision_layer_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    collection = (
        "constants"
        if path[-1] in ("std_bias", "std_scale", "input_min", "input_max", "output_min", "output_max")
        else "params"
    )
    return (collection, *path)


GEMMA4_PROJECTOR_TENSORS = {"embedding_projection.weight": ("projection", "kernel")}
"""The Gemma 4 embedder's tensors past `embed_vision.` (or its audio's), each to its leaf."""


def gemma4_vision_name(path: tuple[str, ...]) -> str:
    """The Gemma 4 vision tensor name of the leaf at `path`, its collection then its trunk path:
    `gemma4_vision_path`'s inverse, for a tower built natively rather than read."""
    trunk = path[1:]
    named = next((name for name, known in _GEMMA4_VISION_TENSORS.items() if known == trunk), None)
    if named is not None:
        return named
    layer, *rest = trunk
    if not layer.startswith("layers_"):
        raise ValueError(f"unknown vision parameter {path!r}")
    norms = {native: name for name, native in SANDWICH_NORMS.items()}
    if len(rest) == 2 and rest[0] in norms:
        rest = [norms[rest[0]], "weight"]
    elif rest[-1] == "kernel":
        rest = [*rest[:-1], "linear", "weight"]
    elif rest[-1] == "scale":
        rest = [*rest[:-1], "weight"]
    return ".".join(("encoder.layers", layer.removeprefix("layers_"), *rest))


def gemma4_tower_layouts(variables: Variables) -> tuple[WeightLayout, ...]:
    """The layouts that write a natively built Gemma 4 tower's tensors back from its collections
    (`gemma4_vision_name`); a read tower's come from its reading (`Gemma4Vision.checkpoint`)."""
    return native_layouts(variables, gemma4_vision_name, gemma4_vision_path)


def gemma4_projector_layouts(params: Variables) -> tuple[WeightLayout, ...]:
    """The layouts that write a natively built Gemma 4 embedder's projection back from its
    parameters, rooted at `params`."""
    names: dict[tuple[str, ...], str] = {("params", *path): name
                                         for name, path in GEMMA4_PROJECTOR_TENSORS.items()}
    return native_layouts({"params": params}, names.__getitem__,
                          lambda name: ("params", *GEMMA4_PROJECTOR_TENSORS[name]))


def translate_gemma4_projector_weights(
    hf_tensors: Mapping[str, Stored], *, param_dtype: str = "float32"
) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """A Gemma 4 embedder's map into its parameter tree, with the layouts that write it back.

    The pre-projection norm carries no scale, so the projection weight is
    the only tensor.
    """
    if set(hf_tensors) != set(GEMMA4_PROJECTOR_TENSORS):
        raise ValueError(f"unknown tensor names {sorted(hf_tensors)}")
    return placed(hf_tensors, GEMMA4_PROJECTOR_TENSORS.__getitem__, param_dtype)


def translate_gemma4_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Gemma4VisionConfig into a Gemma4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The head
    width comes from head_dim or the hidden size divided by the head count.
    Standardization and activation clipping retain
    their reference buffers outside the trainable parameter collection.
    """
    vision = _vision_section(hf_config)
    geometry = trunk_geometry(vision)
    if vision.get("attention_bias", False):
        raise ValueError("attention_bias=True needs biased maps this trunk lacks")
    if "output_proj_dims" in vision:
        raise ValueError(
            f"output_proj_dims ({vision['output_proj_dims']!r}) changes the "
            "projector width this record leaves to the text width")
    heads = geometry["num_heads"]
    return {
        "class": "gemma4", "fields": {
        **geometry,
        "num_key_value_heads": _records.integer(
            vision.get("num_key_value_heads", heads), "num_key_value_heads"
        ),
        "head_dim": _records.integer(vision.get("head_dim", geometry["hidden_size"] // heads), "head_dim"),
        "patch_size": _records.integer(vision.get("patch_size", 16), "patch_size"),
        "pooling_kernel_size": _records.integer(vision.get("pooling_kernel_size", 3), "pooling_kernel_size"),
        "position_embedding_size": _records.integer(
            vision.get("position_embedding_size", 10240), "position_embedding_size"
        ),
        "hidden_act": activation_name(vision, ("gelu_pytorch_tanh", "gelu"), "gelu_pytorch_tanh",
                                      ("hidden_activation", "hidden_act")),
        "rms_norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        "rope_theta": rope_base(vision, 100.0, axial=True),
        "standardize": bool(vision.get("standardize", False)),
        "use_clipped_linears": bool(vision.get("use_clipped_linears", False)),
    }}


def export_gemma4_vision_config(tower: Gemma4Vision) -> Mapping[str, object]:
    """A Gemma4Vision value as the Gemma4VisionConfig that
    `translate_gemma4_vision_config` reads back to it. An unset head_dim is
    written as the width it stands for."""
    return {
        "model_type": "gemma4_vision",
        "hidden_size": tower.hidden_size,
        "intermediate_size": tower.intermediate_size,
        "num_hidden_layers": tower.num_layers,
        "num_attention_heads": tower.num_heads,
        "num_key_value_heads": tower.num_key_value_heads,
        "head_dim": tower.hidden_size // tower.num_heads if tower.head_dim is None else tower.head_dim,
        "patch_size": tower.patch_size,
        "pooling_kernel_size": tower.pooling_kernel_size,
        "position_embedding_size": tower.position_embedding_size,
        "hidden_activation": tower.hidden_act,
        "rms_norm_eps": tower.rms_norm_eps,
        "rope_parameters": {"rope_type": "default", "rope_theta": tower.rope_theta},
        "standardize": tower.standardize,
        "use_clipped_linears": tower.use_clipped_linears,
    }


def translate_gemma4_projector_config(vision: Mapping[str, object], text_width: int,
                                      input_width: int | None = None) -> Record:
    """A Gemma 4 wrapper's projector fields: decoder width, norm epsilon, and the width of the raw media
    it projects where no encoder precedes it (`input_width`, Gemma 4 Unified's)."""
    fields: dict[str, object] = {
        "text_width": int(text_width),
        "norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps")}
    if input_width is not None:
        fields["input_features"] = int(input_width)
    return {"class": "gemma4", "fields": fields}


Gemma4Vision.checkpoint = TowerLayout("vision_tower.", gemma4_vision_path)
Gemma4Projector.checkpoint = ProjectorLayout("embed_vision.", GEMMA4_PROJECTOR_TENSORS,
                                             read=translate_gemma4_projector_weights)
