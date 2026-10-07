"""The Gemma 4 vision trunk and its projector, with their checkpoint maps
(transformers 5.16.1 `models/gemma4/modeling_gemma4.py`).
"""

import dataclasses
import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike

from dew import records
from dew.interop.weights import translate_parameters
from dew.nn.activations import activation
from dew.nn.attention import RMSNorm
from dew.nn.precision import at_least_fp32
from dew.nn.rope import inverse_frequencies
from dew.objectives.base import Variables
from dew.registry import Record

from .common import (
    _PROJECTOR_PATHS,
    ProjectorBase,
    TowerBase,
    TowerGeometry,
    _vision_section,
    projector_weight_path,
)


def _gemma4_rope_tables(positions: jax.Array, head_dim: int,
                        theta: float, *, dtype: DTypeLike) -> tuple[jax.Array, jax.Array]:
    """The 2D rotary tables of the Gemma 4 vision attention, as cos/sin.

    Each spatial dim carries its own frequencies over half the head
    (modeling_gemma4.py, Gemma4VisionRotaryEmbedding.
    compute_default_rope_parameters): the angles double up within a dim and
    the dims concatenate, so `positions` [B, P, 2] yields [B, P, head_dim].
    """
    spatial = head_dim // 2
    inv_freq = inverse_frequencies(theta, spatial, dtype=dtype)
    angles = positions.astype(dtype)[:, :, :, None] * inv_freq
    doubled = jnp.concatenate([angles, angles], axis=-1)
    cos = jnp.concatenate([jnp.cos(doubled[:, :, 0]), jnp.cos(doubled[:, :, 1])],
                          axis=-1)
    sin = jnp.concatenate([jnp.sin(doubled[:, :, 0]), jnp.sin(doubled[:, :, 1])],
                          axis=-1)
    return cos, sin


def _gemma4_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The half rotation on each spatial half, broadcast over the heads.

    Each half turns the NeoX way (modeling_gemma4.py, rotate_half and
    apply_multidimensional_rope): the halves split again and the second
    quarter crosses negated over the first.
    """
    halves = jnp.split(values, 2, axis=-1)
    angles = jnp.split(cos, 2, axis=-1)
    turns = jnp.split(sin, 2, axis=-1)
    rotated = [half * angle + jnp.concatenate(
        [-half[..., half.shape[-1] // 2:], half[..., :half.shape[-1] // 2]],
        axis=-1) * turn for half, angle, turn in zip(halves, angles, turns, strict=True)]
    return jnp.concatenate(rotated, axis=-1)


class Gemma4ClippableLinear(nn.Dense):
    """The reference linear map with optional nontrainable activation bounds.

    Subclassing Dense keeps the existing kernel path; HF's extra linear
    name belongs to checkpoint conversion, not another numerical operation.
    """

    use_clipped_linears: bool = False

    @nn.compact
    def __call__(self, inputs):
        if self.use_clipped_linears:
            low = self.variable("constants", "input_min", lambda: jnp.array(-jnp.inf, jnp.float32))
            high = self.variable("constants", "input_max", lambda: jnp.array(jnp.inf, jnp.float32))
            inputs = jnp.clip(inputs, low.value.astype(inputs.dtype), high.value.astype(inputs.dtype))
        output = super().__call__(inputs)
        if self.use_clipped_linears:
            low = self.variable("constants", "output_min", lambda: jnp.array(-jnp.inf, jnp.float32))
            high = self.variable("constants", "output_max", lambda: jnp.array(jnp.inf, jnp.float32))
            output = jnp.clip(output, low.value.astype(output.dtype), high.value.astype(output.dtype))
        return output


class Gemma4VisionAttention(nn.Module):
    """Grouped-query attention with scaled q/k/v norms and the 2D rotary.

    The four maps are bias-free, the queries and keys norm with a scale and
    the values without (modeling_gemma4.py, Gemma4VisionAttention), and the
    scores run unscaled with the softmax in fp32 (its scaling is 1.0, not
    the shared kernel's 1/sqrt(d), so the few lines sit here).
    """

    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False
    head_dim: int | None = None

    @property
    def features_per_head(self) -> int:
        return self.hidden_size // self.num_heads if self.head_dim is None else self.head_dim

    def setup(self):
        dense = functools.partial(Gemma4ClippableLinear, use_bias=False,
                                  use_clipped_linears=self.use_clipped_linears,
                                  dtype=self.dtype, precision=self.precision)
        head_dim = self.features_per_head
        self.q_proj = dense(self.num_heads * head_dim, name="q_proj")
        self.k_proj = dense(self.num_key_value_heads * head_dim,
                            name="k_proj")
        self.v_proj = dense(self.num_key_value_heads * head_dim,
                            name="v_proj")
        self.o_proj = dense(self.hidden_size, name="o_proj")
        self.q_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="q_norm")
        self.k_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="k_norm")
        self.v_norm = RMSNorm(epsilon=self.rms_norm_eps, with_scale=False,
                              dtype=self.dtype, name="v_norm")

    def __call__(self, hidden_states, cos, sin, valid=None) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.features_per_head
        # The norms read the heads (modeling_gemma4.py,
        # Gemma4VisionAttention.forward): project, split, then normalize.
        query = self.q_norm(self.q_proj(hidden_states).reshape(batch, length, -1, head_dim))
        key = self.k_norm(self.k_proj(hidden_states).reshape(batch, length, -1, head_dim))
        value = self.v_norm(self.v_proj(hidden_states).reshape(batch, length, -1, head_dim))
        query = _gemma4_rope(query, cos[:, :, None, :], sin[:, :, None, :])
        key = _gemma4_rope(key, cos[:, :, None, :], sin[:, :, None, :])
        repeats = self.num_heads // self.num_key_value_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=self.precision)
        if valid is not None:
            scores = jnp.where(valid[:, None, None, :], scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probs, value, precision=self.precision)
        return self.o_proj(attended.reshape(batch, length, self.num_heads * head_dim))


class Gemma4VisionMLP(nn.Module):
    """Gated feed-forward without biases: act(gate) times up, then down.

    The reference reads the activation from the config (modeling_gemma4.py,
    Gemma4VisionMLP); only the two Gaussian forms map.
    """

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "gelu_pytorch_tanh"
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False

    def setup(self):
        dense = functools.partial(Gemma4ClippableLinear, use_bias=False,
                                  use_clipped_linears=self.use_clipped_linears,
                                  dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.intermediate_size, name="gate_proj")
        self.up_proj = dense(self.intermediate_size, name="up_proj")
        self.down_proj = dense(self.hidden_size, name="down_proj")

    def __call__(self, hidden_states):
        if self.hidden_act not in ("gelu_pytorch_tanh", "gelu"):
            raise ValueError(
                f"hidden_act {self.hidden_act!r} is not expressible: this MLP "
                "runs gelu_pytorch_tanh or gelu")
        return self.down_proj(activation(self.hidden_act)(self.gate_proj(hidden_states))
                              * self.up_proj(hidden_states))


class Gemma4VisionEncoderLayer(nn.Module):
    """RMS sandwich around attention and the gated MLP, both residual.

    The four norms all carry a scale (modeling_gemma4.py,
    Gemma4VisionEncoderLayer): one before and one after each half.
    """

    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_key_value_heads: int
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False
    head_dim: int | None = None

    def setup(self):
        norm = functools.partial(RMSNorm, epsilon=self.rms_norm_eps, dtype=self.dtype)
        self.input_layernorm = norm(name="input_layernorm")
        self.self_attn = Gemma4VisionAttention(
            self.hidden_size, self.num_heads, self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype, precision=self.precision,
            use_clipped_linears=self.use_clipped_linears, name="self_attn")
        self.post_attention_layernorm = norm(name="post_attention_layernorm")
        self.pre_feedforward_layernorm = norm(name="pre_feedforward_layernorm")
        self.mlp = Gemma4VisionMLP(
            self.hidden_size, self.intermediate_size, self.hidden_act,
            dtype=self.dtype, precision=self.precision,
            use_clipped_linears=self.use_clipped_linears, name="mlp")
        self.post_feedforward_layernorm = norm(name="post_feedforward_layernorm")

    def __call__(self, hidden_states, cos, sin, valid=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin, valid)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4VisionTransformer(nn.Module):
    """The Gemma 4 vision trunk, param layout of `Gemma4VisionConfig`.

    The processor supplies [B, patches, patch_pixels] and (x, y) position IDs
    with (-1, -1) for padding. Fixed NCHW images are patchified in the same
    HWC order. Outputs retain padded soft-token slots so their shape remains
    static under JIT; callers select valid features using processor lengths.
    """

    config: "Gemma4Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.head_dim is None and cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        head_dim = cfg.hidden_size // cfg.num_heads if cfg.head_dim is None else cfg.head_dim
        if head_dim <= 0 or head_dim % 4:
            raise ValueError(
                f"the head width ({head_dim}) must split "
                "over the two rotary dims and their halves")
        if cfg.num_heads % cfg.num_key_value_heads:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must repeat num_key_value_heads "
                f"({cfg.num_key_value_heads})")
        self.patch_embed = nn.Dense(
            cfg.hidden_size, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = self.param(
            "position_table", nn.initializers.normal(0.02),
            (2, cfg.position_embedding_size, cfg.hidden_size))
        self.layers = [
            Gemma4VisionEncoderLayer(
                cfg.hidden_size, cfg.intermediate_size, cfg.num_heads,
                cfg.num_key_value_heads, cfg.hidden_act,
                head_dim=cfg.head_dim,
                rms_norm_eps=cfg.rms_norm_eps,
                dtype=self.dtype, precision=self.precision,
                use_clipped_linears=cfg.use_clipped_linears, name=f"layers_{index}")
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
        if pixels.ndim == 4:
            if pixel_position_ids is not None:
                raise ValueError("position IDs accompany patch pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            stride = cfg.patch_size * kernel
            if channels != 3 or height % stride or width % stride:
                raise ValueError(f"Gemma4 images must have three channels and sides divisible by {stride}")
            rows, columns = height // cfg.patch_size, width // cfg.patch_size
            patches = pixels.reshape(batch, channels, rows, cfg.patch_size, columns, cfg.patch_size)
            pixels = patches.transpose(0, 2, 4, 3, 5, 1).reshape(batch, rows * columns, -1)
            x = jnp.tile(jnp.arange(columns), rows)
            y = jnp.repeat(jnp.arange(rows), columns)
            pixel_position_ids = jnp.broadcast_to(jnp.stack([x, y], axis=-1), (batch, rows * columns, 2))
        if pixels.ndim != 3 or pixels.shape[-1] != 3 * cfg.patch_size ** 2:
            raise ValueError("Gemma4 patch pixels must be [B, patches, 3 * patch_size**2]")
        if pixel_position_ids is None or pixel_position_ids.shape != (*pixels.shape[:2], 2):
            raise ValueError("Gemma4 patch pixels require aligned [B, patches, 2] position IDs")
        if not jnp.issubdtype(pixel_position_ids.dtype, jnp.integer):
            raise ValueError("pixel_position_ids must be integers")
        if pixels.shape[1] % kernel ** 2:
            raise ValueError("patch count must divide into whole pooling blocks")
        valid = (pixel_position_ids >= 0).all(axis=-1)
        safe = jnp.maximum(pixel_position_ids, 0)
        hidden_states = self.patch_embed(2 * (pixels - 0.5))
        table = jnp.asarray(self.position_table, hidden_states.dtype)
        positional = table[0, safe[..., 0]] + table[1, safe[..., 1]]
        hidden_states = hidden_states + jnp.where(valid[..., None], positional, 0)
        head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_heads
        cos, sin = _gemma4_rope_tables(pixel_position_ids, head_dim, cfg.rope_theta,
                                       dtype=at_least_fp32(hidden_states.dtype))
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, valid)
        output_length = pixels.shape[1] // kernel ** 2
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
        return TowerGeometry(patch_size=self.patch_size, block_size=self.pooling_kernel_size)


class Gemma4ProjectorModule(nn.Module):
    """Soft tokens to text width: a scale-free RMS norm, then the map.

    The reference keeps this beside the tower (modeling_gemma4.py,
    Gemma4MultimodalEmbedder): the pre-projection norm carries no weight and
    the projection no bias.
    """

    text_width: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.pre_norm = RMSNorm(epsilon=self.norm_eps, with_scale=False,
                                dtype=self.dtype, name="pre_norm")
        self.projection = nn.Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="projection")

    def __call__(self, image_features) -> jax.Array:
        return self.projection(self.pre_norm(image_features))


@dataclasses.dataclass(frozen=True)
class Gemma4Projector(ProjectorBase):
    """Gemma 4's projector fields: the decoder width and the norm epsilon."""

    text_width: int
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return Gemma4ProjectorModule(text_width=self.text_width, norm_eps=self.norm_eps)


_GEMMA4_VISION_TENSORS = {
    "patch_embedder.input_proj.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table": ("position_table",),
    "std_bias": ("std_bias",),
    "std_scale": ("std_scale",),
}
_GEMMA4_VISION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEMMA4_VISION_MLP = ("gate_proj", "up_proj", "down_proj")
_GEMMA4_VISION_NORMS = ("input_layernorm", "post_attention_layernorm",
                        "pre_feedforward_layernorm", "post_feedforward_layernorm")


def _gemma4_vision_layer_path(parts) -> tuple[str, ...] | None:
    """`encoder.layers.N...` into the layer's path."""
    if len(parts) < 5 or parts[:2] != ["encoder", "layers"] or not parts[2].isdigit():
        return None
    layer = f"layers_{parts[2]}"
    if len(parts) == 5 and parts[3] in _GEMMA4_VISION_NORMS and parts[4] == "weight":
        return (layer, parts[3], "scale")
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


def translate_gemma4_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Gemma 4 parameters plus native FP32 frozen and clipping buffers."""
    return translate_parameters(hf_tensors, gemma4_vision_path, param_dtype)


def translate_gemma4_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """A Gemma 4 embedder's map into its parameter tree.

    The pre-projection norm carries no scale, so the projection weight is
    the only tensor.
    """
    if set(hf_tensors) != set(_PROJECTOR_PATHS["gemma4"]):
        raise ValueError(f"unknown tensor names {sorted(hf_tensors)}")
    return translate_parameters(hf_tensors, lambda name: projector_weight_path("gemma4", name), param_dtype)


def translate_gemma4_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Gemma4VisionConfig into a Gemma4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The head
    width comes from head_dim or the hidden size divided by the head count.
    Standardization and activation clipping retain
    their reference buffers outside the trainable parameter collection.
    """
    vision = _vision_section(hf_config)
    hidden = records.integer(vision["hidden_size"], "hidden_size")
    heads = records.integer(vision["num_attention_heads"], "num_attention_heads")
    head_dim = vision.get("head_dim", hidden // heads)
    if records.integer(head_dim, "head_dim") <= 0 or records.integer(head_dim, "head_dim") % 4:
        raise ValueError(
            f"head_dim ({head_dim}) must split into the two rotary dimensions and their halves")
    activation = str(vision.get("hidden_activation", vision.get("hidden_act",
                                                                "gelu_pytorch_tanh")))
    if activation not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_activation {activation!r} is not expressible: this trunk "
            "runs gelu_pytorch_tanh or gelu")
    if vision.get("attention_bias", False):
        raise ValueError("attention_bias=True needs biased maps this trunk lacks")
    if records.number(vision.get("attention_dropout", 0.0), "attention_dropout"):
        raise ValueError("attention_dropout is training-time; this trunk runs eval")
    if "output_proj_dims" in vision:
        raise ValueError(
            f"output_proj_dims ({vision['output_proj_dims']!r}) changes the "
            "projector width this record leaves to the text width")
    rope = records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    if rope.get("rope_type", "default") != "default":
        raise ValueError(
            f"rope_type {rope.get('rope_type')!r} is not expressible: this trunk "
            "runs the default 2D rotary")
    return {
        "class": "gemma4", "fields": {
        "hidden_size": hidden,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": heads,
        "num_key_value_heads": records.integer(
            vision.get("num_key_value_heads", heads), "num_key_value_heads"
        ),
        "head_dim": records.integer(head_dim, "head_dim"),
        "patch_size": records.integer(vision.get("patch_size", 16), "patch_size"),
        "pooling_kernel_size": records.integer(vision.get("pooling_kernel_size", 3), "pooling_kernel_size"),
        "position_embedding_size": records.integer(
            vision.get("position_embedding_size", 10240), "position_embedding_size"
        ),
        "hidden_act": activation,
        "rms_norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        "rope_theta": records.number(rope.get("rope_theta", vision.get("rope_theta", 100.0)), "rope_theta"),
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


def translate_gemma4_projector_config(vision: Mapping[str, object],
                                      text_width: int) -> Record:
    """A Gemma 4 wrapper's projector fields: decoder width, norm epsilon."""
    return {
        "class": "gemma4", "fields": {
        "text_width": int(text_width),
        "norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }}
