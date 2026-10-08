"""The Qwen 3.5 vision trunk and its merger, with their checkpoint maps
(transformers 5.16.1 `models/qwen3_5/modeling_qwen3_5.py`).
"""

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew._model_types import _QWEN35_VISION_TYPES
from dew.interop.weights import checkpoint_array, translate_parameters
from dew.nn.activations import gelu_exact
from dew.nn.attention import LayerNorm
from dew.nn.inputs import AttentionMetadata
from dew.nn.text_encoders import clip_layer
from dew.objectives.base import Variables
from dew.registry import Record

from .common import ProjectorBase, TowerBase, TowerGeometry, _vision_section


def _qwen35_interp_taps(index: jax.Array, size: int | jax.Array, side: int) -> tuple[jax.Array, jax.Array]:
    """Bilinear taps into a `side`-long table for positions along one axis.

    The closed form of the align-corners linspace the reference resamples
    with (vision_utils.py, _interpolation_axis_taps_weights): each position
    reads its floor and ceiling with the linear hat weights.
    """
    src = index.astype(jnp.float32) * (side - 1) / jnp.maximum(size - 1, 1)
    floor = jnp.floor(src).astype(jnp.int32)
    distance = (src - floor.astype(jnp.float32))[..., None]
    taps = jnp.stack([floor, floor + 1], axis=-1).clip(0, side - 1)
    weights = jnp.concatenate([1 - distance, distance], axis=-1).clip(min=0)
    return taps, weights


def _qwen35_pos_embeds(table: jax.Array, rows: jax.Array, cols: jax.Array,
                       grid_height: int | jax.Array, grid_width: int | jax.Array) -> jax.Array:
    """The learned table resampled to one image's patch grid.

    Bilinear over the square table's rows and columns separately, outer
    product of the taps (vision_utils.py,
    get_vision_interpolation_indices_and_weights).
    """
    side = round(len(table) ** 0.5)
    row_taps, row_weights = _qwen35_interp_taps(rows, grid_height, side)
    col_taps, col_weights = _qwen35_interp_taps(cols, grid_width, side)
    indices = (row_taps[..., :, None] * side + col_taps[..., None, :]).reshape(*rows.shape, 4)
    weights = (row_weights[..., :, None] * col_weights[..., None, :]).reshape(*rows.shape, 4)
    return (table[indices] * weights[..., None]).sum(axis=-2)


class Qwen35VisionTransformer(nn.Module):
    """The Qwen 3.5 vision trunk, param layout of `Qwen3_5VisionConfig`.

    Packed processor pixels are padded to [B, patches, patch_pixels] and
    accompanied by each row's [time, height, width] patch grid. Attention stays
    inside each frame and excludes padded keys. Fixed NCHW images use the
    processor's channel-then-time patch order. The return value is the
    pre-merge sequence; the projector owns the merger.

    Each block is CLIP's layer (`clip_layer`) at a 1e-6 epsilon, its queries,
    keys and values in one map, rotated by the patch grid: heights then
    widths share one frequency table over half the head, and the two turn
    together as the text rope's halves do (modeling_qwen3_5.py,
    Qwen3_5VisionRotaryEmbedding and apply_rotary_pos_emb_vision).
    """

    config: "Qwen35Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        self.patch_embed = nn.Dense(
            cfg.hidden_size, use_bias=True, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = nn.Embed(cfg.num_position_embeddings, cfg.hidden_size,
                                       dtype=self.dtype, name="position_table")
        side = cfg.hidden_size // cfg.num_heads // 2
        self.blocks = [
            clip_layer(cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, cfg.num_position_embeddings,
                       activation=cfg.hidden_act, eps=1e-6, rotary_axes=(side, side), packed=True,
                       dtype=self.dtype, precision=self.precision, name=f"blocks_{index}")
            for index in range(cfg.depth)]

    def __call__(self, pixel_values, grid_thw=None) -> jax.Array:
        cfg = self.config
        pixels = jnp.asarray(pixel_values)
        merge = cfg.spatial_merge_size
        patch = cfg.patch_size
        if pixels.ndim == 4:
            if grid_thw is not None:
                raise ValueError("grid_thw accompanies packed pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            stride = patch * merge
            if channels != cfg.in_channels or height % stride or width % stride:
                raise ValueError(
                    f"pixel_values must be {cfg.in_channels}-channel images tiled by {stride}px blocks"
                )
            rows, columns = height // patch, width // patch
            blocks = pixels.reshape(batch, channels, rows // merge, merge, patch,
                                    columns // merge, merge, patch)
            blocks = blocks.transpose(0, 2, 5, 3, 6, 1, 4, 7)
            flat = blocks.reshape(batch, rows * columns, channels, 1, patch, patch)
            frames = jnp.broadcast_to(flat, (batch, rows * columns, channels,
                                            cfg.temporal_patch_size, patch, patch))
            pixels = frames.reshape(batch, rows * columns, -1)
            grid_thw = jnp.broadcast_to(jnp.array([1, rows, columns], jnp.int32), (batch, 3))
        if pixels.ndim != 3 or pixels.shape[-1] != cfg.in_channels * cfg.temporal_patch_size * patch ** 2:
            raise ValueError(
                "packed Qwen pixels must be [B, patches, C * temporal_patch_size * patch_size**2]"
            )
        batch, length, _ = pixels.shape
        if grid_thw is None or grid_thw.shape != (batch, 3):
            raise ValueError("packed Qwen pixels require aligned [B, 3] grid_thw")
        grid = jnp.asarray(grid_thw)
        heights, widths = grid[:, 1:2], grid[:, 2:3]
        area = heights * widths
        within = jnp.arange(length)[None, :] % area
        blocks_wide = widths // merge
        rows = (within // (merge * merge * blocks_wide)) * merge + (within // merge) % merge
        columns = ((within // (merge * merge)) % blocks_wide) * merge + within % merge
        hidden_states = self.patch_embed(pixels)
        table = jnp.asarray(self.position_table.embedding, hidden_states.dtype)
        hidden_states = hidden_states + _qwen35_pos_embeds(table, rows, columns, heights, widths)
        metadata = AttentionMetadata(rotary_positions=jnp.stack([rows, columns], axis=-1))
        # Attention stays inside each frame, and padding, segment 0, sees nothing.
        valid = jnp.arange(length)[None, :] < jnp.prod(grid, axis=1, keepdims=True)
        frames = jnp.where(valid, jnp.arange(length)[None, :] // area + 1, 0)
        for block in self.blocks:
            hidden_states = block(hidden_states, segment_ids=frames, attention_metadata=metadata)
        # A block without hyper-connections hands on the plain residual.
        assert isinstance(hidden_states, jax.Array)
        return hidden_states


@dataclasses.dataclass(frozen=True)
class Qwen35Vision(TowerBase):
    """A Qwen 3.5 trunk's geometry, under the reference's field names."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304

    def build(self) -> nn.Module:
        return Qwen35VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class Qwen35ProjectorModule(nn.Module):
    """The merger: norm, group each merge block, map through an exact GELU.

    The reference keeps this inside the vision model (modeling_qwen3_5.py,
    Qwen3_5VisionPatchMerger); it reads the trunk sequence here so the tower
    and projector parities stay separate.
    """

    hidden_size: int
    spatial_merge_size: int
    out_hidden_size: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")
        grown = self.hidden_size * self.spatial_merge_size ** 2
        self.fc1 = nn.Dense(grown, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="fc1")
        self.fc2 = nn.Dense(self.out_hidden_size, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="fc2")

    def __call__(self, image_features) -> jax.Array:
        batch, count, _ = image_features.shape
        block = self.spatial_merge_size ** 2
        grown = self.hidden_size * block
        if count % block:
            raise ValueError(
                f"{count} patch features do not group into "
                f"{self.spatial_merge_size}x{self.spatial_merge_size} merge blocks")
        grouped = self.norm(image_features).reshape(batch, count // block, grown)
        return self.fc2(gelu_exact(self.fc1(grouped)))


@dataclasses.dataclass(frozen=True)
class Qwen35Projector(ProjectorBase):
    """Qwen 3.5's projector fields: the trunk width, the merge size, the
    merged width the text embeddings read directly."""

    vision_width: int
    merge_size: int
    out_width: int

    def build(self) -> nn.Module:
        return Qwen35ProjectorModule(hidden_size=self.vision_width,
                                     spatial_merge_size=self.merge_size,
                                     out_hidden_size=self.out_width)


_QWEN35_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "pos_embed.weight": ("position_table", "embedding"),
}


# A block's norms and maps as `clip_layer`'s decoder block names them.
_BLOCK_NORMS = {"norm1": "input_layernorm", "norm2": "post_attention_layernorm"}
_BLOCK_MAPS = {("attn", "qkv"): ("self_attn", "qkv_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "linear_fc1"): ("mlp", "up_proj"), ("mlp", "linear_fc2"): ("mlp", "down_proj")}


def _qwen35_vision_block_path(parts) -> tuple[str, ...] | None:
    """`blocks.N...` into the block's path."""
    if (
        len(parts) < 4
        or parts[0] != "blocks"
        or not parts[1].isdigit()
        or parts[-1] not in ("weight", "bias")
    ):
        return None
    block, leaf = f"blocks_{parts[1]}", parts[-1]
    if len(parts) == 4 and parts[2] in _BLOCK_NORMS:
        return (block, _BLOCK_NORMS[parts[2]], "scale" if leaf == "weight" else "bias")
    if len(parts) == 5 and (parts[2], parts[3]) in _BLOCK_MAPS:
        return (block, *_BLOCK_MAPS[parts[2], parts[3]], "kernel" if leaf == "weight" else "bias")
    return None


def qwen35_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Qwen 3.5 vision tensor name into its path in a trunk tree.

    The merger lives under its own prefix and maps with the projector; the
    rotary table is a buffer the checkpoint leaves out. Anything else unknown
    raises ValueError.
    """
    if hf_name.split(".")[0] == "merger":
        return None
    path = _QWEN35_VISION_TENSORS.get(hf_name) or _qwen35_vision_block_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_qwen35_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Qwen 3.5 vision tensors into a trunk parameter tree.

    The patch convolution carries [out, in, time, h, w] and lands as one map;
    the trunk's buffer already holds the viewed channel order, so the rows
    are a plain reshape and every other leaf rides the shared transpose.
    """
    rest = {name: tensor for name, tensor in hf_tensors.items()
            if name != "patch_embed.proj.weight"}
    params = translate_parameters(rest, qwen35_vision_path, param_dtype)
    conv = checkpoint_array(hf_tensors["patch_embed.proj.weight"], param_dtype)
    params.setdefault("patch_embed", {})["kernel"] = np.ascontiguousarray(
        conv.transpose(1, 2, 3, 4, 0).reshape(-1, conv.shape[0]))
    return params


def translate_qwen35_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Qwen3_5VisionConfig into a Qwen35Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The
    released 0.8B checkpoint spells the tower's model_type qwen3_5, which the
    config class rewrites on load; both spellings map. The position table
    must be square, and the activation one the shared MLP runs.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "qwen3_5_vision") not in _QWEN35_VISION_TYPES:
        raise ValueError(
            f"vision model_type {vision.get('model_type')!r} is not the Qwen 3.5 tower")
    table = records.integer(vision["num_position_embeddings"], "num_position_embeddings")
    if int(table ** 0.5) ** 2 != table:
        raise ValueError(
            f"num_position_embeddings ({table}) is not a square, this trunk "
            "resamples a square table")
    activation = str(vision.get("hidden_act", "gelu_pytorch_tanh"))
    if activation not in ("quick_gelu", "gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this trunk runs the "
            "shared MLP's activations")
    return {
        "class": "qwen3_5", "fields": {
        "depth": records.integer(vision["depth"], "depth"),
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "hidden_act": activation,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_heads": records.integer(vision["num_heads"], "num_heads"),
        "in_channels": records.integer(vision.get("in_channels", 3), "in_channels"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "spatial_merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "temporal_patch_size": records.integer(vision["temporal_patch_size"], "temporal_patch_size"),
        "num_position_embeddings": table,
    }}


def translate_qwen35_projector_config(hf_config: Mapping[str, object],
                                      text_width: int) -> Record:
    """A Qwen 3.5 wrapper's projector fields: trunk width, merge, output.

    The merged features enter the text embeddings directly, so a merger width
    beside the decoder width refuses.
    """
    vision = _vision_section(hf_config)
    merged = records.integer(vision["out_hidden_size"], "out_hidden_size")
    if merged != int(text_width):
        raise ValueError(
            f"out_hidden_size ({merged}) is not the decoder width ({text_width}), "
            "the merger output enters the text embeddings as it is")
    return {
        "class": "qwen3_5", "fields": {
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "out_width": merged,
    }}
