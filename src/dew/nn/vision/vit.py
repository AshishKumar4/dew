"""The shared vision transformer and the merge projector.

A family's config states its `ViTLayout`: the patch map and the learned position table it resamples,
the block (its norm, heads and feed-forward), the 2D rotary's axis order, and per layer the attention
(full over a frame, a band of keys either side of each query, or a square window of patches), the
token order the layer runs in and whether it adds attention sinks. Packed pixels arrive as the
family's processor writes them, `[B, patches, pixels]` with each row's `[time, height, width]` grid;
NCHW images use either a convolution or packed-pixel projection. The layout also states class and
register placement, learned-position arithmetic and which rows return after the final norm.

Each family's reference, transformers 5.16.1's `models/qwen3_5/modeling_qwen3_5.py` and
`models/muse_glimmer/modeling_muse_glimmer.py` and MiMo-V2.6's `modeling_mimo_v2.py`, names its parts;
`qwen35`, `mimo` and `muse_glimmer` hold the configs and checkpoint maps.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, Initializer, PrecisionLike

from dew.batch import coordinates
from dew.interop.unread import held
from dew.interop.weights import ParamTree, checkpoint_array, insert, placed
from dew.nn.activations import activation as named_activation
from dew.nn.attention import LayerNorm, RMSNorm
from dew.nn.backbones.decoder_block import DecoderBlock
from dew.nn.blocks import table_rows, torch_bicubic_resize
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.flops import matching_lengths, patch_lengths, projection_flops
from dew.nn.inputs import AttentionMetadata
from dew.nn.mlp import GatedMLP
from dew.nn.precision import float32_dot_general
from dew.nn.sharding import logical_axes
from dew.nn.text_encoders import clip_layer

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

Attention = Literal["full", "band", "window"]
Order = Literal["patches", "columns", "windows"]


@dataclasses.dataclass(frozen=True)
class SpecialTokens:
    """A class token and its unpositioned registers. Registers follow a leading class token."""

    cls: Literal["first", "last"] | None = None
    registers: int = 0
    singleton: bool = False
    compute_dtype: bool = True
    initializer: Initializer = nn.initializers.zeros

    @property
    def count(self) -> int:
        return int(self.cls is not None) + self.registers


@dataclasses.dataclass(frozen=True)
class LearnedPositions:
    """The square patch table and its resizing arithmetic; a class position follows the token order."""

    side: int
    mode: Literal["fixed", "bilinear", "bicubic"] = "bilinear"
    singleton: bool = False
    compute_dtype: bool = True
    align_corners: bool = True
    antialias: bool = False
    initializer: Initializer = nn.Embed.embedding_init


class _PositionTable(nn.Module):
    """The stored rank of a learned table, including checkpoints with a singleton batch axis."""

    rows: int
    features: int
    specification: LearnedPositions
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, indices=None):
        shape = (self.rows, self.features)
        if self.specification.singleton:
            shape = (1, *shape)
        table = self.param("embedding", self.specification.initializer, shape).reshape(
            self.rows, self.features)
        return table if indices is None else table_rows(table, indices, self.dtype or table.dtype)


def _documents(segment):
    """The `Coordinates` of a layer's `segment` ids, each token keeping its index as its position (the
    rotation is the metadata's own); None for a layer that attends unmasked."""
    if segment is None:
        return None
    return coordinates(segments=segment, positions=jnp.broadcast_to(jnp.arange(segment.shape[1]),
                                                                     segment.shape))


@dataclasses.dataclass(frozen=True)
class ViTLayout:
    """The geometry and wiring of one family's tower.

    `attention`, `order` and `sinks` name each layer's: its keys (a frame's every patch, `band` keys
    either side in the layer's order, or the patches of its `window`-patch square), the order its
    tokens run in (the processor's, merge blocks column by column, or window by window), and whether
    a learned logit per head joins the first key of each frame (MiMo's sinks). `rotary_columns_first`
    turns the first rotary half by the column and the second by the row, both counted from
    `rotary_origin`; `positions` keeps a fixed grid or resizes its square table by its stated
    interpolation. Bilinear sampling without aligned corners gives out-of-range taps zero weight
    (`F.grid_sample`'s zero padding). `merge_blocks` is the processor's patch order: merge block by
    merge block, else raster; `channels_first` its pixel order in a patch, channels then time, else
    time then channels. `mlp_packed` keeps a gated feed-forward's gate and up maps in one (as
    `packed` the attention's queries, keys and values). `post_norm` names the norm after the
    blocks, a layer or an RMS norm, or none. `output` selects the sequence, patches alone or a
    `[B, rows, columns, F]` patch grid; `post_norm_scope` distinguishes CLIP's normalized class row
    from a normalized whole sequence.
    """

    depth: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    activation: str | Callable[[jax.Array], jax.Array]
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    norm_type: Literal["rms", "layer", "fast_layer"]
    norm_eps: float
    norm_bias: bool
    mlp_bias: bool
    packed: bool
    patch_bias: bool
    attention: tuple[Attention, ...]
    order: tuple[Order, ...]
    sinks: tuple[bool, ...]
    positions: LearnedPositions | None = None
    special_tokens: SpecialTokens = SpecialTokens()
    patch_projection: Literal["dense", "conv"] = "dense"
    image_size: int | None = None
    rotary: Literal["half", "adjacent"] | None = "half"
    pre_norm: bool = False
    post_norm: Literal["layer", "rms"] | None = None
    post_norm_scope: Literal["sequence", "class"] = "sequence"
    post_norm_affine: bool = True
    outer_norm_eps: float = 1e-5
    mlp_packed: bool = False
    output: Literal["sequence", "patches", "patch_grid"] = "sequence"
    branch_scale: float | None = None
    rope_theta: float = 10000.0
    rotary_columns_first: bool = False
    rotary_origin: int = 0
    band: int = 0
    window: int = 0
    merge_blocks: bool = True
    channels_first: bool = True

    def __post_init__(self):
        if not len(self.attention) == len(self.order) == len(self.sinks) == self.depth:
            raise ValueError(f"{self.depth} layers need an attention, an order and a sinks flag each")
        if self.rotary is not None and self.head_dim % 4:
            raise ValueError(f"head_dim {self.head_dim} splits into no height and width rotary pairs")
        if self.special_tokens.registers and self.special_tokens.cls != "first":
            raise ValueError("register tokens follow a leading class token")
        if self.post_norm_scope == "class" and self.special_tokens.cls is None:
            raise ValueError("a class-only output norm needs a class token")


def _interp_taps(index, size, side: int, align_corners: bool):
    """Bilinear taps into a `side`-long table for positions along one axis, and their weights:
    transformers' `_interpolation_axis_taps_weights` (vision_utils.py) with aligned corners, or with
    half-pixel centres and out-of-range taps weighing zero."""
    index = index.astype(jnp.float32)
    if align_corners:
        source = index * (side - 1) / jnp.maximum(size - 1, 1)
    else:
        source = (index + 0.5) * side / size - 0.5
    floor = jnp.floor(source)
    raw = floor.astype(jnp.int32)[..., None] + jnp.arange(2)
    distance = (source - floor)[..., None]
    weights = jnp.concatenate([1 - distance, distance], axis=-1).clip(min=0)
    if not align_corners:
        weights = weights * ((raw >= 0) & (raw <= side - 1))
    return jnp.clip(raw, 0, side - 1), weights


def resampled_positions(table, rows, columns, heights, widths, *, align_corners: bool, dtype=None):
    """The learned square table resampled to each patch's grid place: bilinear over its rows and
    columns separately, the outer product of the taps (vision_utils.py,
    get_vision_interpolation_indices_and_weights)."""
    side = round(len(table) ** 0.5)
    row_taps, row_weights = _interp_taps(rows, heights, side, align_corners)
    column_taps, column_weights = _interp_taps(columns, widths, side, align_corners)
    indices = (row_taps[..., :, None] * side + column_taps[..., None, :]).reshape(*rows.shape, 4)
    weights = (row_weights[..., :, None] * column_weights[..., None, :]).reshape(*rows.shape, 4)
    gathered = table_rows(table, indices, table.dtype if dtype is None else dtype)
    return (gathered * weights[..., None]).sum(axis=-2)


def vit_block(layout: ViTLayout, index: int, *, dtype: Dtype | None,
              precision: PrecisionLike) -> DecoderBlock:
    """Layer `index`'s block (`clip_layer`): a pre-norm attention over the 2D rotary, over a band of
    keys where the layout says, and a pre-norm feed-forward, both residual."""
    band = layout.attention[index] == "band"
    return clip_layer(layout.hidden_size, layout.num_heads, layout.intermediate_size, 1,
                      kv_heads=layout.num_kv_heads, head_dim=layout.head_dim, activation=layout.activation,
                      eps=layout.norm_eps, norm_type=layout.norm_type, norm_bias=layout.norm_bias,
                      mlp_bias=layout.mlp_bias, mlp_packed=layout.mlp_packed,
                      rotary_axes=None if layout.rotary is None else (layout.head_dim // 2,) * 2,
                      rotary_pairs=layout.rotary or "half", rope_theta=layout.rope_theta,
                      branch_scale=layout.branch_scale,
                      packed=layout.packed, window=layout.band if band else None, dtype=dtype,
                      precision=precision, name=f"blocks_{index}")


def _gather(x, index):
    """`x`'s rows `[B, L, ...]` taken in `index` `[B, L]` order; None takes them as they are."""
    if x is None or index is None:
        return x
    return jnp.take_along_axis(x, index.reshape(*index.shape, *(1,) * (x.ndim - 2)), axis=1)


@logical_axes({("patch_embed",): (None, None, None, "embed"),
               ("position_table",): (None, None, "embed")})
class VisionTransformer(nn.Module):
    """A family's trunk (`ViTLayout`): patches mapped and placed, the blocks, and the trunk sequence
    in merge-block order.

    Packed pixels `[B, patches, pixels]` come with each row's `[time, height, width]` patch grid, a
    row's padding past its patches; attention stays inside a frame, or a window of it, and padding
    (segment 0) sees nothing. NCHW images are one frame each, repeated along time, packed in the
    processor's order. Each layer runs in its own token order, the tokens, their rotary places and
    their frames gathered into it from the previous layer's; the sequence returns to the processor's
    order after the last.
    """

    layout: ViTLayout
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        lay = self.layout
        if lay.positions is not None:
            self.position_table = _PositionTable(
                lay.positions.side ** 2 + int(lay.special_tokens.cls is not None), lay.hidden_size,
                lay.positions, dtype=self.dtype if lay.positions.compute_dtype else None,
                name="position_table")
        if lay.post_norm == "layer":
            self.post_norm = LayerNorm(epsilon=lay.outer_norm_eps, use_scale=lay.post_norm_affine,
                                       use_bias=lay.post_norm_affine, dtype=self.dtype, name="post_norm")
        elif lay.post_norm == "rms":
            self.post_norm = RMSNorm(epsilon=lay.outer_norm_eps, with_scale=lay.post_norm_affine,
                                     dtype=self.dtype, name="post_norm")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (self.layout.patch_size, self.layout.patch_size), rows)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        lay = self.layout
        length, = matching_lengths(lengths)
        pixels = lay.in_channels * lay.temporal_patch_size * lay.patch_size ** 2
        return projection_flops(lengths, pixels, lay.hidden_size) + sum(
            vit_block(lay, index, dtype=self.dtype, precision=self.precision).model_flops(
                (length + lay.special_tokens.count).astype(np.int64)) for index in range(lay.depth))

    def __call__(self, pixel_values, grid_thw=None) -> jax.Array:
        return self._finished(self._sequence(pixel_values, grid_thw), pixel_values)

    def sequence_and_pool(self, pixel_values, grid_thw=None) -> tuple[jax.Array, jax.Array]:
        """The selected sequence and its class row through the declared post norm."""
        cls = self.layout.special_tokens.cls
        if cls is None:
            raise ValueError("class pooling needs a class token")
        hidden = self._sequence(pixel_values, grid_thw)
        pooled = hidden[:, 0 if cls == "first" else -1]
        if self.layout.post_norm is not None and self.layout.post_norm_scope == "class":
            pooled = self.post_norm(pooled)
        return self._finished(hidden, pixel_values), pooled

    @nn.compact
    def _sequence(self, pixel_values, grid_thw=None) -> jax.Array:
        lay = self.layout
        pixels, grid, images = self._packed(jnp.asarray(pixel_values), grid_thw)
        length = pixels.shape[1]
        merge = lay.spatial_merge_size
        heights, widths = grid[:, 1:2], grid[:, 2:3]
        area = heights * widths
        place = jnp.arange(length)[None, :]
        within = place % area
        if lay.merge_blocks:
            blocks_wide = widths // merge
            rows = (within // (merge * merge * blocks_wide)) * merge + (within // merge) % merge
            columns = ((within // (merge * merge)) % blocks_wide) * merge + within % merge
        else:
            rows, columns = within // widths, within % widths
        valid = place < jnp.prod(grid, axis=1, keepdims=True)
        frame = place // area

        hidden = self._embedded(pixel_values, pixels, rows, columns, heights, widths, images)
        tokens = lay.special_tokens
        outer = functools.partial(LayerNorm, epsilon=lay.outer_norm_eps, dtype=self.dtype)
        if lay.pre_norm:
            hidden = outer(name="pre_norm")(hidden)

        rotary = None
        if lay.rotary is not None:
            axes = (columns, rows) if lay.rotary_columns_first else (rows, columns)
            rotary = jnp.stack(axes, axis=-1) + lay.rotary_origin
            if tokens.count:
                origin = jnp.zeros((rotary.shape[0], tokens.count, 2), rotary.dtype)
                rotary = jnp.concatenate([origin, rotary] if tokens.cls == "first"
                                         else [rotary, origin], axis=1)
        last = jnp.iinfo(jnp.int32).max
        # Each order's key: a token's place in it, padding last.
        keys = {"patches": place, "columns": None, "windows": None}
        if "columns" in lay.order:
            # Merge blocks column by column, each block's patches as they came (MiMo's window index).
            unit, blocks_wide = within // (merge * merge), widths // merge
            keys["columns"] = (frame * area + (unit % blocks_wide) * (heights // merge) * merge * merge
                               + (unit // blocks_wide) * merge * merge + within % (merge * merge))
        windows = None
        if "window" in lay.attention or "windows" in lay.order:
            side = lay.window
            across = -(-widths // side)
            square = (rows // side) * across + columns // side
            windows = frame * (-(-heights // side)) * across + square
            keys["windows"] = windows * side * side + (rows % side) * side + columns % side
        orders = {name: None if name == "patches"
                  else jnp.argsort(jnp.where(valid, key, last), axis=1, stable=True)
                  for name, key in keys.items() if key is not None}

        segments = {
            # An NCHW image is one frame with no padding: its full layers attend unmasked.
            "full": None if images else jnp.where(valid, frame + 1, 0),
            "band": jnp.where(valid, frame + 1, 0),
            "window": None if windows is None else jnp.where(valid, windows + 1, 0),
        }
        current = "patches"
        for index in range(lay.depth):
            wanted = lay.order[index]
            if wanted != current:
                leaving, entering = orders[current], orders[wanted]
                if leaving is not None:
                    back = jnp.argsort(leaving, axis=1)
                    entering = back if entering is None else jnp.take_along_axis(back, entering, axis=1)
                hidden = _gather(hidden, entering)
                current = wanted
            order = orders[current]
            segment = _gather(segments[lay.attention[index]], order)
            bias = None
            if lay.sinks[index]:
                sinks = self.param(f"sinks_{index}", nn.initializers.zeros, (lay.num_heads,))
                frames = _gather(jnp.where(valid, frame + 1, 0), order)
                first = (frames != jnp.pad(frames, ((0, 0), (1, 0)))[:, :-1]) & (frames > 0)
                bias = jnp.where(first[:, None, None, :], sinks[None, :, None, None], 0.0)
            metadata = AttentionMetadata(coordinates=_documents(segment),
                                         rotary_positions=_gather(rotary, order), position_bias=bias)
            hidden = vit_block(lay, index, dtype=self.dtype, precision=self.precision)(
                hidden, attention_metadata=metadata)
        leaving = orders[current]
        if leaving is not None:
            hidden = _gather(hidden, jnp.argsort(leaving, axis=1))
        if lay.post_norm is not None and lay.post_norm_scope == "sequence":
            hidden = self.post_norm(hidden)
        if not lay.merge_blocks and merge > 1:
            # Raster patches into merge blocks, each block's patches consecutive (pixel_shuffle's gather).
            block = frame * area + (rows // merge) * merge * widths + (columns // merge) * merge * merge
            key = block + (rows % merge) * merge + columns % merge
            hidden = _gather(hidden, jnp.argsort(jnp.where(valid, key, last), axis=1, stable=True))
        return hidden

    def _embedded(self, pixel_values, pixels, rows, columns, heights, widths, images):
        """Patch projection, positioned patch/class rows, and unpositioned register insertion."""
        lay = self.layout
        if lay.patch_projection == "conv":
            if not images:
                raise ValueError("convolutional patch projection takes NCHW images")
            hidden = Conv(lay.hidden_size, (lay.patch_size, lay.patch_size),
                          strides=(lay.patch_size, lay.patch_size), padding="VALID", use_bias=lay.patch_bias,
                          dtype=self.dtype, precision=self.precision, name="patch_embed")(
                              jnp.asarray(pixel_values).transpose(0, 2, 3, 1))
            hidden = hidden.reshape(hidden.shape[0], pixels.shape[1], lay.hidden_size)
        else:
            hidden = Dense(lay.hidden_size, use_bias=lay.patch_bias, dtype=self.dtype,
                           precision=self.precision, name="patch_embed",
                           dot_general=float32_dot_general)(pixels)
        tokens = lay.special_tokens
        class_position = None
        if lay.positions is not None:
            spec = lay.positions
            table = self.position_table()
            compute = hidden.dtype if spec.compute_dtype else table.dtype
            if tokens.cls is not None:
                class_position = (table[:1] if tokens.cls == "first" else table[-1:]).astype(compute)
                table = table[1:] if tokens.cls == "first" else table[:-1]
            if spec.mode == "bilinear":
                position = resampled_positions(table, rows, columns, heights, widths,
                                                align_corners=spec.align_corners, dtype=compute)
            elif spec.mode == "bicubic":
                if not images:
                    raise ValueError("bicubic positions take an image's rectangular patch grid")
                height, width = (side // lay.patch_size for side in jnp.shape(pixel_values)[2:])
                position = table.astype(compute).reshape(1, spec.side, spec.side, lay.hidden_size)
                if (height, width) != (spec.side, spec.side):
                    position = torch_bicubic_resize(position, height, width, antialias=spec.antialias)
                position = position.reshape(1, height * width, lay.hidden_size)
            else:
                position = self.position_table(jnp.arange(table.shape[0]) + int(tokens.cls == "first"))[None]
            hidden = hidden + position
        if tokens.cls is not None:
            shape = (1, 1, lay.hidden_size) if tokens.singleton else (lay.hidden_size,)
            cls = self.param("class_token", tokens.initializer, shape)
            if tokens.compute_dtype:
                cls = cls.astype(hidden.dtype)
            if class_position is not None:
                cls = cls + class_position
            cls = jnp.broadcast_to(cls, (hidden.shape[0], 1, lay.hidden_size))
            leading = [cls]
            if tokens.registers:
                registers = self.param("register_tokens", tokens.initializer,
                                       (1, tokens.registers, lay.hidden_size))
                if tokens.compute_dtype:
                    registers = registers.astype(hidden.dtype)
                leading.append(jnp.broadcast_to(registers, (hidden.shape[0], tokens.registers,
                                                             lay.hidden_size)))
            hidden = jnp.concatenate([*leading, hidden] if tokens.cls == "first" else [hidden, cls], axis=1)
        return hidden

    def _finished(self, hidden: jax.Array, pixel_values) -> jax.Array:
        """Select the public rows and, when requested, restore their patch grid."""
        lay, tokens = self.layout, self.layout.special_tokens
        if lay.output != "sequence":
            start = tokens.count if tokens.cls == "first" else 0
            stop = -1 if tokens.cls == "last" else None
            hidden = hidden[:, start:stop]
        if lay.output == "patch_grid":
            if jnp.ndim(pixel_values) != 4:
                raise ValueError("a tower returning its patch grid takes NCHW images")
            height, width = jnp.shape(pixel_values)[2:]
            hidden = hidden.reshape(hidden.shape[0], height // lay.patch_size, width // lay.patch_size,
                                    lay.hidden_size)
        return hidden

    def _packed(self, pixels, grid_thw):
        """Packed pixels and their grids [B, 3], with the input's image/packed distinction."""
        lay = self.layout
        merge, patch = lay.spatial_merge_size, lay.patch_size
        if pixels.ndim == 4:
            if grid_thw is not None:
                raise ValueError("grid_thw accompanies packed pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            if lay.image_size is not None and (height, width) != (lay.image_size, lay.image_size):
                raise ValueError(f"pixel_values must be {lay.image_size}x{lay.image_size} images")
            stride = patch * merge
            if channels != lay.in_channels or (lay.patch_projection == "dense"
                                               and (height % stride or width % stride)):
                raise ValueError(f"pixel_values must be {lay.in_channels}-channel images tiled by "
                                 f"{stride}px blocks")
            rows, columns = height // patch, width // patch
            grid = jnp.broadcast_to(jnp.asarray([1, rows, columns], jnp.int32), (batch, 3))
            if lay.patch_projection == "conv":
                # Convolution consumes the original image. Its patch sequence needs only geometry,
                # not a second pixel packing (which would also reshape Explicit batch axes).
                geometry = jax.ShapeDtypeStruct(
                    (batch, rows * columns, channels * lay.temporal_patch_size * patch ** 2), pixels.dtype)
                return geometry, grid, True
            if lay.merge_blocks:
                blocks = pixels.reshape(batch, channels, rows // merge, merge, patch, columns // merge, merge,
                                        patch).transpose(0, 2, 5, 3, 6, 1, 4, 7)
            else:
                blocks = pixels.reshape(batch, channels, rows, patch, columns, patch).transpose(
                    0, 2, 4, 1, 3, 5)
            flat = blocks.reshape(batch, rows * columns, 1, channels, patch, patch)
            frames = jnp.broadcast_to(
                flat, (batch, rows * columns, lay.temporal_patch_size, channels, patch, patch))
            if lay.channels_first:
                frames = frames.transpose(0, 1, 3, 2, 4, 5)
            pixels = frames.reshape(batch, rows * columns,
                                    channels * lay.temporal_patch_size * patch ** 2)
            return pixels, grid, True
        if lay.image_size is not None:
            raise ValueError("a fixed-image tower takes NCHW images")
        if pixels.ndim != 3 or pixels.shape[-1] != lay.in_channels * lay.temporal_patch_size * patch ** 2:
            raise ValueError("packed pixels must be [B, patches, C * temporal_patch_size * patch_size**2]")
        if grid_thw is None or jnp.shape(grid_thw) != (pixels.shape[0], 3):
            raise ValueError("packed pixels require aligned [B, 3] grid_thw")
        return pixels, jnp.asarray(grid_thw), False


class MergeProjector(nn.Module):
    """Each merge block's features as one vector, mapped to the decoder width.

    The trunk sequence comes in merge-block order (`VisionTransformer`). With `norm`, a layer norm
    first (Qwen 3.5's and MiMo's merger); `bias` gives it and both maps their biases. Each block's
    `merge**2` features join unit by unit, or channel by channel where `channel_major` (Muse
    Glimmer's `pixel_shuffle`), and pass two maps with the checkpoint's `activation` (its
    `hidden_act` name) between them, and after the second too where `activation_after` (Muse
    Glimmer's adapter).
    """

    hidden_size: int
    spatial_merge_size: int
    width: int
    out_width: int
    norm: bool = True
    channel_major: bool = False
    bias: bool = True
    activation: str = "gelu"
    activation_after: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, image_features) -> jax.Array:
        batch, count, _ = image_features.shape
        block = self.spatial_merge_size ** 2
        if count % block:
            raise ValueError(f"{count} patch features do not group into "
                             f"{self.spatial_merge_size}x{self.spatial_merge_size} merge blocks")
        if self.norm:
            image_features = LayerNorm(epsilon=1e-6, use_bias=self.bias, dtype=self.dtype,
                                       name="norm")(image_features)
        grouped = image_features.reshape(batch, count // block, block, self.hidden_size)
        if self.channel_major:
            grouped = grouped.swapaxes(-1, -2)
        grouped = grouped.reshape(batch, count // block, block * self.hidden_size)
        act = named_activation(self.activation)
        mlp = GatedMLP(self.width, self.out_width, in_features=grouped.shape[-1], activation=act,
                        activation_after=self.activation_after, use_bias=self.bias, names=("fc1", "fc2"),
                        dtype=self.dtype, precision=self.precision)
        nn.share_scope(self, mlp)
        return mlp(grouped)


PATCH_CONVOLUTION = "patch_embed.proj.weight"
"""Where the Qwen-style towers keep their patch map: a `[out, in, time, h, w]` convolution."""


def translate_vit_weights(hf_tensors: Mapping[str, np.ndarray],
                          path_of: Callable[[str], tuple[str, ...] | None],
                          param_dtype: str = "float32") -> tuple[ParamTree, tuple[WeightLayout, ...]]:
    """A trunk's variables from its tensors through its family's `path_of`, rooted at the collection,
    the patch convolution `PATCH_CONVOLUTION`, where the checkpoint has one, landing as one Dense map
    over the processor's channel, time and pixel order at `params/patch_embed/kernel`; with the
    layouts that write each tensor back from them (`placed`)."""
    from dew.interop.streaming import WeightLayout

    variables, layouts = placed({name: tensor for name, tensor in hf_tensors.items()
                                 if name != PATCH_CONVOLUTION}, path_of, param_dtype)
    if PATCH_CONVOLUTION in hf_tensors:
        conv = checkpoint_array(hf_tensors[PATCH_CONVOLUTION], param_dtype)
        dense = held(conv.transpose(1, 2, 3, 4, 0).reshape(-1, conv.shape[0]), contiguous=True)
        insert(variables, ("params", "patch_embed", "kernel"), dense, PATCH_CONVOLUTION)
        # The Dense map's transpose is the convolution, its input axes flattened.
        kernel = ("params", "patch_embed", "kernel")
        layouts += (WeightLayout(PATCH_CONVOLUTION, (kernel,), conv.shape, (1, 0)),)
    return variables, layouts
