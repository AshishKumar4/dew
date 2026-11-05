"""The shared vision transformer and the merge projector.

A family's config states its `ViTLayout`: the patch map and the learned position table it resamples,
the block (its norm, heads and feed-forward), the 2D rotary's axis order, and per layer the attention
(full over a frame, a band of keys either side of each query, or a square window of patches) or a
gated convolution over the frame's patch grid, the token order the layer runs in and whether it adds
attention sinks. Packed pixels arrive as the
family's processor writes them, `[B, patches, pixels]` with each row's `[time, height, width]` grid;
NCHW images use either a convolution or packed-pixel projection. The layout also states class and
register placement, learned-position arithmetic and which rows return after the final norm.

Each family's reference, transformers 5.16.1's `models/qwen3_5/modeling_qwen3_5.py` and
`models/muse_glimmer/modeling_muse_glimmer.py`, MiMo-V2.6's `modeling_mimo_v2.py` and LFM3-VL's
`modeling_lfm3_vl.py`, names its parts; `qwen35`, `mimo`, `muse_glimmer` and `lfm3` hold the configs and
checkpoint maps.
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
from dew.interop.unread import Stored, held
from dew.interop.weights import Tree, checkpoint_array, insert, placed
from dew.nn.activations import activation as named_activation
from dew.nn.backbones.decoder_block import DecoderBlock
from dew.nn.blocks import table_rows, torch_bicubic_resize, torch_cubic_weights
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.flops import matching_lengths, patch_lengths, projection_flops
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixers.short_conv import GatedGridConv
from dew.nn.mlp import GatedMLP
from dew.nn.norms import LayerNorm, RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import sinusoid_frequencies, sinusoids
from dew.nn.scatter import gather_positions
from dew.nn.sharding import declared
from dew.nn.text_encoders import clip_layer

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

Attention = Literal["full", "band", "window", "conv", "item"]
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
    border: bool = False
    """Out-of-range bilinear taps held at the table's edge, as `F.interpolate` holds them (MoonViT's),
    where `F.grid_sample`'s zero padding weighs them zero; bicubic taps are held there always."""
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
        # A singleton batch axis a checkpoint holds names none.
        axes = (None,) * (len(shape) - 1) + ("embed",)
        table = self.param("embedding", declared(self.specification.initializer, axes), shape).reshape(
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

    `attention`, `order` and `sinks` name each layer's: its keys (a frame's every patch, a row's every
    patch across its frames (`item`, MoonViT's), `band` keys
    either side in the layer's order, or the patches of its `window`-patch square) or, for `conv`, a gated
    `conv_kernel`-square convolution over its frame's patch grid in the attention's place (LFM3-VL's,
    `GatedGridConv`, which reads each patch's grid place as its rotary position), the order its tokens run
    in (the processor's, merge blocks column by column, or window by window), and whether a learned logit
    per head joins the first key of each frame (MiMo's sinks). `rotary_columns_first` turns the first
    rotary half by the column and the second by the row, both counted from `rotary_origin`, or under
    `rotary_interleaved` the two axes pair by pair (MoonViT's); `positions` keeps a fixed grid or resizes
    its square table by its stated interpolation. Bilinear sampling without aligned corners gives out-of-
    range taps zero weight (`F.grid_sample`'s zero padding). `merge_blocks` is the processor's patch
    order: merge block by merge block, else raster; `channels_first` its pixel order in a patch, channels
    then time, else time then channels. `mlp_packed` keeps a gated feed-forward's gate and up maps in one
    (as `packed` the attention's queries, keys and values). `post_norm` names the norm after the blocks, a
    layer or an RMS norm, or none. `output` selects the sequence, patches alone or a `[B, rows, columns,
    F]` patch grid; `post_norm_scope` distinguishes CLIP's normalized class row from a normalized whole
    sequence. `pre_norm` names the norm after the patch map, as `post_norm` does. `attention_bias` biases
    the attention's maps, and `qk_norm` norms each head's queries and keys (an RMS norm at the layers'
    epsilon) before the rotary; under `scale_after_cast` every RMS norm weighs its normalized input after
    its cast back, as Llama's does. An RMS norm's epsilon None (`norm_eps`, `outer_norm_eps`) is the
    machine epsilon of the dtype its input accumulates in (`RMSNorm.epsilon`), as torch's
    RMSNorm(eps=None) takes it (Kimi K3's MoonViT).
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
    norm_eps: float | None
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
    pre_norm: Literal["layer", "rms"] | None = None
    post_norm: Literal["layer", "rms"] | None = None
    post_norm_scope: Literal["sequence", "class"] = "sequence"
    post_norm_affine: bool = True
    outer_norm_eps: float | None = 1e-5
    mlp_packed: bool = False
    output: Literal["sequence", "patches", "patch_grid"] = "sequence"
    frame_positions: int | None = None
    """A fixed 1-D sinusoidal table of this many frames, MAE's computed in float32
    (`dew.nn.rope.sinusoid_frequencies`' `powered`), added with the learned positions to each frame's
    patches of a row of more than one frame (MoonViT's `Learnable2DInterpPosEmbDivided_fixed`)."""
    pool_frames: bool = False
    """Whether a row's merge blocks are averaged over its frames after the blocks, the row returning its
    first frame's places and zeros past them (MoonViT's temporal pooling, `tpool_patch_merger`)."""
    branch_scale: float | None = None
    rope_theta: float = 10000.0
    rotary_columns_first: bool = False
    rotary_interleaved: bool = False
    rotary_origin: int = 0
    band: int = 0
    window: int = 0
    merge_blocks: bool = True
    channels_first: bool = True
    attention_bias: bool = True
    qk_norm: bool = False
    conv_kernel: int = 3
    scale_after_cast: bool = False

    def __post_init__(self):
        if not len(self.attention) == len(self.order) == len(self.sinks) == self.depth:
            raise ValueError(f"{self.depth} layers need an attention, an order and a sinks flag each")
        if (self.norm_eps is None and (self.norm_type != "rms" or self.qk_norm)) or (
                self.outer_norm_eps is None and "layer" in (self.pre_norm, self.post_norm)):
            raise ValueError("an epsilon of the computing dtype's (None) is an RMS norm's, as torch's "
                             "RMSNorm(eps=None) takes it; a layer norm or a query and key norm states its "
                             "own")
        if self.rotary is not None and self.head_dim % 4:
            raise ValueError(f"head_dim {self.head_dim} splits into no height and width rotary pairs")
        if self.special_tokens.registers and self.special_tokens.cls != "first":
            raise ValueError("register tokens follow a leading class token")
        if self.post_norm_scope == "class" and self.special_tokens.cls is None:
            raise ValueError("a class-only output norm needs a class token")
        if "conv" in self.attention and (self.rotary is None or self.rotary_columns_first
                                         or self.rotary_origin or self.special_tokens.count):
            raise ValueError("a grid convolution reads each patch's (row, column) from 0 as its rotary "
                             "position, which a tower without its rotary, turning columns first, counting "
                             "from another origin or holding special tokens does not state")


def _interp_taps(index, size, side: int, align_corners: bool, dtype, *, border: bool = False,
                 cubic: bool = False):
    """Bilinear taps into a `side`-long table for positions along one axis, and their weights in
    `at_least_fp32(dtype)`: transformers' `_interpolation_axis_taps_weights` (vision_utils.py) with
    aligned corners, or with half-pixel centres and out-of-range taps weighing zero, or under `border`
    held at the table's edge (`F.interpolate`'s); `cubic`'s four taps by the Keys kernel
    (`dew.nn.blocks.torch_cubic_weights`), held at the edge."""
    index = index.astype(at_least_fp32(dtype))
    if align_corners:
        source = index * (side - 1) / jnp.maximum(size - 1, 1)
    else:
        source = (index + 0.5) * side / size - 0.5
    floor = jnp.floor(source)
    distance = source - floor
    if cubic:
        raw = floor.astype(jnp.int32)[..., None] + jnp.arange(-1, 3)
        return jnp.clip(raw, 0, side - 1), torch_cubic_weights(distance)
    raw = floor.astype(jnp.int32)[..., None] + jnp.arange(2)
    weights = jnp.concatenate([1 - distance[..., None], distance[..., None]], axis=-1).clip(min=0)
    if not (align_corners or border):
        weights = weights * ((raw >= 0) & (raw <= side - 1))
    return jnp.clip(raw, 0, side - 1), weights


def resampled_positions(table, rows, columns, heights, widths, *, align_corners: bool, dtype=None,
                        border: bool = False, cubic: bool = False):
    """The learned square table resampled to each patch's grid place: bilinear or bicubic over its rows
    and columns separately, the outer product of the taps (vision_utils.py,
    get_vision_interpolation_indices_and_weights; `_interp_taps`)."""
    side, dtype = round(len(table) ** 0.5), table.dtype if dtype is None else dtype
    taps = functools.partial(_interp_taps, side=side, align_corners=align_corners, dtype=dtype,
                             border=border, cubic=cubic)
    (row_taps, row_weights), (column_taps, column_weights) = taps(rows, heights), taps(columns, widths)
    taps = row_taps.shape[-1] * column_taps.shape[-1]
    indices = (row_taps[..., :, None] * side + column_taps[..., None, :]).reshape(*rows.shape, taps)
    weights = (row_weights[..., :, None] * column_weights[..., None, :]).reshape(*rows.shape, taps)
    gathered = table_rows(table, indices, dtype)
    return (gathered * weights[..., None]).sum(axis=-2)


def _stated(epsilon: float | None) -> float:
    """A layer norm's epsilon, which a layout states (`ViTLayout.__post_init__`)."""
    if epsilon is None:
        raise ValueError("a layer norm states its epsilon; its input dtype's (None) is an RMS norm's")
    return epsilon


def vit_block(layout: ViTLayout, index: int, *, dtype: Dtype | None,
              precision: PrecisionLike) -> DecoderBlock:
    """Layer `index`'s block (`clip_layer`): a pre-norm attention over the 2D rotary, over a band of
    keys where the layout says, or a grid convolution in its place, and a pre-norm feed-forward, both
    residual."""
    band = layout.attention[index] == "band"
    mixer = None
    if layout.attention[index] == "conv":
        mixer = functools.partial(GatedGridConv, emb_features=layout.hidden_size, kernel=layout.conv_kernel,
                                  dtype=dtype, precision=precision)
    return clip_layer(layout.hidden_size, layout.num_heads, layout.intermediate_size, 1,
                      kv_heads=layout.num_kv_heads, head_dim=layout.head_dim, activation=layout.activation,
                      eps=layout.norm_eps, norm_type=layout.norm_type,
                      norm_bias=layout.norm_bias,
                      mlp_bias=layout.mlp_bias, mlp_packed=layout.mlp_packed,
                      rotary_axes=None if layout.rotary is None else (layout.head_dim // 2,) * 2,
                      rotary_pairs=layout.rotary or "half", rotary_interleaved=layout.rotary_interleaved,
                      rope_theta=layout.rope_theta,
                      branch_scale=layout.branch_scale, attention_bias=layout.attention_bias,
                      qk_norm=layout.qk_norm, scale_after_cast=layout.scale_after_cast, mixer=mixer,
                      packed=layout.packed, window=layout.band if band else None, dtype=dtype,
                      precision=precision, name=f"blocks_{index}")


def _gather(x, index):
    """`x`'s rows `[B, L, ...]` taken in `index` `[B, L]` order; None takes them as they are."""
    if x is None or index is None:
        return x
    return gather_positions(x, index)


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
        outer = lay.outer_norm_eps
        if lay.post_norm == "layer":
            self.post_norm = LayerNorm(epsilon=_stated(outer), use_scale=lay.post_norm_affine,
                                       use_bias=lay.post_norm_affine, dtype=self.dtype, name="post_norm")
        elif lay.post_norm == "rms":
            self.post_norm = RMSNorm(epsilon=outer, with_scale=lay.post_norm_affine,
                                     scale_after_cast=lay.scale_after_cast, dtype=self.dtype,
                                     name="post_norm")

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

        hidden = self._embedded(pixel_values, pixels, rows, columns, heights, widths, images,
                                frame=frame, times=grid[:, 0:1])
        tokens = lay.special_tokens
        if lay.pre_norm == "layer":
            hidden = LayerNorm(epsilon=_stated(lay.outer_norm_eps), dtype=self.dtype,
                               name="pre_norm")(hidden)
        elif lay.pre_norm == "rms":
            hidden = RMSNorm(epsilon=lay.outer_norm_eps,
                             scale_after_cast=lay.scale_after_cast,
                             dtype=self.dtype, name="pre_norm")(hidden)

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
            "conv": jnp.where(valid, frame + 1, 0),
            "item": jnp.where(valid, 1, 0),
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
        if lay.pool_frames:
            # Each place's mean over the row's frames, at the place's own index in its first frame, which
            # the merge blocks then gather as one frame's.
            summed = jax.vmap(lambda row, at: jnp.zeros_like(row).at[at].add(row))(
                jnp.where(valid[..., None], hidden, 0), jnp.where(valid, within, 0))
            # A padded row's grid holds no frame: it divides by one, its places kept zero.
            frames = jnp.maximum(grid[:, 0:1, None], 1).astype(summed.dtype)
            hidden = jnp.where((valid & (place < area))[..., None], summed / frames, 0)
        if not lay.merge_blocks and merge > 1:
            # Raster patches into merge blocks, each block's patches consecutive (pixel_shuffle's gather).
            block = frame * area + (rows // merge) * merge * widths + (columns // merge) * merge * merge
            key = block + (rows % merge) * merge + columns % merge
            hidden = _gather(hidden, jnp.argsort(jnp.where(valid, key, last), axis=1, stable=True))
        return hidden

    def _embedded(self, pixel_values, pixels, rows, columns, heights, widths, images, *, frame, times):
        """Patch projection, positioned patch/class rows, and unpositioned register insertion; a row of
        more than one frame (`times`) adds each `frame`'s fixed position (`ViTLayout.frame_positions`)."""
        lay = self.layout
        if lay.patch_projection == "conv":
            if not images:
                raise ValueError("convolutional patch projection takes NCHW images")
            hidden = Conv(lay.hidden_size, (lay.patch_size, lay.patch_size),
                          strides=(lay.patch_size, lay.patch_size), padding="VALID", use_bias=lay.patch_bias,
                          kernel_init=declared(nn.initializers.lecun_normal(), (None, None, None, "embed")),
                          bias_init=declared(nn.initializers.zeros_init(), ("embed",)),
                          dtype=self.dtype, precision=self.precision, name="patch_embed")(
                              jnp.asarray(pixel_values).transpose(0, 2, 3, 1))
            hidden = hidden.reshape(hidden.shape[0], pixels.shape[1], lay.hidden_size)
        else:
            hidden = Dense(lay.hidden_size, use_bias=lay.patch_bias, dtype=self.dtype,
                           precision=self.precision, name="patch_embed", kernel_axes=(None, "embed"),
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
            if spec.mode == "bilinear" or (spec.mode == "bicubic" and not images):
                # A packed row's grid is its own: bicubic positions resample per place (`F.interpolate`'s).
                position = resampled_positions(table, rows, columns, heights, widths,
                                                align_corners=spec.align_corners, dtype=compute,
                                                border=spec.border, cubic=spec.mode == "bicubic")
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
            if lay.frame_positions is not None:
                width = lay.hidden_size // 2
                frequencies = sinusoid_frequencies(width, width, dtype=np.float32, powered=True)
                table = sinusoids(np.arange(lay.frame_positions, dtype=np.float32)[:, None] * frequencies,
                                  xp=np)
                at = jnp.minimum(frame, lay.frame_positions - 1)
                position = position + jnp.where((times > 1)[..., None], jnp.asarray(table, compute)[at], 0)
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
    first at `norm_eps` (Qwen 3.5's, MiMo's and LFM3-VL's); `bias` gives both maps their biases, and the
    norm its own where `norm_bias` is unset (LFM3-VL's norm keeps its bias either way). Each block's
    `merge**2` features join unit by unit, or channel by channel where `channel_major` (Muse
    Glimmer's `pixel_shuffle`), and pass two maps with the checkpoint's `activation` (its
    `hidden_act` name) between them, and after the second too where `activation_after` (Muse
    Glimmer's adapter); with `post_norm`, an RMS norm at `norm_eps` after them (Kimi K3's
    PatchMergerMLPV2).
    """

    hidden_size: int
    spatial_merge_size: int
    width: int
    out_width: int
    norm: bool = True
    norm_eps: float = 1e-6
    norm_bias: bool | None = None
    channel_major: bool = False
    bias: bool = True
    activation: str = "gelu"
    activation_after: bool = False
    post_norm: bool = False
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
            norm_bias = self.bias if self.norm_bias is None else self.norm_bias
            image_features = LayerNorm(epsilon=self.norm_eps, use_bias=norm_bias, dtype=self.dtype,
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
        projected = mlp(grouped)
        if self.post_norm:
            projected = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="post_norm")(projected)
        return projected


PATCH_CONVOLUTION = "patch_embed.proj.weight"
"""Where the Qwen-style towers keep their patch map: a `[out, in, time, h, w]` convolution, or MoonViT's
`[out, in, h, w]`."""


def translate_vit_weights(hf_tensors: Mapping[str, Stored],
                          path_of: Callable[[str], tuple[str, ...] | None],
                          param_dtype: str = "float32") -> tuple[Tree[Stored], tuple[WeightLayout, ...]]:
    """A trunk's variables from its tensors through its family's `path_of`, rooted at the collection,
    the patch convolution `PATCH_CONVOLUTION`, where the checkpoint has one, landing as one Dense map
    over the processor's channel, time and pixel order at `params/patch_embed/kernel`; with the
    layouts that write each tensor back from them (`placed`)."""
    from dew.interop.streaming import WeightLayout

    variables, layouts = placed({name: tensor for name, tensor in hf_tensors.items()
                                 if name != PATCH_CONVOLUTION}, path_of, param_dtype)
    if PATCH_CONVOLUTION in hf_tensors:
        conv = checkpoint_array(hf_tensors[PATCH_CONVOLUTION], param_dtype)
        dense = held(conv.transpose(*range(1, conv.ndim), 0).reshape(-1, conv.shape[0]), contiguous=True)
        insert(variables, ("params", "patch_embed", "kernel"), dense, PATCH_CONVOLUTION)
        # The Dense map's transpose is the convolution, its input axes flattened.
        kernel = ("params", "patch_embed", "kernel")
        layouts += (WeightLayout(PATCH_CONVOLUTION, (kernel,), conv.shape, (1, 0)),)
    return variables, layouts
