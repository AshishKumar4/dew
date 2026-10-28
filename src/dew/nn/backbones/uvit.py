"""Denoise patches with the two U-shaped token transformers.

UViT takes the time and the text as tokens beside the patches. The U-DiT
takes them as adaLN-Zero modulation instead. Both skip each first-half
block's output into its mirror in the second half.
"""

from functools import partial
from typing import Literal

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.nn.dense import Dense, DenseGeneral

from ..attention import LayerNorm
from ..blocks import sinusoidal_time
from ..conv import Conv
from ..dit import (
    ROPE_THETA,
    ModulatedBlock,
    PatchEmbedding,
    RematChoice,
    RematLadder,
    _TransformerOptions,
    remat_block,
    scan_ordered,
)
from ..flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from ..mlp import GatedMLP
from ..precision import at_least_fp32, float32_dot_general
from ..rope import rotary_freqs
from ..scan_orders import unscanned


def u_walk(x, down, middle, merges, up, run):
    """The U-shaped walk of a token stack: each of the `down` blocks' outputs saved, the `middle`
    block, then each of the `up` blocks after its `merges` projection of the stream concatenated with
    the skip saved last. `run(block, x)` applies a block with the stack's own conditioning."""
    skips = []
    for block in down:
        x = run(block, x)
        skips.append(x)
    x = run(middle, x)
    for merge, block in zip(merges, up, strict=True):
        x = run(block, merge(jnp.concatenate([x, skips.pop()], axis=-1)))
    return x


class UViT(_TransformerOptions):
    """Denoises patches as U-ViT does, following baofff/U-ViT's libs/uvit_t2i.py.

    U-ViT is from Bao et al. (2023). A time token and, when there is text, one
    token per text state come before the patches: [time, text..., patches].

    - The time token is the sinusoidal embedding [cos, sin] of the time
      multiplied by `time_scale`, passed through a d -> 4d -> d SiLU MLP when
      `mlp_time_embed` is set. U-ViT reads timesteps on a 0 to 999 scale,
      while Dew's processes give a time in [0, 1].
    - A dense layer projects the text states, and a `patch_size` convolution
      embeds the patches.
    - A learned position table covers every token: one row for the time,
      `text_tokens` rows for the text, and rows for the patches of an
      `image_size` image, each part using its rows from the first.

    Every block is a plain pre-norm transformer block, x + attention(LN(x))
    then x + MLP(LN(x)), with q, k and v projections without bias, an output
    projection with bias, and an MLP `mlp_ratio` wide on exact GELU.
    num_layers / 2 blocks go down, one sits in the middle and num_layers / 2
    go up; each up block first projects [x, skip] back to the width. A final
    LN and dense layer read the patches, which are unpatchified and, with
    `conv`, refined by a 3x3 convolution. The text attends unmasked, padding
    included, as U-ViT's CLIP states do. A Hilbert `scan_order` gathers the
    embedded patches along the curve (`dew.nn.dit.scan_ordered`), and the
    position rows follow that order.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_time_embed: bool = False
    conv: bool = True
    time_scale: float = 999.0
    text_tokens: int = 77
    image_size: int = 512
    scan_order: Literal["raster", "hilbert"] = "raster"

    @nn.nowrap
    def projections(self, input_channels: int = 0, text_features: int = 0) -> dict[str, tuple[int, int]]:
        width = self.emb_features
        maps = {'patch_embed': (self.patch_size**2 * input_channels, width),
                'context_embed': (text_features, width), 'skip_linear': (2 * width, width),
                'decoder_pred': (width, self.patch_size**2 * self.output_channels)}
        if self.mlp_time_embed:
            maps.update({'time_embed_0': (width, 4 * width), 'time_embed_2': (4 * width, width)})
        return maps

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("UViT requires multimodal_flops with input channels and text geometry")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def multimodal_flops(self, lengths, context_lengths, input_channels: int,
                         text_features: int) -> float:
        """The time token, U-shaped skips, all input maps and final image convolution."""
        image, text = matching_lengths(lengths, context_lengths)
        joined, rows = (image + text + 1).astype(np.int64), np.ones(image.shape, np.int64)
        width, patch = self.emb_features, self.patch_size**2
        maps = self.projections(input_channels, text_features)
        count = (projection_flops(lengths, *maps['patch_embed'])
                 + projection_flops(context_lengths, *maps['context_embed']))
        block = ModulatedBlock(width, self.num_heads, modulated=False, **self._block_options(), parent=None)
        count += (self.num_layers + 1) * block.model_flops(joined)
        count += (self.num_layers // 2) * projection_flops(joined, *maps['skip_linear'])
        # decoder_pred reads time/text too, before __call__ slices to the image.
        count += projection_flops(joined, *maps['decoder_pred'])
        if self.mlp_time_embed:
            count += sum(projection_flops(rows, *maps[name]) for name in ('time_embed_0', 'time_embed_2'))
        if self.conv:
            count += self._final_layer(parent=None).model_flops((image * patch).astype(np.int64),
                                                               self.output_channels)
        return count

    @nn.nowrap
    def image_flops(self, lengths, input_channels: int, text_features: int | None = None) -> float:
        if text_features is not None:
            raise MissingFlopsCount("UViT text tokens need multimodal_flops, not only a pooled width")
        return self.multimodal_flops(lengths, np.zeros(np.shape(lengths), np.int64), input_channels, 0)

    def setup(self):
        if self.num_layers % 2:
            raise ValueError(f"UViT splits its layers into a down and an up half; {self.num_layers} "
                             "is odd")
        maps = self.projections()
        def dense(name, **options):
            key = 'skip_linear' if name.startswith('skip_linear_') else name
            return Dense(maps[key][1], dtype=self.dtype, precision=self.precision, name=name, **options,
                         dot_general=float32_dot_general)
        self.patch_embed = PatchEmbedding(patch_size=self.patch_size, embedding_dim=self.emb_features,
                                          dtype=self.dtype, precision=self.precision, name="patch_embed")
        patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = self.param("pos_embed", nn.initializers.truncated_normal(0.02),
                                    (1, 1 + self.text_tokens + patches, self.emb_features))
        if self.mlp_time_embed:
            self.time_mlp = GatedMLP(4 * self.emb_features, self.emb_features, activation=nn.silu,
                                      use_bias=True, names=("time_embed_0", "time_embed_2"),
                                      dtype=self.dtype, precision=self.precision)
            nn.share_scope(self, self.time_mlp)
        self.context_embed = dense("context_embed")
        block = partial(ModulatedBlock, features=self.emb_features, num_heads=self.num_heads, modulated=False,
                        qkv_bias=False, gelu_approximate=False, **self._block_options())
        half = self.num_layers // 2
        self.in_blocks = [block(name=f"in_blocks_{i}") for i in range(half)]
        self.mid_block = block(name="mid_block")
        self.skip_linear = [dense(f"skip_linear_{i}") for i in range(half)]
        self.out_blocks = [block(name=f"out_blocks_{i}") for i in range(half)]
        self.norm = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        self.decoder_pred = dense("decoder_pred", kernel_init=nn.initializers.zeros)
        if self.conv:
            self.final_layer = self._final_layer(name="final_layer")

    def _final_layer(self, **scope):
        return Conv(self.output_channels, (3, 3), padding="SAME", dtype=self.dtype,
                    precision=self.precision, **scope)

    def __call__(self, x, temb, textcontext=None, train: bool = False):
        _, H, W, _ = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        rows = self.pos_embed.shape[1] - 1 - self.text_tokens
        if num_patches > rows:
            raise ValueError(f"{num_patches} patches exceed the position table's {rows} "
                             f"(image_size {self.image_size})")
        patches, inverse = self.patch_embed.scanned(x, self.scan_order)

        wide = at_least_fp32(self.dtype)
        time = sinusoidal_time(jnp.asarray(temb, wide) * self.time_scale, self.emb_features, dtype=wide)
        if self.mlp_time_embed:
            time = self.time_mlp(time.astype(self.dtype))
        tokens = [time.astype(patches.dtype)[:, None]]
        positions = [self.pos_embed[:, :1]]
        if textcontext is not None:
            text = textcontext.hidden
            if text.shape[1] > self.text_tokens:
                raise ValueError(f"{text.shape[1]} text states exceed the position table's "
                                 f"{self.text_tokens}")
            tokens.append(self.context_embed(text.astype(self.dtype)).astype(patches.dtype))
            positions.append(self.pos_embed[:, 1:1 + text.shape[1]])
        tokens.append(patches)
        # Each patch's learned position follows it into the scan order (`scan_ordered`), as its token did.
        raster = self.pos_embed[:, 1 + self.text_tokens:1 + self.text_tokens + num_patches]
        positions.append(scan_ordered(raster, self.scan_order, H // self.patch_size, W // self.patch_size)[0])
        x = jnp.concatenate(tokens, axis=1) + jnp.concatenate(positions, axis=1).astype(patches.dtype)
        extras = x.shape[1] - num_patches

        x = u_walk(x, self.in_blocks, self.mid_block, self.skip_linear, self.out_blocks,
                   lambda block, x: block(x, conditioning=None, freqs_cis=None, train=train))

        out = self.decoder_pred(self.norm(x))[:, extras:]
        image = unscanned(out, inverse, self.patch_size, H, W, self.output_channels)
        return self.final_layer(image) if self.conv else image


class SimpleUDiT(RematLadder, _TransformerOptions):
    """A U-shaped DiT: `SimpleDiT`'s adaLN-Zero blocks with the first half's
    outputs skipping into the second half through a dense layer over the
    concatenation. Position comes from RoPE over the sequence index, so a
    hilbert scan carries the rotation of its curve index and no 2D signal.

    `adaln_silu` False and `text_pooling` "all" are FlaxDiff 0.2's U-DiT: its
    blocks project the conditioning vector without a SiLU and it averages the
    text over every position. `dew.interop.flaxdiff` loads its checkpoints
    with them.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.0
    remat: RematChoice = False
    scan_order: Literal["raster", "hilbert"] = "raster"
    adaln_silu: bool = True
    text_pooling: Literal["real", "all"] = "real"

    @nn.nowrap
    def projections(self, input_channels: int = 0) -> dict[str, tuple[int, int]]:
        return {'up_dense': (2 * self.emb_features, self.emb_features)}

    def _block(self, remat: RematChoice = False, **scope) -> ModulatedBlock:
        return remat_block(ModulatedBlock, remat)(features=self.emb_features, num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate, adaln_silu=self.adaln_silu,
                                                 **self._block_options(), **scope)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("SimpleUDiT requires image_flops(lengths, input_channels, text_features)")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def image_flops(self, lengths, input_channels: int, text_features: int | None = None) -> float:
        """Patch/time/pooled maps, modulated blocks and the up-path skip maps."""
        return (PatchEmbedding(self.patch_size, self.emb_features, parent=None).model_flops(
                    lengths, input_channels)
                + self._conditioning(self.emb_features, text_pooling=self.text_pooling).model_flops(
                    lengths, text_features)
                + (self.num_layers + 1) * self._block(parent=None).model_flops(lengths)
                + (self.num_layers // 2) * projection_flops(lengths, *self.projections()['up_dense'])
                + self._output(self.patch_size, self.output_channels).model_flops(lengths, self.emb_features))

    def setup(self):
        assert self.num_layers % 2 == 0, "num_layers must be even for U-Net structure"
        half_layers = self.num_layers // 2

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="patch_embed"
        )
        self.conditioning = self._conditioning(self.emb_features, text_pooling=self.text_pooling)

        block = partial(self._block, self.remat)
        self.down_blocks = [block(name=f"down_block_{i}") for i in range(half_layers)]
        self.mid_block = block(name="mid_block")
        self.up_dense = [
            DenseGeneral(
                features=self.projections()['up_dense'][1],
                dtype=self.dtype,
                precision=self.precision,
                name=f"up_dense_{i}", dot_general=float32_dot_general
            ) for i in range(half_layers)
        ]
        self.up_blocks = [block(name=f"up_block_{i}") for i in range(half_layers)]

        self.output = self._output(self.patch_size, self.output_channels)

    def __call__(self, x, temb, textcontext=None, train: bool = False):
        _, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            "Image dimensions must be divisible by patch size"
        )

        x_seq, hilbert_inv_idx = self.patch_embed.scanned(x, self.scan_order)

        cond_emb = self.conditioning(temb, textcontext)
        freqs_cis = rotary_freqs(jnp.arange(x_seq.shape[1]), self.emb_features // self.num_heads,
                                 ROPE_THETA, dtype=at_least_fp32(x_seq.dtype))

        x_seq = u_walk(x_seq, self.down_blocks, self.mid_block, self.up_dense, self.up_blocks,
                       lambda block, x: block(x, cond_emb, freqs_cis, train))

        return self.output(x_seq, hilbert_inv_idx, H, W)
