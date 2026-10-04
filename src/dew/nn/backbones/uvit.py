"""Denoise patches with the two U-shaped token transformers.

UViT takes the time and the text as tokens beside the patches. The U-DiT
takes them as adaLN-Zero modulation instead. Both skip each first-half
block's output into its mirror in the second half.
"""

from functools import partial
from typing import Literal

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.registry import models

from ..attention import LayerNorm
from ..conv import Conv
from ..dit import (
    ROPE_THETA,
    ConditioningEmbed,
    ModulatedBlock,
    PatchEmbedding,
    PatchSequenceOutput,
    RematChoice,
    remat_block,
)
from ..precision import at_least_fp32
from ..rope import rotary_freqs
from ..scan_orders import hilbert_patchify, hilbert_unpatchify, unpatchify
from .unet_condition import sinusoidal_time


@models("uvit")
class UViT(nn.Module):
    """Denoise patches as U-ViT does (Bao et al. 2023), baofff/U-ViT's
    libs/uvit_t2i.py.

    A time token and, given text, one token per text state lead the patches:
    [time, text..., patches]. The time token is the sinusoidal embedding
    [cos, sin] of the time times `time_scale` (U-ViT reads its timesteps on a
    0 to 999 scale, Dew's processes hand a time in [0, 1]), through a
    d -> 4d -> d SiLU MLP when `mlp_time_embed`; the text states are projected
    by a dense layer; the patches are a `patch_size` convolution. A learned
    position table covers every token: one row for the time, `text_tokens`
    for the text and the patches of an `image_size` image, used from the
    first row of each part. Every block is a plain pre-norm transformer
    block, x + attention(LN(x)) and then x + MLP(LN(x)): q, k and v without
    bias, the output projection with one, the MLP `mlp_ratio` wide on exact
    GELU. num_layers / 2 blocks go down, one sits in the middle and
    num_layers / 2 go up, each up block first projecting [x, skip] back to
    the width. A final LN and dense layer read the patches, which are
    unpatchified and, with `conv`, refined by a 3x3 convolution. The text
    attends unmasked, padding included, as U-ViT's CLIP states do. A Hilbert
    `scan_order` sequences the patches along the curve, each projected by a
    dense layer, and its position rows follow the sequence.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    mlp_time_embed: bool = False
    conv: bool = True
    time_scale: float = 999.0
    text_tokens: int = 77
    image_size: int = 512
    force_fp32_for_softmax: bool = True
    attention_impl: str = "auto"  # an AttentionImpl
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5
    scan_order: Literal["raster", "hilbert"] = "raster"

    def setup(self):
        if self.num_layers % 2:
            raise ValueError(f"UViT splits its layers into a down and an up half; {self.num_layers} "
                             "is odd")
        dense = partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        if self.scan_order == "hilbert":
            self.hilbert_proj = dense(self.emb_features, name="hilbert_projection")
        else:
            self.patch_embed = PatchEmbedding(patch_size=self.patch_size, embedding_dim=self.emb_features,
                                              dtype=self.dtype, precision=self.precision, name="patch_embed")
        patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = self.param("pos_embed", nn.initializers.truncated_normal(0.02),
                                    (1, 1 + self.text_tokens + patches, self.emb_features))
        if self.mlp_time_embed:
            self.time_mlp = [dense(4 * self.emb_features, name="time_embed_0"),
                             dense(self.emb_features, name="time_embed_2")]
        self.context_embed = dense(self.emb_features, name="context_embed")
        block = partial(ModulatedBlock, features=self.emb_features, num_heads=self.num_heads, modulated=False,
                        mlp_ratio=self.mlp_ratio, qkv_bias=False, gelu_approximate=False, dtype=self.dtype,
                        precision=self.precision, force_fp32_for_softmax=self.force_fp32_for_softmax,
                        norm_epsilon=self.norm_epsilon, attention_impl=self.attention_impl)
        half = self.num_layers // 2
        self.in_blocks = [block(name=f"in_blocks_{i}") for i in range(half)]
        self.mid_block = block(name="mid_block")
        self.skip_linear = [dense(self.emb_features, name=f"skip_linear_{i}") for i in range(half)]
        self.out_blocks = [block(name=f"out_blocks_{i}") for i in range(half)]
        self.norm = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        self.decoder_pred = dense(self.patch_size ** 2 * self.output_channels,
                                  kernel_init=nn.initializers.zeros, name="decoder_pred")
        if self.conv:
            self.final_layer = Conv(self.output_channels, (3, 3), padding="SAME", dtype=self.dtype,
                                    precision=self.precision, name="final_layer")

    def __call__(self, x, temb, textcontext=None, train: bool = False):
        _, H, W, _ = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        rows = self.pos_embed.shape[1] - 1 - self.text_tokens
        if num_patches > rows:
            raise ValueError(f"{num_patches} patches exceed the position table's {rows} "
                             f"(image_size {self.image_size})")
        inverse = None
        if self.scan_order == "hilbert":
            patches, inverse = hilbert_patchify(x, self.patch_size)
            patches = self.hilbert_proj(patches)
        else:
            patches = self.patch_embed(x)

        wide = at_least_fp32(self.dtype)
        time = sinusoidal_time(jnp.asarray(temb, wide) * self.time_scale, self.emb_features, dtype=wide)
        if self.mlp_time_embed:
            time = self.time_mlp[1](nn.silu(self.time_mlp[0](time.astype(self.dtype))))
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
        positions.append(self.pos_embed[:, 1 + self.text_tokens:1 + self.text_tokens + num_patches])
        x = jnp.concatenate(tokens, axis=1) + jnp.concatenate(positions, axis=1).astype(patches.dtype)
        extras = x.shape[1] - num_patches

        skips = []
        for block in self.in_blocks:
            x = block(x, conditioning=None, freqs_cis=None, train=train)
            skips.append(x)
        x = self.mid_block(x, conditioning=None, freqs_cis=None, train=train)
        for skip_linear, block in zip(self.skip_linear, self.out_blocks, strict=True):
            x = skip_linear(jnp.concatenate([x, skips.pop()], axis=-1))
            x = block(x, conditioning=None, freqs_cis=None, train=train)

        out = self.decoder_pred(self.norm(x))[:, extras:]
        if inverse is not None:
            image = hilbert_unpatchify(out, inverse, self.patch_size, H, W, self.output_channels)
        else:
            image = unpatchify(out, self.patch_size, H, W, self.output_channels)
        return self.final_layer(image) if self.conv else image


@models("simple_udit")
class SimpleUDiT(nn.Module):
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
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: str = "auto"  # an AttentionImpl
    remat: RematChoice = False
    norm_epsilon: float = 1e-5
    scan_order: Literal["raster", "hilbert"] = "raster"
    adaln_silu: bool = True
    text_pooling: Literal["real", "all"] = "real"

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
        if self.scan_order == "hilbert":
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )
        self.conditioning = ConditioningEmbed(
            emb_features=self.emb_features,
            mlp_ratio=self.mlp_ratio,
            dtype=self.dtype,
            precision=self.precision,
            text_pooling=self.text_pooling,
        )

        block = partial(
            remat_block(ModulatedBlock, self.remat),
            features=self.emb_features,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            norm_epsilon=self.norm_epsilon,
            adaln_silu=self.adaln_silu,
        )
        self.down_blocks = [block(name=f"down_block_{i}") for i in range(half_layers)]
        self.mid_block = block(name="mid_block")
        self.up_dense = [
            nn.DenseGeneral(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name=f"up_dense_{i}"
            ) for i in range(half_layers)
        ]
        self.up_blocks = [block(name=f"up_block_{i}") for i in range(half_layers)]

        self.output = PatchSequenceOutput(
            patch_size=self.patch_size,
            output_channels=self.output_channels,
            norm_epsilon=self.norm_epsilon,
            dtype=self.dtype,
            precision=self.precision,
        )

    def __call__(self, x, temb, textcontext=None, train: bool = False):
        _, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            "Image dimensions must be divisible by patch size"
        )

        hilbert_inv_idx = None
        if self.scan_order == "hilbert":
            patches_raw, hilbert_inv_idx = hilbert_patchify(x, self.patch_size)
            x_seq = self.hilbert_proj(patches_raw)
        else:
            x_seq = self.patch_embed(x)

        cond_emb = self.conditioning(temb, textcontext)
        freqs_cis = rotary_freqs(jnp.arange(x_seq.shape[1]), self.emb_features // self.num_heads,
                                 ROPE_THETA, dtype=at_least_fp32(x_seq.dtype))

        skips = []
        for i in range(self.num_layers // 2):
            x_seq = self.down_blocks[i](x_seq, cond_emb, freqs_cis, train)
            skips.append(x_seq)

        x_seq = self.mid_block(x_seq, cond_emb, freqs_cis, train)

        for i in range(self.num_layers // 2):
            skip_conn = skips.pop()
            x_seq = jnp.concatenate([x_seq, skip_conn], axis=-1)
            x_seq = self.up_dense[i](x_seq)
            x_seq = self.up_blocks[i](x_seq, cond_emb, freqs_cis, train)

        return self.output(x_seq, hilbert_inv_idx, H, W)
