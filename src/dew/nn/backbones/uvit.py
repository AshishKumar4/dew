"""The two U-shaped token transformers: UViT, whose blocks take the time and
the text as tokens, and the U-DiT, whose blocks are adaLN-Zero modulated."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional, Literal
from flax.typing import Dtype, PrecisionLike
from functools import partial

from ..attention import TransformerBlock, rotary_freqs
from ..blocks import FourierEmbedding, TimeProjection
from ..scan_orders import hilbert_patchify, hilbert_unpatchify, unpatchify
from ..dit import (
    ROPE_THETA, ConditioningEmbed, PatchEmbedding, PatchSequenceOutput, ModulatedBlock,
    remat_block,
)
from dew.registry import models
from ..sharding import logical_axes


@models("uvit")
@logical_axes({}, heuristic=(("text_proj",), ("up_dense_*",), ("pos_encoding",), ("final_*conv*",)))
class UViT(nn.Module):
    """U-ViT (Bao et al. 2023): the time embedding and the text are tokens
    beside the patches, the blocks are plain transformer blocks, and the
    first half's outputs skip into the second half through a dense layer
    over the concatenation. Position is a learned table over the raster
    index, sized for a 512 pixel image.

    `add_residualblock_output` refines the unpatchified prediction with two
    convolutions over it and the input image.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    use_projection: bool = False
    use_self_and_cross: bool = False
    force_fp32_for_softmax: bool = True
    attention_impl: Optional[str] = None
    activation: Callable = jax.nn.swish
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    add_residualblock_output: bool = False
    norm_inputs: bool = False
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-5
    scan_order: Literal["raster", "hilbert"] = "raster"

    def setup(self):
        assert self.num_layers % 2 == 0, "num_layers must be even for U-Net structure"
        half_layers = self.num_layers // 2
        norm = partial(nn.LayerNorm, epsilon=self.norm_epsilon, dtype=self.dtype)

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

        max_patches = (512 // self.patch_size)**2
        self.pos_encoding = self.param('pos_encoding',
                                       jax.nn.initializers.normal(stddev=0.02),
                                       (1, max_patches, self.emb_features))

        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features)
        ], name="time_embed")

        self.text_proj = nn.DenseGeneral(
            features=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="text_proj"
        )

        block = partial(
            TransformerBlock,
            heads=self.num_heads,
            dim_head=self.emb_features // self.num_heads,
            dtype=self.dtype, precision=self.precision, use_projection=self.use_projection,
            use_self_and_cross=self.use_self_and_cross,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            only_pure_attention=False, norm_inputs=self.norm_inputs,
            explicitly_add_residual=self.explicitly_add_residual,
            norm_epsilon=self.norm_epsilon,
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

        self.final_norm = norm(name="final_norm")
        self.final_proj = nn.Dense(
            features=self.patch_size ** 2 * self.output_channels,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="final_proj"
        )

        if self.add_residualblock_output:
            self.final_conv1 = nn.Conv(
                features=64, kernel_size=(3, 3), strides=(1, 1),
                dtype=self.dtype, precision=self.precision, name="final_conv1"
            )
            self.final_norm_conv = norm(name="final_norm_conv")
            self.final_conv2 = nn.Conv(
                features=self.output_channels, kernel_size=(3, 3), strides=(1, 1),
                dtype=jnp.float32,
                precision=self.precision, name="final_conv2"
            )

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        original_img = x
        B, H, W, C = original_img.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        hilbert_inv_idx = None
        if self.scan_order == "hilbert":
            patches_raw, hilbert_inv_idx = hilbert_patchify(x, self.patch_size)
            x_patches = self.hilbert_proj(patches_raw)
        else:
            x_patches = self.patch_embed(x)

        assert num_patches <= self.pos_encoding.shape[
            1], f"Number of patches {num_patches} exceeds max_len {self.pos_encoding.shape[1]} in positional encoding"
        x_patches = x_patches + self.pos_encoding[:, :num_patches, :]

        time_token = self.time_embed(temb.astype(jnp.float32))
        time_token = jnp.expand_dims(time_token.astype(self.dtype), axis=1)

        if textcontext is not None:
            text_tokens = self.text_proj(textcontext.hidden.astype(self.dtype))
            x = jnp.concatenate([x_patches, time_token, text_tokens], axis=1)
        else:
            x = jnp.concatenate([x_patches, time_token], axis=1)

        skips = []
        for i in range(self.num_layers // 2):
            x = self.down_blocks[i](x)
            skips.append(x)

        x = self.mid_block(x)

        for i in range(self.num_layers // 2):
            skip_conn = skips.pop()
            x = jnp.concatenate([x, skip_conn], axis=-1)
            x = self.up_dense[i](x)
            x = self.up_blocks[i](x)

        x_patches_out = self.final_proj(self.final_norm(x)[:, :num_patches, :])

        if self.scan_order == "hilbert":
            assert hilbert_inv_idx is not None, "Hilbert inverse index missing"
            x_image = hilbert_unpatchify(
                x_patches_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        else:
            x_image = unpatchify(x_patches_out, self.patch_size, H, W, self.output_channels)

        if self.add_residualblock_output:
            x_image = jnp.concatenate(
                [original_img.astype(self.dtype), x_image], axis=-1)

            x_image = self.final_conv1(x_image)
            x_image = self.final_norm_conv(x_image)
            x_image = self.activation(x_image)
            x_image = self.final_conv2(x_image)

        return x_image


@models("simple_udit")
@logical_axes({}, heuristic=(("up_dense_*",),))
class SimpleUDiT(nn.Module):
    """A U-shaped DiT: `SimpleDiT`'s adaLN-Zero blocks with the first half's
    outputs skipping into the second half through a dense layer over the
    concatenation. Position comes from RoPE over the sequence index, so a
    hilbert scan carries the rotation of its curve index and no 2D signal.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: Optional[str] = None
    remat: bool = False
    norm_epsilon: float = 1e-5
    scan_order: Literal["raster", "hilbert"] = "raster"

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

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        hilbert_inv_idx = None
        if self.scan_order == "hilbert":
            patches_raw, hilbert_inv_idx = hilbert_patchify(x, self.patch_size)
            x_seq = self.hilbert_proj(patches_raw)
        else:
            x_seq = self.patch_embed(x)

        cond_emb = self.conditioning(temb, textcontext)
        freqs_cis = rotary_freqs(jnp.arange(x_seq.shape[1]), self.emb_features // self.num_heads,
                                 ROPE_THETA)

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
