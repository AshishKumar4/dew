"""MM-DiT, the SD3-style multimodal DiT, and a hierarchical variant.

The block is the dual-stream block SD3 and Flux run (`joint.DoubleStreamBlock`):
text and image tokens keep separate qkv/mlp/modulation weights and mix
through a single joint attention over the concatenated sequence.
"""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense

from ..dit import (
    ROPE_THETA,
    RematChoice,
    RematLadder,
    _AttentionStackOptions,
    _DiTStackOptions,
    remat_block,
    rope_for_scan,
)
from ..flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from ..norms import LayerNorm
from ..precision import at_least_fp32, float32_dot_general
from ..rope import rotary_freqs
from ..scan_orders import patchify, unpatchify
from .joint import DoubleStreamBlock

PATCH_SIDE = 2


def _block(stack: _AttentionStackOptions, remat: RematChoice, features: int, heads: int,
           name: str | None = None) -> DoubleStreamBlock:
    """One dual-stream block: `DoubleStreamBlock`, the block SD3 and Flux run,
    with the text first, adaLN-Zero modulation, the stack's layer norms and
    its queries' and keys' at 1e-6, rotate-half pairs, and its dropout."""
    return remat_block(DoubleStreamBlock, remat)(
        features, heads, features // heads, context_first=True, qk_norm=stack.qk_norm,
        epsilon=stack.norm_epsilon, qk_epsilon=1e-6, mlp_hidden=int(features * stack.mlp_ratio),
        zero_modulation=True, rotary_pairs="half", dropout_rate=stack.dropout_rate, dtype=stack.dtype,
        precision=stack.precision, attention_impl=stack.attention_impl,
        force_fp32_for_softmax=stack.force_fp32_for_softmax, name=name)


class SimpleMMDiT(_DiTStackOptions):
    """SD3-style MM-DiT: a plain stack of dual-stream blocks, the image tokens
    rotated by their raster index (`rope_for_scan`)."""

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("SimpleMMDiT requires multimodal_flops with image/text input widths")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def multimodal_flops(self, lengths, context_lengths, input_channels: int, text_features: int) -> float:
        """All input/output maps, pooled/time conditioning and joint blocks."""
        matching_lengths(lengths, context_lengths)
        width = self.emb_features
        count = (self._embedding(self.patch_size, width, self.scan_order).model_flops(lengths, input_channels)
                 + projection_flops(context_lengths, *self.projections(text_features)['txt_embed'])
                 + self._conditioning(width).model_flops(lengths, text_features))
        block = _block(self, remat=False, features=width, heads=self.num_heads)
        return (count + self.num_layers * block.multisequence_flops(lengths, context_lengths)
                + self._output(self.patch_size, self.output_channels, modulated=True).model_flops(
                    lengths, width))

    @nn.nowrap
    def projections(self, text_features: int = 0) -> dict[str, tuple[int, int]]:
        return {'txt_embed': (text_features, self.emb_features)}

    def setup(self):
        self.embed = self._embedding(self.patch_size, self.emb_features, self.scan_order)
        self.conditioning = self._conditioning(self.emb_features)
        # text tokens enter the sequence, so they need their own projection
        self.txt_embed = Dense(
            features=self.projections()['txt_embed'][1], dtype=self.dtype,
            precision=self.precision, name="txt_embed", dot_general=float32_dot_general)
        self.blocks = [_block(self, self.remat, self.emb_features, self.num_heads, f"mmdit_block_{i}")
                       for i in range(self.num_layers)]
        self.output = self._output(self.patch_size, self.output_channels, modulated=True)

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `textcontext`, which runs as a second stream through every block."""
        return "textcontext"

    def __call__(self, x, temb, textcontext, train: bool = False):
        _, H, W, _ = x.shape

        img, inv_idx = self.embed(x)
        txt = self.txt_embed(textcontext.hidden)
        cond_emb = self.conditioning(temb, textcontext)
        rotation = rope_for_scan(img, self.emb_features // self.num_heads, self.scan_order)

        for block in self.blocks:
            img, txt = block(img, txt, cond_emb, rotation, train)

        return self.output(img, inv_idx, H, W, conditioning=cond_emb)


class PatchMerging(nn.Module):
    """Merges each 2x2 group of patches into one token of `out_features`,
    halving the grid on the way down: a Swin-style layer norm over the
    concatenated group, then a projection."""
    out_features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        return projection_flops(lengths, PATCH_SIDE**2 * input_features, self.out_features)

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        B, _, C = x.shape
        merged = patchify(x.reshape(B, H_patches, W_patches, C), PATCH_SIDE)
        merged = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")(merged)
        merged = Dense(
            features=self.out_features,
            dtype=self.dtype,
            precision=self.precision,
            name="projection", dot_general=float32_dot_general
        )(merged)
        return merged, H_patches // PATCH_SIDE, W_patches // PATCH_SIDE

class PatchExpanding(nn.Module):
    """Expands each token into a 2x2 group of `out_features` tokens, doubling
    the grid on the way up: a projection to the group's width, a layer norm,
    then the rearrangement."""
    out_features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5

    @property
    def expanded_features(self) -> int:
        return PATCH_SIDE**2 * self.out_features

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        return projection_flops(lengths, input_features, self.expanded_features)

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        B = x.shape[0]
        x = Dense(
            features=self.expanded_features,
            dtype=self.dtype,
            precision=self.precision,
            name="projection", dot_general=float32_dot_general
        )(x)
        x = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")(x)
        new_H, new_W = H_patches * PATCH_SIDE, W_patches * PATCH_SIDE
        expanded = unpatchify(x, PATCH_SIDE, new_H, new_W, self.out_features)
        return expanded.reshape(B, new_H * new_W, self.out_features), new_H, new_W


class HierarchicalMMDiT(RematLadder, _AttentionStackOptions):
    """U-shaped MM-DiT: dual-stream blocks per stage with patch merging on the
    way down and expansion + skip fusion on the way up.

    Raster order only: the merge/expand grid reshapes assume row-major token
    order, so a hilbert scan would scramble the neighborhoods being merged.
    """
    output_channels: int = 3
    base_patch_size: int = 8  # Patch size at the *finest* resolution level (stage 0)
    emb_features: Sequence[int] = (512, 768, 1024)  # Feature dims for stages, fine to coarse
    num_layers: Sequence[int] = (4, 4, 14)  # Layers per stage, fine to coarse
    num_heads: Sequence[int] = (8, 12, 16)  # Heads per stage, fine to coarse
    remat: RematChoice = False

    @nn.nowrap
    def projections(self, text_features: int = 0) -> dict[str, tuple[int, int]]:
        return {name: shape for stage, width in enumerate(self.emb_features)
                for name, shape in ((f'cond_proj_stage{stage}', (self.emb_features[0], width)),
                                    (f'txt_embed_stage{stage}', (text_features, width)),
                                    (f'fusion_{stage}', (2 * width, width)))}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("HierarchicalMMDiT requires multimodal_flops with image/text input widths")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        coarsest = self.base_patch_size * PATCH_SIDE**(len(self.emb_features) - 1)
        patch_lengths(sample_shape, (coarsest, coarsest), rows)
        return patch_lengths(sample_shape, (self.base_patch_size, self.base_patch_size), rows)

    @nn.nowrap
    def multimodal_flops(self, lengths, context_lengths, input_channels: int, text_features: int) -> float:
        """Per-stage streams, patch merges/expansions, skip maps and output."""
        image, _ = matching_lengths(lengths, context_lengths)
        rows, base = np.ones(image.shape, np.int64), self.emb_features[0]
        if np.any(image % PATCH_SIDE**(2 * (len(self.emb_features) - 1))):
            raise MissingFlopsCount("HierarchicalMMDiT needs whole patches at its coarsest stage")
        count = (self._embedding(self.base_patch_size, base).model_flops(lengths, input_channels)
                 + self._conditioning(base).model_flops(lengths, text_features))
        maps = self.projections(text_features)
        for stage, (width, layers, heads) in enumerate(zip(
                self.emb_features, self.num_layers, self.num_heads, strict=True)):
            level = (image / PATCH_SIDE**(2 * stage)).astype(np.int64)
            count += (projection_flops(rows, *maps[f'cond_proj_stage{stage}'])
                      + projection_flops(context_lengths, *maps[f'txt_embed_stage{stage}']))
            block = _block(self, remat=False, features=width, heads=heads)
            count += (1 if stage == len(self.emb_features) - 1 else 2) * layers * block.multisequence_flops(
                level, context_lengths)
            if stage < len(self.emb_features) - 1:
                below = self.emb_features[stage + 1]
                count += PatchMerging(below, parent=None).model_flops(level // PATCH_SIDE**2, width)
                count += PatchExpanding(width, parent=None).model_flops(level // PATCH_SIDE**2, below)
                count += projection_flops(level, *maps[f'fusion_{stage}'])
        return count + self._output(self.base_patch_size, self.output_channels, modulated=True).model_flops(
            lengths, base)

    def stage_blocks(self, stage: int, prefix: str) -> list:
        """Build one stage's dual-stream blocks, at that stage's width and heads."""
        return [_block(self, self.remat, self.emb_features[stage], self.num_heads[stage],
                       f"{prefix}_block_stage{stage}_{i}")
                for i in range(self.num_layers[stage])]

    def encoder_path(self, num_stages: int):
        """Build the encoder, fine to coarse: each stage's blocks and its merger.

        Sets `encoder_blocks`, one list per stage, and `patch_mergers`, one
        between each pair of stages.
        """
        self.encoder_blocks = [self.stage_blocks(s, "encoder") for s in range(num_stages)]
        self.patch_mergers = [
            PatchMerging(
                out_features=self.emb_features[s + 1],
                dtype=self.dtype,
                precision=self.precision,
                norm_epsilon=self.norm_epsilon,
                name=f"patch_merger_{s}"
            ) for s in range(num_stages - 1)
        ]

    def decoder_path(self, num_stages: int):
        """Build the decoder, coarse to fine, for stages N-2 down to 0.

        Sets `patch_expanders`, the `fusion_layers` that take an expanded
        stage beside its skip, and `decoder_blocks`. All three are indexed
        by decoder step, not by stage.
        """
        decoder_stages = list(range(num_stages - 2, -1, -1))
        self.patch_expanders = [
            PatchExpanding(
                out_features=self.emb_features[s],
                dtype=self.dtype,
                precision=self.precision,
                norm_epsilon=self.norm_epsilon,
                name=f"patch_expander_{s}"
            ) for s in decoder_stages
        ]
        self.fusion_layers = [
            nn.Sequential([
                LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype),
                Dense(features=self.projections()[f'fusion_{s}'][1], dtype=self.dtype,
                      precision=self.precision, dot_general=float32_dot_general),
            ], name=f"fusion_{s}") for s in decoder_stages
        ]
        self.decoder_blocks = [self.stage_blocks(s, "decoder") for s in decoder_stages]

    def setup(self):
        """Build the patch embedding, the two paths of stages, and the head.

        `cond_projs` and `txt_embeds` carry the conditioning and the text
        into each stage's own width. `encoder_path` and `decoder_path` set
        the blocks and the resampling layers between them.
        """
        assert len(self.emb_features) == len(self.num_layers) == len(self.num_heads), \
            "Feature dimensions, layers, and heads must have the same number of stages"
        num_stages = len(self.emb_features)

        self.embed = self._embedding(self.base_patch_size, self.emb_features[0])
        # Base conditioning at the finest dim, projected per stage
        self.conditioning = self._conditioning(self.emb_features[0])
        self.cond_projs = [
            Dense(features=self.projections()[f'cond_proj_stage{i}'][1], dtype=self.dtype,
                  precision=self.precision, name=f"cond_proj_stage{i}", dot_general=float32_dot_general)
            for i in range(num_stages)
        ]
        # Per-stage text streams (dims differ per stage)
        self.txt_embeds = [
            Dense(features=self.projections()[f'txt_embed_stage{i}'][1], dtype=self.dtype,
                  precision=self.precision, name=f"txt_embed_stage{i}", dot_general=float32_dot_general)
            for i in range(num_stages)
        ]

        self.encoder_path(num_stages)
        self.decoder_path(num_stages)

        self.output = self._output(self.base_patch_size, self.output_channels, modulated=True)

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `textcontext`, which runs as a second stream through every block."""
        return "textcontext"

    def __call__(self, x, temb, textcontext, train: bool = False):
        _, H, W, _ = x.shape
        num_stages = len(self.emb_features)
        assert (
            H % (self.base_patch_size * (PATCH_SIDE ** (num_stages - 1))) == 0
            and W % (self.base_patch_size * (PATCH_SIDE ** (num_stages - 1))) == 0
        ), (
            f"Image dimensions ({H},{W}) must be divisible by effective coarsest patch size "
            f"{self.base_patch_size * (PATCH_SIDE ** (num_stages - 1))}"
        )

        img, _ = self.embed(x)
        cond_base = self.conditioning(temb, textcontext)
        conds = [proj(cond_base) for proj in self.cond_projs]
        txts = [embed(textcontext.hidden) for embed in self.txt_embeds]

        # --- Encoder path ---
        H_P, W_P = H // self.base_patch_size, W // self.base_patch_size
        skips = {}
        for stage in range(num_stages):
            txt = txts[stage]
            rotation = rotary_freqs(
                jnp.arange(img.shape[1]), self.emb_features[stage] // self.num_heads[stage],
                ROPE_THETA, dtype=at_least_fp32(img.dtype))
            for block in self.encoder_blocks[stage]:
                img, txt = block(img, txt, conds[stage], rotation, train)
            skips[stage] = img
            if stage < num_stages - 1:
                img, H_P, W_P = self.patch_mergers[stage](img, H_P, W_P)

        # --- Decoder path ---
        for i, stage in enumerate(range(num_stages - 2, -1, -1)):
            img, H_P, W_P = self.patch_expanders[i](img, H_P, W_P)
            img = self.fusion_layers[i](jnp.concatenate([img, skips[stage]], axis=-1))
            txt = txts[stage]
            rotation = rotary_freqs(
                jnp.arange(img.shape[1]), self.emb_features[stage] // self.num_heads[stage],
                ROPE_THETA, dtype=at_least_fp32(img.dtype))
            for block in self.decoder_blocks[i]:
                img, txt = block(img, txt, conds[stage], rotation, train)

        return self.output(img, None, H, W, conditioning=conds[0])
