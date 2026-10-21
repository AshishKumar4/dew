"""JEPA encoders and predictors, built from the shared DiT blocks.

A JEPA encoder is the DiT structure without the diffusion parts: patchify with
the 2D sincos signal, run ModulatedBlocks over the tokens, then normalize.
There is no timestep to condition on, so the blocks run in their unmodulated
(plain pre-norm) mode. The mixer is still configurable, and mixer patterns
with 'ssm' give a linear-time S5 encoder.

Position never comes from RoPE here. Both the encoder and the predictor work
on a masked subset of the sequence, where a token's index in the sequence is
not its position on the grid, so the spatial blocks run unrotated and the 2D
sincos embedding that stays with each token carries all the position
information.

The image encoder and predictor compute what V-JEPA's do (facebookresearch/jepa,
image mode, mask-token predictor), its exact (erf) GELU in every MLP
included (`ModulatedBlock.gelu_approximate=False`, torch's 1 + erf form).
On an A100, a training step of a ViT-S/16 encoder at 224 with a 6-layer
predictor, bf16, batch 64, took 26.68/26.72 ms with it against 27.86 with
the tanh GELU and 29.57/29.59 with jax's erfc form (ABAB, 50 steps after 5
of warmup, equal peak memory; Colab job dew-gpu-c20-gelu-a100-job-1).
"""

# tests/test_jepa_source.py checks these modules against V-JEPA's own code.

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.nn.dense import Dense

from ..attention import LayerNorm
from ..dit import (
    ModulatedBlock,
    _JepaStackOptions,
    build_block_pattern,
    factorized_walk,
    scan_ordered_pos_embed,
)
from ..flops import MissingFlopsCount, counted_flops, patch_lengths, projection_flops, real_lengths
from ..scan_orders import ScanOrder
from ..sharding import constrain, down_projection, rows_like, rows_taken
from .dit import gather_tokens


class TokenStack(_JepaStackOptions):
    """A stack of unmodulated blocks over a token sequence."""
    features: int
    num_layers: int

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The stack's declared pattern, with real sequence lengths for each block."""
        return sum(counted_flops(block, lengths) for block in self._blocks())

    def setup(self):
        self.blocks = self._blocks()

    @nn.nowrap
    def _blocks(self) -> list[ModulatedBlock]:
        pattern = build_block_pattern(self.num_layers, self.ssm_attention_ratio)
        return [
            ModulatedBlock(
                features=self.features,
                num_heads=self.num_heads,
                mixer='ssm' if kind == 'ssm' else 'attention',
                modulated=False,
                gelu_approximate=False,
                **self._block_options(),
                ssm_state_dim=self.ssm_state_dim,
                bidirectional_ssm=self.bidirectional_ssm,
                name=f"block_{i}",
            ) for i, kind in enumerate(pattern)
        ]

    def __call__(self, tokens, freqs_cis=None, train: bool = False):
        for block in self.blocks:
            tokens = block(tokens, conditioning=None, freqs_cis=freqs_cis, train=train)
        return tokens


class FactorizedTokenStack(_JepaStackOptions):
    """Spatial then temporal blocks over [B, T, N, F], as in VideoDiT.

    Time is a real 1D axis that masking never touches, so the temporal half
    is rotated by frame index while the spatial half runs unrotated.
    """
    features: int
    num_layers: int

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("FactorizedTokenStack requires spatiotemporal_flops(frames, patches)")

    @nn.nowrap
    def spatiotemporal_flops(self, frames, patches) -> float:
        """Spatial rows per frame and temporal rows per spatial patch."""
        real_lengths(frames)
        real_lengths(patches)
        if np.shape(frames) != np.shape(patches):
            raise ValueError("factorized FLOPs need matching frame/patch row shapes")
        options = self._stack_options()
        spatial = TokenStack(self.features, self.num_layers, **options, parent=None)
        temporal = TokenStack(self.features, self.num_layers,
                              **{**options, 'ssm_attention_ratio': 'all-attn'}, parent=None)
        return (spatial.model_flops(np.repeat(patches, frames))
                + temporal.model_flops(np.repeat(frames, patches)))

    def setup(self):
        def stack(name, num_layers, ratio):
            options = self._stack_options()
            options['ssm_attention_ratio'] = ratio
            return TokenStack(
                features=self.features, num_layers=num_layers, **options, name=name)

        # one spatial and one temporal block per layer, built as single-block
        # stacks so the two halves can be interleaved
        pattern = build_block_pattern(self.num_layers, self.ssm_attention_ratio)
        self.spatial = [stack(f"spatial_{i}", 1, "all-ssm" if kind == 'ssm' else "all-attn")
                        for i, kind in enumerate(pattern)]
        self.temporal = [stack(f"temporal_{i}", 1, "all-attn") for i in range(self.num_layers)]

    def __call__(self, tokens, train: bool = False):
        B, T, N, F = tokens.shape
        tokens = factorized_walk(tokens.reshape(B * T, N, F), T, self.features // self.num_heads, [
            (lambda x, stack=spatial: stack(x, train=train),
             lambda x, rotation, stack=temporal: stack(x, freqs_cis=rotation, train=train))
            for spatial, temporal in zip(self.spatial, self.temporal, strict=True)])
        return tokens.reshape(B, T, N, F)


class JepaEncoder(_JepaStackOptions):
    """ViT over an image, optionally restricted to a subset of its patches."""
    patch_size: int = 16
    emb_features: int = 384
    num_layers: int = 12
    num_heads: int = 6
    scan_order: ScanOrder = "raster"

    stack_type: ClassVar[type[TokenStack | FactorizedTokenStack]] = TokenStack
    """The layers between the patches and the norm.

    `TokenStack` runs over one image's tokens and `FactorizedTokenStack` over a
    clip's frames.
    """

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("JepaEncoder requires image_flops(lengths, input_channels)")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        """Image patch lengths; a video needs its frame/patch geometry separately."""
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def image_flops(self, lengths, input_channels: int, text_features: int | None = None) -> float:
        """The patch projection plus selected tokens' transformer stack."""
        if text_features is not None:
            raise MissingFlopsCount("JepaEncoder has no text conditioning projection")
        stack = self.stack_type(self.emb_features, self.num_layers, **self._stack_options(), parent=None)
        return (self._embedding(self.patch_size, self.emb_features).model_flops(lengths, input_channels)
                + counted_flops(stack, lengths))

    def setup(self):
        self.embed = self._embedding(self.patch_size, self.emb_features, self.scan_order)
        self.stack = self.stack_type(
            features=self.emb_features, num_layers=self.num_layers,
            **self._stack_options(),
        )
        self.norm = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")

    def __call__(self, x, token_idx=None, train: bool = False):
        tokens, _ = self.embed(x)
        if token_idx is not None:
            tokens = gather_tokens(tokens, token_idx)
        return self.norm(self.stack(tokens, train=train))

    def hidden_states(self, x, *, train: bool = False, token_idx=None):
        """Return the encoder's representation of `x`, the normed tokens its
        call gives, of the patches `token_idx` selects when given."""
        return self(x, token_idx, train=train)


class JepaVideoEncoder(JepaEncoder):
    """Factorized spatial-temporal encoder over (B, T, H, W, C).

    token_idx selects a tubelet: the same patch positions in every frame, so
    the factorized layout survives masking untouched.
    """

    stack_type = FactorizedTokenStack

    @nn.nowrap
    def spatiotemporal_flops(self, frames, patches, input_channels: int) -> float:
        """Patch maps and factorized work, not flattened into one attention square."""
        stack = FactorizedTokenStack(self.emb_features, self.num_layers, **self._stack_options(), parent=None)
        lengths = (real_lengths(frames) * real_lengths(patches)).astype(np.int64)
        return (projection_flops(lengths, self.patch_size**2 * input_channels, self.emb_features)
                + stack.spatiotemporal_flops(frames, patches))

    def __call__(self, x, token_idx=None, train: bool = False):
        B, T, H, W, C = x.shape
        tokens, _ = self.embed(x.reshape(B * T, H, W, C))
        if token_idx is not None:
            tokens = gather_tokens(tokens, jnp.repeat(token_idx, T, axis=0))
        tokens = tokens.reshape(B, T, tokens.shape[1], self.emb_features)
        return self.norm(self.stack(tokens, train=train))


class JepaPredictor(_JepaStackOptions):
    """A narrow transformer that maps context embeddings to target embeddings.

    Context tokens are projected down, mask tokens stand in for the targets,
    and both get the sincos signal for the grid position they belong to.

    The projections in and out split no tensor width, so under a tensor axis
    each runs where `down_projection` places it. When the measured link is
    fast enough to gather the results, each tensor shard projects only its
    own tokens, as multi-head latent attention's down-projections do.
    """
    # Run on every token of every shard, the two projections made
    # tools/layout_parity's JEPA compute 1.11 times one device's FLOPs on tensor4.
    grid: tuple[int, int] = (14, 14)
    emb_features: int = 384      # encoder width, in and out
    predictor_features: int = 192
    num_layers: int = 6
    num_heads: int = 6
    scan_order: str = 'raster'
    factorized: bool = False     # space-time blocks, for video

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("JepaPredictor requires multisequence_flops(context_lengths, target_lengths)")

    @nn.nowrap
    def multisequence_flops(self, context_lengths, target_lengths) -> float:
        """Projected context plus target tokens, followed by the target-only head."""
        if self.factorized:
            raise MissingFlopsCount("JepaPredictor needs frames and spatial patch counts for factorized work")
        context, target = real_lengths(context_lengths), real_lengths(target_lengths)
        if context.shape != target.shape:
            raise ValueError("JEPA FLOPs need matching context/target row shapes")
        stack = TokenStack(self.predictor_features, self.num_layers, **self._stack_options(), parent=None)
        return (stack.model_flops((context + target).astype(np.int64))
                + projection_flops(context_lengths, self.emb_features, self.predictor_features)
                + projection_flops(target_lengths, self.predictor_features, self.emb_features))

    def setup(self):
        self.proj_in = Dense(features=self.predictor_features, dtype=self.dtype,
                             precision=self.precision, name="proj_in")
        self.mask_token = self.param(
            "mask_token", nn.initializers.normal(0.02), (1, 1, self.predictor_features))
        stack = FactorizedTokenStack if self.factorized else TokenStack
        self.stack = stack(
            features=self.predictor_features, num_layers=self.num_layers,
            **self._stack_options(),
        )
        self.norm = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        self.proj_out = Dense(features=self.emb_features, dtype=self.dtype,
                              precision=self.precision, name="proj_out")

    def __call__(self, context, context_idx, target_idx, train: bool = False):
        """Map context embeddings [B, (T,) N_ctx, F] to predictions [B, (T,) N_tgt, F]."""
        pos_embed = jnp.asarray(
            scan_ordered_pos_embed(self.predictor_features, *self.grid, self.scan_order),
            dtype=self.dtype or jnp.float32)

        def positions(indices):
            # Each row's own indices read the replicated table where the row is placed.
            pos = rows_taken(pos_embed, indices)           # [B, N, P]
            return pos[:, None] if self.factorized else pos

        num_target_tokens = target_idx.shape[-1]
        context = _projected(self.proj_in, context) + positions(context_idx)
        targets = self.mask_token + positions(target_idx)
        if self.factorized:
            targets = jnp.broadcast_to(
                targets, (*context.shape[:-2], num_target_tokens, self.predictor_features))

        tokens = jnp.concatenate([context, rows_like(targets, context)], axis=-2)
        tokens = self.stack(tokens, train=train)
        return _projected(self.proj_out, self.norm(tokens[..., -num_target_tokens:, :]))


def _projected(dense: Dense, x: jax.Array) -> jax.Array:
    """`dense(x)` over `[batch, ..., tokens, width]`, run where
    `down_projection` places a projection of `x` to `dense.features`: its
    positions spread over the tensor axis, or whole on every shard."""
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    place = down_projection(flat, dense.features)
    return constrain(dense(constrain(flat, place)), place).reshape(*x.shape[:-1], dense.features)
