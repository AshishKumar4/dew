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
image mode, mask-token predictor) with one difference: their MLPs use the
tanh GELU where V-JEPA's use exact (erf) GELU. Exact GELU
(`ModulatedBlock.gelu_approximate=False`) made a training step 3.2% slower
on an RTX 4080, and Dew loads no published I-JEPA or V-JEPA weights that
would need it.
"""

# tests/test_jepa_source.py checks these modules against V-JEPA's own code with
# the GELU swapped. The 3.2% was a ViT-S/16 encoder at 224 with a 6-layer
# predictor, bf16, batch 64: 36.35-36.51 ms against 35.24-35.30.

from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.registry import models

from ..attention import LayerNorm
from ..dit import ROPE_THETA, ModulatedBlock, _JepaStackOptions, build_block_pattern, scan_ordered_pos_embed
from ..precision import at_least_fp32
from ..rope import rotary_freqs
from ..sharding import constrain, down_projection
from .dit import gather_tokens


class TokenStack(_JepaStackOptions):
    """A stack of unmodulated blocks over a token sequence."""
    features: int
    num_layers: int

    def setup(self):
        pattern = build_block_pattern(self.num_layers, self.ssm_attention_ratio)
        self.blocks = [
            ModulatedBlock(
                features=self.features,
                num_heads=self.num_heads,
                mixer='ssm' if kind == 'ssm' else 'attention',
                modulated=False,
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
        freqs_temporal = rotary_freqs(jnp.arange(T), self.features // self.num_heads, ROPE_THETA,
                                      dtype=at_least_fp32(tokens.dtype))

        tokens = tokens.reshape(B * T, N, F)
        for spatial, temporal in zip(self.spatial, self.temporal, strict=True):
            tokens = spatial(tokens, train=train)
            tokens = tokens.reshape(B, T, N, F).transpose(0, 2, 1, 3).reshape(B * N, T, F)
            tokens = temporal(tokens, freqs_cis=freqs_temporal, train=train)
            tokens = tokens.reshape(B, N, T, F).transpose(0, 2, 1, 3).reshape(B * T, N, F)
        return tokens.reshape(B, T, N, F)


@models("jepa_encoder")
class JepaEncoder(_JepaStackOptions):
    """ViT over an image, optionally restricted to a subset of its patches."""
    patch_size: int = 16
    emb_features: int = 384
    num_layers: int = 12
    num_heads: int = 6
    scan_order: Literal["raster", "hilbert", "zigzag"] = "raster"

    stack_type: ClassVar[type[TokenStack | FactorizedTokenStack]] = TokenStack
    """The layers between the patches and the norm.

    `TokenStack` runs over one image's tokens and `FactorizedTokenStack` over a
    clip's frames.
    """

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


@models("jepa_video_encoder")
class JepaVideoEncoder(JepaEncoder):
    """Factorized spatial-temporal encoder over (B, T, H, W, C).

    token_idx selects a tubelet: the same patch positions in every frame, so
    the factorized layout survives masking untouched.
    """

    stack_type = FactorizedTokenStack

    def __call__(self, x, token_idx=None, train: bool = False):
        B, T, H, W, C = x.shape
        tokens, _ = self.embed(x.reshape(B * T, H, W, C))
        if token_idx is not None:
            tokens = gather_tokens(tokens, jnp.repeat(token_idx, T, axis=0))
        tokens = tokens.reshape(B, T, tokens.shape[1], self.emb_features)
        return self.norm(self.stack(tokens, train=train))


@models("jepa_predictor")
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

    def setup(self):
        self.proj_in = nn.Dense(features=self.predictor_features, dtype=self.dtype,
                                precision=self.precision, name="proj_in")
        self.mask_token = self.param(
            "mask_token", nn.initializers.normal(0.02), (1, 1, self.predictor_features))
        stack = FactorizedTokenStack if self.factorized else TokenStack
        self.stack = stack(
            features=self.predictor_features, num_layers=self.num_layers,
            **self._stack_options(),
        )
        self.norm = LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        self.proj_out = nn.Dense(features=self.emb_features, dtype=self.dtype,
                                 precision=self.precision, name="proj_out")

    def __call__(self, context, context_idx, target_idx, train: bool = False):
        """Map context embeddings [B, (T,) N_ctx, F] to predictions [B, (T,) N_tgt, F]."""
        pos_embed = jnp.asarray(
            scan_ordered_pos_embed(self.predictor_features, *self.grid, self.scan_order),
            dtype=self.dtype or jnp.float32)

        def positions(indices):
            pos = pos_embed[indices]                       # [B, N, P]
            return pos[:, None] if self.factorized else pos

        num_target_tokens = target_idx.shape[-1]
        context = _projected(self.proj_in, context) + positions(context_idx)
        targets = self.mask_token + positions(target_idx)
        if self.factorized:
            targets = jnp.broadcast_to(
                targets, (*context.shape[:-2], num_target_tokens, self.predictor_features))

        tokens = jnp.concatenate([context, targets], axis=-2)
        tokens = self.stack(tokens, train=train)
        return _projected(self.proj_out, self.norm(tokens[..., -num_target_tokens:, :]))


def _projected(dense: nn.Dense, x: jax.Array) -> jax.Array:
    """`dense(x)` over `[batch, ..., tokens, width]`, run where
    `down_projection` places a projection of `x` to `dense.features`: its
    positions spread over the tensor axis, or whole on every shard."""
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    place = down_projection(flat, dense.features)
    return constrain(dense(constrain(flat, place)), place).reshape(*x.shape[:-1], dense.features)
