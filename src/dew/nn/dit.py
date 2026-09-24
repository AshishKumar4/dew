"""
Shared machinery for the DiT family.

Every DiT-style model here is the same sandwich: patchify in some scan order,
add a 2D sincos position signal, run adaLN-Zero modulated blocks over the
token sequence, and unpatchify back. The models differ in the token mixer
inside the block (attention or S5 SSM) and in how the blocks are arranged
(plain stack, U-shaped skips, hybrid patterns). This module owns the
sandwich; the model files arrange blocks.
"""

import inspect
import math
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from .attention import LayerNorm, NormalAttention
from .blocks import FourierEmbedding, TimeProjection
from .conv import Conv
from .precision import fp32_result_dot_general
from .rope import rotary_freqs
from .scan_orders import (
    build_2d_sincos_pos_embed,
    hilbert_indices,
    hilbert_patchify,
    hilbert_unpatchify,
    inverse_permutation,
    unpatchify,
    zigzag_indices,
    zigzag_patchify,
)
from .sharding import MLP_HIDDEN, RESIDUAL, constrain, logical_axes
from .ssm import BidirectionalS5Layer, S5Layer, SpatialFusionConv

SCAN_ORDERS = ('raster', 'hilbert', 'zigzag')

# The rotary base every model here rotates with, RoFormer's.
ROPE_THETA = 10000.0


@struct.dataclass
class TextContext:
    """Encoded text a model conditions on: `hidden` `[B, L, D]` from the text
    tower and `mask` `[B, L]`, ones on the real tokens and zeros on the padding,
    which any pooling over L weights by."""
    hidden: jax.Array
    mask: jax.Array


def masked_mean(x, mask):
    """The mean of `x` `[B, L, D]` over L, counting only the rows `mask`
    `[B, L]` marks; a padded row moves nothing, and a row with no real
    tokens at all contributes zero."""
    weights = jnp.asarray(mask, x.dtype)[:, :, None]
    counted = jnp.sum(weights, axis=1)
    return jnp.sum(x * weights, axis=1) / jnp.maximum(counted, 1)


def scan_indices(scan_order: str, H_P: int, W_P: int):
    """Forward permutation for a scan order (None for raster)."""
    if scan_order == 'hilbert':
        return hilbert_indices(H_P, W_P)
    if scan_order == 'zigzag':
        return zigzag_indices(H_P, W_P)
    return None


def scan_ordered_pos_embed(emb_dim: int, H_P: int, W_P: int, scan_order: str):
    """2D sincos position embedding permuted into the scan order, so token i of
    the sequence carries the signal for the 2D position it came from."""
    pos_embed = build_2d_sincos_pos_embed(emb_dim, H_P, W_P)
    order = scan_indices(scan_order, H_P, W_P)
    return pos_embed if order is None else pos_embed[order]


def build_block_pattern(num_layers: int, ssm_attention_ratio: str = "3:1",
                        block_pattern: Sequence[str] | None = None):
    """Per-layer mixer choice from a ratio string like '3:1', 'all-ssm', 'all-attn'."""
    if block_pattern is not None:
        if len(block_pattern) != num_layers:
            raise ValueError("block_pattern names every layer's mixer; got "
                             f"{len(block_pattern)} entries for {num_layers} layers")
        return list(block_pattern)
    if ssm_attention_ratio == "all-ssm":
        return ['ssm'] * num_layers
    if ssm_attention_ratio == "all-attn":
        return ['attn'] * num_layers
    n_ssm, n_attn = (int(part) for part in ssm_attention_ratio.split(':'))
    unit = ['ssm'] * n_ssm + ['attn'] * n_attn
    return (unit * (num_layers // len(unit) + 1))[:num_layers]


class PatchEmbedding(nn.Module):
    """Non-overlapping `patch_size` patches through one convolution, as a
    row-major token sequence `[B, H_P * W_P, embedding_dim]`."""
    patch_size: int
    embedding_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        batch, height, width, _ = x.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        x = Conv(features=self.embedding_dim,
                 kernel_size=(self.patch_size, self.patch_size),
                 strides=(self.patch_size, self.patch_size),
                 dtype=self.dtype,
                 precision=self.precision)(x)
        return jnp.reshape(x, (batch, -1, self.embedding_dim))


@logical_axes({("ada_proj",): ("embed", "modulation")})
class AdaLNParams(nn.Module):
    """The six adaLN-Zero modulation vectors of one block, `[B, 1, 6 * features]`,
    from the conditioning vector.

    SiLU then a zero-init projection, as in the DiT paper: without the
    nonlinearity every block's modulation would be an affine map of the same
    shared vector, and the zero init makes every block the identity at start.
    """
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, conditioning):
        if conditioning.ndim == 2:
            conditioning = jnp.expand_dims(conditioning, axis=1)
        return nn.Dense(
            features=6 * self.features,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="ada_proj"
        )(nn.silu(conditioning))


@logical_axes({("patch_embed", "Conv_0"): (None, None, None, "embed")},
              heuristic=(("hilbert_projection",),))
class PatchSequenceEmbed(nn.Module):
    """Patchify in raster/hilbert/zigzag order and add the 2D sincos signal.

    Returns `(tokens, inv_idx)`; `inv_idx` restores row-major order on the
    way out and is None for raster.
    """
    patch_size: int
    emb_features: int
    scan_order: str = 'raster'
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        assert self.scan_order in SCAN_ORDERS, f"Unknown scan order {self.scan_order}"
        if self.scan_order == 'raster':
            self.patch_embed = PatchEmbedding(
                patch_size=self.patch_size,
                embedding_dim=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="patch_embed",
            )
        else:
            # The patches arrive already permuted, so a dense projection of
            # the raw pixels replaces the strided convolution.
            self.scan_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection",
            )

    def __call__(self, x):
        _, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by patch size"
        H_P, W_P = H // self.patch_size, W // self.patch_size

        inv_idx = None
        if self.scan_order == 'hilbert':
            patches_raw, inv_idx = hilbert_patchify(x, self.patch_size)
            tokens = self.scan_proj(patches_raw)
        elif self.scan_order == 'zigzag':
            patches_raw, inv_idx = zigzag_patchify(x, self.patch_size)
            tokens = self.scan_proj(patches_raw)
        else:
            tokens = self.patch_embed(x)

        pos_embed = scan_ordered_pos_embed(
            self.emb_features, H_P, W_P, self.scan_order)
        tokens = tokens + jnp.asarray(pos_embed, dtype=tokens.dtype)[None, :, :]
        return tokens, inv_idx


@logical_axes({("time_embed", "layers_2"): ("mlp", "embed")},
              heuristic=(("time_embed", "layers_1"), ("text_context_proj",)))
class ConditioningEmbed(nn.Module):
    """Fourier time embedding + the text projection mean-pooled over the real
    tokens, summed into the single conditioning vector the adaLN modulation
    consumes."""
    emb_features: int
    mlp_ratio: int = 4
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features * self.mlp_ratio,
                           dtype=self.dtype, precision=self.precision),
            nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision),
        ], name="time_embed")
        self.text_proj = nn.Dense(
            features=self.emb_features, dtype=self.dtype,
            precision=self.precision, name="text_context_proj")

    def __call__(self, temb, textcontext: TextContext | None = None):
        cond_emb = self.time_embed(temb)
        if textcontext is not None:
            text_emb = self.text_proj(textcontext.hidden)
            cond_emb = cond_emb + masked_mean(text_emb, textcontext.mask)
        return cond_emb


@logical_axes({("final_ada_proj",): ("embed", "modulation"),
               ("final_proj",): ("embed", "output")})
class PatchSequenceOutput(nn.Module):
    """Final norm + zero-init fp32 head + unpatchify for any scan order and
    any (non-square included) patch grid."""
    patch_size: int
    output_channels: int
    modulated: bool = False  # adaLN shift/scale on the final norm (DiT FinalLayer)
    norm_epsilon: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, tokens, inv_idx, H, W, conditioning=None):
        features = tokens.shape[-1]
        x_out = LayerNorm(
            epsilon=self.norm_epsilon, use_scale=not self.modulated,
            use_bias=not self.modulated, dtype=self.dtype, name="final_norm")(tokens)
        if self.modulated:
            assert conditioning is not None, "modulated output head needs the conditioning vector"
            if conditioning.ndim == 2:
                conditioning = jnp.expand_dims(conditioning, axis=1)
            shift, scale = jnp.split(nn.Dense(
                features=2 * features,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=nn.initializers.zeros,
                name="final_ada_proj",
            )(nn.silu(conditioning)), 2, axis=-1)
            x_out = x_out * (1 + scale) + shift

        x_out = nn.Dense(
            features=self.patch_size * self.patch_size * self.output_channels,
            dtype=self.dtype,
            precision=self.precision,
            # The loss is computed in fp32, so the result and the accumulation
            # are fp32 while the operands stay in the compute dtype.
            dot_general=fp32_result_dot_general(self.precision),
            kernel_init=nn.initializers.zeros,
            name="final_proj",
        )(x_out)

        if inv_idx is not None:
            return hilbert_unpatchify(x_out, inv_idx, self.patch_size, H, W, self.output_channels)
        return unpatchify(x_out, self.patch_size, H, W, self.output_channels)


# A fused attention forward reaches a remat policy as one of these
# primitives, not as a dot: jax wraps its cuDNN kernel in a custom_vjp whose
# forward returns the attention output together with the softmax statistics
# its backward pass consumes.
FUSED_ATTENTION_FORWARD = frozenset(
    {'dot_product_attention_fwd', 'dot_product_attention_fwd_wrapper'})

_DOTS_AND_ATTENTION_OUTPUT = jax.checkpoint_policies.save_from_both_policies(
    jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    jax.checkpoint_policies.save_only_these_names('attention_output'))


def saved_through_remat(prim, *args, **params) -> bool:
    """The values a rematerialized block keeps instead of recomputing.

    Three kinds. Unbatched matmul outputs, which keeps the recompute cheap
    while leaving the reference path's [B, H, Q, K] scores, a batched dot, out
    of the residuals. Whatever `scaled_dot_product_attention` returns, which
    it labels 'attention_output'. And the whole fused attention forward: a
    name can only mark that primitive's output, and its backward pass also
    needs the softmax statistics, so a policy that saves the output alone
    still replays the entire flash forward. Saving the primitive keeps both,
    which is what takes a step's fused forward calls from two per layer back
    to one.
    """
    return (str(prim) in FUSED_ATTENTION_FORWARD
            or _DOTS_AND_ATTENTION_OUTPUT(prim, *args, **params))


RematChoice = bool | Literal['dots', 'full']
"""A diffusion backbone's `remat`: False keeps every activation, True or
'dots' recomputes a block but keeps its matmul outputs and attention
forward, 'full' recomputes the whole block from its inputs."""


def remat_block(block_cls, enabled: RematChoice, policy: str | None = 'dots'):
    """Optionally rematerialize a block class.

    Recomputing a block during the backward pass trades extra compute for a
    large drop in activation memory, which caps trainable model size. The
    default policy is `saved_through_remat`, which keeps the block's cheap
    matmul outputs and its attention forward. Blocks carrying complex
    intermediates (the S5 mixer) must pass policy=None: saving a residual
    goes through jax.lax.reduce_precision, which only accepts floating dtypes.

    `train` selects a Python branch, so it has to stay static; that also means
    callers must pass it positionally for jax to see it as such.
    """
    if not enabled:
        return block_cls
    if enabled == 'full':
        policy = None
    names = list(inspect.signature(block_cls.__call__).parameters)
    return nn.remat(
        block_cls,
        static_argnums=tuple(i for i, name in enumerate(names) if name == 'train'),
        policy=(saved_through_remat if policy == 'dots' else None),
    )


@logical_axes({("mlp", "layers_0"): ("embed", "mlp"), ("mlp", "layers_2"): ("mlp", "embed")},
              heuristic=(("ssm",), ("spatial_fusion",)))
class ModulatedBlock(nn.Module):
    """adaLN-Zero modulated residual block with a pluggable token mixer.

    mixer='attention' gives the standard DiT block, rotated by the `freqs_cis`
    a call passes (None leaves the tokens unrotated); mixer='ssm' replaces
    attention with a bidirectional S5 scan, optionally followed by
    Spatial-Mamba style 2D state fusion, and ignores freqs_cis.

    modulated=False drops the adaLN-Zero conditioning path entirely, leaving a
    plain pre-norm residual block with learned affine norms, the ViT block a
    JEPA encoder needs, where there is no timestep to condition on.
    """
    features: int
    num_heads: int
    mixer: str = 'attention'
    modulated: bool = True
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    use_gating: bool = True
    qk_norm: bool = False
    attention_impl: str = "auto"  # an AttentionImpl
    # ssm mixer options
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    use_2d_fusion: bool = False
    scan_order: str = 'raster'  # needed to un-permute for the 2D fusion conv

    def setup(self):
        assert self.mixer in ('attention', 'ssm'), f"Unknown mixer {self.mixer}"
        hidden_features = int(self.features * self.mlp_ratio)

        if self.modulated:
            self.ada_params_module = AdaLNParams(
                self.features, dtype=self.dtype, precision=self.precision)
        # Without modulation the norms carry their own affine, since there is
        # no conditioning vector left to supply the shift and scale
        affine = not self.modulated
        self.norm1 = LayerNorm(
            epsilon=self.norm_epsilon, use_scale=affine, use_bias=affine,
            dtype=self.dtype, name="norm1")
        self.norm2 = LayerNorm(
            epsilon=self.norm_epsilon, use_scale=affine, use_bias=affine,
            dtype=self.dtype, name="norm2")

        if self.mixer == 'attention':
            self.attention = NormalAttention(
                query_dim=self.features,
                heads=self.num_heads,
                dim_head=self.features // self.num_heads,
                dtype=self.dtype,
                precision=self.precision,
                use_bias=True,
                qk_norm=self.qk_norm,
                attention_impl=self.attention_impl,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
            )
        else:
            ssm_cls = BidirectionalS5Layer if self.bidirectional_ssm else S5Layer
            self.ssm = ssm_cls(features=self.features, state_dim=self.ssm_state_dim, dtype=self.dtype, name="ssm")
            if self.use_2d_fusion:
                assert self.scan_order in SCAN_ORDERS, f"Unknown scan_order {self.scan_order}"
                self.spatial_fusion = SpatialFusionConv(
                    features=self.features,
                    dilations=(1, 2, 3),
                    kernel_size=3,
                    dtype=self.dtype,
                    precision=self.precision,
                    name="spatial_fusion",
                )

        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            # Column-parallel under a tensor axis; the activation holds the
            # place so the layers keep their names.
            lambda hidden: nn.gelu(constrain(hidden, MLP_HIDDEN)),
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision),
        ])
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def _apply_2d_fusion(self, ssm_output):
        """Un-permute scan-ordered SSM output to a 2D grid, fuse, re-permute back."""
        B, S, F = ssm_output.shape
        # square patch grid; S is a static python int at trace time
        H_P = math.isqrt(S)
        W_P = H_P
        assert H_P * W_P == S, (
            f"2D fusion requires a square patch grid; got S={S} which is not a "
            f"perfect square.")

        scan_fwd = scan_indices(self.scan_order, H_P, W_P)
        if scan_fwd is None:
            return self.spatial_fusion(ssm_output.reshape(B, H_P, W_P, F)).reshape(B, S, F)
        row_major = ssm_output[:, inverse_permutation(scan_fwd), :]
        fused = self.spatial_fusion(row_major.reshape(B, H_P, W_P, F)).reshape(B, S, F)
        return fused[:, scan_fwd, :]

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis, train: bool = False):
        if self.modulated:
            scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
                self.ada_params_module(conditioning), 6, axis=-1
            )
        else:
            assert conditioning is None, "an unmodulated block takes no conditioning"
            # Identity modulation, so the block body below collapses to a
            # plain pre-norm residual block without branching on the mode
            scale_mlp = shift_mlp = scale_attn = shift_attn = 0.0
            gate_mlp = gate_attn = 1.0

        # The token stream sits where the batch does before and after each
        # sublayer, as the decoder's does.
        skip = constrain(x, RESIDUAL)
        x_modulated = self.norm1(skip) * (1 + scale_attn) + shift_attn
        if self.mixer == 'attention':
            mixer_output = self.attention(x_modulated, freqs_cis=freqs_cis)
        else:
            mixer_output = self.ssm(x_modulated)
            if self.use_2d_fusion:
                mixer_output = self._apply_2d_fusion(mixer_output)
        mixer_output = self.dropout(mixer_output, deterministic=not train)

        if self.use_gating:
            skip = constrain(skip + gate_attn * mixer_output, RESIDUAL)
        else:
            skip = constrain(skip + mixer_output, RESIDUAL)

        x_mlp_modulated = self.norm2(skip) * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(x_mlp_modulated)
        mlp_output = self.dropout(mlp_output, deterministic=not train)

        if self.use_gating:
            return constrain(skip + gate_mlp * mlp_output, RESIDUAL)
        return constrain(skip + mlp_output, RESIDUAL)


def rope_for_scan(seq_len: int, head_dim: int, scan_order: str):
    """RoPE frequencies for a token sequence in `scan_order`: the rotation at
    every index for raster, where the index is a position, and None for the
    hilbert and zigzag orders, where it is not (the 2D sincos embedding
    carries position there)."""
    if scan_order != 'raster':
        return None
    return rotary_freqs(jnp.arange(seq_len), head_dim, ROPE_THETA)
