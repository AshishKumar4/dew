"""Kimi Delta Attention: the linear-attention layer of GLM-5.3-Flash.

KDA is the gated delta rule of `dew.nn.linear` with a different recurrence
decay and parameterisation (`Glm5NextTextLinearAttention`,
modeling_glm5_next.py:584-733):

- The decay is a vector per head over the key dimensions, `S <- S *
  exp(g_t)[:, None]` (modeling_glm5_next.py:468-471, 529-532).
- `g = f_b_proj(f_a_proj(x)) + dt_bias` over `[H, Dk]`, and with
  `linear_lower_bound` the decay is `lower_bound * sigmoid(exp(A_log) * g)`
  instead of `-exp(A_log) * softplus(g)` (:319-335).
- `beta = sigmoid(b_proj(x))` per head (:697).
- q, k and v have their own projections and depthwise convs, stored apart
  as the release stores them and concatenated at call time in the
  reference's order (:607-615, 642-649).
- The output gate `g_b_proj(g_a_proj(x))` drives a sigmoid-gated weighted
  RMSNorm, `o_norm` (:339-358, 729-730), then `o_proj`.
- q and k are l2-normalised inside the rule as `x / sqrt(sum x^2 + eps)`
  (:416-424).

`chunk_kimi_delta_attention` (:482-578) is the gated delta rule's chunked form
with a decay per key dimension, `dew.nn.kernels.recurrence_chunk.chunk_gated_delta_rule`: on a
GPU fla's chunked KDA ported to Pallas (`dew.nn.kernels.fla_kda`), elsewhere
the portable math in fp32; the recurrent form is
`dew.nn.kernels.recurrence_chunk.recurrent_delta_rule` (:428-478). tests/test_kda.py holds the
math to a float64 oracle of the reference, and tests/test_fla_kda.py the port.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import logistic
from dew.nn.dense import Dense

from .flops import conv_flops, delta_flops, projection_flops
from .inputs import PLAIN_CALL, MixerCall, coordinates_of
from .kernels import recurrence
from .kernels.recurrence_chunk import CHUNK_SIZE, Chunk, chunk, chunk_gated_delta_rule, kernel_routed
from .kv_cache import Opened, compute_dtype
from .linear import (
    DepthwiseConv1d,
    RMSNormGated,
    decode_delta_rule,
    decode_kernel,
    fused_token,
    held_scan,
    scan_cache,
    sequence_delta_rule,
    sequence_split,
)
from .mixer_base import MixerBase, MixerContext
from .precision import at_least_fp32, float32_dot_general
from .residuals import DELTA_RULE_RESIDUALS
from .sharding import LogicalAxes

KDA_AXES: dict[str, LogicalAxes] = {
    # The attention mixer's projections under its names (`ATTENTION_AXES`).
    "q_proj": ("embed", "heads"),
    "k_proj": ("embed", "kv"),
    "v_proj": ("embed", "kv"),
    "o_proj": ("attention", "embed"),
    # The gates' low-rank pairs and the beta projection are KDA's own.
    "b_proj": ("embed", "kv"),
    "f_a_proj": ("embed", None),
    "f_b_proj": (None, "heads"),
    "g_a_proj": ("embed", None),
    "g_b_proj": (None, "heads"),
    "g_proj": ("embed", "heads"),
}
"""The logical axes of KDA's projections, by name, as it declares them where it makes them."""


class KimiDeltaAttention(nn.Module):
    """The token mixer of a GLM-5.3-Flash `linear_attention` layer.

    Parameter names are the release's: `q_proj`, `k_proj`, `v_proj`,
    `{q,k,v}_conv1d/weight` `[H Dk, 1, K]`, `f_a_proj`, `f_b_proj`, `dt_bias`
    `[H Dk]` and `A_log` `[H]` directly under the layer (transformers renames
    them under `forget_gate`), `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm/weight`
    `[Dk]`, `o_proj`. `lower_bound` is `linear_lower_bound`. The decode state is
    `GatedDeltaNet`'s pair of `cache` leaves.

    `full_rank_gate` is Kimi K3's output gate, one `g_proj` from the model width
    to the heads (`use_full_rank_gate`, modeling_kimi_linear.py:531-537, 651-656
    of moonshotai/Kimi-K3 at f831ab6); the rest of K3's layer is this one (fla
    `chunk_kda`, fla-core 0.5.2, fla/ops/kda/gate.py:57-70).
    """

    reads: ClassVar[frozenset[str]] = frozenset({"forks"})
    """Of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`), the documents that continue another
    (`dew.batch.Forks`), each read from the memory and the conv history its parent leaves
    (`dew.nn.linear.forked_conv`, `forked_scan`); no mask, runs or bias, which a recurrence reading every
    earlier token of its document has no way to narrow."""

    emb_features: int
    num_heads: int
    head_dim: int
    conv_kernel: int = 4
    lower_bound: float | None = -5.0
    full_rank_gate: bool = False
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def qkv_features(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def declared(self) -> dict:
        """What the layer declares of the chunked rule's operands at every call of it (`recurrence.Hooks`):
        its write gates a sigmoid's, at most one, of keys and queries the conv l2-normalized, each of at most
        unit norm (`write_bound`, `unit_queries`), and its least log decay a token (`decay_bound`)."""
        return {"write_bound": 1.0, "unit_queries": True, "decay_bound": self.lower_bound}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Q/K/V/output, low-rank decay and gate, convs, and chunked delta work."""
        inner = self.qkv_features
        direct = 4 * inner + self.head_dim + self.num_heads
        low_rank = self.head_dim * inner
        if self.full_rank_gate:
            direct += inner
        else:
            direct += self.head_dim
            low_rank += self.head_dim * inner
        return (projection_flops(lengths, self.emb_features, direct)
                + projection_flops(lengths, 1, low_rank)
                + 3 * conv_flops(lengths, inner, inner, self.conv_kernel, groups=inner)
                + delta_flops(lengths, self.num_heads, self.head_dim, self.head_dim))

    def setup(self):
        def dense(features: int, *, name: str) -> Dense:
            return Dense(features, use_bias=False, dtype=self.dtype, precision=self.precision,
                         dot_general=float32_dot_general, name=name, kernel_axes=KDA_AXES[name])

        self.q_proj = dense(self.qkv_features, name='q_proj')
        self.k_proj = dense(self.qkv_features, name='k_proj')
        self.v_proj = dense(self.qkv_features, name='v_proj')
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            DepthwiseConv1d(features=self.qkv_features, kernel=self.conv_kernel, name=f'{name}_conv1d')
            for name in 'qkv')
        self.f_a_proj = dense(self.head_dim, name='f_a_proj')
        self.f_b_proj = dense(self.qkv_features, name='f_b_proj')
        self.dt_bias = self.param('dt_bias', nn.initializers.zeros, (self.qkv_features,), jnp.float32)
        self.A_log = self.param('A_log', nn.initializers.zeros, (self.num_heads,), jnp.float32)
        self.b_proj = dense(self.num_heads, name='b_proj')
        if self.full_rank_gate:
            self.g_proj = dense(self.qkv_features, name='g_proj')
        else:
            self.g_a_proj = dense(self.head_dim, name='g_a_proj')
            self.g_b_proj = dense(self.qkv_features, name='g_b_proj')
        # The reference's fp32 norm with its weight, then the sigmoid of the
        # gate (Glm5NextTextRMSNormGated, modeling_glm5_next.py:346-358).
        self.o_norm = RMSNormGated(
            epsilon=self.norm_eps, activation="sigmoid", dtype=self.dtype, name="o_norm"
        )
        self.o_proj = dense(self.emb_features, name='o_proj')

    @property
    def _gates(self) -> recurrence.Gates:
        """The forget gate's law (`recurrence.gate_law`): a decay a key dimension,
        `Glm5NextTextForgetGate.forward`'s (modeling_glm5_next.py:319-335), bounded below by `lower_bound`
        where the layer has one, else the softplus step's (its exp accurate, torch's softplus Dew's one, its
        value past 20 the gate's within float32's rounding)."""
        return recurrence.Gates(bound=self.lower_bound)

    def _decay(self, x, valid=None):
        """`Glm5NextTextForgetGate.forward` (modeling_glm5_next.py:319-335): `[B, S, H, Dk]` in fp32
        (`at_least_fp32`), the recurrence's gate law (`_gates`, `recurrence.gated`) at the layer's own
        boundary: its rate exponentiated in its parameter's dtype, as the layer has always taken it, every
        route (decode, chunked, sharded) handed the decay it forms, a padded token's (`valid`) gate formed
        from zeros (`recurrence.gated`)."""
        B, S, _ = x.shape
        gate = self.f_b_proj(self.f_a_proj(x))
        gate = gate.astype(at_least_fp32(gate.dtype)).reshape(B, S, self.num_heads, self.head_dim)
        bias = jnp.reshape(self.dt_bias, (self.num_heads, self.head_dim))
        decay, _ = recurrence.gated(self._gates, gate, self.A_log[:, None], bias, None, valid)
        return decay

    @nn.nowrap
    def cache_spec(self, rows: int) -> dict[str, Opened]:
        """The layer's decode cache for `rows` rows (`scan_cache`): the conv's history of its queries, keys
        and values and the recurrent state, both in at least fp32."""
        wide = at_least_fp32(compute_dtype(self, self.dtype))
        return scan_cache(rows, self.conv_kernel, 3 * self.qkv_features, 'recurrent_state',
                          (self.num_heads, self.head_dim, self.head_dim), wide, wide)

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        decode, attention_metadata = call.decode, call.metadata
        B, S, _ = x.shape
        tokens = coordinates_of(attention_metadata, B, S)
        valid = tokens.valid
        if valid is not None:
            # The reference zeroes padded rows before projecting them
            # (apply_mask_to_padding_states, modeling_glm5_next.py:361-370).
            x = jnp.where(valid[:, :, None], x, 0)
        projections = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
        if decode:
            # At one row XLA lowers each projection to a reduction and fuses the three into their
            # concatenate, which ran 34 of an H100's 151 us a step (`held_conv1d`'s barrier, behind it).
            projections = jax.lax.optimization_barrier(projections)
        projected = jnp.concatenate(projections, axis=-1)
        wide = at_least_fp32(projected.dtype)
        taps = jnp.concatenate([conv()[0] for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)])
        g = self._decay(x, valid)
        beta = logistic(self.b_proj(x).astype(wide))
        if valid is not None:
            # exp(0) = 1 and beta = 0 preserve the memory across a padded slot.
            g = jnp.where(valid[:, :, None, None], g, 0.0)
            beta = jnp.where(valid[:, :, None], beta, 0.0)
        gate = self.g_proj(x) if self.full_rank_gate else self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, S, self.num_heads, self.head_dim)
        token = None
        forked = tokens.forks is not None
        # An initialization runs the general path, where the layer makes its parameters.
        if decode and S == 1 and not forked and not self.is_initializing() and decode_kernel(
                jax.ShapeDtypeStruct((B, self.num_heads, self.head_dim, self.head_dim), wide),
                groups=self.num_heads, fuses=True) == 'fused':
            token = self._fused_token(taps, g, beta, gate, valid, tokens.starts, projected.dtype)

        def split(part):
            return part.reshape(B, S, self.num_heads, self.head_dim)

        # The chunked op, where the layer has made its parameters, gates and norms its output in its epilogue
        # (`_chunk_hooks`), as the decode token does.
        normed = not self.is_initializing()
        chunked = S > 1 or forked
        # The epilogue's gate, weight and output dtype, the sequence's either path's.
        hooks = self._chunk_hooks(projected.dtype, normed)
        sides = {'z': gate, 'weight': self.o_norm.variables['params']['weight']} if normed else {}
        normed_dtype = (self.o_norm.dtype or projected.dtype) if normed else None

        def scan(convolved, *, starts, valid, held, forked=None):
            query, key, value = (split(part) for part in convolved)
            if not chunked:
                return decode_delta_rule(query, key, value, g, beta, held, starts=starts)
            # The batch's documents' firsts, built once with it (`Coordinates.firsts`); a forked call's
            # documents from its table (`held_scan`, `Chunk.loads`), its padded tokens keeping the memory
            # (`Chunk.valid`). The layer's decay its own (`_decay`), its write gate the op's.
            loads, stores, table = (None, None, 0) if forked is None else forked
            return chunk(hooks, Chunk(
                query, key, value, log_decay=g, beta=beta, state=held, starts=starts,
                firsts=tokens.firsts if forked is None else None,
                loads=loads, stores=stores, held=table, valid=tokens.valid if forked is None else valid,
                **sides), chunk_size=self.chunk_size, max_segments=tokens.max_segments,
                dtype=normed_dtype)

        # The decode state is GatedDeltaNet's pair, here in fp32 (`cache_spec`). The conv
        # normalises the queries and keys (`l2norm`) and rounds once, to the
        # dtype the rule takes: the projections' where the port runs it, as
        # fla's layer hands them over, the wide one for the math.
        rows, keys, values = (split(part) for part in jnp.split(projected, 3, axis=-1))
        # beta a sigmoid's, of the queries and keys the conv l2-normalized (`causal_conv1d`'s pieces).
        ported = S > 1 and kernel_routed(rows, keys, values, g, beta, **self.declared)
        heads = ((self.qkv_features, self.head_dim),) * 2 + ((self.qkv_features, None),)
        shards = sequence_split(decode, tokens)
        if shards > 1:
            rule = functools.partial(chunk_gated_delta_rule, max_segments=tokens.max_segments,
                                     **self.declared)
            # Each shard's rule gates and norms its output in its epilogue, as the unsplit op does.
            out = sequence_delta_rule(projected, taps, g, beta, tokens, valid, shards=shards,
                                      key_heads=self.num_heads, widths=(self.head_dim,) * 3, parts=heads,
                                      conv_dtype=projected.dtype if ported else wide, rule=rule,
                                      chunk_size=self.chunk_size, output=hooks.output, dtype=normed_dtype,
                                      **sides)
        else:
            out = held_scan(self, projected, taps, None, scan,
                            (self.num_heads, self.head_dim, self.head_dim), decode=decode, coordinates=tokens,
                            spec=self.cache_spec(B) if decode else None, state='recurrent_state',
                            state_dtype=wide, conv_dtype=wide, parts=heads,
                            dtype=projected.dtype if ported else wide, token=token,
                            # The chunked op names the recurrence's output before its gated norm.
                            named=not (chunked and normed))
        if token is not None:
            # The fused step gated and normalized its output (`FUSED`), in its epilogue's dtype.
            return self.o_proj(out.astype(projected.dtype))
        if not (chunked and normed):
            # The scan read back in its projections' dtype, as Kimi's delta attention reads it.
            out = self.o_norm(out.astype(projected.dtype), gate)
        return self.o_proj(out.reshape(B, S, self.qkv_features))

    def _chunk_hooks(self, operand, normed: bool) -> recurrence.Hooks:
        """The sequence's chunked op (`chunk`) on the decode token's form (`FUSED`), the conv and the l2 norm
        before it (the conv's pieces, `held_scan`): its decay the layer's (`_decay`) and, where `normed`,
        the sigmoid-gated norm in its epilogue, its output read back in `operand`, under the layer's declared
        bounds (`declared`)."""
        output = recurrence.OutputNorm(self.norm_eps, 'sigmoid', read_in=jnp.dtype(operand).name)
        return dataclasses.replace(FUSED, conv=None, l2norm=False, scale=self.head_dim ** -0.5,
                                   output=output if normed else None, **self.declared)

    def _fused_token(self, taps, g, beta, gate, valid, starts, operand):
        """A decode token's conv, update and gated norm in one launch (`linear.fused_token` under
        `FUSED`): its decay a key dimension, write gate and output gate a row's, as the layer forms them, its
        output read back in `operand`, the input projection's dtype, before the norm."""
        hooks = dataclasses.replace(FUSED, output=recurrence.OutputNorm(self.norm_eps, 'sigmoid',
                                                                        read_in=jnp.dtype(operand).name))
        return fused_token(hooks, taps,
                           heads=(self.num_heads, self.num_heads, self.num_heads),
                           widths=(self.head_dim, self.head_dim, self.head_dim),
                           per_row={'log_decay': g[:, 0], 'beta': beta[:, 0], 'z': gate[:, 0]}, per_head={},
                           weight=lambda: self.o_norm.variables['params']['weight'], valid=valid,
                           dtype=self.o_norm.dtype, starts=starts)


FUSED = recurrence.Hooks(conv=recurrence.Conv('silu'), l2norm=True, decay='vector')
"""Kimi delta attention's decode token whole in the recurrence's step: its conv and silu, the queries' and
keys' l2 norm, the delta rule under a decay a key dimension and a write gate, the sigmoid-gated norm (the
layer's epsilon, `KimiDeltaAttention._fused_token`)."""


@dataclasses.dataclass(frozen=True)
class KimiDeltaAttentionMixer(MixerBase):
    """The `kimi_delta_attention` kind, by GLM-5.3-Flash's config fields:
    `linear_num_heads` heads of `linear_head_dim`, the depthwise conv's
    window and the forget gate's lower bound (configuration_glm5_next.py:143-146),
    and Kimi K3's `use_full_rank_gate` output gate."""

    residuals = DELTA_RULE_RESIDUALS
    # Each document from the state and conv history its parent leaves (`dew.nn.linear.forked_scan`).
    reads_forks = True
    # Its cached prefill resets its state at each packed document's start (`dew.nn.linear`).
    packed_cache = True

    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    use_full_rank_gate: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError(
                "kimi_delta_attention requires causal=True; its recurrence has no bidirectional mode"
            )
        return functools.partial(
            KimiDeltaAttention,
            emb_features=ctx.emb_features,
            num_heads=self.linear_num_heads,
            head_dim=self.linear_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            lower_bound=self.linear_lower_bound,
            full_rank_gate=self.use_full_rank_gate,
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
