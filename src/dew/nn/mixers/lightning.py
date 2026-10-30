"""Lightning attention: MiniMax-Text-01's and MiniMax-M1's linear-attention layer.

A head's memory decays by one constant a token and takes each token's key and value, and a query
reads it (`MiniMaxText01LightningAttention`, modeling_minimax_text_01.py at MiniMaxAI/MiniMax-Text-01
a7351bf2, :210-339):

    S_t = exp(-r) S_t-1 + k_t v_t^T          o_t = q_t^T S_t

with `q`, `k` and `v` one projection under the model's activation, split per head in that order
(:261-264), and `r` the head's rate: ALiBi's slope for the head (`_build_slope_tensor`, :1234-1256)
times `1 - layer / (layers - 1) + 1e-5` (:1348), so the first layer forgets fastest, each factor in
fp32 as the reference multiplies its fp32 slopes by it; the layer is data the stack hands each
(`MixerCall.depth`), so the lightning layers are one kind and scan as one. The output is
RMS-normed over every head at once (`MiniMaxText01RMSNorm`, eps 1e-6), gated by a sigmoid of the
layer's input through `output_gate`, and projected back by `out_proj` (:328-334). The reference
computes it in blocks of 256 tokens (:281-313), the recurrence's chunked form, which is
`dew.nn.kernels.recurrence_chunk.chunk` without the delta rule; a decode token is the recurrence's
step (:316-326). A padded slot neither decays nor writes the memory: the reference zeroes its
values (:280-281), whose memory a left-padded row holds at zero anyway.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.mesh import LayoutRefused
from dew.nn.dense import Dense
from dew.nn.flops import projection_flops, ssd_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import CHUNK_SIZE, Chunk, chunk
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.sharding import logical_axes, sequence_shards

LIGHTNING = recurrence.Hooks(decay='scalar', delta=False)
"""The recurrence's form: one log decay a head, no delta rule, no write gate, no scale."""

ACTIVATIONS = {'silu': nn.silu}
"""The activations the projection takes, by the config's `hidden_act`: every release's SiLU."""


def lightning_slopes(heads: int) -> tuple[float, ...]:
    """Each head's slope, ALiBi's (`get_slopes`, :1237-1251), in double precision as the reference's
    Python computes them before it stores them in fp32."""

    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** index for index in range(n)]

    def slopes(n: int) -> list[float]:
        if math.log2(n).is_integer():
            return power_of_two(n)
        closest = 2 ** math.floor(math.log2(n))
        return power_of_two(closest) + slopes(2 * closest)[0::2][:n - closest]

    return tuple(slopes(heads))


@logical_axes({
    ("qkv_proj",): ("embed", "linear"),
    ("output_gate",): ("embed", "linear"),
    ("out_proj",): ("attention", "embed"),
})
class LightningAttention(nn.Module):
    """The token mixer of a MiniMax `linear_attention` layer. Parameter names are the release's:
    `qkv_proj` (each head's query, key and value side by side), `output_gate`, `norm/scale` over
    every head's channels, `out_proj`. `slopes` are its heads' (`lightning_slopes`), scaled by its
    place among `layers` (`MixerCall.depth`). The decode state is `recurrent_state` `[B, H, D, D]` in
    at least fp32, which the first decode call only allocates."""

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token
    of its document, with no mask or bias to narrow it."""

    emb_features: int
    num_heads: int
    head_dim: int
    slopes: tuple[float, ...]
    layers: int
    activation: str = 'silu'
    chunk_size: int = CHUNK_SIZE
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def features(self) -> int:
        return self.num_heads * self.head_dim

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The projections in and out, and the memory's write and read a token (`ssd_flops`'s count of
        the same two contractions)."""
        return (projection_flops(lengths, self.emb_features, 4 * self.features)
                + projection_flops(lengths, self.features, self.emb_features)
                + ssd_flops(lengths, self.num_heads, self.head_dim, self.head_dim))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """The projections, without the memory's contractions."""
        return self.model_flops(lengths) - ssd_flops(lengths, self.num_heads, self.head_dim, self.head_dim)

    def setup(self):
        if len(self.slopes) != self.num_heads:
            raise ValueError(f"a lightning layer holds one slope a head: {len(self.slopes)} slopes for "
                             f"{self.num_heads} heads")
        if self.layers < 2:
            raise ValueError(f"a lightning layer's rate falls from the first of its stack's layers to the "
                             f"last, so the stack holds two or more, not {self.layers}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"the projection's activation is one of {sorted(ACTIVATIONS)}, got "
                             f"{self.activation!r}")
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.qkv_proj = dense(3 * self.features, name='qkv_proj')
        self.output_gate = dense(self.features, name='output_gate')
        self.norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True, dtype=self.dtype, name='norm')
        self.out_proj = dense(self.emb_features, name='out_proj')

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        batch, length, _ = x.shape
        tokens = coordinates_of(call.metadata, batch, length)
        shards = sequence_shards()
        if shards > 1:
            raise LayoutRefused(f"lightning attention carries its memory along the whole sequence, which a "
                                f"sequence axis of {shards} splits; run it on a mesh with sequence=1")
        qkv = ACTIVATIONS[self.activation](self.qkv_proj(x))
        heads = qkv.reshape(batch, length, self.num_heads, 3 * self.head_dim)
        query, key, value = jnp.split(heads, 3, axis=-1)
        wide = at_least_fp32(qkv.dtype)
        if call.depth is None:
            raise ValueError("a lightning layer's rates follow its place in the stack, which a decoder's "
                             "stack hands it (MixerCall.depth)")
        # The release's fp32 slopes times its fp32 factor for the layer (:1348), in the state's dtype.
        factors = 1 - np.arange(self.layers) / (self.layers - 1) + 1e-5
        rates = jnp.asarray(self.slopes, wide) * jnp.asarray(factors, wide)[jnp.asarray(call.depth)[0, 0]]
        log_decay = jnp.broadcast_to(-rates, (batch, length, self.num_heads))
        if tokens.valid is not None:
            # exp(0) keeps the memory and a zero value writes nothing.
            log_decay = jnp.where(tokens.valid[:, :, None], log_decay, 0.0)
            value = jnp.where(tokens.valid[:, :, None, None], value, 0)
        held = None
        if call.decode:
            allocated = self.has_variable('cache', 'recurrent_state')
            held = self.variable('cache', 'recurrent_state', jnp.zeros,
                                 (batch, self.num_heads, self.head_dim, self.head_dim), wide)
            if not allocated:
                # The allocation-only call: init_cache's dummy token leaves no memory behind.
                return self.out_proj(jnp.zeros((batch, length, self.features), qkv.dtype))
        if held is not None and length == 1:
            out, _, final = recurrence.step_reference(LIGHTNING, recurrence.Step(
                state=held.value, query=query[:, 0], key=key[:, 0], value=value[:, 0],
                log_decay=log_decay[:, 0], active=None if tokens.valid is None else tokens.valid[:, 0]))
            out = out[:, None]
        else:
            out, final = chunk(LIGHTNING, Chunk(query, key, value, log_decay=log_decay,
                                                state=None if held is None else held.value,
                                                starts=tokens.starts, firsts=tokens.firsts),
                               chunk_size=self.chunk_size, max_segments=tokens.max_segments)
        if held is not None:
            held.value = final.astype(held.value.dtype)
        # The recurrence's output under both names a remat policy reads it by (`RESIDUALS`).
        out = checkpoint_name(checkpoint_name(out, 'scan'), 'context')
        out = self.norm(out.reshape(batch, length, self.features))
        return self.out_proj(nn.sigmoid(self.output_gate(x)) * out)


@dataclasses.dataclass(frozen=True)
class LightningAttentionMixer(MixerBase):
    """The `lightning_attention` kind: MiniMax's linear-attention layer in a stack of
    `num_hidden_layers`, whose place in it sets its heads' rates (`LightningAttention`), by the
    reference's names; `hidden_act` the projection's activation and `block_size` the chunk the
    recurrence runs in, which changes no value but its rounding. The heads and their width are the
    context's: the layer's attention geometry, without its key/value heads, rotary or window."""

    residuals = ('context', 'scan')
    reads_depth = True

    num_hidden_layers: int = 2
    hidden_act: str = 'silu'
    block_size: int = CHUNK_SIZE

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("lightning_attention requires causal=True; its recurrence has no "
                             "bidirectional mode")
        return functools.partial(
            LightningAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            slopes=lightning_slopes(ctx.num_heads),
            layers=self.num_hidden_layers,
            activation=self.hidden_act,
            chunk_size=self.block_size,
            dtype=ctx.dtype,
            precision=ctx.precision)
