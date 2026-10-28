"""Mamba: the selective state-space mixer (Gu & Dao 2023, arXiv 2312.00752).

Each of the `D` inner channels carries its own `[N]` state:

    h_t <- exp(dt_t A) * h_t-1 + dt_t B_t x_t          y_t = h_t . C_t + D x_t

`A = -exp(A_log)` is `[D, N]`, a decay per channel and state dimension, so
unlike Mamba-2's scalar per head (`dew.nn.mixers.mamba2`) the recurrence is
no masked matmul; it runs as a linear scan. `dt_t = softplus(dt_proj(dt_raw))`
is the selective step, from a rank-`time_step_rank` projection, and `B_t`,
`C_t` are `[N]` per token, shared by every channel.

dew computes transformers 5.16.1's `MambaMixer` (modeling_mamba.py:285-482):
`in_proj` yields `[x, z]`, the depthwise causal conv with its bias runs over
`x` with silu after, `x_proj` yields `[dt, B, C]`, `dt_proj` widens the step
and adds its fp32 bias before the softplus, the scan
(`mamba_selective_scan`, 175-282) runs in fp32, the `D` skip is added and
the output gated by `silu(z)` before `out_proj`. The decode step is
`mamba_selective_state_update` (129-172). Parameter names are the
checkpoint's: in_proj, conv1d/{weight,bias}, x_proj, dt_proj/{kernel,bias},
A_log, D, out_proj.

The scan runs in chunks of `chunk_size` tokens: within a chunk an
associative scan over `(exp(dt A), dt B x)` pairs, the reference's
`combine_fn` (231-234), and across chunks the carried `[B, D, N]` state.
Mamba's block has no feed-forward (`MambaBlock`, 505-529: norm, mixer,
residual), which `CausalTransformer` spells as an `mlp_features` of 0.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.mesh import SEQUENCE_AXIS, LayoutRefused
from dew.nn.blocks import normal_kernel
from dew.nn.dense import Dense
from dew.nn.flops import conv_flops, projection_flops, ssd_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of, sequence_blocks
from dew.nn.kernels.recurrence_chunk import RESET_DECAY
from dew.nn.linear import DepthwiseConv1d, entering_state, held_scan, over_sequence, sequence_conv
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import accumulated, at_least_fp32, float32_dot_general, float32_einsum, float32_products
from dew.nn.sharding import logical_axes, rows_like, sequence_shards
from dew.nn.ssm import affine_spans

CHUNK_SIZE = 64
"""Tokens per associative scan. Each chunk materialises its `[B, C, D, N]`
decays and writes, so the chunk bounds the scan's memory; the output does
not depend on it."""


def selective_scan(x, dt, A, B, C, D, state=None, starts=None, chunk_size: int = CHUNK_SIZE):
    """`mamba_selective_scan` (modeling_mamba.py:175-282) after the softplus.

    `x` and `dt` `[B, S, D]`, `A` `[D, N]`, `B` and `C` `[B, S, N]`, `D` `[D]`;
    returns `(output [B, S, D], state [B, D, N])` computed in `at_least_fp32`
    and cast back. `starts` `[B, S]` marks packed documents' first tokens,
    whose entering state drops. A call shorter than a chunk scans its own
    length, so a decode step is the recurrence's one step.
    """
    dtype, work = x.dtype, at_least_fp32(x.dtype)
    x, dt, A, B, C, D = (jnp.asarray(t, work) for t in (x, dt, A, B, C, D))
    batch, length, channels = x.shape
    chunk_size = min(chunk_size, length)
    carried = (rows_like(jnp.zeros((batch, channels, A.shape[-1]), work), x) if state is None
               else jnp.asarray(state, work))
    pad = (chunk_size - length % chunk_size) % chunk_size

    def chunks(t):  # [B, S, ...] -> [NC, B, C, ...], chunks leading for the scan
        return sequence_blocks(t, chunk_size, leading=True)

    # A padded step is dt 0, which neither decays nor writes, and no reset.
    resets = jnp.zeros((batch, length), jnp.bool_) if starts is None else jnp.asarray(starts, jnp.bool_)

    def one_chunk(h, step):
        x_c, dt_c, B_c, C_c, reset_c = step   # [B, C, D] twice, [B, C, N] twice, [B, C]
        decay = jnp.exp(dt_c[..., None] * A)                # [B, C, D, N]
        decay = jnp.where(reset_c[..., None, None], 0.0, decay)
        write = (dt_c * x_c)[..., None] * B_c[:, :, None, :]
        # The reference's `combine_fn` (modeling_mamba.py:231-234).
        decays, writes = jax.lax.associative_scan(affine_spans, (decay, write), axis=1)
        states = decays * h[:, None] + writes
        read = float32_einsum('bcdn,bcn->bcd', states, C_c)
        return states[:, -1], read

    final, out = jax.lax.scan(one_chunk, carried, tuple(chunks(t) for t in (x, dt, B, C, resets)))
    out = jnp.moveaxis(out, 0, 1).reshape(batch, length + pad, channels)[:, :length] + x * D
    return out.astype(dtype), final.astype(dtype)


@logical_axes({
    ("in_proj",): ("embed", "linear"),
    ("out_proj",): ("attention", "embed"),
})
class Mamba(nn.Module):
    """The token mixer of a Mamba layer, `MambaMixer` (modeling_mamba.py:285-482).

    `intermediate_size` is the config's `expand * hidden_size`, `state_size`
    each channel's `N`, and `time_step_rank` the width of the raw step.

    The decode state is `ssm_state` `[B, D, N]` and `conv_state` `[B, K-1, D]`
    in the `cache` collection (`held_scan`); a call holding it scans from
    the held state. Padded rows (`attention_metadata.valid`) are zeroed
    before the projection and take a zero step, and packed documents
    (`segment_ids`) run as if alone.
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token
    of its document, with no mask or bias to narrow it."""

    emb_features: int
    intermediate_size: int
    state_size: int = 16
    time_step_rank: int = 48
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    chunk_size: int = CHUNK_SIZE
    init_std: float | None = None
    """Normal std of in_proj, x_proj and the conv taps; None keeps lecun normal."""
    output_init_std: float | None = None
    """Normal std of out_proj; None follows init_std."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Input/output maps, conv and the recurrent state write/read, 4*D*N.

        The delta*B*x outer write and C*h read each cost 2*D*N per token.
        Pointwise exp(delta*A)*h and gates are excluded, the same MAC
        convention used for GDN/KDA and SSD, independent of chunk size.
        """
        inner = self.intermediate_size
        return (projection_flops(lengths, self.emb_features, 3 * inner)
                + projection_flops(lengths, inner, 2 * self.time_step_rank + 2 * self.state_size)
                + conv_flops(lengths, inner, inner, self.conv_kernel, groups=inner)
                + ssd_flops(lengths, 1, self.state_size, inner))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Frozen learned maps and conv, excluding the scan's state write and read (`ssd_flops`)."""
        return self.model_flops(lengths) - ssd_flops(lengths, 1, self.state_size, self.intermediate_size)

    def setup(self):
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size counts tokens per chunk, got {self.chunk_size}")
        dense = functools.partial(Dense, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.in_proj = dense(2 * self.intermediate_size, use_bias=self.use_bias, name='in_proj',
                             **normal_kernel(self.init_std))
        self.conv1d = DepthwiseConv1d(features=self.intermediate_size, kernel=self.conv_kernel,
                                      use_bias=self.use_conv_bias, init_std=self.init_std, name='conv1d')
        self.x_proj = dense(self.time_step_rank + 2 * self.state_size, use_bias=False, name='x_proj',
                            **normal_kernel(self.init_std))
        self.dt_proj = _StepProjection(self.intermediate_size, self.time_step_rank, dtype=self.dtype,
                                       precision=self.precision, name='dt_proj')
        # The reference's init (init_mamba_weights, modeling_mamba.py:337-343):
        # A_log = log(1..N) on every channel, D ones.
        self.A_log = self.param('A_log', _log_state_index, (self.intermediate_size, self.state_size))
        self.D = self.param('D', nn.initializers.ones, (self.intermediate_size,), jnp.float32)
        self.out_proj = dense(self.emb_features, use_bias=self.use_bias, name='out_proj', **normal_kernel(
            self.init_std if self.output_init_std is None else self.output_init_std))

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        decode, attention_metadata = call.decode, call.metadata
        batch, length, _ = x.shape
        tokens = coordinates_of(attention_metadata, batch, length)
        valid = tokens.valid
        shards = sequence_shards()
        if shards > 1 and decode:
            raise LayoutRefused(
                f"mamba under a sequence axis of {shards} runs whole training sequences; a decode step "
                "holds one state per row. Decode on a mesh with sequence=1.")
        if valid is not None:
            x = jnp.where(valid[:, :, None], x, 0)
        mixed, gate = jnp.split(self.in_proj(x), 2, axis=-1)
        # The conv and the scan run in fp32, as the reference's torch path
        # casts them, or in the model's own dtype where it is wider.
        wide = at_least_fp32(mixed.dtype)
        taps, bias = self.conv1d()
        if shards > 1:
            # Each shard convolves its slice reading the real tokens before it from the earlier ones
            # (`sequence_conv`); the step, B and C come token by token outside; each shard scans from
            # zero and again from the state the earlier ones leave it (`_sequence_scan`).
            convolved = over_sequence(functools.partial(_sequence_conv, dtype=wide), shards,
                                      (mixed.astype(wide), tokens.segments, valid), ((taps, bias),))
            xs, step, B, C = self._projected(convolved, valid)
            A = -jnp.exp(jnp.asarray(self.A_log, wide))
            out = over_sequence(functools.partial(_sequence_scan, chunk_size=self.chunk_size), shards,
                                (xs, step, B, C, tokens.starts), ((A, jnp.asarray(self.D, wide)),))
            out = checkpoint_name(checkpoint_name(out, 'scan'), 'context')
        else:
            out = held_scan(self, mixed, taps, bias, self._scan,
                            (self.intermediate_size, self.state_size),
                            decode=decode, coordinates=tokens, state_dtype=wide, conv_dtype=wide, dtype=wide)
        if out is None:
            # Allocation only: init_cache's dummy token must not consume a
            # position or leave state behind.
            return self.out_proj(jnp.zeros((batch, length, self.intermediate_size), self.dtype))
        if valid is not None:
            out = jnp.where(valid[:, :, None], out, 0)
        gated = out * nn.silu(gate.astype(wide))
        return self.out_proj(gated.astype(gate.dtype))

    def _scan(self, convolved, *, starts, valid, held):
        """The selective scan of the convolved `[B, S, D]` inner channels
        (`_projected`). Returns `[B, S, D]` before the gate and the state
        leaving it."""
        xs, step, B, C = self._projected(convolved, valid)
        wide = xs.dtype
        return selective_scan(xs, step, -jnp.exp(jnp.asarray(self.A_log, wide)), B, C,
                              jnp.asarray(self.D, wide), held, starts, self.chunk_size)

    def _projected(self, convolved, valid):
        """The scan's operands from the convolved `[B, S, D]` inner channels, token by token: the
        channels (padding zero), the step (the raw step, B and C from `x_proj`, then the raw step
        through `dt_proj`; zero on padding), B and C."""
        xs = convolved
        if valid is not None:
            xs = jnp.where(valid[:, :, None], xs, 0)
        raw, B, C = jnp.split(self.x_proj(xs), [self.time_step_rank, self.time_step_rank + self.state_size],
                              axis=-1)
        step = self.dt_proj(raw)
        if valid is not None:
            # A zero step neither decays nor writes the state.
            step = jnp.where(valid[:, :, None], step, 0.0)
        return xs, step, B, C


def _sequence_conv(mixed, segments, valid, taps, bias, *, dtype):
    """This shard's slice of the conv, inside a `shard_map` that holds the sequence axis manual
    (`over_sequence`): each real token reading the `K-1` real tokens before it in its document, from the
    earlier shards too (`sequence_conv`)."""
    return sequence_conv(mixed, taps, bias, segments, valid, dtype=dtype)


def _sequence_scan(xs, step, B, C, starts, A, D, *, chunk_size: int):
    """This shard's slice of the selective scan, inside a `shard_map` that holds the sequence axis manual
    (`over_sequence`): run from zero, the state it leaves and its span's log decay `A sum(dt)` (a document
    start in it zeroing the decay, `RESET_DECAY`) give the state the earlier shards leave it
    (`entering_state`), and the scan runs again from that state, a start in it dropping it as on one
    device."""
    _, leaving = selective_scan(xs, step, A, B, C, D, None, starts, chunk_size)
    decay = jnp.sum(step, axis=1)[..., None] * A                    # [B, D, N]
    if starts is not None:
        decay = jnp.where(jnp.any(starts, axis=1)[:, None, None], RESET_DECAY, decay)
    out, _ = selective_scan(xs, step, A, B, C, D, entering_state(decay, leaving, SEQUENCE_AXIS), starts,
                            chunk_size)
    return out


class _StepProjection(nn.Module):
    """`dt_proj`: the raw step widened to every channel, then its bias and
    the softplus in at least fp32, where the reference adds the fp32 bias
    inside its scan (modeling_mamba.py:467-468, 203-206). The reference's
    init draws the kernel uniformly within rank ** -0.5 and the bias as the
    inverse softplus of a step (init_mamba_weights, 344-357)."""

    features: int
    rank: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, raw):
        bound = self.rank ** -0.5
        kernel = self.param(
            'kernel', lambda key, shape: jax.random.uniform(key, shape, jnp.float32, -bound, bound),
            (self.rank, self.features))
        bias = self.param('bias', inverse_softplus_step, (self.features,))
        dtype = self.dtype if self.dtype is not None else raw.dtype
        product = functools.partial(jnp.matmul, precision=float32_products(dtype, self.precision))
        widened = accumulated(product, raw.astype(dtype), kernel.astype(dtype))
        wide = at_least_fp32(widened.dtype)
        return nn.softplus(widened.astype(wide) + bias.astype(wide))


def _log_state_index(key, shape):
    """`A_log = log(1..N)` on every channel, the S4D-real init (modeling_mamba.py:339-341)."""
    del key
    return jnp.broadcast_to(jnp.log(jnp.arange(1, shape[1] + 1, dtype=jnp.float32)), shape)


def inverse_softplus_step(key, shape):
    """Mamba's timestep bias, `init_mamba_weights`' dt_proj bias
    (modeling_mamba.py:349-357) and `init_mamba2_weights`' dt_bias
    (modeling_mamba2.py:434-442) at the reference configs' defaults: a step
    log-uniform between time_step_min 0.001 and time_step_max 0.1, floored at
    time_step_floor 1e-4, then the inverse of softplus so the forward's
    softplus recovers it."""
    step = jnp.exp(jax.random.uniform(key, shape) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    step = jnp.maximum(step, 1e-4)
    return step + jnp.log(-jnp.expm1(-step))


@dataclasses.dataclass(frozen=True)
class MambaMixer(MixerBase):
    """The `mamba` kind, by the reference config's field names
    (configuration_mamba.py). It ignores the context's attention geometry: an
    SSM has no keys to cache or positions to rotate.
    """

    residuals = ('context', 'scan', 'conv')

    intermediate_size: int = 1536
    state_size: int = 16
    time_step_rank: int = 48
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    chunk_size: int = CHUNK_SIZE

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("mamba requires causal=True; its recurrence has no bidirectional mode")
        return functools.partial(
            Mamba,
            emb_features=ctx.emb_features,
            intermediate_size=self.intermediate_size,
            state_size=self.state_size,
            time_step_rank=self.time_step_rank,
            conv_kernel=self.conv_kernel,
            use_bias=self.use_bias,
            use_conv_bias=self.use_conv_bias,
            chunk_size=self.chunk_size,
            init_std=ctx.init_std,
            output_init_std=ctx.output_init_std,
            dtype=ctx.dtype,
            precision=ctx.precision)
