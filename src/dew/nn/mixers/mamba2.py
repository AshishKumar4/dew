"""Mamba-2: the structured state-space duality mixer (Dao & Gu 2024).

An SSD layer is a linear recurrence over a `[P, N]` state per head: with a
scalar decay `a_t = exp(dt_t * A)` per head and step, the state reads

    S_t <- a_t S_t-1 + dt_t x_t B_t^T          y_t = S_t C_t + D x_t

where `x_t` is `[P]` (the head's channels), `B_t` and `C_t` are `[N]`
(shared by the `H // G` heads of one group) and `dt_t = softplus(dt_raw +
dt_bias)` is the selective step. `A = -exp(A_log)` is one scalar per head,
so the decay is a scalar times the identity, which is what makes the
chunked form a masked matmul: within a chunk the output is the "attention"
`(C B^T * L) x` with `L[i, j] = exp(sum_{j<k<=i} dt_k A)`, and the chunk's
state crosses to the next through one decayed rank-N write.

dew computes transformers 5.16.1's `mamba2_chunk_scan`
(modeling_mamba2.py:254-357) in the same order: `segment_sum` as its masked
cumulative sum (73-90), the intra-chunk `Y_diag`, the per-chunk states, the
inter-chunk recurrence (as a scan over the chunks where the reference
materialises the `[NC+1, NC+1]` decay matrix; the two are the same
products) and `Y_off`, the `D` skip added in fp32, and the whole scan in
fp32 as the reference casts. The decode step is `mamba2_selective_state_update`
(192-251). Both live here beside their gated-delta-rule counterparts'
shapes so tests/test_mamba2.py can hold them to the reference's numbers.

The scan over the chunks is `xla_chunk_scan` here, and on a GPU or a TPU
whose tiling covers the geometry it is `dew.nn.kernels.ssd`'s Pallas kernel,
chosen at trace time from the backend the way attention's 'auto' chooses
cudnn. The two compute the same scan: the XLA form is the oracle the kernel's
tests hold it to and the path every other backend and shape takes. Decoding
stays on the recurrent form either way.

Around the scan: `in_proj` yields `[z, x, B, C, dt]` in that order
(`projected_states.split([intermediate, conv_dim, num_heads])`, 484-486,
with `conv_dim = intermediate + 2 G N`), the depthwise causal conv with its
bias runs over `[x, B, C]` with silu after (496-520), the gated norm
multiplies by `silu(z)` before it normalises over the whole intermediate
width (`MambaRMSNormGated`, 105-121, the `norm_before_gate=False` form),
and `out_proj` brings the width back. Parameter names are the
checkpoint's: in_proj, conv1d/{weight,bias}, A_log, dt_bias, D, norm,
out_proj.

Mamba-2's block has no feed-forward (`Mamba2Block`, 608-632: norm, mixer,
residual), which `CausalTransformer` spells as an `mlp_features` of 0.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.inputs import AttentionMetadata
from dew.nn.kernels.ssd import ssd_chunk_scan, ssd_kernel_platform
from dew.nn.linear import _masked_conv1d, causal_conv1d
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.sharding import logical_axes

CHUNK_SIZE = 256
"""The reference's default `chunk_size` (configuration_mamba2.py). The
chunked and the recurrent form agree at any size."""


def segment_sum(x):
    """`segment_sum` (modeling_mamba2.py:73-90): over the last axis of `x`,
    `out[..., i, j] = sum_{j < k <= i} x[..., k]` on and below the diagonal,
    `-inf` above, as the reference's masked cumulative sum rather than a
    difference of cumulative sums, so exp of it is the same numbers."""
    size = x.shape[-1]
    strict = jnp.tril(jnp.ones((size, size), jnp.bool_), -1)
    expanded = jnp.where(strict, jnp.broadcast_to(x[..., :, None], (*x.shape, size)), 0.0)
    summed = jnp.cumsum(expanded, axis=-2)
    inclusive = jnp.tril(jnp.ones((size, size), jnp.bool_))
    return jnp.where(inclusive, summed, -jnp.inf)


def _expand_groups(x, num_heads: int):
    """`[B, S, G, N] -> [B, S, H, N]`, the reference's `repeat_interleave`:
    the `H // G` heads of one group read the same B and C."""
    return jnp.repeat(x, num_heads // x.shape[2], axis=2)


def xla_chunk_scan(x_c, B_c, C_c, a_c, carried):
    """The scan over chunks in plain XLA, the oracle and the fallback for
    `dew.nn.kernels.ssd`. `x_c` `[NC, B, C, H, P]` already scaled by `dt`,
    `B_c` and `C_c` `[NC, B, C, H, N]` with the groups expanded, `a_c`
    `[NC, B, H, C]` the per-step `A dt`, `carried` `[B, H, P, N]` the state
    entering the sequence; returns the output in `x_c`'s layout and the state
    leaving it."""
    a_cumsum = jnp.cumsum(a_c, axis=-1)
    # 1. The intra-chunk output: L[i, j] = exp(segsum) masks C_i . B_j.
    L = jnp.exp(segment_sum(a_c))                           # [NC, B, H, C, C]
    G = jnp.einsum('nbihs,nbjhs->nbhij', C_c, B_c)
    y_diag = jnp.einsum('nbhij,nbjhp->nbihp', G * L, x_c)
    # 2. Each chunk's own write into the state, decayed to the chunk's end.
    decay_states = jnp.exp(a_cumsum[..., -1:] - a_cumsum)   # [NC, B, H, C]
    states = jnp.einsum('nbjhs,nbhj,nbjhp->nbhps', B_c, decay_states, x_c)

    # 3. The state crossing the chunks: the reference's `decay_chunk @ states`
    # over the padded [NC+1, NC+1] segment sums is this product carried
    # chunk by chunk.
    def one_chunk(previous, step):
        step_states, chunk_decay = step
        return jnp.exp(chunk_decay)[..., None, None] * previous + step_states, previous

    final, previous = jax.lax.scan(one_chunk, carried, (states, a_cumsum[..., -1]))
    # 4. The state each chunk reads, decayed to every position of it.
    y_off = jnp.einsum('nbihs,nbhps,nbhi->nbihp', C_c, previous, jnp.exp(a_cumsum))
    return y_diag + y_off, final


def chunk_ssd(x, dt, A, B, C, D, state=None, chunk_size: int = CHUNK_SIZE):
    """The chunked SSD scan, `mamba2_chunk_scan` (modeling_mamba2.py:254-357).

    `x` `[B, S, H, P]`, `dt` `[B, S, H]` already through softplus and the
    limit, `A` `[H]`, `B` and `C` `[B, S, G, N]`, `D` `[H]`; returns
    `(output [B, S, H, P], state [B, H, P, N])`, fp32 throughout as the
    reference, cast back to the input's dtype.

    The scan over chunks runs on the Pallas kernel where `ssd_kernel_platform`
    takes the backend and the geometry, and on `xla_chunk_scan` everywhere
    else. The two agree to fp32 tolerance (tests/test_ssd_kernel.py).
    """
    dtype = x.dtype
    x, dt, A, B, C, D = (jnp.asarray(t, jnp.float32) for t in (x, dt, A, B, C, D))
    batch, length, heads, head_dim = x.shape
    state_size = B.shape[-1]
    B, C = _expand_groups(B, heads), _expand_groups(C, heads)
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    total = length + pad
    skip = D[None, None, :, None] * x                      # the reference's D_residual
    x = x * dt[..., None]                                   # discretised input
    a = A[None, None, :] * dt                               # [B, T, H] log decay per step

    def chunks(t):  # [B, T, ...] -> [NC, B, C, ...], chunks leading for the scan
        blocked = t.reshape(batch, total // chunk_size, chunk_size, *t.shape[2:])
        return jnp.moveaxis(blocked, 1, 0)

    x_c, B_c, C_c = chunks(x), chunks(B), chunks(C)         # [NC, B, C, H, ...]
    a_c = jnp.moveaxis(chunks(a), 3, 2)                     # [NC, B, H, C]
    carried = (jnp.zeros((batch, heads, head_dim, state_size), jnp.float32) if state is None
               else jnp.asarray(state, jnp.float32))
    platform = ssd_kernel_platform(chunk_size, head_dim, state_size)
    scanned, final = (xla_chunk_scan(x_c, B_c, C_c, a_c, carried) if platform is None else
                      ssd_chunk_scan(x_c, B_c, C_c, a_c, carried, platform))
    output = jnp.moveaxis(scanned, 0, 1).reshape(batch, total, heads, head_dim) + skip
    return output[:, :length].astype(dtype), final.astype(dtype)


def recurrent_ssd(x, dt, A, B, C, D, state=None):
    """One token at a time, `mamba2_selective_state_update`
    (modeling_mamba2.py:192-251) as a scan over time, in fp32. Operands as
    `chunk_ssd` takes them."""
    dtype = x.dtype
    x, dt, A, B, C, D = (jnp.asarray(t, jnp.float32) for t in (x, dt, A, B, C, D))
    batch, _, heads, head_dim = x.shape
    B, C = _expand_groups(B, heads), _expand_groups(C, heads)
    if state is None:
        state = jnp.zeros((batch, heads, head_dim, B.shape[-1]), jnp.float32)

    def one_token(s, step):
        x_t, dt_t, B_t, C_t = step                          # [B, H, P], [B, H], [B, H, N]
        s = s * jnp.exp(dt_t * A)[..., None, None] + (dt_t[..., None] * x_t)[..., None] * B_t[..., None, :]
        return s, jnp.einsum('bhps,bhs->bhp', s, C_t) + x_t * D[None, :, None]

    final, out = jax.lax.scan(
        one_token, jnp.asarray(state, jnp.float32),
        tuple(jnp.moveaxis(t, 1, 0) for t in (x, dt, B, C)))
    return jnp.moveaxis(out, 0, 1).astype(dtype), final.astype(dtype)


class Conv1dTaps(nn.Module):
    """The depthwise conv's taps `[D, 1, K]` and bias `[D]`, the checkpoint's
    `conv1d.{weight,bias}` (`nn.Conv1d(groups=conv_dim)`, modeling_mamba2.py:392-399)."""

    features: int
    kernel: int = 4
    use_bias: bool = True

    @nn.compact
    def __call__(self):
        weight = self.param('weight', nn.initializers.lecun_normal(), (self.features, 1, self.kernel))
        bias = (self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
                if self.use_bias else None)
        return weight, bias


class MambaRMSNormGated(nn.Module):
    """`MambaRMSNormGated` (modeling_mamba2.py:105-121): the gate first, the
    norm over the whole width after, both in fp32, the weight applied to the
    cast-back values."""

    epsilon: float = 1e-5
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x, gate):
        dtype = self.dtype if self.dtype is not None else x.dtype
        y = x.astype(jnp.float32) * nn.silu(gate.astype(jnp.float32))
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + self.epsilon)
        scale = self.param('weight', nn.initializers.ones, (x.shape[-1],), jnp.float32)
        return (scale.astype(dtype) * y.astype(dtype)).astype(dtype)


@logical_axes({
    ("in_proj",): ("embed", "linear"),
    ("out_proj",): ("attention", "embed"),
}, heuristic=(("conv1d",),))
class Mamba2(nn.Module):
    """The token mixer of a Mamba-2 layer, `Mamba2Mixer` (modeling_mamba2.py:360-588).

    `intermediate = num_heads * head_dim` is the layer's own width (the
    config's `expand * hidden_size`), `state_size` the `N` of each head's
    `[head_dim, N]` state and `n_groups` how many distinct B and C the heads
    share. `time_step_limit` clamps the softplus'd step, the reference's
    `dt_limit`; its default is no clamp.

    The decode state is two leaves in the flax `cache` collection,
    `ssm_state` `[B, H, P, N]` and `conv_state` `[B, conv_dim, K-1]`,
    allocated at the batch the first decode-mode call sees and advanced by
    every later one, as the gated delta net holds its own. Prefill and
    decode share one path: a call of one token runs the recurrent step, a
    longer one the chunked scan with the held state as its initial state.

    Padded rows (`attention_metadata.valid`) are zeroed before the
    projection as the reference's `apply_mask_to_padding_states` does; the
    conv reads a row's real tokens as a stream across them, and a padded
    slot takes a zero step (`dt = 0`), which leaves the state as it was.
    """

    emb_features: int
    num_heads: int
    head_dim: int
    state_size: int = 128
    n_groups: int = 1
    conv_kernel: int = 4
    chunk_size: int = CHUNK_SIZE
    use_bias: bool = False
    use_conv_bias: bool = True
    time_step_limit: tuple[float, float] = (0.0, float('inf'))
    norm_eps: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_features(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size

    def setup(self):
        if self.num_heads % self.n_groups:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of n_groups ({self.n_groups})")
        if self.conv_kernel < 2:
            raise ValueError(
                f"the causal conv needs a history, so a kernel of at least 2, got {self.conv_kernel}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size counts tokens per chunk, got {self.chunk_size}")
        dense = functools.partial(nn.Dense, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision)
        self.in_proj = dense(2 * self.intermediate_size + 2 * self.n_groups * self.state_size + self.num_heads,
                             name='in_proj')
        self.conv1d = Conv1dTaps(features=self.conv_features, kernel=self.conv_kernel,
                                 use_bias=self.use_conv_bias, name='conv1d')
        # The reference's init: A_log = log(1..H), dt_bias the inverse
        # softplus of a step drawn between time_step_min and max, D ones
        # (init_mamba2_weights, modeling_mamba2.py:428-442).
        self.A_log = self.param('A_log', _log_head_index, (self.num_heads,))
        self.dt_bias = self.param('dt_bias', _inverse_softplus_step, (self.num_heads,))
        self.D = self.param('D', nn.initializers.ones, (self.num_heads,), jnp.float32)
        self.norm = MambaRMSNormGated(epsilon=self.norm_eps, dtype=self.dtype, name='norm')
        self.out_proj = dense(self.emb_features, name='out_proj')

    def _conv(self, mixed, valid, conv_state):
        """The conv over `[B, D, S]` with the bias before silu, from the
        held history when decoding, and the history it leaves."""
        weight, bias = self.conv1d()
        taps = jnp.asarray(weight[:, 0, :], jnp.float32)
        bias = None if bias is None else jnp.asarray(bias, jnp.float32)
        if valid is not None:
            return _masked_conv1d(mixed, taps, valid, conv_state, bias=bias)
        length = mixed.shape[-1]
        history = mixed if conv_state is None else jnp.concatenate([conv_state, mixed], axis=2)
        out = causal_conv1d(history, taps, bias=bias)[..., -length:]
        return out, history[:, :, -(self.conv_kernel - 1):]

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        del positions, segment_ids, kv_store
        batch, length, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if valid is not None and valid.shape != (batch, length):
            raise ValueError(f"row validity must be {(batch, length)}, got {valid.shape}")
        if valid is not None:
            x = jnp.where(valid[:, :, None], x, 0)
        projected = self.in_proj(x)
        gate, mixed, dt = jnp.split(
            projected, [self.intermediate_size, self.intermediate_size + self.conv_features], axis=-1)
        # The conv and the scan run in fp32, as the reference's torch path
        # casts them (modeling_mamba2.py:271, 305).
        mixed = jnp.moveaxis(mixed.astype(jnp.float32), 2, 1)   # [B, D, S]
        ssm = None
        conv_state = None
        if decode:
            allocated = self.has_variable('cache', 'ssm_state')
            conv_state = self.variable('cache', 'conv_state', jnp.zeros,
                                       (batch, self.conv_features, self.conv_kernel - 1), jnp.float32)
            ssm = self.variable('cache', 'ssm_state', jnp.zeros,
                                (batch, self.num_heads, self.head_dim, self.state_size), jnp.float32)
            if not allocated:
                # Allocation only, as the delta net: init_cache's dummy token
                # must not consume a position or leave state behind.
                return self.out_proj(jnp.zeros((batch, length, self.intermediate_size), self.dtype))
        mixed, history = self._conv(mixed, valid, None if conv_state is None else conv_state.value)
        if conv_state is not None:
            conv_state.value = history
        mixed = jnp.moveaxis(mixed, 2, 1)
        xs, B, C = jnp.split(
            mixed, [self.intermediate_size, self.intermediate_size + self.n_groups * self.state_size], axis=-1)
        xs = xs.reshape(batch, length, self.num_heads, self.head_dim)
        B = B.reshape(batch, length, self.n_groups, self.state_size)
        C = C.reshape(batch, length, self.n_groups, self.state_size)
        step = nn.softplus(dt.astype(jnp.float32) + self.dt_bias.astype(jnp.float32))
        step = jnp.clip(step, *self.time_step_limit)
        if valid is not None:
            # A zero step neither decays nor writes the state.
            step = jnp.where(valid[:, :, None], step, 0.0)
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        held = None if ssm is None else ssm.value
        out, final = (recurrent_ssd(xs, step, A, B, C, self.D, held) if length == 1 else
                      chunk_ssd(xs, step, A, B, C, self.D, held, self.chunk_size))
        if ssm is not None:
            ssm.value = final
        out = out.reshape(batch, length, self.intermediate_size)
        if valid is not None:
            out = jnp.where(valid[:, :, None], out, 0)
        return self.out_proj(self.norm(out, gate))


def _log_head_index(key, shape):
    """`A_log = log(1..H)`, the S4D-real init (modeling_mamba2.py:430-431)."""
    del key
    return jnp.log(jnp.arange(1, shape[0] + 1, dtype=jnp.float32))


def _inverse_softplus_step(key, shape, minimum: float = 0.001, maximum: float = 0.1, floor: float = 1e-4):
    """`init_mamba2_weights`' dt_bias (modeling_mamba2.py:434-442): a step
    log-uniform between the config's time_step_min and max, floored, then
    the inverse of softplus so the forward's softplus recovers it."""
    step = jnp.exp(jax.random.uniform(key, shape) * (jnp.log(maximum) - jnp.log(minimum)) + jnp.log(minimum))
    step = jnp.maximum(step, floor)
    return step + jnp.log(-jnp.expm1(-step))


@mixers("mamba2")
@dataclasses.dataclass(frozen=True)
class Mamba2Mixer(MixerBase):
    """The `mamba2` kind, by the reference config's field names
    (configuration_mamba2.py): `num_heads` heads of `head_dim`, a state of
    `state_size` per head, `n_groups` distinct B/C, the conv's `conv_kernel`,
    the scan's `chunk_size`, the biases and the step's `time_step_limit`.

    The kind ignores the context's attention geometry (num_kv_heads,
    head_dim, the window, rope, KV sharing): an SSM has no keys to cache and
    no positions to rotate. Its own `num_heads` and `head_dim` are the
    checkpoint's, which need not match the model's attention heads.
    """

    num_heads: int = 128
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    chunk_size: int = CHUNK_SIZE
    use_bias: bool = False
    use_conv_bias: bool = True
    time_step_limit: tuple[float, float] = (0.0, float('inf'))

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("mamba2 requires causal=True; its recurrence has no bidirectional mode")
        return functools.partial(
            Mamba2,
            emb_features=ctx.emb_features,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.state_size,
            n_groups=self.n_groups,
            conv_kernel=self.conv_kernel,
            chunk_size=self.chunk_size,
            use_bias=self.use_bias,
            use_conv_bias=self.use_conv_bias,
            time_step_limit=(float(self.time_step_limit[0]), float(self.time_step_limit[1])),
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
