"""Mamba-2: the structured state-space duality mixer (Dao & Gu 2024).

An SSD layer is a linear recurrence over a `[P, N]` state per head:

    S_t <- a_t S_t-1 + dt_t x_t B_t^T          y_t = S_t C_t + D x_t

`a_t = exp(dt_t * A)` is a scalar decay per head and step. `x_t` is `[P]`,
the head's channels. `B_t` and `C_t` are `[N]`, shared by the `H // G` heads
of a group. `dt_t = softplus(dt_raw + dt_bias)` is the selective step.

`A = -exp(A_log)` is one scalar per head, so the decay is a scalar times the
identity. That is what makes the chunked form a masked matmul: within a
chunk the output is the "attention" `(C B^T * L) x` with
`L[i, j] = exp(sum_{j<k<=i} dt_k A)`, and the chunk's state crosses to the
next through one decayed rank-N write.

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
from jax.sharding import PartitionSpec as P

from dew.nn.inputs import AttentionMetadata
from dew.nn.kernels.ssd import ssd_chunk_scan, ssd_kernel_platform
from dew.nn.linear import _masked_conv1d, causal_conv1d, document_conv1d, document_starts
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.sharding import SEQUENCE_AXIS, logical_axes, row_axes, sequence_shards

CHUNK_SIZE = 256
"""The reference's default `chunk_size` (configuration_mamba2.py). The
chunked and the recurrent form agree at any size."""

RESET_DECAY = -1e4
"""The log decay a document's first token takes in place of `A dt`: the
state entering it is multiplied by `exp(-1e4)`, which is 0 in fp32. A step's
own `A dt` is never positive, so every span containing a reset sums to at
most this and decays to exactly 0, and a span without one never sees it.
It stays finite because the Pallas kernel forms its segment sums as a
matmul against a 0/1 triangle, where an infinity times 0 is a NaN; and it
stays small enough that thousands of resets in one shard's total decay are
nowhere near fp32's range."""


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


def _chunk_writes(x_c, B_c, a_c):
    """Each chunk's own write into the state, decayed to the chunk's end:
    `sum_j exp(sum_{j<k<C} a_k) x_j B_j^T`, `[NC, B, H, P, N]`.

    The decay from step j to the chunk's end is the suffix sum after j, a
    reverse cumulative sum shifted by one, where the reference takes the
    difference `acs_C-1 - acs_j` of two prefix sums. The two are the same
    number in exact arithmetic; the suffix sum never subtracts, which is
    what keeps a document reset exact (`RESET_DECAY`): a reset before j sits
    in both prefix sums, and their difference would lose every digit below
    its magnitude."""
    suffix = jax.lax.cumsum(a_c, axis=a_c.ndim - 1, reverse=True)
    to_end = jnp.exp(jnp.pad(suffix[..., 1:], [(0, 0)] * (a_c.ndim - 1) + [(0, 1)]))
    return jnp.einsum('nbjhs,nbhj,nbjhp->nbhps', B_c, to_end, x_c)


def _carry(previous, step):
    """One chunk (or one shard) across the state: decay what entered by the
    span's total log decay and add the span's own write."""
    writes, log_decay = step
    return jnp.exp(log_decay)[..., None, None] * previous + writes, previous


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
    states = _chunk_writes(x_c, B_c, a_c)
    # 3. The state crossing the chunks: the reference's `decay_chunk @ states`
    # over the padded [NC+1, NC+1] segment sums is this product carried
    # chunk by chunk.
    final, previous = jax.lax.scan(_carry, carried, (states, a_cumsum[..., -1]))
    # 4. The state each chunk reads, decayed to every position of it.
    y_off = jnp.einsum('nbihs,nbhps,nbhi->nbihp', C_c, previous, jnp.exp(a_cumsum))
    return y_diag + y_off, final


def _entering_state(decay, write, axis: str):
    """The state entering this shard of a sequence the mesh axis `axis`
    splits, inside a `shard_map` that holds it manual.

    Shard r's span of the recurrence is affine in the state entering it:
    it leaves `exp(A_r) h + B_r`, with `decay` the span's total log decay
    `A_r` `[B, H]` and `write` the state `B_r` `[B, H, P, N]` it leaves from
    zero. Two spans in order compose to one, `(A_1, B_1)` then `(A_2, B_2)`
    being `(A_1 + A_2, exp(A_2) B_1 + B_2)`, an associative operation whose
    identity `(0, 0)` is what `ppermute` hands a shard with no source. So
    the prefix over shards is a Kogge-Stone scan: at shift 1, 2, 4, ...
    each shard receives the pair `shift` shards back and composes it in
    front of its own, which leaves every shard the composition of itself
    and all shards before it after `ceil(log2 n)` rounds; one more shift by
    a single shard makes it exclusive, the state entering this shard, zero
    on the first. Each round moves one pair per shard, where gathering all
    of them moves n; lm-engine's `_SerialPrefixScan`
    (`sequence_mixer_blocks/mamba2/op.py`) folds the gathered pairs
    serially. Autodiff transposes each `ppermute` into the reverse one."""
    shards = jax.lax.axis_size(axis)

    def shifted(value, shift: int):
        return jax.lax.ppermute(value, axis, [(r, r + shift) for r in range(shards - shift)])

    shift = 1
    while shift < shards:
        earlier_decay, earlier_write = shifted(decay, shift), shifted(write, shift)
        write = jnp.exp(decay)[..., None, None] * earlier_write + write
        decay = earlier_decay + decay
        shift *= 2
    return shifted(write, 1)


def chunk_ssd(x, dt, A, B, C, D, state=None, chunk_size: int = CHUNK_SIZE, starts=None):
    """The chunked SSD scan, `mamba2_chunk_scan` (modeling_mamba2.py:254-357).

    `x` `[B, S, H, P]`, `dt` `[B, S, H]` already through softplus and the
    limit, `A` `[H]`, `B` and `C` `[B, S, G, N]`, `D` `[H]`; returns
    `(output [B, S, H, P], state [B, H, P, N])`, fp32 throughout as the
    reference, cast back to the input's dtype.

    `starts` `[B, S]` marks the tokens that open a packed document: the
    state entering such a token is dropped, so no document reads another's.

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
    if starts is not None:
        a = jnp.where(jnp.pad(starts, ((0, 0), (0, pad)))[..., None], RESET_DECAY, a)

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


def recurrent_ssd(x, dt, A, B, C, D, state=None, starts=None):
    """One token at a time, `mamba2_selective_state_update`
    (modeling_mamba2.py:192-251) as a scan over time, in fp32. Operands as
    `chunk_ssd` takes them; a token `starts` marks sees a zero state."""
    dtype = x.dtype
    x, dt, A, B, C, D = (jnp.asarray(t, jnp.float32) for t in (x, dt, A, B, C, D))
    batch, length, heads, head_dim = x.shape
    B, C = _expand_groups(B, heads), _expand_groups(C, heads)
    if state is None:
        state = jnp.zeros((batch, heads, head_dim, B.shape[-1]), jnp.float32)
    if starts is None:
        starts = jnp.zeros((batch, length), jnp.bool_)

    def one_token(s, step):
        x_t, dt_t, B_t, C_t, start = step                   # [B, H, P], [B, H], [B, H, N], [B]
        decay = jnp.where(start[:, None], 0.0, jnp.exp(dt_t * A))
        s = s * decay[..., None, None] + (dt_t[..., None] * x_t)[..., None] * B_t[..., None, :]
        return s, jnp.einsum('bhps,bhs->bhp', s, C_t) + x_t * D[None, :, None]

    final, out = jax.lax.scan(
        one_token, jnp.asarray(state, jnp.float32),
        tuple(jnp.moveaxis(t, 1, 0) for t in (x, dt, B, C, starts)))
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

    Packed documents (`segment_ids`) run as if each ran alone: the conv
    reads nothing across a segment change and the scan drops the state at
    it, as mamba_ssm does for its `seq_idx`. Without segment ids the state
    and the conv run across the whole row.

    Under a mesh whose sequence axis is above one, a training or scoring
    call runs the conv and the scan inside a `shard_map` over that axis
    (`_sequence_mix`): each shard's conv reads the previous shard's last
    `K-1` tokens through a `ppermute`, and its scan runs from zero and
    then adds what the state the earlier shards leave (`_entering_state`)
    contributes to each output, packed documents resetting both across
    shard boundaries as within one. Decoding and rows with
    padding slots hold one state per row and are refused there.
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

    def _conv(self, mixed, taps, bias, valid, conv_state, segments):
        """The conv over `[B, D, S]` with the bias before silu, from the
        held history when decoding, and the history it leaves."""
        if valid is not None or segments is not None:
            valid = jnp.ones(mixed.shape[::2], bool) if valid is None else valid
            return _masked_conv1d(mixed, taps, valid, conv_state, bias=bias, segments=segments)
        length = mixed.shape[-1]
        history = mixed if conv_state is None else jnp.concatenate([conv_state, mixed], axis=2)
        out = causal_conv1d(history, taps, bias=bias)[..., -length:]
        return out, history[:, :, -(self.conv_kernel - 1):]

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        del positions, kv_store
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
        mixed, dt = mixed.astype(jnp.float32), dt.astype(jnp.float32)
        weight, bias = self.conv1d()
        taps = jnp.asarray(weight[:, 0, :], jnp.float32)
        bias = None if bias is None else jnp.asarray(bias, jnp.float32)
        heads = (jnp.asarray(self.dt_bias, jnp.float32), -jnp.exp(self.A_log.astype(jnp.float32)),
                 jnp.asarray(self.D, jnp.float32))
        scan = functools.partial(
            _ssd, num_heads=self.num_heads, head_dim=self.head_dim, n_groups=self.n_groups,
            state_size=self.state_size, chunk_size=self.chunk_size,
            time_step_limit=self.time_step_limit)
        shards = sequence_shards()
        if decode or valid is not None:
            if shards > 1:
                raise ValueError(
                    f"mamba2 under a sequence axis of {shards} runs whole training sequences; "
                    "decoding and rows with padding slots (attention_metadata.valid) hold one "
                    "state per row. Run them on a mesh with sequence=1.")
            out = self._stateful(mixed, dt, taps, bias, heads, scan, decode, valid, segment_ids)
            if out is None:
                # Allocation only, as the delta net: init_cache's dummy token
                # must not consume a position or leave state behind.
                return self.out_proj(jnp.zeros((batch, length, self.intermediate_size), self.dtype))
        elif shards > 1:
            out = _over_sequence(functools.partial(_sequence_mix, scan=scan, axis=SEQUENCE_AXIS),
                                 shards, mixed, dt, segment_ids, (taps, bias, *heads))
        else:
            out = _sequence_mix(mixed, dt, segment_ids, (taps, bias, *heads), scan=scan)
        return self.out_proj(self.norm(out, gate))

    def _stateful(self, mixed, dt, taps, bias, heads, scan, decode: bool, valid, segments):
        """The scan of a call that holds per-row state: a decode step or
        prefill advancing the flax cache, or rows with padding slots. None
        for the allocation-only decode call. Packed `segments` reset the conv
        and the state at each document's first real token; the held state
        counts as the call's first document, which it continues."""
        batch = mixed.shape[0]
        mixed = jnp.moveaxis(mixed, 2, 1)                   # [B, D, S]
        ssm = conv_state = None
        if decode:
            allocated = self.has_variable('cache', 'ssm_state')
            conv_state = self.variable('cache', 'conv_state', jnp.zeros,
                                       (batch, self.conv_features, self.conv_kernel - 1), jnp.float32)
            ssm = self.variable('cache', 'ssm_state', jnp.zeros,
                                (batch, self.num_heads, self.head_dim, self.state_size), jnp.float32)
            if not allocated:
                return None
        mixed, history = self._conv(mixed, taps, bias, valid,
                                    None if conv_state is None else conv_state.value, segments)
        if conv_state is not None:
            conv_state.value = history
        starts = None if segments is None else document_starts(segments, valid=valid)
        out, final = scan(mixed, dt, *heads, starts=starts, valid=valid,
                          held=None if ssm is None else ssm.value)
        if ssm is not None:
            ssm.value = final
        if valid is not None:
            out = jnp.where(valid[:, :, None], out, 0)
        return out


def _ssd(convolved, dt, dt_bias, A, D, *, num_heads: int, head_dim: int, n_groups: int,
         state_size: int, chunk_size: int, time_step_limit, starts=None, valid=None, held=None,
         axis: str | None = None):
    """The scan after the conv: `convolved` `[B, conv_dim, S]` split into
    `x`, `B` and `C`, the step through softplus with its bias and the limit,
    then the recurrent step for one token and the chunked scan otherwise.
    Returns the output `[B, S, H * P]` and the state leaving it.

    Under `axis`, this shard's slice of a sequence inside a `shard_map`,
    the scan runs from zero and the output is then corrected for the state
    `h` the earlier shards leave, which the recurrence carries linearly: the
    state at token t is its zero-start value plus `exp(acs_t) h`, with
    `acs_t` the inclusive sum of this shard's `A dt` (a document start's
    `RESET_DECAY` in it takes the term to 0), so the output gains
    `C_t . exp(acs_t) h`. The zero-start scan's final state is the shard's
    own write `B_r` the exchange needs (`_entering_state`); the state
    returned is that zero-start one."""
    batch, _, length = convolved.shape
    mixed = jnp.moveaxis(convolved, 2, 1)
    intermediate = num_heads * head_dim
    xs, B, C = jnp.split(mixed, [intermediate, intermediate + n_groups * state_size], axis=-1)
    xs = xs.reshape(batch, length, num_heads, head_dim)
    B = B.reshape(batch, length, n_groups, state_size)
    C = C.reshape(batch, length, n_groups, state_size)
    step = jnp.clip(nn.softplus(dt + dt_bias), *time_step_limit)
    if valid is not None:
        # A zero step neither decays nor writes the state.
        step = jnp.where(valid[:, :, None], step, 0.0)
    if length == 1 and axis is None:
        out, final = recurrent_ssd(xs, step, A, B, C, D, held, starts)
    else:
        out, final = chunk_ssd(xs, step, A, B, C, D, held, chunk_size, starts=starts)
    if axis is not None:
        log_decay = A * step                                # [B, S, H]
        if starts is not None:
            log_decay = jnp.where(starts[..., None], RESET_DECAY, log_decay)
        entering = _entering_state(jnp.sum(log_decay, axis=1), final, axis)
        entering = entering.reshape(batch, n_groups, num_heads // n_groups, head_dim, state_size)
        decayed = jnp.exp(jnp.cumsum(log_decay, axis=1)).reshape(
            batch, length, n_groups, num_heads // n_groups)
        carried = jnp.einsum('bsgn,bgkpn,bsgk->bsgkp', C, entering, decayed)
        out = out + carried.reshape(batch, length, num_heads, head_dim)
    return out.reshape(batch, length, intermediate), final


def _sequence_mix(mixed, dt, segments, weights, *, scan, axis: str | None = None):
    """The conv and the scan over whole training sequences, or over this
    shard's slice of them inside a `shard_map` that holds `axis` manual.

    `mixed` `[B, S, conv_dim]` and `dt` `[B, S, H]` in fp32, `segments`
    `[B, S]` the packed documents or None, `weights` the conv's taps and
    bias and the heads' `dt_bias`, `A` and `D`. Under `axis` the conv reads
    the previous shard's last `K-1` tokens, and their segments, through one
    `ppermute` (the first shard receives zeros, the history a sequence
    starts from), and the scan's output is corrected for the state the
    earlier shards leave (`_ssd`). A packed document resets both: the conv reads nothing across a
    segment change and the scan drops the state at it, wherever the change
    falls, a shard boundary included."""
    taps, bias, dt_bias, A, D = weights
    width = taps.shape[1] - 1
    mixed = jnp.moveaxis(mixed, 2, 1)                       # [B, D, S]
    batch, channels, length = mixed.shape
    history = jnp.zeros((batch, channels, width), mixed.dtype)
    before = None if segments is None else jnp.broadcast_to(segments[:, :1], (batch, width))
    if axis is not None:
        if length < width:
            raise ValueError(
                f"each sequence shard holds {length} tokens, fewer than the {width} the "
                f"conv reads behind its first; use fewer sequence shards or longer sequences")
        shards = jax.lax.axis_size(axis)
        forward = [(shard, shard + 1) for shard in range(shards - 1)]
        history = jax.lax.ppermute(mixed[..., length - width:], axis, forward)
        if segments is not None:
            before = jax.lax.ppermute(segments[:, length - width:], axis, forward)
    if segments is None:
        convolved = causal_conv1d(jnp.concatenate([history, mixed], axis=2), taps, bias=bias)[..., width:]
        starts = None
    else:
        convolved = document_conv1d(history, mixed, before, segments, taps, bias)
        starts = document_starts(segments, before[:, -1])
    out, _ = scan(convolved, dt, dt_bias, A, D, starts=starts, axis=axis)
    return out


def _over_sequence(mix, shards: int, mixed, dt, segments, weights):
    """Run `mix` (`_sequence_mix` under the sequence axis) in a `shard_map`
    that splits the sequence over the mesh's sequence axis and the rows over
    `row_axes`, with the weights whole on every shard. The layer's
    projections and its gated norm act token by token and stay with GSPMD
    outside it. Every other axis goes manual too, replicated, as
    `exchanged_heads_attention` takes them and for its reasons: the SSD's
    Pallas kernel refuses to lower with an axis left to the partitioner,
    and the pipeline vmaps its stages over the stage axis."""
    batch, length, _ = mixed.shape
    if length % shards:
        raise ValueError(
            f"mamba2 splits the sequence of {length} tokens {shards} ways over the "
            "mesh's sequence axis, and it does not divide")
    mesh = jax.sharding.get_abstract_mesh()
    tokens = P(row_axes(batch) or None, SEQUENCE_AXIS)
    manual = {axis for axis in mesh.axis_names if axis not in mesh.manual_axes}
    # Pallas kernels state no varying-manual-axes type for their outputs,
    # as the attention exchange notes, so the check is off here too.
    split = jax.shard_map(mix, in_specs=(tokens, tokens, tokens, P()), out_specs=tokens,
                          axis_names=manual, check_vma=False)
    return split(mixed, dt, segments, weights)


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
