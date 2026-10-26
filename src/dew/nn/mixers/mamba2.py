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

dew computes transformers 5.16.1's `mamba2_chunk_scan` (modeling_mamba2.py:254-357)
as the recurrence template's chunked op (`chunk_ssd`, `recurrence_chunk.chunk`): no
delta rule, a rated decay `A dt`, the `D` skip added in fp32, and the whole scan
in at least fp32 as the reference casts its operands. The decode step is
`mamba2_selective_state_update` (192-251, `recurrent_ssd`). tests/test_mamba2.py
holds both to the reference's numbers.

On a GPU whose target row names it the chunked scan is mamba_ssm's own
kernels (`dew.nn.kernels.mamba_ssd`, `recurrence_chunk.SCAN` names where not),
which multiply `x`, `B` and `C` in their own dtype into fp32 sums as
mamba_ssm does; on a TPU whose tiling covers the geometry `dew.nn.kernels.ssd`'s
Pallas kernel; elsewhere the template's portable math, chosen at trace time.
Decoding stays on the recurrent form
either way. The conv sums in at least fp32 and hands the scan its output in
the model's compute dtype, as mamba_ssm's `causal_conv1d_fn` hands its scan
(`mamba_split_conv1d_scan_combined`); the step, the decay and the states
stay in at least fp32.

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
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from dew.mesh import SEQUENCE_AXIS, LayoutRefused, logical_spec
from dew.nn.attention import repeat_kv_heads, unweighted_rmsnorm
from dew.nn.blocks import normal_kernel
from dew.nn.dense import Dense
from dew.nn.flops import conv_flops, projection_flops, ssd_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.kernels.generation import measured_kernel
from dew.nn.kernels.recurrence import Hooks
from dew.nn.kernels.recurrence_chunk import RESET_DECAY, Chunk, chunk
from dew.nn.linear import DepthwiseConv1d, causal_conv1d, held_conv1d, held_scan, real_tail
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.mixers.mamba import inverse_softplus_step
from dew.nn.precision import at_least_fp32, float32_dot_general, float32_einsum
from dew.nn.sharding import logical_axes, manual_map, sequence_shards

CHUNK_SIZE = 256
"""The reference's default `chunk_size` (configuration_mamba2.py). The
chunked and the recurrent form agree at any size."""

def _entering_state(decay, write, axis: str):
    """The state entering this shard of a sequence the mesh axis `axis`
    splits, inside a `shard_map` that holds it manual.

    Shard r's span is affine in its entering state: it leaves `exp(A_r) h +
    B_r`, with `decay` `A_r` `[B, H]` and `write` `B_r` `[B, H, P, N]`. Spans
    compose as `(A_1 + A_2, exp(A_2) B_1 + B_2)`, associative with identity
    `(0, 0)`, what `ppermute` hands a shard with no source. So the prefix is a
    Kogge-Stone scan, shifts 1, 2, 4, ... for `ceil(log2 n)` rounds and one more
    to make it exclusive (zero on the first shard), one pair per shard per
    round where lm-engine's `_SerialPrefixScan` gathers all n. Autodiff
    transposes each `ppermute`."""
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
    """The chunked SSD scan, `mamba2_chunk_scan` (modeling_mamba2.py:254-357), as the recurrence
    template's chunked op (`recurrence_chunk.chunk`): no delta rule, a rated decay `A dt` with `dt`
    the write gate, `x` the values, `B` the keys, `C` the queries and the skip `D x`.

    `x` `[B, S, H, P]`, `dt` `[B, S, H]` through softplus and the limit, `A` `[H]`, `B` and `C`
    `[B, S, G, N]`, `D` `[H]`; returns `(output [B, S, H, P], state [B, H, P, N])` in
    `at_least_fp32` of the operands' common dtype, the output unrounded for what a caller adds to it
    (the earlier shards' state, `_ssd`). `starts` `[B, S]` marks packed documents' first tokens,
    whose entering state drops. The op runs it on mamba_ssm's kernels where their port takes the
    call (`recurrence_chunk.SCAN`), on `x`, `B` and `C` in their common dtype; else in
    `at_least_fp32` of it, on the TPU's Pallas kernel where it takes the geometry, else the portable
    math.
    """
    operand = jnp.result_type(x, B, C)
    work = at_least_fp32(operand)
    x, B, C = (jnp.asarray(t, operand) for t in (x, B, C))
    dt, A, D = (jnp.asarray(t, work) for t in (dt, A, D))
    held = None if state is None else jnp.swapaxes(jnp.asarray(state, work), -1, -2)
    out, final = chunk(SSD, Chunk(C, B, x, beta=dt, rate=A, skip=D, state=held, starts=starts),
                       chunk_size=chunk_size, dtype=work)
    return out, jnp.swapaxes(final, -1, -2)


SSD = Hooks(decay='scalar', rated=True, delta=False, skip=True)
"""The SSD as the recurrence template's form: a rated decay a head, no delta rule, the skip."""


def recurrent_ssd(x, dt, A, B, C, D, state=None, starts=None):
    """One token at a time, `mamba2_selective_state_update`
    (modeling_mamba2.py:192-251) as a scan over time, in fp32 (the input's
    own dtype where it is wider). Operands and results as `chunk_ssd`'s; a
    token `starts` marks sees a zero state."""
    work = at_least_fp32(jnp.result_type(x, B, C))
    x, dt, A, B, C, D = (jnp.asarray(t, work) for t in (x, dt, A, B, C, D))
    batch, length, heads, head_dim = x.shape
    # The `H // G` heads of one group read the same B and C, the reference's `repeat_interleave`.
    B, C = repeat_kv_heads(B, heads), repeat_kv_heads(C, heads)
    if state is None:
        state = jnp.zeros((batch, heads, head_dim, B.shape[-1]), work)
    if starts is None:
        starts = jnp.zeros((batch, length), jnp.bool_)

    def one_token(s, step):
        x_t, dt_t, B_t, C_t, start = step                   # [B, H, P], [B, H], [B, H, N], [B]
        decay = jnp.where(start[:, None], 0.0, jnp.exp(dt_t * A))
        s = s * decay[..., None, None] + (dt_t[..., None] * x_t)[..., None] * B_t[..., None, :]
        read = float32_einsum('bhps,bhs->bhp', s, C_t)
        return s, read + x_t * D[None, :, None]

    final, out = jax.lax.scan(
        one_token, jnp.asarray(state, work),
        tuple(jnp.moveaxis(t, 1, 0) for t in (x, dt, B, C, starts)))
    return jnp.moveaxis(out, 0, 1), final


class MambaRMSNormGated(nn.Module):
    """`MambaRMSNormGated` (modeling_mamba2.py:105-121): the gate first, the
    norm over the whole width after, both in fp32 (`at_least_fp32`), the weight applied to the
    cast-back values."""

    epsilon: float = 1e-5
    groups: int = 1
    """Nemotron-H normalizes each B/C group's channels separately after the gate."""
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x, gate):
        dtype = self.dtype if self.dtype is not None else x.dtype
        wide = at_least_fp32(x.dtype)
        if self.groups < 1 or x.shape[-1] % self.groups:
            raise ValueError(f"norm groups ({self.groups}) must divide the width ({x.shape[-1]})")
        y = x.astype(wide) * nn.silu(gate.astype(wide))
        y = unweighted_rmsnorm(y.reshape(*y.shape[:-1], self.groups, -1), self.epsilon).reshape(y.shape)
        scale = self.param('weight', nn.initializers.ones, (x.shape[-1],), jnp.float32)
        return (scale.astype(dtype) * y.astype(dtype)).astype(dtype)


@logical_axes({
    ("in_proj",): ("embed", "linear"),
    ("out_proj",): ("attention", "embed"),
})
class Mamba2(nn.Module):
    """The token mixer of a Mamba-2 layer, `Mamba2Mixer` (modeling_mamba2.py:360-588).

    `intermediate = num_heads * head_dim` (the config's `expand *
    hidden_size`), `state_size` each head's `N`, `n_groups` the distinct B and
    C. `time_step_limit` is the reference's `dt_limit`, by default no clamp.

    The decode state is `ssm_state` `[B, H, P, N]` and `conv_state`
    `[B, K-1, conv_dim]` in the `cache` collection; a one-token call runs the
    recurrent step and a longer one the chunked scan from the held state.
    Padded rows (`attention_metadata.valid`) are zeroed before the projection
    (`apply_mask_to_padding_states`) and take a zero step. Packed documents
    (`segment_ids`) run as if alone, as mamba_ssm's `seq_idx` does.

    Under a mesh sequence axis above one, training and scoring run the conv and
    scan in a `shard_map` (`_sequence_mix`): the conv reads the previous shard's
    last `K-1` tokens by `ppermute` and the scan adds the earlier shards' state
    (`_entering_state`); decoding and padded rows are refused there.
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    """None of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token
    of its document, with no mask or bias to narrow it."""

    emb_features: int
    num_heads: int
    head_dim: int
    state_size: int = 128
    n_groups: int = 1
    conv_kernel: int = 4
    chunk_size: int = CHUNK_SIZE
    use_bias: bool = False
    use_conv_bias: bool = True
    norm_groups: int = 1
    time_step_limit: tuple[float, float] = (0.0, float('inf'))
    norm_eps: float = 1e-5
    init_std: float | None = None
    """Normal std of in_proj and the conv taps (lm-engine draws both from the
    same std); None keeps lecun normal."""
    output_init_std: float | None = None
    """Normal std of out_proj; None follows init_std."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Projected SSD, conv and its recurrent state write/read, independent of chunk size."""
        projected = 3 * self.intermediate_size + 2 * self.n_groups * self.state_size + self.num_heads
        return (projection_flops(lengths, self.emb_features, projected)
                + conv_flops(lengths, self.conv_features, self.conv_features,
                             self.conv_kernel, groups=self.conv_features)
                + ssd_flops(lengths, self.num_heads, self.state_size, self.head_dim))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Frozen learned maps and conv, excluding the scan's state contractions (`ssd_flops`)."""
        return self.model_flops(lengths) - ssd_flops(lengths, self.num_heads, self.state_size, self.head_dim)

    @property
    def conv_features(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size

    def setup(self):
        if self.num_heads % self.n_groups:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of n_groups ({self.n_groups})")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size counts tokens per chunk, got {self.chunk_size}")
        dense = functools.partial(
            Dense, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
            dot_general=float32_dot_general
        )
        self.in_proj = dense(
            2 * self.intermediate_size + 2 * self.n_groups * self.state_size + self.num_heads,
            name="in_proj",
            **normal_kernel(self.init_std),
        )
        self.conv1d = DepthwiseConv1d(features=self.conv_features, kernel=self.conv_kernel,
                                      use_bias=self.use_conv_bias, init_std=self.init_std, name='conv1d')
        # The reference's init: A_log = log(1..H), dt_bias the inverse
        # softplus of a step drawn between its default time_step_min and
        # max, D ones (init_mamba2_weights, modeling_mamba2.py:428-442).
        self.A_log = self.param('A_log', _log_head_index, (self.num_heads,))
        self.dt_bias = self.param('dt_bias', inverse_softplus_step, (self.num_heads,))
        self.D = self.param('D', nn.initializers.ones, (self.num_heads,), jnp.float32)
        self.norm = MambaRMSNormGated(
            epsilon=self.norm_eps, groups=self.norm_groups, dtype=self.dtype, name='norm')
        self.out_proj = dense(self.emb_features, name='out_proj', **normal_kernel(
            self.init_std if self.output_init_std is None else self.output_init_std))

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        decode, attention_metadata = call.decode, call.metadata
        batch, length, _ = x.shape
        tokens = coordinates_of(attention_metadata, batch, length)
        valid = tokens.valid
        if valid is not None:
            x = jnp.where(valid[:, :, None], x, 0)
        projected = self.in_proj(x)
        gate, mixed, dt = jnp.split(
            projected, [self.intermediate_size, self.intermediate_size + self.conv_features], axis=-1)
        # The conv sums in at least fp32 and hands the scan the compute
        # dtype, as mamba_ssm's conv hands its scan; the step, the decay and
        # the states stay in at least fp32 (`at_least_fp32`).
        compute = mixed.dtype
        wide = at_least_fp32(compute)
        dt = dt.astype(wide)
        taps, bias = self.conv1d()
        heads = (jnp.asarray(self.dt_bias, wide), -jnp.exp(self.A_log.astype(wide)),
                 jnp.asarray(self.D, wide))
        scan = functools.partial(
            _ssd, num_heads=self.num_heads, head_dim=self.head_dim, n_groups=self.n_groups,
            state_size=self.state_size, chunk_size=self.chunk_size,
            time_step_limit=self.time_step_limit)
        shards = sequence_shards()
        if shards > 1:
            if decode:
                raise LayoutRefused(
                    f"mamba2 under a sequence axis of {shards} runs whole training sequences; a decode "
                    "step holds one state per row. Decode on a mesh with sequence=1.")
            # Wide across the shards, so a halo's gradient and its shard's own add before rounding.
            out = _over_sequence(functools.partial(_sequence_mix, scan=scan, dtype=compute),
                                 shards, mixed.astype(wide), dt, tokens.segments, tokens.starts, valid,
                                 (taps, bias, *heads))
            if valid is not None:
                out = jnp.where(valid[:, :, None], out, 0)
            return self.out_proj(self.norm(out, gate))
        out = held_scan(self, mixed, taps, bias, functools.partial(scan, dt=dt, heads=heads),
                        (self.num_heads, self.head_dim, self.state_size),
                        decode=decode, coordinates=tokens, state_dtype=wide, conv_dtype=wide,
                        dtype=compute)
        if out is None:
            # Allocation only, as the delta net: init_cache's dummy token
            # must not consume a position or leave state behind.
            return self.out_proj(jnp.zeros((batch, length, self.intermediate_size), self.dtype))
        if valid is not None:
            out = jnp.where(valid[:, :, None], out, 0)
        return self.out_proj(self.norm(out, gate))



def _ssd(convolved, dt, heads, *, num_heads: int, head_dim: int, n_groups: int,
         state_size: int, chunk_size: int, time_step_limit, starts=None, valid=None, held=None,
         axis: str | None = None):
    """The scan after the conv: `convolved` `[B, S, conv_dim]` split into
    `x`, `B` and `C`, the step through softplus with its bias and the limit,
    then the recurrent step for one token and the chunked scan otherwise.
    Returns the output `[B, S, H * P]` and the zero-start state leaving it.

    Under `axis` the scan runs from zero and the output gains `C_t . exp(acs_t)
    h` for the state `h` the earlier shards leave, `acs_t` the inclusive sum of
    this shard's `A dt` (a `RESET_DECAY` in it zeroes the term). The zero-start
    final state is the shard's write `B_r` (`_entering_state`)."""
    dt_bias, A, D = heads
    batch, length, _ = convolved.shape
    intermediate = num_heads * head_dim
    xs, B, C = jnp.split(convolved, [intermediate, intermediate + n_groups * state_size], axis=-1)
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
        carried = float32_einsum('bsgn,bgkpn,bsgk->bsgkp', C, entering, decayed)
        out = out + carried.reshape(batch, length, num_heads, head_dim)
    return out.reshape(batch, length, intermediate), final


def _sequence_mix(mixed, dt, segments, starts, valid, weights, *, scan, dtype):
    """The conv and the scan over this shard's slice of whole training
    sequences, inside a `shard_map` that holds the sequence axis manual; a
    single shard runs `held_scan` instead.

    `mixed` `[B, S, conv_dim]` the projection, convolved in at least fp32
    and handed to the scan in `dtype`, the model's compute dtype, `dt`
    `[B, S, H]` in fp32, `segments` `[B, S]` and their documents' `starts` or None, `valid` `[B, S]`
    the real tokens or None, `weights` the conv's taps and bias and the heads' `dt_bias`, `A` and
    `D`. The conv reads the previous shard's last `K-1` tokens and segments by one `ppermute` (zeros
    on the first) and the scan is corrected for the earlier shards' state (`_ssd`). A segment change
    resets both wherever it falls. With `valid`, as `held_scan` runs a row with padding, a padded slot
    neither reads nor advances the conv or the state: each real token's conv reads the `K-1` real
    tokens before it in its document, wherever they lie (`_halo`), and a padded slot's step is
    zero."""
    taps, bias, dt_bias, A, D = weights
    width = taps.shape[1] - 1
    length = mixed.shape[1]
    if length < width:
        raise LayoutRefused(
            f"each sequence shard holds {length} tokens, fewer than the {width} the "
            f"conv reads behind its first; use fewer sequence shards or longer sequences")
    if valid is not None:
        halo = _halo(mixed, valid, segments, width)
        convolved, _ = held_conv1d(mixed, taps, valid, halo, bias=bias, segments=segments, dtype=dtype)
        out, _ = scan(convolved, dt, (dt_bias, A, D), starts=starts, valid=valid, axis=SEQUENCE_AXIS)
    else:
        forward = [(shard, shard + 1) for shard in range(jax.lax.axis_size(SEQUENCE_AXIS) - 1)]
        history = jax.lax.ppermute(mixed[:, length - width:], SEQUENCE_AXIS, forward)
        owners = None
        if segments is not None:
            before = jax.lax.ppermute(segments[:, length - width:], SEQUENCE_AXIS, forward)
            owners = jnp.concatenate([before, segments], axis=1)
        stream = jnp.concatenate([history, mixed], axis=1)
        convolved = causal_conv1d(stream, taps, bias=bias, segments=owners, dtype=dtype)[:, width:]
        out, _ = scan(convolved, dt, (dt_bias, A, D), starts=starts, axis=SEQUENCE_AXIS)
    # The scan's output as `held_scan` names it, a mixer's 'context' and a recurrence's 'scan'
    # (a remat policy's `RESIDUALS`).
    return checkpoint_name(checkpoint_name(out, 'scan'), 'context')


def _halo(mixed, valid, segments, width: int):
    """The `width` real tokens before this shard's first, `[B, width, conv_dim]`, from every earlier
    shard of the sequence axis: each shard's last real tokens (`real_tail`) gathered, then taken in
    order, so a shard holding fewer than `width` real tokens passes on the ones before it too. Rows of
    another document than this shard's first real token's are zeros, where its conv starts afresh."""
    tail, documents, count = real_tail(mixed, valid, segments, width)
    tails, all_documents, counts = (jax.lax.all_gather(value, SEQUENCE_AXIS)
                                    for value in (tail, documents, count))
    here = jax.lax.axis_index(SEQUENCE_AXIS)
    halo = jnp.zeros_like(tail)
    halo_documents = jnp.full_like(documents, -1)
    slots = jnp.arange(width)[None, :]
    for shard in range(tails.shape[0]):
        # A shard's tail holds its last `kept` real tokens at its end, zeros before them; behind them
        # come the halo's last `width - kept`.
        kept = jnp.minimum(counts[shard], width)[:, None]
        behind = slots < width - kept
        rows = jnp.minimum(kept + slots, width - 1)
        joined = jnp.where(behind[:, :, None], jnp.take_along_axis(halo, rows[:, :, None], axis=1),
                           tails[shard])
        joined_documents = jnp.where(behind, jnp.take_along_axis(halo_documents, rows, axis=1),
                                     all_documents[shard])
        earlier = shard < here
        halo = jnp.where(earlier, joined, halo)
        halo_documents = jnp.where(earlier, joined_documents, halo_documents)
    if segments is None:
        return halo
    first_real = jnp.argmax(jnp.asarray(valid, bool), axis=1)[:, None]
    first = jnp.take_along_axis(jnp.asarray(segments, jnp.int32), first_real, axis=1)
    return jnp.where((halo_documents == first)[:, :, None], halo, 0)


def _over_sequence(mix, shards: int, mixed, dt, segments, starts, valid, weights):
    """Run `mix` (`_sequence_mix` under the sequence axis) in a `manual_map`
    that splits the rows and the sequence as the rule table places a
    `[batch, length, ...]` activation, with the weights whole on every
    shard. The layer's projections and its gated norm act token by token
    and stay with GSPMD outside it."""
    length = mixed.shape[1]
    if length % shards:
        raise LayoutRefused(
            f"mamba2 splits the sequence of {length} tokens {shards} ways over the "
            "mesh's sequence axis, and it does not divide")
    tokens = logical_spec(("activation_batch", "activation_length"), mixed.shape[:2])
    split = manual_map(mix, (tokens, tokens, tokens, tokens, tokens, P()), tokens)
    return split(mixed, dt, segments, starts, valid, weights)


def _log_head_index(key, shape):
    """`A_log = log(1..H)`, the S4D-real init (modeling_mamba2.py:430-431)."""
    del key
    return jnp.log(jnp.arange(1, shape[0] + 1, dtype=jnp.float32))


@dataclasses.dataclass(frozen=True)
class Mamba2Mixer(MixerBase):
    """The `mamba2` kind, by the reference config's field names
    (configuration_mamba2.py). It ignores the context's attention geometry: an
    SSM has no keys to cache or positions to rotate, and its `num_heads` and
    `head_dim` are the checkpoint's own.
    """

    residuals = ('context', 'scan', 'conv')

    @property
    def keeps_triton_gemm(self) -> bool:
        """Where the scan runs on XLA (the GPU's target row names no port), its
        small batched dots gain from the fusions, 7.7% on an A100; on the
        port a Mamba-2 130M step ran 3% faster without them."""
        return measured_kernel('ssd', 'xla') != 'pallas'

    num_heads: int = 128
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    chunk_size: int = CHUNK_SIZE
    use_bias: bool = False
    use_conv_bias: bool = True
    norm_groups: int = 1
    """One for Mamba-2's whole-width norm; n_groups for Nemotron-H's grouped norm."""
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
            norm_groups=self.norm_groups,
            time_step_limit=(float(self.time_step_limit[0]), float(self.time_step_limit[1])),
            norm_eps=ctx.norm_eps,
            init_std=ctx.init_std,
            output_init_std=ctx.output_init_std,
            dtype=ctx.dtype,
            precision=ctx.precision)
