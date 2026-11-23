"""RWKV4's WKV (row 1455; fla's `fused_recurrent_rwkv4`, fla/ops/rwkv4): a recurrence a channel, each
channel's past held as a numerator and a denominator scaled by a running maximum,

    out_t  = (e^{eps - tau} alpha + e^{u + k_t - tau} v_t) / (e^{eps - tau} beta + e^{u + k_t - tau}),
             tau = max(u + k_t, eps)
    eps'   = max(w + eps, k_t),  alpha' = e^{w + eps - eps'} alpha + e^{k_t - eps'} v_t,
             beta' = e^{w + eps - eps'} beta + e^{k_t - eps'}

with `w = -exp(time_decay)` a channel's decay and `u` (time_first) the current token's bonus. Width one a
channel holds no product, so its selector (`WKV4`) runs it on the recurrence template's channel walk
(`templates.recurrence.wkv4`) where it takes the call, else on the portable math: a span's carry composes
with the next's associatively (`joined`), one `associative_scan` of every token's `(e^{k} v, e^{k}, max k, 1)`
over the sequence, segmented at packed documents' first tokens."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from dew.nn.precision import at_least_fp32
from dew.nn.sharding import grouped_heads, manual_map

from . import generation
from .generation import INHERENT, Choice
from .recurrence_chunk import _VARYING, _per_product
from .selection import Kernel, Op, Requirement, select
from .templates import scan as block_scan

EMPTY = -jnp.inf
"""An empty past's running maximum: a span that holds no token, so it never stands for a token's (a finite
one, -1e38, outweighed a first token's key of -2e38 and read it as 0 / 0). Its `exp` against any token's is
zero, its slope zero: every span joined after the entering one holds a token, and a token reads its own key
beside it (`tau`), so no maximum is two empty ones'."""


class SpanCarry(NamedTuple):
    """A span of tokens' carry: its numerator and denominator scaled by `e^{-top}`, `top` their running
    maximum (fla's alpha, beta and eps), `tokens` its tokens (each decaying what precedes it once) and
    `fresh` whether it begins a document (what precedes it is dropped)."""

    numerator: jax.Array
    denominator: jax.Array
    top: jax.Array
    tokens: jax.Array
    fresh: jax.Array


def _weighed(kept, earlier, added, later):
    """Two spans' values at their larger maximum: `kept` times the earlier's, `added` times the later's."""
    return kept * earlier + added * later


def joined(decay, *, numerators: Callable = _weighed, denominators: Callable = _weighed):
    """The composition of a span with the span after it (`associative_scan`'s operator): the earlier one's
    past decayed by `decay` once a token of the later one, both rescaled to their larger maximum, where the
    later one begins no document; `numerators` and `denominators` weigh each value (`_weighed`'s, or another
    form's, the template's extended floats, `templates.recurrence.wkv4`)."""

    def join(before: SpanCarry, after: SpanCarry) -> SpanCarry:
        aged = before.top + after.tokens * decay
        top = jnp.maximum(aged, after.top)
        kept, added = jnp.exp(aged - top), jnp.exp(after.top - top)
        # The carries' own type (`SpanCarry`, or a form's holding its values otherwise), so the selection
        # between the joined and the dropped carry meets one tree.
        joined = after._replace(numerator=numerators(kept, before.numerator, added, after.numerator),
                                denominator=denominators(kept, before.denominator, added, after.denominator),
                                top=top, tokens=before.tokens + after.tokens, fresh=before.fresh | after.fresh)
        dropped = after._replace(tokens=joined.tokens, fresh=joined.fresh)
        return jax.tree.map(lambda one, other: jnp.where(after.fresh, other, one), joined, dropped)
    return join


def _operands(time_decay, time_first, key, value, state):
    """`wkv4`'s operands in its working dtype (`at_least_fp32` of the inputs'): the decay `-exp(time_decay)`,
    the bonus, the keys and values, and the carry entering, an empty past's (no numerator or denominator,
    the running maximum `EMPTY`) where `state` is None; and the inputs' dtype."""
    dtype = jnp.result_type(key, value)
    work = at_least_fp32(dtype)
    key, value = jnp.asarray(key, work), jnp.asarray(value, work)
    batch, _, channels = key.shape
    if state is None:
        state = jnp.stack([jnp.zeros((batch, channels), work), jnp.zeros((batch, channels), work),
                           jnp.full((batch, channels), EMPTY, work)], axis=1)
    return (-jnp.exp(jnp.asarray(time_decay, work)), jnp.asarray(time_first, work), key, value,
            jnp.asarray(state, work), dtype)


@dataclasses.dataclass(frozen=True)
class WkvCase:
    """A WKV (`wkv4`) as its selector reads it (`WKV4`): whether a shard_map that checks varying axes holds
    its operands, its keys' and values' dtype, its channels, and its device's instruction family (None off a
    GPU)."""

    varying: bool
    dtype: np.dtype
    channels: int
    family: str | None = None


WKV4_SUPPORTS: tuple[Requirement[WkvCase], ...] = (
    Requirement('varying', lambda case: not case.varying, _VARYING, row='ledger 822'),
    Requirement('gpu', lambda case: case.family is not None,
                "no GPU of an instruction family the template's bundles cover: off one the template "
                "interprets, as the CPU suite runs it where a call names it", row=INHERENT, kind='measured'),
    Requirement('dtype', lambda case: case.dtype in (jnp.bfloat16, jnp.float16, jnp.float32),
                lambda case: f"{case.dtype} keys and values; the kernels take 16-bit and fp32 ones",
                row=INHERENT),
    Requirement('channels', lambda case: case.channels % block_scan.THREADS == 0,
                lambda case: f"{case.channels} channels, no whole number of a warpgroup's "
                             f"{block_scan.THREADS}", row='ledger 1455'),
)
"""What the template's WKV (`templates.recurrence.wkv4`) needs of a call: no caller's `shard_map` that
checks varying axes, a GPU of a family its bundles cover, 16-bit or fp32 keys and values, and whole
warpgroups of channels."""

WKV4: Op[WkvCase] = Op(Choice.WKV4, (Kernel('mosaic_gpu', WKV4_SUPPORTS), Kernel('xla', (), generic=True)))
"""The WKV's selector: the recurrence template's channel walk (`templates.recurrence.wkv4`) where it takes
the call, else the portable scan (`scanned`)."""


def wkv4_kernel(key, value) -> str:
    """The kernel the WKV's selector (`WKV4`) gives a call over `key` and `value` `[B, S, C]` on the target,
    reported as its route: of the dtype the kernels run them in, their common one (a call's keys alone in
    bf16 beside fp64 values under x64 took the kernels, which ran them in fp32, ReviewRecurrenceBatch)."""
    from .templates import bundles

    dtype = jnp.dtype(jnp.result_type(key, value))
    on_gpu = generation.backend() == 'gpu'
    bundled = on_gpu and bundles.of(generation.device_generation(), dtype) is not None
    family = generation.family(generation.device_generation()) if bundled else None
    # Either operand a checked `shard_map` varies refuses the kernels (replicated keys beside row-varying
    # values among them, ReviewRecurrenceBatch).
    varying = bool(jax.typeof(key).mat.varying or jax.typeof(value).mat.varying)
    case = WkvCase(varying=varying, dtype=dtype, channels=key.shape[-1], family=family)
    return select(WKV4, case).kernel


def wkv4(time_decay, time_first, key, value, state=None, starts=None):
    """RWKV4's WKV over a sequence. `time_decay` and `time_first` `[C]` (the decay `w = -exp(time_decay)`,
    the bonus `u`), `key` and `value` `[B, S, C]`; `state` `[B, 3, C]` the carry entering (fla's alpha,
    beta and eps; an empty past where None); `starts` `[B, S]` each packed document's first token, which
    enters an empty past. Returns `(out [B, S, C] in the inputs' dtype, state [B, 3, C] in at least fp32)`,
    as fla's `fused_recurrent_rwkv4` (its state's last token), on the kernel its selector gives each product
    (`WKV4`, `recurrence_chunk._per_product`): the recurrence template's walk, interpreted off a GPU where a
    call names it, per shard of rows on a mesh (`manual_map`); or the portable scan (`scanned`). A call of
    no tokens walks none: its final carry is the one entering."""
    if key.shape[1] == 0:
        *_, state, dtype = _operands(time_decay, time_first, key, value, state)
        return jnp.zeros(key.shape, dtype), state

    def template(decay, bonus, key, value, state, *given):
        from .templates.recurrence import mainloop, wkv4 as launches

        family, interpreted = mainloop.issued(jnp.float32)
        return launches.wkv4(decay, bonus, key, value, state, given[0] if given else None, family=family,
                             interpret=interpreted)

    def run(kernel, time_decay, time_first, key, value, state):
        if kernel != 'mosaic_gpu':
            return scanned(time_decay, time_first, key, value, state, starts)
        decay, bonus, _, _, carry, dtype = _operands(time_decay, time_first, key, value, state)
        rows, _ = grouped_heads((*key.shape[:2], 1, key.shape[-1]), 1)
        three, one = P(rows, None, None), P(None)
        operands = (decay, bonus, jnp.asarray(key, dtype), jnp.asarray(value, dtype), carry,
                    *([] if starts is None else [starts]))
        specs = (one, one, three, three, three, P(rows, None))[:len(operands)]
        return manual_map(template, specs, (three, three))(*operands)

    # The decay and bonus (operands 0 and 1) are the call's parameters.
    return _per_product(WKV4.bound(lambda product: wkv4_kernel(key, value)), run, time_decay, time_first, key,
                        value, state, parameters=(0, 1))


def scanned(time_decay, time_first, key, value, state=None, starts=None):
    """`wkv4` on the portable math: one associative scan of its spans' carries (`joined`); operands and
    results as `wkv4`'s."""
    decay, bonus, key, value, state, dtype = _operands(time_decay, time_first, key, value, state)
    batch, length, channels = key.shape
    fresh = (jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool))[..., None]
    fresh = jnp.broadcast_to(fresh, key.shape)
    entering = SpanCarry(state[:, 0, None], state[:, 1, None], state[:, 2, None],
                         jnp.zeros((batch, 1, channels), key.dtype), jnp.zeros((batch, 1, channels), bool))
    tokens = SpanCarry(value, jnp.ones_like(key), key, jnp.ones_like(key), fresh)
    spans = jax.tree.map(lambda first, rest: jnp.concatenate([first, rest], axis=1), entering, tokens)
    carried = jax.lax.associative_scan(joined(decay), spans, axis=1)
    # Each token reads the carry before it (an empty one at a document's first token) and itself at its
    # bonus.
    before = jax.tree.map(lambda x: x[:, :-1], carried)
    numerator, denominator = (jnp.where(fresh, 0.0, x) for x in (before.numerator, before.denominator))
    top = jnp.where(fresh, EMPTY, before.top)
    own = bonus + key
    tau = jnp.maximum(own, top)
    past, present = jnp.exp(top - tau), jnp.exp(own - tau)
    out = (past * numerator + present * value) / (past * denominator + present)
    final = jnp.stack([carried.numerator[:, -1], carried.denominator[:, -1], carried.top[:, -1]], axis=1)
    return out.astype(dtype), final


def recurrent_wkv4(time_decay, time_first, key, value, state=None, starts=None):
    """`wkv4`'s recurrence a token at a time as fla's kernel steps it (fused_recurrent.py:94-114), in at
    least fp32 (float64 where the inputs are); operands and results as `wkv4`'s."""
    decay, bonus, key, value, state, dtype = _operands(time_decay, time_first, key, value, state)
    batch, length, _ = key.shape
    starts = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)

    def one_token(carry, step):
        alpha, beta, eps = carry
        k, v, begun = step
        alpha, beta = (jnp.where(begun[:, None], 0.0, x) for x in (alpha, beta))
        eps = jnp.where(begun[:, None], EMPTY, eps)
        own = bonus + k
        tau = jnp.maximum(own, eps)
        past, present = jnp.exp(eps - tau), jnp.exp(own - tau)
        out = (past * alpha + present * v) / (past * beta + present)
        aged = decay + eps
        eps = jnp.maximum(aged, k)
        kept, added = jnp.exp(aged - eps), jnp.exp(k - eps)
        return (kept * alpha + added * v, kept * beta + added, eps), out

    entering = (state[:, 0], state[:, 1], state[:, 2])
    steps = tuple(jnp.moveaxis(x, 1, 0) for x in (key, value, starts))
    final, out = jax.lax.scan(one_token, entering, steps)
    return jnp.moveaxis(out, 0, 1).astype(dtype), jnp.stack(final, axis=1)
