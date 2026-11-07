"""The recurrence template's forward over a call's chunks: its operands laid in rows of sequences and padded
to whole blocks and chunks (`prepared`), its launches in turn (`templates.recurrence.mainloop`'s: the
inverse and the weighted keys and values, the states, the output or the fused scan), and the values its fast
backward reads kept (`kept_forward`)."""

import dataclasses
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.inputs import cu_seqlens
from dew.nn.kernels import chunked_launch, generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.recurrence.mainloop import (
    CHUNK,
    PARTS,
    Call,
    Operand,
    Tiles,
    _intra,
    _intra_solve,
    _output,
    _parts,
    _scan,
    _solved,
    _solves_apart,
    _states,
    _weighted,
    _weighted_values,
    block_of,
    family_of,
    per_key_refusal,
    value_unit,
)


def zero_padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size`."""
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


def _layout(batch: int, length: int, starts, max_segments: int | None, firsts=None):
    """A call's tokens a chunk at a time: `(chunks, taken [chunks], sequences [N, 5], laid, unlaid)`, each
    chunk's tokens, each sequence's first chunk, chunks, row and whether it begins and ends its row
    (`_states`), and the maps from `[B, T, ...]` to the chunks' rows `[chunks * C, ...]` (a chunk's rows past
    its tokens zero) and back. An aligned call's chunks are its rows' own, unmoved."""
    tokens = batch * length
    if starts is None and length % CHUNK == 0:
        chunks = tokens // CHUNK
        sequences = np.stack([np.arange(batch) * (length // CHUNK), np.full(batch, length // CHUNK),
                              np.arange(batch), np.ones(batch), np.ones(batch)], 1)

        return (chunks, jnp.full((chunks,), CHUNK, jnp.int32), jnp.asarray(sequences, jnp.int32),
                lambda x: x.reshape(tokens, *x.shape[2:]), lambda x: x.reshape(batch, length, *x.shape[1:]))
    if starts is None:
        spans = chunked_launch.sequences(batch, length, chunk=CHUNK)
    else:
        documents = length if max_segments is None else max_segments
        spans = chunked_launch.sequences(batch, length, cu_seqlens(starts, batch * (documents + 1), firsts),
                                         chunk=CHUNK)
    table, seqs = jnp.asarray(spans.chunks), jnp.asarray(spans.spans)
    chunks, taken = table.shape[0], table[:, 1]
    first, count = seqs[:, 0], seqs[:, 1]
    sizes = -(-count // CHUNK)
    live = count > 0
    sequences = jnp.stack([jnp.cumsum(sizes) - sizes, sizes, first // length, live & (first % length == 0),
                           live & ((first + count) % length == 0)], 1).astype(jnp.int32)
    place = jnp.arange(CHUNK)
    real = (place[None, :] < taken[:, None]).reshape(-1)
    index = jnp.where(real, (table[:, :1] + place[None, :]).reshape(-1), tokens)
    rows = jnp.zeros((tokens,), jnp.int32).at[index].set(jnp.arange(chunks * CHUNK, dtype=jnp.int32),
                                                         mode="drop")

    def laid(x):
        flat = x.reshape(tokens, *x.shape[2:])
        kept = real.reshape((-1,) + (1,) * (flat.ndim - 1))
        return jnp.where(kept, flat[jnp.minimum(index, tokens - 1)], jnp.zeros((), x.dtype))

    def unlaid(x):
        return x[rows].reshape(batch, length, *x.shape[1:])

    return chunks, taken, sequences, laid, unlaid


def forward(q, k, v, g, beta, h0=None, *, tiles: Tiles, starts=None, firsts=None,
            max_segments: int | None = None,
            scale: float | None = None, delta: bool = True, wide: bool = False,
            family: mosaic.Family | None = None, interpret: bool = False, floor: float | None = None,
            rounded: bool = False, ranged: bool = False, skip=None):
    """The chunked rule's forward on this mainloop: `(o [B, T, HV, Dv], final [B, HV, Dk, Dv])` for `q`, `k`
    `[B, T, H, Dk]`, `v` `[B, T, HV, Dv]`, the natural log decay `g`, a head's `[B, T, HV]` (the gated delta
    rule) or a key dimension's `[B, T, HV, Dk]` (Kimi delta attention), the write gate `beta` `[B, T, HV]` and
    each row's initial memory `h0` `[B, HV, Dk, Dv]` (zero where None), as `recurrence_chunk.fla_rule` takes
    them: `starts` `[B, T]` marks packed documents' first tokens, at most `max_segments` a row (the row's
    length where None, which the chunks' layout is sized by), each a sequence of its own from a zero memory
    but a row's first, which takes `h0` unless its first token begins a document; the queries scaled by
    `scale` (`Dk^-1/2` where None); the output in the operands' common dtype, fp32 where `wide`, the final
    memory fp32. Without the `delta` rule nothing is corrected and each token writes its values gated by
    `beta` (the identity inverse: Mamba2's SSD through its rated decay's log gates, the gated linear
    attentions, GLA's with a decay a key dimension). Keys and values padded to whole blocks (`block_of`), at
    any width whose launches the device holds (the template's legality, `template.planned`'s shared
    memory and registers a launch). Its states and output launches take `tiles` (the template's config,
    `templates.recurrence.template`), and its launches issue by `family`'s bundle (the device's, `family_of`,
    where None), interpreted where `interpret`. A decay a key dimension takes the layer's declared `floor`
    (`per_key_refusal`). A `rounded` call takes the run's fast forms' numbers: without the delta rule
    (Mamba2's SSD, whose 16-bit numbers are mamba_ssm's) it passes its values between launches in the
    operands' dtype, and an fp32 call's products take its operands as TF32 where its family's products take
    them transposed (mma.sync's), else in `FAST_PARTS` bf16 parts (`prepared`, `_parts`). A `ranged` 16-bit
    delta rule, whose form proves no bound on its chunk inverse, solves it at fp32's range (`_merged`). A
    `skip` `[HV]` adds `D v` to the output of a call whose write gate takes the keys' side
    (`Call.gated_keys`, Mamba2's SSD), as mamba_ssm's output kernel adds it."""
    out, final, _ = kept_forward(q, k, v, g, beta, h0, starts=starts, firsts=firsts,
                                 max_segments=max_segments, scale=scale, delta=delta, tiles=tiles, wide=wide,
                                 family=family, interpret=interpret, floor=floor, keeps=False,
                                 rounded=rounded, ranged=ranged, skip=skip)
    return out, final


def kept_forward(q, k, v, g, beta, h0=None, *, tiles: Tiles, starts=None, firsts=None,
                 max_segments: int | None = None,
                 scale: float | None = None, delta: bool = True, wide: bool = False,
                 family: mosaic.Family | None = None, interpret: bool = False,
                 floor: float | None = None, keeps: bool = True,
                 rounded: bool = False, ranged: bool = False,
                 skip=None) -> tuple[jax.Array, jax.Array, tuple]:
    """`forward`'s `(o, final)` and what the fast backward takes rather than compute again
    (`mainloop_backward.gradients`' `inverse` and `kept`): each chunk's inverse under a decay a head
    (`_solved`'s, None otherwise or where the call `keeps` none, `forward`'s), as fla's backward keeps its
    `A`, and the launches' weighted keys, memories and corrected values, `(w, h, corrected)`, which the port
    keeps unless a remat rung recomputes them (`recurrence_chunk.RECOMPUTED`), the caller's choice."""
    # A forward keeping nothing for a backward under a decay a head runs its states and output in one walk
    # (`_scan`), no memory kept for a backward to take, where its family's products read their operands from
    # shared memory (warpgroup MMA's) and a 16-bit call's operands are one part: where they read registers
    # (mma.sync's), the walk's two phases' operand fragments held its thread past its registers at every tile
    # (an sm87's and an A100's), as the output launch's two phases at once did (`_output_held`); an fp32
    # call's parts spilled it at an H100's widest (132 bytes at two roles of 16 columns, jsf-scan4 at
    # d54ec6b02).
    dtype = jnp.result_type(q, k, v)
    reads = family_of(dtype) if family is None else family
    scans = (not keeps and g.ndim == 3 and jnp.dtype(dtype).itemsize == 2 and reads is not None
             and not reads.operands_in_registers)
    run = prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                   tiles=tiles, family=family, interpret=interpret, floor=floor, keeps=keeps, firsts=firsts,
                   rounded=rounded, ranged=ranged, states=not scans)
    call, tiles = run.call, run.tiles
    out_dtype = jnp.float32 if wide else run.dtype
    if scans:
        o, final = _scan(run.q, run.keys, run.a, run.v, run.hi, run.lo, run.beta, run.h0, run.sequences,
                         run.taken, call, tiles, scale=run.scale, out_dtype=out_dtype, skip=skip)
        final = run.final_of(final)
    else:
        o = _output(run.q, run.k, run.h, run.corrected, run.hi, run.lo, run.taken, run.beta, call, tiles,
                    scale=run.scale, out_dtype=out_dtype, scores=run.scores, skip=skip)
        final = run.final
    o = run.unlaid(o.reshape(call.rows, call.heads, call.value_width))[..., :run.value_width]
    assert final is not None, "a forward runs its states launch"
    return o, final, (run.a, (run.w, run.h, run.corrected))


@dataclasses.dataclass(frozen=True)
class _Run:
    """A call's operands laid a chunk at a time and the forward's values before the output (`prepared`):
    what the output launch and the backward's take."""

    call: Call
    tiles: Tiles
    dtype: jnp.dtype
    scale: float
    key_width: int
    value_width: int
    batch: int
    q: jax.Array
    k: jax.Array
    v: jax.Array
    hi: jax.Array
    lo: jax.Array
    beta: jax.Array
    a: jax.Array | None
    w: jax.Array | None
    keys: jax.Array
    """The keys the memory takes: the laid keys under a decay a head, gated to each chunk's end
    (`algebra.key_decayed_write`, `_weighted`'s) under one a key dimension."""
    scores: jax.Array | None
    h: jax.Array | None
    corrected: jax.Array | None
    final: jax.Array | None
    taken: jax.Array
    sequences: jax.Array
    laid: Callable
    unlaid: Callable
    u: jax.Array | None = None
    """The states launch's values: the weighted values under the delta rule, else the values laid."""
    h0: jax.Array | None = None
    """Each row's initial memory `[B * HV * Dk, Dv]` fp32, as the states launch takes it."""

    def final_of(self, final: jax.Array) -> jax.Array:
        """A states walk's final memories `[B * HV * Dk, Dv]` as the call's `[B, HV, Dk, Dv]`."""
        width, columns = self.call.width, self.call.value_width
        return final.reshape(self.batch, self.call.heads, width, columns)[:, :, :self.key_width,
                                                                          :self.value_width]


def prepared(q, k, v, g, beta, h0, *, starts, max_segments, scale, delta, tiles, family, interpret,
             firsts=None,
             inverse=None, kept=None, floor: float | None = None, keeps: bool = True,
             rounded: bool = False, ranged: bool = False, states: bool = True) -> _Run:
    """`forward`'s operands checked, padded and laid, and its launches before the output run: under a decay a
    head the solve's but where its `inverse` is given, and the weighted and states launches' but where their
    `kept` values, `(w, h, corrected)`, are (`kept_forward`'s; its final memory then None), or but the
    states launch's where not `states` (a walk forming the output too, `_scan`, runs it; its memories,
    corrected values and final memory then None). The solve stores its inverse where the call `keeps` it
    (`_solved`)."""
    family = family_of(jnp.result_type(q, k, v)) if family is None else family
    if family is None:
        raise NotImplementedError(f"the mainloop has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    batch, length, _, key_width = k.shape
    heads, value_width = v.shape[2:]
    dtype = jnp.result_type(q, k, v)
    per_key = g.ndim == 4
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)):
        raise NotImplementedError(f"the mainloop takes 16-bit and fp32 operands, not {dtype}")
    if per_key and (refused := per_key_refusal(dtype)):
        raise NotImplementedError(refused)
    scale = key_width ** -0.5 if scale is None else scale
    block = block_of(family)
    # The values padded to whole blocks where a fast backward reads them (its launches a block at a time),
    # for a forward keeping nothing to whole value tiles (`value_unit`) its walks' and output's blocks divide
    # (`Call.value_block`).
    unit = block if keeps else math.lcm(value_unit(family), tiles.block_v * tiles.walkers,
                                        tiles.block_o or tiles.block_v)
    width, columns = -(-key_width // block) * block, -(-value_width // unit) * unit
    q, k = (zero_padded(x.astype(dtype), 3, width) for x in (q, k))
    v = zero_padded(v.astype(dtype), 3, columns)
    g = zero_padded(g.astype(jnp.float32), 3, width) if per_key else g.astype(jnp.float32)
    # The delta rule's write gate in the operands' dtype, as fla's layer holds it; without the delta rule the
    # reference's own fp32 (Mamba-2's `dt`).
    beta = (beta.astype(dtype) if delta else beta).astype(jnp.float32)
    h0 = (jnp.zeros((batch, heads, width, columns), jnp.float32) if h0 is None
          else zero_padded(zero_padded(h0.astype(jnp.float32), 2, width), 3, columns))
    if starts is not None:
        starts = jnp.asarray(starts, bool)
        h0 = jnp.where(starts[:, 0, None, None, None], 0.0, h0)
    chunks, taken, sequences, laid, unlaid = _layout(batch, length, starts, max_segments, firsts)
    rows = chunks * CHUNK
    given = Operand(rows, _parts(dtype, family, rounded), dtype)
    # What the launches pass between them: the delta rule's in the operands' dtype, as fla's rounds it (the
    # port is its reference). Without it, a `rounded` call's (the SSD's under the run's fast forms, which its
    # route resolves, `recurrence_chunk.fast_rounding`) in the operands' dtype too, as mamba_ssm rounds its
    # scores and entered memories for its products (`_chunk_scan_fwd`'s `cb` and `prev_states`), its memory
    # summed and passed in fp32: on an H100 Nemotron-H 4B's SSD forward took 0.581 ms against fp32's three
    # parts' 2.031 and the port's 0.606, 30B's 0.249 against 0.593 and 0.229 (ssd-held-h100-fdab2424c). Else,
    # and under the accurate forms, fp32's three parts, as the portable math holds them (a 16-bit call's
    # rounded put the output 1.7 times and the final memory up to 540 times the portable math's distance from
    # float64).
    held = given if delta or rounded else Operand(rows, PARTS, jnp.dtype(jnp.float32))
    call = Call(heads, heads // k.shape[2], width, columns, rows, given, held, per_key, delta, family,
                interpret, generation.gpu_target(), ranged=ranged,
                factored=not per_key or (floor is not None and algebra.factored_range(floor)))
    beta = laid(beta).reshape(rows, heads).T
    q, k, v = (given.split(laid(x).reshape(rows, -1)) for x in (q, k, v))
    a, w, u, keys, scores, system = None, None, v, k, None, []
    decays = laid(g).reshape(rows, -1)
    if per_key:
        # A decay a key dimension's prefix halves `[R, HV * Dk]` and its scores, and its system where the
        # call solves it (no inverse given), one launch's (`_intra`).
        solves = delta and kept is None and inverse is None
        scores, *system, hi, lo = _intra(q, k, decays, beta, taken, call, scale=scale, solves=solves)
    else:
        # A decay a head's a head's row of vectors `[HV, R]`.
        hi, lo = (x.reshape(rows, heads).T for x in algebra.prefix(decays.reshape(chunks, CHUNK, -1)))
    if per_key and delta and kept is not None:
        # The forward's inverse, weights, memories and corrected values given (`kept`): its scores and gated
        # keys alone formed again, the weighted launch without the delta rule's system.
        (keys,) = _weighted(k, v, None, hi, lo, beta, taken, dataclasses.replace(call, delta=False))
        a = inverse
    elif per_key and delta and inverse is not None:
        # The forward's inverse given without its states (a recomputed forward, `delta_memory`): its weights
        # from the inverse, the system's launch and the solve not repeated.
        a = inverse
        w, keys = _weighted(k, v, a, hi, lo, beta, taken, call)
        u = _weighted_values(v, a, beta, taken, call)
    elif per_key and delta:
        a = _intra_solve(system[0], taken, call)
        w, keys = _weighted(k, v, a, hi, lo, beta, taken, call)
        u = _weighted_values(v, a, beta, taken, call)
    elif per_key:
        (keys,) = _weighted(k, v, None, hi, lo, beta, taken, call)
    elif delta and inverse is None and not states:
        # A walk correcting its values itself (`_scan`) takes the inverse alone.
        (a,) = _solved(k, v, hi, lo, beta, taken, call, applies=False)
    elif delta and inverse is None:
        # The forward's residuals come in pairs: a recomputed forward has neither.
        assert kept is None, "a kept forward keeps its inverse"
        if _solves_apart(call):
            (a,) = _solved(k, v, hi, lo, beta, taken, call, applies=False)
            w, u = _weighted(k, v, a, hi, lo, beta, taken, call)
        else:
            a, w, u = _solved(k, v, hi, lo, beta, taken, call, keeps=keeps)
    elif delta:
        a = inverse
        if kept is None:
            w, u = _weighted(k, v, a, hi, lo, beta, taken, call)
    h0 = h0.reshape(-1, columns)
    run = _Run(call, tiles, dtype, scale, key_width, value_width, batch, q, k, v, hi, lo, beta, a, w, keys,
               scores, None, None, None, taken, sequences, laid, unlaid, u=u, h0=h0)
    if kept is not None:
        # The forward's saved states, with the delta rule's weights or without them (`w` None).
        (w, h, corrected) = kept
        return dataclasses.replace(run, w=w, h=h, corrected=corrected)
    if not states:
        return run
    h, corrected, final = _states(keys, w, u, hi, lo, beta, h0, sequences, taken, call, tiles)
    return dataclasses.replace(run, h=h, corrected=corrected, final=run.final_of(final))
