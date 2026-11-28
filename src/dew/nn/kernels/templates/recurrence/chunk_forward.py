"""The recurrence template's forward over a call's chunks: its operands laid in rows of sequences and padded
to whole blocks and chunks (`prepared`), its launches in turn (`templates.recurrence.mainloop`'s: the
inverse and the weighted keys and values, the states, then the output), and the values its fast backward
reads kept (`kept_forward`)."""

import contextlib
import contextvars
import dataclasses
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import chunked_launch, generation, recurrence, recurrence_algebra as algebra, tables
from dew.nn.kernels.chunked_launch import zero_padded
from dew.nn.kernels.templates import mosaic, product
from dew.nn.kernels.templates.recurrence import own_pairs
from dew.nn.kernels.templates.recurrence.intra import _intra, _prefix
from dew.nn.kernels.templates.recurrence.mainloop import (
    PLANNED,
    Call,
    Operand,
    Tiles,
    Walked,
    _parts,
    _solved,
    _solves_apart,
    _states,
    _weighted,
    family_of,
    padded_widths,
    per_key_refusal,
)
from dew.nn.kernels.templates.recurrence.output import Normed, output_launch
from dew.nn.kernels.templates.recurrence.table import Table, turning_rows

_OBSERVED: contextvars.ContextVar[list[Call] | None] = contextvars.ContextVar(
    "recurrence_observed_calls", default=None)
"""Dispatched `Call` records collected by `observed()` outside tile planning (`PLANNED`)."""


@contextlib.contextmanager
def observed() -> Iterator[list[Call]]:
    """Collect each dispatched `Call` constructed by `prepared()` outside tile planning (`PLANNED`)."""
    calls: list[Call] = []
    token = _OBSERVED.set(calls)
    try:
        yield calls
    finally:
        _OBSERVED.reset(token)


def _record_observed(call: Call) -> None:
    """Record `call` when an `observed()` block is active outside tile planning (`PLANNED`)."""
    if (calls := _OBSERVED.get()) is not None and PLANNED.get() is None:
        calls.append(call)


def _layout(batch: int, length: int, starts, max_segments: int | None, firsts=None, *, chunk: int,
            forked: bool = False):
    """A call's tokens `chunk` at a time: `(chunks, taken [chunks], sequences [N, 5], laid, unlaid)`, each
    chunk's tokens, each sequence's walk (`Walked`; `_states`), and the maps from `[B, T, ...]` to the
    chunks' rows
    `[chunks * C, ...]` (a chunk's rows past its tokens zero) and back. An aligned call's chunks are its
    rows' own, unmoved. A `forked` call's documents are laid as a packed call's are, each from a chunk's
    edge, and each row walks its documents' chunks in turn, one sequence a row, its documents taking their
    memories up from its table (`chunked_launch.ForkedChunks`)."""
    tokens = batch * length
    if starts is None and length % chunk == 0:
        chunks = tokens // chunk
        # Each row one sequence, entering its memory and leaving it.
        sequences = tables.packed(Walked(np.arange(batch) * (length // chunk),
                                                 np.full(batch, length // chunk), np.arange(batch),
                                                 np.ones(batch), np.ones(batch)), np)

        return (chunks, jnp.full((chunks,), chunk, jnp.int32), jnp.asarray(sequences, jnp.int32),
                lambda x: x.reshape(tokens, *x.shape[2:]), lambda x: x.reshape(batch, length, *x.shape[1:]))
    if starts is None:
        spans = chunked_launch.sequences(batch, length, chunk=chunk)
    else:
        bound = length if max_segments is None else max_segments
        spans = chunked_launch.packed_sequences(starts, firsts, bound, chunk=chunk)
    table = tables.unpacked(chunked_launch.ChunkSpan, lambda i: jnp.asarray(spans.chunks)[:, i])
    span = tables.unpacked(chunked_launch.Span, lambda i: jnp.asarray(spans.spans)[:, i])
    chunks, taken = table.taken.shape[0], table.taken
    sizes = -(-span.tokens // chunk)
    live = span.tokens > 0
    sequences = tables.packed(Walked(jnp.cumsum(sizes) - sizes, sizes, span.memory,
                                             live & (span.takes == 1), live & (span.gives == 1)))
    if forked:
        # A row's documents' chunks follow one another (`cu_seqlens`' order): its walk their first's on.
        row = jnp.clip(span.memory, 0, batch - 1)
        opening = jax.ops.segment_min(jnp.where(live, jnp.cumsum(sizes) - sizes, chunks), row,
                                      num_segments=batch)
        walked = jax.ops.segment_sum(jnp.where(live, sizes, 0), row, num_segments=batch)
        sequences = tables.packed(Walked(opening, walked, jnp.arange(batch), jnp.ones(batch, jnp.int32),
                                         jnp.ones(batch, jnp.int32)))
    place = jnp.arange(chunk)
    real = (place[None, :] < taken[:, None]).reshape(-1)
    index = jnp.where(real, (table.start[:, None] + place[None, :]).reshape(-1), tokens)
    rows = jnp.zeros((tokens,), jnp.int32).at[index].set(jnp.arange(chunks * chunk, dtype=jnp.int32),
                                                         mode="drop")

    def laid(x):
        flat = x.reshape(tokens, *x.shape[2:])
        kept = real.reshape((-1,) + (1,) * (flat.ndim - 1))
        return jnp.where(kept, flat[jnp.minimum(index, tokens - 1)], jnp.zeros((), x.dtype))

    def unlaid(x):
        return x[rows].reshape(batch, length, *x.shape[1:])

    return chunks, taken, sequences, laid, unlaid


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ForkedTable:
    """A forked call's table as the template takes it, its documents laid as a packed call's are, each from a
    chunk's edge (`chunked_launch.ForkedChunks`): each row's given memories `given` `[B, S', HV, Dk, Dv]`
    fp32, the slot each chunk takes up and stores its memory in, `loads` and `stores` `[NC]`
    (`table.Table`), its entry write `entries` `[NC, HV, Dk, Dv]` fp32 where the call gives one, the row each
    chunk walks (`rows` `[NC]`), and the stored slots a row holds, `held` (static: its walks' outputs are
    shaped by it)."""

    given: jax.Array
    loads: jax.Array
    stores: jax.Array
    entries: jax.Array | None
    rows: jax.Array
    held: int = dataclasses.field(metadata={'static': True})
    adds: jax.Array | None = None
    turning: tuple | None = None
    """Each chunk's load rotation `(cos, sin)` `[NC, HV, A]` (`chunked_launch.ForkedChunks.turning`), where
    a turned call gives one."""


def tabled(forked: ForkedTable, width: int, columns: int) -> Table:
    """`forked` as the walks take it (`table.Table`): its memories padded to the call's key and value
    widths and laid as the memories are, a row past each end where it turns them (`Table.pad`), its slots
    a chunk's, its load rotation a key row's coefficients (`table.turning_rows`)."""
    pad = 0 if forked.turning is None else 1

    def laid(x):
        x = zero_padded(zero_padded(x.astype(jnp.float32), x.ndim - 2, width), x.ndim - 1, columns)
        return jnp.pad(x.reshape(-1, columns), ((pad, pad), (0, 0)))

    entries = None if forked.entries is None else laid(forked.entries)
    turning = None
    if forked.turning is not None:
        cos, sin = forked.turning
        turning = tuple(part.reshape(part.shape[0], -1).astype(jnp.float32)
                        for part in turning_rows(cos, sin, width))
    return Table(laid(forked.given), forked.loads.reshape(-1).astype(jnp.int32),
                 forked.stores.reshape(-1).astype(jnp.int32), entries, forked.rows, forked.given.shape[1],
                 forked.held, None if forked.adds is None else forked.adds.reshape(-1).astype(jnp.int32),
                 turning)


def forward(q, k, v, g, beta, h0=None, *, tiles: Tiles, starts=None, firsts=None,
            max_segments: int | None = None,
            scale: float | None = None, delta: bool = True, wide: bool = False,
            family: mosaic.Family | None = None, interpret: bool = False,
            rounded: bool = False, apart: int = 0, values_gated: bool = False, unit_pairs: bool = False,
            decay_bound: float | None = None,
                ranged: bool = False, skip=None,
            forked: ForkedTable | None = None, reads=None, normed: recurrence.OutputNorm | None = None,
            gate=None, weight=None):
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
    any width whose launches the device holds (the template's legality, `template.planned`'s shared memory and
    registers a launch). Its states and output launches take `tiles` (the template's config,
    `templates.recurrence.template`), and its launches issue by `family`'s bundle (the device's, `family_of`,
    where None), interpreted where `interpret`. A `rounded` call takes the run's fast forms' numbers: without
    the delta rule (Mamba2's SSD, whose 16-bit numbers are mamba_ssm's, or fla's simple GLA, its write gated
    on the values where `values_gated`, `Call.values_gated`) it passes its values between launches in the
    operands' dtype, and an fp32 call's products take its operands as TF32 where its family's products
    take them transposed (mma.sync's), else in `FAST_PARTS` bf16 parts (`prepared`, `_parts`). A `ranged`
    16-bit delta rule, whose form proves no bound on its chunk inverse, solves it at fp32's range (`_merged`).
    A `skip` `[HV]` of a call without the delta rule adds `D v` to the output: in the output launch where the
    write gate takes the keys' side (`Call.gated_keys`, Mamba2's SSD), as mamba_ssm's output kernel adds it,
    else to the output after it, landed fp32 and rounded once, the launches holding the values gated. A
    `forked` call's documents, laid from chunks' edges, take their memories up from its table
    (`ForkedTable`). The delta rule's keys erase along reads of their own where `reads` `[B, T, H, Dk]` is
    given (`recurrence.Hooks.erase` 'read', the preconditioned delta rules'): `S' = (I - beta k b^T) D S +
    beta k v^T`, the system's rows and the weights the reads' (`beta b k^T`, `w = X (b beta exp g)`), the
    memory's writes and the scores the keys' (`_solved`; under a decay a key dimension `_paired` and
    `_key_weighted`). Where `normed` (`recurrence.OutputNorm`) is given, the output leaves through the gated
    RMS norm in the output launch's epilogue (`output.Normed`), its gate `gate` `[B, T, HV, Dv]` and `weight`
    `[Dv]`, in the operands' dtype."""
    out, final, _ = kept_forward(q, k, v, g, beta, h0, starts=starts, firsts=firsts,
                                 max_segments=max_segments, scale=scale, delta=delta, tiles=tiles, wide=wide,
                                 family=family, interpret=interpret, keeps=False,
                                 rounded=rounded, apart=apart, values_gated=values_gated,
                                 unit_pairs=unit_pairs,
                                 decay_bound=decay_bound, ranged=ranged, skip=skip, forked=forked,
                                 reads=reads, normed=normed, gate=gate, weight=weight)
    return out, final


def kept_forward(q, k, v, g, beta, h0=None, *, tiles: Tiles, starts=None, firsts=None,
                 max_segments: int | None = None,
                 scale: float | None = None, delta: bool = True, wide: bool = False,
                 family: mosaic.Family | None = None, interpret: bool = False,
                 keeps: bool = True,
                 rounded: bool = False, apart: int = 0, values_gated: bool = False, unit_pairs: bool = False,
                 decay_bound: float | None = None,
                     ranged: bool = False,
                 skip=None, forked: ForkedTable | None = None,
                 reads=None, normed: recurrence.OutputNorm | None = None, gate=None,
                 weight=None) -> tuple:
    """`forward`'s `(o, final)` and what the fast backward takes rather than compute again
    (`mainloop_backward.gradients`' `inverse` and `kept`): each chunk's inverse under a decay a head
    (`_solved`'s, None otherwise or where the call `keeps` none, `forward`'s), as fla's backward keeps its
    `A`, and the launches' weighted keys, memories, corrected values, a forked call's stored slots and under
    a decay a key dimension its prefix, `(w, h, corrected, stored, prefix)`, which the port keeps unless a
    remat rung recomputes them (`residuals.RECOMPUTED`), the caller's choice: the prefix, kept as fla_kda
    keeps its decay's cumulative sum, spares the backward's intra walk its launch, Kimi Linear's 162 us of an
    A100's 8056 us training (M-JS-212). Where `normed` is given (`forward`'s), the output is the epilogue's,
    and where the call `keeps` its residuals `(normalized, read back)`: the output the norm read, in its
    `read_in` dtype, which the norm's backward takes (`template.mosaic_rule`)."""
    assert normed is None or skip is None or not delta, "an epilogue follows a skip the output launch adds"
    run = prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                   tiles=tiles, family=family, interpret=interpret, keeps=keeps, firsts=firsts,
                   rounded=rounded, apart=apart, values_gated=values_gated, unit_pairs=unit_pairs,
                   decay_bound=decay_bound,
                       ranged=ranged, forked=forked,
                   out_dtype=jnp.float32 if wide else None, skipped=skip is not None, reads=reads)
    call, tiles, out_dtype = run.call, run.tiles, run.landed
    # The skip in the launch that holds the values as given (`Call.gated_keys`); where they hold them gated,
    # added to the output after it, landed fp32 (the output rounded once).
    fused = skip if call.gated_keys else None
    epilogue, sides = None, {}
    if normed is not None:
        epilogue = Normed(normed.epsilon, normed.activation, jnp.dtype(normed.read_in or jnp.float32).name,
                          run.value_width, kept=keeps)
        columns = call.value_width
        sides = {"gate": run.laid(zero_padded(gate, 3, columns)).reshape(call.rows, -1),
                 "weight": zero_padded(weight, 0, columns)}
    o = output_launch(run.q, run.k, run.h, run.corrected, run.prefix, run.taken, run.beta, call, tiles,
                      scale=run.scale, out_dtype=out_dtype, scores=run.scores, skip=fused, normed=epilogue,
                      **sides)
    final = run.final

    def unlaid(o):
        return run.unlaid(o.reshape(call.rows, call.heads, call.value_width))[..., :run.value_width]

    if epilogue is not None:
        assert fused is not None or skip is None, "the epilogue follows the skip"
        return (tuple(map(unlaid, o)) if epilogue.kept else unlaid(o)), final, (
            run.a, (run.w, run.h, run.corrected, run.stored, run.prefix if call.per_key else None))
    o = unlaid(o)
    if fused is None and skip is not None:
        o = recurrence.skip_added(o, skip, v, jnp.float32).astype(jnp.float32 if wide else run.dtype)
    assert final is not None, "a forward runs its states launch"
    prefix = run.prefix if call.per_key else None
    return o, final, (run.a, (run.w, run.h, run.corrected, run.stored, prefix))


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
    prefix: tuple
    """The decay's prefix within each chunk: under a decay a head its compensated halves, a head's row of
    vectors each `[HV, R]` (`algebra.prefix`'s pair); under one a key dimension its limbs `[R, HV * Dk]` each
    (`intra._prefix`'s, `Call.prefix_limbs`: its rounded sum alone where its decays are bounded)."""
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
    """Each row's entering memory `[B * HV * Dk, Dv]` fp32, as the states launch takes it."""
    table: Table | None = None
    """A forked call's table, as its walks take it (`tabled`)."""
    landed: jnp.dtype | None = None
    """The dtype the output launches land the output in (`prepared`'s)."""
    stored: jax.Array | None = None
    """A forked call's stored slots, the finals its storing documents left `[B * held * HV * Dk, Dv]` fp32
    (`mainloop._states`'), where its table holds any."""
    reads: jax.Array | None = None
    """The reads the delta rule's keys erase along, laid as the keys are (`forward`'s `reads`), where the
    call takes them."""

    def shaped(self, memory: jax.Array) -> jax.Array:
        """Memories `[B * HV * Dk, Dv]` as the call's `[B, HV, Dk, Dv]`."""
        width, columns = self.call.width, self.call.value_width
        return memory.reshape(self.batch, self.call.heads, width, columns)[:, :, :self.key_width,
                                                                            :self.value_width]


def _paired(q, k, decays, beta, taken, call, *, scale: float, solves: bool, prefix=None, reads=None) -> tuple:
    """The scores, the system's inverse less its unit diagonal where the call `solves` (`intra._intra`'s),
    and the decay's prefix (`intra._prefix`'s, or `prefix` where given, the forward's kept): the prefix's
    launch, then each sub-chunk's own pairs and their system's inverses a launch of their own
    (`own_pairs.intra_own`), then the walk placing the own scores and merging the inverse; the system's rows
    the `reads`' where given."""
    prefix = _prefix(decays, taken, call) if prefix is None else prefix
    products = ("scores", "system") if solves else ("scores",)
    own = own_pairs.intra_own(q, k, prefix, beta, taken, call, scale=scale, products=products, reads=reads)
    return (*_intra(q, k, prefix, beta, taken, own, call, scale=scale, solves=solves, reads=reads), prefix)


def _taken(dtype, family: mosaic.Family | None, *, per_key: bool, rounded: bool, delta: bool,
           skipped: bool) -> mosaic.Family:
    """The bundle a call of these forms issues by (`family`, the device's where None, `family_of`), refusing
    a call the mainloop does not take (`prepared`'s): a device no bundle claims, operands neither 16-bit nor
    fp32, a decay a key dimension's form it refuses (`per_key_refusal`), and a skip beside the delta rule,
    whose gradient no launch of the delta rule takes (the chunked op adds it after the template)."""
    family = family_of(dtype) if family is None else family
    if family is None:
        raise NotImplementedError(f"the mainloop has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)):
        raise NotImplementedError(f"the mainloop takes 16-bit and fp32 operands, not {dtype}")
    if per_key and (refused := per_key_refusal(dtype, rounded=rounded)):
        raise NotImplementedError(refused)
    if skipped and delta:
        raise NotImplementedError("the template takes the skip of a call without the delta rule alone, its "
                                  "reverse walk the skip's gradient (`mainloop_backward._reverse`); the "
                                  "delta rule's is added after it (`recurrence_chunk.chunk`)")
    return family


def _key_weighted(k, v, a, prefix: tuple, beta, taken, call, reads) -> tuple:
    """Under a decay a key dimension `(w, u, the keys decayed to their chunk's end)` (`mainloop._weighted`'s),
    the weights the `reads`' where given (`recurrence.Hooks.erase` 'read', `w = X (b beta exp g)`) and the
    decayed keys the keys' apart (the weighted launch without the delta rule's system)."""
    if reads is None:
        return _weighted(k, v, a, prefix, beta, taken, call)
    w, u, _ = _weighted(reads, v, a, prefix, beta, taken, call)
    (keys,) = _weighted(k, v, None, prefix, beta, taken, dataclasses.replace(call, delta=False))
    return w, u, keys


def _reads_taken(reads, k, *, delta: bool) -> None:
    """Refuses `reads` (`forward`'s) the mainloop takes none of: but of the keys' shape, under the delta
    rule."""
    if reads is not None and (not delta or reads.shape != k.shape):
        raise NotImplementedError("the mainloop's keys erase along reads of their own, of the keys' shape, "
                                  "under the delta rule alone: the portable math takes the rest")


def prepared(q, k, v, g, beta, h0, *, starts, max_segments, scale, delta, tiles, family, interpret,
             firsts=None, inverse=None, kept=None, keeps: bool = True, rounded: bool = False, apart: int = 0,
             values_gated: bool = False, unit_pairs: bool = False,
             decay_bound: float | None = None, ranged: bool = False, forked=None,
             out_dtype=None, skipped: bool = False, reads=None) -> _Run:
    """`forward`'s operands checked, padded and laid, and its launches before the output run: under a decay a
    head the solve's but where its `inverse` is given, and the weighted and states launches' but where their
    `kept` values, `(w, h, corrected, stored, prefix)`, are (`kept_forward`'s; its final memory then None;
    under a decay a key dimension the prefix's launch where `prefix` is), its output
    landing in `out_dtype` (the operands' where None), or in fp32 where a skip is added to it after the
    launches (`skipped`, `kept_forward`), so it rounds once (`_Run.landed`). A
    skip is a call's without the delta rule alone (`skipped` with `delta` refused). The solve stores its
    inverse where the call `keeps` it (`_solved`). A forked call's documents, laid as a packed call's are (its
    `starts`, `firsts` and `max_segments`), give `forked` (`tabled`): each row's walk takes up each document's
    memory from its table (`table.Table`). Where `reads` is given (`forward`'s), the solve and weighted
    launches form the system's rows and the weights from them (`_reads_taken`)."""
    dtype, per_key = jnp.result_type(q, k, v), g.ndim == 4
    _reads_taken(reads, k, delta=delta)
    family = _taken(dtype, family, per_key=per_key, rounded=rounded, delta=delta, skipped=skipped)
    batch, length, _, key_width = k.shape
    heads, value_width = v.shape[2:]
    scale = key_width ** -0.5 if scale is None else scale
    width, columns = padded_widths(family, key_width, value_width, kept=keeps, per_key=per_key, tiles=tiles)
    given_reads = () if reads is None else (reads,)
    q, k, *reads = (zero_padded(x.astype(dtype), 3, width) for x in (q, k, *given_reads))
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
    chunks, taken, sequences, laid, unlaid = _layout(batch, length, starts, max_segments, firsts,
                                                     chunk=tiles.chunk, forked=forked is not None)
    rows = chunks * tiles.chunk
    given = Operand(rows, _parts(dtype, family, rounded), dtype)
    # What the launches pass between them: the delta rule's in the operands' dtype, as fla's rounds it (the
    # port is its reference). Without it, a `rounded` call's (the SSD's under the run's fast forms, which its
    # route resolves, `recurrence_chunk.fast_rounding`) in the operands' dtype too, as mamba_ssm rounds its
    # scores and entered memories for its products (`_chunk_scan_fwd`'s `cb` and `prev_states`), its memory
    # summed and passed in fp32: on an H100 Nemotron-H 4B's SSD forward took 0.581 ms against fp32's three
    # parts' 2.031 and the port's 0.606, 30B's 0.249 against 0.593 and 0.229 (ssd-held-h100-fdab2424c); and
    # fla's simple GLA's, its write gated on the values (`values_gated`, `Call.values_gated`). Else,
    # and under the accurate forms, fp32's three parts, as the portable math holds them (a 16-bit call's
    # rounded put the output 1.7 times and the final memory up to 540 times the portable math's distance from
    # float64).
    held = given if delta or rounded else Operand(rows, product.bf16_part_count(), jnp.dtype(jnp.float32))
    if apart and (delta or per_key or tiles.chunk % apart):
        raise NotImplementedError("the mainloop reads a token's state apart from its group's writes under a "
                                  "decay a head without the delta rule alone, its chunk whole groups: the "
                                  "portable math takes the rest")
    # Interpreted, the target the interpreted bundle is of (`generation.interpreted_target`,
    # `mainloop.issued`), so a CPU's run takes the launches a plan for it takes.
    target = generation.interpreted_target() if interpret else generation.gpu_target()
    call = Call(heads, heads // k.shape[2], width, columns, rows, given, held, per_key, delta, family,
                interpret, target, tiles.chunk, ranged=ranged, apart=apart, values_gated=values_gated,
                unit_pairs=unit_pairs, decay_bound=decay_bound, producers=tiles.producers,
                residents=tiles.residents, keys=tiles.keys, alone=not keeps)
    _record_observed(call)
    table = None if forked is None else tabled(forked, width, columns)
    # A skip reads the values a launch holds as given (`Call.gated_keys`); else it is added after them.
    landed = jnp.dtype(jnp.float32 if skipped and not call.gated_keys else out_dtype or dtype)
    beta = laid(beta).reshape(rows, heads).T
    q, k, v, *reads = (given.split(laid(x).reshape(rows, -1)) for x in (q, k, v, *reads))
    # The system's rows and the weights the reads' where the call takes them, else the keys'.
    reads, rows_of = (reads[0], reads[0]) if reads else (None, k)
    a, w, u, keys, scores, system = None, None, v, k, None, []
    decays = laid(g).reshape(rows, -1)
    if per_key:
        # A decay a key dimension's prefix's limbs `[R, HV * Dk]` and its scores, and its system's inverse
        # where the call solves it (no inverse given), merged after the walk's pairs (`_paired`).
        solves = delta and kept is None and inverse is None
        scores, *system, prefix = _paired(q, k, decays, beta, taken, call, scale=scale, solves=solves,
                                          prefix=None if kept is None else kept[4], reads=reads)
    else:
        # A decay a head's halves, a head's row of vectors `[HV, R]` each.
        halves = algebra.prefix(decays.reshape(chunks, call.chunk, -1))
        prefix = tuple(x.reshape(rows, heads).T for x in halves)
    if per_key and delta and kept is not None:
        # The forward's inverse, weights, memories and corrected values given (`kept`): its scores and gated
        # keys alone formed again, the weighted launch without the delta rule's system.
        (keys,) = _weighted(k, v, None, prefix, beta, taken, dataclasses.replace(call, delta=False))
        a = inverse
    elif per_key and delta and inverse is not None:
        # The forward's inverse given without its states (a recomputed forward, `delta_memory`): its weights
        # from the inverse, the system's launch and the solve not repeated.
        a = inverse
        w, u, keys = _key_weighted(k, v, a, prefix, beta, taken, call, reads)
    elif per_key and delta:
        # The walk's inverse (`_paired`'s, merged from the own blocks' after the pairs).
        a = system[0]
        w, u, keys = _key_weighted(k, v, a, prefix, beta, taken, call, reads)
    elif per_key:
        (keys,) = _weighted(k, v, None, prefix, beta, taken, call)
    elif delta and inverse is None:
        # The forward's residuals come in pairs: a recomputed forward has neither.
        assert kept is None, "a kept forward keeps its inverse"
        if _solves_apart(call):
            (a,) = _solved(k, v, *prefix, beta, taken, call, applies=False, reads=reads)
            w, u = _weighted(rows_of, v, a, prefix, beta, taken, call)
        else:
            a, w, u = _solved(k, v, *prefix, beta, taken, call, keeps=keeps, reads=reads)
    elif delta:
        a = inverse
        if kept is None:
            w, u = _weighted(rows_of, v, a, prefix, beta, taken, call)
    h0 = h0.reshape(-1, columns)
    run = _Run(call, tiles, dtype, scale, key_width, value_width, batch, q, k, v, prefix, beta, a,
               w, keys, scores, None, None, None, taken, sequences, laid, unlaid, u=u, h0=h0,
               table=table, landed=landed, reads=reads)
    if kept is not None:
        # The forward's saved states, with the delta rule's weights or without them (`w` None).
        (w, h, corrected, stored, _) = kept
        return dataclasses.replace(run, w=w, h=h, corrected=corrected, stored=stored)
    h, corrected, final, stored = _states(keys, w, u, prefix, beta, h0, sequences, taken, call, tiles,
                                          run.table)
    return dataclasses.replace(run, h=h, corrected=corrected, final=run.shaped(final), stored=stored)
