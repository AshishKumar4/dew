"""A forked call's token walk over its table (the chunked op's `_walk`'s, `Chunk.loads`): each document's
first token enters the memory its table slot holds, and its last leaves its state in its stored slot, a
chunk's documents walked a segment at a time, the walk's replay and adjoint through the same hooks, so the
table's and the given memories' cotangents are the walk's: the chunked op's portable math's own seam."""

import jax
import jax.numpy as jnp

from dew.algebra import scan
from dew.nn.sharding import varying_like

from .chunked_launch import entry_write


def table_read(given, table, slot):
    """The memory at one-based table slot `slot` (`Chunk.loads`): `given`'s `[S', ...]` first, then
    `table`'s `[held, ...]` stored ones, zero at slot zero."""
    count = given.shape[0]
    taken = given[jnp.clip(slot - 1, 0, count - 1)]
    if table.shape[0]:
        taken = jnp.where(slot > count, table[jnp.clip(slot - count - 1, 0, table.shape[0] - 1)], taken)
    return jnp.where(slot > 0, taken, jnp.zeros((), taken.dtype))


def table_stored(table, slot, memory):
    """`table` `[held, ...]` with `memory` in one-based stored slot `slot`, as it was at slot zero."""
    if not table.shape[0]:
        return table
    at = jnp.maximum(slot - 1, 0)
    return table.at[at].set(jnp.where(slot > 0, memory.astype(table.dtype), table[at]))


def taken_up(h, added: bool = False, turned: bool = False):
    """A walk's memory entering (the chunked op's `_walk`'s `h`), and its forked operands where it walks a
    forked call (`h` a tuple: the given memories, each token's load, store and document (`Forked.tokens`,
    chunked) and, where given, each document's added slot (`added`, `Chunk.adds`), the entry write's operands
    and, `turned`, the load rotation's `(cos, sin)` each document's)."""
    if isinstance(h, tuple):
        given, loads, stores, begun, *rest = h
        adds, rest = (rest[0], rest[1:]) if added else (None, rest)
        turning = tuple(rest[-2:]) if turned else None
        entry = rest[:-2] if turned else rest
        return given, (loads, stores, begun, adds, tuple(entry) or None, turning)
    return h, None


def _opened(given, table, slot, adds, entry, document, turning=None):
    """The memory a document's first token enters (the chunked op's `_walk`): its table slot's (`table_read`),
    the slot it adds to it where the call gives them (`adds` `[D]`, `Chunk.adds`: zero, none), turned by its
    document's load rotation where the call gives one (`Forked.turning` `(cos, sin)` `[D, A]`: minus its own
    turns, so it reads in its own last token's frame, `templates.recurrence.turned`), and where the call gives
    one, its entry write (its keys turned so already), `weight sum_r key_r value_r^T` of document `document`'s
    operands (`entry` `(key [D, R, Dk], value [D, R, Dv], weight [D])`), formed for that document alone."""
    taken = table_read(given, table, slot)
    if adds is not None:
        taken = taken + table_read(given, table, adds[document])
    if turning is not None:
        taken = scan.rotated(taken, *(part[document] for part in turning), axis=-2)
    if entry is None:
        return taken
    key, value, weight = (x[document] for x in entry)
    # The weight inside the contraction, so its gradient is one too (`dot_general`'s accumulation): summed
    # after, the SSD port's took 1.6 to 1.9 times the error each document alone takes from float64.
    # Its terms summed aligned to the largest, which may leave fp32's range where the sum is in it.
    written = entry_write(key, value, weight)
    return taken + written.astype(taken.dtype)


def _hooks(given, entry, adds, loads, stores, begun, turning=None):
    """A forked chunk's token walk's own hooks (`Derived.replay`'s `walked`), its walk carrying the table and
    the call's given memories and entry operands beside it, so their cotangents are the walk's: before a
    document's first token, the state it enters its memory (`_opened`); after its last, its state left in
    its stored slot (`table_stored`)."""
    def before(t, h, carried):
        table, given, entry = carried
        taken = _opened(given, table, loads[t], adds, entry, begun[t], turning)
        return jnp.where(loads[t] >= 0, taken.astype(h.dtype), h), carried

    def after(t, h, carried):
        table, given, entry = carried
        return table_stored(table, stores[t], h), given, entry

    return before, after


def forked_steps(forms, operands, given, forked, held: int, documents: int, decayed, segments):
    """A forked walk forward (the chunked op's `_walk`), its documents in place: each chunk's outputs, the
    memory entering each chunk, the row's last memory, and the walk's table at its end. The walk takes the
    row's segments in token order, a segment a document's tokens within one chunk, each from the memory
    entering it: the chunk's own where its document continued from the chunk before, else its document's,
    taken up from the table at its first token (`_opened`, the table as the segments before left it, so a
    document stores and a later one of the same chunk loads). Each segment's outputs and the state it leaves
    are its window of the chunk's forms (`Derived.prepare`'s `window`), the state stored in its document's
    slot where the segment ends a storing document, and the chunk's next memory where it ends the chunk. The
    segments are at most the chunks and the documents, each one window's work, as many as a row laid a chunk a
    document took; no value is sized by the documents' bound. Its scan states its trips, the chunks and the
    bound's documents (a count of its work, `instrumentation.step_flops`, and a checkify of a vmap of it
    hold), but walks a window only below `segments`, every row's most (`most_segments`, outside the rows' map,
    so a step past them is skipped, not selected away); a row's own steps past its last segment change
    nothing. Every step walking a window, a 2048-token row of three documents at chunk 64 took 1290.8 ms of
    armada's CPU at a bound of 2048 against 43.2 at its own (row 7952); a while_loop stopping at the last
    segment counted nothing and broke a checkify of a vmap (train 1239, #3785)."""
    loads, stores, begun, adds, entry, turning = forked
    chunks, count = loads.shape
    place = jnp.arange(count)
    starts = loads >= 0
    # Each segment's first token, in token order: each document's first, and a chunk's first where its
    # document continues; past the row's segments, a sentinel past its tokens.
    marks = (starts | (place == 0)).reshape(-1)
    # One place past the segments' most: the walk stops at its first sentinel, each segment's end the next
    # origin (the row's end at its last).
    origins = jnp.nonzero(marks, size=chunks + documents + 1, fill_value=chunks * count)[0]
    # Every carry typed as each operand that reaches it varies (a checked `shard_map`'s manual axes): the
    # table's given memories, the chunks' operands (a laid erase's corrections among them) and the fork's.
    reaching = (given, *jax.tree.leaves(operands), *jax.tree.leaves(forked))
    decayed = varying_like(decayed, *reaching)
    zero_decay, zero = jnp.zeros_like(decayed), varying_like(jnp.zeros_like(given[0]), *reaching)

    def segment(carried, first, after):
        """The walk past the segment from token place `first` up to `after` (the next segment's origin, or
        the row's end), a real one: the loop stops at the first sentinel."""
        (chunk_in, held_chunk, leaving), table, outputs, entering = carried
        # A row's sentinel past its own segments walks its last chunk's first window, its result dropped.
        chunk, origin = jnp.minimum(first // count, chunks - 1), first % count
        through = jnp.where(after // count == chunk, after % count - 1, count - 1)
        # Entering a new chunk, its memory is the one the chunk before left.
        new = chunk != held_chunk
        memory_in = jax.tree.map(lambda a, b: jnp.where(new, a, b), leaving, chunk_in)
        entering = entering.at[chunk].set(jnp.where(new, memory_in[1], entering[chunk]))
        continuing = ~starts[chunk, origin]
        memory = jnp.where(continuing, memory_in[1],
                           _opened(given, table, loads[chunk, origin], adds, entry, begun[chunk, origin],
                                   turning))
        window = forms.prepare(jax.tree.map(lambda x: x[chunk], operands), window=(origin, through))
        decay = jnp.where(continuing, memory_in[0], zero_decay)
        left, kept = forms.step((decay, memory), window)
        out = forms.mapped(window, (decay, memory), kept)
        rows = (place >= origin) & (place <= through)
        outputs = outputs.at[chunk].set(jnp.where(rows[:, None], out, outputs[chunk]))
        table = table_stored(table, stores[chunk, through], left[1])
        ends = through == count - 1
        leaving = jax.tree.map(lambda a, b: jnp.where(ends, a, b), left, leaving)
        return (memory_in, chunk, leaving), table, outputs, entering

    # A forked walk's forms are a single state's, which keep nothing for their outputs (a pair walks none).
    shaped = jax.eval_shape(lambda chunk: forms.mapped(forms.prepare(chunk), (decayed, zero), None),
                            jax.tree.map(lambda x: x[0], operands))
    table = varying_like(jnp.zeros((held, *given.shape[1:]), given.dtype), *reaching)
    start = (((decayed, zero), varying_like(jnp.asarray(-1), origins), (decayed, zero)), table,
             varying_like(jnp.zeros((chunks, *shaped.shape), shaped.dtype), *reaching),
             varying_like(jnp.zeros((chunks, *zero.shape), zero.dtype), *reaching))
    def walked(carried, at):
        first = origins[at]
        moved = segment(carried, first, origins[at + 1])
        return jax.tree.map(lambda new, old: jnp.where(first < chunks * count, new, old), moved, carried)

    def step(carried, at):
        return jax.lax.cond(at < segments, walked, lambda carried, at: carried, carried, at), None

    ((_, _, last), table, outputs, entering), _ = jax.lax.scan(step, start, jnp.arange(chunks + documents))
    return outputs, entering, last[1], table


def most_segments(h, added: bool, turned: bool):
    """A forked walk's most segments over its rows and heads (`forked_steps`'), each chunk's first token
    and each document's, from its loads `[rows, heads, chunks, C]` (`taken_up`); None for an unforked
    walk. The skip it bounds runs where the fork's layout is the same for the whole map: under an outer
    vmap over differing layouts it is batched there, and the walk's conditional runs both branches, as the
    scan before it did (correct, its saving gone)."""
    _, forked = taken_up(h, added, turned)
    if forked is None:
        return None
    loads = forked[0]
    marks = (loads >= 0) | (jnp.arange(loads.shape[-1]) == 0)
    return jnp.max(jnp.sum(marks, axis=(-2, -1)))


def forked_replay(replay, operands, given, forked, held: int):
    """A forked walk token by token (the chunked op's `_walk`'s replay): its outputs and the row's last
    memory."""
    loads, stores, begun, adds, entry, turning = forked
    # The carries typed as every operand that reaches them varies (a checked `shard_map`'s manual axes).
    reaching = (given, *jax.tree.leaves(operands), *jax.tree.leaves(forked))
    table = varying_like(jnp.zeros((held, *given.shape[1:]), given.dtype), *reaching)

    def chunked(carried, chunk):
        h, table = carried
        chunk, load, store, begin = chunk
        before, after = _hooks(given, entry, adds, load, store, begin, turning)
        outputs, h, (table, _, _) = replay(chunk, h, (before, after, (table, given, entry)))
        return (h, table), outputs

    (h, _), outputs = jax.lax.scan(chunked, (varying_like(jnp.zeros_like(given[0]), *reaching), table),
                                   (operands, loads, stores, begun))
    return outputs, h


def forked_adjoint(backward, operands, given, forked, entering, table, d_outputs, d_final):
    """A forked walk backward (the chunked op's `_walk`'s adjoint), from the row's last memory: each chunk's
    token walk from the memory entering it differentiated with its hooks (`_hooks`), the table it reads the
    walk's last (each slot stored once, before every document that loads it), so a stored slot's cotangent
    gathers each document that took it up and joins the walk's at the token that stored it, and a given
    memory's and an entry write's operands' are their documents'. Returns the chunks' cotangents and the
    walk's operands' (`(given, loads, stores, begun, adds?, entry?, turning?)`, the integers held in their
    place, the load rotation's cotangent none: a turned call's turns take theirs in the crossing form,
    `templates.recurrence.turned`)."""
    loads, stores, begun, adds, entry, turning = forked

    def chunked(carried, chunk):
        d_h, d_walked = carried
        chunk, h, d_out, load, store, begin = chunk
        before, after = _hooks(given, entry, adds, load, store, begin, turning)
        d_entering, d_chunk, d_walked = backward(chunk, (None, h), d_h, d_out,
                                                 (before, after, (table, given, entry), d_walked))
        return (d_entering, d_walked), d_chunk

    zeros = jax.tree.map(jnp.zeros_like, (table, given, entry))
    (_, (_, d_given, d_entry)), d_operands = jax.lax.scan(
        chunked, (d_final, zeros), (operands, entering, d_outputs, loads, stores, begun), reverse=True)
    return d_operands, (d_given, loads, stores, begun, *(() if adds is None else (adds,)),
                        *(() if entry is None else d_entry),
                        *(() if turning is None else tuple(jnp.zeros_like(part) for part in turning)))
