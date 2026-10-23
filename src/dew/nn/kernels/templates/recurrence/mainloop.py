"""The recurrence template's Mosaic GPU mainloop (design §6: one skeleton a template, over an instruction
family's bundle, `mosaic.Family`): the gated delta rule's forward, a decay a head, as fla's four launches
(`dew.nn.kernels.fla_gdn`: its fused solve, its launch split, 64-token chunks), each program calling the
algebra's tile functions (`recurrence_algebra`) between the bundle's products. It names no instruction, layout
or barrier: the family streams a program's tiles into shared memory (`stream`, `walk`), issues its products
(`product`), stages a register operand for one (`stage`) and lays its tiles and vectors (`rows_of`, `iota`).

- solve: a chunk's `(I + A)^-1` for `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)` (`algebra.solved`),
  the chunk one block: a family's product is a warpgroup's 64 rows, which hold no 16-row block of their own,
  so the forward substitution runs over the chunk's rows and no blocks merge;
- weighted: `w = A (k beta exp2(g))` and `u = A (v beta)` (`algebra.weighted`);
- states: each sequence's chunks in order over a value block, the memory `[Dk, BV]` carried in fp32 key
  slices of 64, each chunk storing the memory it enters and its corrected values (`algebra.corrected`,
  `scalar_write`, `carried`);
- output: each chunk's output from the memory it entered and its own pairs (`algebra.chunk_output`).

Every launch takes whole 64-token chunks: a packed or ragged call's tokens are laid out a chunk at a time
first, a chunk's rows past its tokens zero and its prefix past them held at its last token (fla's masked
loads), so a program's loads and stores are whole tiles; an aligned call's chunks are its rows as they are.
The decay's prefix is compensated, a `(hi, lo)` pair in base 2 within each chunk (`prefix`).

The tensor cores take 16-bit operands alone (mma.sync's), so an fp32 operand is three bf16 parts summing to it
exactly (`_split`) and its product the six of the parts' whose orders sum under three, an fp32 product's
accuracy. `exact` keeps every value between the launches and every product's operand the algebra rounds to
fp32 (the inverse, `w`, `u`, the memories, the corrected values and the scores), as the token-by-token
recurrence computes them; without it a 16-bit call rounds them to its own dtype, as fla does
(`fla_gdn.chunk_gated_delta_rule`'s numbers)."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import precision
from dew.nn.inputs import cu_seqlens
from dew.nn.kernels import chunked_launch, recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.recurrence import _sm80

FAMILY: mosaic.Family = _sm80.MMA_SYNC
"""The bundle the mainloop issues by."""

CHUNK = 64
"""Tokens a chunk, fla's."""

BLOCK = 64
"""A tile's width: the keys' and values' column blocks, and the memory's key slices."""

WIDEST = 128
"""The widest keys and values a call takes: a program's tiles in shared memory at 128 fit an A100's block."""

LOG2_E = math.log2(math.e)
"""log2(e) to double precision, the prefix's base-2 scale (fla's RCP_LN2 is 1.93e-8 short of it)."""

PARTS = 3
"""The bf16 parts of an fp32 operand: 8 of its significand's 24 bits each."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A forward's tiles: the value block the states and output launches split the values into (`block_v`),
    and the chunks the states launch has in flight (`stages`)."""

    block_v: int
    stages: int


DEFAULT = {1: Tiles(64, 2), PARTS: Tiles(64, 1)}
"""The tiles by the parts of the states launch's widest operand (three hold one chunk in flight)."""


def _split(x) -> tuple:
    """fp32 `x` as three bf16 parts summing to it exactly, largest first: each rounding's remainder is
    exact."""
    first = x.astype(jnp.bfloat16)
    rest = x - first.astype(jnp.float32)
    second = rest.astype(jnp.bfloat16)
    return first, second, (rest - second.astype(jnp.float32)).astype(jnp.bfloat16)


def _parts(dtype) -> int:
    """The parts a product takes an operand of `dtype` as: three for fp32, one for 16 bits."""
    return PARTS if jnp.dtype(dtype) == jnp.float32 else 1


class _Parts:
    """A tile in shared memory as the 16-bit parts its products take (one, or an fp32 tile's three, `_split`),
    each a view of a stream's slot, and `dtype` the tile's own."""

    def __init__(self, refs, dtype):
        self.refs, self.dtype = tuple(refs), jnp.dtype(dtype)

    @property
    def shape(self) -> tuple:
        return self.refs[0].shape

    @property
    def T(self) -> "_Parts":
        return _Parts((ref.T for ref in self.refs), self.dtype)

    def value(self) -> jax.Array:
        """The tile in registers, in its dtype: its parts summed in fp32, exactly."""
        if len(self.refs) == 1:
            return self.refs[0][...]
        return functools.reduce(jnp.add, (ref[...].astype(jnp.float32) for ref in self.refs))


class _Issuer:
    """The algebra's products (`algebra.Issue`) on the family: an operand a `_Parts` tile in shared memory, or
    a register value the family stages first (`stage`) into its product's `buffers` (by product name, a tile a
    part), an fp32 one as its parts. Operands of three parts issue the products whose parts' orders sum under
    three, smallest first."""

    def __init__(self, buffers: dict | None = None):
        self.buffers = buffers or {}

    def _operand(self, x, name: str) -> tuple:
        if isinstance(x, _Parts):
            return x.refs
        values = _split(x) if _parts(x.dtype) > 1 else (x,)
        buffers = self.buffers[name]
        return tuple(FAMILY.stage(value, buffer) for value, buffer in zip(values, buffers, strict=True))

    def __call__(self, name: str, lhs, rhs) -> jax.Array:
        left, right = self._operand(lhs, name), self._operand(rhs, name)
        shape = (lhs.shape[0], rhs.shape[1])
        count = max(len(left), len(right))
        terms = sorted(((i, j) for i in range(len(left)) for j in range(len(right)) if i + j < count),
                       key=lambda term: -sum(term))
        return functools.reduce(jnp.add, (FAMILY.product(left[i], right[j], shape) for i, j in terms))


def _diagonal(size: int) -> jax.Array:
    """A `[size, size]` tile's diagonal, in the family's layout."""
    shape = (size, size)
    return FAMILY.iota(shape, 0) == FAMILY.iota(shape, 1)


def _across(vector: jax.Array) -> jax.Array:
    """A column's vector laid as a row's (`algebra.Across`): its diagonal's row sums, exactly."""
    return jnp.sum(jnp.where(_diagonal(vector.shape[0]), vector[None, :], 0.0), axis=1)


def _down(vector: jax.Array) -> jax.Array:
    """A row's vector laid as a column's, `_across`'s inverse. Two loads of one view in the two layouts are
    one load to Mosaic GPU's layout inference, which then finds none for it."""
    return jnp.sum(jnp.where(_diagonal(vector.shape[0]), vector[:, None], 0.0), axis=0)


@dataclasses.dataclass(frozen=True)
class _Operand:
    """A launch's operand in global memory, `[parts * rows, columns]`: its parts (`_split`) stacked `rows`
    apart, each `[rows, columns]`, and `dtype` its value's (fp32 where three parts)."""

    rows: int
    parts: int
    dtype: jnp.dtype

    @property
    def stored(self):
        """The dtype its parts are stored in."""
        return jnp.bfloat16 if self.parts > 1 else self.dtype

    def views(self, ref, row, rows: int, column, columns: int) -> list:
        """Its parts' views `[rows, columns]` from `row` and `column`."""
        return [ref.at[pl.ds(p * self.rows + row, rows), pl.ds(column, columns)] for p in range(self.parts)]

    def store(self, ref, value, row, rows: int, column, columns: int) -> None:
        """`value` `[rows, columns]` as its parts at `row` and `column`."""
        pieces = _split(value.astype(jnp.float32)) if self.parts > 1 else (value,)
        for view, piece in zip(self.views(ref, row, rows, column, columns), pieces, strict=True):
            view[...] = piece.astype(view.dtype)

    def shape(self, columns: int) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.parts * self.rows, columns), self.stored)

    def split(self, x):
        """`x` `[rows, columns]` in its dtype as its stacked parts."""
        return jnp.concatenate(_split(x.astype(jnp.float32)), 0) if self.parts > 1 else x.astype(self.dtype)


@dataclasses.dataclass(frozen=True)
class _Call:
    """A call's static geometry: value heads over key heads (`group`), key and value widths (whole blocks),
    the chunk-laid rows (`rows`), the inputs' operand form (`given`, the queries', keys' and values') and the
    values passed between the launches' (`held`), and whether it is interpreted."""

    heads: int
    group: int
    width: int
    value_width: int
    rows: int
    given: _Operand
    held: _Operand
    interpret: bool

    @property
    def chunks(self) -> int:
        return self.rows // CHUNK

    @property
    def memories(self) -> _Operand:
        """The memories the chunks enter, `[chunks * heads * width, value_width]` as `held`."""
        return dataclasses.replace(self.held, rows=self.chunks * self.heads * self.width)


def _streamed(**blocks) -> dict:
    """Each named stream's parts' blocks, `name0`, `name1`, ...: `blocks[name]` its tile's shape and its
    `_Operand`."""
    return {f"{name}{p}": (shape, operand.stored) for name, (shape, operand) in blocks.items()
            for p in range(operand.parts)}


def _tile(slots, name: str, operand: _Operand, rows=None, columns=None) -> _Parts:
    """Stream `name`'s slot (its parts `name0`, `name1`, ...) as a `_Parts`, at a block of `rows` or `columns`
    (`pl.ds`) where given."""
    def view(ref):
        if rows is not None:
            ref = ref.at[rows, :]
        return ref if columns is None else ref.at[:, columns]

    return _Parts((view(slots[f"{name}{p}"]) for p in range(operand.parts)), operand.dtype)


def _views(**parted) -> dict:
    """A stream step's views by stream part name, `parted[name]` a list of an operand's parts' views."""
    return {f"{name}{p}": view for name, views in parted.items() for p, view in enumerate(views)}


def _staging(operand: _Operand, *shapes) -> tuple:
    """A product site's staging tiles, a part of `operand`'s form each, for each of `shapes`."""
    return tuple(tuple(plgpu.SMEM(shape, operand.stored, transforms=_sm80.swizzled(*shape, operand.stored))
                       for _ in range(operand.parts)) for shape in shapes)


def _kernel(program, call: _Call, *, out_type, grid, grid_names, blocks: dict, stages: int,
            staging: tuple = ()):
    """`program(*refs, streams, staging)` as a kernel over `grid`: its streams of `blocks` (`stages` deep) and
    its staging tiles scoped."""
    def entry(*arrays):
        def scoped(streams, staged):
            program(*arrays, streams=FAMILY.opened(streams), staging=staged)

        pl.run_scoped(scoped, FAMILY.streams(blocks, stages, 1), staging)

    return FAMILY.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names, consumers=1,
                         interpret=call.interpret)


def _walk(streams, views, lo, hi, step, start, call: _Call):
    """The program's one role: the family's loads of `views` and its walk of `step` from `start` over steps
    `lo` to `hi` (at least one)."""
    out = []

    def producer():
        FAMILY.stream(streams, views, lo, hi)

    def consumer(_):
        out.append(FAMILY.walk(streams, ((step, lo, hi),), start, (), call.interpret)[0])

    FAMILY.roles(0, 1, consumer, producer)
    return out[0]


def _solve(k, hi, lo, beta, taken, call: _Call):
    """`(I + A)^-1` of each chunk and value head, `[R, HV * C]` as `call.held` (`algebra.solved`)."""
    width, held = call.width, dataclasses.replace(call.held, rows=call.rows)

    def program(k_ref, hi_ref, lo_ref, beta_ref, taken_ref, a_ref, *, streams, staging):
        del staging
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)

        def views(_):
            return _views(keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width))

        def step(index, slots, carry, accumulators):
            del index

            def keys(i, c):
                del i
                return _tile(slots, "keys", call.given, columns=pl.ds(c * BLOCK, BLOCK))

            gate, low, betas = (FAMILY.rows_of(ref.at[h, span]) for ref in (hi_ref, lo_ref, beta_ref))
            inverse = algebra.solved(keys, [gate], [low], [betas], width // BLOCK, _Issuer(),
                                     laid=([_down(gate)], [_down(low)]), across=_across)[0, 0]
            held.store(a_ref, inverse, row, CHUNK, h * CHUNK, CHUNK)
            return carry, accumulators

        @pl.when(taken_ref[n] > 0)
        def _():
            _walk(streams, views, 0, 1, step, (), call)

    return _kernel(program, call, out_type=held.shape(call.heads * CHUNK), grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=_streamed(keys=((CHUNK, width), call.given)),
                   stages=1)(k, hi, lo, beta, taken)


def _weighted(k, v, a, hi, lo, beta, taken, call: _Call):
    """`(w [R, HV * Dk], u [R, HV * Dv])` as `call.held` (`algebra.weighted`): the inverse against the decayed
    write-gated keys and against the write-gated values."""
    width, value_width = call.width, call.value_width
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(k_ref, v_ref, a_ref, hi_ref, lo_ref, beta_ref, taken_ref, w_ref, u_ref, *, streams, staging):
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)

        def views(_):
            return _views(inverse=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                          keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width),
                          values=call.given.views(v_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index
            (scaled,) = staging
            issue = _Issuer({"apply_u": scaled, "apply_w": scaled})
            inverse = _tile(slots, "inverse", held)
            betas = FAMILY.rows_of(beta_ref.at[h, span])
            decay = algebra.raised(FAMILY.rows_of(hi_ref.at[h, span]), FAMILY.rows_of(lo_ref.at[h, span]))
            decay = lax.broadcast_in_dim(decay, (CHUNK, BLOCK), (0,))
            for name, columns, out, base, decayed in (("values", value_width, u_ref, h * value_width, None),
                                                      ("keys", width, w_ref, h * width, decay)):
                for c in range(columns // BLOCK):
                    tile = _tile(slots, name, call.given, columns=pl.ds(c * BLOCK, BLOCK)).value()
                    product = algebra.weighted("apply_w" if decayed is not None else "apply_u", inverse,
                                               tile.astype(held.dtype), betas, issue, decayed)
                    held.store(out, product, row, CHUNK, base + c * BLOCK, BLOCK)
            return carry, accumulators

        @pl.when(taken_ref[n] > 0)
        def _():
            _walk(streams, views, 0, 1, step, (), call)

    blocks = _streamed(inverse=((CHUNK, CHUNK), held), keys=((CHUNK, width), call.given),
                       values=((CHUNK, value_width), call.given))
    out_type = (held.shape(call.heads * width), held.shape(call.heads * value_width))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=blocks, stages=1,
                   staging=_staging(held, (CHUNK, BLOCK)))(k, v, a, hi, lo, beta, taken)


def _states(k, w, u, hi, lo, h0, sequences, call: _Call, tiles: Tiles):
    """`(h [chunks * HV * Dk, Dv], corrected [R, HV * Dv], final [B * HV * Dk, Dv])`: the memory each chunk
    enters and the corrected values as `call.held`, and each row's final memory, fp32, each sequence's chunks
    in order over a value block (`algebra.corrected`, `scalar_write`, `carried`). `sequences` `[N, 5]` is each
    sequence's first chunk, its chunks, its row and whether it begins and ends its row; `h0` `[B * HV * Dk,
    Dv]` fp32 each row's initial memory."""
    width, value_width, bv = call.width, call.value_width, tiles.block_v
    slices = width // BLOCK
    held, memories = dataclasses.replace(call.held, rows=call.rows), call.memories
    batch = h0.shape[0] // (call.heads * width)
    initial = _Operand(h0.shape[0], 1, jnp.dtype(jnp.float32))

    def program(k_ref, w_ref, u_ref, hi_ref, lo_ref, h0_ref, seq_ref, h_ref, corrected_ref, final_ref, *,
                streams, staging):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        (staged,) = staging
        issue = _Issuer({"corrected": staged, "write": staged})

        def memory(base, i):
            return (base * call.heads + h) * width + i * BLOCK

        def views(index):
            row_ = (first + index) * CHUNK
            return _views(w=held.views(w_ref, row_, CHUNK, h * width, width),
                          keys=call.given.views(k_ref, row_, CHUNK, h // call.group * width, width),
                          values=held.views(u_ref, row_, CHUNK, h * value_width + j * bv, bv))

        def step(index, slots, state, accumulators):
            n = first + index
            span = pl.ds(n * CHUNK, CHUNK)
            for i, part in enumerate(state):
                memories.store(h_ref, part, memory(n, i), BLOCK, j * bv, bv)
            corrected = algebra.corrected(
                _tile(slots, "values", held).value(),
                lambda i: _tile(slots, "w", held, columns=pl.ds(i * BLOCK, BLOCK)), state, held.dtype, issue)
            held.store(corrected_ref, corrected, n * CHUNK, CHUNK, h * value_width + j * bv, bv)
            # The chunk's last prefix as a row's vector of it, each of whose entries is the decay a key
            # slice's rows take (Mosaic GPU's warpgroup lowering bitcasts no scalar, which `algebra.raised`
            # does).
            last, last_lo = (jnp.full((CHUNK,), ref[h, n * CHUNK + CHUNK - 1]) for ref in (hi_ref, lo_ref))
            gated, decay = algebra.scalar_write(corrected, last, FAMILY.rows_of(hi_ref.at[h, span]), last_lo,
                                                FAMILY.rows_of(lo_ref.at[h, span]))
            state = algebra.carried(
                state, lambda i: _tile(slots, "keys", call.given, columns=pl.ds(i * BLOCK, BLOCK)), gated,
                [decay[:, None]] * slices, held.dtype, issue)
            return state, accumulators

        initials = _streamed(**{f"initial{i}": ((BLOCK, bv), initial) for i in range(slices)})

        def entered(allocated):
            def initial_views(_):
                return _views(**{f"initial{i}": initial.views(h0_ref, memory(row, i), BLOCK, j * bv, bv)
                                 for i in range(slices)})

            def taken(index, slots, carry, accumulators):
                del index, carry
                # A finite memory times zero where the sequence begins no row (a select takes no scalar
                # condition over a tile in Mosaic GPU's warpgroup lowering).
                kept = jnp.where(begins > 0, 1.0, 0.0)
                return ([_tile(slots, f"initial{i}", initial).value() * kept for i in range(slices)],
                        accumulators)

            return _walk(FAMILY.opened(allocated), initial_views, 0, 1, taken,
                         [FAMILY.zeros((BLOCK, bv))] * slices, call)

        @pl.when(count > 0)
        def _():
            state = pl.run_scoped(entered, FAMILY.streams(initials, 1, 1))
            state = _walk(streams, views, 0, count, step, state, call)

            @pl.when(ends > 0)
            def _():
                for i, part in enumerate(state):
                    final_ref.at[pl.ds(memory(row, i), BLOCK), pl.ds(j * bv, bv)][...] = part

    out_type = (memories.shape(value_width), held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(w=((CHUNK, width), held), keys=((CHUNK, width), call.given),
                       values=((CHUNK, bv), held))
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=tiles.stages,
                   staging=_staging(held, (BLOCK, bv)))(k, w, u, hi, lo, h0, sequences)


def _output(q, k, h, corrected, hi, lo, taken, call: _Call, tiles: Tiles, *, scale: float, out_dtype):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype` (`algebra.chunk_output`)."""
    width, value_width, bv = call.width, call.value_width, tiles.block_v
    held, memories = dataclasses.replace(call.held, rows=call.rows), call.memories

    def program(q_ref, k_ref, h_ref, v_ref, hi_ref, lo_ref, taken_ref, o_ref, *, streams, staging):
        j, n, h = lax.axis_index("blocks"), lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (scores,) = staging

        def views(_):
            return _views(queries=call.given.views(q_ref, row, CHUNK, h // call.group * width, width),
                          keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width),
                          entered=memories.views(h_ref, (n * call.heads + h) * width, width, j * bv, bv),
                          values=held.views(v_ref, row, CHUNK, h * value_width + j * bv, bv))

        def step(index, slots, carry, accumulators):
            del index

            def block(name, operand, c, *, along_rows=False):
                place = pl.ds(c * BLOCK, BLOCK)
                return _tile(slots, name, operand, **({"rows": place} if along_rows else {"columns": place}))

            gate, low = FAMILY.rows_of(hi_ref.at[h, span]), FAMILY.rows_of(lo_ref.at[h, span])
            out = algebra.chunk_output(
                functools.partial(block, "queries", call.given),
                functools.partial(block, "entered", memories, along_rows=True),
                functools.partial(block, "keys", call.given), width // BLOCK, gate, low,
                _tile(slots, "values", held), scale, _Issuer({"local": scores}),
                laid=(_down(gate), _down(low)))
            o_ref.at[span, pl.ds(h * value_width + j * bv, bv)][...] = out.astype(o_ref.dtype)
            return carry, accumulators

        @pl.when(taken_ref[n] > 0)
        def _():
            _walk(streams, views, 0, 1, step, (), call)

    blocks = _streamed(queries=((CHUNK, width), call.given), keys=((CHUNK, width), call.given),
                       entered=((width, bv), memories), values=((CHUNK, bv), held))
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, call.chunks, call.heads),
                   grid_names=("blocks", "chunks", "heads"), blocks=blocks, stages=1,
                   staging=_staging(held, (CHUNK, CHUNK)))(q, k, h, corrected, hi, lo, taken)


def prefix(g):
    """The decay's prefix within each chunk, compensated, in base 2: `g` `[..., C, H]` (natural log) to `(hi,
    lo)`, each token's `g log2(e)` an exact pair (its product with log2(e)'s fp32 rounding error-free,
    `precision.two_product`, plus its product with the rounding's remainder) summed compensated
    (`precision.compensated_add`), so a difference of two positions is as precise as its own range."""
    scale = np.float32(LOG2_E)
    product, error = precision.two_product(g, jnp.full_like(g, scale), bit_split=True)
    error = error + g * float(np.float32(LOG2_E - float(scale)))
    return lax.associative_scan(precision.compensated_add, (product, error), axis=g.ndim - 2)


def _padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size`."""
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


def _layout(batch: int, length: int, starts, max_segments: int | None):
    """A call's tokens a chunk at a time: `(chunks, taken [chunks], sequences [N, 5], laid, unlaid)`, each
    chunk's tokens, each sequence's first chunk, chunks, row and whether it begins and ends its row
    (`_states`), and the maps from `[B, T, ...]` to the chunks' rows `[chunks * C, ...]` (a chunk's rows past
    its tokens zero) and back. An aligned call's chunks are its rows' own, unmoved."""
    tokens = batch * length
    if starts is None and length % CHUNK == 0:
        chunks = tokens // CHUNK
        sequences = np.stack([np.arange(batch) * (length // CHUNK), np.full(batch, length // CHUNK),
                              np.arange(batch), np.ones(batch), np.ones(batch)], 1)

        def laid(x):
            return x.reshape(tokens, *x.shape[2:])

        def unlaid(x):
            return x.reshape(batch, length, *x.shape[1:])

        return chunks, jnp.full((chunks,), CHUNK, jnp.int32), jnp.asarray(sequences, jnp.int32), laid, unlaid
    if starts is None:
        spans = chunked_launch.sequences(batch, length, chunk=CHUNK)
    else:
        documents = length if max_segments is None else max_segments
        spans = chunked_launch.sequences(batch, length, cu_seqlens(starts, batch * (documents + 1)),
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


def forward(q, k, v, g, beta, h0=None, *, starts=None, max_segments: int | None = None,
            scale: float | None = None, exact: bool = True, tiles: Tiles | None = None, wide: bool = False,
            interpret: bool = False):
    """The gated delta rule's forward on this mainloop, a decay a head: `(o [B, T, HV, Dv], final [B, HV, Dk,
    Dv])` for `q`, `k` `[B, T, H, Dk]`, `v` `[B, T, HV, Dv]`, the natural log decay `g` `[B, T, HV]`, the
    write gate `beta` `[B, T, HV]` and each row's initial memory `h0` `[B, HV, Dk, Dv]` (zero where None), as
    `recurrence_chunk.fla_rule` takes them: `starts` `[B, T]` marks packed documents' first tokens, at most
    `max_segments` a row (the row's length where None, which the chunks' layout is sized by), each a sequence
    of its own from a zero memory but a row's first, which takes `h0` unless its first token begins a
    document; the queries scaled by `scale` (`Dk^-1/2` where None); the output in the operands' common dtype,
    fp32 where `wide`, the final memory fp32. `exact` keeps the values the launches pass and the algebra's
    rounded operands fp32 (module docstring). Keys and values at most `WIDEST` wide, padded to whole blocks.
    """
    batch, length, _, key_width = k.shape
    heads, value_width = v.shape[2:]
    dtype = jnp.result_type(q, k, v)
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)):
        raise NotImplementedError(f"the mainloop takes 16-bit and fp32 operands, not {dtype}")
    if max(key_width, value_width) > WIDEST:
        raise NotImplementedError(f"the mainloop takes keys and values at most {WIDEST} wide")
    scale = key_width ** -0.5 if scale is None else scale
    width, columns = (-(-x // BLOCK) * BLOCK for x in (key_width, value_width))
    q, k = (_padded(x.astype(dtype), 3, width) for x in (q, k))
    v = _padded(v.astype(dtype), 3, columns)
    beta = beta.astype(dtype).astype(jnp.float32)
    h0 = (jnp.zeros((batch, heads, width, columns), jnp.float32) if h0 is None
          else _padded(_padded(h0.astype(jnp.float32), 2, width), 3, columns))
    if starts is not None:
        starts = jnp.asarray(starts, bool)
        h0 = jnp.where(starts[:, 0, None, None, None], 0.0, h0)
    chunks, taken, sequences, laid, unlaid = _layout(batch, length, starts, max_segments)
    rows = chunks * CHUNK
    inner = jnp.dtype(jnp.float32) if exact else dtype
    given, held = _Operand(rows, _parts(dtype), dtype), _Operand(rows, _parts(inner), inner)
    call = _Call(heads, heads // k.shape[2], width, columns, rows, given, held, interpret)
    hi, lo = prefix(laid(g.astype(jnp.float32)).reshape(chunks, CHUNK, heads))
    hi, lo, beta = (x.reshape(rows, heads).T for x in (hi, lo, laid(beta)))
    q, k, v = (given.split(laid(x).reshape(rows, -1)) for x in (q, k, v))
    a = _solve(k, hi, lo, beta, taken, call)
    w, u = _weighted(k, v, a, hi, lo, beta, taken, call)
    tiles = DEFAULT[held.parts] if tiles is None else tiles
    h, corrected, final = _states(k, w, u, hi, lo, h0.reshape(-1, columns), sequences, call, tiles)
    o = _output(q, k, h, corrected, hi, lo, taken, call, tiles, scale=scale,
                out_dtype=jnp.float32 if wide else dtype)
    o = unlaid(o.reshape(rows, heads, columns))[..., :value_width]
    return o, final.reshape(batch, heads, width, columns)[:, :, :key_width, :value_width]
