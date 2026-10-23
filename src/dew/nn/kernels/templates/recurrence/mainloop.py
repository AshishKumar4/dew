"""The recurrence template's Mosaic GPU mainloop (design §6: one skeleton a template, over an instruction
family's bundle, `mosaic.Family`): the chunked rule's forward as fla's launches (`dew.nn.kernels.fla_gdn`,
`fla_kda`: the fused solve, the launch split, 64-token chunks), each program calling the algebra's tile
functions (`recurrence_algebra`) between the bundle's products, for every rule the chunked op's hooks name
(`recurrence.Hooks`) as values rather than code paths: a decay a head, a key dimension's or none (zero), with
or without the delta rule (without it the identity inverse: no solve, each token's write its values gated),
Mamba2's rated decay through its log gates. It names no instruction, layout or barrier: the call's family
(`FAMILIES`, the device's: mma.sync's, wgmma's) runs a program's roles (`roles`), streams its tiles into
shared memory (`stream`, `walk`), issues its products into fresh accumulators (`summed`, `products`), stages
a register operand for one (`staged`) and lays its tiles and vectors (`rows_of`, `columns_of`, `iota`); its
staging tiles' swizzle is Mosaic GPU's inference.

- solve (a decay a head, the delta rule): a chunk's `(I + A)^-1` for `A = tril(beta_s (k_s . k_t) exp2(g_s -
  g_t), -1)` (`algebra.system`), inverted by doubling on the chunk's tile (`algebra.doubled`): a family's
  product is a warpgroup's 64 rows, which hold no 16-row block of their own;
- intra (a decay a key dimension): each chunk's scores and system, every pair's decay its own
  (`algebra.key_decayed_row`), the system inverted in the same program under the delta rule;
- weighted: `w = A (k beta exp2(g))` and `u = A (v beta)` (`algebra.weighted`), and under a decay a key
  dimension the keys decayed to their chunk's end (`algebra.key_decayed_write`);
- states: each sequence's chunks in order over a value block, the memory `[Dk, BV]` carried in fp32 key
  slices of 64, each chunk storing the memory it enters and its corrected values (`algebra.corrected`,
  `scalar_write`, `carried`);
- output: each chunk's output from the memory it entered and its own pairs (`algebra.chunk_output`,
  `key_decayed_output`).

Every launch takes whole 64-token chunks: a packed or ragged call's tokens are laid out a chunk at a time
first, a chunk's rows past its tokens zero and its prefix past them held at its last token (fla's masked
loads), so a program's loads and stores are whole tiles; an aligned call's chunks are its rows as they are.
The decay's prefix is compensated, a `(hi, lo)` pair in base 2 within each chunk (`prefix`).

A 16-bit call's numbers are fla's (`fla_gdn.chunk_gated_delta_rule`'s): the values the launches pass (the
inverse, `w`, `u`, the memories and the corrected values) and the operands the algebra rounds are in its
dtype, the memory fp32 across chunks. An fp32 call keeps them fp32; as the tensor cores take 16-bit operands
alone (mma.sync's), an fp32 operand is three bf16 parts summing to it exactly (`_split`) and its product the
six of the parts' whose orders sum under three, an fp32 product's accuracy.

Decode is the states launch's program at one token: a program a (row, head, value block), the memory in from
the row's state and out to it (the states program's entered and final memories), its step the algebra's token
update (`recurrence.update`, the chunk form's one-token case) where a chunk's is `corrected` and `carried`,
between a prologue (the conv's halo and the gates) and an epilogue (the gated RMS norm) on the same bundle."""

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import precision
from dew.nn.inputs import cu_seqlens
from dew.nn.kernels import chunked_launch, generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import mosaic, mosaic_sm90
from dew.nn.kernels.templates.recurrence import _sm80

FAMILIES: dict[str, mosaic.Family] = {"mma.sync": _sm80.MMA_SYNC, "wgmma": mosaic_sm90.WGMMA}
"""The bundles the mainloop issues by, by the instruction family each claims (`generation.family`)."""

INTERPRETED = mosaic_sm90.WGMMA
"""The bundle Mosaic GPU's interpreter runs the mainloop by off a GPU of a family it claims (a CPU): wgmma's,
whose producer role, TMA rings and barriers the interpreter models (its products the dots they compute)."""


def family_of() -> mosaic.Family | None:
    """The bundle of the device's instruction family, None where the mainloop has none (a CPU's)."""
    return FAMILIES.get(generation.family(generation.device_generation()))

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


DEFAULT = {1: Tiles(64, 2), PARTS: Tiles(32, 1)}
"""The tiles by the parts an operand takes: an fp32 call's three hold one chunk in flight, 32 values wide, so
the states and output launches' tiles at 128-wide keys fit an A100's block (163 KB of shared memory)."""


def _split(x, parts: int = PARTS) -> tuple:
    """fp32 `x` as `parts` bf16 parts, largest first, each the last's remainder rounded (exact): three sum to
    `x` exactly, two to 16 of its significand's bits."""
    pieces, rest = [], x
    for _ in range(parts):
        piece = rest.astype(jnp.bfloat16)
        pieces.append(piece)
        rest = rest - piece.astype(jnp.float32)
    return tuple(pieces)


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

    def astype(self, dtype) -> "_Parts":
        """The tile as `dtype`'s, which a product takes it as: its own dtype alone."""
        if jnp.dtype(dtype) != self.dtype:
            raise TypeError(f"a {self.dtype} tile in shared memory taken as {jnp.dtype(dtype)}")
        return self

    def value(self) -> jax.Array:
        """The tile in registers, in its dtype: its parts summed in fp32, exactly."""
        if len(self.refs) == 1:
            return self.refs[0][...]
        return functools.reduce(jnp.add, (ref[...].astype(jnp.float32) for ref in self.refs))


class _Issuer:
    """The algebra's products (`algebra.Issue`) on the call's family: an operand a `_Parts` tile in shared
    memory, or a register value the family stages first (`staged`) into its product's `buffers` (by product
    name, a tile a part), an fp32 one as its parts, as many as its buffers (one, a cast to theirs). A product
    whose buffers are a list of such sets keeps each value it stages there for the products after it that
    take the same value, staging a new one into the set its last product read longest ago. Operands of three
    parts issue the products whose parts' orders sum under three, smallest first; of two, under two."""

    def __init__(self, call: "_Call", buffers: dict | None = None):
        self.call, self.buffers = call, buffers or {}
        # By product name: each set's value (held, so its identity stays its own) and its last use.
        self.held: dict = {}

    def _operand(self, x, name: str, other=None) -> tuple:
        if isinstance(x, _Parts):
            return x.refs
        if isinstance(x, algebra.Transposed):
            # Staged as it is and read transposed (`algebra.Transposed`).
            return tuple(ref.T for ref in self._operand(x.value, name, other))
        buffers = self.buffers[name]
        if isinstance(buffers, list):
            held = self.held.setdefault(name, [[None, -1] for _ in buffers])
            uses = max(use for _, use in held) + 1
            for place, (value, _) in enumerate(held):
                if value is x:
                    held[place][1] = uses
                    return buffers[place]
            free = [place for place in range(len(buffers)) if held[place][0] is not other]
            place = min(free, key=lambda place: held[place][1])
            held[place] = [x, uses]
            buffers = buffers[place]
        values = _split(x, len(buffers)) if len(buffers) > 1 else (x,)
        family = self.call.family
        return tuple(family.staged(value, buffer) for value, buffer in zip(values, buffers, strict=True))

    def __call__(self, name: str, lhs, rhs) -> jax.Array:
        left = self._operand(lhs, name, rhs)
        right = self._operand(rhs, name, lhs)
        shape = (lhs.shape[0], rhs.shape[1])
        count = max(len(left), len(right))
        terms = sorted(((i, j) for i in range(len(left)) for j in range(len(right)) if i + j < count),
                       key=lambda term: -sum(term))
        family, interpret = self.call.family, self.call.interpret

        def product(i: int, j: int) -> jax.Array:
            return family.summed(shape, lambda acc: family.products(acc, left[i], right[j], interpret),
                                 interpret)

        # Each part's product on its own, summed in fp32 smallest first: a tensor core's accumulation
        # truncates (mma.sync's), so a part accumulated onto a larger one loses its last bit where a sum
        # rounds it.
        return functools.reduce(jnp.add, (product(i, j) for i, j in terms))


def _diagonal(size: int, family: mosaic.Family) -> jax.Array:
    """A `[size, size]` tile's diagonal, in `family`'s layout."""
    shape = (size, size)
    return family.iota(shape, 0) == family.iota(shape, 1)


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

    def vectors(self, ref, row, column, columns: int) -> list:
        """Its parts' views of row `row`, `[columns]` from `column`."""
        return [ref.at[p * self.rows + row, pl.ds(column, columns)] for p in range(self.parts)]

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
    values passed between the launches' (`held`), whether its decay is a key dimension's (`per_key`), whether
    it runs the delta rule (`delta`), the bundle its launches issue by (`family`), and whether they are
    interpreted."""

    heads: int
    group: int
    width: int
    value_width: int
    rows: int
    given: _Operand
    held: _Operand
    per_key: bool
    delta: bool
    family: mosaic.Family
    interpret: bool

    @property
    def gates(self) -> _Operand:
        """A decay a key dimension's prefix halves, `[rows, heads * width]` fp32 each."""
        return _Operand(self.rows, 1, jnp.dtype(jnp.float32))

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


def _views(block: int | None = None, **parted) -> dict:
    """A launch's streams' views by stream part name (`Family.opened`'s `(ref, block)`), `parted[name]` a list
    of an operand's parts' views: each step `block` rows of its view, or the whole view in one step where
    None."""
    return {f"{name}{p}": (view, view.shape[0] if block is None else block) for name, views in parted.items()
            for p, view in enumerate(views)}


def _staging(operand: _Operand, *shapes) -> tuple:
    """A product site's staging tiles, a part of `operand`'s form each, for each of `shapes`."""
    return tuple(tuple(plgpu.SMEM(shape, operand.stored) for _ in range(operand.parts)) for shape in shapes)


def _merges(call: _Call) -> tuple:
    """The inverse's staging tiles (`algebra.doubled`'s 'merge' products, both operands fp32 registers): three
    sets, the system's kept across the levels and each level's inverse across its two products, each set a
    `[C, C]` tile an operand's part: an fp32 call's three bf16 parts, a 16-bit call's one fp16, 11 significand
    bits past the TF32 fla merges in (10) where the inverse is rounded to the call's 8 after."""
    dtype, parts = (jnp.bfloat16, PARTS) if call.held.parts > 1 else (jnp.float16, 1)
    tile = (CHUNK, CHUNK)
    return tuple(tuple(plgpu.SMEM(tile, dtype) for _ in range(parts)) for _ in range(3))


def _merging(staging, call: _Call) -> _Issuer:
    """The inverse's products on `_merges`' staging tiles."""
    return _Issuer(call, {"merge": list(staging)})


def _taken(index: int):
    """A chunk's program's liveness (`_kernel`'s `live`): its chunk, the `index`-th input's entry (`taken`),
    holds tokens."""
    return lambda refs: refs[index][lax.axis_index("chunks")] > 0


def _kernel(program, call: _Call, *, out_type, grid, grid_names, blocks: dict, stages: int, live,
            staging: tuple = (), once: dict | None = None):
    """`program(*refs, streams, staging)` as a kernel over `grid`, run where `live(refs)` holds (a laid
    call's chunks and sequences are as many as its worst case, most of them empty): its streams' allocations
    (of `blocks`, `stages` deep, and of `once`'s blocks, loaded once, one deep; `Family.opened` takes them
    with their views) and its staging tiles scoped, named for its launch (`recurrence_solve`, ...)."""
    family = call.family
    streams = {**family.streams(blocks, stages, 1), **family.streams(once or {}, 1, 1)}

    def entry(*arrays):
        def scoped(staged, *, streams):
            program(*arrays, streams=streams, staging=staged)

        # An empty program scopes nothing (Mosaic GPU's interpreter takes a scope's allocations in turn).
        pl.when(live(arrays))(lambda: family.scoped(scoped, staging, streams=streams))

    entry.__name__ = f"recurrence{program.__qualname__.split('.')[0]}"

    return family.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names, consumers=1,
                         interpret=call.interpret)


def _walk(streams, views, step, call: _Call):
    """The program's roles over its one step: the family's loads of `views` (`_views`) into `streams` (its
    allocations) and its walk of `step`, which stores what the program computes."""
    family = call.family
    streams = family.opened(streams, views)

    def consumer(_):
        family.walk(streams, ((step, 0, 1),), (), (), call.interpret)

    family.roles(1, consumer, lambda: family.stream(streams, views, 0, 1))


def _solve(k, hi, lo, beta, taken, call: _Call):
    """`(I + A)^-1` of each chunk and value head under a decay a head, `[R, HV * C]` as `call.held`, fla's
    kkt_solve in one program: the keys' products (`algebra.keyed`) inverted by doubling on the chunk's tile
    (`algebra.doubled`), its merges in fp16 in a 16-bit call (`_merges`)."""
    width, held, family = call.width, dataclasses.replace(call.held, rows=call.rows), call.family

    def program(k_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref, taken_ref, a_ref, *, streams,
                staging):
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        views = _views(keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width))

        def step(index, slots, carry, accumulators):
            del index
            keys = [_tile(slots, "keys", call.given, columns=pl.ds(c * BLOCK, BLOCK))
                    for c in range(width // BLOCK)]
            issue = _Issuer(call)
            product = functools.reduce(jnp.add, (issue("intra", key, key.T) for key in keys))
            rows = [family.rows_of(ref.at[h, span]) for ref in (hi_ref, lo_ref, beta_ref)]
            columns = [family.columns_of(ref.at[h, span]) for ref in (column_hi_ref, column_lo_ref)]
            system = algebra.keyed(product, *rows, *columns)
            # The inverse less its unit diagonal, which `algebra.weighted` adds outside its product.
            inverse = (algebra.doubled(-system, _merging(staging, call))
                       - jnp.where(_diagonal(CHUNK, family), 1.0, 0.0))
            held.store(a_ref, inverse, row, CHUNK, h * CHUNK, CHUNK)
            return carry, accumulators

        _walk(streams, views, step, call)

    # The prefix twice, read as a row's vectors and a column's: Mosaic GPU's layout inference takes two loads
    # of one view as one.
    return _kernel(program, call, out_type=held.shape(call.heads * CHUNK), grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=_streamed(keys=((CHUNK, width), call.given)),
                   stages=1, live=_taken(6), staging=_merges(call))(k, hi, lo, beta, hi, lo, taken)


def _intra(q, k, hi, lo, beta, taken, call: _Call, *, scale: float):
    """`((I + A)^-1, Aqk)` of each chunk and value head under a decay a key dimension, `[R, HV * C]` each as
    `call.held`: the chunk's scores and its keys' system, every pair's decay its own
    (`algebra.key_decayed_row` over the chunk's tokens at once, each row's pairs past its token masked after),
    and the system inverted (`algebra.doubled`): fla_kda's intra rows and solve, fused. Without the delta
    rule, `(Aqk,)` alone (the system computed and not inverted, its loop the scores')."""
    width, held, gates = call.width, dataclasses.replace(call.held, rows=call.rows), call.gates
    family = call.family

    def program(q_ref, k_ref, hi_ref, lo_ref, beta_ref, taken_ref, *outs, streams, staging):
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed, columns = h // call.group * width, pl.ds(h * width, width)
        views = _views(queries=call.given.views(q_ref, row, CHUNK, keyed, width),
                       keys=call.given.views(k_ref, row, CHUNK, keyed, width),
                       gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                       lows=gates.views(lo_ref, row, CHUNK, h * width, width))

        def earlier(j):
            """The chunk's `j`-th token's prefix halves and key, each a column's vector."""
            token = row + j
            key = functools.reduce(jnp.add, (family.columns_of(view).astype(jnp.float32)
                                             for view in call.given.vectors(k_ref, token, keyed, width)))
            gate, low = (family.columns_of(ref.at[token, columns]) for ref in (hi_ref, lo_ref))
            return gate, low, key

        def step(index, slots, carry, accumulators):
            del index
            query, key = (_tile(slots, name, call.given).value().astype(jnp.float32)
                          for name in ("queries", "keys"))
            gate, low = (_tile(slots, name, gates).value() for name in ("gates", "lows"))
            weighted = key * family.rows_of(beta_ref.at[h, span])[:, None]
            scores, system = algebra.key_decayed_row(query, weighted, gate, low, earlier, CHUNK - 1, CHUNK,
                                                     scale)
            rows, cols = (family.iota((CHUNK, CHUNK), dimension) for dimension in (0, 1))
            if call.delta:
                inverse = algebra.doubled(-jnp.where(rows > cols, system, 0.0), _merging(staging, call))
                held.store(outs[0], inverse - jnp.where(rows == cols, 1.0, 0.0), row, CHUNK, h * CHUNK, CHUNK)
            held.store(outs[-1], jnp.where(rows >= cols, scores, 0.0), row, CHUNK, h * CHUNK, CHUNK)
            return carry, accumulators

        _walk(streams, views, step, call)

    blocks = _streamed(queries=((CHUNK, width), call.given), keys=((CHUNK, width), call.given),
                       gates=((CHUNK, width), gates), lows=((CHUNK, width), gates))
    out_type = (held.shape(call.heads * CHUNK),) * (1 + call.delta)
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=blocks, stages=1, live=_taken(5),
                   staging=_merges(call) if call.delta else ())(q, k, hi, lo, beta, taken)


def _weighted(k, v, a, hi, lo, beta, taken, call: _Call):
    """`(w [R, HV * Dk], u [R, HV * Dv])` as `call.held` (`algebra.weighted`): the inverse against the decayed
    write-gated keys and against the write-gated values; under a decay a key dimension, also the keys decayed
    to their chunk's end, `[R, HV * Dk]` as `call.held` (`algebra.key_decayed_write`), which the memory takes,
    and those alone without the delta rule (`a` None)."""
    width, value_width, gates, family = call.width, call.value_width, call.gates, call.family
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(k_ref, v_ref, a_ref, hi_ref, lo_ref, beta_ref, taken_ref, *outs, streams, staging):
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        given = {"keys": call.given.views(k_ref, row, CHUNK, h // call.group * width, width)}
        if call.per_key:
            given.update(gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                         lows=gates.views(lo_ref, row, CHUNK, h * width, width))
        if call.delta:
            given.update(inverse=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                         values=call.given.views(v_ref, row, CHUNK, h * value_width, value_width))
        views = _views(**given)

        def step(index, slots, carry, accumulators):
            del index
            if call.delta:
                (scaled,) = staging
                issue = _Issuer(call, {"apply_u": scaled, "apply_w": scaled})
                inverse = _tile(slots, "inverse", held)
                betas = family.rows_of(beta_ref.at[h, span])
                for c in range(value_width // BLOCK):
                    tile = _tile(slots, "values", call.given, columns=pl.ds(c * BLOCK, BLOCK)).value()
                    u = algebra.weighted("apply_u", inverse, tile.astype(held.dtype), betas, issue)
                    held.store(outs[1], u, row, CHUNK, h * value_width + c * BLOCK, BLOCK)
                if not call.per_key:
                    decay = algebra.raised(family.rows_of(hi_ref.at[h, span]),
                                           family.rows_of(lo_ref.at[h, span]))
                    decay = lax.broadcast_in_dim(decay, (CHUNK, BLOCK), (0,))
            for c in range(width // BLOCK):
                block = pl.ds(c * BLOCK, BLOCK)
                tile = _tile(slots, "keys", call.given, columns=block).value()
                if call.per_key:
                    gate, low = (_tile(slots, name, gates, columns=block).value()
                                 for name in ("gates", "lows"))
                    decay = algebra.raised(gate, low)
                    ends = pl.ds(h * width + c * BLOCK, BLOCK)
                    last, last_low = (family.columns_of(ref.at[row + CHUNK - 1, ends])
                                      for ref in (hi_ref, lo_ref))
                    written = algebra.key_decayed_write(tile, gate, low, last, last_low)
                    held.store(outs[-1], written, row, CHUNK, h * width + c * BLOCK, BLOCK)
                if call.delta:
                    w = algebra.weighted("apply_w", inverse, tile.astype(held.dtype), betas, issue, decay)
                    held.store(outs[0], w, row, CHUNK, h * width + c * BLOCK, BLOCK)
            return carry, accumulators

        _walk(streams, views, step, call)

    given = {"keys": ((CHUNK, width), call.given)}
    if call.per_key:
        given.update(gates=((CHUNK, width), gates), lows=((CHUNK, width), gates))
    if call.delta:
        given.update(inverse=((CHUNK, CHUNK), held), values=((CHUNK, value_width), call.given))
    out_type = ((held.shape(call.heads * width), held.shape(call.heads * value_width)) * call.delta
                + (held.shape(call.heads * width),) * call.per_key)
    # Without the delta rule the inverse's place takes the keys, an operand no step reads.
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=_streamed(**given), stages=1, live=_taken(6),
                   staging=_staging(held, (CHUNK, BLOCK)) if call.delta else ())(
        k, v, k if a is None else a, hi, lo, beta, taken)


def _states(k, w, u, hi, lo, beta, h0, sequences, call: _Call, tiles: Tiles):
    """`(h [chunks * HV * Dk, Dv], corrected [R, HV * Dv], final [B * HV * Dk, Dv])`: the memory each chunk
    enters and the corrected values as `call.held`, and each row's final memory, fp32, each sequence's chunks
    in order over a value block (`algebra.corrected`, `carried`). Under the delta rule `u` is the weighted
    values and `w` the weighted keys (`_weighted`); without it (`w` None) `u` is the values, each chunk's
    write gated by `beta` (`algebra.weighted`'s identity inverse) and nothing corrected. A decay a head gates
    each chunk's write to its end (`algebra.scalar_write`); under one a key dimension `k` is the keys decayed
    so (`_weighted`'s third) and the memory's rows take their own decays. `sequences` `[N, 5]` is each
    sequence's first chunk, its chunks, its row and whether it begins and ends its row; `h0` `[B * HV * Dk,
    Dv]` fp32 each row's initial memory."""
    width, value_width, bv, family = call.width, call.value_width, tiles.block_v, call.family
    slices = width // BLOCK
    held, memories = dataclasses.replace(call.held, rows=call.rows), call.memories
    batch = h0.shape[0] // (call.heads * width)
    initial = _Operand(h0.shape[0], 1, jnp.dtype(jnp.float32))
    delta = w is not None
    values = held if delta else call.given
    # The keys the memory takes: a key head's, or a value head's decayed to their chunk's end.
    keys, keyed = (held, 1) if call.per_key else (call.given, call.group)

    def program(k_ref, w_ref, u_ref, hi_ref, lo_ref, beta_ref, h0_ref, seq_ref, h_ref, corrected_ref,
                final_ref, *, streams, staging):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        (staged,) = staging
        issue = _Issuer(call, {"corrected": staged, "write": staged})

        def memory(base, i):
            return (base * call.heads + h) * width + i * BLOCK

        # Each step a chunk of every row's operands, by its index among the call's chunks.
        views = _views(CHUNK, **({"w": held.views(w_ref, 0, call.rows, h * width, width)} if delta else {}),
                       keys=keys.views(k_ref, 0, call.rows, h // keyed * width, width),
                       values=values.views(u_ref, 0, call.rows, h * value_width + j * bv, bv))
        initial_views = _views(**{f"initial{i}": initial.views(h0_ref, memory(row, i), BLOCK, j * bv, bv)
                                  for i in range(slices)})
        chunks = family.opened({name: streams[name] for name in views}, views)
        entering = family.opened({name: streams[name] for name in initial_views}, initial_views)

        def step(n, slots, state, accumulators):
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(state):
                memories.store(h_ref, part, memory(n, i), BLOCK, j * bv, bv)
            tile = _tile(slots, "values", values).value()
            if delta:
                def weights(i):
                    return _tile(slots, "w", held, columns=pl.ds(i * BLOCK, BLOCK))

                corrected = algebra.corrected(tile, weights, state, held.dtype, issue)
            else:
                written = algebra.weighted("apply_u", None, tile, family.rows_of(beta_ref.at[h, span]), issue)
                corrected = algebra.corrected(written, None, state, held.dtype, issue)
            held.store(corrected_ref, corrected, n * CHUNK, CHUNK, h * value_width + j * bv, bv)
            if call.per_key:
                gated = corrected
                decay = [algebra.raised(*(family.rows_of(ref.at[last, pl.ds(h * width + i * BLOCK, BLOCK)])
                                          for ref in (hi_ref, lo_ref)))[:, None] for i in range(slices)]
            else:
                # The chunk's last prefix as a row's vector of it, each of whose entries is the decay a key
                # slice's rows take (Mosaic GPU's warpgroup lowering bitcasts no scalar, which
                # `algebra.raised` does).
                ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
                gated, decayed = algebra.scalar_write(corrected, ending, family.rows_of(hi_ref.at[h, span]),
                                                      ending_low, family.rows_of(lo_ref.at[h, span]))
                decay = [decayed[:, None]] * slices
            separate = None
            if held.parts > 1:
                # An fp32 call's last write outside the product (`algebra.carried`), held to the fp32
                # recurrence; a 16-bit call's is fla's product.
                def key_rows(i):
                    """The chunk's last key's slice `i`, a row's vector: its parts' rows summed, exactly."""
                    views = keys.vectors(k_ref, last, h // keyed * width + i * BLOCK, BLOCK)
                    return functools.reduce(jnp.add, (family.rows_of(view).astype(jnp.float32)
                                                      for view in views))

                ends = family.iota(gated.shape, 0) == CHUNK - 1
                separate = key_rows, jnp.sum(jnp.where(ends, gated, 0.0), axis=0)
            state = algebra.carried(
                state, lambda i: _tile(slots, "keys", keys, columns=pl.ds(i * BLOCK, BLOCK)), gated, decay,
                held.dtype, issue, last=separate)
            return state, accumulators

        def taken(index, slots, carry, accumulators):
            del index, carry
            # A finite memory times zero where the sequence begins no row (a select takes no scalar
            # condition over a tile in Mosaic GPU's warpgroup lowering).
            kept = jnp.where(begins > 0, 1.0, 0.0)
            return [_tile(slots, f"initial{i}", initial).value() * kept for i in range(slices)], accumulators

        def consumer(_):
            state = family.walk(entering, ((taken, 0, 1),), [family.zeros((BLOCK, bv))] * slices, (),
                                call.interpret)[0]
            state = family.walk(chunks, ((step, first, first + count),), state, (), call.interpret,
                                first=first)[0]

            @pl.when(ends > 0)
            def _():
                for i, part in enumerate(state):
                    final_ref.at[pl.ds(memory(row, i), BLOCK), pl.ds(j * bv, bv)][...] = part

        def producer():
            family.stream(entering, initial_views, 0, 1)
            family.stream(chunks, views, first, first + count)

        family.roles(1, consumer, producer)

    out_type = (memories.shape(value_width), held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(**({"w": ((CHUNK, width), held)} if delta else {}), keys=((CHUNK, width), keys),
                       values=((CHUNK, bv), values))
    initials = _streamed(**{f"initial{i}": ((BLOCK, bv), initial) for i in range(slices)})
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=tiles.stages,
                   live=lambda refs: refs[7][lax.axis_index("sequences"), 1] > 0,
                   staging=_staging(held, (BLOCK, bv)), once=initials)(
        k, k if w is None else w, u, hi, lo, beta, h0, sequences)


def _output(q, k, h, corrected, hi, lo, taken, call: _Call, tiles: Tiles, *, scale: float, out_dtype,
            scores=None):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype`: under a decay a head its own pairs' scores
    (`algebra.chunk_output`), under one a key dimension the intra launch's (`scores`, `_intra`;
    `algebra.key_decayed_output`)."""
    width, value_width, bv, gates = call.width, call.value_width, tiles.block_v, call.gates
    held, memories, family = dataclasses.replace(call.held, rows=call.rows), call.memories, call.family

    def program(q_ref, k_ref, h_ref, v_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, o_ref, *,
                streams, staging):
        j, n, h = lax.axis_index("blocks"), lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        if call.per_key:
            given = {"gates": gates.views(hi_ref, row, CHUNK, h * width, width),
                     "lows": gates.views(lo_ref, row, CHUNK, h * width, width),
                     "scores": held.views(k_ref, row, CHUNK, h * CHUNK, CHUNK)}
        else:
            given = {"keys": call.given.views(k_ref, row, CHUNK, h // call.group * width, width)}
        views = _views(queries=call.given.views(q_ref, row, CHUNK, h // call.group * width, width),
                       entered=memories.views(h_ref, (n * call.heads + h) * width, width, j * bv, bv),
                       values=held.views(v_ref, row, CHUNK, h * value_width + j * bv, bv), **given)

        def step(index, slots, carry, accumulators):
            del index

            def block(name, operand, c, *, along_rows=False):
                place = pl.ds(c * BLOCK, BLOCK)
                return _tile(slots, name, operand, **({"rows": place} if along_rows else {"columns": place}))

            entered = functools.partial(block, "entered", memories, along_rows=True)
            value = _tile(slots, "values", held)
            if call.per_key:
                def decays(c):
                    return tuple(block(name, gates, c).value() for name in ("gates", "lows"))

                out = algebra.key_decayed_output(
                    lambda c: block("queries", call.given, c).value(), decays, entered, width // BLOCK,
                    _tile(slots, "scores", held).value().astype(jnp.float32), value, scale,
                    _Issuer(call, {"memory": staged, "local": staged}))
            else:
                gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
                out = algebra.chunk_output(
                    functools.partial(block, "queries", call.given), entered,
                    functools.partial(block, "keys", call.given), width // BLOCK, gate, low, value, scale,
                    _Issuer(call, {"local": staged}), laid=(family.columns_of(column_hi_ref.at[h, span]),
                                                            family.columns_of(column_lo_ref.at[h, span])))
            o_ref.at[span, pl.ds(h * value_width + j * bv, bv)][...] = out.astype(o_ref.dtype)
            return carry, accumulators

        _walk(streams, views, step, call)

    if call.per_key:
        given = {"gates": ((CHUNK, width), gates), "lows": ((CHUNK, width), gates),
                 "scores": ((CHUNK, CHUNK), held)}
    else:
        given = {"keys": ((CHUNK, width), call.given)}
    blocks = _streamed(queries=((CHUNK, width), call.given), entered=((width, bv), memories),
                       values=((CHUNK, bv), held), **given)
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    # The prefix twice, as the solve's.
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, call.chunks, call.heads),
                   grid_names=("blocks", "chunks", "heads"), blocks=blocks, stages=1, live=_taken(8),
                   staging=_staging(held, (CHUNK, BLOCK)))(q, k if scores is None else scores, h, corrected,
                                                           hi, lo, hi, lo, taken)


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
            scale: float | None = None, delta: bool = True, tiles: Tiles | None = None, wide: bool = False,
            family: mosaic.Family | None = None, interpret: bool = False):
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
    attentions, GLA's with a decay a key dimension). Keys and values at most `WIDEST` wide, padded to whole
    blocks; under a decay a key dimension in an fp32 call at most a block wide (its tiles' three parts fill a
    block's shared memory past that). Its launches issue by `family`'s bundle (the device's, `family_of`,
    where None), interpreted where `interpret`."""
    run = _prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                    tiles=tiles, family=family, interpret=interpret)
    call, tiles = run.call, run.tiles
    o = _output(run.q, run.k, run.h, run.corrected, run.hi, run.lo, run.taken, call, tiles, scale=run.scale,
                out_dtype=jnp.float32 if wide else run.dtype, scores=run.scores)
    o = run.unlaid(o.reshape(call.rows, call.heads, call.value_width))[..., :run.value_width]
    return o, run.final


@dataclasses.dataclass(frozen=True)
class _Run:
    """A call's operands laid a chunk at a time and the forward's values before the output (`_prepared`):
    what the output launch and the backward's take."""

    call: _Call
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
    scores: jax.Array | None
    h: jax.Array
    corrected: jax.Array
    final: jax.Array
    taken: jax.Array
    sequences: jax.Array
    laid: Callable
    unlaid: Callable


def _prepared(q, k, v, g, beta, h0, *, starts, max_segments, scale, delta, tiles, family, interpret) -> _Run:
    """`forward`'s operands checked, padded and laid, and its launches before the output run."""
    family = family_of() if family is None else family
    if family is None:
        raise NotImplementedError(f"the mainloop has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    batch, length, _, key_width = k.shape
    heads, value_width = v.shape[2:]
    dtype = jnp.result_type(q, k, v)
    per_key = g.ndim == 4
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)):
        raise NotImplementedError(f"the mainloop takes 16-bit and fp32 operands, not {dtype}")
    widest = BLOCK if per_key and dtype == jnp.float32 else WIDEST
    if max(key_width, value_width) > widest:
        raise NotImplementedError(f"the mainloop takes keys and values at most {widest} wide here")
    scale = key_width ** -0.5 if scale is None else scale
    width, columns = (-(-x // BLOCK) * BLOCK for x in (key_width, value_width))
    q, k = (_padded(x.astype(dtype), 3, width) for x in (q, k))
    v = _padded(v.astype(dtype), 3, columns)
    g = _padded(g.astype(jnp.float32), 3, width) if per_key else g.astype(jnp.float32)
    beta = beta.astype(dtype).astype(jnp.float32)
    h0 = (jnp.zeros((batch, heads, width, columns), jnp.float32) if h0 is None
          else _padded(_padded(h0.astype(jnp.float32), 2, width), 3, columns))
    if starts is not None:
        starts = jnp.asarray(starts, bool)
        h0 = jnp.where(starts[:, 0, None, None, None], 0.0, h0)
    chunks, taken, sequences, laid, unlaid = _layout(batch, length, starts, max_segments)
    rows = chunks * CHUNK
    given = held = _Operand(rows, _parts(dtype), dtype)
    call = _Call(heads, heads // k.shape[2], width, columns, rows, given, held, per_key, delta, family,
                 interpret)
    hi, lo = prefix(laid(g).reshape(chunks, CHUNK, -1))
    # A decay a head's prefix a head's row of vectors `[HV, R]`, one a key dimension's tiles `[R, HV * Dk]`.
    hi, lo = (x.reshape(rows, -1) if per_key else x.reshape(rows, heads).T for x in (hi, lo))
    beta = laid(beta).reshape(rows, heads).T
    q, k, v = (given.split(laid(x).reshape(rows, -1)) for x in (q, k, v))
    w, u, keys, scores = None, v, k, None
    if per_key and delta:
        a, scores = _intra(q, k, hi, lo, beta, taken, call, scale=scale)
        w, u, keys = _weighted(k, v, a, hi, lo, beta, taken, call)
    elif per_key:
        (scores,) = _intra(q, k, hi, lo, beta, taken, call, scale=scale)
        (keys,) = _weighted(k, v, None, hi, lo, beta, taken, call)
    elif delta:
        a = _solve(k, hi, lo, beta, taken, call)
        w, u = _weighted(k, v, a, hi, lo, beta, taken, call)
    tiles = DEFAULT[held.parts] if tiles is None else tiles
    h, corrected, final = _states(keys, w, u, hi, lo, beta, h0.reshape(-1, columns), sequences, call, tiles)
    final = final.reshape(batch, heads, width, columns)[:, :, :key_width, :value_width]
    return _Run(call, tiles, dtype, scale, key_width, value_width, batch, q, k, v, hi, lo, beta,
                a if delta and not per_key else None, w, scores, h, corrected, final, taken, sequences, laid,
                unlaid)


def gradients(q, k, v, g, beta, h0, do, dfinal, *, starts=None, max_segments: int | None = None,
              scale: float | None = None, family: mosaic.Family | None = None, interpret: bool = False):
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call: `(dq, dk, dv, dg, dbeta, dh0)` for the output's and final memory's gradients `do`, `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_reverse`, `_query_key`, `_weighted_back`, the decay's gradient summed token by token
    (`algebra.summed_back`) and the memory's decay's term added, a key head's value heads summed. Its
    rounding is fla's; the accurate backward is the portable math's token adjoint."""
    run = _prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=True,
                    tiles=None, family=family, interpret=interpret)
    call = run.call
    if call.per_key or call.held.parts > 1:
        raise NotImplementedError("the fast backward takes a decay a head in a 16-bit call")
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width
    batch = run.batch
    laid_do = call.given.split(run.laid(_padded(do.astype(run.dtype), 3, columns)).reshape(rows, -1))
    dht = _padded(_padded(dfinal.astype(jnp.float32), 2, width), 3, columns).reshape(-1, columns)
    dh, dv2, dh0 = _reverse(run.q, run.k, run.w, laid_do, run.hi, run.lo, dht, run.sequences, call,
                            scale=run.scale)
    dq, dk, dw, decay = _query_key(run.q, run.k, run.corrected, laid_do, dv2, run.h, dh, run.hi, run.lo,
                                   run.taken, call, scale=run.scale)
    dk_w, dv, decay_w, dbeta = _weighted_back(run.a, run.k, run.v, dw, dv2, run.hi, run.lo, run.beta,
                                              run.taken, call)
    chunks = call.chunks

    def per_chunk(x):
        """`[R, HV * C]` as `[chunks, HV, C, C]`."""
        return jnp.moveaxis(x.reshape(chunks, CHUNK, heads, CHUNK), 2, 1)

    terms = per_chunk(decay.sum(0) + decay_w)
    diagonal = jnp.eye(CHUNK, dtype=bool)
    after = jnp.sum(jnp.where(diagonal, terms, 0.0), -1)
    pairs = jnp.where(diagonal, 0.0, terms)
    dg = jax.vmap(jax.vmap(algebra.summed_back))(pairs, after)
    # The memory's decay across each chunk: its gradient `sum(S dS)` through the decay `exp(g_last)` from
    # the chunk's start, every token's.
    memory = jnp.sum((run.h.astype(jnp.float32) * dh.astype(jnp.float32)).reshape(chunks, heads, -1), -1)
    ending = (run.hi + run.lo).reshape(heads, chunks, CHUNK)[..., -1]
    dg = dg + (memory * jnp.exp2(ending.T))[..., None]
    dg = jnp.moveaxis(dg, 1, 2).reshape(rows, heads)
    dbeta = jnp.sum(jnp.where(diagonal, per_chunk(dbeta), 0.0), -1)
    dbeta = jnp.moveaxis(dbeta, 1, 2).reshape(rows, heads)

    def headed(x, size):
        """`[R, HV * size]` laid back as `[B, T, HV, size]`."""
        return run.unlaid(x.reshape(rows, heads, size))

    def keyed(x):
        """A value head's gradient `[B, T, HV, Dk]` summed onto its key head's."""
        return x.reshape(*x.shape[:2], heads // call.group, call.group, width).sum(3)[..., :run.key_width]

    dq = keyed(headed(dq, width))
    dk = keyed(headed(dk, width) + headed(dk_w, width))
    dv = headed(dv, columns)[..., :run.value_width]
    dh0 = dh0.reshape(batch, heads, width, columns)[:, :, :run.key_width, :run.value_width]
    if starts is not None:
        dh0 = jnp.where(jnp.asarray(starts, bool)[:, 0, None, None, None], 0.0, dh0)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), run.unlaid(dg).astype(g.dtype),
            run.unlaid(dbeta).astype(beta.dtype), dh0)


# The fast backward of a decay a head under the delta rule (fla's DAG through the algebra's backward tile
# functions, `algebra.local_gradient` ... `algebra.summed_back`), a 16-bit call's: the memory's reverse
# recurrence a sequence at a time (`_reverse`), the queries', keys' and weights' gradients a key block of a
# chunk at a time (`_query_key`), and the WY representation's (`_weighted_back`); the decay's gradient summed
# token by token in XLA. A decay's terms leave a launch as a `[C, C]` tile a chunk and value head, its pairs
# strictly lower and each token's terms from its chunk's start on its diagonal, which `summed_back` never
# reads as a pair.


def _flipped(x, call: _Call):
    """`x` `[parts * R, columns]` with its chunks in reverse order: walked forward, a sequence's chunks from
    its last (a chunk `n`'s rows at chunk `chunks - 1 - n`)."""
    parts = x.shape[0] // call.rows
    return x.reshape(parts, call.chunks, CHUNK, -1)[:, ::-1].reshape(x.shape)


def _reverse(q, k, w, do, hi, lo, dht, sequences, call: _Call, *, scale: float):
    """`(dh [chunks * HV * Dk, Dv], dv [R, HV * Dv], dh0 [B * HV * Dk, Dv])`: the gradient of the memory each
    chunk leaves as `call.memories`, the corrected values' gradient as `call.held` (their own output's,
    `algebra.local_gradient`, and the memory's, `algebra.reverse_carried`) and each row's initial memory's
    gradient, fp32, each sequence's chunks from its last over a value block (fla's dhu and dv_local in one
    launch). `q`, `k`, `w` and `do` are laid as the forward's; `dht` `[B * HV * Dk, Dv]` fp32 each row's
    final memory's gradient."""
    width, value_width, family = call.width, call.value_width, call.family
    bv, slices = BLOCK, width // BLOCK
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    batch = dht.shape[0] // (call.heads * width)
    final = _Operand(dht.shape[0], 1, jnp.dtype(jnp.float32))
    chunks = call.chunks

    def program(q_ref, k_ref, w_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, dht_ref, seq_ref,
                dh_ref, dv_ref, dh0_ref, *, streams, staging):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        (staged,) = staging
        issue = _Issuer(call, dict.fromkeys(("local_dv", "values_dv", "reads", "weights_dv"), staged))

        def memory(base, i):
            return (base * call.heads + h) * width + i * BLOCK

        key_columns = h // call.group * width
        views = _views(CHUNK, queries=given.views(q_ref, 0, call.rows, key_columns, width),
                       keys=given.views(k_ref, 0, call.rows, key_columns, width),
                       w=held.views(w_ref, 0, call.rows, h * width, width),
                       out=given.views(do_ref, 0, call.rows, h * value_width + j * bv, bv))
        final_views = _views(**{f"final{i}": final.views(dht_ref, memory(row, i), BLOCK, j * bv, bv)
                                for i in range(slices)})
        walked = family.opened({name: streams[name] for name in views}, views)
        leaving = family.opened({name: streams[name] for name in final_views}, final_views)

        def step(f, slots, grad, accumulators):
            n = chunks - 1 - f
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(grad):
                memories.store(dh_ref, part, memory(n, i), BLOCK, j * bv, bv)

            def block(name, operand, i):
                return _tile(slots, name, operand, columns=pl.ds(i * BLOCK, BLOCK))

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            out = _tile(slots, "out", given)
            local = algebra.local_gradient(
                functools.partial(block, "keys", given), functools.partial(block, "queries", given), slices,
                gate, low, out, scale, issue, laid=(family.columns_of(column_hi_ref.at[h, span]),
                                                    family.columns_of(column_lo_ref.at[h, span])))
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            grad, dv = algebra.reverse_carried(
                grad, functools.partial(block, "keys", given), functools.partial(block, "w", held),
                functools.partial(block, "queries", given), out.value(), local, gate, low, ending,
                ending_low, scale, held.dtype, issue)
            held.store(dv_ref, dv, n * CHUNK, CHUNK, h * value_width + j * bv, bv)
            return grad, accumulators

        def taken(index, slots, carry, accumulators):
            del index, carry
            kept = jnp.where(ends > 0, 1.0, 0.0)
            return [_tile(slots, f"final{i}", final).value() * kept for i in range(slices)], accumulators

        start = chunks - first - count

        def consumer(_):
            grad = family.walk(leaving, ((taken, 0, 1),), [family.zeros((BLOCK, bv))] * slices, (),
                               call.interpret)[0]
            grad = family.walk(walked, ((step, start, start + count),), grad, (), call.interpret,
                               first=start)[0]

            @pl.when(begins > 0)
            def _():
                for i, part in enumerate(grad):
                    dh0_ref.at[pl.ds(memory(row, i), BLOCK), pl.ds(j * bv, bv)][...] = part

        def producer():
            family.stream(leaving, final_views, 0, 1)
            family.stream(walked, views, start, start + count)

        family.roles(1, consumer, producer)

    out_type = (memories.shape(value_width), held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                       w=((CHUNK, width), held), out=((CHUNK, bv), given))
    finals = _streamed(**{f"final{i}": ((BLOCK, bv), final) for i in range(slices)})
    flipped = [_flipped(x, call) for x in (q, k, w, do)]
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=2,
                   live=lambda refs: refs[9][lax.axis_index("sequences"), 1] > 0,
                   staging=_staging(held, (BLOCK, bv)), once=finals)(*flipped, hi, lo, hi, lo, dht,
                                                                     sequences)


def _query_key(q, k, corrected, do, dv, h, dh, hi, lo, taken, call: _Call, *, scale: float):
    """`(dq, dk [R, HV * Dk] fp32, dw [R, HV * Dk] as call.held, decay [blocks, R, HV * C] fp32)`: each
    chunk's and key block's queries', keys' and weights' gradients a value head's (fla's dqkwg,
    `algebra.query_key_gradients`), and its decay's terms over the block's key columns, the pairs strictly
    lower and the terms from the chunk's start on the diagonal. `h` and `dh` the memories each chunk enters
    and its leaving memory's gradient (`call.memories`)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    blocks = value_width // BLOCK

    def program(q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, hi_ref, lo_ref, column_hi_ref,
                column_lo_ref, taken_ref, dq_ref, dk_ref, dw_ref, decay_ref, *, streams, staging):
        c, n, h = lax.axis_index("blocks"), lax.axis_index("chunks"), lax.axis_index("heads")
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        (staged,) = staging
        key_columns = h // call.group * width + c * BLOCK
        memory = (n * call.heads + h) * width + c * BLOCK
        views = _views(queries=given.views(q_ref, row, CHUNK, key_columns, BLOCK),
                       keys=given.views(k_ref, row, CHUNK, key_columns, BLOCK),
                       values=held.views(v_ref, row, CHUNK, h * value_width, value_width),
                       out=given.views(do_ref, row, CHUNK, h * value_width, value_width),
                       value_grads=held.views(dv_ref, row, CHUNK, h * value_width, value_width),
                       entered=memories.views(h_ref, memory, BLOCK, 0, value_width),
                       leaving=memories.views(dh_ref, memory, BLOCK, 0, value_width))

        def step(index, slots, carry, accumulators):
            del index

            def column(name, operand, j):
                return _tile(slots, name, operand, columns=pl.ds(j * BLOCK, BLOCK))

            def whole(name, operand, i):
                del i
                return _tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            ((dq, dk, dw),), (pairs, after) = algebra.query_key_gradients(
                functools.partial(whole, "queries", given), functools.partial(whole, "keys", given), 1,
                functools.partial(column, "values", held), functools.partial(column, "out", given),
                functools.partial(column, "value_grads", held),
                lambda i, j: column("entered", memories, j), lambda i, j: column("leaving", memories, j),
                blocks, gate, low, ending, ending_low, None, scale, held.dtype,
                _Issuer(call, {"scores_dq": [staged]}), laid=(family.columns_of(column_hi_ref.at[h, span]),
                                                              family.columns_of(column_lo_ref.at[h, span])))
            out_columns = pl.ds(h * width + c * BLOCK, BLOCK)
            dq_ref.at[span, out_columns][...] = dq
            dk_ref.at[span, out_columns][...] = dk
            held.store(dw_ref, dw, row, CHUNK, h * width + c * BLOCK, BLOCK)
            decay_ref.at[c, span, pl.ds(h * CHUNK, CHUNK)][...] = jnp.where(_diagonal(CHUNK, family),
                                                                            after[:, None], pairs)
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(queries=((CHUNK, BLOCK), given), keys=((CHUNK, BLOCK), given),
                      values=((CHUNK, value_width), held), out=((CHUNK, value_width), given),
                      value_grads=((CHUNK, value_width), held), entered=((BLOCK, value_width), memories),
                      leaving=((BLOCK, value_width), memories))
    wide = jax.ShapeDtypeStruct((call.rows, call.heads * width), jnp.float32)
    out_type = (wide, wide, held.shape(call.heads * width),
                jax.ShapeDtypeStruct((width // BLOCK, call.rows, call.heads * CHUNK), jnp.float32))
    return _kernel(program, call, out_type=out_type, grid=(width // BLOCK, call.chunks, call.heads),
                   grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1, live=_taken(11),
                   staging=_staging(held, (CHUNK, CHUNK)))(q, k, corrected, do, dv, h, dh, hi, lo, hi, lo,
                                                           taken)


def _weighted_back(a, k, v, dw, du, hi, lo, beta, taken, call: _Call):
    """`(dk [R, HV * Dk], dv [R, HV * Dv], decay [R, HV * C], dbeta [R, HV * C])`, fp32: the WY
    representation's backward a chunk and value head at a time (fla's prepare_wy_repr_bwd,
    `algebra.weighted_gradient`) from `w`'s and `u`'s gradients `dw`, `du` (`call.held`), the inverse `a` less
    its unit diagonal (`_solve`'s); the decay's terms as `_query_key`'s, the write gate's gradient on a
    tile's diagonal."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, k_ref, v_ref, dw_ref, du_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref,
                taken_ref, dk_ref, dv_ref, decay_ref, dbeta_ref, *, streams, staging):
        n, h = lax.axis_index("chunks"), lax.axis_index("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       keys=given.views(k_ref, row, CHUNK, h // call.group * width, width),
                       values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                       w_grads=held.views(dw_ref, row, CHUNK, h * width, width),
                       u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            def column(name, operand, j):
                return _tile(slots, name, operand, columns=pl.ds(j * BLOCK, BLOCK))

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            names = ("inverse_t", "inverse_dx", "inverse_l", "keys_dl", "keys_dlt")
            key_grads, value_grads, d_beta, (pairs, after) = algebra.weighted_gradient(
                _tile(slots, "lower", held), functools.partial(column, "keys", given), width // BLOCK,
                functools.partial(column, "values", given), value_width // BLOCK,
                functools.partial(column, "w_grads", held), functools.partial(column, "u_grads", held), beta,
                gate, low, held.dtype, _Issuer(call, dict.fromkeys(names, staged)),
                laid=(family.columns_of(column_hi_ref.at[h, span]),
                      family.columns_of(column_lo_ref.at[h, span])))
            for i, grad in enumerate(key_grads):
                dk_ref.at[span, pl.ds(h * width + i * BLOCK, BLOCK)][...] = grad
            for j, grad in enumerate(value_grads):
                dv_ref.at[span, pl.ds(h * value_width + j * BLOCK, BLOCK)][...] = grad
            diagonal = _diagonal(CHUNK, family)
            decay_ref.at[span, pl.ds(h * CHUNK, CHUNK)][...] = jnp.where(diagonal, after[:, None], pairs)
            dbeta_ref.at[span, pl.ds(h * CHUNK, CHUNK)][...] = jnp.where(diagonal, d_beta[:, None], 0.0)
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), keys=((CHUNK, width), given),
                      values=((CHUNK, value_width), given), w_grads=((CHUNK, width), held),
                      u_grads=((CHUNK, value_width), held))

    def wide(columns):
        return jax.ShapeDtypeStruct((call.rows, call.heads * columns), jnp.float32)

    out_type = (wide(width), wide(value_width), wide(CHUNK), wide(CHUNK))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(10),
                   staging=_staging(held, (CHUNK, CHUNK)))(a, k, v, dw, du, hi, lo, beta, hi, lo, taken)
