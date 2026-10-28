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
  product is a warpgroup's 64 rows, which hold no 16-row block of their own; and in the same program the
  weighted keys and values below, the inverse applied in registers (`_solved`);
- intra (a decay a key dimension): each chunk's scores and system, each pair's decay factored over its
  sub-chunks on the tensor cores (`algebra.key_decayed_pairs`, which the call's declared floor bounds), the
  system inverted in the same program under the delta rule;
- weighted: `w = A (k beta exp2(g))` and `u = A (v beta)` (`algebra.weighted`) from a stored inverse (a
  decay a key dimension's, the backward's recomputed forward), and under a decay a key dimension the keys
  decayed to their chunk's end (`algebra.key_decayed_write`);
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

Decode is not on the mainloop: a one-token step takes `recurrence.step` (row 1424 for its Mosaic form)."""

import contextlib
import contextvars
import dataclasses
import functools
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import precision
from dew.nn.inputs import cu_seqlens
from dew.nn.kernels import chunked_launch, generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import bundles, mosaic, mosaic_sm80, mosaic_sm90, resources
from dew.nn.kernels.templates.gemm import cost

INTERPRETED = mosaic_sm90.WGMMA
"""The bundle Mosaic GPU's interpreter runs the mainloop by off a GPU of a family it claims (a CPU): wgmma's,
whose producer role, TMA rings and barriers the interpreter models (its products the dots they compute)."""


def bundle_dtype(dtype) -> jnp.dtype:
    """The 16-bit operands a call of `dtype`'s products take on a bundle (`bundles.of`'s dtype): its own, an
    fp32 call's bf16 parts."""
    return jnp.dtype(jnp.bfloat16 if jnp.dtype(dtype) == jnp.float32 else dtype)


def family_of(dtype=jnp.bfloat16) -> mosaic.Family | None:
    """The bundle of the device's instruction family for a call of `dtype` (`bundles.of`), None where the
    mainloop has none (a CPU's)."""
    return bundles.of(generation.device_generation(), bundle_dtype(dtype))

CHUNK = 64
"""Tokens a chunk, fla's."""


def block_of(family: mosaic.Family) -> int:
    """A tile's width on `family`'s bundle: the keys' and values' column blocks and the memory's key slices,
    the widest depth its products step over in 16-bit operands (`Family.depths`), whose bf16 parts an fp32
    call's products take too."""
    return family.depths(jnp.bfloat16)[0]

LOG2_E = math.log2(math.e)
"""log2(e) to double precision, the prefix's base-2 scale (fla's RCP_LN2 is 1.93e-8 short of it)."""

PARTS = 3
"""The bf16 parts of an fp32 operand: 8 of its significand's 24 bits each."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A forward's tiles, the recurrence template's config, derived from the call's shape and the device's
    and its bundle's facts (`templates.recurrence.template.Mosaic.derived`): the value block the states
    launch splits the values into (`block_v`), the chunks it has in flight (`stages`), and the output
    launch's value block (`block_o`), the states launch's where None."""

    block_v: int
    stages: int
    block_o: int | None = None

    @property
    def output_block(self) -> int:
        return self.block_v if self.block_o is None else self.block_o


BF16_LARGEST = float(jnp.finfo(jnp.bfloat16).max)
"""bf16's largest finite value, `(2 - 2^-7) 2^127`, an fp32 value."""


def _split(x, parts: int = PARTS) -> tuple:
    """fp32 `x` as `parts` bf16 parts, largest first, each the last's remainder rounded (exact): three sum to
    `x` exactly, two to 16 of its significand's bits. The first saturates at bf16's largest (an fp32 value
    past it rounds to infinity in bf16, its remainder then NaN), the next parts taking the difference."""
    pieces, rest = [], x
    for _ in range(parts):
        piece = jnp.clip(rest, -BF16_LARGEST, BF16_LARGEST).astype(jnp.bfloat16)
        pieces.append(piece)
        rest = rest - piece.astype(jnp.float32)
    return tuple(pieces)


def per_key_refusal(dtype, floor: float | None) -> str | None:
    """Why the mainloop refuses a call of `dtype` under a decay a key dimension, None where it takes it: its
    intra launch factors each pair's decay over the chunk's sub-chunks (`algebra.key_decayed_pairs`), which
    needs the layer's declared `floor` bounding a sub-chunk's range (`algebra.factored_range`), and rounds
    each side's gate once, where the fp32 recurrence rounds each pair's (1.98 times its distance from
    float64 in an fp32 call, fam-t)."""
    if floor is None or not algebra.factored_range(floor):
        return f"a decay a key dimension needs a declared floor its sub-chunks factor under, not {floor}"
    if jnp.dtype(dtype) == jnp.float32:
        return "an fp32 call's decay a key dimension is the portable math's"
    return None


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
        if isinstance(x, algebra.Exact):
            # One part, in the product's own tile for exact operands (`{name}.exact`).
            (buffer,) = self.buffers[f"{name}.exact"]
            return (self.call.family.staged(x.value, buffer),)
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

    def store(self, ref, value, row, rows: int, column, columns: int, *, family: mosaic.Family,
              smem, wait: bool = True) -> None:
        """`value` `[rows, columns]` as its parts at `row` and `column`, through `smem` (a `[rows, columns]`
        tile of its stored dtype the launch lands its stores in, `_landed`), its last copy left in flight
        unless `wait` (`Family.stored`, `_walk_stores`)."""
        pieces = _split(value.astype(jnp.float32)) if self.parts > 1 else (value,)
        family.stored([piece.astype(self.stored) for piece in pieces],
                      self.views(ref, row, rows, column, columns), smem=smem, wait=wait)

    def landing(self, rows: int, columns: int) -> tuple:
        """A store's `landing` entry (`_kernel`): a `[rows, columns]` tile in its stored dtype."""
        return (rows, columns), self.stored

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
    target: generation.GpuTarget

    @property
    def gates(self) -> _Operand:
        """A decay a key dimension's prefix halves, `[rows, heads * width]` fp32 each."""
        return _Operand(self.rows, 1, jnp.dtype(jnp.float32))

    @property
    def chunks(self) -> int:
        return self.rows // CHUNK

    @property
    def block(self) -> int:
        """A tile's width (`block_of`)."""
        return block_of(self.family)

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


def _views(index_map=None, /, **parted) -> dict:
    """A launch's streams' views by stream part name (`Family.opened`'s `(ref, index_map)`, `mosaic.viewed`),
    `parted[name]` a list of an operand's parts' views: step `i` the `i`-th tile of the stream's rows down
    its view, as wide as the view (the whole view where the stream's tile is), or the tile `index_map(i)`
    names."""
    return {f"{name}{p}": (view, _down if index_map is None else index_map) for name, views in parted.items()
            for p, view in enumerate(views)}


def _column(call: _Call, slots, name: str, operand: _Operand, j):
    """Stream `name`'s slot's `j`-th block of `call.block` columns (`_tile`)."""
    return _tile(slots, name, operand, columns=pl.ds(j * call.block, call.block))


def _block_stored(call: _Call, ref, row, column, smem):
    """An algebra's emit (`emit(kind, index, grad)`) storing each block's gradient as it comes, in the
    operands' dtype (as fla's prepare_wy_repr_bwd writes them), to `ref`'s `index`-th block of `call.block`
    columns from `column` on the chunk's rows from `row`, through `smem` (`_Operand.store`)."""
    held = dataclasses.replace(call.held, rows=call.rows)

    def stored(kind: str, index: int, grad) -> None:
        del kind
        held.store(ref, grad, row, CHUNK, column + index * call.block, call.block, family=call.family,
                   smem=smem)

    return stored


def _prefix_reader(family: mosaic.Family, streamed: bool, refs: dict, head, span):
    """A reader of a chunk's prefix as a row's vectors `(gate, low)` and a column's `(column_gate,
    column_low)` (`refs`' views by name, `_vector_views`'), from their streams' slots where `streamed`, else
    from global memory at `head` and `span`."""
    def read(slots):
        found = {name: slots[f"{name}0"] if streamed else ref.at[head, span] for name, ref in refs.items()}
        return ([family.rows_of(found[name]) for name in ("gate", "low")],
                tuple(family.columns_of(found[name]) for name in ("column_gate", "column_low")))

    return read


def _vector_views(chunk, head, **refs) -> dict:
    """The streams of a chunk's vectors (`refs[name]`, a decay a head's `[HV, R]` prefix halves or write
    gates), each its own stream `{name}0` of the head's row at the chunk (`_vector_blocks`), where the
    family's copies move a one-dimensional block (`Family.copies_vectors`), none where they do not."""
    return {f"{name}0": (ref.at[head], lambda i: (chunk,)) for name, ref in refs.items()}


def _vector_blocks(call: _Call, *names) -> dict:
    """The streamed blocks of `_vector_views`' streams, a chunk's fp32 each, where the family copies them."""
    return {f"{name}0": ((CHUNK,), jnp.float32) for name in names} if call.family.copies_vectors else {}


def _down(step) -> tuple:
    """A view's block indices at `step`: its `step`-th row tile, from its first column."""
    return step, 0


def _staging(operand: _Operand, *shapes) -> tuple:
    """A product site's staging tiles, a part of `operand`'s form each, for each of `shapes`."""
    return tuple(tuple(plgpu.SMEM(shape, operand.stored) for _ in range(operand.parts)) for shape in shapes)


def _merges(call: _Call) -> tuple:
    """The inverse's staging tiles (`algebra.doubled`'s 'merge' products, both operands fp32 registers): three
    sets, the system's kept across the levels and each level's inverse across its two products, each set a
    `[C, C]` tile an operand's part: an fp32 call's three bf16 parts, a 16-bit call's one fp16, 11 significant
    bits as the TF32 fla merges its blocks in, where the inverse is rounded to the call's 8 after. fla inverts
    its 16-row diagonal blocks by fp32 forward substitution and merges in TF32; every doubling level here
    rounds to fp16, about 2^-12 absolute on the inverse under aligned keys, a sixth of the bf16 rounding
    both arms share. fp16's range holds the inverse of keys the layers normalize (`l2norm`: each entry at
    most `beta |k_s| |k_t|`, at most one); unnormalized keys with `beta |k|^2` past two grow it
    geometrically, past fp16's largest in about 16 tokens, where TF32's stays finite (template numerics
    review 2026-10-26, P3-4)."""
    dtype, parts = (jnp.bfloat16, PARTS) if call.held.parts > 1 else (jnp.float16, 1)
    tile = (CHUNK, CHUNK)
    return tuple(tuple(plgpu.SMEM(tile, dtype) for _ in range(parts)) for _ in range(3))


def _summing(call: _Call) -> tuple:
    """`algebra.summed_back`'s staging: the fp32 operand's three bf16 parts and the triangle's one
    (`algebra.Exact`) its 'summed' product takes, and the scratch its columns' sums across warps take
    (`_columns`)."""
    return (*_staging(_Operand(CHUNK, PARTS, jnp.dtype(jnp.float32)), (CHUNK, CHUNK)),
            (plgpu.SMEM((CHUNK, CHUNK), jnp.bfloat16),), _column_scratch(call))


def _column_scratch(call: _Call) -> tuple:
    """The shared memory the family's column sums take (`Family.column_scratch` fp32 entries), none where
    its reductions take Mosaic GPU's implicit scratch."""
    entries = call.family.column_scratch
    return (plgpu.SMEM((entries,), jnp.float32),) if entries else ()


def _columns(call: _Call, reserved: tuple):
    """A `[rows, columns]` value's column sums in the family's reduction (`Family.column_sums`), through the
    shared memory `_column_scratch` reserved for them."""
    return lambda value: call.family.column_sums(value, reserved[0] if reserved else None, call.interpret)


def _summed(staging: tuple, call: _Call) -> tuple:
    """`algebra.summed_back`'s issuer and column sums on `_summing`'s staging."""
    parts, exact, scratch = staging
    return _Issuer(call, {"summed": parts, "summed.exact": exact}), _columns(call, scratch)


def _merging(staging, call: _Call) -> _Issuer:
    """The inverse's products on `_merges`' staging tiles."""
    return _Issuer(call, {"merge": list(staging)})


def _taken(index: int):
    """A chunk's program's liveness (`_kernel`'s `live`): its chunk, the `index`-th input's entry (`taken`),
    holds tokens."""
    return lambda refs: refs[index][_axis("chunks")] > 0


_TILE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("tile", default=None)
"""A persistent launch's current tile (`_kernel`'s `persistent`): its grid's indices by name."""

_WALKED: contextvars.ContextVar[list | None] = contextvars.ContextVar("walked", default=None)
"""Where a persistent launch's program gives its walk (`_walk`) for its tile instead of running it."""


def _axis(name: str):
    """The program's index along its grid's `name` axis: a persistent launch's tile's (`_TILE`), else the
    kernel's own grid's."""
    tile = _TILE.get()
    return lax.axis_index(name) if tile is None else tile[name]


@dataclasses.dataclass(frozen=True)
class Work:
    """A program's work as a launch states it for the cost model: its products' FLOPs, each an operand's
    part's (`_kernel` counts an fp32 call's parts), its exponentials, and the steps its walk takes."""

    flops: float
    steps: float = 1
    exponentials: float = 0


@dataclasses.dataclass(frozen=True)
class _Held:
    """What a launch's consumer thread holds in registers, as its program states it: the tiles it carries
    across its steps (`carried`, each `(rows, columns, dtype)`) and the tiles each of its phases holds at once
    (`phases`, a tuple of such lists): its accumulators, an operand's parts, a product's pairs and their
    masks."""

    carried: Sequence = ()
    phases: Sequence = ()

    def registers(self, family: mosaic.Family) -> int:
        """A consumer thread's registers at the program's peak (`resources.live_registers`)."""
        return resources.live_registers(family, self.carried, *self.phases)


def _f32(rows: int, columns: int) -> tuple:
    """An fp32 tile `[rows, columns]` (`_Held`'s)."""
    return rows, columns, jnp.float32


def _held(operand: _Operand, rows: int, columns: int) -> list:
    """`operand`'s tile `[rows, columns]` as registers hold it: its parts, each in its stored dtype."""
    return [(rows, columns, operand.stored)] * operand.parts


def _masks(rows: int, columns: int) -> list:
    """The int32 tile a tile's masks compare, its rows' lag behind its columns (`algebra._lags`); the
    inverse's doubling compares its rows and columns apart, two (`_masks(...) * 2`)."""
    return [(rows, columns, jnp.int32)]


def _fragments(call: _Call, *tiles) -> list:
    """The operand tiles `(operand, rows, columns)` a product reads from registers where the family's
    products read them there (`Family.operands_in_registers`, mma.sync's), none where they read shared
    memory (wgmma's)."""
    if not call.family.operands_in_registers:
        return []
    return [tile for operand, rows, columns in tiles for tile in _held(operand, rows, columns)]


@dataclasses.dataclass(frozen=True)
class Launch:
    """A launch as planned (`planned`): its name (`recurrence_states`, ...), its programs and each one's
    threads (its bundle's warpgroups, `Family.warpgroups`), the shared memory a program allocates (its
    streams' slots and barriers and its staging tiles, as its bundle declares them), the registers a consumer
    thread holds at its program's peak (`resources.live_registers` of the tiles it states), its steps, the
    chunks in flight, the bytes a step streams in, a program's tensor FLOPs and exponentials, the bytes it
    reads and writes in global memory, and the programs a multiprocessor holds of it at once
    (`resources.plan`)."""

    name: str
    programs: int
    threads: int
    shared: int
    registers: int
    steps: float
    stages: int
    step_bytes: int
    flops: float
    exponentials: float
    device_bytes: int
    resident: int = 1


_PLANNED: contextvars.ContextVar[list | None] = contextvars.ContextVar("planned", default=None)
"""Where `planned` collects its launches; None where they run."""


def _scope(*allocations, reduces: bool = True, after: tuple = ()) -> resources.Scope:
    """A program's allocations by lifetime (`resources.Scope`, whose bytes `resources.shared_bytes` counts as
    Mosaic GPU's resource estimate does): a pipelined family's streams (`mosaic_sm80.Pipeline`, which
    `emit_pipeline` allocates in its walk's own scope, each its stages of each tile) its walks, every other
    tile and barrier the program's, the tiles it scopes `after` its walks (`_final_landing`), and whether it
    `reduces` across warps: Mosaic GPU reserves the reduction scratch for a kernel with any reduction and none
    without (an A100's solve launch asked for its tiles' 90112 bytes alone, SE-family-a100-0c8018352)."""
    walks, program = [], []
    for x in jax.tree.leaves(allocations, is_leaf=lambda x: isinstance(x, mosaic_sm80.Pipeline)):
        if isinstance(x, mosaic_sm80.Pipeline):
            walks.append(tuple(plgpu.SMEM((x.stages, *shape), dtype) for shape, dtype in x.blocks.values()))
        else:
            program.append(x)
    return resources.Scope(program=tuple(program), walks=tuple(walks), after=tuple(after), reduces=reduces)


def _inverse_flops() -> float:
    """The tensor FLOPs of a chunk's inverse (`algebra.doubled`): two whole-tile products a doubling level,
    from two-wide blocks to the chunk's (`log2(CHUNK) - 1` levels), each `2 C^3`, masked after they issue."""
    return 4 * CHUNK ** 3 * (int(math.log2(CHUNK)) - 1)


def _products(parts: int) -> int:
    """The products an fp32 call's issue takes for one of its operands' (`_Issuer`): its parts' pairs whose
    orders sum under `parts`."""
    return sum(1 for i in range(parts) for j in range(parts) if i + j < parts)


def _walk_stores(family: mosaic.Family, stores: int):
    """A walk step's `stores` stores, each through its own landing tile and left in flight (`wait=False`):
    `before()` ahead of each waits until at most `stores - 1` of the thread's stores still read their tiles,
    the oldest of them the same tile's in the step before, so a step's stores overlap the next step's work
    (the states walk waited out each, 9.5% of its samples, SE-sass-h100-be7214123). `_sequence_walk` waits
    them out after its last step."""
    return lambda: family.store_wait(stores - 1)


def _memory_landing(call: _Call, bv: int) -> dict:
    """A walk's memory's landing tiles, one a key slice `[block, bv]` (`_walk_stores`): `{memory0, ...}`."""
    return {f"memory{i}": call.memories.landing(call.block, bv) for i in range(call.width // call.block)}


def _landing_keys(landing: dict, family: mosaic.Family) -> dict:
    """Each store's landing tile by name (`_landed`'s keys): its own, or on a family whose stores finish when
    they return (`Family.walks_scoped`) one a shape and dtype, its stores' in turn: an A100's states launch
    held three 4 KB tiles of one form, which kept its walk one stage deep (two needed 15 KB less)."""
    shared = family.walks_scoped
    return {name: f"{shape}-{jnp.dtype(dtype).name}" if shared else name
            for name, (shape, dtype) in landing.items()}


def _landed(landing: dict, family: mosaic.Family) -> dict:
    """A launch's stores' shared-memory tiles, `landing[name]` a store's `(shape, dtype)`, by key
    (`_landing_keys`): every family's stores go through shared memory (`Family.stored`, `Family.added`; a TMA
    copy, or rows through registers where a GPU has none), since a product layout's register store of a tile
    to global memory leaves partial 32-byte sectors (JAX's `plan_gmem_transfer` refuses it)."""
    keys = _landing_keys(landing, family)
    return {keys[name]: plgpu.SMEM(shape, dtype) for name, (shape, dtype) in landing.items()}


def _planned_launch(call: _Call, scope: Callable[[int], resources.Scope], registers: int, *, floor: int,
                    most: int) -> resources.LaunchPlan:
    """A launch's ring depth and the programs a multiprocessor holds of it (`resources.plan`, at its stated
    `registers` and its streams `scope`d at each depth from `floor` to `most`), or where it holds at none (a
    config past the device, which `template.Mosaic.legal` refuses) one stage and one program."""
    found = resources.plan([resources.Launch(scope, 1, registers, floor=floor, most=most)], call.family,
                           call.target)
    if found:
        return found[0]
    return resources.LaunchPlan(floor, 1, resources.shared_bytes(call.family, scope(floor)))


def _laid(call: _Call, scope: Callable[[int], resources.Scope], registers: int, tiles: int, *,
          stages: int | None) -> tuple[resources.LaunchPlan, int]:
    """A launch's plan and its programs: a persistent launch's (`stages` None) as many as a multiprocessor
    holds at once, at most its `tiles`, its rings two deep where they run on (`mosaic.RingFamily`'s) and two
    deep hold as many programs as one, so a tile's loads go out under the last tile's work: a launch of
    one-step tiles whose program holds a multiprocessor's registers alone (each launch's 255 registers a
    thread an H100 holds one of, long-scoreboard stalls 51 to 73% of its time, SE-family-h100-80a38c02d; an
    A100's launches held one program each, a quarter of the warps fla's held, SE-family-a100-8c5b5b16a);
    any other a program a tile at its `stages`."""
    if stages is None:
        most = 2 if isinstance(call.family, mosaic.RingFamily) else 1
        found = _planned_launch(call, scope, registers, floor=1, most=most)
        return found, min(tiles, (call.target.multiprocessors or 1) * found.programs)
    return _planned_launch(call, scope, registers, floor=stages, most=stages), tiles


def _stated(work: Work, tiles: int, arrays, out_type) -> pl.CostEstimate:
    """A launch's work over its `tiles` as it states it to XLA and to Dew's FLOP count
    (`templates.gemm.cost`): the model's products (an fp32 call's parts aside), its exponentials, and its
    operands' and outputs' bytes once (row 1895's kernel report: unstated, the recurrence made MFU a lower
    bound)."""
    stated = cost(int(work.flops * tiles), *arrays, *jax.tree.leaves(out_type))
    return dataclasses.replace(stated, transcendentals=int(work.exponentials * tiles))


def _kernel(program, call: _Call, *, out_type, grid, grid_names, blocks: dict, stages: int, live,
            work: Work, held: _Held, staging: tuple = (), once: dict | None = None,
            landing: dict | None = None, persistent: bool = False, reduces: bool = True,
            after: tuple = ()):
    """`program(*refs, streams, staging, landing)` as a kernel over `grid`, run where `live(refs)` holds (a
    laid call's chunks and sequences are as many as its worst case, most of them empty): its streams'
    allocations (of `blocks`, `stages` deep, and of `once`'s blocks, loaded once, one deep; `Family.opened`
    takes them with their views), its staging tiles and its stores' (`landing`, `_landed`) scoped, named for
    its launch (`recurrence_solved`, ...). `held` is what a consumer thread holds in registers (`_Held`).
    Where `planned` collects launches, the launch's record instead (`Launch`, its `work`), and zeros.

    A `persistent` launch runs `grid`'s tiles on as many programs as the device's multiprocessors hold at
    once, its streams' rings running on across them (`_planned_launch`); its program gives its tile's walk
    (`_walk`) for the grid's indices `_axis` reads. A program that `reduces` none takes no reduction scratch
    (`_scope`; the compile gate holds its plan's shared memory to at least the compiled kernel's).
    `after` are the `(shape, dtype)` tiles its program scopes after its walks (`_final_landing`), which its
    plan counts beside them as its family scopes them."""
    family = call.family
    landed = _landed(landing or {}, family)
    staging = (*staging, landed)
    tiles = math.prod(grid)
    threads = 128 * family.warpgroups(1)

    def allocations(depth: int) -> dict:
        # The walked streams and those loaded once, each its own allocation (a pipelined family's stages are
        # its allocation's), which a program opens with their views by name ('walked', 'once').
        opened = {"walked": family.streams(blocks, depth, 1)}
        if once:
            opened["once"] = family.streams(once, 1, 1)
        return opened

    def scope(depth: int) -> resources.Scope:
        return _scope(allocations(depth), staging, reduces=reduces, after=after)

    found, programs = _laid(call, scope, held.registers(family), tiles, stages=None if persistent else stages)
    stages = found.stages
    streams = allocations(stages)
    name = f"recurrence{program.__qualname__.split('.')[0]}"
    collected = _PLANNED.get()
    if collected is not None:
        def recorded(*arrays):
            outs = jax.tree.leaves(out_type)
            moved = sum(math.prod(x.shape) * jnp.dtype(x.dtype).itemsize for x in (*arrays, *outs))
            stepped = sum(math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in blocks.values())
            # A persistent program's work is its tiles' (`work` is a tile's), the most any program takes.
            each = -(-tiles // programs)
            flops = work.flops * _products(call.held.parts) * each
            collected.append(Launch(name, programs, threads, resources.shared_bytes(family, scope(stages)),
                                    held.registers(family), work.steps * each, stages, stepped, flops,
                                    work.exponentials * each, moved, found.programs))
            return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), out_type)

        return recorded

    def entry(*arrays):
        def scoped(staged, *, streams):
            *staged, lands = staged
            keys = _landing_keys(landing or {}, family)
            lands = {name: lands[keys[name]] for name in landing or {}}
            if not persistent:
                program(*arrays, streams=streams, staging=tuple(staged), landing=lands)
                return

            def tile(index) -> dict:
                """The grid's indices of the persistent loop's tile `index` (its first axis the slowest)."""
                flat, indices = index[0], {}
                for axis, extent in reversed(tuple(zip(grid_names, grid, strict=True))):
                    flat, indices[axis] = flat // extent, flat % extent
                return indices

            def given(at: dict) -> tuple:
                """The tile's walk as its program gives it (`_walk`'s), and whether the tile holds tokens."""
                walked: list = []
                tile_token, walk_token = _TILE.set(at), _WALKED.set(walked)
                try:
                    program(*arrays, streams=streams, staging=tuple(staged), landing=lands)
                    taken = live(arrays)
                finally:
                    _WALKED.reset(walk_token)
                    _TILE.reset(tile_token)
                (found,) = walked
                return found, taken

            def schedule(index):
                at = tile(index)
                (views, _, _, steps), taken = given(at)
                return taken, steps, views, at

            def consumer(_, at, walk, store, *, active):
                del store
                if not active:
                    return
                (_, step, prologue, _), _ = given(at)
                token = _TILE.set(at)
                try:
                    walk(step if prologue is None else functools.partial(step, loaded=prologue()), None, ())
                finally:
                    _TILE.reset(token)

            family.persist(streams["walked"], (tiles,), schedule, consumer, consumers=1, axis="programs",
                           outputs=(), interpret=call.interpret)

        if persistent:
            family.scoped(scoped, staging, streams=streams)
        else:
            # An empty program scopes nothing (Mosaic GPU's interpreter takes a scope's allocations in turn).
            pl.when(live(arrays))(lambda: family.scoped(scoped, staging, streams=streams))

    entry.__name__ = name

    def launched(*arrays):
        laid = ({"grid": (programs,), "grid_names": ("programs",)} if persistent
                else {"grid": grid, "grid_names": grid_names})
        return family.kernel(entry, out_type=out_type, **laid, consumers=1, interpret=call.interpret,
                             name=name, cost=_stated(work, tiles, arrays, out_type))(*arrays)

    return launched


def _walk(streams, views, step, call: _Call, *, steps: int = 1, prologue: Callable | None = None):
    """The program's roles over its `steps` steps (one by default): the family's loads of `views` (`_views`)
    into `streams` (its allocations) and its walk of `step`, which stores what the program computes. Where a
    `prologue` is given, the consumer runs it before its walk waits on the streams' first tiles and its
    step takes what it returns (`step(index, slots, carry, accumulators, loaded)`): its own loads from
    global memory (a chunk's prefix and write gates) in flight beside the streams', not after them."""
    family = call.family
    given = _WALKED.get()
    if given is not None:
        # A persistent launch's tile: its kernel walks it (`_kernel`'s `persistent`).
        given.append((views, step, prologue, steps))
        return
    streams = family.opened(streams["walked"], views)

    def consumer(_):
        walked = step if prologue is None else functools.partial(step, loaded=prologue())
        family.walk(streams, ((walked, 0, steps),), (), (), call.interpret)

    family.roles(1, consumer, lambda: family.stream(streams, views, 0, steps))


def _solved(k, v, hi, lo, beta, taken, call: _Call, *, keeps: bool = True):
    """`(a [R, HV * C], w [R, HV * Dk], u [R, HV * Dv])` as `call.held` of each chunk and value head under a
    decay a head, fla's kkt_solve and recompute_w_u in one program (FlashQLA's first kernel): the keys'
    products (`algebra.keyed`) inverted by doubling on the chunk's tile (`algebra.doubled`, its merges in
    fp16 in a 16-bit call, `_merges`), stored less its unit diagonal (`a`, the backward's), and applied in
    registers to the decayed write-gated keys and the write-gated values (`algebra.weighted`), the inverse
    staged once for every block's product: one launch's tiles and one load of the keys, where two launches
    stored and loaded the inverse between them. Where it `keeps` no inverse (a forward no backward reads,
    `a` None), its store's landing tile goes: an A100's launch then holds two programs a multiprocessor
    (an inverse's 8 KB past half its shared memory kept it at one, SE-family-a100-0c8018352)."""
    width, value_width, family = call.width, call.value_width, call.family
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(k_ref, v_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref, taken_ref, *outputs,
                streams, staging, landing):
        a_ref, w_ref, u_ref = outputs if keeps else (None, *outputs)
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        views = _views(keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width),
                       values=call.given.views(v_ref, row, CHUNK, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "write": beta_ref, "column_gate": column_hi_ref,
                   "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            # The chunk's vectors streamed with its tiles, loaded a tile ahead (`_vector_views`).
            views |= _vector_views(n, h, **vectors)
        merges, (lower_tile, scaled_tile) = staging[:3], (staging[:2] if len(staging) == 3 else staging[3:])

        def read(refs):
            """The vectors as a row's (gates, lows, writes) and a column's (gates, lows), from their streams'
            slots or from global memory."""
            refs = {name: refs[f"{name}0"] if streamed else ref.at[h, span] for name, ref in vectors.items()}
            return ([family.rows_of(refs[name]) for name in ("gate", "low", "write")],
                    [family.columns_of(refs[name]) for name in ("column_gate", "column_low")])

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low, betas), columns = found

            def block(name, c):
                return _tile(slots, name, call.given, columns=pl.ds(c * call.block, call.block))

            product = functools.reduce(jnp.add, (_Issuer(call)("intra", key, key.T)
                                                 for key in map(functools.partial(block, "keys"),
                                                                range(width // call.block))))
            system = algebra.keyed(product, gate, low, betas, *columns)
            # The inverse less its unit diagonal, which `algebra.weighted` adds outside its product.
            lower = (algebra.doubled(-system, _merging(merges, call))
                     - jnp.where(_diagonal(CHUNK, family), 1.0, 0.0))
            if keeps:
                held.store(a_ref, lower, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            # The inverse staged once (the first set) for every block's product, a block staged in the
            # second: `_Issuer` keeps a staged value for the products after it that take it.
            sets = [lower_tile, scaled_tile]
            issue = _Issuer(call, {"apply_u": sets, "apply_w": sets})
            for c in range(value_width // call.block):
                u = algebra.weighted("apply_u", lower, block("values", c).value().astype(held.dtype), betas,
                                     issue)
                held.store(u_ref, u, row, CHUNK, h * value_width + c * call.block, call.block, family=family,
                           smem=landing["block"])
            decay = lax.broadcast_in_dim(algebra.raised(gate, low), (CHUNK, call.block), (0,))
            for c in range(width // call.block):
                w = algebra.weighted("apply_w", lower, block("keys", c).value().astype(held.dtype), betas,
                                     issue, decay)
                held.store(w_ref, w, row, CHUNK, h * width + c * call.block, call.block, family=family,
                           smem=landing["block"])
            return carry, accumulators

        _walk(streams, views, step, call, prologue=None if streamed else lambda: read(None))

    # The keys' products and the inverse's doubling, then the inverse against each key and value block.
    work = Work(2 * CHUNK * CHUNK * width + _inverse_flops() + 2 * CHUNK * CHUNK * (width + value_width))
    # The keys' products' sum and the system's pairs; then the doubling's inverse, power and product, beside
    # the masks (an fp32 call's merges split the inverse and the power into their parts); then the inverse
    # held beside a block's product, its decay and its keys or values.
    held_tiles = _Held(phases=(
        [_f32(CHUNK, CHUNK)] * 2 + _fragments(call, (call.given, CHUNK, call.block), (call.given, CHUNK,
                                                                                         call.block)),
        [_f32(CHUNK, CHUNK)] * 3 + _masks(CHUNK, CHUNK) * 2
        + (_held(held, CHUNK, CHUNK) * 2 if held.parts > 1 else []),
        [_f32(CHUNK, CHUNK)] + [_f32(CHUNK, call.block)] * 2 + _held(call.given, CHUNK, call.block)
        + _fragments(call, (held, CHUNK, CHUNK), (call.given, CHUNK, call.block))))
    out_type = (held.shape(call.heads * width), held.shape(call.heads * value_width))
    if keeps:
        out_type = (held.shape(call.heads * CHUNK), *out_type)
    # The inverse's application stages in the doubling's merge tiles where they are its form (an fp32 call's
    # bf16 parts, a block a chunk wide), done by then: apart, an fp32 call's launch took 242864 bytes of
    # shared memory, past an H100's block (SE-family-h100-e0de17766).
    merging = _merges(call)
    reused = (held.parts > 1 and merging[0][0].dtype == held.stored and call.block == CHUNK)
    staging = merging if reused else (*merging, _staging(held, (CHUNK, CHUNK))[0],
                                      _staging(held, (CHUNK, call.block))[0])
    landing = {"square": held.landing(CHUNK, CHUNK)} if keeps else {}
    landing["block"] = held.landing(CHUNK, call.block)
    solved = _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                     grid_names=("chunks", "heads"),
                     blocks=_streamed(keys=((CHUNK, width), call.given),
                                      values=((CHUNK, value_width), call.given))
                     | _vector_blocks(call, "gate", "low", "write", "column_gate", "column_low"),
                     stages=1, live=_taken(7), work=work, held=held_tiles, staging=staging, landing=landing,
                     persistent=True, reduces=False)(k, v, hi, lo, beta, hi, lo, taken)
    return solved if keeps else (None, *solved)


def _intra(q, k, hi, lo, beta, taken, call: _Call, *, scale: float):
    """`((I + A)^-1, Aqk)` of each chunk and value head under a decay a key dimension, `[R, HV * C]` each as
    `call.held`: the chunk's scores and its keys' system, each pair's decay factored over the chunk's
    sub-chunks (`algebra.key_decayed_pairs`, which the call's declared floor bounds), their products on the
    tensor cores, and the system inverted (`algebra.doubled`): fla_kda's intra rows and solve, fused.
    Without the delta rule, `(Aqk,)` alone."""
    width, held, gates = call.width, dataclasses.replace(call.held, rows=call.rows), call.gates
    family = call.family

    def program(q_ref, k_ref, hi_ref, lo_ref, beta_ref, taken_ref, *outs, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = _views(queries=call.given.views(q_ref, row, CHUNK, keyed, width),
                       keys=call.given.views(k_ref, row, CHUNK, keyed, width),
                       gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                       lows=gates.views(lo_ref, row, CHUNK, h * width, width))
        pairing, *merges = staging

        def step(index, slots, carry, accumulators):
            del index

            def block(name, operand, c):
                return _tile(slots, name, operand, columns=pl.ds(c * call.block, call.block)).value()

            def operand(name):
                return lambda c: block(name, call.given, c).astype(jnp.float32)

            def reference(sub, c):
                """The sub-chunk's first token's prefix halves on key block `c`, a column's vectors, from
                global memory (a step's tiles carry their bundle's transforms, which a traced row does not
                commute through)."""
                columns = pl.ds(h * width + c * call.block, call.block)
                return tuple(family.columns_of(ref.at[row + sub * algebra.SUB, columns])
                             for ref in (hi_ref, lo_ref))

            def gated(c):
                return block("gates", gates, c), block("lows", gates, c)

            pairs = functools.partial(algebra.key_decayed_pairs, columns=width // call.block, gates=gated,
                                      reference=reference, issue=_Issuer(call, {"pairs": pairing}),
                                      chunk=CHUNK)
            scores = pairs(operand("queries"), operand("keys"), strict=False) * scale
            held.store(outs[-1], scores, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            if call.delta:
                betas = family.rows_of(beta_ref.at[h, span])
                system = pairs(lambda c: operand("keys")(c) * betas[:, None], operand("keys"), strict=True)
                rows, cols = (family.iota((CHUNK, CHUNK), dimension) for dimension in (0, 1))
                inverse = algebra.doubled(-system, _merging(tuple(merges), call))
                held.store(outs[0], inverse - jnp.where(rows == cols, 1.0, 0.0), row, CHUNK, h * CHUNK, CHUNK,
                           family=family, smem=landing["square"])
            return carry, accumulators

        _walk(streams, views, step, call)

    blocks = _streamed(queries=((CHUNK, width), call.given), keys=((CHUNK, width), call.given),
                       gates=((CHUNK, width), gates), lows=((CHUNK, width), gates))
    out_type = (held.shape(call.heads * CHUNK),) * (1 + call.delta)
    # Each sub-chunk's and key block's gated rows and keys and their product, for the scores and the system,
    # then the inverse's doubling.
    pairs = (CHUNK // algebra.SUB) * (1 + call.delta)
    work = Work(pairs * 2 * CHUNK * CHUNK * width + (_inverse_flops() if call.delta else 0),
                exponentials=pairs * 2 * CHUNK * width)
    # The sum of the pairs, beside a sub-chunk's and key block's prefix halves, exponent, gated rows and
    # keys; then the doubling's, as the solve's.
    held_tiles = _Held(phases=(
        [_f32(CHUNK, CHUNK)] + [_f32(CHUNK, call.block)] * 4 + _masks(CHUNK, call.block) * 2
        + _fragments(call, (_pairing_operand(call), CHUNK, call.block), (_pairing_operand(call), CHUNK,
                                                                              call.block)),
        [_f32(CHUNK, CHUNK)] * 4 + _masks(CHUNK, CHUNK) * 2))
    staging = (_pairing(call), *(_merges(call) if call.delta else ()))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=blocks, stages=1, live=_taken(5), work=work,
                   held=held_tiles, staging=staging, landing={"square": held.landing(CHUNK, CHUNK)},
                   persistent=True)(q, k, hi, lo, beta, taken)


def _pairing_operand(call: _Call) -> _Operand:
    """The form the factored pairs' fp32 operands take to their products (`algebra.key_decayed_pairs`): in a
    16-bit call, fp32 itself where the family's products take it as TF32 (`Family.takes_tf32`, warpgroup
    MMA's), one product a pair as fla's sub-chunk products take, else two bf16 parts (16 of their bits,
    past TF32's); an fp32 call's three bf16 parts."""
    if call.held.parts > 1:
        return _Operand(call.rows, PARTS, jnp.dtype(jnp.float32))
    return _Operand(call.rows, 1 if call.family.takes_tf32 else 2, jnp.dtype(jnp.float32))


def _pairing(call: _Call) -> list:
    """The factored pairs' staging (`_Issuer`'s 'pairs'): two sets, the gated rows' and the gated keys',
    each a `[C, block]` tile a part (`_pairing_operand`)."""
    return [_staging(_pairing_operand(call), (CHUNK, call.block))[0] for _ in range(2)]


def _weighted(k, v, a, hi, lo, beta, taken, call: _Call):
    """`(w [R, HV * Dk], u [R, HV * Dv])` as `call.held` (`algebra.weighted`): the inverse against the decayed
    write-gated keys and against the write-gated values; under a decay a key dimension, also the keys decayed
    to their chunk's end, `[R, HV * Dk]` as `call.held` (`algebra.key_decayed_write`), which the memory takes,
    and those alone without the delta rule (`a` None)."""
    width, value_width, gates, family = call.width, call.value_width, call.gates, call.family
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(k_ref, v_ref, a_ref, hi_ref, lo_ref, beta_ref, taken_ref, *outs, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        given = {"keys": call.given.views(k_ref, row, CHUNK, h // call.group * width, width)}
        if call.per_key:
            given.update(gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                         lows=gates.views(lo_ref, row, CHUNK, h * width, width))
        if call.delta:
            given.update(inverse=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                         values=call.given.views(v_ref, row, CHUNK, h * value_width, value_width))
        views = _views(**given)
        vectors = {**({"write": beta_ref} if call.delta else {}),
                   **({} if call.per_key else {"gate": hi_ref, "low": lo_ref})}
        streamed = family.copies_vectors
        if streamed:
            # The chunk's vectors streamed with its tiles, loaded a tile ahead (`_vector_views`).
            views |= _vector_views(n, h, **vectors)

        def read(refs):
            """The write gates and a decay a head's prefix, a row's vectors, from their streams' slots or from
            global memory."""
            rows = {name: family.rows_of(refs[f"{name}0"] if streamed else ref.at[h, span])
                    for name, ref in vectors.items()}
            return rows.get("write"), None if call.per_key else (rows["gate"], rows["low"])

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            betas, prefix = found
            # The inverse, the write gates and the issuer under the delta rule, which a decay a head's call
            # of this launch always takes (`_prepared`).
            applied, decay = None, None
            if call.delta:
                (scaled,) = staging
                issue = _Issuer(call, {"apply_u": scaled, "apply_w": scaled})
                inverse = _tile(slots, "inverse", held)
                applied = inverse, betas, issue
                for c in range(value_width // call.block):
                    tile = _tile(slots, "values", call.given,
                                 columns=pl.ds(c * call.block, call.block)).value()
                    u = algebra.weighted("apply_u", inverse, tile.astype(held.dtype), betas, issue)
                    held.store(outs[1], u, row, CHUNK, h * value_width + c * call.block, call.block,
                               family=family, smem=landing["block"])
            if prefix is not None:
                decay = lax.broadcast_in_dim(algebra.raised(*prefix), (CHUNK, call.block), (0,))
            for c in range(width // call.block):
                block = pl.ds(c * call.block, call.block)
                tile = _tile(slots, "keys", call.given, columns=block).value()
                if call.per_key:
                    gate, low = (_tile(slots, name, gates, columns=block).value()
                                 for name in ("gates", "lows"))
                    decay = algebra.raised(gate, low)
                    ends = pl.ds(h * width + c * call.block, call.block)
                    last, last_low = (family.columns_of(ref.at[row + CHUNK - 1, ends])
                                      for ref in (hi_ref, lo_ref))
                    written = algebra.key_decayed_write(tile, gate, low, last, last_low)
                    held.store(outs[-1], written, row, CHUNK, h * width + c * call.block, call.block,
                               family=family, smem=landing["block"])
                if applied is not None:
                    inverse, betas, issue = applied
                    w = algebra.weighted("apply_w", inverse, tile.astype(held.dtype), betas, issue, decay)
                    held.store(outs[0], w, row, CHUNK, h * width + c * call.block, call.block, family=family,
                               smem=landing["block"])
            return carry, accumulators

        _walk(streams, views, step, call, prologue=None if streamed else lambda: read(None))

    given = {"keys": ((CHUNK, width), call.given)}
    if call.per_key:
        given.update(gates=((CHUNK, width), gates), lows=((CHUNK, width), gates))
    if call.delta:
        given.update(inverse=((CHUNK, CHUNK), held), values=((CHUNK, value_width), call.given))
    out_type = ((held.shape(call.heads * width), held.shape(call.heads * value_width)) * call.delta
                + (held.shape(call.heads * width),) * call.per_key)
    # Without the delta rule the inverse's place takes the keys, an operand no step reads.
    work = Work(2 * CHUNK * CHUNK * (width + value_width) if call.delta else 0,
                exponentials=CHUNK * width if call.per_key else 0)
    # The inverse beside a block's product, its decay and its keys or values; under a decay a key dimension
    # the block's prefix halves and its decayed write too.
    held_tiles = _Held(phases=(
        _held(held, CHUNK, CHUNK) + [_f32(CHUNK, call.block)] * 2 + _held(call.given, CHUNK, call.block)
        + [_f32(CHUNK, call.block)] * 3 * call.per_key
        + _fragments(call, (held, CHUNK, CHUNK), (call.given, CHUNK, call.block)),))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"),
                   blocks=_streamed(**given) | _vector_blocks(call, *(("write",) if call.delta else ()),
                                                              *(() if call.per_key else ("gate", "low"))),
                   stages=1, live=_taken(6),
                   work=work, held=held_tiles,
                   staging=_staging(held, (CHUNK, call.block)) if call.delta else (),
                   landing={"block": held.landing(CHUNK, call.block)}, persistent=True)(
        k, v, k if a is None else a, hi, lo, beta, taken)


def _sequence_walk(call: _Call, memory: _Operand, name: str, entering: tuple, walked: tuple, step, start,
                   count, kept, written, out, bv: int, *, landing, prefetch: Callable | None = None) -> None:
    """A sequence's memory over a value block `bv` wide, carried in one role pair: the memory it takes, each
    key slice's tile `{name}{i}` of `memory` (`entering`, the opened streams and their views), zero where
    `kept` is 0; `step` over its chunks `[start, start + count)` of `walked` (the same); and the memory it
    gives written to each slice's view `out(i)` where `written` is nonzero, through `landing` (a slice's
    shared tile, `_landed`, or where None one scoped after the walk, `_final_landing`) (`_states`', and
    `_reverse`'s from its last chunk). Where
    `prefetch(n)` is given, it loads chunk `n`'s own values from global memory (its prefix), a step ahead:
    each step issues the next chunk's loads before its own work and takes the ones the step before issued
    (`step(n, slots, state, accumulators, loaded)`), so their latency is the step's, not on its path."""
    family, slices = call.family, call.width // call.block
    walked_step = step
    if prefetch is not None:
        last = start + count - 1

        def walked_step(n, slots, carry, accumulators):
            state, loaded = carry
            ahead = prefetch(jnp.minimum(n + 1, last))
            state, accumulators = step(n, slots, state, accumulators, loaded)
            return (state, ahead), accumulators
    (entered, entered_views), (chunks, views) = entering, walked

    def taken(index, slots, carry, accumulators):
        del index, carry
        # A finite memory times zero where the sequence takes none (a select takes no scalar condition over a
        # tile in Mosaic GPU's warpgroup lowering).
        scale = jnp.where(kept > 0, 1.0, 0.0)
        return [_tile(slots, f"{name}{i}", memory).value() * scale for i in range(slices)], accumulators

    def consumer(_):
        state = family.walk(entered, ((taken, 0, 1),), [family.zeros((call.block, bv))] * slices, (),
                            call.interpret)[0]
        carry = state if prefetch is None else (state, prefetch(start))
        carry = family.walk(chunks, ((walked_step, start, start + count),), carry, (), call.interpret,
                            first=start)[0]
        state = carry if prefetch is None else carry[0]
        # The steps' stores left in flight (`_walk_stores`) done reading their tiles before the program ends.
        family.store_wait(0)

        @pl.when(written > 0)
        def _():
            def store(smem):
                family.stored(list(state), [out(i) for i in range(slices)], smem=smem)

            if landing is None:
                pl.run_scoped(store, plgpu.SMEM((call.block, bv), jnp.float32))
            else:
                store(landing)

    def producer():
        family.stream(entered, entered_views, 0, 1)
        family.stream(chunks, views, start, start + count)

    family.roles(1, consumer, producer)


def _final_landing(call: _Call, bv: int) -> tuple[dict, tuple]:
    """`_sequence_walk`'s final memory's landing, a key slice `[block, bv]` fp32: `(landing, after)`, the
    launch's own tile (`landing['final']`) where its family's walks share its scope, else one the walk
    scopes after it (`after`, `_kernel`'s; `Family.walks_scoped`), which takes no more than the walk's
    tiles: an A100's states launch then holds two stages at two programs a multiprocessor."""
    final = ((call.block, bv), jnp.float32)
    return ({}, (final,)) if call.family.walks_scoped else ({"final": final}, ())


def _gated_write(call: _Call, corrected, loaded, prefix: tuple, last, head) -> tuple:
    """`_states`' step's write gated to its chunk's end and each key slice's decay `[block, 1]` across the
    chunk: under a decay a key dimension the corrected values as they are and each slice's rows' decays at
    the chunk's last token (`prefix`, its halves' refs `[R, HV * Dk]`); under one a head the chunk's write
    gated by `algebra.scalar_write` and its last prefix, a row's vector each slice's rows take, from
    `loaded` (`_states`' prefetch, a step ahead)."""
    family, slices = call.family, call.width // call.block
    if call.per_key:
        columns = [pl.ds(head * call.width + i * call.block, call.block) for i in range(slices)]
        return corrected, [algebra.raised(*(family.rows_of(ref.at[last, span]) for ref in prefix))[:, None]
                           for span in columns]
    assert loaded is not None, "a decay a head's walk prefetches its prefix"
    (ending, ending_low), (gate, low), _ = loaded
    gated, decayed = algebra.scalar_write(corrected, ending, gate, ending_low, low)
    return gated, [decayed[:, None]] * slices


def _states_held(call: _Call, values: _Operand, keys: _Operand, bv: int) -> _Held:
    """What `_states`' consumer thread holds: the memory carried, its key slices in fp32, and the next
    chunk's prefix vectors (`prefetch`); a step's corrected and written values beside the values' block and
    a memory slice as a product's operand (and, where products read registers, a block of the weighted keys
    and of the keys)."""
    held = call.held
    return _Held(carried=[_f32(call.width, bv)] + [_f32(CHUNK, 1)] * 5, phases=(
        [_f32(CHUNK, bv)] * 2 + _held(values, CHUNK, bv) + _held(held, call.block, bv)
        + _fragments(call, (held, CHUNK, call.block), (keys, CHUNK, call.block)),))


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
    slices = width // call.block
    held, memories = dataclasses.replace(call.held, rows=call.rows), call.memories
    batch = h0.shape[0] // (call.heads * width)
    initial = _Operand(h0.shape[0], 1, jnp.dtype(jnp.float32))
    delta = w is not None
    values = held if delta else call.given
    # The keys the memory takes: a key head's, or a value head's decayed to their chunk's end.
    keys, keyed = (held, 1) if call.per_key else (call.given, call.group)

    def program(k_ref, w_ref, u_ref, hi_ref, lo_ref, beta_ref, h0_ref, seq_ref, h_ref, corrected_ref,
                final_ref, *, streams, staging, landing):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        staged, scratch = staging
        issue = _Issuer(call, {"corrected": staged, "write": staged})

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        # Each step a chunk of every row's operands, by its index among the call's chunks.
        views = _views(**({"w": held.views(w_ref, 0, call.rows, h * width, width)} if delta else {}),
                       keys=keys.views(k_ref, 0, call.rows, h // keyed * width, width),
                       values=values.views(u_ref, 0, call.rows, h * value_width + j * bv, bv))
        initial_views = _views(**{f"initial{i}": initial.views(h0_ref, memory(row, i), call.block, j * bv, bv)
                                  for i in range(slices)})
        chunks = family.opened(streams["walked"], views)
        entering = family.opened(streams["once"], initial_views)

        def prefetch(n):
            """Chunk `n`'s prefix under a decay a head: its rows' halves and its last's, a row's vectors
            (Mosaic GPU's warpgroup lowering bitcasts no scalar, which `algebra.raised` does), and its write
            gates without the delta rule."""
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            ends = tuple(jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            rows = tuple(family.rows_of(ref.at[h, span]) for ref in (hi_ref, lo_ref))
            return ends, rows, None if delta else family.rows_of(beta_ref.at[h, span])

        # Each key slice of the memory and the corrected values, a store each in flight.
        before_store = _walk_stores(family, width // call.block + 1)

        def step(n, slots, state, accumulators, loaded=None):
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(state):
                before_store()
                memories.store(h_ref, part, memory(n, i), call.block, j * bv, bv, family=family,
                               smem=landing[f"memory{i}"], wait=False)
            tile = _tile(slots, "values", values).value()
            if delta:
                def weights(i):
                    return _tile(slots, "w", held, columns=pl.ds(i * call.block, call.block))

                corrected = algebra.corrected(tile, weights, state, held.dtype, issue)
            else:
                betas = family.rows_of(beta_ref.at[h, span]) if loaded is None else loaded[2]
                written = algebra.weighted("apply_u", None, tile, betas, issue)
                corrected = algebra.corrected(written, None, state, held.dtype, issue)
            before_store()
            held.store(corrected_ref, corrected, n * CHUNK, CHUNK, h * value_width + j * bv, bv,
                       family=family, smem=landing["corrected"], wait=False)
            gated, decay = _gated_write(call, corrected, loaded, (hi_ref, lo_ref), last, h)
            separate = None
            if held.parts > 1:
                # An fp32 call's last write outside the product (`algebra.carried`), held to the fp32
                # recurrence; a 16-bit call's is fla's product.
                def key_rows(i):
                    """The chunk's last key's slice `i`, a row's vector: its parts' rows summed, exactly."""
                    views = keys.vectors(k_ref, last, h // keyed * width + i * call.block, call.block)
                    return functools.reduce(jnp.add, (family.rows_of(view).astype(jnp.float32)
                                                      for view in views))

                ends = family.iota((CHUNK, bv), 0) == CHUNK - 1
                separate = key_rows, _columns(call, scratch)(jnp.where(ends, gated, 0.0))
            state = algebra.carried(
                state, lambda i: _tile(slots, "keys", keys, columns=pl.ds(i * call.block, call.block)), gated,
                decay, held.dtype, issue, last=separate)
            return state, accumulators

        _sequence_walk(call, initial, "initial", (entering, initial_views), (chunks, views), step, first,
                       count, begins, ends,
                       lambda i: final_ref.at[pl.ds(memory(row, i), call.block), pl.ds(j * bv, bv)], bv,
                       prefetch=None if call.per_key else prefetch, landing=landing.get("final"))

    out_type = (memories.shape(value_width), held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(**({"w": ((CHUNK, width), held)} if delta else {}), keys=((CHUNK, width), keys),
                       values=((CHUNK, bv), values))
    initials = _streamed(**{f"initial{i}": ((call.block, bv), initial) for i in range(slices)})
    # A sequence's chunks, the mean over the laid sequences; two products a chunk.
    steps = call.chunks / sequences.shape[0]
    work = Work(4 * CHUNK * width * bv * steps, steps)
    final_landing, after = _final_landing(call, bv)
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=tiles.stages,
                   live=lambda refs: refs[7][lax.axis_index("sequences"), 1] > 0, work=work,
                   held=_states_held(call, values, keys, bv),
                   staging=(*_staging(held, (call.block, bv)), _column_scratch(call)), once=initials,
                   landing={**_memory_landing(call, bv), "corrected": held.landing(CHUNK, bv),
                            **final_landing},
                   after=after,
                   # Its last write's column sums an fp32 call's alone (`_columns`).
                   reduces=held.parts > 1)(
        k, k if w is None else w, u, hi, lo, beta, h0, sequences)


def _output_streams(call: _Call, bv: int) -> tuple[dict, tuple]:
    """`_output`'s streamed blocks and its staging tiles at value blocks `bv` wide: the queries, a value
    block's memory entered and values; under a decay a head the keys, the scores staged once for every
    value block's product; under one a key dimension the prefix tiles and the intra launch's scores, a key
    block's product staged."""
    width, held, memories, gates = call.width, call.held, call.memories, call.gates
    if call.per_key:
        given = {"gates": ((CHUNK, width), gates), "lows": ((CHUNK, width), gates),
                 "scores": ((CHUNK, CHUNK), held)}
        staging = _staging(held, (CHUNK, call.block))
    else:
        given = {"keys": ((CHUNK, width), call.given)}
        staging = _staging(held, (CHUNK, CHUNK))
    blocks = _streamed(queries=((CHUNK, width), call.given), entered=((width, bv), memories),
                       values=((CHUNK, bv), held), **given)
    return blocks, staging


def _output_held(call: _Call, bv: int) -> _Held:
    """What `_output`'s consumer thread holds: a value block's memory and local products' fp32 sums beside
    its values (and, where products read registers, a key block's queries and memory slice and the scores);
    under a decay a head the first step's scores too, their sum, their pairs' decays and their masks (an
    fp32 call's split into parts), held with the value block's tiles, the step's whole set (as its phases
    alone, a 64-wide value block's output spilled 16 bytes a thread on an A100's at 200 stated registers);
    under one a key dimension a key block's prefix halves and the scores in fp32."""
    held, memories = call.held, call.memories
    values = [_f32(CHUNK, bv)] * 2 + _held(held, CHUNK, bv) + _fragments(
        call, (call.given, CHUNK, call.block), (memories, call.block, bv), (held, CHUNK, CHUNK))
    if call.per_key:
        return _Held(phases=(values + [_f32(CHUNK, call.block)] * 2 + [_f32(CHUNK, CHUNK)],))
    return _Held(phases=(values + [_f32(CHUNK, bv)] + [_f32(CHUNK, CHUNK)] * 3 + _masks(CHUNK, CHUNK)
                         + (_held(held, CHUNK, CHUNK) if held.parts > 1 else []),))


def _output(q, k, h, corrected, hi, lo, taken, call: _Call, tiles: Tiles, *, scale: float, out_dtype,
            scores=None):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype`, a program a chunk and value head and a step each
    value block: under a decay a head its own pairs' scores taken once, at the first step, and staged for
    every block's (`algebra.chunk_scores`, `chunk_output`), under one a key dimension the intra launch's
    (`scores`, `_intra`; `algebra.key_decayed_output`)."""
    width, value_width, bv, gates = call.width, call.value_width, tiles.output_block, call.gates
    held, memories, family = dataclasses.replace(call.held, rows=call.rows), call.memories, call.family
    steps = value_width // bv

    def program(q_ref, k_ref, h_ref, v_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, o_ref, *,
                streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        if call.per_key:
            given = {"gates": gates.views(hi_ref, row, CHUNK, h * width, width),
                     "lows": gates.views(lo_ref, row, CHUNK, h * width, width),
                     "scores": held.views(k_ref, row, CHUNK, h * CHUNK, CHUNK)}
        else:
            given = {"keys": call.given.views(k_ref, row, CHUNK, h // call.group * width, width)}
        # Step `i` the `i`-th value block: its columns of the memory entered and of the values.
        queried = call.given.views(q_ref, row, CHUNK, h // call.group * width, width)
        entering = memories.views(h_ref, (n * call.heads + h) * width, width, 0, value_width)
        views = {**_views(lambda i: (0, 0), queries=queried, **given),
                 **_views(lambda i: (0, i), entered=entering,
                          values=held.views(v_ref, row, CHUNK, h * value_width, value_width))}

        def prologue():
            """A decay a head's prefix as a row's vectors and a column's, from global memory before the walk
            waits on its tiles. Not streamed: four more rings' stream bookkeeping spilled a value block 64
            wide's output (388 bytes on an H100's, 0a261aae0)."""
            return (tuple(family.rows_of(ref.at[h, span]) for ref in (hi_ref, lo_ref)),
                    tuple(family.columns_of(ref.at[h, span]) for ref in (column_hi_ref, column_lo_ref)))

        def step(index, slots, carry, accumulators, loaded=None):
            def block(name, operand, c, *, along_rows=False):
                place = pl.ds(c * call.block, call.block)
                return _tile(slots, name, operand, **({"rows": place} if along_rows else {"columns": place}))

            entered = functools.partial(block, "entered", memories, along_rows=True)
            value = _tile(slots, "values", held)
            if call.per_key:
                def decays(c):
                    return tuple(block(name, gates, c).value() for name in ("gates", "lows"))

                out = algebra.key_decayed_output(
                    lambda c: block("queries", call.given, c).value(), decays, entered, width // call.block,
                    _tile(slots, "scores", held).value().astype(jnp.float32), value, scale,
                    _Issuer(call, {"memory": staged, "local": staged}))
            else:
                assert loaded is not None, "a decay a head's prefix loads in the prologue"
                (gate, low), laid = loaded
                # The tiles taken before the first step's branch: a slot is the stream's, every step's.
                queries, keys = ([block(name, call.given, c) for c in range(width // call.block)]
                                 for name in ("queries", "keys"))

                @pl.when(index == 0)
                def _():
                    scored = algebra.chunk_scores(
                        queries.__getitem__, keys.__getitem__, width // call.block, gate, low, held.dtype,
                        _Issuer(call), laid=laid)
                    pieces = _split(scored.astype(jnp.float32), held.parts) if held.parts > 1 else (scored,)
                    for piece, buffer in zip(pieces, staged, strict=True):
                        family.staged(piece, buffer)

                out = algebra.chunk_output(queries.__getitem__, entered, width // call.block, gate, low,
                                           _Parts(staged, held.dtype), value, scale, _Issuer(call))
            view = o_ref.at[span, pl.ds(h * value_width + index * bv, bv)]
            family.stored([out.astype(o_ref.dtype)], [view], smem=landing["out"])
            return carry, accumulators

        _walk(streams, views, step, call, steps=steps, prologue=None if call.per_key else prologue)

    blocks, staging = _output_streams(call, bv)
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    # The prefix twice, as the solve's.
    # The chunk's scores once (the intra launch's under a decay a key dimension), then each value block's
    # memory and local products.
    scored = 0 if call.per_key else 2 * CHUNK * CHUNK * width
    work = Work(scored + 2 * CHUNK * (width + CHUNK) * value_width, steps)
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=blocks, stages=1, live=_taken(8), work=work,
                   held=_output_held(call, bv), staging=staging, landing={"out": ((CHUNK, bv), out_dtype)},
                   # A 16-bit call's takes no reduction (an A100's asked for its tiles' 49152 bytes alone).
                   persistent=True, reduces=held.parts > 1)(
        q, k if scores is None else scores, h, corrected, hi, lo, hi, lo, taken)


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
            family: mosaic.Family | None = None, interpret: bool = False, floor: float | None = None):
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
    any width whose launches the device holds (the template's legality, `planned`'s shared memory and
    registers a launch). Its states and output launches take `tiles` (the template's config,
    `templates.recurrence.template`), and its launches issue by `family`'s bundle (the device's, `family_of`,
    where None), interpreted where `interpret`. A decay a key dimension takes the layer's declared `floor`
    (`per_key_refusal`)."""
    out, final, _ = kept_forward(q, k, v, g, beta, h0, starts=starts, firsts=firsts,
                                 max_segments=max_segments, scale=scale, delta=delta, tiles=tiles, wide=wide,
                                 family=family, interpret=interpret, floor=floor, keeps=False)
    return out, final


def kept_forward(q, k, v, g, beta, h0=None, *, tiles: Tiles, starts=None, firsts=None,
                 max_segments: int | None = None,
                 scale: float | None = None, delta: bool = True, wide: bool = False,
                 family: mosaic.Family | None = None, interpret: bool = False,
                 floor: float | None = None, keeps: bool = True) -> tuple[jax.Array, jax.Array, tuple]:
    """`forward`'s `(o, final)` and what the fast backward takes rather than compute again (`gradients`'
    `inverse` and `kept`): each chunk's inverse under a decay a head (`_solved`'s, None otherwise or where
    the call `keeps` none, `forward`'s), as fla's backward keeps its `A`, and the launches' weighted keys,
    memories and corrected values, `(w, h, corrected)`, which the port keeps unless a remat rung recomputes
    them (`recurrence_chunk.RECOMPUTED`), the caller's choice."""
    run = _prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=delta,
                    tiles=tiles, family=family, interpret=interpret, floor=floor, keeps=keeps, firsts=firsts)
    call, tiles = run.call, run.tiles
    o = _output(run.q, run.k, run.h, run.corrected, run.hi, run.lo, run.taken, call, tiles, scale=run.scale,
                out_dtype=jnp.float32 if wide else run.dtype, scores=run.scores)
    o = run.unlaid(o.reshape(call.rows, call.heads, call.value_width))[..., :run.value_width]
    assert run.final is not None, "a forward runs its states launch"
    return o, run.final, (run.a, (run.w, run.h, run.corrected))


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
    final: jax.Array | None
    taken: jax.Array
    sequences: jax.Array
    laid: Callable
    unlaid: Callable


def _prepared(q, k, v, g, beta, h0, *, starts, max_segments, scale, delta, tiles, family, interpret,
              firsts=None,
              inverse=None, kept=None, floor: float | None = None, keeps: bool = True) -> _Run:
    """`forward`'s operands checked, padded and laid, and its launches before the output run: under a decay a
    head the solve's but where its `inverse` is given, and the weighted and states launches' but where their
    `kept` values, `(w, h, corrected)`, are (`kept_forward`'s; its final memory then None). The solve stores
    its inverse where the call `keeps` it (`_solved`)."""
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
    if per_key and (refused := per_key_refusal(dtype, floor)):
        raise NotImplementedError(refused)
    scale = key_width ** -0.5 if scale is None else scale
    block = block_of(family)
    width, columns = (-(-x // block) * block for x in (key_width, value_width))
    q, k = (_padded(x.astype(dtype), 3, width) for x in (q, k))
    v = _padded(v.astype(dtype), 3, columns)
    g = _padded(g.astype(jnp.float32), 3, width) if per_key else g.astype(jnp.float32)
    # The delta rule's write gate in the operands' dtype, as fla's layer holds it; without the delta rule the
    # reference's own fp32 (Mamba-2's `dt`).
    beta = (beta.astype(dtype) if delta else beta).astype(jnp.float32)
    h0 = (jnp.zeros((batch, heads, width, columns), jnp.float32) if h0 is None
          else _padded(_padded(h0.astype(jnp.float32), 2, width), 3, columns))
    if starts is not None:
        starts = jnp.asarray(starts, bool)
        h0 = jnp.where(starts[:, 0, None, None, None], 0.0, h0)
    chunks, taken, sequences, laid, unlaid = _layout(batch, length, starts, max_segments, firsts)
    rows = chunks * CHUNK
    given = _Operand(rows, _parts(dtype), dtype)
    # What the launches pass between them: the delta rule's in the operands' dtype, as fla's rounds it (the port is
    # its reference); without the delta rule fp32, its three parts, as the portable math and mamba_ssm hold their
    # scores, written values and memories (a 16-bit call's rounded to bf16 put the template's output 1.7 times and
    # its final memory up to 540 times the portable math's distance from float64, on an H100 and interpreted).
    held = given if delta else _Operand(rows, PARTS, jnp.dtype(jnp.float32))
    call = _Call(heads, heads // k.shape[2], width, columns, rows, given, held, per_key, delta, family,
                 interpret, generation.gpu_target())
    hi, lo = prefix(laid(g).reshape(chunks, CHUNK, -1))
    # A decay a head's prefix a head's row of vectors `[HV, R]`, one a key dimension's tiles `[R, HV * Dk]`.
    hi, lo = (x.reshape(rows, -1) if per_key else x.reshape(rows, heads).T for x in (hi, lo))
    beta = laid(beta).reshape(rows, heads).T
    q, k, v = (given.split(laid(x).reshape(rows, -1)) for x in (q, k, v))
    a, w, u, keys, scores = None, None, v, k, None
    if per_key and delta:
        a, scores = _intra(q, k, hi, lo, beta, taken, call, scale=scale)
        w, u, keys = _weighted(k, v, a, hi, lo, beta, taken, call)
    elif per_key:
        (scores,) = _intra(q, k, hi, lo, beta, taken, call, scale=scale)
        (keys,) = _weighted(k, v, None, hi, lo, beta, taken, call)
    elif delta and inverse is None:
        # The forward's residuals come in pairs: a recomputed forward has neither.
        assert kept is None, "a kept forward keeps its inverse"
        a, w, u = _solved(k, v, hi, lo, beta, taken, call, keeps=keeps)
    elif delta:
        a = inverse
        if kept is None:
            w, u = _weighted(k, v, a, hi, lo, beta, taken, call)
    if kept is not None and delta and not per_key:
        (w, h, corrected), final = kept, None
    else:
        h, corrected, final = _states(keys, w, u, hi, lo, beta, h0.reshape(-1, columns), sequences, call,
                                      tiles)
        final = final.reshape(batch, heads, width, columns)[:, :, :key_width, :value_width]
    return _Run(call, tiles, dtype, scale, key_width, value_width, batch, q, k, v, hi, lo, beta,
                None if per_key else a, w, scores, h, corrected, final, taken, sequences, laid,
                unlaid)


def planned(q, k, v, g, beta, h0=None, *, tiles: Tiles, family: mosaic.Family, starts=None,
            max_segments: int | None = None, delta: bool = True, backward: bool = False,
            floor: float | None = None, target: generation.GpuTarget | None = None) -> tuple[Launch, ...]:
    """The launches a call over operands of these shapes (`jax.ShapeDtypeStruct`s, `forward`'s) runs at
    `tiles` by `family`'s bundle on `target` (the device's, or the assumed one's, where None), in order, as
    planned (`Launch`): its arrays abstract (`jax.eval_shape`), its kernels declared and not built
    (`_kernel`); with `backward`, the fast backward's too (`gradients`)."""
    found: list[Launch] = []

    def run(q, k, v, g, beta, h0, starts):
        token = _PLANNED.set(found)
        try:
            out, final, (inverse, kept) = kept_forward(q, k, v, g, beta, h0, tiles=tiles, starts=starts,
                                                       max_segments=max_segments, delta=delta, family=family,
                                                       floor=floor)
            if backward:
                gradients(q, k, v, g, beta, h0, out, final, tiles=tiles, starts=starts,
                          max_segments=max_segments, family=family, inverse=inverse, kept=kept)
        finally:
            _PLANNED.reset(token)

    with contextlib.nullcontext() if target is None else generation.assumed(target):
        jax.eval_shape(run, q, k, v, g, beta, h0, starts)
    return tuple(found)


def holds(launches: tuple[Launch, ...], family: mosaic.Family, shared: int) -> bool:
    """Whether each of `launches` fits a block of `shared` bytes and its consumer threads hold their registers
    (`Launch.registers` within `Family.registers`): a generated launch spills nothing."""
    return all(launch.shared <= shared and launch.registers <= family.registers(1) for launch in launches)


def backward_holds(q, k, v, g, beta, h0=None, *, tiles: Tiles, family: mosaic.Family, shared: int,
                   starts=None, max_segments: int | None = None) -> bool:
    """Whether the fast backward's launches (`gradients`'s, past the forward's) hold on `family`'s bundle at
    `tiles` within a block of `shared` bytes (`holds`); where they do not, the rule's gradient is the
    portable math's."""
    shapes = [None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v, g, beta, h0,
                                                                                    starts)]
    *operands, starts = shapes
    forward = planned(*operands, tiles=tiles, family=family, starts=starts, max_segments=max_segments)
    both = planned(*operands, tiles=tiles, family=family, starts=starts, max_segments=max_segments,
                   backward=True)
    return holds(both[len(forward):], family, shared)


def gradients(q, k, v, g, beta, h0, do, dfinal, *, tiles: Tiles, starts=None, firsts=None,
              max_segments: int | None = None, scale: float | None = None,
              family: mosaic.Family | None = None, interpret: bool = False,
              inverse=None, kept=None):
    """The fast backward of the gated delta rule, a decay a head (`forward`'s, `g` `[B, T, HV]`), in a 16-bit
    call: `(dq, dk, dv, dg, dbeta, dh0)` for the output's and final memory's gradients `do`, `dfinal`, fla's
    DAG (recomputing the forward's values, as fla does) through the algebra's backward tile functions:
    `_local_back`, `_reverse`, `_scores_back`, `_query_key`, `_system_back`, `_values_back`, `_keys_back`,
    the decay's gradient summed token by token
    (`algebra.summed_back`) and the memory's decay's term added, a key head's value heads summed; the
    forward's `inverse` and `kept` values where given (`kept_forward`'s) rather than computed again, its
    states launch at the forward's `tiles` where they are not. Its rounding is fla's; the accurate backward is
    the portable math's token adjoint."""
    run = _prepared(q, k, v, g, beta, h0, starts=starts, max_segments=max_segments, scale=scale, delta=True,
                    tiles=tiles, family=family, interpret=interpret, inverse=inverse, kept=kept,
                    firsts=firsts)
    call = run.call
    if call.per_key or call.held.parts > 1:
        raise NotImplementedError("the fast backward takes a decay a head in a 16-bit call")
    rows, heads, width, columns = call.rows, call.heads, call.width, call.value_width
    batch = run.batch
    laid_do = call.given.split(run.laid(_padded(do.astype(run.dtype), 3, columns)).reshape(rows, -1))
    dht = _padded(_padded(dfinal.astype(jnp.float32), 2, width), 3, columns).reshape(-1, columns)
    local = _local_back(run.q, run.k, laid_do, run.hi, run.lo, run.taken, call, scale=run.scale)
    dh, dv2, dh0 = _reverse(run.q, run.k, run.w, laid_do, local, run.hi, run.lo, dht, run.sequences, call,
                            tiles, scale=run.scale)
    ds, decay_pairs = _scores_back(run.q, run.k, run.corrected, laid_do, run.hi, run.lo, run.taken, call,
                                   scale=run.scale)
    dq, dk, dw, after, leaving, memory = _query_key(run.q, run.k, run.corrected, laid_do, dv2, run.h, dh, ds,
                                                    run.hi, run.lo, run.taken, call, scale=run.scale)
    d_l = _system_back(run.a, run.k, run.v, dw, dv2, run.hi, run.lo, run.beta, run.taken, call)
    dv, dbeta_u = _values_back(run.a, run.v, dv2, run.beta, run.taken, call)
    dk_w, dbeta_k, after_w, decay_w = _keys_back(run.a, run.k, dw, d_l, run.hi, run.lo, run.beta, run.taken,
                                                 call)
    dbeta = dbeta_u + dbeta_k
    chunks = call.chunks
    # Each token's decay's gradient, the launches' (`algebra.summed_back` is linear in its terms) and the
    # memory's decay across each chunk: its gradient `sum(S dS)` through the decay `exp(g_last)` from the
    # chunk's start, every token's.
    memory = memory.reshape(-1, heads, chunks, call.block).sum((0, 3))
    ending = (run.hi + run.lo).reshape(heads, chunks, CHUNK)[..., -1]
    # The query/key launch's range terms each token's through them (`algebra.summed_back`'s ranges' part):
    # `after` from each token to the chunk's end, `leaving` from its start to before it.
    after, leaving = (x.sum(0).reshape(heads, chunks, CHUNK) for x in (after, leaving))
    after = after + after_w.reshape(heads, chunks, CHUNK)
    # `leaving`'s sum before each token, never its own term added and taken away: a token's own `leaving` is
    # order one where the terms before it are `exp(-30)`'s under steep decays, which the sum rounded away
    # (an H100's decay gradient under decays near zero 140 times the port's distance from float64, whose
    # `leaving` enters its strictly lower pairs, fla `chunk_bwd_dqkwg`; SE-numerics-h100-fd61e11eeb).
    before = jnp.concatenate((jnp.zeros_like(leaving[..., :1]), jnp.cumsum(leaving[..., :-1], -1)), -1)
    ranges = jnp.flip(jnp.cumsum(jnp.flip(after, -1), -1), -1) + before
    dg = ((decay_pairs + decay_w).reshape(heads, chunks, CHUNK) + ranges
          + (memory * jnp.exp2(ending))[..., None])
    dg, dbeta = dg.reshape(heads, rows).T, dbeta.T

    def headed(x, size):
        """`[R, HV * size]` laid back as `[B, T, HV, size]`."""
        return run.unlaid(x.reshape(rows, heads, size))

    def keyed(x):
        """A value head's gradient `[B, T, HV, Dk]` summed onto its key head's."""
        return x.reshape(*x.shape[:2], heads // call.group, call.group, width).sum(3)[..., :run.key_width]

    # The launches' gradients in the operands' dtype (fla's), summed in fp32.
    dq, dk, dk_w, dv = (x.astype(jnp.float32) for x in (dq, dk, dk_w, dv))
    key_heads = heads // call.group
    def key_headed(x):
        """`[R, H * Dk]`, a key head's, laid back as `[B, T, H, Dk]`."""
        return run.unlaid(x.reshape(rows, key_heads, width))[..., :run.key_width]

    dq = key_headed(dq)
    dk = key_headed(dk) + keyed(headed(dk_w, width))
    dv = headed(dv, columns)[..., :run.value_width]
    dh0 = dh0.reshape(batch, heads, width, columns)[:, :, :run.key_width, :run.value_width]
    if starts is not None:
        dh0 = jnp.where(jnp.asarray(starts, bool)[:, 0, None, None, None], 0.0, dh0)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), run.unlaid(dg).astype(g.dtype),
            run.unlaid(dbeta).astype(beta.dtype), dh0)


# The fast backward of a decay a head under the delta rule (fla's DAG through the algebra's backward tile
# functions, `algebra.local_gradient` ... `algebra.summed_back`), a 16-bit call's: the memory's reverse
# recurrence a sequence at a time (`_reverse`), each chunk's scores' gradient (`_scores_back`), the queries',
# keys' and weights' gradients a key block of a chunk at a time (`_query_key`), and the WY representation's
# (`_system_back`, `_values_back`, `_keys_back`); each launch writes its decay's terms a chunk and value
# head, its pairs' summed token by token (`summed_pairs`) and its ranges' by XLA's cumulative sums, which
# XLA adds.


def _local_back(q, k, do, hi, lo, taken, call: _Call, *, scale: float):
    """`[R, HV * Dv]` as `call.held`: the corrected values' gradient from each chunk's own output
    (`algebra.local_gradient`, fla's chunk_bwd_dv_local), rounded as fla's, a program a chunk and value head
    (persistent): apart from the memory's reverse walk (`_reverse`), whose tiles beside these spilled 80
    bytes a thread on an H100's."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(q_ref, k_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, dv_ref, *,
                streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = _views(queries=given.views(q_ref, row, CHUNK, keyed, width),
                       keys=given.views(k_ref, row, CHUNK, keyed, width),
                       out=given.views(do_ref, row, CHUNK, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            views |= _vector_views(n, h, **vectors)

        read = _prefix_reader(family, streamed, vectors, h, span)

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low), laid = found
            (local,) = staging

            def block(name, c):
                return _tile(slots, name, given, columns=pl.ds(c * call.block, call.block))

            # The local scores once, against the output's gradient's whole row.
            dv = algebra.local_gradient(
                functools.partial(block, "keys"), functools.partial(block, "queries"), width // call.block,
                gate, low, _tile(slots, "out", given), scale, _Issuer(call, {"local_dv": local}), laid=laid)
            held.store(dv_ref, dv, row, CHUNK, h * value_width, value_width, family=family,
                       smem=landing["row"])
            return carry, accumulators

        _walk(streams, views, step, call, prologue=None if streamed else lambda: read(None))

    tiles = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                      out=((CHUNK, value_width), given))
    tiles |= _vector_blocks(call, "gate", "low", "column_gate", "column_low")
    # The local scores, then their product with the output's gradient.
    work = Work(2 * CHUNK * CHUNK * (width + value_width), exponentials=CHUNK * CHUNK)
    # The local scores, their decays and masks and their narrow form, and their product and its cast.
    held_tiles = _Held(phases=([_f32(CHUNK, CHUNK)] * 2 + _masks(CHUNK, CHUNK) + _held(held, CHUNK, CHUNK)
                               + [_f32(CHUNK, value_width)] + _held(held, CHUNK, value_width)
                               + _fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),))
    return _kernel(program, call, out_type=held.shape(call.heads * value_width),
                   grid=(call.chunks, call.heads), grid_names=("chunks", "heads"), blocks=tiles, stages=1,
                   live=_taken(7), work=work, held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)),
                   landing={"row": held.landing(CHUNK, value_width)}, persistent=True)(
        q, k, do, hi, lo, hi, lo, taken)


def _reverse(q, k, w, do, local, hi, lo, dht, sequences, call: _Call, tiles: Tiles, *, scale: float):
    """`(dh [chunks * HV * Dk, Dv], dv [R, HV * Dv], dh0 [B * HV * Dk, Dv])`: the gradient of the memory each
    chunk leaves as `call.memories`, the corrected values' gradient as `call.held` (their own output's,
    `local`, `_local_back`'s, and the memory's, `algebra.reverse_carried`) and each row's initial memory's
    gradient, fp32, each sequence's chunks from its last over a value block, the states launch's
    (`tiles.block_v`; fla's dhu). `q`, `k`, `w`, `do` and `local` are laid as the forward's; `dht`
    `[B * HV * Dk, Dv]` fp32 each row's final memory's gradient."""
    width, value_width, family = call.width, call.value_width, call.family
    bv, slices = tiles.block_v, width // call.block
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    batch = dht.shape[0] // (call.heads * width)
    final = _Operand(dht.shape[0], 1, jnp.dtype(jnp.float32))
    chunks = call.chunks

    def program(q_ref, k_ref, w_ref, do_ref, local_ref, hi_ref, lo_ref, dht_ref, seq_ref, dh_ref, dv_ref,
                dh0_ref, *, streams, staging, landing):
        j, s, h = lax.axis_index("blocks"), lax.axis_index("sequences"), lax.axis_index("heads")
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        staged, (second,) = staging
        # The decayed output gradient fp32 as two parts against the queries (fla's split product, 16 of its
        # significand's bits): one part put the initial memory's gradient 5 to 7 times the port's distance
        # from float64. Its first part reuses the memory products' tile, each consumed before the next
        # stages.
        issue = _Issuer(call, {**dict.fromkeys(("values_dv", "weights_dv"), staged),
                               "reads": (*staged, second)})

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        key_columns = h // call.group * width
        # Step `f` the chunk `chunks - 1 - f`: a sequence's chunks from its last, walked forward.
        views = _views(lambda f: (chunks - 1 - f, 0),
                       queries=given.views(q_ref, 0, call.rows, key_columns, width),
                       keys=given.views(k_ref, 0, call.rows, key_columns, width),
                       w=held.views(w_ref, 0, call.rows, h * width, width),
                       out=given.views(do_ref, 0, call.rows, h * value_width + j * bv, bv),
                       local=held.views(local_ref, 0, call.rows, h * value_width + j * bv, bv))
        final_views = _views(**{f"final{i}": final.views(dht_ref, memory(row, i), call.block, j * bv, bv)
                                for i in range(slices)})
        walked = family.opened(streams["walked"], views)
        leaving = family.opened(streams["once"], final_views)

        # Each key slice of the memory's gradient and the corrected values', a store each in flight.
        before_store = _walk_stores(family, width // call.block + 1)

        def step(f, slots, grad, accumulators):
            n = chunks - 1 - f
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(grad):
                before_store()
                memories.store(dh_ref, part, memory(n, i), call.block, j * bv, bv, family=family,
                               smem=landing[f"memory{i}"], wait=False)

            def block(name, operand, i):
                return _tile(slots, name, operand, columns=pl.ds(i * call.block, call.block))

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            out = _tile(slots, "out", given)
            local = _tile(slots, "local", held).value().astype(jnp.float32)
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            grad, dv = algebra.reverse_carried(
                grad, functools.partial(block, "keys", given), functools.partial(block, "w", held),
                functools.partial(block, "queries", given), out.value(), local, gate, low, ending,
                ending_low, scale, held.dtype, issue)
            before_store()
            held.store(dv_ref, dv, n * CHUNK, CHUNK, h * value_width + j * bv, bv, family=family,
                       smem=landing["corrected"], wait=False)
            return grad, accumulators

        _sequence_walk(call, final, "final", (leaving, final_views), (walked, views), step,
                       chunks - first - count, count, ends, begins,
                       lambda i: dh0_ref.at[pl.ds(memory(row, i), call.block), pl.ds(j * bv, bv)], bv,
                       landing=landing.get("final"))

    out_type = (memories.shape(value_width), held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                       w=((CHUNK, width), held), out=((CHUNK, bv), given), local=((CHUNK, bv), held))
    finals = _streamed(**{f"final{i}": ((call.block, bv), final) for i in range(slices)})
    # The memory's carried gradient: three memory products a chunk.
    steps = chunks / sequences.shape[0]
    work = Work(2 * 3 * CHUNK * width * bv * steps, steps)
    # The memory's gradient carried, its key slices in fp32, and the one it gives the chunk before as it is
    # summed; a step's corrected values' two gradients and its local one beside the output's gradient, a
    # memory slice as an operand, and the stored tiles' casts.
    held_tiles = _Held(carried=[_f32(width, bv)] * 2, phases=(
        [_f32(CHUNK, bv)] * 3 + _held(given, CHUNK, bv)
        + _held(held, call.block, bv) + _held(memories, call.block, bv) + _held(held, CHUNK, bv)
        + _fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block)),))
    final_landing, after = _final_landing(call, bv)
    return _kernel(program, call, out_type=out_type, grid=(value_width // bv, sequences.shape[0], call.heads),
                   grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=2, work=work,
                   held=held_tiles,
                   live=lambda refs: refs[8][lax.axis_index("sequences"), 1] > 0,
                   staging=(*_staging(held, (call.block, bv)), (plgpu.SMEM((CHUNK, bv), held.stored),)),
                   once=finals,
                   landing={**_memory_landing(call, bv), "corrected": held.landing(CHUNK, bv),
                            **final_landing},
                   after=after)(q, k, w, do, local, hi, lo, dht, sequences)


def _query_key_views(call: _Call, keyed: tuple, served: tuple, memory: tuple, *, row, key_columns, first,
                     chunk, block) -> dict:
    """`_query_key`'s streams' views (`_views`) for a program's key block `block` of chunk `chunk`, its key
    head's first value head `first`: the queries' and keys' block (`keyed`, their refs), and each step `i`
    the group's `i`-th value head's values, output gradient and corrected values' gradient (`served`) at
    its columns, and its entered memory and leaving memory's gradient (`memory`) a head's rows apart."""
    width, value_width, group = call.width, call.value_width, call.group
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    q_ref, k_ref = keyed
    operands = dict(zip(("values", "out", "value_grads"), ((held, served[0]), (given, served[1]),
                                                           (held, served[2])), strict=True))
    memory_row = (chunk * call.heads + first) * width + block * call.block
    return {**_views(lambda i: (0, 0), queries=given.views(q_ref, row, CHUNK, key_columns, call.block),
                     keys=given.views(k_ref, row, CHUNK, key_columns, call.block)),
            **_views(lambda i: (0, i), **{name: operand.views(ref, row, CHUNK, first * value_width,
                                                              group * value_width)
                                          for name, (operand, ref) in operands.items()}),
            **_views(lambda i: (i * (width // call.block), 0),
                     **{name: memories.views(ref, memory_row, group * width, 0, value_width)
                        for name, ref in zip(("entered", "leaving"), memory, strict=True)})}


def _gradient_stored(call: _Call, kind: str, block: int, grad, *, weights: Sequence, sums: dict, first,
                     landing: dict) -> None:
    """`algebra.memory_gradients`' emit in `_query_key`: the weights' gradient (`kind` 'dw') in the
    operands' dtype to `weights` (its parts' views) as it is final, as fla's dqkwg writes it; the queries'
    and keys' ('dq', 'dk') to `sums[kind]` summed over the key head's value heads in fp32, stored at the
    first (`first`) and added at the rest (`Family.added`: a TMA reduction, or whole rows through
    registers), each through its launch's landing tiles."""
    del block
    family = call.family
    if kind == "dw":
        pieces = _split(grad.astype(jnp.float32)) if call.held.parts > 1 else (grad,)
        family.stored([piece.astype(call.held.stored) for piece in pieces], weights, smem=landing["weights"])
        return

    @pl.when(first)
    def _():
        family.stored([grad], [sums[kind]], smem=landing["sums"])

    @pl.when(jnp.logical_not(first))
    def _():
        family.added(grad, sums[kind], call.interpret, smem=landing["sums"])


def _query_key_held(call: _Call) -> _Held:
    """What `_query_key`'s consumer thread holds (`algebra.memory_gradients`): a block's gradient's fp32 sum,
    its memory product and its scores' product, a block's queries or keys in fp32 for the range terms, and a
    block of the memories entered and left for `sum(S dS)`; where products read registers, a block of each
    operand."""
    held, memories, given = call.held, call.memories, call.given
    return _Held(phases=(
        [_f32(CHUNK, call.block)] * 4 + [_f32(call.block, call.block)] * 2
        + _fragments(call, (given, CHUNK, call.block), (held, CHUNK, call.block), (memories, call.block,
                                                                                      call.block)),))


def _query_key(q, k, corrected, do, dv, h, dh, ds, hi, lo, taken, call: _Call, *, scale: float):
    """`(dq, dk [R, H * Dk] fp32, dw [R, HV * Dk] as call.held, after, leaving [blocks, HV, R], memory
    [blocks, HV, chunks * block] fp32)`: each chunk's and key block's queries', keys' and weights' gradients
    (fla's dqkwg's memory products, `algebra.memory_gradients`, given the scores' gradient `ds` `[R, HV * C]`
    as `call.held`, `_scores_back`'s), a program a key head's and a step each value head it serves, the
    queries' and keys' summed over them in fp32 where they leave (each program's own tiles: a register
    carry across the steps spilled), as the port's sum; the decay's range terms over the block's key
    columns, each token's (`algebra.memory_gradients`' `(after, leaving)`, which `gradients` sums over their
    ranges: summed here, their `[C, C]` tiles spilled 376 bytes a thread on an H100's), and the block's rows
    of `sum(S dS)`, the memory's decay's gradient across the chunk. `h` and `dh` the memories each chunk
    enters and its leaving memory's gradient (`call.memories`)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, memories, given = dataclasses.replace(call.held, rows=call.rows), call.memories, call.given
    blocks, group = value_width // call.block, call.group

    def program(q_ref, k_ref, v_ref, do_ref, dv_ref, h_ref, dh_ref, ds_ref, hi_ref, lo_ref, taken_ref,
                dq_ref, dk_ref, dw_ref, after_ref, leaving_ref, memory_ref, *, streams, staging, landing):
        del staging
        c, n, keyed = lax.axis_index("blocks"), lax.axis_index("chunks"), lax.axis_index("heads")
        row, span, last = n * CHUNK, pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
        key_columns = keyed * width + c * call.block
        first = keyed * group
        views = _query_key_views(call, (q_ref, k_ref), (v_ref, do_ref, dv_ref), (h_ref, dh_ref), row=row,
                                 key_columns=key_columns, first=first, chunk=n, block=c)
        # Step `i` the group's `i`-th value head's scores' gradient.
        views |= _views(lambda i: (0, i), ds=held.views(ds_ref, row, CHUNK, first * CHUNK, group * CHUNK))

        def step(index, slots, carry, accumulators):
            h = first + index

            column = functools.partial(_column, call, slots)

            def whole(name, operand, i):
                del i
                return _tile(slots, name, operand)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            ending, ending_low = (jnp.full((CHUNK,), ref[h, last]) for ref in (hi_ref, lo_ref))
            first_head = index == 0

            stored = functools.partial(
                _gradient_stored, call, weights=held.views(dw_ref, row, CHUNK, h * width + c * call.block,
                                                           call.block),
                sums={"dq": dq_ref.at[span, pl.ds(key_columns, call.block)],
                      "dk": dk_ref.at[span, pl.ds(key_columns, call.block)]},
                first=first_head, landing=landing)

            after, leaving = algebra.memory_gradients(
                functools.partial(whole, "queries", given), functools.partial(whole, "keys", given), 1,
                functools.partial(column, "values", held), functools.partial(column, "out", given),
                functools.partial(column, "value_grads", held),
                lambda i, j: column("entered", memories, j), lambda i, j: column("leaving", memories, j),
                blocks, _tile(slots, "ds", held), gate, low, ending, ending_low, None, scale, _Issuer(call),
                emit=stored)
            after_ref.at[c, h, span][...] = after
            leaving_ref.at[c, h, span][...] = leaving
            # `sum(S dS)` over the block's rows, each row's, read here rather than both memories again.
            memory_ref.at[c, h, pl.ds(n * call.block, call.block)][...] = functools.reduce(jnp.add, (
                jnp.sum(column("entered", memories, j).value().astype(jnp.float32)
                        * column("leaving", memories, j).value().astype(jnp.float32), axis=1)
                for j in range(blocks)))
            # The step's reduce-adds ('dq', 'dk', `Family.added`) waited on by design, not by 'dw''s store
            # being emitted after them: Mosaic GPU waits on no copy at a kernel's exit (template numerics
            # review 2026-10-26, P3-4).
            family.store_wait(0)
            return carry, accumulators

        _walk(streams, views, step, call, steps=group)

    tiles = _streamed(queries=((CHUNK, call.block), given), keys=((CHUNK, call.block), given),
                      values=((CHUNK, value_width), held), out=((CHUNK, value_width), given),
                      value_grads=((CHUNK, value_width), held), entered=((call.block, value_width), memories),
                      leaving=((call.block, value_width), memories), ds=((CHUNK, CHUNK), held))
    keyed_out = jax.ShapeDtypeStruct((call.rows, call.heads // group * width), jnp.float32)
    terms = jax.ShapeDtypeStruct((width // call.block, call.heads, call.rows), jnp.float32)
    out_type = (keyed_out, keyed_out, held.shape(call.heads * width), terms, terms,
                jax.ShapeDtypeStruct((width // call.block, call.heads, call.chunks * call.block),
                                     jnp.float32))
    # Each value head of the group: three memory products and the scores' gradient's two.
    each = 3 * CHUNK * value_width * call.block + 2 * CHUNK * CHUNK * call.block
    work = Work(2 * each * group, group)
    return _kernel(program, call, out_type=out_type,
                   grid=(width // call.block, call.chunks, call.heads // group),
                   grid_names=("blocks", "chunks", "heads"), blocks=tiles, stages=1, live=_taken(10),
                   work=work, held=_query_key_held(call),
                   landing={"weights": held.landing(CHUNK, call.block),
                            "sums": ((CHUNK, call.block), jnp.float32)})(
        q, k, corrected, do, dv, h, dh, ds, hi, lo, taken)


def _scores_back(q, k, corrected, do, hi, lo, taken, call: _Call, *, scale: float):
    """`(ds [R, HV * C] as call.held, decay [HV, R] fp32)`: each chunk and value head's scores' gradient and
    the decay's gradient from its pair terms (`algebra.scores_gradient`, `algebra.summed_pairs`), fla's
    dqkwg's part the memory's products do not take, a program a chunk and value head (persistent): apart
    from `_query_key`, each launch's tiles a thread's registers hold (one launch spilled 1.5 KB a thread on
    an H100's, SE bwd-probe at a49c8da50)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given
    blocks = value_width // call.block

    def program(q_ref, k_ref, v_ref, do_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref, ds_ref,
                decay_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = _views(queries=given.views(q_ref, row, CHUNK, keyed, width),
                       keys=given.views(k_ref, row, CHUNK, keyed, width),
                       values=held.views(v_ref, row, CHUNK, h * value_width, value_width),
                       out=given.views(do_ref, row, CHUNK, h * value_width, value_width))
        vectors = {"gate": hi_ref, "low": lo_ref, "column_gate": column_hi_ref, "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            views |= _vector_views(n, h, **vectors)

        read = _prefix_reader(family, streamed, vectors, h, span)

        def step(index, slots, carry, accumulators, loaded=None):
            del index
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            (gate, low), laid = found

            column = functools.partial(_column, call, slots)

            summed, columns = _summed(tuple(staging), call)
            ds, pairs = algebra.scores_gradient(
                functools.partial(column, "queries", given), functools.partial(column, "keys", given),
                width // call.block, functools.partial(column, "values", held),
                functools.partial(column, "out", given), blocks, gate, low, scale, held.dtype, _Issuer(call),
                laid=laid)
            held.store(ds_ref, ds, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, columns)
            return carry, accumulators

        _walk(streams, views, step, call, prologue=None if streamed else lambda: read(None))

    tiles = _streamed(queries=((CHUNK, width), given), keys=((CHUNK, width), given),
                      values=((CHUNK, value_width), held), out=((CHUNK, value_width), given))
    tiles |= _vector_blocks(call, "gate", "low", "column_gate", "column_low")
    out_type = (held.shape(call.heads * CHUNK), jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    # The scores' gradient, the scores and the pairs' sums.
    work = Work(2 * CHUNK * CHUNK * (value_width + width + CHUNK), exponentials=CHUNK * CHUNK)
    # The scores' gradient fp32 and narrow, the pairs' decays, the scores and the pair terms, the masks, and
    # the pairs' sum's product.
    held_tiles = _Held(phases=([_f32(CHUNK, CHUNK)] * 5 + _held(held, CHUNK, CHUNK) + _masks(CHUNK, CHUNK)
                               + _fragments(call, (given, CHUNK, call.block), (given, CHUNK, call.block)),))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(8), work=work,
                   held=held_tiles, staging=_summing(call), landing={"square": held.landing(CHUNK, CHUNK)},
                   persistent=True)(q, k, corrected, do, hi, lo, hi, lo, taken)


def _system_back(a, k, v, dw, du, hi, lo, beta, taken, call: _Call):
    """`dL [R, HV * C]` as `call.held`: the WY representation's backward's system part a chunk and value head
    at a time (fla's prepare_wy_repr_bwd's, `algebra.system_gradient`) from `w`'s and `u`'s gradients `dw`,
    `du` (`call.held`) and the inverse `a` less its unit diagonal (`_solved`'s), which `_keys_back` takes.
    Apart from `_keys_back` and `_values_back`, each launch's tiles a thread's registers hold (in one they
    spilled 360 bytes a thread on an H100's)."""
    width, value_width, family = call.width, call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, k_ref, v_ref, dw_ref, du_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref,
                taken_ref, dl_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       keys=given.views(k_ref, row, CHUNK, h // call.group * width, width),
                       values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                       w_grads=held.views(dw_ref, row, CHUNK, h * width, width),
                       u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(_column, call, slots)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            d_l = algebra.system_gradient(
                _tile(slots, "lower", held), functools.partial(column, "keys", given), width // call.block,
                functools.partial(column, "values", given), value_width // call.block,
                functools.partial(column, "w_grads", held), functools.partial(column, "u_grads", held), beta,
                gate, low, held.dtype, _Issuer(call, dict.fromkeys(("inverse_t", "inverse_dx", "inverse_l"),
                                                                   staged)),
                laid=(family.columns_of(column_hi_ref.at[h, span]),
                      family.columns_of(column_lo_ref.at[h, span])))
            held.store(dl_ref, d_l, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), keys=((CHUNK, width), given),
                      values=((CHUNK, value_width), given), w_grads=((CHUNK, width), held),
                      u_grads=((CHUNK, value_width), held))
    # The weighted keys' and values' products into the inverse's gradient, and its products through it.
    work = Work(2 * CHUNK * CHUNK * (width + value_width + 2 * CHUNK))
    # A block's keys or values scaled, narrow, beside the inverse's gradient; then it masked, its products
    # through the inverse and the pairs' decays with their masks (`algebra.system_gradient`).
    held_tiles = _Held(phases=(
        [_f32(CHUNK, CHUNK), _f32(CHUNK, call.block), *_held(held, CHUNK, call.block),
         *_fragments(call, (held, CHUNK, call.block), (held, CHUNK, call.block))],
        [_f32(CHUNK, CHUNK)] * 4 + _held(held, CHUNK, CHUNK) + _masks(CHUNK, CHUNK)
        + _fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, CHUNK))))
    return _kernel(program, call, out_type=held.shape(call.heads * CHUNK), grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(10), work=work,
                   held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)),
                   landing={"square": held.landing(CHUNK, CHUNK)})(a, k, v, dw, du, hi, lo, beta, hi, lo,
                                                                   taken)


def _values_back(a, v, du, beta, taken, call: _Call):
    """`dv [R, HV * Dv]` as `call.held` and `dbeta [HV, R]` fp32: the WY representation's backward's values'
    part a chunk and value head at a time (`algebra.values_gradient`): the values' gradient and beta's
    through `u`."""
    value_width, family = call.value_width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, v_ref, du_ref, beta_ref, taken_ref, dv_ref, dbeta_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        (staged,) = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       values=given.views(v_ref, row, CHUNK, h * value_width, value_width),
                       u_grads=held.views(du_ref, row, CHUNK, h * value_width, value_width))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(_column, call, slots)
            stored = _block_stored(call, dv_ref, row, h * value_width, landing["block"])

            dbeta_ref.at[h, span][...] = algebra.values_gradient(
                _tile(slots, "lower", held), functools.partial(column, "values", given),
                value_width // call.block, functools.partial(column, "u_grads", held),
                family.rows_of(beta_ref.at[h, span]), held.dtype, _Issuer(call, {"inverse_t": staged}),
                emit=stored)
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), values=((CHUNK, value_width), given),
                      u_grads=((CHUNK, value_width), held))
    out_type = (held.shape(call.heads * value_width),
                jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32))
    work = Work(2 * CHUNK * CHUNK * value_width)
    # A value block's gradient in fp32 and narrow, its product through the inverse, and the values in fp32.
    held_tiles = _Held(phases=(
        [_f32(CHUNK, call.block)] * 3 + _held(held, CHUNK, call.block)
        + _fragments(call, (held, CHUNK, CHUNK), (held, CHUNK, call.block)),))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(4), work=work,
                   held=held_tiles, staging=_staging(held, (CHUNK, call.block)),
                   landing={"block": held.landing(CHUNK, call.block)})(a, v, du, beta, taken)


def _keys_back(a, k, dw, dl, hi, lo, beta, taken, call: _Call):
    """`dk [R, HV * Dk]` as `call.held` and `(dbeta, after, decay [HV, R])` fp32: the WY representation's
    backward's keys' part a chunk and value head at a time (`algebra.keys_gradient`), given the system's
    gradient `dl` (`_system_back`'s): the keys' gradient through `w` and through the system, beta's
    through both, the decay's range terms through `w`, and its gradient from the pair terms
    (`algebra.summed_pairs`, in the launch)."""
    width, family = call.width, call.family
    held, given = dataclasses.replace(call.held, rows=call.rows), call.given

    def program(a_ref, k_ref, dw_ref, dl_ref, hi_ref, lo_ref, beta_ref, taken_ref, dk_ref, dbeta_ref,
                after_ref, decay_ref, *, streams, staging, landing):
        n, h = _axis("chunks"), _axis("heads")
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        staged, *summing = staging
        views = _views(lower=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                       keys=given.views(k_ref, row, CHUNK, h // call.group * width, width),
                       w_grads=held.views(dw_ref, row, CHUNK, h * width, width),
                       system=held.views(dl_ref, row, CHUNK, h * CHUNK, CHUNK))

        def step(index, slots, carry, accumulators):
            del index

            column = functools.partial(_column, call, slots)

            gate, low = family.rows_of(hi_ref.at[h, span]), family.rows_of(lo_ref.at[h, span])
            beta = family.rows_of(beta_ref.at[h, span])
            stored = _block_stored(call, dk_ref, row, h * width, landing["block"])

            d_beta, after, pairs = algebra.keys_gradient(
                _tile(slots, "lower", held), functools.partial(column, "keys", given), width // call.block,
                functools.partial(column, "w_grads", held), _tile(slots, "system", held), beta, gate, low,
                held.dtype, _Issuer(call, dict.fromkeys(("inverse_t", "keys_dl", "keys_dlt"), staged)),
                emit=stored)
            summed, columns = _summed(tuple(summing), call)
            decay_ref.at[h, span][...] = algebra.summed_pairs(pairs, summed, columns)
            dbeta_ref.at[h, span][...] = d_beta
            after_ref.at[h, span][...] = after
            return carry, accumulators

        _walk(streams, views, step, call)

    tiles = _streamed(lower=((CHUNK, CHUNK), held), keys=((CHUNK, width), given),
                      w_grads=((CHUNK, width), held), system=((CHUNK, CHUNK), held))
    tokens = jax.ShapeDtypeStruct((call.heads, call.rows), jnp.float32)
    out_type = (held.shape(call.heads * width), tokens, tokens, tokens)
    # Each key block's products through `w` and the system, and the keys' own pairs.
    work = Work(2 * CHUNK * CHUNK * (4 * width))
    # The system's gradient narrow and scaled by beta beside a key block's gradient terms and their sum;
    # then the keys' own pairs, the pair terms, and the pairs' sums' masks and product
    # (`algebra.keys_gradient`).
    held_tiles = _Held(phases=(
        _held(held, CHUNK, CHUNK) * 2 + [_f32(CHUNK, call.block)] * 5
        + _fragments(call, (held, CHUNK, CHUNK), (given, CHUNK, call.block), (held, CHUNK, call.block)),
        _held(held, CHUNK, CHUNK) + [_f32(CHUNK, CHUNK)] * 4 + _masks(CHUNK, CHUNK)))
    return _kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                   grid_names=("chunks", "heads"), blocks=tiles, stages=1, live=_taken(7), work=work,
                   held=held_tiles, staging=_staging(held, (CHUNK, CHUNK)) + _summing(call),
                   landing={"block": held.landing(CHUNK, call.block)})(a, k, dw, dl, hi, lo, beta, taken)
