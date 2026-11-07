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
  sub-chunks on the tensor cores (`algebra.key_block_pairs`, which the call's declared floor bounds), the
  system inverted by a launch of its own under the delta rule;
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
dtype, the memory fp32 across chunks. An fp32 call keeps them fp32, and its products take each operand as
the run's forms ask (`_parts`): under the accurate forms as three bf16 parts summing to it exactly
(`product.formed`), its product the six of the parts' whose orders sum under three, an fp32 product's
accuracy; under the fast forms as itself, a TF32 operand, on a family whose products take transposed TF32
operands (mma.sync's, `_fast_parts`: JAX's default precision for an fp32 product, and fla's, which its
recording holds it to), else as `FAST_PARTS` bf16 parts.

The fast backward's launches and the plan of a call's launches, forward and backward, are
`mainloop_backward`'s, on this module's machinery. Decode is not on the mainloop: a one-token step takes
`recurrence.step` (row 1424 for its Mosaic form)."""

import contextvars
import dataclasses
import functools
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, recurrence_algebra as algebra, tuning
from dew.nn.kernels.templates import bundles, mosaic, product, resources
from dew.nn.kernels.templates.cost import Work, estimate
from dew.nn.kernels.templates.operands import Issuer, Parts


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


def value_unit(family: mosaic.Family) -> int:
    """The values' padded width's step on `family`'s bundle: a staged operand's narrowest block of 16-bit
    columns (`Family.narrowest_columns`), which a value tile (`Call.value_block`) and a walk's value block
    divide."""
    return family.narrowest_columns(jnp.bfloat16)


def block_of(family: mosaic.Family) -> int:
    """A tile's width on `family`'s bundle: the keys' and values' column blocks and the memory's key slices,
    the widest depth its products step over in 16-bit operands (`Family.depths`), whose bf16 parts an fp32
    call's products take too."""
    return family.depths(jnp.bfloat16)[0]

PARTS = -(-(jnp.finfo(jnp.float32).nmant + 1) // (jnp.finfo(jnp.bfloat16).nmant + 1))
"""The bf16 parts of an fp32 operand: its significand's 24 bits (23 stored and the implicit one) a bf16
part's 8 at a time, each part the rounding error of those before it, three."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A forward's tiles, the recurrence template's config, derived from the call's shape and the device's
    and its bundle's facts (`templates.recurrence.template.Mosaic.derived`): the value block the states
    launch splits the values into (`block_v`), the chunks it has in flight (`stages`), the output
    launch's value block (`block_o`), the states launch's where None, and the value blocks of one sequence
    and head a walk program takes, one a consumer role beside one producer (`walkers`: the states and
    reverse launches'; their chains overlap on one multiprocessor and their keys stream once). `forward`
    are the tiles a forward without a gradient takes where they differ: no fast backward reads its states
    launch's block, which a training call's tiles narrow to hold it."""

    block_v: int
    stages: int
    block_o: int | None = None
    walkers: int = 1
    forward: "Tiles | None" = None

    @property
    def output_block(self) -> int:
        return self.block_v if self.block_o is None else self.block_o


def per_key_refusal(dtype) -> str | None:
    """Why the mainloop refuses a call of `dtype` under a decay a key dimension, None where it takes it: its
    intra launch factors each pair's decay over the chunk's sub-chunks (`algebra.key_block_pairs`), each
    sub-chunk's own pairs too where the layer's declared floor bounds a sub-chunk's range
    (`algebra.factored_range`), else each its own decay (`algebra.key_diagonal_pairs`), and rounds each
    side's gate once, where the fp32 recurrence rounds each pair's (1.98 times its distance from float64 in
    an fp32 call, fam-t)."""
    if jnp.dtype(dtype) == jnp.float32:
        return "an fp32 call's decay a key dimension is the portable math's"
    return None


FAST_PARTS = 2
"""The bf16 parts of an fp32 operand under the run's fast forms where its products take no transposed TF32
operand (`_fast_parts`): 16 of its significand's 24 bits, past TF32's 10, JAX's default precision for an
fp32 product (and fla's), their products the parts' three pairs whose orders sum under two. TF32 itself
takes no transposed operand on a warpgroup MMA ('Only f16 WGMMA supports transposes', tf32-sm90 at the gaps'
first head)."""


def _fast_parts(family: mosaic.Family) -> int:
    """The parts an fp32 operand takes under the run's fast forms on `family`: itself, as TF32, where its
    products take transposed TF32 operands (`Family.transposes_tf32`, mma.sync's), JAX's default precision for
    an fp32 product and fla's (Triton's tf32 dot); else `FAST_PARTS` bf16 parts."""
    return 1 if family.takes_tf32 and family.transposes_tf32 else FAST_PARTS


def _parts(dtype, family: mosaic.Family, fast: bool = False) -> int:
    """The parts a product takes an operand of `dtype` as on `family`: one for 16 bits; for fp32
    `_fast_parts` under the run's fast forms (`fast`), else three."""
    if jnp.dtype(dtype) != jnp.float32:
        return 1
    return _fast_parts(family) if fast else PARTS


def diagonal(size: int, family: mosaic.Family) -> jax.Array:
    """A `[size, size]` tile's diagonal, in `family`'s layout."""
    shape = (size, size)
    return family.iota(shape, 0) == family.iota(shape, 1)


@dataclasses.dataclass(frozen=True)
class Operand:
    """A launch's operand in global memory, `[parts * rows, columns]`: its parts (`product.formed`) stacked
    `rows` apart, each `[rows, columns]`, and `dtype` its value's (fp32 where three parts)."""

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
        unless `wait` (`Family.stored`, `walk_stores`)."""
        pieces = product.formed(value.astype(jnp.float32), self.parts) if self.parts > 1 else (value,)
        family.stored([piece.astype(self.stored) for piece in pieces],
                      self.views(ref, row, rows, column, columns), smem=smem, wait=wait)

    def landing(self, rows: int, columns: int) -> tuple:
        """A store's `landing` entry (`launch_kernel`): a `[rows, columns]` tile in its stored dtype."""
        return (rows, columns), self.stored

    def shape(self, columns: int) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.parts * self.rows, columns), self.stored)

    def split(self, x):
        """`x` `[rows, columns]` in its dtype as its stacked parts."""
        if self.parts == 1:
            return x.astype(self.dtype)
        return jnp.concatenate(product.formed(x.astype(jnp.float32), self.parts), 0)

    def joined(self, x):
        """Its stacked parts `x` as the value `[rows, columns]` they sum to, fp32 (`split`'s inverse)."""
        return x.astype(jnp.float32).reshape(self.parts, -1, x.shape[-1]).sum(0)


@dataclasses.dataclass(frozen=True)
class Call:
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
    given: Operand
    held: Operand
    per_key: bool
    delta: bool
    family: mosaic.Family
    interpret: bool
    target: generation.GpuTarget
    ranged: bool = False
    """Whether its solve holds the chunk inverse at fp32's range: a 16-bit delta rule whose form proves no
    bound on the inverse's entries (`recurrence.Hooks.inverse_bound`), its merges in bf16 parts
    (`_merging`)."""
    factored: bool = True
    """Whether a decay a key dimension's pairs factor over whole sub-chunks (`algebra.factored_range` of the
    call's declared floor): where not, each sub-chunk's own pairs are each their own decay
    (`algebra.key_diagonal_pairs`)."""

    @property
    def gated_keys(self) -> bool:
        """Whether the write gate takes the keys' side: a call without the delta rule whose values pass
        between launches in its operands' 16-bit dtype (a rounded call under the run's fast forms,
        `prepared`), the values themselves then held and the gate scaling the keys' rows, the scores' columns
        and their gradients in fp32 before their products round, as mamba_ssm places its `dt`: gating the
        values, `x dt` rounded put a token's write one rounding past mamba_ssm's (B = 1, x = 129/128,
        dt = 129/256: 130/256 against 16641/32768)."""
        return not self.delta and self.held.parts == 1

    @property
    def gates(self) -> Operand:
        """A decay a key dimension's prefix halves, `[rows, heads * width]` fp32 each."""
        return Operand(self.rows, 1, jnp.dtype(jnp.float32))

    @property
    def chunks(self) -> int:
        return self.rows // CHUNK

    @property
    def block(self) -> int:
        """A tile's width (`block_of`)."""
        return block_of(self.family)

    @property
    def value_block(self) -> int:
        """A value tile's width where a launch takes the values' whole width a block at a time: the widest
        dividing both the tile's width and the values' (`value_unit`'s multiple for a forward keeping
        nothing, `prepared`). Padded to the tile's width, Nemotron-H 4B's 80-wide heads ran 128 wide, the
        forward's every launch 1.6 times the columns and 0.30 ms of XLA's pads and slices around its 1.25
        on an A100 (ssd-kernels-a100-d8d788006)."""
        return math.gcd(self.value_width, self.block)

    @property
    def memories(self) -> Operand:
        """The memories the chunks enter, `[chunks * heads * width, value_width]` as `held`."""
        return dataclasses.replace(self.held, rows=self.chunks * self.heads * self.width)


def stream_blocks(**blocks) -> dict:
    """Each named stream's parts' blocks, `name0`, `name1`, ...: `blocks[name]` its tile's shape and its
    `Operand`."""
    return {f"{name}{p}": (shape, operand.stored) for name, (shape, operand) in blocks.items()
            for p in range(operand.parts)}


def stream_tile(slots, name: str, operand: Operand, rows=None, columns=None) -> Parts:
    """Stream `name`'s slot (its parts `name0`, `name1`, ...) as a `Parts`, at a block of `rows` or `columns`
    (`pl.ds`) where given."""
    def view(ref):
        if rows is not None:
            ref = ref.at[rows, :]
        return ref if columns is None else ref.at[:, columns]

    return Parts((view(slots[f"{name}{p}"]) for p in range(operand.parts)), operand.dtype)


def stream_views(index_map=None, /, **parted) -> dict:
    """A launch's streams' views by stream part name (`Family.opened`'s `(ref, index_map)`, `mosaic.viewed`),
    `parted[name]` a list of an operand's parts' views: step `i` the `i`-th tile of the stream's rows down
    its view, as wide as the view (the whole view where the stream's tile is), or the tile `index_map(i)`
    names."""
    return {f"{name}{p}": (view, _down if index_map is None else index_map) for name, views in parted.items()
            for p, view in enumerate(views)}


def column_tile(call: Call, slots, name: str, operand: Operand, j, width: int | None = None):
    """Stream `name`'s slot's `j`-th block of `width` columns, `call.block` where None (`stream_tile`)."""
    width = call.block if width is None else width
    return stream_tile(slots, name, operand, columns=pl.ds(j * width, width))


def block_stored(call: Call, ref, row, column, smem, width: int | None = None):
    """An algebra's emit (`emit(kind, index, grad)`) storing each block's gradient as it comes, in the
    operands' dtype (as fla's prepare_wy_repr_bwd writes them), to `ref`'s `index`-th block of `width`
    columns (`call.block` where None) from `column` on the chunk's rows from `row`, through `smem`
    (`Operand.store`)."""
    held = dataclasses.replace(call.held, rows=call.rows)
    width = call.block if width is None else width

    def stored(kind: str, index: int, grad) -> None:
        del kind
        held.store(ref, grad, row, CHUNK, column + index * width, width, family=call.family, smem=smem)

    return stored


def prefix_reader(family: mosaic.Family, streamed: bool, refs: dict, head, span):
    """A reader of a chunk's prefix as a row's vectors `(gate, low)` and a column's `(column_gate,
    column_low)` (`refs`' views by name, `vector_views`'), from their streams' slots where `streamed`, else
    from global memory at `head` and `span`."""
    def read(slots):
        found = {name: slots[f"{name}0"] if streamed else ref.at[head, span] for name, ref in refs.items()}
        return ([family.rows_of(found[name]) for name in ("gate", "low")],
                tuple(family.columns_of(found[name]) for name in ("column_gate", "column_low")))

    return read


def vector_views(chunk, head, **refs) -> dict:
    """The streams of a chunk's vectors (`refs[name]`, a decay a head's `[HV, R]` prefix halves or write
    gates), each its own stream `{name}0` of the head's row at the chunk (`vector_blocks`), where the
    family's copies move a one-dimensional block (`Family.copies_vectors`), none where they do not."""
    return {f"{name}0": (ref.at[head], lambda i: (chunk,)) for name, ref in refs.items()}


def vector_blocks(call: Call, *names) -> dict:
    """The streamed blocks of `vector_views`' streams, a chunk's fp32 each, where the family copies them."""
    return {f"{name}0": ((CHUNK,), jnp.float32) for name in names} if call.family.copies_vectors else {}


def _down(step) -> tuple:
    """A view's block indices at `step`: its `step`-th row tile, from its first column."""
    return step, 0


def staging_tiles(operand: Operand, *shapes) -> tuple:
    """A product site's staging tiles, a part of `operand`'s form each, for each of `shapes`."""
    return tuple(tuple(plgpu.SMEM(shape, operand.stored) for _ in range(operand.parts)) for shape in shapes)


def _merges(call: Call) -> tuple:
    """The inverse's staging tiles (`algebra.doubled`'s 'merge' products, both operands fp32 registers): three
    sets, the system's kept across the levels and each level's inverse across its two products, each set a
    `[C, C]` tile an operand's part: an fp32 call's three bf16 parts, a 16-bit call's one fp16, 11 significant
    bits as the TF32 fla merges its blocks in, where the inverse is rounded to the call's 8 after. fla inverts
    its 16-row diagonal blocks by fp32 forward substitution and merges in TF32; every doubling level here
    rounds to fp16, about 2^-12 absolute on the inverse under aligned keys, a sixth of the bf16 rounding
    both arms share. fp16's range holds the inverse and each merge's product of it where the call's form
    proves its entries at most two (`recurrence.Hooks.inverse_bound`: unit keys, write gates at most two).
    Unnormalized keys with `beta |k|^2` past two grow it geometrically, past fp16's largest in about 16 tokens
    (row 2381): such a `ranged` call's merges take fp32's range in bf16 parts (`_merged`), two of them, where
    three spilled the solve on an A100 and an H100 and one put a normalized call's inverse 2.1 to 2.7 times as
    far from float64; two keep 16 bits, past the TF32 fla merges in."""
    merged = _merged(call)
    tile = (CHUNK, CHUNK)
    return tuple(tuple(plgpu.SMEM(tile, merged.stored) for _ in range(merged.parts)) for _ in range(3))


def _merged(call: Call) -> Operand:
    """The form the doubling's merges take the inverse and its power in (`_merges`): an fp32 call's; a
    `ranged` 16-bit call's fp32 in its fast forms' parts (`_fast_parts`), fp32's range at TF32's precision or
    past it; else one fp16 part."""
    if call.held.dtype == jnp.float32:
        return call.held
    if call.ranged:
        return Operand(call.rows, _fast_parts(call.family), jnp.dtype(jnp.float32))
    return Operand(call.rows, 1, jnp.dtype(jnp.float16))


def summing_staging(call: Call) -> tuple:
    """`algebra.summed_pairs`' staging: the fp32 operand's three bf16 parts its 'summed' product takes,
    staged, none where the family's products read their left operands from registers
    (`Family.operands_in_registers`, `summed_issuer`: `Issuer`'s register operand), the triangle's one part
    (`product.Exact`), and the scratch its columns' sums across warps take (`column_sums`). Staged on an A100,
    the parts' 24 KB put the scores' backward at keys 256 wide past its block (174080 bytes against 166912,
    wide-bwd at b3de8f293)."""
    staged = (() if call.family.operands_in_registers
              else staging_tiles(Operand(CHUNK, PARTS, jnp.dtype(jnp.float32)), (CHUNK, CHUNK)))
    return (*staged, (plgpu.SMEM((CHUNK, CHUNK), jnp.bfloat16),), column_sums_scratch(call))


def column_sums_scratch(call: Call) -> tuple:
    """The shared memory the family's column sums take (`Family.column_scratch` fp32 entries), none where
    its reductions take Mosaic GPU's implicit scratch."""
    entries = call.family.column_scratch
    return (plgpu.SMEM((entries,), jnp.float32),) if entries else ()


def column_sums(call: Call, reserved: tuple):
    """A `[rows, columns]` value's column sums in the family's reduction (`Family.column_sums`), through the
    shared memory `column_sums_scratch` reserved for them."""
    return lambda value: call.family.column_sums(value, reserved[0] if reserved else None, call.interpret)


def summed_issuer(staging: tuple, call: Call) -> tuple:
    """`algebra.summed_pairs`' issuer and column sums on `summing_staging`'s staging: its fp32 operand's parts
    staged, or held in registers where the family's products read them there."""
    *staged, exact, scratch = staging
    parts = PARTS if call.family.operands_in_registers else staged[0]
    return Issuer(call, {"summed": parts, "summed.exact": exact}), column_sums(call, scratch)


def _merging(staging, call: Call) -> Issuer:
    """The inverse's products on `_merges`' staging tiles."""
    return Issuer(call, {"merge": list(staging)})


def chunk_liveness(index: int):
    """A chunk's program's liveness (`launch_kernel`'s `live`): its chunk, the `index`-th input's entry
    (`taken`), holds tokens."""
    return lambda at, refs: refs[index][at["chunks"]] > 0


@dataclasses.dataclass(frozen=True)
class Held:
    """What a launch's consumer thread holds in registers, as its program states it: the tiles it carries
    across its steps (`carried`, each `(rows, columns, dtype)`) and the tiles each of its phases holds at once
    (`phases`, a tuple of such lists): its accumulators, an operand's parts, a product's pairs and their
    masks."""

    carried: Sequence = ()
    phases: Sequence = ()

    def registers(self, family: mosaic.Family, addressed=()) -> int:
        """A consumer thread's registers at the program's peak (`resources.live_registers`), beside its
        addresses into the shared tiles it stages and lands itself (`addressed`, `_addressed`)."""
        return resources.live_registers(family, self.carried, *self.phases, addressed=addressed)


def _addressed(staging: tuple, landing: dict) -> list:
    """The shared tiles a launch's consumer thread reads and writes through its registers, each `(shape,
    dtype)` a role's own (`resources.addresses`): its staging tiles and its stores' landings."""
    staged = [(tile.shape, tile.dtype) for tile in jax.tree.leaves(staging, is_leaf=lambda leaf: hasattr(
        leaf, "shape"))]
    return staged + list(landing.values())


def f32(rows: int, columns: int) -> tuple:
    """An fp32 tile `[rows, columns]` (`Held`'s)."""
    return rows, columns, jnp.float32


def held_tile(operand: Operand, rows: int, columns: int) -> list:
    """`operand`'s tile `[rows, columns]` as registers hold it: its parts, each in its stored dtype."""
    return [(rows, columns, operand.stored)] * operand.parts


def masks(rows: int, columns: int) -> list:
    """The int32 tile a tile's masks compare, its rows' lag behind its columns (`algebra._lags`); the
    inverse's doubling compares its rows and columns apart, two (`masks(...) * 2`)."""
    return [(rows, columns, jnp.int32)]


def _product(call: Call, lhs: tuple, rhs: tuple, *, rhs_transposed: bool = False,
             rhs_swizzle: int | None = None) -> list:
    """The operand fragments one product `[rows, depth] @ [depth, columns]` holds at once where the family's
    products read registers (`Family.operands_in_registers`), its left operand `lhs` `(operand, rows,
    depth)` a step deep and its right one `rhs` `(operand, depth, columns)` a sub-step
    (`Family.fragment_depth`), or where it is staged transposed (`rhs_transposed`: its depth its stage's
    columns, a key tile's in `Q Kᵀ`) a whole swizzle of its depth (`rhs_swizzle` bytes, the widest its rows
    fill where None), as its load takes it (`Family.rhs_fragment_depth`), each in the warps replicating it
    (`lhs_copies`, `rhs_copies`): the GEMM's own rule (`resources.operand_fragments`, each operand's parts);
    none where they read shared memory."""
    (left, rows, depth), (right, _, columns) = lhs, rhs
    return resources.operand_fragments(call.family, rows, depth, columns, left.stored, right.stored,
                                       lhs_parts=left.parts, rhs_parts=right.parts,
                                       rhs_transposed=rhs_transposed, rhs_swizzle=rhs_swizzle)


def _widest(*tiles: list) -> list:
    """The largest of `tiles`' live sets by their words: of products issued in turn, one's fragments."""
    return max(tiles, key=lambda held: sum(r * c * jnp.dtype(d).itemsize for r, c, d in held))


def fragments(call: Call, *tiles) -> list:
    """The operand tiles `(operand, rows, columns)` a product reads from registers where the family's
    products read them there (`Family.operands_in_registers`, mma.sync's), none where they read shared
    memory (wgmma's)."""
    if not call.family.operands_in_registers:
        return []
    return [tile for operand, rows, columns in tiles for tile in held_tile(operand, rows, columns)]


@dataclasses.dataclass(frozen=True)
class Launch:
    """A launch as planned (`template.planned`): its name (`recurrence_states`, ...), its programs
    and each one's threads (its bundle's warpgroups, `Family.warpgroups`), the shared memory a program
    allocates (its streams' slots and barriers and its staging tiles, as its bundle declares them), the
    registers a consumer thread holds at its program's peak (`resources.live_registers` of the tiles it
    states), its steps, the chunks in flight, the bytes a step streams in, a program's tensor FLOPs and
    exponentials, the bytes it reads and writes in global memory, the programs a multiprocessor holds of it at
    once (`resources.plan`; zero where it holds none, a config past the device), its consumer roles, and the
    bundle it issues by (`family`: the call's, or the layout a launch takes for itself, `_solving`), which
    every check of it reads."""

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
    consumers: int = 1
    family: mosaic.Family | None = None


PLANNED: contextvars.ContextVar[list | None] = contextvars.ContextVar("planned", default=None)
"""Where `template.planned` collects its launches; None where they run."""


STATIC = ("call", "tiles", "scale", "keeps", "applies", "out_dtype", "solves")
"""A launch's static arguments, by name (`lowered_once`): its call's geometry, bundle and target, its tiles
and its constants; its every other argument an array or None."""


lowered_once = tuning.lowered_once(STATIC, unless=lambda: PLANNED.get() is not None)
"""A launch lowered once a configuration (`tuning.lowered_once`): a launch's program closes over its call,
so each call traced its own and a hybrid model's step lowered its layers' identical kernels one by one.
Planning (`template.planned`) traces every call, each launch recording itself (`launch_kernel`)."""


def shared_allocation(call: Call, blocks: dict, staging: tuple, landing: dict, *, once: dict | None = None,
                      consumers: int = 1, swizzles: dict[str, int] | None = None, reduces: bool = True,
                      after: tuple = ()
                      ) -> tuple[Callable[[int], dict], Callable[[int], resources.Scope], tuple, dict]:
    """A launch's shared memory at a ring depth, as `launch_kernel` lays it: its walked streams' and its
    streams loaded once' allocations (each its own; a pipelined family's stages are its allocation's), which
    a program opens with their views by name ('walked', 'once'), the scope they make with its staging and
    landing tiles a role (`_roles_staged`, `_landed`), whose bytes `resources.shared_bytes` counts, and those
    staging tiles (its landing's last) and landing tiles: the one account a launch's legality and its plan
    both read."""
    family = call.family
    staged, landed = _roles_staged(staging, landing, consumers)
    staged = (*staged, _landed(landed, family))

    def allocations(depth: int) -> dict:
        opened = {"walked": family.streams(blocks, depth, consumers, swizzles=swizzles)}
        if once:
            opened["once"] = family.streams(once, 1, consumers)
        return opened

    def scope(depth: int) -> resources.Scope:
        return allocation_scope(family, allocations(depth), staged, reduces=reduces, after=after)

    return allocations, scope, staged, landed


def allocation_scope(family: mosaic.Family, streams: dict, staged: tuple, *, reduces: bool = True,
           after: tuple = ()) -> resources.Scope:
    """A program's allocations by lifetime (`resources.Scope`, whose bytes `resources.shared_bytes` counts as
    Mosaic GPU's resource estimate does): its `streams`' as the family states where they live
    (`Family.lifetimes`: the program's, or each walk's own), its `staged` tiles and barriers the program's,
    the tiles it scopes `after` its walks (`final_memory_landing`), and whether it `reduces` across warps:
    Mosaic GPU reserves the reduction scratch for a kernel with any reduction and none without (an A100's
    solve launch asked for its tiles' 90112 bytes alone, SE-family-a100-0c8018352)."""
    program, walks = list(jax.tree.leaves(staged)), []
    for allocated in streams.values():
        held, walked = family.lifetimes(allocated)
        program += jax.tree.leaves(held)
        walks += walked
    return resources.Scope(program=tuple(program), walks=tuple(walks), after=tuple(after), reduces=reduces)


def _inverse_flops() -> float:
    """The tensor FLOPs of a chunk's inverse (`algebra.doubled`): two whole-tile products a doubling level,
    from two-wide blocks to the chunk's (`log2(CHUNK) - 1` levels), each `2 C^3`, masked after they issue."""
    return 4 * CHUNK ** 3 * (int(math.log2(CHUNK)) - 1)


def issued_flops(call: Call, flops: float, left: Operand | None = None,
                 right: Operand | None = None) -> float:
    """`flops` of products of `left` by `right` (the model's `2 m k n` each) as their issuer takes them: once
    for each pair of the operands' parts it issues (`product.terms`), each operand the call's held one
    where None."""
    left, right = left or call.held, right or call.held
    return flops * len(product.terms(left.parts, right.parts))


def walk_stores(family: mosaic.Family, stores: int):
    """A walk step's `stores` stores, each through its own landing tile and left in flight (`wait=False`):
    `before()` ahead of each waits until at most `stores - 1` of the thread's stores still read their tiles,
    the oldest of them the same tile's in the step before, so a step's stores overlap the next step's work
    (the states walk waited out each, 9.5% of its samples, SE-sass-h100-be7214123). `sequence_walk` waits
    them out after its last step."""
    return lambda: family.store_wait(stores - 1)


def memory_landing(call: Call, bv: int) -> dict:
    """A walk's memory's landing tiles, one a key slice `[block, bv]` (`walk_stores`): `{memory0, ...}`."""
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


def _planned_launch(call: Call, scope: Callable[[int], resources.Scope], registers: int, *, floor: int,
                    most: int, consumers: int = 1) -> resources.LaunchPlan | None:
    """A launch's ring depth and the programs a multiprocessor holds of it (`resources.plan`, at its stated
    `registers` and its streams `scope`d at each depth from `floor` to `most`), None where it holds at none: a
    config past the device, whose record says so (`Launch.resident` zero) and `template.Mosaic.legal`
    refuses."""
    found = resources.plan([resources.Launch(scope, consumers, registers, floor=floor, most=most)],
                           call.family, call.target)
    return found[0] if found else None


def _laid(call: Call, scope: Callable[[int], resources.Scope], registers: int, tiles: int, *,
          stages: int | tuple[int, int] | None,
          consumers: int = 1) -> tuple[resources.LaunchPlan | None, int]:
    """A launch's plan and its programs: a persistent launch's (`stages` None) as many as a multiprocessor
    holds at once, at most its `tiles`, its rings up to two deep where they run on across its tiles (a
    family's whose walks do not scope their streams apart, `Family.walks_scoped`), at the plan that hides the
    most latency (`resources.hidden`), so a tile's loads go out under the last tile's work: a launch of
    one-step tiles whose program holds a multiprocessor's registers alone (each launch's 255 registers a
    thread an H100 holds one of, long-scoreboard stalls 51 to 73% of its time, SE-family-h100-80a38c02d; an
    A100's launches held one program each, a quarter of the warps fla's held, SE-family-a100-8c5b5b16a);
    any other a program a tile at its `stages`, or at the depth `resources.plan` takes between a `(floor,
    most)` pair's; None where no multiprocessor holds it (`_planned_launch`), a persistent launch then a
    program a multiprocessor."""
    if stages is None:
        most = 1 if call.family.walks_scoped else 2
        found = _planned_launch(call, scope, registers, floor=1, most=most)
        return found, min(tiles, (call.target.multiprocessors or 1) * (found.programs if found else 1))
    floor, most = stages if isinstance(stages, tuple) else (stages, stages)
    return _planned_launch(call, scope, registers, floor=floor, most=most, consumers=consumers), tiles


def _stated(work: Work, tiles: int, hardware: float, arrays, out_type) -> pl.CostEstimate:
    """A launch's work over its `tiles` as it states it to XLA and to Dew's FLOP count
    (`templates.cost.estimate`), the hardware's: its products as issued (`hardware` a tile, `issued_flops`:
    an fp32 operand's bf16 parts each a product), its exponentials, and its operands' and outputs' bytes
    once (row 1895's kernel report: unstated, the recurrence made MFU a lower bound). The model's share,
    `work.flops` a tile, the launch declares apart (`Family.kernel`'s `model_flops`)."""
    return estimate(hardware * tiles, *arrays, *jax.tree.leaves(out_type),
                    transcendentals=work.exponentials * tiles)


def _tile_at(index, grid: tuple, grid_names: tuple) -> dict:
    """`grid`'s indices by name of a persistent loop's tile `index` (its first axis the slowest)."""
    flat, indices = index[0], {}
    for axis, extent in reversed(tuple(zip(grid_names, grid, strict=True))):
        flat, indices[axis] = flat // extent, flat % extent
    return indices


def launch_kernel(program, call: Call, *, out_type, grid, grid_names, blocks: dict, stages: int | tuple, live,
                  work: Work, held: Held, staging: tuple = (), once: dict | None = None,
                  landing: dict | None = None, persistent: bool = False, reduces: bool = True,
                  after: tuple = (), consumers: int = 1, swizzles: dict[str, int] | None = None,
                  name: str | None = None, hardware: float | None = None):
    """`program(at, *refs, streams, staging, landing)` as a kernel over `grid`, `at` a tile's indices by the
    grid's axis names, run where `live(at, refs)` holds (a laid call's chunks and sequences are as many as its
    worst case, most of them empty): its streams' allocations (of `blocks`, `stages` deep, and of `once`'s
    blocks, loaded once, one deep; `Family.opened` takes them with their views), its staging tiles and its
    stores' (`landing`, `_landed`) scoped, named for its launch (`recurrence_solved`, ...; `name` where
    given). The program returns its walk (`mosaic.Walk`), which the kernel runs (`Transport.run`), or None
    where it runs its own roles (`sequence_walk`'s). `held` is what a consumer thread holds in registers
    (`Held`). Where `template.planned` collects launches, the launch's record (`Launch`, its
    `work`) and zeros. `work` states the model's products a tile, `hardware` their FLOPs as issued
    (`issued_flops`, the held operand's pairs where None), which its cost and Dew's FLOP count state.

    A `persistent` launch runs `grid`'s tiles on as many programs as the device's multiprocessors hold at
    once, its streams' rings running on across them (`_planned_launch`): the family's `persist` takes each
    tile's walk from its program at the tile's indices (`_tile_at`). A program that `reduces` none takes no
    reduction scratch (`_scope`; the compile gate holds its plan's shared memory to at least the compiled
    kernel's).
    `after` are the `(shape, dtype)` tiles its program scopes after its walks (`final_memory_landing`), which
    its plan counts beside them as its family scopes them. A program of `consumers` consumer roles (a walk's,
    `Tiles.walkers`) holds each staging and landing tile once a role, along a leading axis a role indexes
    (`role_tiles`): one role's products' tiles are no other's to write while its products read them."""
    family = call.family
    registers = held.registers(family, _addressed(staging, landing or {}))
    allocations, scope, staging, landing = shared_allocation(call, blocks, staging, landing or {}, once=once,
                                                             consumers=consumers, swizzles=swizzles,
                                                             reduces=reduces,
                                                             after=after)
    tiles = math.prod(grid)
    hardware = issued_flops(call, work.flops) if hardware is None else hardware
    threads = 128 * family.warpgroups(consumers)

    found, programs = _laid(call, scope, registers, tiles, stages=None if persistent else stages,
                            consumers=consumers)
    # Where no multiprocessor holds the launch, its record says so (`Launch.resident` zero), which the
    # template's legality refuses, and it builds at its shallowest depth, as a probe of the config compiles
    # it.
    shallowest = 1 if persistent else stages[0] if isinstance(stages, tuple) else stages
    stages = found.stages if found else shallowest
    streams = allocations(stages)
    name = name or f"recurrence{program.__qualname__.split('.')[0]}"
    collected = PLANNED.get()
    if collected is not None:
        def recorded(*arrays):
            outs = jax.tree.leaves(out_type)
            moved = sum(math.prod(x.shape) * jnp.dtype(x.dtype).itemsize for x in (*arrays, *outs))
            stepped = sum(math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in blocks.values())
            # A persistent program's work is its tiles' (`work` is a tile's), the most any program takes.
            each = -(-tiles // programs)
            flops = hardware * each
            collected.append(Launch(name, programs, threads, resources.shared_bytes(family, scope(stages)),
                                    registers, work.span * each, stages, stepped, flops,
                                    work.exponentials * each, moved, found.programs if found else 0,
                                    consumers, family))
            return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), out_type)

        return recorded

    def entry(*arrays):
        def scoped(staged, products, *, streams):
            with family.products_in(products):
                return run(staged, streams=streams)

        def run(staged, *, streams):
            *staged, lands = staged
            keys = _landing_keys(landing, family)
            lands = {name: lands[keys[name]] for name in landing}

            def given(at: dict) -> mosaic.Walk | None:
                """The tile's walk as its program gives it at its grid indices `at`, or None where the program
                runs its own roles (`sequence_walk`'s)."""
                return program(at, *arrays, streams=streams, staging=tuple(staged), landing=lands)

            if not persistent:
                walked = given({axis: lax.axis_index(axis) for axis in grid_names})
                if walked is not None:
                    family.run(streams["walked"], walked, call.interpret)
                return

            def tile(at: dict) -> mosaic.Walk:
                """The tile's walk, which a persistent launch's program returns."""
                walked = given(at)
                assert walked is not None, "a persistent launch's program returns its tile's walk"
                return walked

            def schedule(index):
                at = _tile_at(index, grid, grid_names)
                walked = tile(at)
                return live(at, arrays), walked.steps, walked.views, at

            def issue(_, at, walk, *, active):
                # A tile's steps store what they compute: nothing is left to finish.
                if active:
                    tile(at).persisted(walk)

            def finish(*_, **__):
                pass

            family.persist(streams["walked"], (tiles,), schedule, issue, finish, consumers=1, axis="programs",
                           outputs=(), interpret=call.interpret)

        # The products' own memory where the family's issuer scopes each (`Issuer.product_scratch`).
        products = family.product_scratch(consumers)
        if persistent:
            family.scoped(scoped, staging, products, streams=streams)
        else:
            # An empty program scopes nothing (Mosaic GPU's interpreter takes a scope's allocations in turn).
            at = {axis: lax.axis_index(axis) for axis in grid_names}
            pl.when(live(at, arrays))(lambda: family.scoped(scoped, staging, products, streams=streams))

    entry.__name__ = name

    def launched(*arrays):
        laid = ({"grid": (programs,), "grid_names": ("programs",)} if persistent
                else {"grid": grid, "grid_names": grid_names})
        # At its plan's residency (`resources.LaunchPlan.programs`), which its registers fit and its roles
        # raise to the share of (two query/key programs where one held, FS-WHY3-h100).
        return family.kernel(entry, out_type=out_type, **laid, consumers=consumers, interpret=call.interpret,
                             name=name, cost=_stated(work, tiles, hardware, arrays, out_type),
                             model_flops=int(work.flops * tiles),
                             resident=found.programs if found else 1)(*arrays)

    return launched


def _solving(call: Call) -> mosaic.Family:
    """The bundle the solve launch issues by: of the layouts the call's device takes (`bundles.layouts`), the
    first whose programs a multiprocessor holds most (`Family.programs`, a consumer role's), the call's own
    where none holds more. The solve's persistent programs walk a chunk at a time on their products'
    latency, one a multiprocessor beside a producer leaving its tensor cores idle: on an H100 PCIe the 27B
    gated delta net's took 0.235 ms on warpgroup MMA's layout without a producer (`mosaic_sm90.FED`, two
    programs a multiprocessor) against 0.356 beside a producer, Qwen3-Next's 0.156 against 0.240, where the
    states launch's walk took 0.187 beside a producer against 0.202 to 0.243 without one
    (JS-fed5-h100-d82eeb171). Interpreted, the interpreter's (`bundles.INTERPRETED`)."""
    if call.interpret:
        return call.family
    layouts = bundles.layouts(call.target.generation, bundle_dtype(call.given.dtype))
    return max((call.family, *layouts), key=lambda layout: layout.programs(1))


def _solved_resources(call: Call, *, keeps: bool, applies: bool, narrow: bool) -> tuple:
    """`_solved`'s work, held tiles, staging tiles and landings: those of its keys' products and the
    inverse's doubling, and where it `applies` the inverse, of its application to each key and value block
    (`narrow` where a value tile is narrower than a block, `Call.value_block`)."""
    width, value_width = call.width, call.value_width
    held = dataclasses.replace(call.held, rows=call.rows)
    # The keys' products and the inverse's doubling, then the inverse against each key and value block.
    work = Work(flops=2 * CHUNK * CHUNK * width + _inverse_flops()
                + 2 * CHUNK * CHUNK * (width + value_width), span=1)
    # As issued: the keys' products of the operands as given, the doubling's merges in their own form
    # (`_merged`: a ranged 16-bit call's fp32 parts), the inverse's application to the given keys and values.
    merged = _merged(call)
    hardware = (issued_flops(call, 2 * CHUNK * CHUNK * width, call.given, call.given)
                + issued_flops(call, _inverse_flops(), merged, merged)
                + issued_flops(call, 2 * CHUNK * CHUNK * (width + value_width), held, call.given))
    # The keys' products' sum and the system's pairs; then the doubling's inverse, power and product, beside
    # the masks (an fp32 call's merges split the inverse and the power into their parts); then the inverse
    # held beside a block's product, its decay and its keys or values.
    held_tiles = Held(phases=(
        [f32(CHUNK, CHUNK)] * 2 + fragments(call, (call.given, CHUNK, call.block), (call.given, CHUNK,
                                                                                         call.block)),
        _doubling_held(call),
        *([[f32(CHUNK, CHUNK)] + [f32(CHUNK, call.block)] * 2 + held_tile(call.given, CHUNK, call.block)
           + fragments(call, (held, CHUNK, CHUNK), (call.given, CHUNK, call.block))] if applies else [])))
    # The inverse's application stages in the doubling's merge tiles where they are its form (an fp32 call's
    # bf16 parts, a block a chunk wide), done by then: apart, an fp32 call's launch took 242864 bytes of
    # shared memory, past an H100's block (SE-family-h100-e0de17766).
    merging = _merges(call)
    reused = (held.parts > 1 and merging[0][0].dtype == held.stored and call.block == CHUNK)
    staging = merging if reused or not applies else (*merging, staging_tiles(held, (CHUNK, CHUNK))[0],
                                                     staging_tiles(held, (CHUNK, call.block))[0])
    if narrow:
        staging = (*staging, staging_tiles(held, (CHUNK, call.value_block))[0])
    landing = {"square": held.landing(CHUNK, CHUNK)} if keeps else {}
    if applies:
        landing["block"] = held.landing(CHUNK, call.block)
    if narrow:
        landing["values"] = held.landing(CHUNK, call.value_block)
    return work, hardware, held_tiles, staging, landing


@lowered_once
def _solved(k, v, hi, lo, beta, taken, call: Call, *, keeps: bool = True, applies: bool = True):
    """`(a [R, HV * C], w [R, HV * Dk], u [R, HV * Dv])` as `call.held` of each chunk and value head under a
    decay a head, fla's kkt_solve and recompute_w_u in one program (FlashQLA's first kernel): the keys'
    products (`algebra.keyed`) inverted by doubling on the chunk's tile (`algebra.doubled`, its merges in
    fp16 in a 16-bit call, `_merges`), stored less its unit diagonal (`a`, the backward's), and applied in
    registers to the decayed write-gated keys and the write-gated values (`algebra.weighted`), the inverse
    staged once for every block's product: one launch's tiles and one load of the keys, where two launches
    stored and loaded the inverse between them. Where it `keeps` no inverse (a forward no backward reads,
    `a` None), its store's landing tile goes: an A100's launch then holds two programs a multiprocessor
    (an inverse's 8 KB past half its shared memory kept it at one, SE-family-a100-0c8018352). Where it
    `applies` none, `(a,)` alone, which `_weighted` applies (`_solves_apart`)."""
    call = dataclasses.replace(call, family=_solving(call))
    width, value_width, family = call.width, call.value_width, call.family
    # The values a tile narrower than a block (`Call.value_block`): their own staging and landing.
    narrow = applies and call.value_block != call.block
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(at, k_ref, v_ref, hi_ref, lo_ref, beta_ref, column_hi_ref, column_lo_ref, taken_ref, *outputs,
                streams, staging, landing):
        a_ref, w_ref, u_ref = ((*outputs, None, None) if not applies else outputs if keeps
                               else (None, *outputs))
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        views = stream_views(keys=call.given.views(k_ref, row, CHUNK, h // call.group * width, width),
                             **({"values": call.given.views(v_ref, row, CHUNK, h * value_width, value_width)}
                                if applies else {}))
        vectors = {"gate": hi_ref, "low": lo_ref, "write": beta_ref, "column_gate": column_hi_ref,
                   "column_low": column_lo_ref}
        streamed = family.copies_vectors
        if streamed:
            # The chunk's vectors streamed with its tiles, loaded a tile ahead (`vector_views`).
            views |= vector_views(n, h, **vectors)
        # The doubling's merges, then the inverse's and a block's tiles where they are not the merges', then a
        # value tile's where it is narrower than a block (`Call.value_block`).
        merges, applied = staging[:3], list(staging[3:])
        values_tile = applied.pop() if narrow else None
        lower_tile, scaled_tile = applied if applied else merges[:2]

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
                return stream_tile(slots, name, call.given, columns=pl.ds(c * call.block, call.block))

            # The key blocks' products into one accumulator: an accumulator a block, an H100's solve at keys
            # 256 wide spilled 224 bytes a thread (wide-sm90 at 8d1145e66).
            product = Issuer(call).summed([("intra", key, key.T) for key in map(
                functools.partial(block, "keys"), range(width // call.block))])
            system = algebra.keyed(product, gate, low, betas, *columns)
            lower = algebra.solved(system, _merging(merges, call))
            if keeps:
                held.store(a_ref, lower, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            if not applies:
                return carry, accumulators
            # The inverse staged once (the first set) for every block's product, a block staged in the
            # second: `Issuer` keeps a staged value for the products after it that take it.
            sets = [lower_tile, scaled_tile]
            issue = Issuer(call, {"apply_u": sets if values_tile is None else [lower_tile, values_tile],
                                  "apply_w": sets})
            vb = call.value_block
            for c in range(value_width // vb):
                values = stream_tile(slots, "values", call.given, columns=pl.ds(c * vb, vb))
                u = algebra.weighted("apply_u", lower, values.value().astype(held.dtype), betas, issue)
                held.store(u_ref, u, row, CHUNK, h * value_width + c * vb, vb, family=family,
                           smem=landing["values" if narrow else "block"])
            decay = lax.broadcast_in_dim(algebra.raised(gate, low), (CHUNK, call.block), (0,))
            for c in range(width // call.block):
                w = algebra.weighted("apply_w", lower, block("keys", c).value().astype(held.dtype), betas,
                                     issue, decay)
                held.store(w_ref, w, row, CHUNK, h * width + c * call.block, call.block, family=family,
                           smem=landing["block"])
            return carry, accumulators

        return mosaic.Walk(views, step, prologue=None if streamed else lambda: read(None))

    out_type = (held.shape(call.heads * width), held.shape(call.heads * value_width)) if applies else ()
    if keeps:
        out_type = (held.shape(call.heads * CHUNK), *out_type)
    work, hardware, held_tiles, staging, landing = _solved_resources(call, keeps=keeps, applies=applies,
                                                                     narrow=narrow)
    solved = launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                           grid_names=("chunks", "heads"),
                           blocks=stream_blocks(keys=((CHUNK, width), call.given),
                                                **({"values": ((CHUNK, value_width),
                                                               call.given)} if applies else {}))
                           | vector_blocks(call, "gate", "low", "write", "column_gate", "column_low"),
                           stages=1, live=chunk_liveness(7), work=work, held=held_tiles, staging=staging,
                           landing=landing, hardware=hardware,
                           persistent=True, reduces=False)(k, v, hi, lo, beta, hi, lo, taken)
    return solved if keeps else (None, *solved)


def _solves_apart(call: Call) -> bool:
    """Whether a decay a head's solve leaves its inverse's application to `_weighted`: where the doubling's
    merges take parts (`_merged`). Beside the application's tiles the solve spilled 1168 bytes a thread in
    an fp32 call's two parts and 196 in a ranged bf16 call's on an H100 (fast32-sm90 at 09bfe984c,
    Qwen3.5-27B's gated delta net)."""
    return _merged(call).parts > 1


def whole_blocks(width: int, block: int) -> list[int]:
    """The widths a row of `width` steps through a whole number of `block`s at a time, widest first, each
    dividing it: a row's every block staged whole (192 columns of 64-column blocks take 192 or 64, never
    96, which cuts a 64-column swizzle)."""
    return [each for each in range(width, 0, -block) if width % each == 0 and each % block == 0]


def fitting_block(call: Call, held: Callable[[int], Held], staged: Callable[[int], tuple], widest: int,
                  unit: int) -> int:
    """A launch's block of a row it takes a block at a time: the widest whole number of `unit`s dividing
    `widest` (`whole_blocks`) whose register model (`held(block)`, `Held`, beside its addresses into its
    staging and landing tiles there, `staged(block)`) fits a consumer thread's registers
    (`Family.registers`), as the launch states them; `unit` where none does, which the launch then
    refuses."""
    family = call.family
    return next((block for block in whole_blocks(widest, unit)
                 if held(block).registers(family, _addressed(*staged(block))) <= family.registers(1)), unit)


def key_block(call: Call, held: Callable[[int], Held], staged: Callable[[int], tuple],
              streamed: Callable[[int], dict] | None = None) -> int:
    """A launch's key block under a decay a key dimension, whose prefix halves, decays and gated keys it
    holds a block at a time in fp32: from the widest of the family's depths for 16-bit operands
    (`Family.depths`) down to the narrowest columns a staged operand takes (`Family.narrowest_columns`), the
    widest that fits a thread's registers (`fitting_block`); where the launch walks its key blocks and
    streams a block at a time (`streamed(block)`, its walked streams there), the block a multiprocessor holds
    most programs of (`_planned_launch`, at its register model and its shared memory), the widest of those.
    Its walks wait on their products' latency: the intra launch's 248 registers a thread and 143 KB held an
    H100 at one program a multiprocessor, its schedulers without a warp to issue 64% of its cycles
    (kda-ncu2-h100-bc481d7de)."""
    family = call.family
    widest, unit = family.depths(jnp.bfloat16)[0], family.narrowest_columns(jnp.bfloat16)
    if streamed is None:
        return fitting_block(call, held, staged, widest, unit)
    resident = functools.partial(_resident, call, held, staged, streamed)
    return max(_key_blocks(call), key=lambda block: (resident(block), block))


def _key_blocks(call: Call) -> list[int]:
    """The key blocks a walk under a decay a key dimension may take (`key_block`), each dividing the keys."""
    family = call.family
    widest, unit = family.depths(jnp.bfloat16)[0], family.narrowest_columns(jnp.bfloat16)
    return [block for block in whole_blocks(widest, unit) if call.width % block == 0]


def _resident(call: Call, held: Callable[[int], Held], staged: Callable[[int], tuple],
              streamed: Callable[[int], dict], block: int) -> int:
    """The programs a multiprocessor holds of a walk at key block `block` (`_planned_launch`, at its register
    model and its shared memory), none past a thread's registers."""
    family = call.family
    staging, landing = staged(block)
    registers = held(block).registers(family, _addressed(staging, landing))
    if registers > family.registers(1):
        return 0
    _, scope, _, _ = shared_allocation(call, streamed(block), staging, landing)
    found = _planned_launch(call, scope, registers, floor=1, most=1)
    return found.programs if found else 0


def _intra_held(call: Call, block: int, outputs: int = 1) -> Held:
    """`_intra_walk`'s consumer thread at key block `block` forming `outputs` pairs' sums, which shared memory
    carries across the key blocks, one output's formed and added at a time: the block's prefix scan (its
    running pair, a shifted pair and the decay); then the factors every output shares beside the block's
    tiles: its prefix halves, the rows' gate down and the keys gated up (`algebra.key_block_pairs`), a pair's
    decay's two terms (`between`: the exp2 of the fractions' difference and the scale of the wholes'), a
    sub-chunk's middle factor and its laid references, each output's gated rows, their masks, and one
    product's operands, its right a key sub-chunk's."""
    pairing = _pairing_operand(call)
    if outputs == 1:
        # A lone output's sum carried in registers across the key blocks, as its launch held it before the
        # walks merged (M-JS-17's A100 parent), beside one gated output's factors.
        return Held(carried=[f32(CHUNK, CHUNK)], phases=(
            [f32(CHUNK, block)] * 5,
            [f32(CHUNK, block)] * 12 + masks(CHUNK, block) * 2
            + fragments(call, (pairing, CHUNK, block), (pairing, algebra.SUB, block))))
    # Live at once: the prefix halves and their separated pair, the rows' grid, the keys and the queries as
    # read, one output's rows and their gated form, the keys gated up, the gate down, a sub-chunk's middle
    # factor and the masked keys; the output's pairs and its sum read back. Its sums held in registers across
    # the key blocks, both outputs' at once, the merged launch spilled 336 to 424 bytes at 32 keys a block on
    # an A100 and an H100 (jsf-scan4 at d54ec6b02).
    return Held(carried=[], phases=(
        [f32(CHUNK, block)] * 5,
        [f32(CHUNK, block)] * 12 + [f32(CHUNK, CHUNK)] * 2 + masks(CHUNK, block) * 2
        + fragments(call, (pairing, CHUNK, block), (pairing, algebra.SUB, block))))


def _doubling_held(call: Call) -> list:
    """A chunk's inverse by doubling as a thread holds it (`algebra.solved`): the inverse, its power and their
    product beside the masks, merges in parts (`_merged`) splitting the inverse and the power into them; the
    system, staged negated for the merges, held no longer. Two parts (`FAST_PARTS`, one accumulator in place,
    `Issuer`) hold one operand's at a time: both at once stated 264 registers for a ranged bf16 call's solve
    apart, which ptxas held spill-free at 255 on an H100 and an A100 (fast32e at cdcd620ef); three parts,
    each product its own accumulator, both (as one, an fp32 call's solve spilled 1588 bytes a thread on an
    H100, gaps4 at 97e2852bf)."""
    merged = _merged(call)
    held = 1 if merged.parts == FAST_PARTS else 2
    return ([f32(CHUNK, CHUNK)] * 3 + masks(CHUNK, CHUNK) * 2
            + (held_tile(merged, CHUNK, CHUNK) * held if merged.parts > 1 else []))


def _systems(call: Call) -> Operand:
    """The form `_intra` passes each chunk's keys' system to `_intra_solve` in: fp32, as fla's intra rows
    store their `Akk` for its solve."""
    return Operand(call.rows, 1, jnp.dtype(jnp.float32))


@lowered_once
def _intra_solve(system, taken, call: Call):
    """`(I + A)^-1` less its unit diagonal `[R, HV * C]` as `call.held` of each chunk and value head's keys'
    system `A` (`_intra`'s, `_systems`' form), inverted by doubling on the chunk's tile (`algebra.doubled`):
    fla_kda's solve. Fused with the pairs, the doubling's tiles beside theirs held an H100's thread at 168
    registers and one program a multiprocessor, 12.5% of its warps' cycles issuing, the launch 0.944 ms of
    Kimi Linear's 1.86 ms forward (kda-ncu-h100-1170e4d25)."""
    held, systems, family = dataclasses.replace(call.held, rows=call.rows), _systems(call), call.family

    def program(at, a_ref, taken_ref, out_ref, *, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row = n * CHUNK
        views = stream_views(system=systems.views(a_ref, row, CHUNK, h * CHUNK, CHUNK))

        def step(index, slots, carry, accumulators):
            del index
            lower = algebra.solved(stream_tile(slots, "system", systems).value(),
                                   _merging(tuple(staging), call))
            held.store(out_ref, lower, row, CHUNK, h * CHUNK, CHUNK, family=family, smem=landing["square"])
            return carry, accumulators

        return mosaic.Walk(views, step)

    return launch_kernel(program, call, out_type=held.shape(call.heads * CHUNK),
                         grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"),
                         blocks=stream_blocks(system=((CHUNK, CHUNK), systems)),
                         stages=2, live=chunk_liveness(1), work=Work(flops=_inverse_flops(), span=1),
                         hardware=issued_flops(call, _inverse_flops(), *(_merged(call),) * 2),
                         held=Held(phases=(_doubling_held(call),)),
                         staging=_merges(call), landing={"square": held.landing(CHUNK, CHUNK)},
                         persistent=True)(
        system, taken)


@lowered_once
def _intra(q, k, g, beta, taken, call: Call, *, scale: float, solves: bool):
    """`(Aqk [R, HV * C], A [R, HV * C] where the call `solves`, hi, lo [R, HV * Dk])` of each chunk and value
    head under a decay a key dimension, one launch (`_intra_walk`): the scores as `call.held`, the keys'
    system where it `solves` (`_systems`' form, the write-gated keys' pairs, strict), which `_intra_solve`
    inverts, and the decay's prefix halves (`call.gates`, `algebra.prefix`'s pair) from the laid decay `g`
    `[R, HV * Dk]` fp32, which every later launch reads. Each pair's decay factored over the chunk's
    sub-chunks (`algebra.key_block_pairs`, which the call's declared floor bounds), their products on the
    tensor cores (fla_kda's intra rows); the scores and the system share every factor in one walk, their sums
    in shared memory, one output's formed at a time (`_intra_walk`), where `_intra_merges` (wgmma's); else
    each output walks a launch of its own, its sum in registers. Where the pairs do not factor
    (`Call.factored`), each sub-chunk's own pairs taking their own decays besides, each output is a launch of
    its own (no model calls one: every model's Kimi delta attention declares its floor)."""
    if solves and not (call.factored and _intra_merges(call)):
        scores, hi, lo = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("scores",))
        (system,) = _intra_walk(q, k, g, beta, taken, call, scale=scale, products=("system",))
        return scores, system, hi, lo
    return _intra_walk(q, k, g, beta, taken, call, scale=scale,
                       products=("scores", "system") if solves else ("scores",))


def _intra_merges(call: Call) -> bool:
    """Whether the scores and the system take one intra walk (`_intra_walk`): where the family's products
    read their operands from shared memory (wgmma's), at a key block at which a multiprocessor holds a
    program of it, both sums in shared memory (an sm86's 99 KB held none of a narrow key dimension's). Where
    they read registers (mma.sync's), each output walks a launch of its own: one walk's sums through shared
    memory beside the fragments took 1.736 ms of an A100's Kimi Linear forward against the two launches'
    0.762 and 0.717 (M-JS-17, W33-a100-1), the forward 2.754 ms against 2.487."""
    if call.family.operands_in_registers:
        return False
    resident = functools.partial(_resident, call, functools.partial(_intra_held, call, outputs=2),
                                 functools.partial(_intra_staged, call, products=("scores", "system")),
                                 functools.partial(_intra_streamed, call, queries=True))
    return any(resident(block) for block in _key_blocks(call))


def _shifted_keys(call: Call, tiles, keys: jax.Array, prefix: tuple):
    """A key block's keys and separated prefix staged in `tiles` below `SUB` rows of any values (a pair they
    reach is masked), each offset's reads its loads shifted by it (`algebra.Shifted`)."""
    mosaic.synced(call.interpret)
    for i, value in enumerate((keys, *algebra.separated(*prefix))):
        plgpu.store(tiles.at[i, pl.ds(algebra.SUB, CHUNK)], value, optimized=False)
    mosaic.synced(call.interpret)
    return lambda e: tuple(call.family.tile_of(tiles.at[i, pl.ds(algebra.SUB - e, CHUNK)])
                           for i in range(3))


def _intra_prefix(call: Call, scanned, decays: jax.Array, keys: int) -> tuple:
    """A key block's prefix halves (`algebra.scanned_prefix`) of its `decays` `[CHUNK, keys]`, staged in
    `scanned` (each half's `CHUNK` zero rows, then its running pair, which the shifts read), left there at
    rows `CHUNK` on and returned in the accumulator's layout. The scan's steps run in a row-major layout,
    each thread's columns consecutive, whose transfers at any row offset touch every bank once: in the
    accumulator's, a fragment's 8 rows `keys` words apart shared 8 banks (19.0M bank conflicts on loads in a
    GH200's Kimi Linear intra launch, M-JS-17); the decay laid in and the prefix out once each."""
    def half_at(half, rows):
        return scanned.at[half, rows]

    def stage(pair):
        # A thread reads other threads' rows: the last shift's reads done before the rows are written again,
        # and the writes made visible before they are read (`mosaic.synced`).
        mosaic.synced(call.interpret)
        for half, value in enumerate(pair):
            plgpu.store(half_at(half, pl.ds(CHUNK, CHUNK)), value, optimized=False)
        mosaic.synced(call.interpret)
        return lambda shift: tuple(_rows_of(half_at(half, pl.ds(CHUNK - shift, CHUNK)), keys)
                                   for half in (0, 1))

    mosaic.synced(call.interpret)
    for half in (0, 1):
        plgpu.store(half_at(half, pl.ds(0, CHUNK)), call.family.zeros((CHUNK, keys)), optimized=False)
    plgpu.store(half_at(0, pl.ds(CHUNK, CHUNK)), decays, optimized=False)
    mosaic.synced(call.interpret)
    stage(algebra.scanned_prefix(_rows_of(half_at(0, pl.ds(CHUNK, CHUNK)), keys), stage, chunk=CHUNK))
    return tuple(call.family.tile_of(half_at(half, pl.ds(CHUNK, CHUNK))) for half in (0, 1))


def _rows_of(ref, keys: int) -> jax.Array:
    """A `[CHUNK, keys]` tile of plain shared rows read in a row-major layout (`plgpu.Layout.WG_STRIDED`):
    each thread its consecutive words, a warp's transfer a contiguous run."""
    layout = plgpu.Layout.WG_STRIDED((CHUNK, keys), vec_size=min(4, CHUNK * keys // 128))
    return plgpu.load(ref, layout=layout, optimized=False)


def _intra_masked(found: jax.Array, *, strict: bool, scale: float, diagonal) -> jax.Array:
    """A key block's pairs `found` of an output as its sum takes them: below the diagonal where `strict` (the
    system), else on and below it and scaled (the scores); where the pairs do not factor, each sub-chunk's own
    pairs besides (`diagonal`, `algebra.key_diagonal_pairs` given its strictness)."""
    if diagonal is not None:
        found = found + diagonal(strict=strict)
    lag = algebra._lags((CHUNK, CHUNK))
    return jnp.where(lag > 0, found, 0.0) if strict else jnp.where(lag >= 0, found, 0.0) * scale


def _intra_summed(call: Call, found: jax.Array, at, *, first) -> None:
    """A key block's masked pairs `found` of an output (`_intra_masked`) added to its sum in shared memory
    `at` (written at the first block), before the next output's are formed (a merged walk's)."""
    mosaic.synced(call.interpret)

    # The first block's written, the rest added, in branches: a scalar condition laid over a tile
    # (`jnp.where`) is no broadcast Mosaic GPU's warpgroup semantics lower.
    @pl.when(first)
    def _():
        plgpu.store(at, found, optimized=False)

    @pl.when(jnp.logical_not(first))
    def _():
        plgpu.store(at, call.family.tile_of(at) + found, optimized=False)


def _intra_sums_stored(call: Call, products: tuple[str, ...], totals: tuple, sums, landing: dict, *, row,
                       head) -> None:
    """The intra walk's sums at its last step, each output's in its form: the scores as the held operand's,
    the system as `_systems`'."""
    held = dataclasses.replace(call.held, rows=call.rows)
    for name, total, ref in zip(products, totals, sums, strict=True):
        form, smem = ((held, landing["square"]) if name == "scores" else (_systems(call), landing["system"]))
        form.store(ref, total, row, CHUNK, head * CHUNK, CHUNK, family=call.family, smem=smem)


def _intra_walk(q, k, g, beta, taken, call: Call, *, scale: float, products: tuple[str, ...]):
    """The intra launch's walk of each chunk's key blocks (`key_block`, the block a multiprocessor holds most
    programs of): each step the block's prefix halves from its decay (`algebra.scanned_prefix`, staged in a
    padded shared tile whose rows the sub-chunks' references read), stored for the later launches, and the
    block's pairs summed into the sums the walk carries, stored at its last step: the scores (the queries'
    pairs, scaled) and, under the delta rule, the keys' system (the write-gated keys' pairs, strict), their
    factors formed once for both (`algebra.key_block_pairs`). Where the call's pairs do not factor
    (`Call.factored`), each sub-chunk's own pairs are each their own decay (`algebra.key_diagonal_pairs`).
    Its prefix an XLA scan of its own, thirteen launches took 0.30 ms of Kimi Linear's 1.57 ms forward on an
    H100; the scores and the system a launch each, each scanned the prefix and gated its keys again, 0.42
    and 0.40 ms of its 1.45 (kda-batch-h100-7e3a5fb48)."""
    width, gates, family = call.width, call.gates, call.family
    held = dataclasses.replace(call.held, rows=call.rows)
    # The scores' launch stores the prefix halves; a launch of the system alone reads no queries.
    scored = "scores" in products
    outputs = len(products)
    keys = key_block(call, functools.partial(_intra_held, call, outputs=outputs),
                     functools.partial(_intra_staged, call, products=products),
                     functools.partial(_intra_streamed, call, queries=scored))
    steps = width // keys

    def program(at, q_ref, k_ref, g_ref, beta_ref, taken_ref, *outs, streams, staging, landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        keyed = h // call.group * width
        views = stream_views(lambda i: (0, i), keys=call.given.views(k_ref, row, CHUNK, keyed, width),
                             decays=gates.views(g_ref, row, CHUNK, h * width, width),
                             **({"queries": call.given.views(q_ref, row, CHUNK, keyed, width)} if scored
                                else {}))
        pairing, scanned, *rest = staging
        summed_at, padded = (rest[0], rest[1:]) if outputs > 1 else (None, rest)
        sums, prefixes = outs[:outputs], outs[outputs:]

        def step(index, slots, totals, accumulators):
            slots.acquire_all()
            prefix = _intra_prefix(call, scanned, stream_tile(slots, "decays", gates).value(), keys)
            for half, ref in zip(prefix if scored else (), prefixes, strict=True):
                gates.store(ref, half, row, CHUNK, h * width + index * keys, keys, family=family,
                            smem=landing["prefix"])

            def at_token(token):
                """The prefix halves at the chunk's `token`, a column's vectors, from the staged scan."""
                return tuple(family.columns_of(scanned.at[half, CHUNK + token]) for half in (0, 1))

            def operand(name):
                return stream_tile(slots, name, call.given).value().astype(jnp.float32)

            def shifted(c):
                del c
                return _shifted_keys(call, padded[0], operand("keys"), prefix)

            # The scores' rows the queries; the system's each key scaled by its write strength, its pairs
            # strict; each formed as its products are.
            def rows(name):
                if name == "scores":
                    return operand("queries")
                return operand("keys") * family.rows_of(beta_ref.at[h, span])[:, None]

            keyed, *gated = pairing
            pairs = algebra.key_block_pairs([functools.partial(rows, name) for name in products],
                                            operand("keys"), prefix, at_token,
                                            Issuer(call, {"pairs": gated}).summed,
                                            lambda up: staged_parts(family, up, keyed), chunk=CHUNK,
                                            diagonal=call.factored)
            masked = []
            for name, found in zip(products, pairs, strict=True):
                diagonal = None if call.factored else functools.partial(
                    algebra.key_diagonal_pairs, lambda c, name=name: rows(name), shifted, 1, lambda c: prefix,
                    chunk=CHUNK)
                masked.append(_intra_masked(found, strict=name == "system", scale=scale, diagonal=diagonal))
                if summed_at is not None:
                    _intra_summed(call, masked[-1], summed_at.at[len(masked) - 1], first=index == 0)
            if summed_at is None:
                totals = (totals[0] + masked[0],)

            @pl.when(index == steps - 1)
            def _():
                if summed_at is not None:
                    mosaic.synced(call.interpret)
                found = (totals if summed_at is None
                         else tuple(family.tile_of(summed_at.at[o]) for o in range(outputs)))
                _intra_sums_stored(call, products, found, sums, landing, row=row, head=h)

            return totals, accumulators

        carry = () if outputs > 1 else (family.as_tile(family.zeros((CHUNK, CHUNK))),)
        return mosaic.Walk(views, step, steps=steps, carry=carry)

    blocks = _intra_streamed(call, keys, queries=scored)
    formed = [held if name == "scores" else _systems(call) for name in products]
    out_type = (*(form.shape(call.heads * CHUNK) for form in formed),
                *((gates.shape(call.heads * width),) * 2 if scored else ()))
    # Each output's product a key sub-chunk's columns; the exponentials a key block's rows gated down and its
    # keys up.
    work = Work(flops=outputs * 2 * CHUNK * CHUNK * width, span=steps,
                exponentials=2 * CHUNK * width)
    staging, landing = _intra_staged(call, keys, products=products)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=(1, 2),
                         live=chunk_liveness(4), work=work,
                         hardware=issued_flops(call, work.flops, *(_pairing_operand(call),) * 2),
                         held=_intra_held(call, keys, outputs), staging=staging, landing=landing,
                         name="recurrence_intra" if scored else "recurrence_intra_system",
                         # Each stream staged at its key block's row: a tile slices only whole swizzle rows.
                         swizzles={name: resources.widest_swizzle(dtype, keys)
                                   for name, (_, dtype) in blocks.items()})(q, k, g, beta, taken)


def _intra_streamed(call: Call, keys: int, *, queries: bool) -> dict:
    """`_intra_walk`'s walked streams at key block `keys`: the keys and the decay a block at a time, and the
    queries where it forms the scores."""
    return stream_blocks(keys=((CHUNK, keys), call.given), decays=((CHUNK, keys), call.gates),
                         **({"queries": ((CHUNK, keys), call.given)} if queries else {}))


def _intra_staged(call: Call, keys: int, *, products: tuple[str, ...]) -> tuple:
    """`_intra_walk`'s staging tiles and landings at key block `keys` forming `products`: the pairs' operands
    (`_pairing`), the prefix scan's halves below as many zero rows (`algebra.scanned_prefix`'s shifts), each
    output's sum across the key blocks, and the stores' landings: the
    scores' square and a block of the prefix halves where it forms the scores, the system's where it forms the
    system."""
    landing = {**({"square": dataclasses.replace(call.held, rows=call.rows).landing(CHUNK, CHUNK),
                   "prefix": call.gates.landing(CHUNK, keys)} if "scores" in products else {}),
               **({"system": _systems(call).landing(CHUNK, CHUNK)} if "system" in products else {})}
    # Where the call's pairs do not factor (`Call.factored`), a key block's keys and prefix halves below a
    # sub-chunk's rows (`algebra.key_diagonal_pairs`' shifted reads).
    # Both read at row offsets no tiling divides (the scan's shifts, the sub-chunks' references), so plain.
    # Unpadded: a tile read in the accumulator's layout takes a row stride a whole number of its tiling's
    # columns (Mosaic GPU's layout inference refused 24 at a 16-column tiling and 80 at a 64-column one,
    # jsf-kdaintra5 and 9), a multiple of 32 words at 32 columns or more, so no stride it takes moves a
    # fragment store's rows off one another's banks.
    padded = () if call.factored else (mosaic.plain((3, algebra.SUB + CHUNK, keys), jnp.float32),)
    scanned = mosaic.plain((2, 2 * CHUNK, keys), jnp.float32)
    # A merged walk's sums in shared memory, each read and written whole, so in swizzled 8-row tiles: plain,
    # rows a multiple of 32 words apart put a fragment's 8 rows in the same banks (41.3M bank conflicts in a
    # GH200's Kimi Linear intra launch at ceb64ccc5d, M-JS-17); a lone output's in registers (`_intra_held`).
    sums = ((mosaic.staged((len(products), CHUNK, CHUNK), jnp.float32, 128),) if len(products) > 1
            else ())
    return (_pairing(call, keys), scanned, *sums, *padded), landing


def _pairing_operand(call: Call) -> Operand:
    """The form the factored pairs' fp32 operands take to their products (`algebra.key_block_pairs`): in a
    16-bit call, fp32 itself where the family's products take it as TF32 (`Family.takes_tf32`, warpgroup
    MMA's), one product a pair as fla's sub-chunk products take, else two bf16 parts (16 of their bits,
    past TF32's); an fp32 call's three bf16 parts."""
    if call.held.parts > 1:
        return Operand(call.rows, PARTS, jnp.dtype(jnp.float32))
    return Operand(call.rows, 1 if call.family.takes_tf32 else 2, jnp.dtype(jnp.float32))


def staged_parts(family: mosaic.Family, value: jax.Array, buffers) -> Parts:
    """`value` staged once in shared memory as its parts in `buffers` (`product.formed`, itself in one), a
    block of whose rows a product takes (`Parts.rows`, `algebra.Staged`)."""
    parts = product.formed(value, len(buffers)) if len(buffers) > 1 else (value,)
    return Parts(tuple(family.staged(part, buffer) for part, buffer in zip(parts, buffers, strict=True)),
                 jnp.float32)


def _pairing(call: Call, block: int) -> list:
    """The factored pairs' staging: the keys gated up, staged once for every output's products
    (`algebra.key_block_pairs`' `staged`), then the gated rows' two sets (`Issuer`'s 'pairs', one formed
    while the other's product reads it), each a `[C, block]` tile a part (`_pairing_operand`)."""
    return [staging_tiles(_pairing_operand(call), (CHUNK, block))[0] for _ in range(3)]


def _weighted_held(call: Call, block: int) -> Held:
    """`_weighted`'s consumer thread at key block `block`: the inverse beside a value block's product, its
    decay and its keys or values; under a decay a key dimension the key block's prefix halves, its range
    decay's two terms (`algebra.range_decay`, as `between`'s) and its decayed write too."""
    held = dataclasses.replace(call.held, rows=call.rows)
    return Held(phases=(
        held_tile(held, CHUNK, CHUNK) + [f32(CHUNK, call.block)] * 2
        + held_tile(call.given, CHUNK, call.block) + [f32(CHUNK, block)] * 5 * call.per_key
        + fragments(call, (held, CHUNK, CHUNK), (call.given, CHUNK, call.block)),))


@lowered_once
def _weighted(k, v, a, hi, lo, beta, taken, call: Call):
    """`(w [R, HV * Dk], u [R, HV * Dv])` as `call.held` (`algebra.weighted`): the inverse against the decayed
    write-gated keys and against the write-gated values; under a decay a key dimension the keys alone,
    `(w, the keys decayed to their chunk's end [R, HV * Dk])` as `call.held` (`algebra.key_decayed_write`,
    which the memory takes), the values' their own launch (`_weighted_values`), and the decayed keys alone
    without the delta rule (`a` None). Formed beside the key walk at its first step, the values' products
    held an H100 thread at 248 registers and one program a multiprocessor (kda-ncu-h100-1170e4d25)."""
    width, value_width, gates, family = call.width, call.value_width, call.gates, call.family
    held = dataclasses.replace(call.held, rows=call.rows)
    # Under a decay a key dimension a walk of the key blocks, the block a multiprocessor holds most programs
    # of (`key_block`): streaming every block at once, the launch held one program at 132 KB on an H100 with
    # 104 registers' need (kda-kernels2 at bc481d7de).
    keyed = (key_block(call, functools.partial(_weighted_held, call),
                       functools.partial(_weighted_staged, call),
                       functools.partial(_weighted_streamed, call)) if call.per_key else call.block)
    steps = width // keyed if call.per_key else 1

    def program(at, k_ref, v_ref, a_ref, hi_ref, lo_ref, beta_ref, taken_ref, *outs, streams, staging,
                landing):
        n, h = at["chunks"], at["heads"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        given = {"keys": call.given.views(k_ref, row, CHUNK, h // call.group * width, width)}
        if call.per_key:
            given.update(gates=gates.views(hi_ref, row, CHUNK, h * width, width),
                         lows=gates.views(lo_ref, row, CHUNK, h * width, width))
        whole = {}
        if call.delta:
            whole.update(inverse=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK))
        if call.delta and not call.per_key:
            whole.update(values=call.given.views(v_ref, row, CHUNK, h * value_width, value_width))
        # A key block a step under a decay a key dimension, the inverse and the values whole.
        views = {**stream_views(lambda i: (0, i), **given), **stream_views(lambda i: (0, 0), **whole)}
        vectors = {**({"write": beta_ref} if call.delta else {}),
                   **({} if call.per_key else {"gate": hi_ref, "low": lo_ref})}
        streamed = family.copies_vectors
        if streamed:
            # The chunk's vectors streamed with its tiles, loaded a tile ahead (`vector_views`).
            views |= vector_views(n, h, **vectors)

        def read(refs):
            """The write gates and a decay a head's prefix, a row's vectors, from their streams' slots or from
            global memory."""
            rows = {name: family.rows_of(refs[f"{name}0"] if streamed else ref.at[h, span])
                    for name, ref in vectors.items()}
            return rows.get("write"), None if call.per_key else (rows["gate"], rows["low"])

        def step(index, slots, carry, accumulators, loaded=None):
            # Every slot waited for each step, the values' too, which the first step alone reads: a load no
            # step waits for leaves its barrier's phases unobserved (the interpreter refuses it).
            slots.acquire_all()
            found = read(slots) if streamed else loaded
            assert found is not None, "a family that streams no vectors loads them in the prologue"
            betas, prefix = found
            # The inverse, the write gates and the issuer under the delta rule, which a decay a head's call
            # of this launch always takes (`prepared`).
            applied, decay = None, None
            if call.delta:
                # A key block's own staging where it is narrower than a value block's.
                issue = Issuer(call, {"apply_u": staging[0], "apply_w": staging[-1]})
                inverse = stream_tile(slots, "inverse", held)
                applied = inverse, betas, issue

                # The values' weights once, a value tile at a time (`Call.value_block`), under a decay a
                # head (a key dimension's are their own launch's, `_weighted_values`).
                for c in range(0 if call.per_key else value_width // call.value_block):
                    vb = call.value_block
                    tile = stream_tile(slots, "values", call.given, columns=pl.ds(c * vb, vb)).value()
                    u = algebra.weighted("apply_u", inverse, tile.astype(held.dtype), betas, issue)
                    held.store(outs[1], u, row, CHUNK, h * value_width + c * vb, vb, family=family,
                               smem=landing["values"])
            if prefix is not None:
                decay = lax.broadcast_in_dim(algebra.raised(*prefix), (CHUNK, keyed), (0,))
            # Under a decay a key dimension the step's own key block (its streams' slots whole), else each.
            for c, block in ([(index, None)] if call.per_key
                             else [(c, pl.ds(c * keyed, keyed)) for c in range(width // keyed)]):
                tile = stream_tile(slots, "keys", call.given, columns=block).value()
                if call.per_key:
                    gate, low = (stream_tile(slots, name, gates).value() for name in ("gates", "lows"))
                    decay = algebra.raised(gate, low)
                    ends = pl.ds(h * width + c * keyed, keyed)
                    last, last_low = (family.columns_of(ref.at[row + CHUNK - 1, ends])
                                      for ref in (hi_ref, lo_ref))
                    written = algebra.key_decayed_write(tile, gate, low, last, last_low)
                    held.store(outs[-1], written, row, CHUNK, h * width + c * keyed, keyed,
                               family=family, smem=landing["keyed"])
                if applied is not None:
                    inverse, betas, issue = applied
                    w = algebra.weighted("apply_w", inverse, tile.astype(held.dtype), betas, issue, decay)
                    held.store(outs[0], w, row, CHUNK, h * width + c * keyed, keyed, family=family,
                               smem=landing["keyed" if call.per_key else "block"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=steps, prologue=None if streamed else lambda: read(None))

    given = _weighted_blocks(call, keyed)
    values = 0 if call.per_key else value_width
    out_type = ((held.shape(call.heads * width),) * call.delta
                + (held.shape(call.heads * value_width),) * (call.delta and not call.per_key)
                + (held.shape(call.heads * width),) * call.per_key)
    # Without the delta rule the inverse's place takes the keys, an operand no step reads.
    work = Work(flops=2 * CHUNK * CHUNK * (width + values) if call.delta else 0, span=1,
                exponentials=CHUNK * width if call.per_key else 0)
    staging, landing = _weighted_staged(call, keyed)
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=_weighted_streamed(call, keyed),
                         stages=(1, 2) if call.per_key else 1, live=chunk_liveness(6),
                         work=dataclasses.replace(work, span=steps), held=_weighted_held(call, keyed),
                         # The inverse applied to the given keys and values.
                         hardware=issued_flops(call, work.flops, held, call.given),
                         staging=staging, landing=landing,
                         # A key block's walk a program a tile; a decay a head's one step a tile, persistent.
                         persistent=not call.per_key,
                         # A key block's streams staged at its row: a tile slices only whole swizzle rows.
                         swizzles={f"{name}{p}": resources.widest_swizzle(operand.stored, keyed)
                                   for name, (_, operand) in given.items()
                                   if name in ("keys", "gates", "lows") for p in range(operand.parts)})(
        k, v, k if a is None else a, hi, lo, beta, taken)


def _weighted_blocks(call: Call, keyed: int) -> dict:
    """`_weighted`'s streams' tiles by name, `(shape, operand)`: the keys and a decay a key dimension's prefix
    halves a key block (`keyed`) wide where it walks them (the keys whole under a decay a head), the inverse
    and the values whole."""
    width = keyed if call.per_key else call.width
    given = {"keys": ((CHUNK, width), call.given)}
    if call.per_key:
        given.update(gates=((CHUNK, width), call.gates), lows=((CHUNK, width), call.gates))
    if call.delta:
        given.update(inverse=((CHUNK, CHUNK), dataclasses.replace(call.held, rows=call.rows)))
    if call.delta and not call.per_key:
        given.update(values=((CHUNK, call.value_width), call.given))
    return given


def _weighted_streamed(call: Call, keyed: int) -> dict:
    """`_weighted`'s walked streams at key block `keyed` (`_weighted_blocks`), its vectors' beside."""
    return stream_blocks(**_weighted_blocks(call, keyed)) | vector_blocks(
        call, *(("write",) if call.delta else ()), *(() if call.per_key else ("gate", "low")))


def _weighted_staged(call: Call, keyed: int) -> tuple:
    """`_weighted`'s staging tiles and landings at key block `keyed`: its products' staged operands where
    the delta rule takes them, a value tile's (`Call.value_block`) and a key block's, one set where they are
    as wide, and their landings: a value tile's, a block's and a key block's own only where it is one."""
    held = dataclasses.replace(call.held, rows=call.rows)
    values = () if call.per_key else ((CHUNK, call.value_block),)
    widths = (*values, (CHUNK, keyed if call.per_key else call.block))
    staging = staging_tiles(held, *dict.fromkeys(widths)) if call.delta else ()
    return staging, {"block": held.landing(CHUNK, call.block),
                     **({"values": held.landing(CHUNK, call.value_block)}
                        if call.delta and not call.per_key else {}),
                     **({"keyed": held.landing(CHUNK, keyed)} if call.per_key else {})}


def _weighted_values_held(call: Call) -> Held:
    """`_weighted_values`' consumer thread: the inverse, a value tile, its write-gated fp32 copy beside the
    product's accumulator (`algebra.weighted`, as `_weighted_held` states a value block's), the product's
    fragments. Its gated copy unstated, an H100's ptxas asked 58 registers of the 48 its residency left
    (jsf-kdaintra6)."""
    held, vb = dataclasses.replace(call.held, rows=call.rows), call.value_block
    products = held_tile(held, CHUNK, CHUNK) + held_tile(call.given, CHUNK, vb) + [f32(CHUNK, vb)] * 2
    return Held(phases=(products + fragments(call, (held, CHUNK, CHUNK), (call.given, CHUNK, vb)),))


@lowered_once
def _weighted_values(v, a, beta, taken, call: Call):
    """`u [R, HV * Dv]` as `call.held` under a decay a key dimension (`algebra.weighted`): the inverse
    against the write-gated values, a program a chunk, head and value tile (`Call.value_block`), with no key
    walk: beside the key walk (`_weighted`) its products held an H100 thread at 248 registers and one
    program a multiprocessor (kda-ncu-h100-1170e4d25), alone 6 a multiprocessor's worth of the walk's
    tiles."""
    vb, family = call.value_block, call.family
    held = dataclasses.replace(call.held, rows=call.rows)

    def program(at, v_ref, a_ref, beta_ref, taken_ref, u_ref, *, streams, staging, landing):
        del taken_ref
        n, h, c = at["chunks"], at["heads"], at["values"]
        row, span = n * CHUNK, pl.ds(n * CHUNK, CHUNK)
        views = stream_views(lambda i: (0, 0), inverse=held.views(a_ref, row, CHUNK, h * CHUNK, CHUNK),
                             values=call.given.views(v_ref, row, CHUNK, h * call.value_width + c * vb, vb))
        streamed = family.copies_vectors
        if streamed:
            views |= vector_views(n, h, write=beta_ref)

        def step(index, slots, carry, accumulators):
            del index
            slots.acquire_all()
            betas = family.rows_of(slots["write0"] if streamed else beta_ref.at[h, span])
            issue = Issuer(call, {"apply_u": staging[0]})
            tile = stream_tile(slots, "values", call.given).value()
            u = algebra.weighted("apply_u", stream_tile(slots, "inverse", held), tile.astype(held.dtype),
                                 betas, issue)
            held.store(u_ref, u, row, CHUNK, h * call.value_width + c * vb, vb, family=family,
                       smem=landing["values"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=1)

    blocks = stream_blocks(inverse=((CHUNK, CHUNK), held), values=((CHUNK, vb), call.given)) | vector_blocks(
        call, "write")
    work = Work(flops=2 * CHUNK * CHUNK * vb, span=1)
    return launch_kernel(program, call, out_type=held.shape(call.heads * call.value_width),
                         grid=(call.chunks, call.heads, call.value_width // vb),
                         grid_names=("chunks", "heads", "values"), blocks=blocks, stages=1,
                         live=chunk_liveness(3), work=work,
                         held=_weighted_values_held(call),
                         hardware=issued_flops(call, work.flops, held, call.given),
                         staging=staging_tiles(held, (CHUNK, vb)),
                         landing={"values": held.landing(CHUNK, vb)})(v, a, beta, taken)


def sequence_walk(call: Call, memory: Operand, name: str, entering: tuple, walked: tuple, step, start,
                  count, kept, written, out, bv: int, *, landing, prefetch: Callable | None = None,
                  walkers: int = 1) -> None:
    """A sequence's memory over `walkers` value blocks `bv` wide, one a consumer role beside one producer
    (`Tiles.walkers`): the memory each role takes, its columns of each key slice's tile `{name}{i}` of
    `memory` (`entering`, the opened streams and their views), zero where `kept` is 0; the role's
    `step(role)` over its chunks `[start, start + count)` of `walked` (the same, its value streams the roles'
    columns side by side); and the memory it gives written to each slice's view `out(role, i)` where
    `written` is nonzero, through `landing(role)` (a slice's shared tile, `_landed`, or where None one scoped
    after the walk, `final_memory_landing`) (`_states`', and `_reverse`'s from its last chunk). Where
    `prefetch(n)` is given, it loads chunk `n`'s own values from global memory (its prefix), a step ahead:
    each step issues the next chunk's loads before its own work and takes the ones the step before issued
    (`step(role)(n, slots, state, accumulators, loaded)`), so their latency is the step's, not on its path."""
    family, slices = call.family, call.width // call.block
    (entered, entered_views), (chunks, views) = entering, walked

    def consumer(wg):
        # Each role's columns and tiles at static offsets (`mosaic.per_role`).
        mosaic.per_role(wg, walkers, walk)

    def walk(role, wg):
        walked_step = role_step = step(role)
        if (fetch := prefetch) is not None:
            last = start + count - 1

            def walked_step(n, slots, carry, accumulators):
                state, loaded = carry
                ahead = fetch(jnp.minimum(n + 1, last))
                state, accumulators = role_step(n, slots, state, accumulators, loaded)
                return (state, ahead), accumulators

        def taken(index, slots, carry, accumulators):
            del index, carry
            # A finite memory times zero where the sequence takes none (a select takes no scalar condition
            # over a tile in Mosaic GPU's warpgroup lowering).
            scale = jnp.where(kept > 0, 1.0, 0.0)
            return [stream_tile(slots, f"{name}{i}", memory, columns=pl.ds(role * bv, bv)).value() * scale
                    for i in range(slices)], accumulators

        # The streams a producer loads, or on a layout without one the role's own feed of them.
        state = family.walk(family.fed(entered, entered_views, 0, 1, wg), ((taken, 0, 1),),
                            [family.zeros((call.block, bv))] * slices, (), call.interpret)[0]
        carry = state if prefetch is None else (state, prefetch(start))
        carry = family.walk(family.fed(chunks, views, start, start + count, wg),
                            ((walked_step, start, start + count),), carry, (), call.interpret,
                            first=start)[0]
        state = carry if prefetch is None else carry[0]
        # The steps' stores left in flight (`walk_stores`) done reading their tiles before the program ends.
        family.store_wait(0)

        @pl.when(written > 0)
        def _():
            def store(smem):
                family.stored(list(state), [out(role, i) for i in range(slices)], smem=smem)

            if landing(role) is None:
                pl.run_scoped(store, plgpu.SMEM((call.block, bv), jnp.float32))
            else:
                store(landing(role))

    def producer():
        family.stream(entered, entered_views, 0, 1)
        family.stream(chunks, views, start, start + count)

    family.roles(walkers, consumer, producer)


def _roles_staged(staging: tuple, landing: dict, consumers: int) -> tuple[tuple, dict]:
    """`launch_kernel`'s staging and landing tiles for `consumers` roles: each once a role along a leading
    axis a role indexes (`role_tiles`) where more than one role shares the program."""
    if consumers == 1:
        return staging, landing
    return (jax.tree.map(lambda tile: plgpu.SMEM((consumers, *tile.shape), tile.dtype), staging),
            {name: ((consumers, *shape), dtype) for name, (shape, dtype) in landing.items()})


def role_tiles(staging, landing: dict, walkers: int):
    """A walk program's role's staging and landing tiles: `launch_kernel`'s, each along a leading axis of one
    tile a role where `walkers` roles share the program, the role's own (`ref.at[role]`)."""
    def of(role):
        if walkers == 1:
            return staging, landing
        return (jax.tree.map(lambda ref: ref.at[role], staging),
                {name: None if ref is None else ref.at[role] for name, ref in landing.items()})

    return of


def final_memory_landing(call: Call, bv: int) -> tuple[dict, tuple]:
    """`sequence_walk`'s final memory's landing, a key slice `[block, bv]` fp32: `(landing, after)`, the
    launch's own tile (`landing['final']`) where its family's walks share its scope, else one the walk
    scopes after it (`after`, `launch_kernel`'s; `Family.walks_scoped`), which takes no more than the walk's
    tiles: an A100's states launch then holds two stages at two programs a multiprocessor."""
    final = ((call.block, bv), jnp.float32)
    return ({}, (final,)) if call.family.walks_scoped else ({"final": final}, ())


def _prefetched(family: mosaic.Family, prefix: tuple, beta_ref, head, n) -> tuple:
    """`_states`' chunk `n`'s prefix under a decay a head: its rows' halves and its last's, a row's vectors
    (Mosaic GPU's warpgroup lowering bitcasts no scalar, which `algebra.raised` does), and its write gates
    where `beta_ref` is given (without the delta rule)."""
    span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
    ends = tuple(jnp.full((CHUNK,), ref[head, last]) for ref in prefix)
    rows = tuple(family.rows_of(ref.at[head, span]) for ref in prefix)
    return ends, rows, None if beta_ref is None else family.rows_of(beta_ref.at[head, span])


def _keys_gated_write(call: Call, slots, keys: Operand, values: Operand, loaded, state, issue, role: int,
                      bv: int) -> list:
    """`_states`' step's memory where the write's gate takes the keys' side (`Call.gated_keys`): each key
    row gated by its write gate and decay to the chunk's end, the values as they are
    (`algebra.keys_gated_carried`), from `loaded` (`_states`' prefetch, a step ahead)."""
    assert loaded is not None, "a decay a head's walk prefetches its prefix"
    (ending, ending_low), (gate, low), betas = loaded
    ends = algebra.range_decay(ending, ending_low, gate, low) * betas
    return algebra.keys_gated_carried(
        state, lambda i: stream_tile(slots, "keys", keys, columns=pl.ds(i * call.block, call.block)),
        stream_tile(slots, "values", values, columns=pl.ds(role * bv, bv)), ends,
        [algebra.raised(ending, ending_low)[:, None]] * (call.width // call.block), call.held.dtype, issue)


def _gated_write(call: Call, corrected, loaded, prefix: tuple, last, head) -> tuple:
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


def _last_write(call: Call, keys: Operand, k_ref, row, final_token, key_columns, columns, gated) -> tuple:
    """`algebra.carried`'s `last` for a chunk at `row` whose last real token is its `final_token`-th: that
    token's key slices as rows' vectors (its parts' rows summed, exactly, from `key_columns` on), its gated
    write's row as a column's (`columns`, `column_sums`), and its row, held to the fp32 recurrence outside the
    write product, as the portable math peels it (`_peeled_chunk_inputs`). A chunk past its document's end
    holds zeros after its tokens: its last slot's write left the last real one in the product, 2.0 times the
    portable math's final memory from float64 under a steep decay (interpreted and on an
    H100)."""
    family = call.family

    def key_rows(i):
        views = keys.vectors(k_ref, row + final_token, key_columns + i * call.block, call.block)
        return functools.reduce(jnp.add, (family.rows_of(view).astype(jnp.float32) for view in views))

    ends = family.iota(gated.shape, 0) == final_token
    return key_rows, columns(jnp.where(ends, gated, 0.0)), final_token


def _states_held(call: Call, values: Operand, keys: Operand, bv: int) -> Held:
    """What `_states`' consumer thread holds: the memory carried, its key slices in fp32, and the next
    chunk's prefix vectors (`prefetch`); a step's corrected and written values beside the values' block and
    a memory slice as a product's operand (and, where products read registers, a block of the weighted keys
    and of the keys); where the write's gate takes the keys' side (`Call.gated_keys`), a key block gated
    in fp32 and narrowed beside the values' block instead."""
    held = call.held
    carried = [f32(call.width, bv)] + [f32(CHUNK, 1)] * 5
    if call.gated_keys:
        # The write's gate on the keys' rows (`algebra.keys_gated_carried`): a key block in fp32 and narrowed
        # for its product, beside the values' block as they are, and the product's own fp32 sum `[K_i, BV]`
        # before it joins the decayed memory slice: without it the model held a 128-wide value block that
        # spilled 520 bytes on an H100 (the resource gate's Nemotron-H 4B SSD forward alone).
        return Held(carried=carried, phases=([
            f32(CHUNK, call.block), f32(call.block, bv), *held_tile(held, CHUNK, call.block),
            *held_tile(values, CHUNK, bv),
            *fragments(call, (held, CHUNK, call.block), (values, CHUNK, bv))],))
    return Held(carried=carried, phases=(
        [f32(CHUNK, bv)] * 2 + held_tile(values, CHUNK, bv) + held_tile(held, call.block, bv)
        + fragments(call, (held, CHUNK, call.block), (keys, CHUNK, call.block)),))


@lowered_once
def _states(k, w, u, hi, lo, beta, h0, sequences, taken, call: Call, tiles: Tiles):
    """`(h [chunks * HV * Dk, Dv], corrected [R, HV * Dv], final [B * HV * Dk, Dv])`: the memory each chunk
    enters and the corrected values as `call.held`, and each row's final memory, fp32, each sequence's chunks
    in order over a value block (`algebra.corrected`, `carried`). Under the delta rule `u` is the weighted
    values and `w` the weighted keys (`_weighted`); without it (`w` None) `u` is the values, each chunk's
    write gated by `beta` (`algebra.weighted`'s identity inverse; on the keys' rows where `Call.gated_keys`,
    `algebra.keys_gated_carried`, the values themselves its corrected values, `u` returned for them, no copy
    stored) and nothing corrected. A decay a
    head gates each chunk's write to its end (`algebra.scalar_write`); under one a key dimension `k` is the
    keys decayed so (`_weighted`'s third) and the memory's rows take their own decays. `sequences` `[N, 5]` is
    each sequence's first chunk, its chunks, its row and whether it begins and ends its row; `h0` `[B * HV *
    Dk, Dv]` fp32 each row's initial memory. A program walks `tiles.walkers` value blocks of its sequence and
    head, one a consumer role, its keys and weighted keys streamed once for all of them."""
    width, value_width, bv, family = call.width, call.value_width, tiles.block_v, call.family
    slices, walkers = width // call.block, tiles.walkers
    walked_width = bv * walkers
    held, memories = dataclasses.replace(call.held, rows=call.rows), call.memories
    batch = h0.shape[0] // (call.heads * width)
    initial = Operand(h0.shape[0], 1, jnp.dtype(jnp.float32))
    delta = w is not None
    values = held if delta else call.given
    # The keys the memory takes: a key head's, or a value head's decayed to their chunk's end.
    keys, keyed = (held, 1) if call.per_key else (call.given, call.group)

    # Where the keys take the write gate, the corrected values are the values as given: none stored.
    stores_corrected = not call.gated_keys

    def program(at, k_ref, w_ref, u_ref, hi_ref, lo_ref, beta_ref, h0_ref, seq_ref, taken_ref, h_ref,
                *outputs, streams, staging, landing):
        corrected_ref, final_ref = outputs if stores_corrected else (None, *outputs)
        j, s, h = at["blocks"], at["sequences"], at["heads"]
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        roled = role_tiles(staging, landing, walkers)

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        # Each step a chunk of every row's operands, by its index among the call's chunks: the program's value
        # blocks side by side, a role's its own columns.
        walked = j * walked_width
        views = stream_views(**({"w": held.views(w_ref, 0, call.rows, h * width, width)} if delta else {}),
                             keys=keys.views(k_ref, 0, call.rows, h // keyed * width, width),
                             values=values.views(u_ref, 0, call.rows, h * value_width + walked, walked_width))
        initial_views = stream_views(**{f"initial{i}": initial.views(h0_ref, memory(row, i), call.block,
                                                                     walked, walked_width)
                                        for i in range(slices)})
        chunks = family.opened(streams["walked"], views)
        entering = family.opened(streams["once"], initial_views)

        prefetch = functools.partial(_prefetched, family, (hi_ref, lo_ref), None if delta else beta_ref, h)
        # Each key slice of the memory and, where stored, the corrected values, a store each in flight.
        before_store = walk_stores(family, width // call.block + int(stores_corrected))

        def step(role):
            (staged, scratch), lands = roled(role)
            issue = Issuer(call, {"corrected": staged, "write": staged})
            column = walked + role * bv
            return functools.partial(_states_step, lands, issue, scratch, column, role)

        def _states_step(lands, issue, scratch, column, role, n, slots, state, accumulators, loaded=None):
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            for i, part in enumerate(state):
                before_store()
                memories.store(h_ref, part, memory(n, i), call.block, column, bv, family=family,
                               smem=lands[f"memory{i}"], wait=False)
            if call.gated_keys:
                # The values as they are the write's operand and their own corrected values: none formed.
                state = _keys_gated_write(call, slots, keys, values, loaded, state, issue, role, bv)
                return state, accumulators
            tile = stream_tile(slots, "values", values, columns=pl.ds(role * bv, bv)).value()
            if delta:
                def weights(i):
                    return stream_tile(slots, "w", held, columns=pl.ds(i * call.block, call.block))

                corrected = algebra.corrected(tile, weights, state, held.dtype, issue)
            else:
                betas = family.rows_of(beta_ref.at[h, span]) if loaded is None else loaded[2]
                written = algebra.weighted("apply_u", None, tile, betas, issue)
                corrected = algebra.corrected(written, None, state, held.dtype, issue)
            before_store()
            held.store(corrected_ref, corrected, n * CHUNK, CHUNK, h * value_width + column, bv,
                       family=family, smem=lands["corrected"], wait=False)
            gated, decay = _gated_write(call, corrected, loaded, (hi_ref, lo_ref), last, h)
            # An fp32 call's last real token's write outside the product; a 16-bit call's is fla's product.
            separate = (_last_write(call, keys, k_ref, n * CHUNK, taken_ref[n] - 1, h // keyed * width,
                                    column_sums(call, scratch), gated) if held.parts > 1 else None)
            state = algebra.carried(
                state, lambda i: stream_tile(slots, "keys", keys, columns=pl.ds(i * call.block, call.block)),
                gated, decay, held.dtype, issue, last=separate)
            return state, accumulators

        sequence_walk(call, initial, "initial", (entering, initial_views), (chunks, views), step, first,
                      count, begins, ends,
                      lambda role, i: final_ref.at[pl.ds(memory(row, i), call.block),
                                                   pl.ds(walked + role * bv, bv)], bv,
                      prefetch=None if call.per_key else prefetch,
                      landing=lambda role: roled(role)[1].get("final"), walkers=walkers)

    out_type, blocks, initials, staging, landing, after, work = _states_launch(
        call, tiles, delta, keys, values, batch, sequences.shape[0], stores_corrected)
    launched = launch_kernel(program, call, out_type=out_type,
                             grid=(value_width // walked_width, sequences.shape[0], call.heads),
                             grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=tiles.stages,
                             live=lambda at, refs: refs[7][at["sequences"], 1] > 0, work=work,
                             consumers=walkers, held=_states_held(call, values, keys, bv), staging=staging,
                             once=initials, landing=landing, after=after,
                             # Its last write's column sums an fp32 call's alone (`column_sums`).
                             reduces=held.parts > 1)(
        k, k if w is None else w, u, hi, lo, beta, h0, sequences, taken)
    if stores_corrected:
        return launched
    assert jnp.dtype(u.dtype) == held.dtype, "the values as given are held so (`Call.gated_keys`)"
    h, final = launched
    return h, u, final


def _states_launch(call: Call, tiles: Tiles, delta: bool, keys: Operand, values: Operand, batch: int,
                   sequences: int, stores_corrected: bool) -> tuple:
    """`_states`' outputs (the memories, the corrected values where it stores them, the final memories),
    its streamed blocks and its initial memories' at `tiles`, its staging and landing tiles and what its
    final memories' landing waits after (`final_memory_landing`), and its `Work` over `sequences`."""
    width, value_width, walked_width = call.width, call.value_width, tiles.block_v * tiles.walkers
    bv = tiles.block_v
    held, initial = dataclasses.replace(call.held, rows=call.rows), Operand(batch * call.heads * width, 1,
                                                                            jnp.dtype(jnp.float32))
    out_type = (call.memories.shape(value_width),
                *((held.shape(call.heads * value_width),) if stores_corrected else ()),
                jax.ShapeDtypeStruct((batch * call.heads * width, value_width), jnp.float32))
    blocks = stream_blocks(**({"w": ((CHUNK, width), held)} if delta else {}), keys=((CHUNK, width), keys),
                           values=((CHUNK, walked_width), values))
    initials = stream_blocks(**{f"initial{i}": ((call.block, walked_width), initial)
                                for i in range(width // call.block)})
    staging = (*staging_tiles(held, (CHUNK, call.block) if call.gated_keys else (call.block, bv)),
               column_sums_scratch(call))
    final_landing, after = final_memory_landing(call, bv)
    landing = {**memory_landing(call, bv),
               **({"corrected": held.landing(CHUNK, bv)} if stores_corrected else {}), **final_landing}
    # A sequence's chunks, the mean over the laid sequences; two products a chunk.
    steps = call.chunks / sequences
    return (out_type, blocks, initials, staging, landing, after,
            Work(flops=4 * CHUNK * width * walked_width * steps, span=steps))


def _scan_held(call: Call, values: Operand, keys: Operand, bv: int) -> Held:
    """What `_scan`'s consumer thread holds: `_states`' memory and prefix vectors carried (`_states_held`),
    beside the chunk's prefix as a column's vectors and its write gates; its step's phase; and the output's
    phase beside the memory: the corrected values and the output's fp32 sum, the chunk's own pairs' scores,
    their decays' two terms and their masks (an fp32 call's split into parts), and one product's fragments
    at a time, the scores', the memory's and the local one's issuing in turn."""
    held, given = call.held, call.given
    states = _states_held(call, values, keys, bv)
    scores = ([f32(CHUNK, CHUNK)] * 3 + masks(CHUNK, CHUNK)
              + (held_tile(held, CHUNK, CHUNK) if held.parts > 1 else []))
    products = _widest(
        _product(call, (given, CHUNK, call.block), (given, call.block, CHUNK), rhs_transposed=True),
        _product(call, (given, CHUNK, call.block), (call.memories, call.block, bv)),
        _product(call, (held, CHUNK, CHUNK), (held, CHUNK, bv)))
    # The delta rule's values corrected in the walk: the predicted values and their gated difference beside
    # the inverse's product.
    applied = [f32(CHUNK, bv)] * 2 + products if call.delta else []
    return Held(carried=[*states.carried, f32(CHUNK, 1)],
                phases=(*states.phases, applied, [f32(CHUNK, bv)] * 2 + scores + products))


@lowered_once
def _scan(q, k, a, v, hi, lo, beta, h0, sequences, taken, call: Call, tiles: Tiles, *, scale: float,
          out_dtype, skip=None):
    """`(o [R, HV * Dv], final [B * HV * Dk, Dv])` of a forward under a decay a head, one walk: `_states`'
    walk over each sequence's chunks in order, a value block a role, each chunk's output (`_output`'s: its
    own pairs' scores, `algebra.chunk_scores`, and `algebra.chunk_output` of the memory it enters and its
    corrected values) formed in the step from the memory in its registers, which no launch stores: the
    memories and corrected values `_states` writes for `_output` to read back, `[chunks * HV * Dk, Dv]` and
    `[R, HV * Dv]` in the values' form, written and read by no launch, nor `_output`'s queries and keys read a
    second time (FlashQLA's fused forward, flash_qla 0.1.3 hopper/fused_fwd.py). Under the delta rule its
    step corrects the values itself from the chunk's inverse less its unit diagonal `a` `[R, HV * C]`
    (`_solved`'s, applying none): `(I + L) (beta (v - exp2(g) k S))`, which `_states` reads as `u - w S`
    from the weighted keys and values `_weighted` writes, `[R, HV * Dk]` and `[R, HV * Dv]`, here neither
    written nor read. `v` the values as given, the rest `_states`' operands and the queries `q`, `scale`
    and `out_dtype` `_output`'s, and a `skip` `[HV]` adds `D v`, from the values as given, where the write
    gate takes the keys' side (`Call.gated_keys`). A decay a key dimension's output takes the intra
    launch's scores (`_output`'s `scores`): not this walk's."""
    assert not call.per_key, "a decay a key dimension's scores are the intra launch's"
    assert skip is None or call.gated_keys, "a skip reads the values a call whose keys take the gate holds"
    width, value_width, bv, family = call.width, call.value_width, tiles.block_v, call.family
    slices, walkers = width // call.block, tiles.walkers
    walked_width = bv * walkers
    held = dataclasses.replace(call.held, rows=call.rows)
    batch = h0.shape[0] // (call.heads * width)
    initial = Operand(h0.shape[0], 1, jnp.dtype(jnp.float32))
    delta = a is not None
    assert delta == call.delta, "the delta rule's walk corrects its values from the chunks' inverses"
    values = keys = call.given

    def program(at, q_ref, k_ref, a_ref, v_ref, hi_ref, lo_ref, beta_ref, h0_ref, seq_ref, taken_ref, *refs,
                streams, staging, landing):
        *skip_ref, o_ref, final_ref = refs
        j, s, h = at["blocks"], at["sequences"], at["heads"]
        first, count, row, begins, ends = (seq_ref[s, i] for i in range(5))
        roled = role_tiles(staging, landing, walkers)

        def memory(base, i):
            return (base * call.heads + h) * width + i * call.block

        walked = j * walked_width
        key_columns = h // call.group * width
        inverse = {"lower": held.views(a_ref, 0, call.rows, h * CHUNK, CHUNK)} if delta else {}
        views = stream_views(**inverse,
                             queries=keys.views(q_ref, 0, call.rows, key_columns, width),
                             keys=keys.views(k_ref, 0, call.rows, key_columns, width),
                             values=values.views(v_ref, 0, call.rows, h * value_width + walked, walked_width))
        initial_views = stream_views(**{f"initial{i}": initial.views(h0_ref, memory(row, i), call.block,
                                                                     walked, walked_width)
                                        for i in range(slices)})
        chunks = family.opened(streams["walked"], views)
        entering = family.opened(streams["once"], initial_views)
        prefetch = functools.partial(_prefetched, family, (hi_ref, lo_ref), beta_ref, h)
        # The output a store each step in flight (`walk_stores`).
        before_store = walk_stores(family, 1)

        def step(role):
            (staged, scratch, scored, entered, local, *applied), lands = roled(role)
            issue = Issuer(call, {"corrected": staged, "write": staged, "memory": entered, "local": local,
                                  **({"apply_u": applied[0]} if applied else {})})
            return functools.partial(_scan_step, lands, issue, scratch, scored, walked + role * bv, role)

        def _scan_step(lands, issue, scratch, scored, column, role, n, slots, state, accumulators, loaded):
            (_, _), (gate, low), betas = loaded
            span, last = pl.ds(n * CHUNK, CHUNK), n * CHUNK + CHUNK - 1
            # The prefix as a column's vectors for the chunk's own pairs' decays and, where the write gate
            # takes the keys' side, the write gates for the scores' columns (`_output`'s prologue's): read in
            # the step, not carried a step ahead beside the rows' (a column's vectors in the walk's carry).
            laid = tuple(family.columns_of(ref.at[h, span]) for ref in (hi_ref, lo_ref))
            column_scale = family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None
            queries, keyed = ([column_tile(call, slots, name, keys, c) for c in range(slices)]
                              for name in ("queries", "keys"))
            tile = stream_tile(slots, "values", values, columns=pl.ds(role * bv, bv))
            # The chunk's own pairs' scores staged for its local product (`_output`'s first step's).
            scores = algebra.chunk_scores(queries.__getitem__, keyed.__getitem__, slices, gate, low,
                                          held.dtype, Issuer(call), laid=laid, column_scale=column_scale)
            pieces = (product.formed(scores.astype(jnp.float32), held.parts) if held.parts > 1 else (scores,))
            for piece, buffer in zip(pieces, scored, strict=True):
                family.staged(piece, buffer)
            corrected = _scan_corrected(call, slots, tile, bv, state, (gate, low), betas, issue)
            # The memory the chunk enters, in its registers, staged as the memories' form for its product.
            out = algebra.chunk_output(queries.__getitem__, state.__getitem__, slices, gate, low,
                                       Parts(scored, held.dtype), corrected, scale, issue)
            if skip_ref:
                out = out + skip_ref[0][h] * tile.value().astype(jnp.float32)
            before_store()
            family.stored([out.astype(o_ref.dtype)], [o_ref.at[span, pl.ds(h * value_width + column, bv)]],
                          smem=lands["out"], wait=False)
            if call.gated_keys:
                state = _keys_gated_write(call, slots, keys, values, loaded, state, issue, role, bv)
                return state, accumulators
            gated, decay = _gated_write(call, corrected, loaded, (hi_ref, lo_ref), last, h)
            separate = (_last_write(call, keys, k_ref, n * CHUNK, taken_ref[n] - 1, key_columns,
                                    column_sums(call, scratch), gated) if held.parts > 1 else None)
            state = algebra.carried(state, lambda i: column_tile(call, slots, "keys", keys, i), gated, decay,
                                    held.dtype, issue, last=separate)
            return state, accumulators

        sequence_walk(call, initial, "initial", (entering, initial_views), (chunks, views), step, first,
                      count, begins, ends,
                      lambda role, i: final_ref.at[pl.ds(memory(row, i), call.block),
                                                   pl.ds(walked + role * bv, bv)], bv,
                      prefetch=prefetch, landing=lambda role: roled(role)[1].get("final"), walkers=walkers)

    blocks, initials, staging, landing, after, work, out_type = _scan_launched(
        call, tiles, out_dtype, steps=call.chunks / sequences.shape[0], rows=batch * call.heads * width)
    return launch_kernel(program, call, out_type=out_type,
                         grid=(value_width // walked_width, sequences.shape[0], call.heads),
                         grid_names=("blocks", "sequences", "heads"), blocks=blocks, stages=tiles.stages,
                         live=lambda at, refs: refs[8][at["sequences"], 1] > 0, work=work, consumers=walkers,
                         held=_scan_held(call, values, keys, bv), staging=staging, once=initials,
                         landing=landing, after=after, reduces=held.parts > 1)(
        q, k, k if a is None else a, v, hi, lo, beta, h0, sequences, taken,
        *(() if skip is None else (skip.astype(jnp.float32),)))


def _scan_corrected(call: Call, slots, tile, bv: int, state, prefix: tuple, betas, issue: Issuer):
    """`_scan`'s chunk's corrected values `[C, bv]` fp32 from its values `tile` and the memory it enters,
    `state`, a value block `bv` wide: the values themselves where the write gate takes the keys' side
    (`Call.gated_keys`); under the delta rule `u - w S` as `(I + L) (beta (v - exp2(g) k S))`, the keys'
    prediction from the memory decayed to each token, its difference from the values write-gated, corrected by
    the chunk's inverse (`lower`); else the write-gated values less nothing (`algebra.corrected`)."""
    held, keys = dataclasses.replace(call.held, rows=call.rows), call.given
    if call.gated_keys:
        return tile
    if call.delta:
        predicted = algebra.corrected(jnp.zeros((CHUNK, bv), jnp.float32),
                                      lambda i: column_tile(call, slots, "keys", keys, i), state, held.dtype,
                                      issue)
        difference = tile.value().astype(jnp.float32) + algebra.raised(*prefix)[:, None] * predicted
        return algebra.weighted("apply_u", stream_tile(slots, "lower", held), difference, betas, issue)
    written = algebra.weighted("apply_u", None, tile.value(), betas, issue)
    return algebra.corrected(written, None, state, held.dtype, issue)


def _scan_launched(call: Call, tiles: Tiles, out_dtype, *, steps: float, rows: int) -> tuple:
    """`_scan`'s streams (the inverse under the delta rule, the queries, keys and a role's values a chunk),
    the entering memories' once-streams, its staging tiles and landings, its work over `steps` chunks a
    sequence (the laid sequences' mean) and its outputs' types (the final memory `rows` long)."""
    width, bv, walkers = call.width, tiles.block_v, tiles.walkers
    walked_width = bv * walkers
    held, keys, delta = dataclasses.replace(call.held, rows=call.rows), call.given, call.delta
    initial = Operand(rows, 1, jnp.dtype(jnp.float32))
    blocks = stream_blocks(**({"lower": ((CHUNK, CHUNK), held)} if delta else {}),
                           queries=((CHUNK, width), keys), keys=((CHUNK, width), keys),
                           values=((CHUNK, walked_width), keys))
    initials = stream_blocks(**{f"initial{i}": ((call.block, walked_width), initial)
                                for i in range(width // call.block)})
    staging = (*staging_tiles(held, (CHUNK, call.block) if call.gated_keys else (call.block, bv)),
               column_sums_scratch(call), *staging_tiles(held, (CHUNK, CHUNK)),
               *staging_tiles(call.memories, (call.block, bv)), *staging_tiles(held, (CHUNK, bv)),
               # The delta rule's corrected values' operand for the inverse's product.
               *(staging_tiles(held, (CHUNK, bv)) if delta else ()))
    final_landing, after = final_memory_landing(call, bv)
    landing = {"out": ((CHUNK, bv), out_dtype), **final_landing}
    # A chunk's products: the memory's write and, under the delta rule, its prediction and the inverse's
    # correction; the scores once a role; the memory's and the local product's.
    corrections = (2 * CHUNK * width + 2 * CHUNK * CHUNK) * walked_width if delta else 0
    work = Work(flops=(2 * CHUNK * width * walked_width + corrections + 2 * CHUNK * CHUNK * width * walkers
                       + 2 * CHUNK * (width + CHUNK) * walked_width) * steps, span=steps)
    out_type = (jax.ShapeDtypeStruct((call.rows, call.heads * call.value_width), out_dtype),
                jax.ShapeDtypeStruct((rows, call.value_width), jnp.float32))
    return blocks, initials, staging, landing, after, work, out_type


def _output_streams(call: Call, bv: int) -> tuple[dict, tuple]:
    """`_output`'s streamed blocks and its staging tiles at value blocks `bv` wide: the queries, a value
    block's memory entered and values; under a decay a head the keys, the scores staged once for every
    value block's product; under one a key dimension the prefix tiles and the intra launch's scores, a key
    block's product staged."""
    width, held, memories, gates = call.width, call.held, call.memories, call.gates
    if call.per_key:
        given = {"gates": ((CHUNK, width), gates), "lows": ((CHUNK, width), gates),
                 "scores": ((CHUNK, CHUNK), held)}
        # The local product's staged scores, then each key block's gated queries.
        staging = (staging_tiles(held, (CHUNK, CHUNK)) + staging_tiles(call.given, *[(CHUNK, call.block)]
                                                                       * (width // call.block)),)
    else:
        given = {"keys": ((CHUNK, width), call.given)}
        staging = staging_tiles(held, (CHUNK, CHUNK))
    blocks = stream_blocks(queries=((CHUNK, width), call.given), entered=((width, bv), memories),
                           values=((CHUNK, bv), held), **given)
    return blocks, staging


def _output_swizzle(call: Call, bv: int, staging: tuple, landing: dict) -> int:
    """The swizzle `_output` stages its keys at under a decay a head (`Kᵀ` read transposed, each load a
    whole swizzle of its depth, `Family.rhs_fragment_depth`): the widest its key rows fill, and where products
    read their operands from shared memory the memory's value-block rows too, whose consumer thread's
    registers fit (`_output_held`, beside its addresses into its `staging` and `landing` tiles), halved down
    to the narrowest Mosaic stages; the narrowest where none fits, which the launch then refuses.
    Nemotron-H 4B's SSD output held a bf16 key block's whole 64-column depth at 128 bytes, 255 registers
    and a spill on an A100; at 64 bytes, 32 columns, it compiled within them (row 2943). Warpgroup MMA reads
    a product's two operands at one transform, and the queries are the left operand of the scores' product
    and the memory's (`Q Kᵀ`, `Q H`): the keys' swizzle is the memory's, which a 32-wide bf16 block's rows
    fill at 64 bytes, and at the keys' 128 Mosaic GPU found no layout for the launch."""
    family = call.family
    swizzle = resources.widest_swizzle(call.given.stored, call.width)
    if not family.operands_in_registers:
        swizzle = min(swizzle, resources.widest_swizzle(call.memories.stored, bv))
    while swizzle > mosaic.NARROWEST_SWIZZLE_BYTES and (
            _output_held(call, bv, swizzle).registers(family, _addressed(staging, landing))
            > family.registers(1)):
        swizzle //= 2
    return swizzle


def _output_held(call: Call, bv: int, swizzle: int | None = None) -> Held:
    """What `_output`'s consumer thread holds. Under a decay a head: a value block's phase, its output's
    fp32 sum (the memory's and the local product's, each its own accumulator where products overlap,
    `Family.products_overlap`), its values and one product's fragments (`_product`); and the first step's
    scores' phase, their sum, their pairs' decays and their masks (an fp32 call's split into parts) and their
    product's fragments. The two phases hold at once where the walk steps over value blocks (the scores'
    branch inside its loop); apart where it takes one value block, the scores staged before its products
    issue. So an A100's need was 232, 240, None and 216 registers at value blocks 16, 32, 64 and 128 wide of
    128 values, and None at 128 of 256; an H100's 222 at 64 and within 255 at 128 (Mosaic GPU forming
    each warpgroup product's descriptors at its use, JAX 0.12.0.1.post33; post32's hoisted descriptors
    held the 128-wide block past 255, where both phases were stated at once). Under a decay a key
    dimension the value block's phase, and apart from it the first step's: a key block's prefix halves, its
    decay raised from them and its queries widened, staged for every step (`algebra.key_gated`)."""
    held, memories, given, family = call.held, call.memories, call.given, call.family
    products = _widest(_product(call, (given, CHUNK, call.block), (memories, call.block, bv)),
                       _product(call, (held, CHUNK, CHUNK), (held, CHUNK, bv)))
    if call.per_key:
        # One accumulator for every product (`Issuer.summed`) beside the values, the intra launch's scores
        # in fp32 and their masked narrow form; apart from it, at the first step, the key blocks' gated
        # queries, staged before any product (an A100's held 64- and 128-wide value blocks at 247 registers,
        # an H100's 128-wide at 248).
        values = [f32(CHUNK, bv), *held_tile(held, CHUNK, bv), f32(CHUNK, CHUNK),
                  *held_tile(held, CHUNK, CHUNK),
                  *products]
        return Held(phases=(values, [f32(CHUNK, call.block)] * 4))
    values = [f32(CHUNK, bv)] * (2 if family.products_overlap else 1) + held_tile(held, CHUNK, bv)
    # A write gate on the keys' side scales the scores' columns (`chunk_scores`' `column_scale`): its vector
    # across a thread's columns, 16 fragments on a 64-wide tile (the lowered liveness of Nemotron-H 4B's SSD
    # output, ssd-evidence5 at 97a10a66b; row 2943).
    scores = ([f32(CHUNK, CHUNK)] * 3 + masks(CHUNK, CHUNK)
              + (held_tile(held, CHUNK, CHUNK) if held.parts > 1 else [])
              + ([f32(CHUNK, CHUNK // 2)] if call.gated_keys else []))
    # The keys staged as they arrive, a token a row: `Kᵀ`'s depth its stage's columns, a whole swizzle of it
    # a load (`swizzle`, `_output_swizzle`'s; a 128-byte one's 64 bf16 columns held Nemotron-H 4B's SSD
    # output 48 registers past a model of 16, 255 and a spill on an A100, where it stated 216; row 2943).
    scored = _product(call, (given, CHUNK, call.block), (given, call.block, CHUNK), rhs_transposed=True,
                      rhs_swizzle=swizzle)
    if call.value_width // bv == 1:
        return Held(phases=(values + products, scores + scored))
    # One product's fragments at a time where both phases hold at once, the products issuing in turn: each
    # phase's own beside the other stated an A100's fp32 call at 288 registers a 16-wide output block, where
    # ptxas needed 248 (tf32-fp32c at 9f82d62c4).
    return Held(phases=(values + scores + _widest(products, scored),))


@lowered_once
def _output(q, k, h, corrected, hi, lo, taken, beta, call: Call, tiles: Tiles, *, scale: float, out_dtype,
            scores=None, skip=None):
    """Each chunk's output `[R, HV * Dv]` in `out_dtype`, a program a chunk and value head and a step each
    value block: under a decay a head its own pairs' scores taken once, at the first step, and staged for
    every block's (`algebra.chunk_scores`, `chunk_output`), under one a key dimension the intra launch's
    (`scores`, `_intra`; `algebra.key_decayed_output`). Where the write gate takes the keys' side
    (`Call.gated_keys`) the scores' columns take `beta` `[HV, R]` before they are rounded, and a `skip`
    `[HV]` adds `D v` from the values the step holds (there the corrected ones)."""
    assert skip is None or call.gated_keys, "a skip reads the values a call whose keys take the gate holds"
    width, value_width, bv, gates = call.width, call.value_width, tiles.output_block, call.gates
    held, memories, family = dataclasses.replace(call.held, rows=call.rows), call.memories, call.family
    steps = value_width // bv

    def program(at, q_ref, k_ref, h_ref, v_ref, hi_ref, lo_ref, column_hi_ref, column_lo_ref, taken_ref,
                beta_ref, *refs, streams, staging, landing):
        *skip_ref, o_ref = refs
        n, h = at["chunks"], at["heads"]
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
        views = {**stream_views(lambda i: (0, 0), queries=queried, **given),
                 **stream_views(lambda i: (0, i), entered=entering,
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
                return stream_tile(slots, name, operand,
                                   **({"rows": place} if along_rows else {"columns": place}))

            entered = functools.partial(block, "entered", memories, along_rows=True)
            value = stream_tile(slots, "values", held)
            if call.per_key:
                local, *gated = staged
                # The tiles taken before the first step's branch: a slot is the stream's, every step's.
                queries, halves = ([block(name, operand, c) for c in range(width // call.block)]
                                   for name, operand in (("queries", call.given), ("gates", gates)))
                lows = [block("lows", gates, c) for c in range(width // call.block)]

                # Each key block's gated queries staged once, at the first step, for every value block's
                # product: computed each step, they were live across the walk's steps beside each one's sum.
                @pl.when(index == 0)
                def _():
                    for c, tile in enumerate(gated):
                        decays = (halves[c].value(), lows[c].value())
                        family.staged(algebra.key_gated(queries[c].value(), decays, scale), tile[0])

                out = algebra.key_decayed_output(
                    lambda c: Parts(gated[c], call.given.dtype), entered, width // call.block,
                    stream_tile(slots, "scores", held).value().astype(jnp.float32), value,
                    Issuer(call, {"local": local}).summed)
            else:
                assert loaded is not None, "a decay a head's prefix loads in the prologue"
                (gate, low), laid = loaded
                # The tiles taken before the first step's branch: a slot is the stream's, every step's.
                queries, keys = ([block(name, call.given, c) for c in range(width // call.block)]
                                 for name in ("queries", "keys"))

                @pl.when(index == 0)
                def _():
                    gated = family.columns_of(beta_ref.at[h, span]) if call.gated_keys else None
                    scored = algebra.chunk_scores(
                        queries.__getitem__, keys.__getitem__, width // call.block, gate, low, held.dtype,
                        Issuer(call), laid=laid, column_scale=gated)
                    pieces = (product.formed(scored.astype(jnp.float32), held.parts) if held.parts > 1
                              else (scored,))
                    for piece, buffer in zip(pieces, staged, strict=True):
                        family.staged(piece, buffer)

                out = algebra.chunk_output(queries.__getitem__, entered, width // call.block, gate, low,
                                           Parts(staged, held.dtype), value, scale, Issuer(call))
            if skip_ref:
                out = out + skip_ref[0][h] * value.value().astype(jnp.float32)
            view = o_ref.at[span, pl.ds(h * value_width + index * bv, bv)]
            family.stored([out.astype(o_ref.dtype)], [view], smem=landing["out"])
            return carry, accumulators

        return mosaic.Walk(views, step, steps=steps, prologue=None if call.per_key else prologue)

    blocks, staging = _output_streams(call, bv)
    out_type = jax.ShapeDtypeStruct((call.rows, call.heads * value_width), out_dtype)
    # The prefix twice, as the solve's.
    # The chunk's scores once (the intra launch's under a decay a key dimension), then each value block's
    # memory and local products.
    scored = 0 if call.per_key else 2 * CHUNK * CHUNK * width
    work = Work(flops=scored + 2 * CHUNK * (width + CHUNK) * value_width, span=steps)
    landing = {"out": ((CHUNK, bv), out_dtype)}
    swizzle = None if call.per_key else _output_swizzle(call, bv, staging, landing)
    keys = None if swizzle is None else {f"keys{p}": swizzle for p in range(call.given.parts)}
    return launch_kernel(program, call, out_type=out_type, grid=(call.chunks, call.heads),
                         grid_names=("chunks", "heads"), blocks=blocks, stages=1, live=chunk_liveness(8),
                         work=work,
                         held=_output_held(call, bv, swizzle), staging=staging, landing=landing,
                         # A 16-bit call's takes no reduction (an A100's asked for its tiles' 49152 bytes
                         # alone).
                         persistent=True, reduces=held.parts > 1, swizzles=keys)(
        q, k if scores is None else scores, h, corrected, hi, lo, hi, lo, taken, beta,
        *(() if skip is None else (skip.astype(jnp.float32),)))
