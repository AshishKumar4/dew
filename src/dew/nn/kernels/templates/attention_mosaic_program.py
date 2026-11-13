"""What the attention template's Mosaic GPU launches share (`attention_mosaic`, design §6): the bundles it
runs on (`bundle`), its configs (`Tiles`, `BackwardTiles`), its legality rule (`refused`), and what a
forward's and a backward's programs read alike: the call's padding to a launch's blocks, its spans' block map
and each tile's coordinates (`_Program`, `_Spans`, `_coords`), its hooks' parameters and state (`_Hooked`,
`_param`)."""

import dataclasses
import functools
import math
from collections.abc import Callable, Iterable
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    axes,
    bundles,
    layout,
    mosaic,
    operands,
    resources,
    swizzling,
)
from dew.nn.kernels.templates.attention import (
    Draw,
    Dropout,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    block_map,
    differentiable,
    floating_params,
    kept_dtype,
    key_tiles,
    power_of_two,
    shared_place,
    whole,
)
from dew.nn.kernels.templates.cost import estimate
from dew.nn.programs import called, equations


def few_rows(family: mosaic.Family, rows: int | None) -> bool:
    """Whether a launch of `rows` rows takes `family` at tiles of fewer rows than a role's products cover: a
    decode step's few packed rows (row 4042) on a family whose warps then lie along its tiles' columns
    (`Family.specialized`: mma.sync's one warp of 16 rows, the four along its keys), its query tile the
    issuer's rows (`query_rows`)."""
    if rows is None or rows >= family.role_rows:
        return False
    padded = -(-rows // family.rows) * family.rows
    return family.specialized(padded, family.role_rows) != family


def fewer_rows(family: mosaic.Family) -> tuple[int, ...]:
    """The consumer rows below a role's a backward launch takes on `family`, largest first: whole divisors of
    the role's rows in the issuer's, each one its warps arrange themselves by (`Issuer.specialized`:
    mma.sync's two warps of 16 rows at 32, its register operands then relayed through shared memory where
    they cross warps, `Issuer.relay_bytes`); none on a family whose role is one issuer's rows (wgmma's)."""
    return tuple(rows for rows in range(family.role_rows - family.rows, 0, -family.rows)
                 if family.role_rows % rows == 0 and family.specialized(rows, rows) != family)


def scopes_products(family: mosaic.Family) -> bool:
    """Whether `family`'s issuer keeps its products' accumulators in scratch a launch scopes
    (`Issuer.product_scratch`: tcgen05's chunks')."""
    return bool(family.product_scratch(1, (family.rows,)))


def qualified(family: mosaic.Family) -> bool:
    """Whether the attention mainloop runs on `family`: one whose consumers issue their products
    (`Family.issues_in_consumers`), whose walk's softmax and its rescales run between them in the registers
    that hold them (wgmma's, mma.sync's, a TPU core's MXU); not tcgen05's, one warp issuing them into tensor
    memory, which the Blackwell template walks (`attention_blackwell`)."""
    return family.issues_in_consumers


def bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
    """The bundle the attention mainloop runs on a device of `generation_name` for operands of `dtype`: the
    shared registry's (`bundles.of`) where attention is qualified on it (`qualified`); None elsewhere, whose
    calls another mainloop takes by legality."""
    # Every launch of the mainloop is written against the family's seams, a TPU core's among them (`tpu`),
    # and scopes its issuer's product scratch (`scoped`: tcgen05's chunks, `Family.scoped`'s `products`).
    family = None if generation_name is None else bundles.of(generation_name, dtype, scoped=True, tpu=True)
    return family if family is not None and qualified(family) else None


def layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
    """The bundles a launch of the attention mainloop takes on a device of `generation_name` for operands of
    `dtype`, in preference order (`bundles.layouts`: a producer's layout, then on warpgroup MMA the layout
    whose consumers feed their own rings, `mosaic_sm90.FED`; given the launch's `rows`, each issuer the
    device has, by the share of its rows they fill), where its family is qualified (`qualified`)."""
    if generation_name is None:
        return ()
    return tuple(family for family in bundles.layouts(generation_name, dtype, rows, scoped=True, tpu=True)
                 if qualified(family))


def runs_backward(family: mosaic.Family) -> bool:
    """Whether the attention backward's launches run on `family`: one the mainloop qualifies (`qualified`)
    whose issuer keeps no product scratch a launch scopes (`scopes_products` not: the backward's launches
    scope none; its tcgen05 form is row 4839's tensor-memory backward). Its dQ launch streams through a
    producer, which a family whose consumers feed their own rings has not (`split_queries`): there the one
    launch, queries first, alone (`MosaicBackward.legal`)."""
    return qualified(family) and not scopes_products(family)


def fuses(accurate: bool, parametrized: bool) -> bool:
    """Whether the backward's one launch takes a call of its form: the fast form's under no score hook with
    parameters, whose gradients no launch writes (`MosaicBackward.legal`); the accurate form and a hook with
    parameters take the two launches alone, the dQ launch's producer among them."""
    return not accurate and not parametrized


def backward_layouts(generation_name: str | None, dtype) -> tuple[mosaic.Family, ...]:
    """The bundles the backward's launches take on a device of `generation_name` for operands of `dtype`, in
    preference order (`bundles.layouts`, unscoped, `runs_backward`, its dQ launch streaming through a
    producer, `produced`): a B200's and consumer Blackwell's the mma.sync rings their consumers feed, then
    those a producer fills; none where the device has none."""
    if generation_name is None:
        return ()
    return tuple(filter(runs_backward, bundles.layouts(generation_name, dtype, tpu=True, produced=True)))


def backward_bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
    """The bundle the backward's dQ launch, or its one launch, runs on (`backward_layouts`' first), None
    where the device has none."""
    return next(iter(backward_layouts(generation_name, dtype)), None)


def loads_parts(family: mosaic.Family) -> bool:
    """Whether a forward's fp32 parts load on `family`: a copy each into a slot's rows (`rings.Pieces`), which
    a producer's loads take (`Transport.copies_in_pieces`) and neither a consumer's own ring
    (`Transport.feeds_itself`) nor cp.async's pipeline does."""
    return family.copies_in_pieces and not family.feeds_itself


def query_rows(family: mosaic.Family, dtype, rows: int) -> int:
    """The least query tile a launch of `rows` rows takes on `family`: a role's whole swizzled rows
    (`extents`), or the issuer's rows where the launch's rows are fewer than a role's (`few_rows`)."""
    if few_rows(family, rows):
        return -(-rows // family.rows) * family.rows
    return extents(family, dtype, rows)[-1]


def bundle_of(tiles: "Tiles | BackwardTiles", dtype) -> mosaic.Family:
    """The bundle `tiles` run on: the current device's (`generation.gpu_target`) layout their family names,
    which their legality matched (`MosaicForward.legal`). A config names its physical layout; which configs
    a call derives is the qualification's (`layouts`), so a config pinned on a family not yet qualified (an
    A100's mma.sync, its resource witness) resolves too. A forward's family is arranged for its consumer's
    share, `block_q` x `block_k` (`Issuer.specialized`, as the GEMM's: mma.sync's warps by the share, whose
    default 64-row atom no 32-row consumer's operands cast to); a backward's as `backward_family` arranges
    it."""
    # Every bundle the device issues the mainloop's launches (a launch of a row each ranks them all,
    # `bundles.layouts`), the backward's dQ launch's producer layout among them (`produced`).
    found = {family.name: family
             for family in bundles.layouts(generation.gpu_target().generation, dtype, 1, scoped=True,
                                           tpu=True, produced=True)}
    family = found.get(tiles.family)
    if family is None:
        raise ValueError(f"{tiles}: this device's attention bundles are {sorted(found)}")
    if isinstance(tiles, Tiles):
        return family.specialized(tiles.block_q, tiles.block_k)
    return backward_family(family, tiles)


def backward_family(family: mosaic.Family, tiles: "BackwardTiles") -> mosaic.Family:
    """`family` arranged for a backward's `tiles`, whose launches transpose them (dQ's rows its queries,
    dK/dV's its keys): by their smaller side where that is fewer rows than a role's (`fewer_rows`), else as
    the family stands (four warps along a role's rows hold whole rows of the accumulators the products then
    read from registers)."""
    side = min(tiles.block_q, tiles.block_k)
    return family.specialized(side, side) if side < family.role_rows else family
"""The instruction families this mainloop runs on (`generation.family`), each by its bundle: wgmma's. A
family without one (mma.sync's, until its thin mainloop) takes another mainloop (`MosaicForward.eligible`,
`MosaicBackward.eligible`)."""


def stages_after_walks(family: mosaic.Family) -> bool:
    """Whether a launch stages its outputs (a forward's output or split sums, dQ's sums) after its walks, in
    their slots (`Family.scoped`'s `after_walks`, `staged_after`): where its family scopes its walks apart
    (`walks_scoped`, cp.async's pipelines), no walk reading them. An A100's forward at width 128 staged its
    output beside its query's and its rings, 101,392 bytes at 128 x 32, one program a multiprocessor, where
    FlashAttention 2 runs two (M-ET-48: 6.25% occupancy, 1.39-1.57x its time)."""
    return family.walks_scoped


def staged_after(after: Callable | None, stages, index, store: Callable) -> None:
    """`store(stage)` of consumer `index`'s output stage: the program's `stages`', or one `after` scopes after
    the walks (`stages_after_walks`)."""
    if after is None:
        store(stages.at[index])
    else:
        after(lambda scoped: store(scoped.at[index]))


def query_swizzle(family: mosaic.Family) -> int | None:
    """The swizzle a forward stages its query block at, the left operand of its logits' product: the
    narrowest where its family loads operands into registers (`Family.operands_in_registers`), 8-row tiles
    32 bytes wide, so a load of the block's fragments one instruction deep (16 of a 16-bit row) is a whole
    tile and its 8 rows' 16-byte reads take every bank once, where a plain block's 256-byte rows take one
    bank group 8 times; layout inference's otherwise (`swizzling.staged`). The query is written once a
    tile."""
    return swizzling.NARROWEST_SWIZZLE_BYTES if family.operands_in_registers else None


def transposed_swizzles(tiles: "Tiles | BackwardTiles", *names: str) -> dict[str, int]:
    """The swizzles `Family.streams` stages the streams `names` at, which a launch's products read
    transposed: the narrowest where `tiles` stage them so (`Tiles.narrow`), the family's own otherwise."""
    return dict.fromkeys(names, swizzling.NARROWEST_SWIZZLE_BYTES) if tiles.narrow else {}


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A forward launch's config: each consumer's query block (`block_q`), the key tile (`block_k`), the key
    and value tiles in flight (`stages`), the consumers a program (`consumers`: on wgmma FlashAttention-3's
    two warpgroups take turns on the tensor cores, one's softmax under the other's products) and the
    instruction family whose bundle runs it (`family`, `bundle`). A fold's call rewalks a row past its reach
    in the natural form at its own key tile and consumers (`relaunch_k`, `relaunch_consumers`; the config's
    where None, `relaunch`), which divide the config's so its padded blocks hold, and the value columns each
    of its launches walks (`relaunch_columns`, a launch a block of them; all of them where None);
    `natural_form` marks the natural form's tiles, whose register model differs (`MosaicForward.registers`),
    and `columns` the value columns a launch at them walks."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    family: str = dataclasses.field(kw_only=True)
    relaunch_k: int | None = None
    relaunch_consumers: int | None = None
    relaunch_columns: int | None = None
    natural_form: bool = False
    columns: int | None = None
    narrow: bool = False
    """Whether the streams its products read transposed stage at the narrowest swizzle
    (`transposed_swizzles`), where its family reads operands from registers: a load one instruction's depth
    of them (`Family.rhs_fragment_depth`), not a wide swizzle tile's, a wide stage's bandwidth traded for
    registers (ET-CENSUS80: a 64 x 192 forward's keys at the widest 162 of its 291 registers)."""

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")

    @property
    def relaunch(self) -> "Tiles":
        """The natural form's rewalk's tiles (`algebra.walked_launches`)."""
        return dataclasses.replace(self, block_k=self.relaunch_k or self.block_k,
                                   consumers=self.relaunch_consumers or self.consumers, relaunch_k=None,
                                   relaunch_consumers=None, relaunch_columns=None, natural_form=True,
                                   columns=self.relaunch_columns)


@dataclasses.dataclass(frozen=True, order=True)
class BackwardTiles:
    """A backward's config, one for its launches: the query block a dQ consumer takes and the key block a
    dK/dV consumer takes (`block_q`, `block_k`, so both launches' logits are one tile's), the tiles in
    flight (`stages`), the consumers a program (`consumers`), whether the fast form runs as one launch
    (`fused`, `_launch_backward`), the instruction family whose bundle runs it (`family`) and the dK/dV
    programs a key block's head widths split among (`columns`: each its share of dK's and dV's columns, its
    logits and their cotangents its own, where a whole width's sums pass a consumer's registers) and the
    dK/dV launch's own layout and consumers (`dkv`)."""

    block_q: int
    block_k: int
    stages: int
    consumers: int
    fused: bool = False
    family: str = dataclasses.field(kw_only=True)
    columns: int = 1
    dkv_family: str | None = None
    dkv_consumers: int | None = None
    staged_logits: bool = False
    """Whether the keys-first dK/dV launch stages its logits' queries (the call's taken by its power) a
    step from the call's own in its consumers' shared memory, where a stream of them beside the call's
    passes the block (width 256's)."""
    shared_dq: bool = False
    """Whether the keys-first fused launch's consumers, each its own keys, exchange their score cotangents
    for a cooperative dQ: each forms its half of dQ's columns over every key and adds it, one tile's dQ added
    once a program (FlashAttention 3's width 128, run_mha_bwd_hdim128: AtomLayoutMdQ 1 over two MMA
    warpgroups; each consumer adding a whole tile over its own keys adds twice, the measured bound,
    ET-DQBOUND-h100)."""
    producer_dq: bool = False
    """Whether a shared dQ's adds are its producer's (`mosaic.Outbox`): each consumer stages its share of a
    tile's dQ, which the producer adds by TMA's reduction a ring's depth behind its loads and frees once
    read, the adds' waits off the consumers' walk (FlashAttention 3's dQ store warp)."""
    shared_keys: bool = False
    """Whether the fused launch's consumers share one key block (`attention_mosaic_coop._coop_launch`):
    each forms the logits of its share of the keys and holds its share of the head's columns of dK, dV and
    dQ, where a consumer's own keys' whole dK and dV sums pass its registers (width 256's)."""
    narrow: bool = False
    """Whether the streams its products read transposed stage at the narrowest swizzle
    (`transposed_swizzles`), where its family reads operands from registers: a load one instruction's depth
    of them (`Family.rhs_fragment_depth`), not a wide swizzle tile's, a wide stage's bandwidth traded for
    registers (ET-CENSUS80: a 64 x 192 forward's keys at the widest 162 of its 291 registers)."""

    @property
    def dkv(self) -> "BackwardTiles":
        """The dK/dV launch's own tiles: its layout and consumers (`dkv_family`, `dkv_consumers`, the
        config's where None), each launch's layout its own (`bundles.layouts`)."""
        return dataclasses.replace(self, family=self.dkv_family or self.family,
                                   consumers=self.dkv_consumers or self.consumers, dkv_family=None,
                                   dkv_consumers=None)

    @property
    def dkv_apart(self) -> bool:
        """Whether the dK/dV launch runs on a layout or a count of consumers of its own (`dkv`): its tiles
        other than the config's own, overrides naming the config's own aside."""
        return self.dkv != dataclasses.replace(self, dkv_family=None, dkv_consumers=None)

    @property
    def queries_first(self) -> bool:
        """Whether the backward is the fused launch queries first (`_fused_launch`), on the config's own
        layout and consumers: each consumer its own keys over the whole query slot, on either layout."""
        return self.fused and not self.dkv_apart

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


def _width(width: int, family: mosaic.Family, dtype) -> int:
    """A head width as a launch stages it: a power of two (`padded_call`) in whole swizzled rows
    (`Family.column_unit`)."""
    return whole(power_of_two(width), family.column_unit(dtype))


def _products(family: mosaic.Family, dtype, *products: resources.Product, narrow: bool = False,
              lhs_swizzle: int | None = None) -> list:
    """The phases a step's products hold where its family's read their operands from registers
    (`resources.product_phases`; none where they read shared memory, wgmma's): each product
    (`resources.Product`) its operands' fragments beside the tiles it holds at once (`beside`: its output, a
    tile it reads in registers), a left operand already in registers (`held`: the weights, the score
    cotangents) no fragment of its own, a right one read as the transpose of its stored tile (`transposed`:
    the keys of the logits, the values of dP) loaded at its stage's swizzle granule
    (`Family.rhs_fragment_depth`; mma.sync: ET-CENSUS80, a 64 x 192 forward's keys' fragments 162 of its 291
    registers at its peak, uncounted), the narrowest where its launch stages them `narrow` (`Tiles.narrow`);
    a left operand in shared memory loaded at its stage's (`lhs_swizzle`: a forward's query block,
    `query_swizzle`; the family's own where None). Products naming one right `source` read one staged
    operand, live through the intervening products too."""
    narrowest = swizzling.NARROWEST_SWIZZLE_BYTES if narrow else None
    return resources.product_phases(family, dtype, *products, lhs_swizzle=lhs_swizzle, rhs_swizzle=narrowest)


def stagings(family: mosaic.Family) -> tuple[bool, ...]:
    """Whether a launch's transposed streams stage narrowest (`Tiles.narrow`), the stagings a derivation
    tries in turn: the family's own swizzle first, then the narrowest where its products read their
    operands from registers (`Family.operands_in_registers`), whose fragments a narrow stage shrinks."""
    return (False, True) if family.operands_in_registers else (False,)


class _Issuing(NamedTuple):
    """What an `operands.Issuer` reads of a launch: its family and whether it interprets."""

    family: mosaic.Family
    interpret: bool


def issued_into(family: mosaic.Family, acc, parts, rhs, interpret: bool):
    """`acc` with each of `parts`' products with `rhs`, a tile in shared memory, issued into it in place
    (`operands.Issuer`'s carried accumulator), each part's left operand in registers and formed as it is
    taken (`algebra.formed_parts`); an fp16 part's (`product.Scaled`) its own accumulator, restored by its
    row's power of two (`algebra.restored`) and added in."""
    issuer = operands.Issuer(_Issuing(family, interpret), {"part": 1}, restored=algebra.restored)
    right = operands.Parts((rhs,), rhs.dtype)
    return issuer.summed((("part", part, right) for part in parts), into=acc)


def issued_pairs(family: mosaic.Family, acc, pairs: Iterable, interpret: bool, *,
                 shape: tuple[int, int] | None = None):
    """`acc`, or a fresh fp32 accumulator of `shape` where None, with each of `pairs`' products (a left
    operand in registers and a tile in shared memory, each formed as it is taken) issued into it in place,
    in order, none read before the last (`operands.Issuer.summed`): an fp32 call's parts' products
    (`attention_mosaic_forward._parted_products`), one accumulator where a product apiece copied each in and
    out (et-fp32regs2: the guarded launch 248 registers and 24 bytes spilled where its bf16 one held 168)."""
    issuer = operands.Issuer(_Issuing(family, interpret), {"part": 1}, restored=algebra.restored)
    products = (("part", lhs, operands.Parts((rhs,), rhs.dtype)) for lhs, rhs in pairs)
    return issuer.summed(products, shape) if acc is None else issuer.summed(products, into=acc)


def _block(config: Tiles | BackwardTiles) -> int:
    """The block a launch at `config` pads a call's queries and keys to: whole programs of query blocks and
    key tiles (a forward's), or of query blocks and key blocks (a backward's dQ and dK/dV programs, a shared
    dQ's dK/dV program a key block each of its consumers': padded to a block of the dQ launch's consumers,
    its last program's last consumer read keys past the padded call, NaN interpreted), its keys whole
    blocks of the dK/dV launch's programs too (`dkv_holders`: a layout and consumers of its own, padded
    to the dQ launch's, its last keys had no program to write them)."""
    if isinstance(config, BackwardTiles):
        keys = (config.dkv.consumers if config.shared_dq else config.consumers) * config.block_k
        keys = math.lcm(keys, dkv_holders(config) * config.block_k)
    else:
        keys = config.block_k
    return math.lcm(config.consumers * config.block_q, keys)


def split_queries(family: mosaic.Family) -> bool:
    """Whether the dK/dV program's consumers share one key block and split each slot's queries between them:
    on a layout whose consumers feed their own rings (`Transport.feeds_itself`), each consumer's step its
    share of the queries' tiles and per-query statistics (`a05-twoconsumer`); else each consumer its own
    keys."""
    return family.feeds_itself


def holders(family: mosaic.Family, consumers: int) -> int:
    """The key blocks a dK/dV program holds: one its consumers share (`split_queries`), else one each."""
    return 1 if split_queries(family) else consumers


def dkv_holders(config: BackwardTiles) -> int:
    """The key blocks a program of the dK/dV launch at `config` holds (`holders`), its layout's own."""
    dkv = config.dkv
    return holders(bundles.named(dkv.family, jnp.bfloat16), dkv.consumers)


def _padded(queries: int, keys: int, config: Tiles | BackwardTiles) -> tuple[int, int]:
    """A call's queries and keys as a launch at `config` pads them (`padded_call`, `_block`)."""
    block = _block(config)
    return whole(max(queries, 1), block), whole(max(keys, 1), block)


def launch_scope(family: mosaic.Family, allocations: tuple, reducing_warpgroups: int, *, reduces: bool = True,
                 products: tuple | None = None) -> resources.Scope:
    """A launch's `allocations` (`(scoped, streams)`, `Family.scoped`'s, and those it scopes after its
    walks where it has them, `after_walks`) by lifetime (`Family.lifetimes`), its issuer's product scratch
    where its `products` take one (`Family.product_held`, its tensor memory's addresses
    `Issuer.product_addresses`), with the algebra's reductions' scratch where it `reduces` (a launch reducing
    its scores, weights or cotangents across warps) and its `reducing_warpgroups`' scratch
    (`resources.Scope`)."""
    scoped, streams, *after = allocations
    held, walks = family.lifetimes(streams)
    return resources.Scope(program=(scoped, *family.product_held(products), *held), walks=walks,
                           after=tuple((tile.shape, tile.dtype) for tile in (after[0] if after else ())),
                           reduces=reduces, reducing_warpgroups=reducing_warpgroups,
                           addresses=family.product_addresses(products))


def extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most what one copy moves (`Family.copy_extent`,
    where the family's copies bound it; else registers bound them, `fits`) and the call's `extent` in whole
    multiples (a longer tile holds padding alone)."""
    return _multiples(family, math.lcm(family.role_rows, family.column_unit(dtype)), extent)


def walked_extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A walked tile's extents on `family`, largest first: whole multiples of its swizzled rows
    (`Family.column_unit`), a walked tile (a forward's or a dQ launch's keys, a dK/dV launch's queries) a
    product's depth and no consumer's rows, within `extents`' bounds: mma.sync's walks take 32-row tiles,
    where its 64-row unit held no guarded state at width 128 (FS-GUARD80-2) and no plain tile's stages at
    width 256 (64 x 64 staged 199,696 bytes of an A100's 166,912, 64 x 32 134,160, et-d256e)."""
    return _multiples(family, family.column_unit(dtype), extent)


def _multiples(family: mosaic.Family, unit: int, extent: int) -> tuple[int, ...]:
    """Whole multiples of `unit`, largest first, at most what one of `family`'s copies moves and the call's
    `extent` in whole multiples (`extents`)."""
    top = -(-max(extent, 1) // unit)
    if family.copy_extent is not None:
        top = min(top, family.copy_extent // unit)
    return tuple(range(top * unit, 0, -unit))


BACKWARD_CONSUMERS = 2
"""The consumers a backward launch takes at most: its exchanges and splits pair them (a dK/dV block's column
shares, a slot's query shares, `Family.exchange`'s two schedules). A third left a 64-row tile 21 rows a share
and the accurate form 30 to 290 times the reference's error (045aa11863e). The forward's walk
takes the family's every consumer (FlashAttention 3's d64 tile: three, 192 rows)."""


def most_consumers(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> int:
    """The consumers a launch of `tiles` takes at most on `family`: a forward's the family's own
    (`Family.consumers`), a backward's at most `BACKWARD_CONSUMERS`."""
    return family.consumers if isinstance(tiles, Tiles) else min(family.consumers, BACKWARD_CONSUMERS)


def whole_tiles(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> bool:
    """Whether a program of `family` takes `tiles`: its query and key tiles positive whole multiples of the
    rows a role's products cover (`Family.rows`), and one to its launch's count of consumers
    (`most_consumers`), whose turns its walks are written for."""
    return (0 < tiles.consumers <= most_consumers(family, tiles) and tiles.block_q > 0 and tiles.block_k > 0
            and tiles.block_q % family.rows == 0 and tiles.block_k % family.rows == 0)


ROUNDED = Refusal('rounds', "a quantized model's weights round at their row's maximum in the anchored "
                  "training walk's forward alone (`Hooks.rounds`), no dropping or resumed call's, and its "
                  "backward is its quantization's own (`attention_quantized`)")
"""Why the mainloops take a quantized call's forward alone, of neither a dropping nor a resumed call."""


def refused(dtype, width: int, value_width: int, hooks: Hooks | None, *, backward: bool,
            family: mosaic.Family) -> Refusal | None:
    """Why this mainloop takes no call of `dtype` operands, head widths `width` and `value_width` and `hooks`
    (its forward, or with `backward` its backward) on `family`'s bundle, or None: its legality rule, which
    the ops' eligibility (`MosaicForward.eligible`, `MosaicBackward.eligible`), the route and the stability
    suite read."""
    hooks = Hooks() if hooks is None else hooks
    sixteen = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
    if hooks.rounds is not None and (backward or hooks.drop is not None or hooks.initial is not None):
        return ROUNDED
    if hooks.drop is not None and not algebra.drops_in_place(dtype, hooks.drop.bound):
        # Dropped weights past the dtype's largest take scaled parts (`algebra.dropped_parts`), which the
        # mainloop's sums in place do not take back (`mosaic.unscaled_part`).
        return Refusal('dropout_rate', f"the Mosaic dropping walk takes {jnp.dtype(dtype).name} weights "
                       f"within its range, not a multiplier of {hooks.drop.bound:g} (a rate of "
                       f"{hooks.drop.rate:g})")
    if jnp.dtype(dtype) not in sixteen:
        return Refusal('dtype', f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(dtype)}")
    if backward and not transposable(hooks):
        return Refusal('hook_layout', "the Mosaic backward holds its tiles keys first, and its score hook "
                       "takes an axis of them by an operation no layout rule takes (`layout.RULES`)")
    widest = max(_width(width, family, dtype), _width(value_width, family, dtype))
    if family.copy_extent is not None and widest > family.copy_extent:
        return Refusal('copy_extent', f"the Mosaic mainloop loads a head {widest} wide in one copy, past the "
                       f"{family.copy_extent} one moves (`Family.copy_extent`)")
    return None


APPROXIMATED = frozenset({jax.lax.exp_p, jax.lax.exp2_p, jax.lax.log_p, jax.lax.log2_p, jax.lax.log1p_p,
                          jax.lax.tanh_p, jax.lax.logistic_p, jax.lax.sin_p, jax.lax.cos_p, jax.lax.sqrt_p,
                          jax.lax.rsqrt_p, jax.lax.pow_p})
"""The operations Mosaic GPU computes in the hardware's approximate forms under `approx_math` (its lowering's
exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt; jax's logistic, log1p and pow lower through them)."""


def approximate(hooks: Hooks, tile: tuple[int, int], value_width: int, *, differentiated: bool) -> bool:
    """Whether a launch running `hooks` on `tile` takes the hardware's approximate transcendentals
    (`Family.kernel`'s `approx_math`): only where no hook computes one Mosaic GPU approximates
    (`APPROXIMATED`), with `differentiated` the score hook's pullback too. The mainloop's own exponentials
    hold check 1 at ex2.approx; a hook's need not: on an H100 a softcap's tanh.approx (2^-11 relative) beside
    an initial state put the state's sums' gradient at 7.04 times the reference's error from float64, every
    hook on par with exact math (the hooked backward, HOOKS-FIX-h100)."""
    traced = _hook_jaxprs(hooks, tile, value_width, differentiated=differentiated)
    return not any(_primitives(jaxpr) & APPROXIMATED for jaxpr in traced)


def reducing(family: mosaic.Family, consumers: int, hooked: bool, *, flagged: bool = False,
             scaled: bool = False, magnitudes: bool = False) -> int:
    """The warpgroups of a launch of `consumers` whose arithmetic may reduce across warps implicitly, each
    reserving its share of Mosaic GPU's reduction scratch (`Family.kernel`'s `reducing_warpgroups`): a
    `hooked` launch's hook's (`hook_reduces`: a unit row, which the fork selects by a reduction across warps,
    ledger 1428; an output hook's, untraced), a `flagged` launch's rows reduced to their block's flag (the
    plain launch that flags its rows, `attention_mosaic_forward._Flags`), a `scaled` step's tile reduced
    to its largest across the warps holding its rows (`algebra.running_parts`, an fp16 fused backward's), or
    a scaled walk's keys' value `magnitudes`, its value tile's columns reduced across the warps holding them
    (`Family.key_magnitudes`, `algebra.SCALED_WALK`), so every warpgroup of a launch that runs such a hook,
    flags or scales; none otherwise."""
    if flagged or scaled or magnitudes or hooked:
        return family.warpgroups(consumers)
    return 0


def hook_reduces(hooks: Hooks | None) -> bool:
    """Whether the score hook may reduce across warps (`reducing`): traced, never run, on a tile with its
    pullback, it forms or reads a value but a scalar or a whole tile (a unit row, `k_pos[:1, :]`, ledger
    1428; a sum along an axis) or nests a computation but a call's (a branch, a loop); a fingerprint (a keyed
    launch's hook) counted so. Gemma 2's softcap forms none, so its launches reserve no scratch: its
    backward's one launch 256 wide on an H100 fits beside it (row 3704: 233,568 bytes with it, past the
    block's 232,448). False without a hook."""
    if hooks is None or hooks.score_mod is None:
        return False
    if not callable(hooks.score_mod):
        return True
    params = tuple((param.kind, kept_dtype(param.value.dtype)) for param in hooks.score_params)
    return _hook_reduces(hooks.score_mod, params)


@functools.lru_cache(maxsize=256)
def _hook_reduces(score_mod, params: tuple) -> bool:
    """`hook_reduces` of `score_mod` under parameters of `params`' kinds and dtypes."""
    tile = (TILE_SIDES[0], TILE_SIDES[0])
    traced = _score_jaxprs(score_mod, params, tile, differentiated=True)
    return any(_off_tile(jaxpr, tile) for jaxpr in traced)


def _off_tile(jaxpr: jax_core.Jaxpr, tile: tuple[int, int]) -> bool:
    """Whether an equation of `jaxpr` forms or reads a value but a scalar or a `tile`, or nests a computation
    but a call's (a jitted helper's, a custom derivative's primal), whose own equations are read so."""
    for eqn in jaxpr.eqns:
        if any(getattr(var.aval, "shape", ()) not in ((), tile) for var in (*eqn.invars, *eqn.outvars)):
            return True
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        on_tile = eqn.primitive.inlined_call and len(nested) == 1 and not _off_tile(nested[0], tile)
        if nested and not on_tile:
            return True
    return False


def _hook_jaxprs(hooks: Hooks, tile: tuple[int, int], value_width: int, *,
                 differentiated: bool = False) -> list[jax_core.Jaxpr]:
    """`hooks` traced as a launch runs them on `tile`, its outputs `value_width` wide: the score hook (and,
    `differentiated`, its pullback along the logits and its floating parameters) and the output hook."""
    traced = []
    # A keyed launch's hook is its fingerprint (`tuning.program`), whose lowering its live call established.
    if callable(hooks.score_mod):
        traced += _score_jaxprs(hooks.score_mod, tuple((param.kind, param.value.dtype)
                                                       for param in hooks.score_params),
                                tile, differentiated=differentiated)
    if callable(hooks.output_mod):
        traced.append(jax.make_jaxpr(hooks.output_mod)(
            jax.ShapeDtypeStruct((tile[0], value_width), jnp.float32),
            *(_entries(param.kind, param.value.dtype, (tile[0], value_width))
              for param in hooks.output_params)).jaxpr)
    return traced


def _entries(kind: Kind, dtype, tile: tuple[int, int]) -> jax.ShapeDtypeStruct:
    """A hook parameter as `_param` gives it on `tile`: a head's scalar, else the tile's entries."""
    return jax.ShapeDtypeStruct(() if kind == Kind.HEAD else tile, kept_dtype(dtype))


def _score_jaxprs(score_mod, params: tuple, tile: tuple[int, int], *,
                  differentiated: bool) -> list[jax_core.Jaxpr]:
    """`score_mod` under parameters of `params`' kinds and dtypes traced on `tile`, and with `differentiated`
    its pullback along the logits and its floating parameters."""
    positions, logits = jax.ShapeDtypeStruct(tile, jnp.int32), jax.ShapeDtypeStruct(tile, jnp.float32)
    values = [_entries(kind, dtype, tile) for kind, dtype in params]
    traced = [jax.make_jaxpr(score_mod)(logits, positions, positions, *values).jaxpr]
    if differentiated:
        floating = floating_params(values)

        def pulled(s, q_pos, k_pos, *values):
            hooked = differentiable(score_mod, q_pos, k_pos, values, floating)
            out, pull = jax.vjp(hooked, s, *(values[index] for index in floating))
            return pull(jnp.ones_like(out))

        traced.append(jax.make_jaxpr(pulled)(logits, positions, positions, *values).jaxpr)
    return traced


TILE_SIDES = (32, 64, 128, 192, 256)
"""The sides a program's score tile may take, at every shape of which a hook's legality traces it
(`transposable`): a consumer's share of a slot's queries (32 of a 64-row slot, `split_queries`) among
them."""


def score_places(hooks: Hooks | None) -> tuple[bool, bool]:
    """Whether the score hook reads its queries' places and its keys' (its `q_pos`, `k_pos`): traced, never
    run, on a tile."""
    if hooks is None or not callable(hooks.score_mod):
        return False, False
    params = tuple((param.kind, kept_dtype(param.value.dtype)) for param in hooks.score_params)
    return _score_places(hooks.score_mod, params)


@functools.lru_cache(maxsize=256)
def _score_places(score_mod, params: tuple) -> tuple[bool, bool]:
    """`score_places` of `score_mod` under parameters of `params`' kinds and dtypes."""
    side = TILE_SIDES[0]
    jaxpr = _score_jaxprs(score_mod, params, (side, side), differentiated=False)[0]
    _, queries, keys, *_ = _live_inputs(jaxpr, [True] * len(jaxpr.outvars))
    return queries, keys


def _live_inputs(jaxpr: jax_core.Jaxpr, live_outputs: list[bool]) -> list[bool]:
    """Which of `jaxpr`'s inputs its `live_outputs` surely read: through each equation whose outputs are live,
    into a call's own jaxpr input by input (a jitted helper's, a custom derivative's primal); an equation of
    any other nested computation (a branch's, a loop's) surely reads none, a place it may read left a tile
    (`_coords`)."""
    live = {var for var, used in zip(jaxpr.outvars, live_outputs, strict=True)
            if used and not isinstance(var, jax_core.Literal)}
    for eqn in reversed(jaxpr.eqns):
        used = [var in live for var in eqn.outvars]
        if not any(used):
            continue
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        # A call (a jitted helper's, a custom derivative's primal): its jaxpr's inputs its own, read where
        # every reader reads a call's program (`dew.nn.programs.called`).
        body = called(eqn)
        reads = [not nested] * len(eqn.invars) if body is None else _live_inputs(body, used)
        live.update(var for var, read in zip(eqn.invars, reads, strict=True)
                    if read and not isinstance(var, jax_core.Literal))
    return [var in live for var in jaxpr.invars]


def varying_slopes(hooks: Hooks | None, graded: tuple[Kind, ...]) -> int:
    """The tiles a backward launch's score-hook pullback forms that vary with its inputs
    (`attention_gradient.scored`): its output and its slope along each input it differentiates, the logits and
    each floating parameter of the kinds whose gradients the launch forms (`graded`, `DQ_GRADS` or
    `DKV_GRADS`: the dK/dV launch takes no query's or head's slope, an A100's and an H100's dK/dV under a
    temperature, a softcap with sinks and ALiBi, and ALiBi holding 3, 3 and 2 tiles at 64 x 64 where all their
    slopes priced 4, 4 and 3), but a constant one (a hook linear in an input slopes 1 along it, which ptxas
    holds no tile of: the fast dQ under a query's temperature and a head's bias, ET-PINPAD, 224 unspilled
    where four slopes priced 258); traced, never run, on a tile, each slope along its own input alone, or
    where the hook takes no forward derivative (a `jax.custom_vjp`'s) counted varying. None without a hook."""
    if hooks is None or not callable(hooks.score_mod):
        return 0
    params = tuple((param.kind, kept_dtype(param.value.dtype)) for param in hooks.score_params)
    return _varying_slopes(hooks.score_mod, params, graded)


@functools.lru_cache(maxsize=256)
def _varying_slopes(score_mod, params: tuple, graded: tuple[Kind, ...]) -> int:
    """`varying_slopes` of `score_mod` under parameters of `params`' kinds and dtypes in a launch forming
    the gradients of the kinds `graded`."""
    tile = (TILE_SIDES[0], TILE_SIDES[0])
    positions, logits = jax.ShapeDtypeStruct(tile, jnp.int32), jax.ShapeDtypeStruct(tile, jnp.float32)
    values = [_entries(kind, dtype, tile) for kind, dtype in params]

    def slope(place: int, s, q_pos, k_pos, *values):
        # The hook's derivative along its input `place` (0 the logits, else that parameter) alone.
        inputs = [s, *values]

        def along(x):
            taken = [*inputs[:place], x, *inputs[place + 1:]]
            return score_mod(taken[0], q_pos, k_pos, *taken[1:])
        return jax.jvp(along, (inputs[place],), (jnp.ones_like(inputs[place]),))[1]

    count = int(_varies(score_mod, logits, positions, positions, *values))
    traced = jax.make_jaxpr(score_mod)(logits, positions, positions, *values).jaxpr
    formed = [index for index in floating_params(values) if params[index][0] in graded]
    for place in (0, *(1 + index for index in formed)):
        # A `jax.custom_vjp` the input reaches takes no forward derivative: that slope is counted untraced.
        reached = _reaches_custom_vjp(traced, {traced.invars[0 if place == 0 else place + 2]})
        count += reached or _varies(functools.partial(slope, place), logits, positions, positions, *values)
    return count


def _reaches_custom_vjp(jaxpr: jax_core.Jaxpr, starts: set) -> bool:
    """Whether any of `starts`, `jaxpr`'s inputs, reaches a `jax.custom_vjp` call along its tangent
    (`_tangent_reach`)."""
    return _tangent_reach(jaxpr, starts)[0]


def _tangent_reach(jaxpr: jax_core.Jaxpr, starts: set) -> tuple[bool, set]:
    """Whether `starts`, `jaxpr`'s inputs, reach a `jax.custom_vjp` call along their tangent, and the
    outputs they reach: no further than a `stop_gradient`, which passes no tangent; through a call whose one
    sub-program takes the call's operands and gives its outputs in order (a jit's, a custom JVP's), by the
    operands reached into that program and the outputs it reaches back out; through any other holding a
    sub-program, a branch or a loop body not followed operand by operand, wherever that program calls one
    (`_primitives`), else to every output."""
    reached = set(starts)
    for equation in jaxpr.eqns:
        hit = [index for index, var in enumerate(equation.invars)
               if not isinstance(var, jax_core.Literal) and var in reached]
        if not hit or equation.primitive is jax_core.primitives.stop_gradient_p:
            continue
        if equation.primitive is jax_core.primitives.custom_vjp_call_p:
            return True, set()
        inner = list(jax_core.jaxprs_in_params(equation.params))
        if (len(inner) == 1 and len(inner[0].invars) == len(equation.invars)
                and len(inner[0].outvars) == len(equation.outvars)):
            found, outputs = _tangent_reach(inner[0], {inner[0].invars[index] for index in hit})
            if found:
                return True, set()
            reached.update(out for out, inner_out in zip(equation.outvars, inner[0].outvars, strict=True)
                           if inner_out in outputs)
            continue
        if any(jax_core.primitives.custom_vjp_call_p in _primitives(program) for program in inner):
            return True, set()
        reached.update(equation.outvars)
    return False, {var for var in jaxpr.outvars if not isinstance(var, jax_core.Literal) and var in reached}


def _varies(function: Callable, *arguments: jax.ShapeDtypeStruct) -> bool:
    """Whether `function`'s output, traced at `arguments`, depends on any of them."""
    jaxpr = jax.make_jaxpr(function)(*arguments).jaxpr
    varying = set(jaxpr.invars)
    for eqn in jaxpr.eqns:
        if any(not isinstance(var, jax_core.Literal) and var in varying for var in eqn.invars):
            varying.update(eqn.outvars)
    return any(not isinstance(var, jax_core.Literal) and var in varying for var in jaxpr.outvars)


def transposable(hooks: Hooks) -> bool:
    """Whether the score hook and its pullback run on a tile held keys first (`[bk, bq]`, the dK/dV
    program's) as written for the reference's `[bq, bk]` (`layout.transposed`): traced, never run, at every
    tile shape a program takes (`TILE_SIDES`), which no operation it has no rule for refuses
    (`layout.Untransposable`), and at each one entry at a time in the logits and its floating parameters
    (`_entrywise`), as the slopes the backward multiplies its cotangents by assume
    (`attention_gradient.scored`)."""
    if not callable(hooks.score_mod):
        return True
    params = tuple((param.kind, kept_dtype(param.value.dtype)) for param in hooks.score_params)
    return _transposable(hooks.score_mod, params)


@functools.lru_cache(maxsize=256)
def _transposable(score_mod, params: tuple) -> bool:
    """`transposable` of `score_mod` under parameters of `params`' kinds and dtypes."""
    try:
        for tile in ((rows, columns) for rows in TILE_SIDES for columns in TILE_SIDES):
            _score_jaxprs(layout.transposed(score_mod), params, tile, differentiated=True)
    except layout.Untransposable:
        return False
    for tile in ((rows, columns) for rows in TILE_SIDES for columns in TILE_SIDES):
        canonical = _score_jaxprs(score_mod, params, tile, differentiated=False)[0]
        floating = floating_params([_entries(kind, dtype, tile) for kind, dtype in params])
        if not _entrywise(canonical, {0, *(3 + index for index in floating)}, tile):
            return False
    return True


def _entrywise(jaxpr: jax_core.Jaxpr, inputs: set[int], tile: tuple[int, int]) -> bool:
    """Whether every operation of `jaxpr` that reads its `inputs` (by place) acts one entry at a time
    (`dew.nn.programs.entrywise`, a scalar's broadcast, a nested call's own operations so), each value it
    reads of them and each it gives a scalar or the `tile`."""
    tainted = {jaxpr.invars[index] for index in inputs}
    for eqn in jaxpr.eqns:
        read = [index for index, var in enumerate(eqn.invars)
                if not isinstance(var, jax_core.Literal) and var in tainted]
        if not read:
            continue
        # Its other operands (coordinates, integer parameters) any shape that broadcasts against the tile.
        shapes = [getattr(var.aval, "shape", ())
                  for var in (*(eqn.invars[index] for index in read), *eqn.outvars)]
        if any(shape not in ((), tile) for shape in shapes):
            return False
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        if nested:
            # A call or a custom derivative's primal (whose rule `layout.transposed` checks).
            if not eqn.primitive.inlined_call or not all(
                    _entrywise(inner, set(read), tile) for inner in nested):
                return False
        elif not layout.entrywise(eqn):
            return False
        tainted.update(eqn.outvars)
    return True


def _primitives(jaxpr: jax_core.Jaxpr) -> set[jax_core.Primitive]:
    """The operations of `jaxpr` and of every jaxpr inside it (`dew.nn.programs.equations`)."""
    return {eqn.primitive for eqn in equations(jaxpr)}


def _columns(array: jax.Array, unit: int) -> jax.Array:
    """`array` with its last axis padded to whole rows of `unit` entries, as the family's swizzled tiles hold
    them (`Family.column_unit`)."""
    return jnp.pad(array, ((0, 0),) * (array.ndim - 1) + ((0, -array.shape[-1] % unit),))


class _Hooked(NamedTuple):
    """A call's hooks as its programs read them: the score hook's parameters and the output hook's, as their
    kinds lay them out (`Hooks.score_params`', `[H]`, `[B, H, Qp]`, `[B, H, Kp]`, `[B, H, Qp, Kp]` or
    `[B, Qp, H, Dv]`), and the softmax's initial state, its maximum and denominator `[B, H, Qp]` and sums
    `[B, H, Qp, Dv]` (None where it starts from none), and probability dropout's draws (`Dropout`, None where
    the call drops nothing)."""

    score: tuple[jax.Array, ...]
    output: tuple[jax.Array, ...]
    initial: tuple[jax.Array, jax.Array, jax.Array] | None
    drop: Dropout | None = None
    scales: tuple[jax.Array, jax.Array] | None = None
    """A call over quantized keys and values' scales, the keys' and the values' `[B, Hk, Kp]` fp32, one a key
    and key head in its row's key order (`attention_decode.attend`'s `scales`); None for 16-bit keys."""
    magnitudes: jax.Array | None = None
    """A scaled walk's keys' value magnitudes `[B, Hk, Kp]` fp32, each key's largest over the launch's value
    columns (`attention_mosaic_forward._largest`), read a key tile's columns at a time as the keys' scales
    are; None where a launch reads them from its value slots (a cache's walk)."""


def _hooked_of(hooks: Hooks, call: PaddedCall, unit: int) -> _Hooked:
    """`hooks` at the padded call's shapes (`padded_call`), the value widths of the output hook's parameters
    and the initial state's sums in whole rows of `unit` entries (`_columns`), as the launch's values are."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        acc = jnp.pad(call.rows(state.acc), widened)
        initial = call.rows(state.m, -jnp.inf), call.rows(state.total), _columns(acc, unit)
    output = tuple(_columns(value, unit) if param.kind == Kind.OUTPUT else value
                   for value, param in zip(call.output_params, hooks.output_params, strict=True))
    return _Hooked(call.score_params, output, initial,
                   placed_draws(hooks.drop, call.query.shape[1], call.key_rows))


def placed_draws(drop: Dropout | None, queries: int, keys: int) -> Dropout | None:
    """`drop` with its global places (`DropPlaces`, a sequence exchange's slice) at a launch's padded
    `queries` and `keys`, both, a slice holding one whole its local places, so a program reads either's a
    tile at a time beside its spans (`_coords`), every launch of a call's, its forward's and its backward's;
    `drop` as it is where it draws at its local places."""
    if drop is None or (drop.queries is None and drop.keys is None):
        return drop

    def padded(places: jax.Array | None, count: int) -> jax.Array:
        places = jnp.arange(count, dtype=jnp.int32) if places is None else places.astype(jnp.int32)
        return jnp.pad(places, (0, count - places.shape[0]))

    return dataclasses.replace(drop, queries=padded(drop.queries, queries), keys=padded(drop.keys, keys))


def _param(family: mosaic.Family, ref, kind: Kind, row, head, first, like, k_first=None, *,
           keys_first: bool = False):
    """A hook parameter's values for a tile `like` of queries from `first` (and keys from `k_first`, a score
    tile's), as `kind` lays them out: the head's scalar, else the tile's entries (a score tile's `[bq, bk]`,
    an output tile's `[bq, Dv]`) in the accumulator's layout, its queries' and keys' broadcast. A score hook
    reads them and its coordinates as the tile's entries, where the reference and the other mainloops give a
    query's and a key's as `[bq, 1]` and `[1, bk]`: Mosaic GPU's layouts hold no singleton axis an elementwise
    operation takes (jax 0.11.2's layout inference finds none for a `[64, 1]` or `[1, 64]` array's convert or
    compare, a lowering probe for an H100), so a hook that reshapes or indexes an axis fails to lower here,
    never silently; a pointwise hook is the same function of either. A score tile whose rows are its keys
    (`keys_first`, `[bk, bq]`) reads a tile parameter's entries from its keys-first copy `[B, H, Kp, Qp]`."""
    queries, keys = like.shape[::-1] if keys_first else like.shape
    if kind is Kind.HEAD:
        return ref[head]
    if kind is not Kind.OUTPUT:
        row, head = shared_place(ref.shape, row, head)
    # Its block at the tile's place along each of its axes (`Kind.axes`), a tile parameter's keys-first
    # copy's keys before its queries.
    laid = kind.axes[:2] + kind.axes[2:][::-1] if keys_first and kind is Kind.TILE else kind.axes
    places = {"batch": row, "heads": head, "queries": pl.ds(first, queries), "keys": pl.ds(k_first, keys)}
    block = ref.at[axes.placed(laid, places)]
    if kind in (Kind.TILE, Kind.OUTPUT):
        return family.tile_of(block)
    # A query's or a key's along the tile's rows or its columns, as the tile lays its queries.
    along_rows = (kind is Kind.QUERY) != keys_first
    return (algebra.expand(family.rows_of(block), like) if along_rows
            else algebra.across(family.columns_of(block), like))


class _Spans(NamedTuple):
    """A spans call's arrays as its programs read them: each query's interval `start`, `end` `[B, Qp]` (a
    call's runs `[B, R, Qp]`, `_spans_of`), and its block map at the launch's tiles (`attention.block_map`):
    the key tiles each query block reads, `lo`, `hi`, and sees whole, `full_lo`, `full_hi` (`[B, Qp //
    block_q]`), and the query tiles reading each key block, `q_lo`, `q_hi` (`[B, Kp // block_k]`, a key
    block's walk's: None for a launch walking query blocks); with
    documents, the queries' and keys' ids (`[B, Qp]`, `[B, Kp]`) and the blocks' uniform ones (`query_doc`,
    `key_doc`)."""

    start: jax.Array
    end: jax.Array
    lo: jax.Array | None
    hi: jax.Array | None
    full_lo: jax.Array | None
    full_hi: jax.Array | None
    q_lo: jax.Array | None
    q_hi: jax.Array | None
    query_ids: jax.Array | None = None
    key_ids: jax.Array | None = None
    query_doc: jax.Array | None = None
    key_doc: jax.Array | None = None


def reached(spec: algebra.Forward, queries: int) -> int | None:
    """The query and key pairs a call of `queries` queries (unpadded) reaches, by its form: every pair of a
    full call; a causal call's, each query's keys up to its own place (`algebra.seen`'s `k_pos <= q_pos`,
    the first key aligned to the first query); None for a spans call, whose pairs its arrays decide, which
    no launch states before it runs."""
    if spec.form == "full":
        return queries * spec.keys
    if spec.form == "causal":
        return sum(min(spec.keys, row + 1) for row in range(queries))
    return None


class TileWalk(NamedTuple):
    """A launch's walk of its tiles as the work it issues reads it (`issued`): its padded `extents` (its
    arrays' query rows and keys, whole programs of both), its tiles' `blocks` (a consumer's query block and
    the key block), the tiles a program holds on its walk's axis (`holders`: its consumers' query blocks, or
    the key blocks of a walk over query tiles, `keyed`), and whether each of them walks every tile any of its
    program's holders reads (`common`: `Family.walk_own` on a layout whose consumers feed their rings, a
    shared dQ's or staged logits' dK/dV walk), a tile its own does not see masked whole."""

    extents: tuple[int, int]
    blocks: tuple[int, int]
    holders: int = 1
    keyed: bool = False
    common: bool = False


def issued(spec: algebra.Forward, tile_walk: TileWalk) -> int | None:
    """The query and key pairs a launch walking `tile_walk` computes: each tile holding a pair its call's form
    reaches, whole, the first key aligned to the first query, as its program reads them (a query block's key
    tiles up to its diagonal, `_Program.walked`, or a key block's query tiles from it, `_Program.readers`):
    every padded row's (a padded consumer walks its keys as a real one does), a causal diagonal tile's
    masked part computed with the rest (on an H100, the tensor pipes' counts of a causal call at 2048
    queries and 128-row tiles: 6.25% past its reached pairs, row 3948), and a common walk's every tile for
    each holder, one wholly past the keys too (`algebra.Forward.walked_keys`); None for a spans call."""
    if spec.form == "spans":
        return None
    (queries, keys), (block_q, block_k) = tile_walk.extents, tile_walk.blocks
    query_tiles, causal = queries // block_q, spec.form == "causal"
    seen = min(-(-spec.keys // block_k), keys // block_k)
    if tile_walk.keyed:
        # A key block's readers, from its diagonal where causal; a block wholly past the keys has none.
        firsts = [min(tile * block_k // block_q, query_tiles) if causal else 0 for tile in range(seen)]
        walks = [query_tiles - first for first in firsts] + [0] * (keys // block_k - seen)
    else:
        walks = [min(-(-(tile + 1) * block_q // block_k), seen) if causal else seen
                 for tile in range(query_tiles)]
    if tile_walk.common:
        holders = tile_walk.holders
        programs = (walks[first:first + holders] for first in range(0, len(walks), holders))
        walks = [len(program) * max(program) for program in programs]
    return block_q * block_k * sum(walks)


def stated(spec: algebra.Forward, batch: int, heads: int, widths: int, *arrays,
           tile_walk: TileWalk) -> pl.CostEstimate | None:
    """A launch's cost as it states it to XLA and to Dew's FLOP count (`dew.telemetry.instrumentation`), the
    hardware's work: a multiply-add a pair its walk (`tile_walk`) issues (`issued`), head and row for each of
    `widths` product columns the launch issues, its recomputed ones (a backward's `S`) and each part a
    cotangent's product takes (`part_count`) among them, a transcendental an issued pair, and the bytes of
    each of `arrays` once; None for a spans call, whose pairs no launch knows (`reached`): its FLOPs count as
    unsized, never as 0. The model's share of it the launch declares apart (`modelled`)."""
    pairs = issued(spec, tile_walk)
    if pairs is None:
        return None
    return estimate(2 * batch * heads * pairs * widths, *arrays, transcendentals=batch * heads * pairs)


def modelled(spec: algebra.Forward, batch: int, heads: int, queries: int, widths: int) -> int | None:
    """The model's share of a launch's work (`stated`'s), which its launch declares to Dew's FLOP count
    (`Family.kernel`'s `model_flops`): `widths` product columns of the model's products it owns (the forward's
    `Q K^T` and `P V`, the backward's two transposes of each, as the library's fused attention counts them),
    a recomputed product or a cotangent's later parts the hardware's alone; None for a spans call."""
    pairs = reached(spec, queries)
    return None if pairs is None else 2 * batch * heads * pairs * widths


def part_count(dtype, accurate: bool) -> int:
    """The products a cotangent's product with an operand of `dtype` takes (`algebra.cotangent_parts`): one in
    the fast form's bf16 and in fp32, three in the accurate form's bf16, two in fp16's either form."""
    held = jax.eval_shape(lambda cotangents: algebra.cotangent_parts(cotangents, dtype, accurate=accurate),
                          jax.ShapeDtypeStruct((8, 8), jnp.float32))
    return len(held)


def _spans_of(spec: algebra.Forward, call: PaddedCall, block_q: int, block_k: int, *, readers: bool = False
              ) -> _Spans | None:
    """A spans call's `_Spans` at `block_q` by `block_k` tiles over its padded keys (None for a full or
    causal call, whose programs take their tiles from their places), with each key block's query readers
    where a launch walks key blocks (`readers`: the backward's; computed for no other, two reductions a
    decode step's forward ran for nothing, ET-DECODE8-h100). A call's runs (`KeySpans`) lie run by
    run, `[B, R, Qp]`, so a program reads each run's bounds as a row (`_coords`); its blocks walk their
    runs' envelopes (`attention.block_map`), each tile masked by the union of its queries' runs
    (`algebra.seen`)."""
    if spec.form != "spans":
        return None
    if call.per_span > 1:
        # Each interval a position's packed rows (`KeySpans.per_span`): its programs read them by position
        # and form their tiles' bounds themselves (`_Program.reach`, `_Program.bounds`), no block map in XLA.
        return _Spans(call.start, call.end, None, None, None, None, None, None)
    blocks = block_map(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key_rows),
                       block_q, block_k)
    start, end = (jnp.moveaxis(bound, 2, 1) if bound.ndim == 3 else bound for bound in (call.start, call.end))
    return _Spans(start, end, blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi,
                  blocks.q_lo if readers else None, blocks.q_hi if readers else None,
                  call.query_ids, call.key_ids, blocks.query_doc, blocks.key_doc)


@dataclasses.dataclass(frozen=True)
class _Program[T: (Tiles, BackwardTiles)]:
    """A program's static geometry: its family's bundle, its tiles, the call's key tiles (`steps`), the slots
    a stream holds, its key count, whether it is causal, its query count (`algebra.Forward.queries`; a
    tile holding padding rows is never whole) and the slices each query tile's keys are cut into (`splits`:
    a split-KV launch's, each slice a program of its own on the grid's query-tile axis, `split`)."""

    family: mosaic.Family
    tiles: T
    steps: int
    stages: int
    keys: int
    causal: bool
    queries: int | None = None
    splits: int = 1
    per_span: int = 1
    """The queries each of a spans call's intervals holds (`KeySpans.per_span`): more than one, its programs
    read the intervals by position (`reach`, `bounds`)."""
    parts: int = 1
    """The bf16 parts each of an fp32 call's tiles is staged in, one above the other (`MosaicForward.parts`):
    its products their parts' (`parted`)."""

    def parted(self, tile, rows: int):
        """`tile` as the parts its products take (`operands.Parts`, each `rows` rows of it, an fp32 tile's),
        or itself where the call's operands are 16-bit."""
        if self.parts == 1:
            return tile
        return operands.Parts(tuple(tile.at[pl.ds(part * rows, rows)] for part in range(self.parts)),
                              jnp.float32)

    def split(self, *, fresh: bool = False):
        """This program's slice of its query tile's keys (`splits`), 0 of one; its grid coordinate read where
        it is used where `fresh` (`Family.program_index`)."""
        if self.splits == 1:
            return jnp.int32(0)
        return lax.rem(self.family.program_index("q_tiles", fresh=fresh), self.splits)

    def padded(self, first) -> jax.Array | bool:
        """Whether the query tile from `first` holds padding rows past the call's queries, some or all of
        them (a call padded to whole programs)."""
        if self.queries is None:
            return False
        return first + self.tiles.block_q > self.queries

    def query_tile(self, count: int, *, fresh: bool = False):
        """This program's query tile of the grid's `count`: a causal call's longest first (its last query
        tiles see the most keys), so the grid's tail holds its shortest programs, as FlashAttention-3's
        scheduler orders them; a full call's in order. Its grid coordinate read where it is used where
        `fresh` (`Family.program_index`). A grid of one tile reads none: read afresh, a decode step's split
        coordinate was divided into its one tile's address on an A100 and its offsets spilled 32 bytes, a
        coordinate read as it is folded (row 5466). A causal call's reversal of a power of two tiles is the
        index's complement (`count - 1 - index`, the same tile), which ptxas forms again where it is used:
        subtracted, a B200's forward at width 256 held it and spilled 8 bytes, complemented none (row
        5466)."""
        if count == 1:
            return jnp.int32(0)
        index = self.family.program_index("q_tiles", fresh=fresh)
        if self.splits > 1:
            index = lax.div(index, self.splits)
        if not self.causal:
            return index
        return index ^ (count - 1) if count & (count - 1) == 0 else count - 1 - index

    def __post_init__(self):
        if not whole_tiles(self.family, self.tiles):
            raise ValueError(f"{self.tiles}: query and key tiles of whole {self.family.rows} rows and "
                             f"{most_consumers(self.family, self.tiles)} consumers at most "
                             f"({self.family.name})")

    def walked(self, first):
        """The key tiles a query block from `first` reads: those holding the call's keys (none wholly past
        them, `algebra.seen`), up to its diagonal where causal (`attention.key_tiles`), and no more than the
        call's (`steps`)."""
        _, hi = key_tiles("causal" if self.causal else "full", first, self.tiles.block_q, self.tiles.block_k,
                          self.keys)
        return jnp.minimum(hi, self.steps).astype(jnp.int32)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: those wholly within the call's keys
        (the padded call's last tiles may all be past them), and below its diagonal."""
        whole = jnp.int32(self.keys // self.tiles.block_k)
        if self.causal:
            whole = jnp.minimum(lax.div(first, self.tiles.block_k), whole)
        return jnp.minimum(jnp.maximum(whole, 0), walked)

    def reach(self, first, spans: _Spans | None, row):
        """The key tiles a query block from `first` reads, `[lo, hi)`, and whether it sees tile `index` whole
        (`whole(index)`, a tile that takes no mask): a full or causal call's from its places (`walked`,
        `whole`), a spans call's from its block map (`spans`, batch row `row`), whole where the ids of
        both blocks agree too."""
        if spans is None:
            walked = self.walked(first)
            # A block holding padding rows is whole nowhere: its padding sees no key (`algebra.seen`).
            count = jnp.where(self.padded(first), 0, self.whole(first, walked))
            return *self._sliced(jnp.int32(0), walked), lambda index: index < count
        if spans.lo is None or spans.hi is None or spans.full_lo is None or spans.full_hi is None:
            # Read by position (`_spans_of`): no block map.
            return self._positions_reach(first, spans, row)
        block = lax.div(first, self.tiles.block_q)
        at = functools.partial(self._entry, row=row)
        full_lo, full_hi = at(spans.full_lo, block), at(spans.full_hi, block)

        def whole(index):
            inside = (index >= full_lo) & (index < full_hi)
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = at(spans.query_doc, block)
            return inside & (doc != 0) & (doc == at(spans.key_doc, index))

        return *self._sliced(at(spans.lo, block), at(spans.hi, block)), whole

    def _positions(self, first, spans: _Spans, row) -> list[tuple[jax.Array, ...]]:
        """The positions a query block from `first` holds rows of, in a call whose intervals each hold
        `per_span` queries (`KeySpans.per_span`): each one's index, its interval `[start, end)` (batch row
        `row`'s, read once) and whether the block holds its rows; as many as the block can hold,
        statically."""
        positions = spans.start.shape[1]
        held = min(positions, self.tiles.block_q // self.per_span + 1)
        at = lax.div(first, self.per_span)
        found = []
        for offset in range(held):
            position = at + offset
            # A position past the call's holds no row: read its last one's, unused.
            index = jnp.minimum(position, positions - 1)
            holds = (position < positions) & (position * self.per_span < first + self.tiles.block_q)
            found.append((position, self._entry(spans.start, index, row=row),
                          self._entry(spans.end, index, row=row), holds))
        return found

    def _positions_reach(self, first, spans: _Spans, row):
        """`reach`'s for a call whose intervals each hold `per_span` queries: the key tiles from the first any
        of its rows' intervals starts in to the last any ends in, none seen whole where the block holds a
        padding row (a decode step's few rows of a tile), else those every one of its intervals covers."""
        block_k, big = self.tiles.block_k, jnp.iinfo(jnp.int32).max
        lo, hi, full_lo, full_hi = jnp.int32(big), jnp.int32(0), jnp.int32(0), jnp.int32(big)
        for _, start, end, holds in self._positions(first, spans, row):
            seen = holds & (end > start)
            lo = jnp.where(seen, jnp.minimum(lo, lax.div(start, block_k)), lo)
            hi = jnp.where(seen, jnp.maximum(hi, pl.cdiv(end, block_k)), hi)
            full_lo = jnp.where(holds, jnp.maximum(full_lo, pl.cdiv(start, block_k)), full_lo)
            full_hi = jnp.where(holds, jnp.minimum(full_hi, lax.div(end, block_k)), full_hi)
        lo = jnp.minimum(lo, hi)
        if self.padded(first) is not False:
            full_hi = jnp.where(self.padded(first), full_lo, full_hi)

        def whole(index):
            return (index >= full_lo) & (index < full_hi)

        return *self._sliced(lo, hi), whole

    def bounds(self, first, spans: _Spans, row) -> tuple[tuple, jax.Array | None]:
        """A query block's intervals and its ids as `_coords` reads them (`_query_bounds`), read once before
        its walk; a call whose intervals each hold `per_span` queries its rows' from their positions', a
        padding row's empty."""
        family, count = self.family, self.tiles.block_q
        if spans.lo is not None:
            return _query_bounds(family, spans, row, first, count)
        place = lax.div(first + family.iota((count,), 0), self.per_span)
        start = end = place * 0
        for position, low, high, holds in self._positions(first, spans, row):
            mine = (place == position) & holds
            start, end = jnp.where(mine, low, start), jnp.where(mine, high, end)
        return ((family.as_row(start), family.as_row(end)),), None

    def _entry(self, ref, index, *, row):
        """A block map's entry `ref[row, index]` as the family reads a launch's scalar (`Family.scalar_of`: a
        GPU thread's load, a TPU core's copy into its scalar memory, whose operands stay in HBM, where a load
        lowers on no core: a v5e's documents call refused to lower, "ANY memory space can only be accessed
        using async_copy")."""
        return self.family.scalar_of(ref.at[row, pl.ds(index, 1)])

    def _sliced(self, lo, hi) -> tuple[jax.Array, jax.Array]:
        """`[lo, hi)` key tiles, this program's slice of them (`split`): the `splits` contiguous slices of a
        query tile's tiles, as long as the others but for rounding, partition them; its whole range of one."""
        if self.splits == 1:
            return lo, hi
        split, count = self.split(), hi - lo
        return lo + lax.div(count * split, self.splits), lo + lax.div(count * (split + 1), self.splits)

    def readers(self, k_first, q_steps: int, spans: _Spans | None, row):
        """The query tiles reading a key block from `k_first`, `[lo, hi)` of the call's `q_steps`, and
        whether tile `index` sees it whole (`whole(index)`): a full or causal call's from its places, from
        its diagonal and none for a block wholly past the keys; a spans call's from its block map, whole
        where the ids of both blocks agree too (`reach`)."""
        if spans is None:
            start, end = self._whole_run(k_first, q_steps)
            return self._first_reader(k_first, q_steps), jnp.int32(q_steps), lambda index: (
                (index >= start) & (index < end))
        block = lax.div(k_first, self.tiles.block_k)
        full_lo, full_hi = spans.full_lo, spans.full_hi
        assert full_lo is not None and full_hi is not None, "a key block's walk reads its block map"
        at = functools.partial(self._entry, row=row)

        def seen_whole(index):
            inside = (block >= at(full_lo, index)) & (block < at(full_hi, index))
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = at(spans.key_doc, block)
            return inside & (doc != 0) & (doc == at(spans.query_doc, index))

        assert spans.q_lo is not None and spans.q_hi is not None, "a key block's walk reads its readers"
        return at(spans.q_lo, block), at(spans.q_hi, block), seen_whole

    def segmented(self, step: Callable, run: tuple[jax.Array, jax.Array] | None, lo, hi) -> tuple:
        """A key block's walk of query tiles `[lo, hi)` as `Family.walk`'s segments, `step` each: where its
        whole tiles run unbroken (`run`, `whole_run`), those one segment of their own, unmasked, and the
        tiles around them masked, so no step chooses at run time, whose two branches' registers join
        (FlashAttention 3's masked and unmasked iterations); else one segment choosing a tile at a time."""
        if run is None:
            return ((step, lo, hi),)
        start, end = (jnp.clip(bound, lo, hi) for bound in run)
        return ((functools.partial(step, masked=True), lo, start),
                (functools.partial(step, masked=False), start, end),
                (functools.partial(step, masked=True), end, hi))

    def whole_run(self, k_first, q_steps: int, spans: _Spans | None) -> tuple[jax.Array, jax.Array] | None:
        """The query tiles `[a, b)` that see the key block from `k_first` whole (`readers`' `whole`), one run
        of them: a causal call's, from its diagonal to its first tile holding padding rows; None for a spans
        call, whose whole tiles need not run unbroken, and for a full call, whose every tile but a padded
        edge's is whole (ET-SEGS-h100: the run its own segment, 2% slower a full launch at width 64, 7%
        faster a causal one at Qwen3-8B's)."""
        return None if spans is not None or not self.causal else self._whole_run(k_first, q_steps)

    def _first_reader(self, k_first, q_steps: int) -> jax.Array:
        """The first query tile that sees the key block from `k_first` in a full or causal call: every one,
        or those from its diagonal (a causal call of more keys than queries leaves its last keys unseen)."""
        lo = jnp.minimum(lax.div(k_first, self.tiles.block_q), q_steps) if self.causal else jnp.int32(0)
        return jnp.where(k_first >= self.keys, q_steps, lo)

    def _whole_run(self, k_first, q_steps: int) -> tuple[jax.Array, jax.Array]:
        """`whole_run`'s of a full or causal call: from past the block's diagonal (none if the block holds
        padding keys) to the first query tile holding padding rows, which sees no block whole (`reach`)."""
        block_q, block_k = self.tiles.block_q, self.tiles.block_k
        lo = self._first_reader(k_first, q_steps)
        start = jnp.minimum(pl.cdiv(k_first + block_k - 1, block_q), q_steps) if self.causal else lo
        start = jnp.where(k_first + block_k > self.keys, q_steps, jnp.maximum(start, lo))
        end = jnp.int32(q_steps if self.queries is None else min(self.queries // block_q, q_steps))
        return start, jnp.maximum(end, start)


def _per_span(spec: algebra.Forward, spans: _Spans | None) -> int:
    """The queries each of a launch's spans' intervals holds (`KeySpans.per_span`): its queries over its
    positions where its programs read them by position (`_spans_of`), else 1."""
    if spans is None or spans.lo is not None:
        return 1
    assert spec.queries is not None  # a call read by position states its queries (`_prepared`)
    return spec.queries // spans.start.shape[1]


def _streamed(reaches: list) -> tuple[jax.Array, jax.Array]:
    """The tiles a program streams: those any of its consumers' `reaches` (`[lo, hi)` each) reads."""
    return (functools.reduce(jnp.minimum, [lo for lo, _ in reaches]),
            functools.reduce(jnp.maximum, [hi for _, hi in reaches]))


def _coords(family: mosaic.Family, spans: _Spans | None, row, first, k_first, shape: tuple[int, int],
            q_pos=None, columns=None, *, keys_first: bool = False, read: tuple[bool, bool] = (False, False),
            draw: Draw | None = None, tile_keys: int | None = None, held: tuple | None = None
            ) -> algebra.Coords:
    """A tile's coordinates (`algebra.Coords`): its queries' and keys' places from `first` and `k_first`, the
    tile's entries in the accumulator's layout (`q_pos` the query places and `columns` the tile's key
    offsets, where the caller holds them; a hook reads them so, `_param`), and a spans call's queries'
    intervals and, with documents, both blocks' ids, and a dropping call's `draw`. A tile whose rows are its
    keys (`keys_first`, `shape` `[bk, bq]`) holds its queries' vectors in a column's layout and its keys' in
    a row's: each one's places a vector broadcast where the step surely reads them (`read`, its queries' and
    its keys', `algebra.places_read`), a tile otherwise. A part of the walk's tile (a consumer's share of
    its keys) states the whole tile's keys (`tile_keys`, `algebra.Coords.key_count`), so a part past the
    call's keys is masked as its tile is. `held` (`_query_bounds`) its queries' intervals and ids a walk
    read once, before its loop, where it holds them."""
    query_axis, key_axis = (1, 0) if keys_first else (0, 1)

    def places(start, axis: int) -> jax.Array:
        # A tile's places along `axis` from a row's or a column's vector broadcast: each element's index its
        # lane's base and an immediate. An iota of the whole tile computes each column's from its place (a
        # remainder by the tile's width, seven instructions at 192), chains LLVM hoists out of the walk and
        # ptxas holds across it (the held-term census: 45-127 such integers live at the peak, D64's 64 x 192
        # forward spilling 344 bytes, D128's training 64 x 64 72); a row's needs no remainder.
        return lax.broadcast_in_dim(start + family.iota((shape[axis],), 0), shape, (axis,))

    # Each one's places a vector broadcast where the step reads them (a row's keys first, a column's queries
    # first); a tile otherwise (Mosaic GPU keeps an unread vector's iota without its layout, which its
    # inference cannot place, row 2977), and the queries' keys first (a query's column vector beside the
    # tile finds no layout).
    queries_read, keys_read = read
    if q_pos is None:
        q_pos = (places(first, query_axis) if queries_read and not keys_first
                 else first + family.iota(shape, query_axis))
    key_start = key_offsets = None
    if columns is not None and not keys_first:
        k_pos = k_first + columns
    elif keys_read and not keys_first:
        # A row's keys as the start its lane decides and the offsets its registers do (`Coords.key_start`).
        base, key_offsets = family.column_offsets(shape)
        key_start = k_first + base
        k_pos = key_start + key_offsets
    else:
        k_pos = places(k_first, key_axis) if keys_read else k_first + family.iota(shape, key_axis)
    keys = pl.ds(k_first, shape[key_axis])
    of_keys = family.rows_of if keys_first else family.columns_of
    drop_at = None
    if draw is not None and draw.queries is not None and draw.keys is not None:
        # A draw at a slice's global places (`_placed_draws`): its queries' and keys' a tile's vectors, which
        # the draw then reads in place of its tables (`algebra.drawn`).
        of_queries = family.columns_of if keys_first else family.rows_of
        drop_at = of_queries(draw.queries.at[pl.ds(first, shape[query_axis])]), of_keys(draw.keys.at[keys])
        draw = dataclasses.replace(draw, queries=None, keys=None)
    if spans is None:
        return algebra.Coords(q_pos, k_pos, drop=draw, keys_first=keys_first, tile_keys=tile_keys,
                              key_start=key_start, key_offsets=key_offsets, drop_at=drop_at)
    bounds, query_ids = held if held is not None else _query_bounds(family, spans, row, first,
                                                                    shape[query_axis], keys_first=keys_first)
    ids = None
    if query_ids is not None and spans.key_ids is not None:
        ids = query_ids, of_keys(spans.key_ids.at[row, keys])
    return algebra.Coords(q_pos, k_pos, bounds, ids, draw, keys_first=keys_first, tile_keys=tile_keys,
                          drop_at=drop_at)


def _query_bounds(family: mosaic.Family, spans: _Spans, row, first, count: int, *, keys_first: bool = False
                  ) -> tuple[tuple, jax.Array | None]:
    """A query block's intervals (one vector of bounds a call's run, `_spans_of`) and its ids, read from the
    spans' arrays: what `_coords` reads a step, which a walk reads once before its loop where its steps
    share its queries (every load in the loop waits its global latency, ET-DECPROF2-h100: a decode step's
    walk stalled on its long scoreboard ten times its barrier's)."""
    rows = pl.ds(first, count)
    of_queries = family.columns_of if keys_first else family.rows_of

    def bounds_of(*run):
        return of_queries(spans.start.at[(row, *run, rows)]), of_queries(spans.end.at[(row, *run, rows)])

    runs = [(run,) for run in range(spans.start.shape[1])] if spans.start.ndim == 3 else [()]
    ids = None
    if spans.query_ids is not None and spans.key_ids is not None:
        ids = of_queries(spans.query_ids.at[row, rows])
    return tuple(bounds_of(*run) for run in runs), ids


def _rows(family: mosaic.Family, ref, row, head, first, count: int):
    """A query block's per-row statistic `[count]` of a `[B, H, Q]` array, in a row's layout."""
    return family.rows_of(ref.at[row, head, pl.ds(first, count)])


def _views_of(refs: dict, row, head) -> dict:
    """The streams' views by name (`Family.opened`, `Family.stream`, `rings.viewed`): each `[B, S, H, D]`
    operand's `[S, D]` at batch row `row` and head `head`, step `j` its `j`-th block of rows."""
    return {name: (ref.at[row, :, head], lambda j: (j, 0)) for name, ref in refs.items()}


def _fused_held(logits: tuple, dtype, *, dropped: bool) -> list:
    """The tiles a fused backward launch holds beside both products and its pullback (`backward_registers`):
    a dropping call's multipliers (`algebra.drawn`); fp16's parts formed at the running scale beside the tile
    and its scaled copy, the other left operand fp32 still live (`algebra.running_parts`: a 128-wide fused
    launch 281 registers live at its peak, the scale's exp2, and 2240 bytes spilled where one tile priced it
    at 246, et-c7512). A score hook's output and slopes are its pullback's, held once: a bf16 step on mma.sync
    issues its later products in loops of their own (`Issuer.looped_products`), so none holds another's
    fragments, an overlap a second count of the slopes had covered (row 5477); an fp16 step's parts are priced
    here, and a family reading its operands from shared memory holds its narrowed weights beside the pullback
    (`backward_registers`)."""
    return (([logits] if dropped else [])
            + ([logits] * 3 if jnp.dtype(dtype) == jnp.dtype(jnp.float16) else []))


def _hook_live(family: mosaic.Family, block_q: int, block_k: int, params: tuple, *, grads: str | None = None,
               keys_first: bool = False) -> tuple[list, int]:
    """A step's score-hook parameters (`params`, each its `Kind` and dtype), live beside its logits: a
    TILE's `[bq, bk]` tile, a KEY's vector in the accumulator's column layout and a QUERY's in its row
    layout (`Family.column_registers`, `row_registers`); a HEAD's scalar no tile; each in the dtype its kernel
    loads it (`kept_dtype`). With `grads`, a backward launch's gradients of them too, fp32, where it carries
    them: a KEY's across its walk in the dK/dV launch (`grads="keys"`), a QUERY's in the dQ launch
    (`grads="queries"`, `DQ_GRADS`); a TILE's gradient is the step's score cotangents through the hook's
    slope, formed in their place and stored, no tile beside them (an A100's dK/dV under a tile bias at
    64 x 64, width 64: 255 registers unspilled and none spilled at a cap of 247, where its tile beside them
    priced 280). A `keys_first` step's
    tile lays its keys along its rows, so a KEY's vector is in the row layout and a QUERY's in the column.
    Returns the step's tiles and the vectors' registers."""
    keys, queries = ((family.row_registers, family.column_registers) if keys_first
                     else (family.column_registers, family.row_registers))
    step, vectors = [], 0
    for kind, dtype in params:
        if kind is Kind.TILE:
            step.append((block_q, block_k, dtype))
        elif kind is Kind.KEY:
            vectors += keys(block_k, dtype) + (keys(block_k, jnp.float32) if grads == "keys" else 0)
        elif kind is Kind.QUERY:
            vectors += queries(block_q, dtype) + (queries(block_q, jnp.float32) if grads == "queries" else 0)
    return step, vectors


PREDICATE_REGISTERS = 7
"""The predicate registers a GPU thread holds (P0 to P6): a step's predicates past them ptxas saves to
general registers (P2R) until the selects that read them (`_anchor_registers`)."""


def _anchor_registers(family: mosaic.Family, block_q: int, block_k: int) -> int:
    """The registers a queries-first backward step's anchor select holds (`at_anchor`,
    `attention_backward_algebra.softmax_cotangents`) across the step, from the compares that form it to the
    selects that read it: its predicates packed a bit an entry; a row pass's of them, a role's rows, pending,
    all but those its predicate registers hold (`PREDICATE_REGISTERS`); and the places it compares, the keys'
    column vector and the anchor keys' row vector, int32. The SASS of dQ under ALiBi at 64 x 64, width 64, on
    an A100: 32 `key != anchor` compares a thread packed into one word, 25 saved by P2R, restored by ISETP and
    FSEL, 43 words at its need (200); the accurate form's at 64 x 64 and width 128 on an H100 1, 24 and 18,
    its need 255; dQ at 128 x 64, width 64, under sinks 2, 27 and 20. Priced beside every phase: beside its
    products alone, the accurate form's dQ under a temperature took 64 x 64 at width 64, where it spills 232
    bytes (ET-PINBWD), its hook's slopes live across the select."""
    entries = block_q * block_k // resources.THREADS
    pending = max(0, family.role_rows * block_k // resources.THREADS - PREDICATE_REGISTERS)
    places = family.column_registers(block_k, jnp.int32) + family.row_registers(block_q, jnp.int32)
    return -(-entries // 32) + pending + places


def _mask_places(family: mosaic.Family, block_q: int, block_k: int) -> list:
    """The int32 tiles a step's masks hold where its key tile's indices do not fold (`Family.folds_indices`):
    the queries' and the keys' places and their difference, each entry's formed apart (an A100's 64 x 96 at
    width 64 under a score hook: ptxas 255 registers and 16 to 56 bytes spilled where the two places priced
    240; at 64 x 192 the mask's two subtractions and compares an entry, `folds_indices`); none where they
    fold."""
    return [] if family.folds_indices(block_k) else [(block_q, block_k, jnp.int32)] * 3


def _converted(block_k: int, *widths: int) -> list[list[tuple]]:
    """A quantized call's phases converting a slot of codes beside its sums (`Family.converted`): its codes
    read in the accumulator's layout, biased to int32 and taken to fp32, a tile of each live at once, the
    keys' `[bk, D]` and then the values' `[bk, Dv]` (an A100's int8 decode at width 128, 64-key tiles:
    priced 108 without them, ptxas held it at 160, FS undercount gate4)."""
    return [[(block_k, width, jnp.int32), (block_k, width, jnp.float32)] for width in dict.fromkeys(widths)]


def forward_registers(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, residuals: bool, *,
                       natural: bool = False, params: tuple = (), width: int | None = None,
                       parts: int = 1, scaled: bool = False, codes: bool = False, positions: int = 0,
                       per_span: int = 1) -> int:
    """The registers a forward consumer holds at `tiles` (`resources.live_registers`): its weighted values'
    sums `[bq, Dv]` fp32 carried, and a step's logits fp32 beside its weights in the operands' dtype. A
    training forward (`_trained`, at its anchor known) holds the other keys' weights beside them too, in the
    operands' dtype, and the key indices int32 its anchor test compares, a thread's one per column its
    fragment holds, which its fragment's two rows share (an MMA accumulator's layout: half the tile's words;
    TWOPASS5-h100: uncounted, 64 x 128 on two consumers derived and spilled, 6.6 ms against 64 x 64's 1.06);
    its peak walk holds less. The `natural` form's exponential (`algebra.natural_exp`) holds at its peak the
    scaled logits' rounded product and its residual (`two_product`), two fp32 tiles, beside the product's
    bit split, an int32 tile, the logits themselves read entry by entry into them (ET-PINSOFT: width 256
    under a softcap at 64 x 64, ptxas 255 unspilled where three fp32 tiles priced 260; the IR's breadth-first
    live set holds the third, ET-HOOKBREAK); a training walk's then its power's integral part built in bits,
    two int32 tiles beside its weights (ET-HOOKBREAK: a natural training walk 174 at width 64, its fold's
    110 and 64 int32). A score hook's parameters are live beside the logits it reads (`_hook_live`), a phase
    of their own: the hook adds them into its logits, dead past it (ET-PINFWD: key-bias and tile-bias at
    64 x 64 on two consumers, width 64, ptxas 168 unspilled where they priced beside the weights 180 and
    196). Where its family reads operands from registers, its products' fragments beside their tiles
    (`_products`): the logits' product of `width`-deep queries and keys, and the values' product of the
    weights. A key tile whose indices do not fold holds its masks' places (`_mask_places`). An fp32 call's
    `parts` (`MosaicForward.parts`) issue its logits' products' terms into one accumulator and form its
    weights' parts one at a time from its fp32 weights, held whole across them, its remainder fp32 and the
    part in the operands' dtype beside them (`attention_mosaic_forward._parted_products`: priced as 16-bit
    weights, the fp32 forward at 64 x 64 on three consumers, width 64, counted 100 where ptxas needed 152,
    and spilled 16 bytes at the 128 three consumers leave, et-fp32model). A `scaled` walk's
    (`MosaicForward.scales_in_kernel`) holds beside its step its weights at the scale and its small weights
    raised, fp32, the small ones' parts, and a part's product's own accumulator `[bq, Dv]` fp32
    (`algebra.at_scale`, `attention_mosaic_forward._accumulated`): priced as one fp32 tile and the
    accumulator, the H100's scaled forward at width 128 spilled 232 bytes (et-scaled-b2); its natural form's
    exponential raises the small ones within the step, beside its tiles and the weights at the scale (priced
    apart from them, its rewalk needed 248 of the 228 priced and spilled 44 bytes, et-scaledneed; an fp32
    call's beside its logits alone, its parts formed after it: priced beside the parts' tiles, an H100's
    scaled fp32 forward at width 64 under a softcap priced 276 and needed 184, et-sneedfp32); an fp32
    call's parts walk its weights at the scale its own fp32 weights (priced apart, its decode step's 292
    refused the 64 x 64 tile ptxas holds in 240, et-fp32decneed, and one fp32 tile fewer 212, et-parity-w); a
    family whose products read registers its values' fragments beside the parts' products (unpriced, an
    A100's at width 128 spilled 128 bytes at 255).
    """
    block_q, block_k = tiles.block_q, tiles.block_k
    hooked, vectors = _hook_live(family, block_q, block_k, params)
    logits, weights = (block_q, block_k, jnp.float32), (block_q, block_k, dtype)
    bits = (block_q, block_k, jnp.int32)
    step = [logits, weights, *([weights] if residuals else [])]
    if parts > 1:
        # Its weights fp32, their remainder fp32 and the part taken from it (`_parted_products`).
        step = [logits, logits, logits, weights, *([weights] if residuals else [])]
    if residuals:
        step.append((block_q // 2, block_k, jnp.int32))
        if family.operands_in_registers:
            # Its values' block's fragments whole beside the step, where the family's products read
            # registers: an A100's training walk at 64 x 128, width 64, loads the step's query, key and
            # value blocks up front, their fragments live at once beside its logits (the liveness pass at
            # its peak, census80b), ptxas 255 registers and 48 to 56 bytes spilled where the step priced 252.
            step.append((block_k, value_width * family.rhs_copies, dtype))
    masks = _mask_places(family, block_q, block_k)
    step += masks
    # The step's tiles the exponential's raised small weights are formed beside: a 16-bit call's step (an
    # inference walk's logits and weights and its masks' places; a training walk's spilled 224 to 248 bytes
    # at 238 of 240 priced so, et-scaledneed5); an fp32 call's parts walk its logits alone, its weights'
    # parts formed after the exponential (`_parted_products`).
    scaled_terms = []
    core = (list(step) if residuals else [logits, weights, *masks]) if parts == 1 else [logits]
    if scaled:
        # Its small weights raised from their exponents, fp32, and a part's product's own accumulator beside
        # the walk's while it issues (`operands.Issuer.summed`), and its weights at the scale fp32 beside
        # them; an fp32 call's parts walk forms the small ones' parts in turn as it forms theirs
        # (`_parted_products`; its weights at the scale held as the step's fp32 weights, an H100's fp32 decode
        # step at 64 x 64 priced 212 where ptxas needed 240, et-parity-w); a 16-bit call's holds the small
        # ones' three parts (`algebra.small_parts`), a family whose products read registers its values'
        # block's fragments whole beside them, which the weights' product and each part's read.
        accumulator = (block_q, value_width, jnp.float32)
        # tcgen05's fp32 parts walk reads each part's product back beside them (priced below), its weights at
        # the scale held no apart (a B200's at 64 x 64 over 16 columns needed 224).
        scaled_terms = [logits] if parts > 1 and scopes_products(family) else [logits, logits]
        step += [*scaled_terms, accumulator]
        if parts == 1:
            step += [weights] * (3 if jnp.dtype(dtype).itemsize == 2 else 1)
            if family.operands_in_registers and not residuals:
                step.append((block_k, value_width * family.rhs_copies, dtype))
        if scopes_products(family) and parts > 1:
            # An fp32 call's small part's product read back from tensor memory in its layout, converted to the
            # walk's and taken back by its power (`algebra.restored`) beside its parts' step (tcgen05's
            # chunks, `Tcgen05ChunksIssuer.summed`): unpriced, a B200's scaled fp32 forward at width 128
            # priced 220 at 64 x 64 over 32 columns and spilled 176 bytes at 248, and 204 over 16 where it
            # needed 224 (et-sneedfp32c). A 16-bit call's step holds no such tile beside its own (its scaled
            # forward at width 64 needed 208 at 64 x 64 on one consumer, et-sneedbf16).
            step += [accumulator] * 3
    phases = [step]
    if family.operands_in_registers:
        # Its weights relaid as the values' product's left operand (`MmaIssuer._left`), a copy in each warp
        # along the columns (`Family.lhs_copies`), beside the step's logits, where `_products` holds a
        # register left operand no fragment: unpriced, three of an sm103's narrow bounded walks priced 225
        # and 241 spilled 12 and 444 bytes at 255; priced, 257 (narrow-replication-grid).
        phases.append([logits, (block_q, block_k * family.lhs_copies, dtype)])
    if natural:
        # The exponential's terms (`algebra.natural_exp`); a scaled walk's raise its small weights within its
        # step (`algebra.raised_weights`), beside the step's own tiles and its weights at the scale, before
        # their parts and the parts' accumulator.
        phases.append([logits] * 2 + [bits] + ([*core, *scaled_terms] if scaled else []))
        if residuals:
            phases.append([logits, weights, weights, bits, bits])
    if scopes_products(family):
        sums = (block_q, value_width, jnp.float32)
        phases = _read_back(phases, logits, sums, parted=parts > 1 and not scaled)
    depth = value_width if width is None else width
    products = _products(
        family, dtype,
        resources.Product(block_q, depth, block_k, beside=[logits], transposed=True),
        resources.Product(block_q, block_k, value_width, beside=[weights], held=True),
        narrow=tiles.narrow, lhs_swizzle=query_swizzle(family))
    carried = [(block_q, value_width, jnp.float32)]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    if codes:
        phases += _converted(block_k, value_width if width is None else width, value_width)
    # A training walk's per-query rows (`TRAINING_ROWS`), in the accumulator's row layout.
    rows = TRAINING_ROWS * family.row_registers(block_q, jnp.float32) if residuals else 0
    walked = max(resources.live_registers(family, carried, *phases, *products), scored) + rows
    return max(walked, _reach_registers(family, block_q, positions, per_span)) if positions else walked


def _read_back(phases: list[list[tuple]], logits: tuple, sums: tuple, *, parted: bool) -> list[list[tuple]]:
    """A forward's `phases` where its products are read back from tensor memory in their layout and converted
    to the walk's, both live at once (tcgen05's chunks, `Tcgen05ChunksIssuer.summed`): the `logits` a second
    time in each phase holding them, and the weighted values' product beside the `sums` it adds into (a B200's
    64 x 64 forward on three consumers, width 64, priced 126 of 160 and spilled 20 bytes; 64 x 128 132 and
    184, et-h1need). An unscaled fp32 call's (`parted`) parts each read back from a fresh accumulator into
    the walk's registers beside its step: an sm103's 64 x 64 decode priced 188 spilled 60 bytes at 248, so
    252 (bounded-removal)."""
    phases = [[*phase, logits] for phase in phases]
    if parted:
        phases[0] += [sums] * 2
    return [*phases, [sums] * 2]


def _reach_registers(family: mosaic.Family, block_q: int, positions: int, per_span: int) -> int:
    """The registers a packed-spans walk's reach holds before its walk (`_Program._positions`, a call of
    `positions` intervals of `per_span` queries each): two 64-bit addresses of its bounds a position its block
    holds rows of, every one loaded at once (bounded-ptxlive2: 260 address words at 128 rows, 132 at 64; an
    sm103's 128 x 64 spilled 1192 bytes, its reach alone removed needing 248, its 64 rows 184)."""
    return family.scalar_registers() + 4 * min(positions, block_q // per_span + 1)


def peaks_registers(family: mosaic.Family, tiles: Tiles, dtype, *, params: tuple, width: int) -> int:
    """The registers a peak walk's consumer holds (`attention_mosaic_forward._launch`'s `peaks`, a training
    call's first walk): it reads the keys alone for each row's largest logit, so it carries a row's maximum
    and holds a step's logits fp32 and its score hook's parameters (`_hook_live`), its masks' places where
    the keys' indices do not fold, and its logits' product's fragments where its family reads operands from
    registers; no weighted values, weights or exponentials (FS-1971-LEAST-ATTN: the forward's walk priced it
    238 where ptxas's least spill-free cap was 112). Where its family's products run past their step
    (`Family.in_flight`: warpgroup MMA's), the next step's logits accumulate beside the step's it reduces;
    tcgen05's read back beside their conversion (one priced, an H100's and a B200's at 64 x 64 width 64 37
    registers where ptxas needed 64 to 72, et-parity-u)."""
    block_q, block_k = tiles.block_q, tiles.block_k
    hooked, vectors = _hook_live(family, block_q, block_k, params)
    logits = (block_q, block_k, jnp.float32)
    # A product read back from tensor memory, converted to the walk's layout and reduced, each live at once
    # (tcgen05's chunks: two priced, a B200's peak walk at 64 x 64, width 128, priced 69 where ptxas needed
    # 80, et-parity-w).
    held = 1 + family.in_flight + 2 * scopes_products(family)
    step = [logits] * held + _mask_places(family, block_q, block_k)
    products = _products(
        family, dtype,
        resources.Product(block_q, width, block_k, beside=[logits], transposed=True),
        narrow=tiles.narrow, lhs_swizzle=query_swizzle(family))
    carried = [(block_q, 1, jnp.float32)]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    return max(resources.live_registers(family, carried, step, *products), scored)


@dataclasses.dataclass(frozen=True)
class ForwardRegisters:
    """A forward call's facts its launches' consumer registers read (`forward_registers`, `peaks_registers`),
    the one computation its derivation plans with (`MosaicForward.registers`) and each launch states its
    kernel's register model by as it lowers (`Family.kernel`'s `registers`, `report.Launch.kernels`): its
    operands' dtype (an fp32 call's, whose launches walk its 16-bit `parts`), its query and value heads'
    widths, whether it keeps a training walk's split sums (`residuals`), whether its logits stay natural
    (`natural`, beside a config's own natural form), its score hook's parameters' kinds and kept dtypes,
    whether its walk scales in its kernel (`scaled`, `MosaicForward.scales_in_kernel`), converts its keys'
    and values' quantized codes (`codes`, `attention_mosaic_forward._Codes`), and a packed call's intervals
    and their queries each (`positions`, `per_span`; none unpacked), whose reach its walk reads first
    (`_reach_registers`)."""
    dtype: Any
    width: int
    value_width: int
    residuals: bool
    natural: bool
    params: tuple
    parts: int
    scaled: bool
    codes: bool = False
    positions: int = 0
    per_span: int = 1

    def at(self, family: mosaic.Family, tiles: Tiles, *, peaks: bool = False,
           columns: int | None = None) -> int | None:
        """A consumer's registers at `tiles` on `family`: its walk's over the values' columns its launch
        walks (`columns`, a block of them; the tiles' own by default), or its peak walk's (`peaks`); None on
        a family that budgets none (`Family.budgets_registers`)."""
        if not family.budgets_registers:
            return None
        width = _width(self.width, family, self.dtype)
        if peaks:
            return peaks_registers(family, tiles, self.dtype, params=self.params, width=width)
        value_width = columns or tiles.columns or _width(self.value_width, family, self.dtype)
        return forward_registers(family, tiles, value_width, self.dtype, self.residuals,
                                 natural=tiles.natural_form or self.natural, params=self.params,
                                 width=width, parts=self.parts, scaled=self.scaled, codes=self.codes,
                                 positions=self.positions, per_span=self.per_span)


DQ_ROWS = 7
"""The per-query rows a dQ step holds across its walk (`backward_registers`)."""


TRAINING_ROWS = 5
"""The per-query rows a training forward's walk holds at its peak (`forward_registers`): the maximum, the
denominator and the anchor it carries, and its anchor reduction's shuffled and loaded rows (the corrected IR
live set, ET-HOOKBREAK at the held term: 3 carried rows and 2 at the peak's shuffle beside its tiles, 174 and
176 at 64 x 64 and width 64 where the model without them held 164 and 166)."""


STATISTICS = MappingProxyType({"peak": jnp.float32, "total": jnp.float32, "anchor": jnp.float32,
                               "delta": jnp.float32, "chosen": jnp.int32})
"""The per-query statistics a dK/dV step reads (`attention_mosaic_scores._scores`' `peak`, `total`, `anchor`,
`delta` and `chosen`, the anchor key's index), in the order of their arrays, and their dtypes: the dK/dV
launch over a shared dQ streams each beside its queries, and the fused launch the first its block holds
(`attention_mosaic_fused.streamed_statistics`); row vectors a fused launch's queries first holds
(`backward_registers`), column vectors keys first, one at a time."""


def _dq_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                  products: list, held: list, vectors: int, *, accurate: bool, slopes: int) -> int:
    """`backward_registers`' dQ consumer's: its dQ sums (and the accurate form's keyed sums) carried, its
    per-query rows, a step's two products, then its score cotangents, one part of them at a time, its
    hook's parameters beside its pullback (`held`) and its products' fragments where they read registers."""
    block_q, block_k = config.block_q, config.block_k
    logits = (block_q, block_k, jnp.float32)
    # dQ's anchor select priced as its SASS holds it (`_anchor_registers`), within the family's held
    # scalars where they are a measured whole (`Family.scalars_whole`); its hook's parameters and
    # gradients beside its pullback, not its products: they stream in and out of its entries (dQ under
    # a tile bias at 64 x 64, width 128: S's and dP's accumulators the only float tiles at its peak).
    anchor = _anchor_registers(family, block_q, block_k) if not family.scalars_whole else 0
    products = products[:2]
    pullback = products[1:] + [(block_q, block_k, jnp.float32)] * slopes + held
    carried = [(block_q, width // config.columns, jnp.float32)] * (2 if accurate else 1)
    cotangents = [(block_q, block_k, jnp.float32), (block_q, block_k, dtype)]
    # Its per-query rows in the accumulator's row layout (`Family.row_registers`), live across the walk:
    # the maximum, the prior total, the anchor, D, the anchor key and the residual and mass sums
    # (DQROWPROOF64/128: seven rows, D64 at 128 queries past the budget).
    rows = DQ_ROWS * family.row_registers(block_q, jnp.float32)
    # Its products' fragments where they read registers (`_products`): the logits', the output
    # cotangents' beside them, and dQ's of the score cotangents in registers. The keys are read by the
    # logits and dQ, so stay live while the intervening product reads the values.
    fragments = _products(
        family, dtype,
        resources.Product(block_q, width, block_k, beside=[logits], transposed=True, source="keys"),
        resources.Product(block_q, value_width, block_k,
                          beside=[logits] * 2, transposed=True, source="values"),
        resources.Product(block_q, block_k, width // config.columns,
                          beside=cotangents[1:], held=True, source="keys"),
        narrow=config.narrow)
    # A score hook's slopes and parameters live until its pullback takes them (`gradient.pulled`).
    hooked = [logits] * slopes + held
    if accurate and family.operands_in_registers:
        # The accurate form's parts (`algebra.formed_parts`) issued against one right operand, the keys,
        # whose fragments its family loads once and holds across them, two of its depth's loads ahead,
        # beside the cotangents, the keyed sums' weights fp32 and the part (an A100's dQ at 64 x 64:
        # one LDSM's registers read by four HMMAs over three parts, 218 to 221 live at its peak), and a
        # hook's pullback beside them (an A100's at 64 x 64 and width 128 under alibi spilled 16 bytes).
        right = resources.operand_fragments(family, block_q, block_k, width // config.columns,
                                            resources.OperandRead(dtype, parts=0),
                                            resources.OperandRead(dtype, retained=True))
        fragments.append([cotangents[0], (block_q, block_k, jnp.float32), cotangents[1], *right, *hooked])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's parts scaled (`product.Scaled`): each part's product its own accumulator, restored and
        # added into dQ's (`operands.Issuer.summed`), beside the cotangents and both parts, formed first.
        fragments.append([cotangents[0], cotangents[1], cotangents[1],
                          (block_q, width // config.columns, jnp.float32)])
    if accurate and hooked:
        # The accurate form's cotangents fp32 formed beside the keyed sums' weights fp32
        # (`softmax_cotangents`) and the hook's pullback: an H100's dQ at 64 x 64, width 64, spilled 128
        # bytes under a softcap and 208 under alibi, a tile a slope past its model (acchook3).
        fragments.append([cotangents[0], logits, *hooked])
    phases = resources.live_registers(family, carried, products, pullback, cotangents, *fragments)
    return phases + anchor + rows + vectors


def backward_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                        accurate: bool, dq: bool, split: bool = False, params: tuple = (),
                        slopes: int = 0, dropped: bool = False,
                        hooked: tuple[bool, bool] = (False, False)) -> int:
    """The registers a backward launch's consumer holds at `config` (`resources.live_registers`): a dQ
    consumer's dQ sums `[bq, D]` fp32 carried (and the accurate form's keyed sums) and its per-query rows,
    beside a step's logits and output cotangents' products fp32, then its score cotangents fp32 and one part
    of them in the operands' dtype at a time (`algebra.formed_parts`; ptxas held the accurate form's three at
    width 128 in 255 registers, unspilled, where all three would be 262); a dK/dV consumer's dK and dV sums
    `[bk, D]`, `[bk, Dv]` fp32 carried, beside a step's two products (a fused launch's queries first, its dQ
    tile `[bq, D]` fp32 too) and its per-query statistics, each of a consumer's share of a slot's queries
    where its consumers `split` them (`split_queries`), else of the slot's; every sum its share of the
    `columns`. A
    queries-first step's masked tile holds its predicates beside its products (`algebra.seen` of its places'
    int tiles, `_coords`, a byte an entry), from its logits' mask to its cotangents' (ptxas: the fused
    launch at 128 x 64 and width 64 spills 60 bytes, its model 252 without them; 64 x 64 at width 128 holds
    255 unspilled, 242); a keys-first step's places are vectors broadcast at each use. A score hook's
    parameters and their gradients the launch forms are live beside its products (`_hook_live`), the places
    it reads as values (`hooked`, its queries' and its keys', `score_places`) in a keys-first launch, and its
    pullback's `slopes` tiles fp32 that vary (`varying_slopes`: its output and its slopes along its
    differentiated inputs) beside its output cotangents' product once the logits' is spent, a phase of their
    own (dQ under a temperature at 128 x 64, width 64, spilled 4316 bytes, its SASS's spills their float
    chains, FMUL, FADD and FSEL, the products' and dQ's accumulators live beside them)."""
    block_q, block_k = config.block_q, config.block_k
    logits = (block_q, block_k, jnp.float32)
    held, vectors = _hook_live(family, block_q, block_k, params, grads="queries" if dq else "keys")
    products = [logits] * 2 + [(block_q, block_k, jnp.int8)] + held
    pullback = products[1:] + [logits] * slopes
    if dq:
        return _dq_registers(family, config, width, value_width, dtype, products, held, vectors,
                             accurate=accurate, slopes=slopes)
    if config.shared_keys:
        # Shared keys (`attention_mosaic_coop`): a consumer's share of the head's columns of dK and dV
        # carried, beside a step's two products over its share of the keys and its share of the dQ tile,
        # its statistics row vectors.
        share = config.consumers
        carried = [(block_k, width // share, jnp.float32), (block_k, value_width // share, jnp.float32)]
        products = [(block_q, block_k // share, jnp.float32)] * 2 + [(block_q, block_k // share, jnp.int8)]
        if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
            # fp16's operands as two parts at the running scale (`attention_mosaic_coop._sixteen_step`): an
            # operand's scaled tile and both its parts beside the products.
            products += [(block_q, block_k // share, jnp.float32), *[(block_q, block_k // share, dtype)] * 2]
        statistics = len(STATISTICS) * family.row_registers(block_q, jnp.float32)
        # A score hook's slopes held beside both products to its pullback (as the fused launch's below).
        slope_tiles = [(block_q, block_k // share, jnp.float32)] * slopes
        return resources.live_registers(family, carried, products + slope_tiles,
                                        [(block_q, width // share, jnp.float32)]) + statistics
    carried = [(block_k, width // config.columns, jnp.float32), (block_k, value_width // config.columns,
                                                                 jnp.float32)]
    if config.fused and not split and not config.shared_dq:
        # Queries first (`attention_mosaic_fused._fused_launch`), on either layout: its statistics row
        # vectors.
        statistics = len(STATISTICS) * family.row_registers(block_q, jnp.float32)
        dq_tile = [(block_q, width, jnp.float32)]
        # Its products' fragments where they read registers (`_products`): the logits' and the output
        # cotangents', then dV's, dK's and dQ's of the weights and score cotangents it stages.
        fragments = _products(
            family, dtype,
            resources.Product(block_q, width, block_k, beside=[logits], transposed=True),
            resources.Product(block_q, value_width, block_k, beside=[logits] * 2, transposed=True),
            resources.Product(block_k, block_q, value_width, beside=[logits]),
            resources.Product(block_k, block_q, width, beside=[logits]),
            resources.Product(block_q, block_k, width, beside=dq_tile))
        if not family.operands_in_registers:
            # A family reading its operands from shared memory holds the weights narrowed for dV's product
            # beside the pullback until they are staged: wgmma's at 64 x 64, width 128, under a softcap held
            # six packed words of them at its peak, where it spilled 104 bytes (softcap2, row 5477).
            fragments.append([*pullback, (block_q, block_k, dtype)])
        held = _fused_held(logits, dtype, dropped=dropped)
        return resources.live_registers(family, carried, products + held, pullback + held, dq_tile,
                                        *fragments) + statistics
    return _keys_first_registers(family, config, width, value_width, dtype, carried,
                                 block_q // config.consumers if split else block_q, accurate=accurate,
                                 params=params, slopes=slopes, hooked=hooked)


def _keys_first_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                          carried: list, queries: int, *, accurate: bool, params: tuple, slopes: int,
                          hooked: tuple[bool, bool]) -> int:
    """`backward_registers`' keys-first dK/dV consumer at `config`, its `queries` a slot's or its share's:
    its `carried` sums beside a step's two products and its hook's live values and places, its statistics in
    its tiles' column layout, and its products' fragments."""
    block_k = config.block_k
    held, vectors = _hook_live(family, queries, block_k, params, grads="keys", keys_first=True)
    # The places a score hook reads (`score_places`), each a vector beside its products until the hook
    # reads it: its mask folds them into its compares, a hook takes them as values (a B200's row-indexed
    # dK/dV at 64 x 64, width 128, over two column shares: 24 bytes spilled at a model of 233, its query
    # places' 16 words among ptxas's live registers, FINAL14).
    vectors += (family.column_registers(queries, jnp.int32) if hooked[0] else 0)
    vectors += (family.row_registers(block_k, jnp.int32) if hooked[1] else 0)
    # Its two products' accumulators, and where its family's products run past their step
    # (`Family.in_flight`: warpgroup MMA's) the next step's beside them in a launch of its own (an H100's
    # keys-first dK/dV at 64 x 64 over two consumers' query shares priced 108 to 124 registers where ptxas
    # needed 136 at every hook, et-parity-w); a fused launch's step waits on its dQ adds between them.
    ahead = 0 if config.fused else family.in_flight
    step = [(queries, block_k, jnp.float32)] * (2 * (1 + ahead)) + held
    pullback = step[1:] + [(queries, block_k, jnp.float32)] * slopes
    # The keys-first step's statistics in its tiles' column layout (`Family.column_registers`), each read
    # where it is used, from global memory or a shared dQ's stream (`STATISTICS`), one's column vector live
    # at a time (an A100's keys-first launch at 64 x 64, width 128: 17 to 18 words at its peak, where all
    # five's are 80); a fused launch's dQ tile of its share `[queries, D]` fp32 beside its score
    # cotangents, a shared dQ's its share of the columns (`BackwardTiles.shared_dq`).
    statistics = family.column_registers(queries, jnp.float32)
    dq_width = width // config.consumers if config.shared_dq else width
    added = [[step[0], (queries, dq_width, jnp.float32)]] if config.fused else []
    # Its products' fragments where they read registers (`_products`), keys first: the logits' and the
    # output cotangents' of the keys against the queries over the whole head, which every column share
    # recomputes (`attention_mosaic_dkv`'s products read the slots whole), then dV's and dK's of the weights
    # and the score cotangents in registers, its share of their columns.
    weights = (block_k, queries, dtype)
    columns = config.columns
    fragments = _products(
        family, dtype,
        resources.Product(block_k, width, queries, beside=step[:1], transposed=True),
        resources.Product(block_k, value_width, queries, beside=step, transposed=True),
        resources.Product(block_k, queries, value_width // columns, beside=[weights], held=True),
        resources.Product(block_k, queries, width // columns, beside=[weights], held=True),
        narrow=config.narrow)
    if accurate and family.operands_in_registers:
        # The accurate form's dK parts issued against one right operand, the queries, held at its whole depth
        # across them, beside the cotangents fp32, dV's weights and the part (an A100's dK/dV at 64 x 64:
        # one LDSM's registers read by HMMAs of all three parts, 64 words of them at its peak; one part:
        # ptxas's least 240 fell to 192).
        right = (queries, width // columns * family.rhs_copies, dtype)
        # A score hook's parameters' key gradients beside them (an A100's at 64 x 64 under a key bias spilled
        # 32 bytes, acchook3).
        fragments.append([step[0], weights, weights, right, *held])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's dV and dK parts scaled (`product.Scaled`): each product its own accumulator, restored and
        # added into dV's or dK's (`operands.Issuer.summed`), beside the cotangents and both parts, formed
        # first.
        restore = (block_k, max(width, value_width) // columns, jnp.float32)
        fragments.append([step[0], weights, weights, restore])
    phases = resources.live_registers(family, carried, step, pullback, *added, *fragments)
    return phases + statistics + vectors


@dataclasses.dataclass(frozen=True)
class BackwardRegisters:
    """A backward call's facts its launches' consumer registers read (`backward_registers`), the one
    computation its derivation plans with (`MosaicBackward._launch_registers`) and each launch states its
    kernel's register model by as it lowers (`Family.kernel`'s `registers`): its operands' dtype, its query
    and value heads' widths, its form (`accurate`), its score hook's parameters' kinds and kept dtypes and
    the tiles its pullback varies in the dQ launch and the dK/dV launch (`slopes`), whether it drops
    entries, whether its config walks queries first (`BackwardTiles.queries_first`), and the places its
    score hook reads, its queries' and its keys' (`hooked`, `score_places`)."""
    dtype: Any
    width: int
    value_width: int
    accurate: bool
    params: tuple
    slopes: tuple[int, int]
    dropped: bool
    queries_first: bool
    hooked: tuple[bool, bool]

    def at(self, tiles: BackwardTiles, *, dq: bool) -> int:
        """A consumer's registers in the dQ launch (`dq`) or the dK/dV launch at `tiles`, on the bundle they
        run on (`bundle_of`)."""
        family = bundle_of(tiles, self.dtype)
        width, value_width = (_width(each, family, self.dtype) for each in (self.width, self.value_width))
        return backward_registers(family, tiles, width, value_width, self.dtype, accurate=self.accurate,
                                  dq=dq, split=split_queries(family) and not self.queries_first,
                                  params=self.params, slopes=self.slopes[0 if dq else 1],
                                  dropped=self.dropped, hooked=self.hooked)

