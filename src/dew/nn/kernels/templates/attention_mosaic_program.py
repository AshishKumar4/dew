"""What the attention template's Mosaic GPU launches share (`attention_mosaic`, design §6): the bundles it
runs on (`bundle`), its configs (`Tiles`, `BackwardTiles`), its legality rule (`refused`), and what a
forward's and a backward's programs read alike: the call's padding to a launch's blocks, its spans' block map
and each tile's coordinates (`_Program`, `_Spans`, `_coords`), its hooks' parameters and state (`_Hooked`,
`_param`)."""

import dataclasses
import functools
import math
import operator
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
    after_walk,
    attention_algebra as algebra,
    axes,
    band,
    bundles,
    emission,
    handoffs,
    layout,
    mosaic,
    operands,
    outputs,
    placed,
    resources,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.attention import (
    TILED,
    WIDEST_POWER,
    Draw,
    Dropout,
    Hooks,
    KeySpans,
    Kind,
    PaddedCall,
    band_origins_of,
    block_map,
    differentiable,
    drops_in_place,
    floating_params,
    head_width,
    kept_dtype,
    key_tiles,
    kind_of,
    shared_place,
    whole,
)
from dew.nn.kernels.templates.cost import Stated, stored
from dew.nn.kernels.templates.hooks import Side
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
    (`Issuer.product_scratch`: tcgen05's chunks' and rows'), asked of a program with a product of each kind
    its scratch takes (one fresh, a register left operand, one issued ahead, a walk's sums): the rows' takes
    exactly those."""
    rows = family.rows
    return bool(family.product_scratch(1, (rows,), ((rows, jnp.bfloat16),), (rows,), (rows,)))


def qualified(family: mosaic.Family) -> bool:
    """Whether the attention mainloop runs on `family`: one whose consumers issue their products
    (`Family.issues_in_consumers`), whose walk's softmax and its rescales run between them in the registers
    that hold them (wgmma's, mma.sync's, tcgen05's chunks, a TPU core's MXU), or one that places a walk's
    declared products on a role of their own (`Issuer.places_on_role`, tcgen05's rows, FlashAttention 4's);
    not a bundle whose one warp issues a persistent tile's (`mosaic_sm100.RINGS`)."""
    return family.issues_in_consumers or family.places_on_role


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
    found = [family for family in bundles.layouts(generation_name, dtype, rows, scoped=True, tpu=True)
             if qualified(family)]
    # A family that places its products on a role first, the others in the bundles' order (tcgen05's rows:
    # nvgpu-lean's B200 at the 4B call, 1.23 ms a launch against the chunks' two launches' 2 x 2.49).
    return tuple(sorted(found, key=lambda family: not family.places_on_role))


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


def scheduled_roles(family: mosaic.Family, consumers: int, consumer: Callable[[int], tuple | None],
                    producer: Callable[[], None], issued: Callable[[placed.Tcgen05Walked], None], *,
                    reducer: Callable[[int], None] | None = None):
    """A backward launch's roles (`Family.scheduled`): its consumers, its producer's loads, and where its
    family places its products on a role (`placed.Tcgen05Walked`) its compute warpgroups `consumer(role)`,
    each built at its static role, its turns and its store its own (`mosaic.per_role`), its issuer
    (`issued(family)`) and its reducers, `reducer(role)`, where it hands its adds to them
    (`handoffs.REDUCER`)."""
    if not family.places_on_role:
        return family.scheduled(consumers, consumer, producer)
    walked = placed.walked(family)

    def compute(role: int, index) -> None:
        del index
        consumer(role)

    joined = () if reducer is None else ((handoffs.REDUCER, consumers),)
    bodies = {} if reducer is None else {handoffs.REDUCER.name: reducer}
    return walked.scheduled(consumers, functools.partial(mosaic.per_role, roles=consumers, run=compute),
                            producer, issuer=lambda: issued(walked), joined=joined, bodies=bodies)


def placed_scratch(family: mosaic.Family, walked: int, dtype, fused: tuple[int, ...], kept: tuple[int, ...]
                   ) -> dict:
    """A backward launch's product scratch where its family places its products on a role (`Family.scoped`'s
    `products`, `placed.Tcgen05Walked.product_scratch`): its scores' and cotangents' turns of its `walked`
    tile each, their narrowed lefts in the operands' `dtype`, a fused walk's dQ's halves over them (`fused`,
    their columns) and its kept sums' columns (`kept`); none elsewhere."""
    if not family.places_on_role:
        return {}
    return {"products": (family.consumers, fused, ((walked, dtype),), (walked, walked), kept)}


def walked_layouts(generation_name: str | None, dtype) -> tuple[mosaic.Family, ...]:
    """The bundles the backward's launches take on a device of `generation_name` for operands of `dtype`
    where their products are placed on a role (`bundles.Launch.walked`: a B200's tcgen05, its dK/dV keys
    first `placed.KEYS`, its dQ queries first `placed.QUERIES`, FlashAttention 4's backward, row 4839); none
    elsewhere."""
    if generation_name is None:
        return ()
    return tuple(family for family in bundles.layouts(generation_name, dtype, walked=True)
                 if family.places_on_role)


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


def bundle_of(tiles: "Tiles | BackwardTiles", dtype, *,
              target: generation.GpuTarget | None = None) -> mosaic.Family:
    """The bundle `tiles` run on: `target`'s (or the current device's `generation.gpu_target`) layout their
    family names, which their legality matched (`MosaicForward.legal`). A config names its physical layout;
    which configs a call derives is the qualification's (`layouts`), so a config pinned on a family not yet
    qualified (an A100's mma.sync, its resource witness) resolves too. A forward's family is arranged for its
    consumer's share, `block_q` x `block_k` (`Issuer.specialized`, as the GEMM's: mma.sync's warps by the
    share, whose default 64-row atom no 32-row consumer's operands cast to); a backward's as
    `backward_family` arranges it (row 11987)."""
    # Every bundle the device issues the mainloop's launches (a launch of a row each ranks them all,
    # `bundles.layouts`), the backward's dQ launch's producer layout among them (`produced`), and its dK/dV
    # launch's keys first on a role (`walked_layouts`).
    name = generation.gpu_target().generation if target is None else target.generation
    found = {family.name: family
             for family in (*bundles.layouts(name, dtype, 1, scoped=True, tpu=True, produced=True),
                            *walked_layouts(name, dtype))}
    family = found.get(tiles.family)
    if family is None:
        raise ValueError(f"{tiles}: this device's attention bundles are {sorted(found)}")
    if isinstance(tiles, Tiles):
        return (family.specialized(tiles.block_q, tiles.block_k)
                if tiles.block_q < family.role_rows else family)
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
    of its launches walks (`relaunch_columns`, a launch a block of them; all of them where None), on another
    family and query block where it names them (`relaunch_family`, `relaunch_q`: a family whose program
    takes no masked rewalk in place, `Issuer.rewalks`, rewalks on the device's register family);
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
    relaunch_family: str | None = None
    relaunch_q: int | None = None
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
        return dataclasses.replace(self, block_q=self.relaunch_q or self.block_q,
                                   block_k=self.relaunch_k or self.block_k,
                                   consumers=self.relaunch_consumers or self.consumers,
                                   family=self.relaunch_family or self.family, relaunch_k=None,
                                   relaunch_consumers=None, relaunch_columns=None, relaunch_family=None,
                                   relaunch_q=None, natural_form=True, columns=self.relaunch_columns)


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
    shared_keys: bool = False
    """Whether the fused launch's consumers share one key block (`attention_mosaic_coop._coop_launch`):
    each forms the logits of its share of the keys and holds its share of the head's columns of dK, dV and
    dQ, where a consumer's own keys' whole dK and dV sums pass its registers (width 256's)."""
    narrow: bool = False
    """Whether the streams its products read transposed stage at the narrowest swizzle
    (`transposed_swizzles`), where its family reads operands from registers: a load one instruction's depth
    of them (`Family.rhs_fragment_depth`), not a wide swizzle tile's, a wide stage's bandwidth traded for
    registers (ET-CENSUS80: a 64 x 192 forward's keys at the widest 162 of its 291 registers)."""
    dkv_blocks: tuple[int, int] | None = None
    """The dK/dV launch's own query tile and key block where they are not the config's (`dkv`): a walk whose
    products a role issues (`placed.Tcgen05Walked`) holds a block of its issuer's rows, the dK/dV launch's
    keys first (`placed.KEYS`, 128 keys a step of 64 queries) where the dQ launch's walks its own tiles
    (`placed.QUERIES`, 128 queries a step of 64 keys, or another family's)."""

    @property
    def dkv(self) -> "BackwardTiles":
        """The dK/dV launch's own tiles: its layout, consumers and blocks (`dkv_family`, `dkv_consumers`,
        `dkv_blocks`, the config's where None), each launch's layout its own (`bundles.layouts`)."""
        block_q, block_k = self.dkv_blocks or (self.block_q, self.block_k)
        return dataclasses.replace(self, family=self.dkv_family or self.family,
                                   consumers=self.dkv_consumers or self.consumers, block_q=block_q,
                                   block_k=block_k, dkv_family=None, dkv_consumers=None, dkv_blocks=None)

    @property
    def dkv_apart(self) -> bool:
        """Whether the dK/dV launch runs on a layout or a count of consumers of its own (`dkv`): its tiles
        other than the config's own, overrides naming the config's own aside."""
        return self.dkv != dataclasses.replace(self, dkv_family=None, dkv_consumers=None, dkv_blocks=None)

    @property
    def queries_first(self) -> bool:
        """Whether the backward is the fused launch queries first (`_fused_launch`), on the config's own
        layout and consumers: each consumer its own keys over the whole query slot, on either layout."""
        return self.fused and not self.dkv_apart

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")


def _width(width: int, family: mosaic.Family, dtype) -> int:
    """A head width as a launch stages it: as `padded_call` pads it (`head_width`: a power of two to 256,
    whole groups of 64 columns past it) in whole swizzled rows (`Family.column_unit`)."""
    return whole(head_width(width), family.column_unit(dtype))


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


def issued_into(family: mosaic.Family, acc, parts, rhs, interpret: bool, *,
                shape: tuple[int, int] | None = None):
    """`acc`, or a fresh fp32 accumulator of `shape` where None, with each of `parts`' products with `rhs`,
    a tile in shared memory, issued into it in place (`operands.Issuer`'s carried accumulator), each part's
    left operand in registers and formed as it is taken (`algebra.formed_parts`); an fp16 part's
    (`product.Scaled`) its own accumulator, restored by its row's power of two (`algebra.restored`) and
    added in."""
    issuer = operands.Issuer(_Issuing(family, interpret), {"part": 1}, restored=algebra.restored)
    right = operands.Parts((rhs,), rhs.dtype)
    products = (("part", part, right) for part in parts)
    return issuer.summed(products, shape) if acc is None else issuer.summed(products, into=acc)


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
    key tiles (a forward's), or of query blocks and key blocks (a backward's dQ and dK/dV programs), its
    keys whole blocks of the dK/dV launch's programs too (`dkv_holders`: a layout and consumers of its own,
    padded to the dQ launch's, its last keys had no program to write them) and its queries whole of its
    query tiles (`BackwardTiles.dkv_blocks`)."""
    if isinstance(config, BackwardTiles):
        keys = math.lcm(config.consumers * config.block_k, dkv_holders(config) * config.dkv.block_k,
                        config.dkv.block_q)
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
    """The key blocks a dK/dV program holds: one its consumers share (`split_queries`, or walk a step each in
    turn where a role issues their products, `placed.KEYS`), else one each."""
    return 1 if split_queries(family) or family.places_on_role else consumers


def dkv_holders(config: BackwardTiles) -> int:
    """The key blocks a program of the dK/dV launch at `config` holds (`holders`), its layout's own."""
    dkv = config.dkv
    return holders(bundles.named(dkv.family, jnp.bfloat16, None), dkv.consumers)


def _padded(queries: int, keys: int, config: Tiles | BackwardTiles) -> tuple[int, int]:
    """A call's queries and keys as a launch at `config` pads them (`padded_call`, `_block`)."""
    block = _block(config)
    return whole(max(queries, 1), block), whole(max(keys, 1), block)


def launch_scope(family: mosaic.Family, allocations: tuple, reducing_warpgroups: int, *, reduces: bool = True,
                 products: tuple | None = None, over_rings: bool = False,
                 values: tuple[dict, tuple] | None = None) -> resources.Scope:
    """A launch's `allocations` (`(scoped, streams)`, `Family.scoped`'s, and those it scopes after its
    walks where it has them, `after_walks`) by lifetime (`Family.lifetimes`), its issuer's product scratch
    where its `products` take one (`Family.product_held`, its tensor memory's addresses
    `Issuer.product_addresses`), its issuer's scratch values where `values` are given
    (`Issuer.value_scratch`: TPU VMEM live tiles, row 10965), with the algebra's reductions' scratch where it
    `reduces` (a launch reducing its scores, weights or cotangents across warps) and its
    `reducing_warpgroups`' scratch (`resources.Scope`)."""
    scoped, streams, *after = allocations
    after_walks = tuple(after[0]) if after else ()
    scratch = () if values is None else family.value_scratch(*values)
    if after_walk.uses_over_rings(family, after_walks, over_rings=over_rings):
        added, rest, _ = after_walk.over_rings_plan(streams, after_walks)
        held, walks = family.lifetimes(rest)
        return resources.Scope(program=(scoped, *family.product_held(products), *scratch, *added, *held),
                               walks=walks, after=(), reduces=reduces,
                               reducing_warpgroups=reducing_warpgroups,
                               addresses=family.product_addresses(products))
    held, walks = family.lifetimes(streams)
    tiles = tuple((tile.shape, tile.dtype) for tile in after_walks)
    return resources.Scope(program=(scoped, *family.product_held(products), *scratch, *held),
                           walks=walks, after=tiles, reduces=reduces,
                           reducing_warpgroups=reducing_warpgroups,
                           addresses=family.product_addresses(products))


def extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most what one copy moves (`Family.copy_extent`,
    where the family's copies bound it; else registers bound them, `fits`) and the call's `extent` in whole
    multiples (a longer tile holds padding alone)."""
    return _multiples(family, math.lcm(family.role_rows, family.column_unit(dtype)), extent)


def walked_unit(family: mosaic.Family, dtype, parts: int = 1) -> int:
    """The row granule a walked tile steps by on `family` for `dtype` (`walked_extents`, `whole_tiles`):
    a placed family's `walked_rows`, a parted fp32 wgmma forward's 64-byte swizzle columns over its staged
    bf16 parts (32 keys so width 256's three parts fit Hopper's 232,448-byte block, rows 6508, 10773), else
    `Family.column_unit`."""
    if family.places_on_role:
        return family.walked_rows
    staged = jnp.bfloat16 if parts > 1 else dtype
    if parts > 1 and not scopes_products(family):
        return min(family.column_unit(staged), swizzling.swizzle_columns(staged, 64))
    return family.column_unit(staged)

def walked_extents(family: mosaic.Family, dtype, extent: int, *, parts: int = 1) -> tuple[int, ...]:
    """A walked tile's extents on `family`, largest first: whole multiples of `walked_unit`, within
    `extents`' bounds."""
    return _multiples(family, walked_unit(family, dtype, parts), extent)

def _multiples(family: mosaic.Family, unit: int, extent: int) -> tuple[int, ...]:
    """Whole multiples of `unit`, largest first, at most what one of `family`'s copies moves and the call's
    `extent` in whole multiples (`extents`)."""
    top = -(-max(extent, 1) // unit)
    if family.copy_extent is not None:
        # Stricter than the tiled box's own limit along its rows (rows / 8 at most `copy_extent`), kept until
        # longer tiles are measured: lifting it lets a derivation offer tiles past 256 rows.
        top = min(top, family.copy_extent // unit)
    return tuple(range(top * unit, 0, -unit))


BACKWARD_CONSUMERS = 2
"""The consumers a backward launch takes at most: its exchanges and splits pair them (a dK/dV block's column
shares, a slot's query shares, `Family.exchange`'s two schedules). A third left a 64-row tile 21 rows a share
and the accurate form 30 to 290 times the reference's error (045aa11863e). The forward's walk
takes the family's every consumer (FlashAttention 3's d64 tile: three, 192 rows)."""


def most_consumers(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> int:
    """The consumers a launch of `tiles` takes at most on `family`: a forward's the family's own
    (`Family.consumers`), a backward's those a launch may name there (`Family.shared_consumers`: an A100's
    paired fused launch's two, `MosaicBackward.legal`), at most `BACKWARD_CONSUMERS`."""
    return (family.consumers if isinstance(tiles, Tiles)
            else min(family.shared_consumers, BACKWARD_CONSUMERS))


def held_walked(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> tuple[int, int]:
    """The rows a program of `family` at `tiles` holds and the tile it walks a step: its key block and its
    query tiles where its family holds keys (`Family.holds_keys`, a keys-first placement), else its query
    block and its key tiles."""
    return (tiles.block_k, tiles.block_q) if family.holds_keys else (tiles.block_q, tiles.block_k)


def column_share(array: jax.Array, count: int) -> outputs.Dim:
    """A program's `count` of `array`'s columns as an output dimension (`outputs.Dim`): its last axis, or that
    axis over the shares it is split into (`BackwardTiles.columns`), each a whole share."""
    width = array.shape[-1]
    if width % count:
        raise ValueError(f"a share of {count} of {width} columns is not a whole one")
    columns = outputs.axis(array, -1)
    return columns if count == width else columns // (width // count)


def whole_tiles(family: mosaic.Family, tiles: "Tiles | BackwardTiles", *, parts: int = 1) -> bool:
    """Whether a program of `family` takes `tiles`: its query tiles positive whole multiples of the rows a
    role's products cover (`Family.rows`), its key tiles of the rows a walked tile comes in
    (`Family.walked_rows`, or `walked_unit` for parted fp32 forwards), and one to its launch's count of
    consumers (`most_consumers`), whose turns its walks are written for; a family whose products a role
    issues (`Issuer.places_on_role`) one product's rows, a row a consumer thread: a keys-first walk's its
    key block (`placed.KEYS`)."""
    held, walked = held_walked(family, tiles)
    rows = held == family.rows if family.places_on_role else held % family.rows == 0
    unit = (min(family.walked_rows, walked_unit(family, jnp.float32, parts))
            if parts > 1 else family.walked_rows)
    return (0 < tiles.consumers <= most_consumers(family, tiles) and tiles.block_q > 0 and tiles.block_k > 0
            and rows and walked % unit == 0)

ROUNDED = Refusal('rounds', "a quantized model's weights round at their row's maximum in the anchored "
                  "training walk's forward alone (`Hooks.rounds`), no dropping or resumed call's, and its "
                  "backward is its quantization's own (`attention_quantized`)")
"""Why the mainloops take a quantized call's forward alone, of neither a dropping nor a resumed call."""


def scaled_drop(dtype, drop, initial) -> bool:
    """Whether a dropping call's forward products leave in-place accumulation (`_accumulated`, row 9507):
    weights past `dtype`'s largest binade (traced row-scaled `product.Scaled` parts, `drops_in_place`) or an
    initial state beside a walk of no headroom at a non-power-of-two multiplier (`sums_units`)."""
    if drop is None:
        return False
    return (not drops_in_place(dtype, drop.reach)
            or (initial is not None and not algebra.headroom(dtype) and drop.bound != drop.reach))


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
    if (backward and hooks.drop is not None and hooks.initial is not None and not algebra.headroom(dtype)
            and hooks.drop.bound != hooks.drop.reach):
        return Refusal('dropout_rate', f"the Mosaic dropping backward takes no initial state beside "
                       f"{jnp.dtype(dtype).name} values, whose walk has no headroom")
    if backward and hooks.drop is not None and not drops_in_place(dtype, hooks.drop.reach):
        return Refusal('dropout_rate', f"the Mosaic dropping backward takes {jnp.dtype(dtype).name} "
                       f"weights within its range, not a multiplier of {hooks.drop.bound:g} (a rate of "
                       f"{hooks.drop.rate:g})")
    if jnp.dtype(dtype) not in sixteen:
        return Refusal('dtype', f"the Mosaic mainloop takes 16-bit operands, not {jnp.dtype(dtype)}")
    if backward and not transposable(hooks):
        return Refusal('hook_layout', "the Mosaic backward holds its tiles keys first, and its score hook "
                       "takes an axis of them by an operation no layout rule takes (`layout.RULES`)")
    if any(each > WIDEST_POWER and head_width(each) != each for each in (width, value_width)):
        return Refusal('width', f"the Mosaic mainloop takes a head past {WIDEST_POWER} wide in whole groups "
                       f"of 64 columns (`head_width`), not {width} and {value_width}")
    if backward and max(width, value_width) > WIDEST_POWER:
        return Refusal('width', f"the Mosaic backward takes heads to {WIDEST_POWER} wide: an absorbed MLA "
                       "call past it is inference's (DeepSeek trains its 192-wide keys and 128-wide values)")
    return None


def approximate(hooks: Hooks, tile: tuple[int, int], value_width: int, *, differentiated: bool,
                query: tuple | None = None) -> bool:
    """Whether a launch running `hooks` on `tile` takes the hardware's approximate transcendentals
    (`Family.kernel`'s `approx_math`): only where no hook computes one Mosaic GPU approximates
    (`emission.APPROXIMATED`), with `differentiated` the score hook's pullback too. The mainloop's own
    exponentials hold check 1 at ex2.approx; a hook's need not: on an H100 a softcap's tanh.approx (2^-11
    relative) beside an initial state put the state's sums' gradient at 7.04 times the reference's error from
    float64, every hook on par with exact math (the hooked backward, HOOKS-FIX-h100)."""
    traced = _hook_jaxprs(hooks, tile, value_width, differentiated=differentiated, query=query)
    return not any(_primitives(jaxpr) & emission.APPROXIMATED for jaxpr in traced)


def reducing(family: mosaic.Family, consumers: int, hooked: bool, *, scaled: bool = False,
             magnitudes: bool = False, joined: tuple = ()) -> int:
    """The warpgroups of a launch of `consumers` whose arithmetic may reduce across warps implicitly, each
    reserving its share of Mosaic GPU's reduction scratch (`Family.kernel`'s `reducing_warpgroups`): a
    `hooked` launch's hook's (`hook_reduces`: a unit row, which the fork selects by a reduction across warps,
    ledger 1428; an output hook's, untraced), a `scaled` step's tile reduced to its largest across the warps
    holding its rows (`algebra.running_parts`, an fp16 fused backward's), or a scaled walk's keys' value
    `magnitudes`, its value tile's columns reduced across the warps holding them (`Family.keys_read`,
    `algebra.SCALED_WALK`), so every warpgroup of a launch that runs such a hook or scales; none otherwise.
    A launch's rows flagged to their block's flag reserve none: the flag is its warpgroup's vote
    (`attention_mosaic_forward._status_stored`, `bar.red.or.pred`)."""
    if scaled or magnitudes or hooked:
        return family.warpgroups(consumers, joined=joined)
    return 0

def forward_reduces(family: mosaic.Family, reducing_warpgroups: int) -> bool:
    """Whether a forward launch's arithmetic reduces across warps, reserving its family's reduction scratch
    (`Family.kernel`'s `reduces`, `resources.Scope.reduces`): where its warps split its rows' columns
    (`Issuer.rows_whole`: mma.sync's warps along the keys), or a warpgroup of it reduces implicitly
    (`reducing`). Its rows' maxima and sums each warp's own otherwise, and its flags and verdicts its
    warpgroup's votes: sm120's forward at width 256 fits whole without the 2,048 bytes."""
    return not family.rows_whole or reducing_warpgroups > 0


def hook_reduces(hooks: Hooks | None) -> bool:
    """Whether the score hook may reduce across warps (`reducing`): traced, never run, on a tile with its
    pullback, it forms or reads a value but a scalar or a whole tile (a unit row, `k_pos[:1, :]`, ledger
    1428; a sum along an axis) or nests a computation but a call's (a branch, a loop); a fingerprint (a keyed
    launch's hook) counted so. Gemma 2's softcap forms none, so its launches reserve no scratch: its
    backward's one launch 256 wide on an H100 fits beside it (row 3704: 233,568 bytes with it, past the
    block's 232,448). False without a hook."""
    if hooks is None or hooks.score.fn is None:
        return False
    if not callable(hooks.score.fn):
        return True
    params = param_types(hooks.score.sides)
    return _hook_reduces(hooks.score.fn, params)


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
                 differentiated: bool = False, query: tuple | None = None) -> list[jax_core.Jaxpr]:
    """`hooks` traced as a launch runs them on `tile`, its outputs `value_width` wide: the score hook (and,
    `differentiated`, its pullback along the logits and its floating parameters), the output hook, and the
    query prologue where the launch takes it on its query tile, `query`'s `(rows, D, dtype)`
    (`attention_mosaic_forward._prologued`: its parameters as the tile's entries, each in its own dtype)."""
    traced = []
    if query is not None and callable(hooks.query.fn):
        rows, width, dtype = query
        traced.append(jax.make_jaxpr(hooks.query.fn)(
            jax.ShapeDtypeStruct((rows, width), dtype), jax.ShapeDtypeStruct((rows, 1), jnp.int32),
            *(jax.ShapeDtypeStruct(() if param.axes == Kind.HEAD else (rows, width), param.value.dtype)
              for param in hooks.query.sides)).jaxpr)
    # A keyed launch's hook is its fingerprint (`tuning.program`), whose lowering its live call established.
    if callable(hooks.score.fn):
        traced += _score_jaxprs(hooks.score.fn, tuple((param.axes, param.value.dtype)
                                                       for param in hooks.score.sides),
                                tile, differentiated=differentiated)
    if callable(hooks.output.fn):
        traced.append(jax.make_jaxpr(hooks.output.fn)(
            jax.ShapeDtypeStruct((tile[0], value_width), jnp.float32),
            *(_entries(kind_of(param), param.value.dtype, (tile[0], value_width))
              for param in hooks.output.sides)).jaxpr)
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
    if hooks is None or not callable(hooks.score.fn):
        return False, False
    params = param_types(hooks.score.sides)
    return _score_places(hooks.score.fn, params)


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
    if hooks is None or not callable(hooks.score.fn):
        return 0
    params = param_types(hooks.score.sides)
    return _varying_slopes(hooks.score.fn, params, graded)


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
    if not callable(hooks.score.fn):
        return True
    params = param_types(hooks.score.sides)
    return _transposable(hooks.score.fn, params)


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
        elif not (layout.entrywise(eqn) or layout.placed(eqn)):
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
    kinds lay them out (`Hooks.score`'s sides', `[H]`, `[B, H, Qp]`, `[B, H, Kp]`, `[B, H, Qp, Kp]` or
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
    query: tuple[jax.Array, ...] = ()
    """The query prologue's parameters (`Hooks.query`'s sides'), `[H]`, `[B, H, Qp]` or `[H, D]`, which a
    consumer reads once a query tile where it takes the prologue (`attention_mosaic_forward._query_of`)."""


def _hooked_of(hooks: Hooks, call: PaddedCall, unit: int) -> _Hooked:
    """`hooks` at the padded call's shapes (`padded_call`), the value widths of the output hook's parameters
    and the initial state's sums in whole rows of `unit` entries (`_columns`), as the launch's values are."""
    initial = None
    if hooks.initial is not None:
        state = hooks.initial
        widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - state.acc.shape[-1]),)
        acc = jnp.pad(call.rows(state.acc), widened)
        initial = call.rows(state.m, -jnp.inf), call.rows(state.total), _columns(acc, unit)
    output = tuple(_columns(value, unit) if param.axes == Kind.OUTPUT else value
                   for value, param in zip(call.output_params, hooks.output.sides, strict=True))
    return _Hooked(call.score_params, output, initial,
                   placed_draws(hooks.drop, call.query.shape[1], call.key_rows), query=call.query_params)


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


def param_types(params: Iterable[Side]) -> tuple:
    """Score-hook parameters as a launch's kernels load them: each its `Kind` (`kind_of`, a plain tuple of
    axes named as the kind it is) and dtype (`kept_dtype`)."""
    return tuple((kind_of(param), kept_dtype(param.value.dtype)) for param in params)


def streams_operands(family: mosaic.Family) -> bool:
    """Whether a launch on `family` loads its score hook's operands on its streams (the spec's load contract,
    kernel-templates.md's hooks: a KEY's and a TILE's blocks ride the stream of keys, row 5741): where a
    producer role loads the streams (`Family.roles`: sm90's and sm100's rings). A family whose consumer
    issues every load (mma.sync's pipeline, a TPU core's, consumer-fed rings) reads them where its step
    does."""
    return any(role.name == "producer" for role in family.roles(1).roles)


def score_stream(index: int) -> str:
    """The stream score-hook parameter `index` rides (`score_streams`)."""
    return f"score{index}"


def riding(family: mosaic.Family, kinds: Iterable[Kind], along: str) -> tuple[int, ...]:
    """The score-hook parameters, by index among `kinds`, that ride a launch's stream along its axis `along`
    (the keys of the forward's and dQ's walks, the queries of dK/dV's): each whose kind lays that axis
    (its `Kind`), where `family` loads its operands so (`streams_operands`)."""
    if not streams_operands(family):
        return ()
    # A band rides a walk along the keys, its block at the column its program's query tile's origin decides
    # (`score_views`); along the queries each step's tile has its own origin, read where its step reads it.
    return tuple(index for index, kind in enumerate(kinds)
                 if (along == "keys" if kind is Kind.BAND else along in kind))


def score_streams(family: mosaic.Family, params: tuple, along: str, held: int, step: int) -> dict:
    """The blocks the riding parameters of `params` (each its `Kind` and dtype, `riding`) stream, as
    `Family.streams` takes them: a step's `step` entries along the walk's axis, of a tile the program's
    `held` rows of the axis it holds (its queries, keys first its keys), of a vector one row."""
    kinds = tuple(kind for kind, _ in params)
    return {score_stream(index): ((held if kinds[index] in TILED else 1, step), params[index][1])
            for index in riding(family, kinds, along)}


def score_views(family: mosaic.Family, refs: tuple, kinds: tuple, along: str, row, head, held,
                band: tuple[int, int] | None = None) -> dict:
    """The views the riding parameters' streams load a step's block from (`score_streams`, `rings.viewed`),
    at the shared batch row and head of each (`shared_place`): a tile's `[Qp, Kp]` (keys first `[Kp, Qp]`)
    at the program's `held` tile and the step's, a vector's `[H, N]` at its head and the step's; a band's
    `[Qp, Kb]` at the held tile and the step's keys' block of its band from the tile's origin (`band`: the
    held tile's rows and a step's keys; its origin's blocks off, clamped within the band: a step past it
    reads a block it leaves, `_param`'s zeros there)."""
    views = {}
    paired = band_paired(refs, kinds)
    for index in riding(family, kinds, along):
        ref = refs[index]
        shared_row, shared_head = shared_place(ref.shape, row, head)
        if kinds[index] is Kind.BAND:
            assert band is not None and paired[index] is not None, "a band rides by its origins and steps"
            (rows, step), origins = band, paired[index]
            origin_row, _ = shared_place(origins.shape, row, 0)
            off = lax.div(family.scalar_of(origins.at[origin_row, 0, pl.ds(held * rows, 1)]), step)
            last = ref.shape[-1] // step - 1
            views[score_stream(index)] = (ref.at[shared_row, shared_head],
                                          lambda at, off=off, last=last: (held, jnp.clip(at - off, 0, last)))
        elif kinds[index] is Kind.TILE:
            views[score_stream(index)] = (ref.at[shared_row, shared_head], lambda step: (held, step))
        else:
            views[score_stream(index)] = (ref.at[shared_row], lambda step, at=shared_head: (at, step))
    return views


def held_params(family: mosaic.Family, refs: tuple, kinds: tuple, row, head, first, count: int, *,
                along: str = "keys") -> tuple:
    """A walk's score-hook parameters it holds across its steps, read once before it (the spec's load
    contract, kernel-templates.md's hooks: a query tile's once), each step taking them (`_param`'s `held`),
    by their `kinds`: a HEAD's scalar and the vector of the tile's held axis (a QUERY's, or where the walk
    is `along` the queries a KEY's) of its `count` entries from `first`, in the accumulator's row layout;
    None for a parameter along the walk's own axis, which each step reads at its own entries."""
    vector = Kind.QUERY if along == "keys" else Kind.KEY
    origins = set(band_origins_of(kinds).values())

    def held(ref, kind, index):
        if kind is Kind.HEAD:
            return ref[head]
        if kind is not vector or index in origins:
            # A band's origins its tiles read as a scalar (`_band_param`), no vector of them.
            return None
        shared_row, shared_head = shared_place(ref.shape, row, head)
        places = {"batch": shared_row, "heads": shared_head, kind[-1]: pl.ds(first, count)}
        return family.rows_of(ref.at[axes.placed(kind, places)])

    return tuple(held(ref, kind, index) for index, (ref, kind) in enumerate(zip(refs, kinds, strict=True)))


def _param(family: mosaic.Family, ref, kind: Kind, row, head, first, like, k_first=None, *,
           keys_first: bool = False, held=None, slot=None, at: tuple = (0, 0), origins=None):
    """A hook parameter's values for a tile `like` of queries from `first` (and keys from `k_first`, a score
    tile's), as `kind` lays them out: the head's scalar, else the tile's entries (a score tile's `[bq, bk]`,
    an output tile's `[bq, Dv]`) in the accumulator's layout, its queries' and keys' broadcast. A score hook
    reads them and its coordinates as the tile's entries, where the reference and the other mainloops give a
    query's and a key's as `[bq, 1]` and `[1, bk]`: Mosaic GPU's layouts hold no singleton axis an elementwise
    operation takes (jax 0.11.2's layout inference finds none for a `[64, 1]` or `[1, 64]` array's convert or
    compare, a lowering probe for an H100), so a hook that reshapes or indexes an axis fails to lower here,
    never silently; a pointwise hook is the same function of either. A score tile whose rows are its keys
    (`keys_first`, `[bk, bq]`) reads a tile parameter's entries from its keys-first copy `[B, H, Kp, Qp]`. A
    walk's `held` value of it (`held_params`), where it holds one, is the tile's, its vector broadcast along
    the tile's rows; a step's `slot` of it where it rides the walk's stream (`score_streams`), the tile's
    share of the slot from `at` (its rows' first and its columns'), a vector's along the tile's columns, the
    stream's axis. A band's (`Kind.BAND`) the tile at the keys' column of its query tile's band from their
    origin (`origins`, its QUERY side after it, a scalar a query tile: one origin a `band.BAND_GROUP`),
    zeros where the tile lies past the band (`band.band_column`), its keys-first copy `[B, H, Kb, Qp]`'s
    where its rows are its keys."""
    queries, keys = like.shape[::-1] if keys_first else like.shape
    if kind is Kind.BAND:
        return _band_param(family, ref, origins, row, head, first, k_first, queries, keys,
                           keys_first=keys_first, slot=slot, at=at)
    if held is not None:
        return held if kind is Kind.HEAD else algebra.expand(held, like)
    if slot is not None:
        rows, columns = like.shape
        if kind is Kind.TILE:
            return family.tile_of(slot.at[pl.ds(at[0], rows), pl.ds(at[1], columns)])
        return algebra.across(family.columns_of(slot.at[0, pl.ds(at[1], columns)]), like)
    if kind is Kind.HEAD:
        return ref[head]
    if kind is not Kind.OUTPUT:
        row, head = shared_place(ref.shape, row, head)
    # Its block at the tile's place along each of its axes (its `Kind`), a tile parameter's keys-first
    # copy's keys before its queries.
    laid = kind[:2] + kind[2:][::-1] if keys_first and kind is Kind.TILE else kind
    places = {"batch": row, "heads": head, "queries": pl.ds(first, queries), "keys": pl.ds(k_first, keys)}
    block = ref.at[axes.placed(laid, places)]
    if kind in (Kind.TILE, Kind.OUTPUT):
        return family.tile_of(block)
    # A query's or a key's along the tile's rows or its columns, as the tile lays its queries.
    along_rows = (kind is Kind.QUERY) != keys_first
    return (algebra.expand(family.rows_of(block), like) if along_rows
            else algebra.across(family.columns_of(block), like))


def _band_param(family: mosaic.Family, ref, origins, row, head, first, k_first, queries: int, keys: int, *,
                keys_first: bool, slot=None, at: tuple = (0, 0)):
    """A band side's tile (`_param`'s): its query tile's origin read as a scalar, the tile at the keys'
    column within the band, from the walk's stream where it rides it (`slot`, `score_views`' block at the
    column, the tile's share from `at`) or from the band itself, zeros past it."""
    origin_row, _ = shared_place(origins.shape, row, 0)
    origin = family.scalar_of(origins.at[origin_row, 0, pl.ds(first, 1)])
    row, head = shared_place(ref.shape, row, head)
    column, inside = band.band_column(origin, k_first, keys, ref.shape[-2 if keys_first else -1])
    if slot is not None:
        tile = family.tile_of(slot.at[pl.ds(at[0], queries), pl.ds(at[1], keys)])
        return mosaic.selected(inside, tile, jnp.zeros(tile.shape, tile.dtype))
    places = {"batch": row, "heads": head, "queries": pl.ds(first, queries), "band": pl.ds(column, keys)}
    laid = Kind.BAND[:2] + Kind.BAND[2:][::-1] if keys_first else Kind.BAND
    tile = family.tile_of(ref.at[axes.placed(laid, places)])
    # Zeros past the band, selected (a splat predicate's, `mosaic.selected`), not scaled: a clamped column's
    # -inf bias times zero is NaN.
    return mosaic.selected(inside, tile, jnp.zeros(tile.shape, tile.dtype))


def band_paired(refs: tuple, kinds: tuple[Kind, ...]) -> tuple:
    """Each score side's origins where it is a band's (its QUERY side after it, `band_origins_of`), else
    None: what `_param` reads a band by."""
    pairs = band_origins_of(kinds)
    return tuple(refs[pairs[index]] if index in pairs else None for index in range(len(kinds)))


ORIGINS_READ = 0
"""What a band's origins side gives its hook (`Kind.BAND`'s, `template.Banded`, which takes and leaves it):
none of its values, its tiles reading them as a scalar a query tile (`_band_param`)."""


def params_read(refs: tuple, kinds: tuple[Kind, ...], read: Callable) -> tuple:
    """Each score side's tile, `read(index, ref, kind, origins)`'s (a `_param` of it), a band's read by its
    origins (`band_paired`), and its origins' side `ORIGINS_READ`, which its hook leaves."""
    origins = band_paired(refs, kinds)
    skipped = set(band_origins_of(kinds).values())
    return tuple(jnp.int32(ORIGINS_READ) if index in skipped else read(index, ref, kind, origins[index])
                 for index, (ref, kind) in enumerate(zip(refs, kinds, strict=True)))


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
    walk_lo: jax.Array | None = None
    walk_offset: jax.Array | None = None
    walk_count: jax.Array | None = None


def indexed_views(views: dict, tile: Callable | None) -> dict:
    """`views` (`rings.viewed`'s, each a ref and its step's block indices) of a walk whose step `j` reads key
    tile `tile(j)` (`_Program.tile_of`); the same views where the walk's steps are its tiles (None)."""
    if tile is None:
        return views
    found = tile
    return {name: (ref, lambda step, index=index: index(found(step))) for name, (ref, index) in views.items()}


def indexed_walk(spans: KeySpans, pair: int, block_k: int) -> tuple[jax.Array, ...]:
    """A call of runs' walks over the key tiles its query blocks of `pair` queries select (the indexed form,
    docs/design/kernel-templates.md §5): each block's ranges of tiles (`BlockMap.ranges`, merged, sorted),
    flattened `[B, Qp // pair * R]`, their first tiles (`walk_lo`) and each range's first step among the
    block's (`walk_offset`, its ranges' lengths before it), and each block's step count (`walk_count`,
    `[B, Qp // pair]`): its walk steps those alone, the tiles between its ranges none of its queries reads."""
    ranges = block_map(spans, pair, block_k).ranges
    assert ranges is not None, "a call's indexed walk reads its runs"
    lo, hi = ranges[..., 0], ranges[..., 1]
    lengths = hi - lo
    offsets = jnp.cumsum(lengths, axis=-1) - lengths
    rows = lo.shape[0]
    return lo.reshape(rows, -1), offsets.reshape(rows, -1), jnp.sum(lengths, axis=-1)


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
    """A launch's walk of its tiles as the work it issues reads it (`walks`): its padded `extents` (its
    arrays' query rows and keys, whole programs of both), its tiles' `blocks` (a consumer's query block and
    the key block), the tiles a program holds on its walk's axis (`holders`: its consumers' query blocks, or
    the key blocks of a walk over query tiles, `keyed`), and whether each of them walks every tile any of its
    program's holders reads (`common`: `Family.walk_own` on a layout whose consumers feed their rings, a
    shared dQ's or staged logits' dK/dV walk), a tile its own does not see masked whole; and the consumers
    that take each of its steps together, each its share of the tile (`sharing`: the fused backward's over
    shared keys, `attention_mosaic_coop`, its consumers each a share of the key block's columns); and the
    contiguous slices a tile's walk is split into, a program each (`splits`, `_Program._sliced`)."""

    extents: tuple[int, int]
    blocks: tuple[int, int]
    holders: int = 1
    keyed: bool = False
    common: bool = False
    sharing: int = 1
    splits: int = 1


def walks(spec: algebra.Forward, tile_walk: TileWalk) -> tuple[int, int] | None:
    """The steps a launch walking `tile_walk` takes, every consumer's of a head, and the steps the longest of
    its programs walks, from one traversal of its holders' tiles (`_tile_ranges`). Its steps: each tile
    holding a pair its call's form reaches, whole, the first key aligned to the first query, as its program
    reads them (a query block's key tiles up to its diagonal, `_Program.walked`, or a key block's query tiles
    from it, `_Program.readers`): every padded row's (a padded consumer walks its keys as a real one does), a
    causal diagonal tile's masked part computed with the rest (on an H100, the tensor pipes' counts of a
    causal call at 2048 queries and 128-row tiles: 6.25% past its reached pairs, row 3948), and a common
    walk's every tile its program's holders read in a slice for each holder, one wholly past the keys too
    (`algebra.Forward.walked_keys`). The work a consumer's step emits (`emission.Role.step`) times these is
    its walks' (`cost.Stated.steps`, `emission.Emitted.issued`). Its longest program's (`cost.Stated.longest`,
    its span): each slice of a program's walk (`slice_of`, each of its holders' alike) as its producer
    streams it, the tiles any of them reads (`_streamed`), as the kernel slices and streams them: ten tiles
    in three slices walk 4, where their mean is 10/3 (row 8844), and a causal pair's second slices of tiles
    `[1, 3)` and `[2, 4)` stream 3 (the base review of 71686face, F3), its holders' common walk of them 16
    steps over a causal 256 by 256 call at 64 by 64 tiles where the envelope sliced after was 12 and 2 (of
    369d0ad7b, F3). None for a spans call."""
    ranges = _tile_ranges(spec, tile_walk)
    if ranges is None:
        return None
    holders, splits = tile_walk.holders, tile_walk.splits
    steps = longest = 0
    for first in range(0, len(ranges), holders):
        program = ranges[first:first + holders]
        for split in range(splits):
            # Each holder's slice first, then the tiles any of them reads, as the kernel streams them
            # (`_Program.reach`, `_streamed`): a common walk's every holder walks them all.
            reaches = [slice_of(lo, hi, split, splits, operator.floordiv) for lo, hi in program]
            lo, hi = _streamed(reaches, min, max)
            steps += len(program) * (hi - lo) if tile_walk.common else sum(hi - lo for lo, hi in reaches)
            longest = max(longest, hi - lo)
    return steps * tile_walk.sharing, longest


def _tile_ranges(spec: algebra.Forward, tile_walk: TileWalk) -> list[tuple[int, int]] | None:
    """Each holder's walk of `tile_walk` as the tiles `[lo, hi)` it reads, by its program's own rules (a query
    block's key tiles up to its diagonal, `attention.key_tiles`; a key block's query tiles from it,
    `first_reader`); None for a spans call."""
    if spec.form == "spans":
        return None
    (queries, keys), (block_q, block_k) = tile_walk.extents, tile_walk.blocks
    query_tiles, causal = queries // block_q, spec.form == "causal"
    if tile_walk.keyed:
        # A key block's readers, as its program finds the first (`first_reader`).
        return [(first_reader(tile * block_k, block_q, query_tiles, spec.keys, causal, minimum=min,
                              div=operator.floordiv, where=_chosen), query_tiles)
                for tile in range(keys // block_k)]
    # A query block's key tiles, as its program walks them (`_Program.walked`).
    return [(0, min(key_tiles(spec.form, tile * block_q, block_q, block_k, spec.keys, minimum=min,
                              div=operator.floordiv)[1], keys // block_k))
            for tile in range(query_tiles)]


def first_reader(k_first, block_q: int, q_steps: int, keys: int, causal: bool, *,
                 minimum: Callable = jnp.minimum, div: Callable = lax.div, where: Callable = jnp.where):
    """The first query tile that sees the key block from `k_first` in a full or causal call of `keys` keys,
    by `minimum`, `div` and `where` (a traced walk's, `_Program._first_reader`; a price's integers', `walks`):
    every one, or those from its diagonal (a causal call of more keys than queries leaves its last keys
    unseen); a block wholly past the keys none, from the last."""
    lo = minimum(div(k_first, block_q), q_steps) if causal else 0
    return where(k_first >= keys, q_steps, lo)


def _chosen(condition: bool, chosen, otherwise):
    """`jnp.where`'s choice of host values (`first_reader`)."""
    return chosen if condition else otherwise


def stated(spec: algebra.Forward, batch: int, heads: int, *arrays, tile_walk: TileWalk) -> Stated | None:
    """A launch's statement of its cost (`cost.Stated`), its work its kernel's emission's
    (`emission.costed`): the steps its walk takes, every head's, and its longest program's (`walks`), and the
    bytes of each of `arrays` once; None for a spans call, whose steps no launch knows (`reached`): its cost
    counts as unsized, never as 0. The model's share of its work the launch declares apart (`modelled`)."""
    walked = walks(spec, tile_walk)
    return None if walked is None else Stated(walked[0] * batch * heads, stored(*arrays), walked[1])


def modelled(spec: algebra.Forward, batch: int, heads: int, queries: int, widths: int) -> int | None:
    """The model's share of a launch's work (its emission's, `stated`), which its launch declares to Dew's
    FLOP count (`Family.kernel`'s `model_flops`): `widths` product columns of the model's products it owns
    (the forward's `Q K^T` and `P V`, the backward's two transposes of each, as the library's fused attention
    counts them), a recomputed product or a cotangent's later parts the hardware's alone; None for a spans
    call."""
    pairs = reached(spec, queries)
    return None if pairs is None else 2 * batch * heads * pairs * widths


def _spans_of(spec: algebra.Forward, call: PaddedCall, block_q: int, block_k: int, *, readers: bool = False,
              pair: int | None = None) -> _Spans | None:
    """A spans call's `_Spans` at `block_q` by `block_k` tiles over its padded keys (None for a full or
    causal call, whose programs take their tiles from their places), with each key block's query readers
    where a launch walks key blocks (`readers`: the backward's; computed for no other, two reductions a
    decode step's forward ran for nothing, ET-DECODE8-h100). A call's runs (`KeySpans`) lie run by
    run, `[B, R, Qp]`, so a program reads each run's bounds as a row (`_coords`); its blocks walk their
    runs' envelopes (`attention.block_map`), each tile masked by the union of its queries' runs
    (`algebra.seen`), or where a launch walks query blocks of `pair` queries the tiles their runs select
    alone (`indexed_walk`)."""
    if spec.form != "spans":
        return None
    if call.per_span > 1:
        # Each interval a position's packed rows (`KeySpans.per_span`): its programs read them by position
        # and form their tiles' bounds themselves (`_Program.reach`, `_Program.bounds`), no block map in XLA.
        return _Spans(call.start, call.end, None, None, None, None, None, None)
    blocks = block_map(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key_rows),
                       block_q, block_k)
    start, end = (jnp.moveaxis(bound, 2, 1) if bound.ndim == 3 else bound for bound in (call.start, call.end))
    walk = (indexed_walk(KeySpans(call.start, call.end, call.query_ids, call.key_ids, call.key_rows), pair,
                         block_k) if pair is not None and call.start.ndim == 3 else (None, None, None))
    return _Spans(start, end, blocks.lo, blocks.hi, blocks.full_lo, blocks.full_hi,
                  blocks.q_lo if readers else None, blocks.q_hi if readers else None,
                  call.query_ids, call.key_ids, blocks.query_doc, blocks.key_doc, *walk)


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
    products: tuple[mosaic.Product, ...] = ()
    """Its walks' steps' products declared as data (`mosaic.Family.walk`'s `products`, which the family
    places): a forward's logits and its weighted values' sums (`attention_mosaic_forward.declared`)."""
    value_columns: int | None = None
    """Where a forward's values are its keys' leading columns (`MosaicForward.in_key`), the first of the key
    columns its value tile views in the key slot; None where it streams its values."""
    training: bool = False
    """Whether this launch is a training/backward walk where padding query rows must never see unmasked
    interior tiles."""

    @property
    def ahead(self) -> bool:
        """Whether it issues a product a step ahead of its read: the logits, where it issues each key
        tile's a step before its softmax reads them (`attention_mosaic_forward.logits_ahead`, row 5368)."""
        return bool(mosaic.ahead_streams(self.products))

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
        if not whole_tiles(self.family, self.tiles, parts=self.parts):
            raise ValueError(f"{self.tiles}: query and key tiles of whole {self.family.rows} and "
                             f"{self.family.walked_rows} rows and {most_consumers(self.family, self.tiles)} "
                             f"consumers at most ({self.family.name})")

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

        if spans.walk_count is not None:
            # An indexed walk's steps (`indexed_walk`): its pair's selected tiles, each step's read through
            # `tile_of`.
            tile = self.tile_of(first, spans, row)
            assert tile is not None
            count = at(spans.walk_count, lax.div(first, self.tiles.consumers * self.tiles.block_q))
            return *self._sliced(jnp.int32(0), count), lambda index: whole(tile(index))
        return *self._sliced(at(spans.lo, block), at(spans.hi, block)), whole

    def tile_of(self, first, spans: _Spans | None, row) -> Callable | None:
        """The key tile a walk's step `index` reads for the query block from `first` (batch row `row`): None
        where it is the step's own index, but where the call's runs select its pair's tiles
        (`indexed_walk`), the step's tile among its pair's ranges, found by a search of their first steps
        (log2 R reads a step, R the ranges a pair holds; held across the walk, 2R scalars would hold 128
        registers at R 64)."""
        if spans is None or spans.walk_count is None:
            return None
        walk_lo, walk_offset = spans.walk_lo, spans.walk_offset
        assert walk_lo is not None and walk_offset is not None
        ranges = walk_lo.shape[1] // spans.walk_count.shape[1]
        base = lax.div(first, self.tiles.consumers * self.tiles.block_q) * ranges
        at = functools.partial(self._entry, row=row)

        def tile(index):
            # The last range whose first step is at or before the step's (an empty range's first step its
            # successor's: the search lands on the latest of equal ones, which holds the step).
            low, span = jnp.int32(0), ranges
            while span > 1:
                half = span // 2
                low = jnp.where(at(walk_offset, base + low + half) <= index, low + half, low)
                span -= half
            return at(walk_lo, base + low) + index - at(walk_offset, base + low)

        return tile

    def _positions(self, first, spans: _Spans, row, step, carry):
        """`carry` through `step(carry, position, start, end, holds)` over the positions a query block from
        `first` holds rows of, in a call whose intervals each hold `per_span` queries (`KeySpans.per_span`):
        each one's index, its interval `[start, end)` (batch row `row`'s) and whether the block holds its
        rows; as many as the block can hold. A loop, its bounds read a position at a time: read at once,
        unrolled, their addresses and values held into the walk (a decode prefill's 64-row block held 33
        positions', ptxas holding its sm89 launch at 248 registers where its walk alone needs 160, row
        5760)."""
        positions = spans.start.shape[1]
        held = min(positions, self.tiles.block_q // self.per_span + 1)
        at = lax.div(first, self.per_span)

        def body(offset, carry):
            position = at + offset
            # A position past the call's holds no row: read its last one's, unused.
            index = jnp.minimum(position, positions - 1)
            holds = (position < positions) & (position * self.per_span < first + self.tiles.block_q)
            return step(carry, position, self._entry(spans.start, index, row=row),
                        self._entry(spans.end, index, row=row), holds)

        return lax.fori_loop(0, held, body, carry)

    def _positions_bounds(self, first, spans: _Spans, row):
        """`(lo, hi, full_lo, full_hi)` key-tile bounds for a per-position decode block (`_positions`):
        `[lo, hi)` covers every visible key, and `[full_lo, full_hi)` covers interior tiles seen whole by
        every real query row in the block. Padding rows (`first + block_q > queries`) only disable whole
        tiles during training/backward; inference decode masks padding rows out of status and decided
        verdicts instead so interior tiles skip per-element masking (row 4045)."""
        block_k, big = self.tiles.block_k, jnp.iinfo(jnp.int32).max

        def step(carry, _, start, end, holds):
            lo, hi, full_lo, full_hi = carry
            seen = holds & (end > start)
            return (jnp.where(seen, jnp.minimum(lo, lax.div(start, block_k)), lo),
                    jnp.where(seen, jnp.maximum(hi, pl.cdiv(end, block_k)), hi),
                    jnp.where(holds, jnp.maximum(full_lo, pl.cdiv(start, block_k)), full_lo),
                    jnp.where(holds, jnp.minimum(full_hi, lax.div(end, block_k)), full_hi))

        lo, hi, full_lo, full_hi = self._positions(
            first, spans, row, step, (jnp.int32(big), jnp.int32(0), jnp.int32(0), jnp.int32(big)))
        lo = jnp.minimum(lo, hi)
        if self.training and self.padded(first) is not False:
            full_hi = jnp.where(self.padded(first), full_lo, full_hi)
        return lo, hi, full_lo, full_hi

    def _positions_reach(self, first, spans: _Spans, row):
        """`reach`'s for a call whose intervals each hold `per_span` queries (`_positions_bounds`)."""
        lo, hi, full_lo, full_hi = self._positions_bounds(first, spans, row)

        def whole(index):
            return (index >= full_lo) & (index < full_hi)

        return *self._sliced(lo, hi), whole

    def reach_segmented(self, first, spans: _Spans | None, row):
        """`reach` paired with `forward_whole_run` from one `_positions_bounds` scan on per-position decode
        spans (`spans.lo is None`, row 4045), avoiding a duplicate span-ref scan."""
        if spans is not None and spans.lo is None:
            lo, hi, full_lo, full_hi = self._positions_bounds(first, spans, row)

            def whole(index):
                return (index >= full_lo) & (index < full_hi)

            run = None if self.training else (full_lo, full_hi)
            return *self._sliced(lo, hi), whole, run
        lo, hi, whole = self.reach(first, spans, row)
        return lo, hi, whole, self.forward_whole_run(first, spans)

    def bounds(self, first, spans: _Spans, row) -> tuple[tuple, jax.Array | None]:
        """A query block's intervals and its ids as `_coords` reads them (`_query_bounds`), read once before
        its walk; a call whose intervals each hold `per_span` queries its rows' from their positions', a
        padding row's empty."""
        family, count = self.family, self.tiles.block_q
        if spans.lo is not None:
            return _query_bounds(family, spans, row, first, count)
        place = lax.div(first + family.iota((count,), 0), self.per_span)

        def step(carry, position, low, high, holds):
            start, end = carry
            mine = (place == position) & holds
            return jnp.where(mine, low, start), jnp.where(mine, high, end)

        start, end = self._positions(first, spans, row, step, (place * 0, place * 0))
        return ((family.as_row(start), family.as_row(end)),), None

    def _entry(self, ref, index, *, row):
        """A block map's entry `ref[row, index]` as the family reads a launch's scalar (`Family.scalar_of`: a
        GPU thread's load, a TPU core's copy into its scalar memory, whose operands stay in HBM, where a load
        lowers on no core: a v5e's documents call refused to lower, "ANY memory space can only be accessed
        using async_copy")."""
        return self.family.scalar_of(ref.at[row, pl.ds(index, 1)])

    def _sliced(self, lo, hi) -> tuple[jax.Array, jax.Array]:
        """`[lo, hi)` key tiles, this program's slice of them (`split`, `slice_of`); all of one slice's."""
        return (lo, hi) if self.splits == 1 else slice_of(lo, hi, self.split(), self.splits)

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

    def segmented(self, step: Callable, run: tuple[jax.Array | None, jax.Array] | None, lo, hi) -> tuple:
        """A key block's walk of query tiles `[lo, hi)` as `Family.walk`'s segments, `step` each: where its
        whole tiles run unbroken (`run`, `whole_run`), those one segment of their own, unmasked, and the
        tiles around them masked, so no step chooses at run time, whose two branches' registers join
        (FlashAttention 3's masked and unmasked iterations); where `run[0]` is None (`forward_whole_run`),
        the unmasked run starts at `lo` so only two segments emit; else one segment choosing a tile at a
        time."""
        if run is None:
            return ((step, lo, hi),)
        raw_start, raw_end = run
        if raw_start is None:
            end = jnp.clip(raw_end, lo, hi)
            return ((functools.partial(step, masked=False), lo, end),
                    (functools.partial(step, masked=True), end, hi))
        start = jnp.clip(raw_start, lo, hi)
        end = jnp.clip(jnp.maximum(raw_end, raw_start), lo, hi)
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

    def forward_whole_run(self, first, spans: _Spans | None) -> tuple[jax.Array | None, jax.Array] | None:
        """The key tiles `[0, whole_end)` a causal inference forward/dQ query block from `first` sees whole
        (`reach`), one unbroken run for `segmented` (rows 10790, 4839); per-position decode interior runs
        come from `reach_segmented`."""
        if spans is not None or not self.causal:
            return None
        walked = self.walked(first)
        return None, jnp.where(self.padded(first), jnp.int32(0), self.whole(first, walked))

    def _first_reader(self, k_first, q_steps: int) -> jax.Array:
        """The first query tile that sees the key block from `k_first` (`first_reader`)."""
        return first_reader(k_first, self.tiles.block_q, q_steps, self.keys, self.causal)

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


def _streamed(reaches: list, minimum: Callable = jnp.minimum, maximum: Callable = jnp.maximum) -> tuple:
    """The tiles a program streams: those any of its consumers' `reaches` (`[lo, hi)` each) reads, by
    `minimum` and `maximum` (a traced walk's; a price's integers' `min` and `max`, `walks`)."""
    return (functools.reduce(minimum, [lo for lo, _ in reaches]),
            functools.reduce(maximum, [hi for _, hi in reaches]))


def slice_of(lo, hi, split, splits: int, div: Callable = lax.div) -> tuple:
    """The `split`th of the `splits` contiguous slices of tiles `[lo, hi)` (`_Program._sliced`), as long as
    the others but for rounding, which partition them, by `div` (a traced walk's `lax.div`; a price's
    integers' `operator.floordiv`, `walks`), alike on a count of tiles, never negative."""
    count = hi - lo
    return lo + div(count * split, splits), lo + div(count * (split + 1), splits)


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


def head_flat(family: mosaic.Family, dtype, width: int) -> bool:
    """Sub-32-bit TPU arrays whose width divides the TPU lane unit (`column_unit`, 128 lanes), bitcast
    to `[B, S, H * D]` with zero XLA transpose on both v5e and v6e (row 10683)."""
    return (family.copies_tiled_rows and jnp.dtype(dtype).itemsize < 4 and width > 0
            and width % family.column_unit(dtype) == 0)


def head_first(family: mosaic.Family, dtype, width: int | None = None) -> bool:
    """Sub-32-bit TPU arrays outside `head_flat` (`width % column_unit != 0`), transposed to
    `[B, H, S, D]` so axis 2 avoids `(2, 1)` bf16 subtile packing (row 10683)."""
    return (family.copies_tiled_rows and jnp.dtype(dtype).itemsize < 4
            and (width is None or not head_flat(family, dtype, width)))


def head_layout(family: mosaic.Family, array: jax.Array) -> jax.Array:
    """Pack `[B, S, H, D]` for `family`: bitcast (`head_flat`), swap (`head_first`), or leave in place."""
    width = array.shape[-1]
    if head_flat(family, array.dtype, width):
        return array.reshape(array.shape[0], array.shape[1], array.shape[2] * width)
    return jnp.swapaxes(array, 1, 2) if head_first(family, array.dtype, width) else array


def head_unlayout(family: mosaic.Family, array: jax.Array, heads: int) -> jax.Array:
    """Inverse of `head_layout`, restoring `[B, S, H, D]`."""
    if array.ndim == 3:
        return array.reshape(array.shape[0], array.shape[1], heads, array.shape[2] // heads)
    return jnp.swapaxes(array, 1, 2) if head_first(family, array.dtype, array.shape[-1]) else array


def head_slice(family: mosaic.Family, row, head, span, dtype, *, width: int | None = None) -> tuple:
    """Ref index for `head_flat` (`(row, span, pl.ds(head * width, width))`), `head_first`, or in-place."""
    if width is not None and head_flat(family, dtype, width):
        offset = pl.multiple_of(head * width, width) if isinstance(head, jax.Array) else head * width
        return (row, span, pl.ds(offset, width))
    return (row, head, span) if head_first(family, dtype, width) else (row, span, head)


def ref_width(ref, heads: int) -> int:
    """Per-head width of a 4D `[B, S, H, D]`/`[B, H, S, D]` or 3D `head_flat` `[B, S, H * D]` ref."""
    return ref.shape[-1] // heads if ref.ndim == 3 else ref.shape[-1]


def _views_of(family: mosaic.Family, refs: dict, row, head, *, heads: int | None = None) -> dict:
    """Stream views by name at `row` and `head` in `family`'s axis order (`head_slice`)."""
    return {name: (ref.at[head_slice(family, row, head, slice(None), ref.dtype,
                                     width=ref.shape[-1] if heads is None else ref_width(ref, heads))],
                   lambda j: (j, 0))
            for name, ref in refs.items()}

def powered_in_kernel(family: mosaic.Family, dtype, *, has_query_mod: bool = False,
                      parts: int = 1) -> bool:
    """Whether consumers apply `spec.prescale` in-kernel in their query stage (rows 5272, 11789)."""
    return has_query_mod or (parts == 1 and jnp.dtype(dtype).itemsize == 2 and family.fixes_up
                             and not family.places_on_role)


class BlockPlan(NamedTuple):
    """Forward value-column block plan (`forward_block_plan`): hook split, `out_dtype`, in-kernel hook."""
    split: bool
    out_dtype: Any
    output_hook: bool

    def epilogue_chunk(self, family: mosaic.Family, value_width: int, dtype, *,
                       training: bool = False, masked: bool = False, kept: bool | None = None) -> int:
        if (training or self.output_hook or masked or not family.budgets_registers
                or (family.places_on_role if kept is None else kept) or value_width <= 128):
            return 0
        ed = dtype if self.out_dtype is None else self.out_dtype
        swizzle = output_stage_swizzle(family, value_width, ed) or stores.STAGED_BYTES
        return stores.staged_columns(value_width, (ed,), swizzle=swizzle)


def output_stage_swizzle(family: mosaic.Family, value_width: int, out_dtype) -> int | None:
    """64B swizzle width for wide (>128) 16-bit register forwards (`stores.staged_columns`, row 5272)."""
    swizzle = swizzling.SWIZZLE_BYTES // 2
    itemsize = jnp.dtype(out_dtype).itemsize
    if (value_width <= 128 or not family.budgets_registers or not family.adds_through_shared
            or family.places_on_role or itemsize != 2
            or stores.staged_columns(value_width, (out_dtype,), swizzle=swizzle) * itemsize != swizzle):
        return None
    return swizzle


def forward_block_plan(total_v: int, columns: int | None, out_dtype, *, has_output_mod: bool,
                       training: bool) -> BlockPlan:
    """Derive hook placement and block `out_dtype` from `value_columns` (row 9542)."""
    split = not training and has_output_mod and value_columns(total_v, columns) < total_v
    return BlockPlan(split, jnp.float32 if split else out_dtype, has_output_mod and not split)


def value_columns(padded: int, columns: int | None) -> int:
    """The value columns a launch emits for padded width `padded` and optional block cap `columns`:
    capped at `padded` so slicing, resource planning and register sizing agree on oversized pins."""
    return min(columns, padded) if columns else padded


STATISTICS = MappingProxyType({"peak": jnp.float32, "total": jnp.float32, "anchor": jnp.float32,
                               "delta": jnp.float32, "chosen": jnp.int32})
"""The per-query statistics a dK/dV step reads (`attention_mosaic_scores._scores`' `peak`, `total`, `anchor`,
`delta` and `chosen`, the anchor key's index), in the order of their arrays, and their dtypes: the dK/dV
launch over a shared dQ streams each beside its queries, and the fused launch the first its block holds
(`attention_mosaic_fused.streamed_statistics`); row vectors a fused launch's queries first holds
(`backward_registers`), column vectors keys first, one at a time."""
STATISTICS_STREAM_ORDER: tuple[str, ...] = tuple(dict.fromkeys(("peak", "total", "delta", *STATISTICS)))
"""Stream priority for per-query statistics (`STATISTICS`): `peak`, `total`, and `delta` first so
the fused backward launch streams the hottest rows before `anchor` and `chosen`."""


def streamed_statistics_order(count: int | None = None, *, anchored: bool = True) -> tuple[str, ...]:
    """The first `count` (or all) per-query statistics streamed in priority order, excluding `chosen` when
    `not anchored` (row 11710)."""
    names = (STATISTICS_STREAM_ORDER if anchored
             else tuple(name for name in STATISTICS_STREAM_ORDER if name != "chosen"))
    return names if count is None else names[:count]


