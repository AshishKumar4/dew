"""What the attention template's Mosaic GPU launches share (`attention_mosaic`, design §6): the bundles it
runs on (`bundle`), its configs (`Tiles`, `BackwardTiles`), its legality rule (`refused`), and what a
forward's and a backward's programs read alike: the call's padding to a launch's blocks, its spans' block map
and each tile's coordinates (`_Program`, `_Spans`, `_coords`), its hooks' parameters and state (`_Hooked`,
`_param`)."""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import NamedTuple

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
    mosaic_sm90,
    operands,
    resources,
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
    power_of_two,
    shared_place,
    whole,
)
from dew.nn.kernels.templates.cost import estimate
from dew.nn.programs import CALLS, equations

QUALIFIED = frozenset({mosaic_sm90.WGMMA.name, mosaic_sm90.FED.name})
"""The bundles the attention mainloop is qualified on: wgmma's, with a producer and with consumers that
feed their own rings (`mosaic_sm90.FED`). mma.sync's (an A100's) joins once every launch it derives
compiles spill-free (the resource gate's sm80 cases, `GATED`; row 2078: check 1 held there,
ET-2078-A100CHECK3)."""


def bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
    """The bundle the attention mainloop runs on a device of `generation_name` for operands of `dtype`: the
    shared registry's (`bundles.of`) where attention is qualified on its family (`QUALIFIED`); None elsewhere,
    whose calls another mainloop takes by legality."""
    family = None if generation_name is None else bundles.of(generation_name, dtype)
    return family if family is not None and family.name in QUALIFIED else None


def layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
    """The bundles a launch of the attention mainloop takes on a device of `generation_name` for operands of
    `dtype`, in preference order (`bundles.layouts`: a producer's layout, then on warpgroup MMA the layout
    whose consumers feed their own rings, `mosaic_sm90.FED`; given the launch's `rows`, each issuer the
    device has, by the share of its rows they fill), where its family is qualified (`bundle`)."""
    if generation_name is None or bundle(generation_name, dtype) is None:
        return ()
    return tuple(family for family in bundles.layouts(generation_name, dtype, rows)
                 if family.name in QUALIFIED)


def bundle_of(tiles: "Tiles | BackwardTiles", dtype) -> mosaic.Family:
    """The bundle `tiles` run on: the current device's (`generation.gpu_target`) layout their family names,
    which their legality matched (`MosaicForward.legal`). A config names its physical layout; which configs
    a call derives is the qualification's (`layouts`), so a config pinned on a family not yet qualified (an
    A100's mma.sync, its resource witness) resolves too."""
    # Every bundle the device issues (a launch of a row each ranks them all, `bundles.layouts`).
    found = {family.name: family for family in bundles.layouts(generation.gpu_target().generation, dtype, 1)}
    family = found.get(tiles.family)
    if family is None:
        raise ValueError(f"{tiles}: this device's attention bundles are {sorted(found)}")
    return family
"""The instruction families this mainloop runs on (`generation.family`), each by its bundle: wgmma's. A
family without one (mma.sync's, until its thin mainloop) takes another mainloop (`MosaicForward.eligible`,
`MosaicBackward.eligible`)."""


def values_deferred(family: mosaic.Family, form: str, splits: int, dtype, dropped: bool) -> bool:
    """Whether a forward's plain walk issues each tile's P·V a step late
    (`attention_mosaic_forward._consumer`): half of FlashAttention 3's overlap within a consumer, its next
    `Q Kᵀ` and the step before's P·V in flight together (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp's
    `fwd_step`), the weights carried a step in registers; its softmax under the P·V, the other half, waits
    on weights in registers the running P·V does not read.
    Where its family's products run asynchronously (`Issuer.products_overlap`) over a producer's loads (a
    loading consumer would wait on its own late release, `Transport.feeds_itself`), every program walks a key
    tile (a full or causal call's whole launch: no spans, no splits), and a tile's weights are one bf16
    operand (no dropout's parts)."""
    return (family.products_overlap and not family.feeds_itself and form in ("full", "causal") and splits == 1
            and jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16) and not dropped)


def query_swizzle(family: mosaic.Family) -> int | None:
    """The swizzle a forward stages its query block at, the left operand of its logits' product: the
    narrowest where its family loads operands into registers (`Family.operands_in_registers`), 8-row tiles
    32 bytes wide, so a load of the block's fragments one instruction deep (16 of a 16-bit row) is a whole
    tile and its 8 rows' 16-byte reads take every bank once, where a plain block's 256-byte rows take one
    bank group 8 times; layout inference's otherwise (`mosaic.staged`). The query is written once a tile."""
    return mosaic.NARROWEST_SWIZZLE_BYTES if family.operands_in_registers else None


def transposed_swizzles(tiles: "Tiles | BackwardTiles", *names: str) -> dict[str, int]:
    """The swizzles `Family.streams` stages the streams `names` at, which a launch's products read
    transposed: the narrowest where `tiles` stage them so (`Tiles.narrow`), the family's own otherwise."""
    return dict.fromkeys(names, mosaic.NARROWEST_SWIZZLE_BYTES) if tiles.narrow else {}


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
    family: str = mosaic_sm90.WGMMA.name
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
    deferred: bool = False
    """Whether its plain inference walk issues each tile's P·V a step late (`values_deferred`), where that
    walk's registers hold the carried weights at these tiles (`MosaicForward.derived`)."""

    def __post_init__(self):
        if self.stages < 2:
            raise ValueError(f"{self}: two stages or more")

    @property
    def relaunch(self) -> "Tiles":
        """The natural form's rewalk's tiles (`algebra.walked_launches`), its P·V in its step (its own tiles'
        deferral not derived)."""
        return dataclasses.replace(self, block_k=self.relaunch_k or self.block_k,
                                   consumers=self.relaunch_consumers or self.consumers, relaunch_k=None,
                                   relaunch_consumers=None, relaunch_columns=None, natural_form=True,
                                   columns=self.relaunch_columns, deferred=False)


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
    family: str = mosaic_sm90.WGMMA.name
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


def _products(family: mosaic.Family, dtype, *products, narrow: bool = False, lhs_swizzle: int | None = None
              ) -> list:
    """The phases a step's products hold where its family's read their operands from registers
    (`resources.operand_fragments`; none where they read shared memory, wgmma's): each product
    `(rows, depth, columns, beside, register_lhs, rhs_transposed)` its operands' fragments beside the tiles
    `beside` it holds at once (its output, a tile it reads in registers), a left operand in registers
    (`register_lhs`: the weights, the score cotangents) no fragment of its own, a right one read as the
    transpose of its stored tile (`rhs_transposed`: the keys of the logits, the values of dP) loaded at its
    stage's swizzle granule (`Family.rhs_fragment_depth`; mma.sync: ET-CENSUS80, a 64 x 192 forward's keys'
    fragments 162 of its 291 registers at its peak, uncounted), the narrowest where its launch stages them
    `narrow` (`Tiles.narrow`); a left operand in shared memory loaded at its stage's (`lhs_swizzle`: a
    forward's query block, `query_swizzle`; the family's own where None)."""
    if not family.operands_in_registers:
        return []
    narrowest = mosaic.NARROWEST_SWIZZLE_BYTES if narrow else None
    return [list(beside) + resources.operand_fragments(family, rows, depth, columns, dtype, dtype,
                                                       lhs_parts=0 if register_lhs else 1,
                                                       rhs_transposed=transposed, lhs_swizzle=lhs_swizzle,
                                                       rhs_swizzle=narrowest if transposed else None)
            for rows, depth, columns, beside, register_lhs, transposed in products]


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


DROPPING = Refusal('dropout_places',
                  "the Mosaic mainloop draws dropout at its own places; a sequence exchange slice draws at "
                  "its global places, a table gather a tile (`DropPlaces`)")
"""Why this mainloop refuses a dropping call that draws at a slice's global places (`refused`)."""


def launch_scope(family: mosaic.Family, allocations: tuple, reducing_warpgroups: int, *, reduces: bool = True
           ) -> resources.Scope:
    """A launch's `allocations` (`(scoped, streams)`, `Family.scoped`'s, and those it scopes after its
    walks where it has them, `after_walks`) by lifetime (`Family.lifetimes`), with the algebra's
    reductions' scratch where it `reduces` (a launch reducing its scores, weights or cotangents across
    warps) and its `reducing_warpgroups`' scratch (`resources.Scope`)."""
    scoped, streams, *after = allocations
    held, walks = family.lifetimes(streams)
    return resources.Scope(program=(scoped, *held), walks=walks,
                           after=tuple((tile.shape, tile.dtype) for tile in (after[0] if after else ())),
                           reduces=reduces, reducing_warpgroups=reducing_warpgroups)


def extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A tile axis's extents on `family`, largest first: whole multiples of the rows a consumer's products
    cover (`Family.role_rows`) and of its swizzled rows (`Family.column_unit`; a tile's keys are its weights'
    product's depth, its queries the dK/dV launch's), at most what one copy moves (`Family.copy_extent`,
    where the family's copies bound it; else registers bound them, `fits`) and the call's `extent` in whole
    multiples (a longer tile holds padding alone)."""
    return _multiples(family, math.lcm(family.role_rows, family.column_unit(dtype)), extent)


def key_extents(family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A forward walk's key tile extents on `family`, largest first: whole multiples of its swizzled rows
    (`Family.column_unit`), a key tile's keys its weights' product's depth and no consumer's rows, within
    `extents`' bounds: mma.sync's guarded walk takes 32-key tiles, where its 64-row unit held no guarded
    state at width 128 (FS-GUARD80-2)."""
    return _multiples(family, family.column_unit(dtype), extent)


def _multiples(family: mosaic.Family, unit: int, extent: int) -> tuple[int, ...]:
    """Whole multiples of `unit`, largest first, at most what one of `family`'s copies moves and the call's
    `extent` in whole multiples (`extents`)."""
    top = -(-max(extent, 1) // unit)
    if family.copy_extent is not None:
        top = min(top, family.copy_extent // unit)
    return tuple(range(top * unit, 0, -unit))


def whole_tiles(family: mosaic.Family, tiles: "Tiles | BackwardTiles") -> bool:
    """Whether a program of `family` takes `tiles`: its query and key tiles positive whole multiples of the
    rows a role's products cover (`Family.rows`), and one to the family's count of consumers
    (`Family.consumers`), whose turns its walks are written for."""
    return (0 < tiles.consumers <= family.consumers and tiles.block_q > 0 and tiles.block_k > 0
            and tiles.block_q % family.rows == 0 and tiles.block_k % family.rows == 0)


def refused(dtype, width: int, value_width: int, hooks: Hooks | None, *, backward: bool,
            family: mosaic.Family) -> Refusal | None:
    """Why this mainloop takes no call of `dtype` operands, head widths `width` and `value_width` and `hooks`
    (its forward, or with `backward` its backward) on `family`'s bundle, or None: its legality rule, which
    the ops' eligibility (`MosaicForward.eligible`, `MosaicBackward.eligible`), the route and the stability
    suite read."""
    hooks = Hooks() if hooks is None else hooks
    sixteen = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
    if hooks.drop is not None and (hooks.drop.queries is not None or hooks.drop.keys is not None):
        return DROPPING
    if hooks.drop is not None and jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's dropped weights take scaled parts (`algebra.weights_parts`), which the mainloop's register
        # products do not take back (`mosaic.unscaled_part`).
        return Refusal('dropout_dtype', "the Mosaic dropping walk takes bf16 operands, not float16")
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


APPROXIMATED = frozenset({"exp", "exp2", "log", "log2", "log1p", "tanh", "logistic", "sin", "cos", "sqrt",
                          "rsqrt", "pow"})
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


def reducing(family: mosaic.Family, consumers: int, *hooks, guarded: bool = False) -> int:
    """The warpgroups of a launch of `consumers` whose arithmetic may reduce across warps implicitly, each
    reserving its share of Mosaic GPU's reduction scratch (`Family.kernel`'s `reducing_warpgroups`): a hook's
    unit row, which the fork selects by a reduction across warps (ledger 1428), and a `guarded` launch's
    keys' value magnitudes, its tile's columns reduced across the warps holding them
    (`Family.key_magnitudes`), or a launch's rows reduced to their block's flag (`guarded` too: the plain
    launch that flags its rows, `attention_mosaic_forward._Flags`), so every warpgroup of a launch that runs a
    hook (`hooks`, its score and output hooks, None where it has none) or guards; none otherwise."""
    if guarded or any(hook is not None for hook in hooks):
        return family.warpgroups(consumers)
    return 0


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
        # A call (a jitted helper's `jaxpr`, a custom derivative's `call_jaxpr`): its jaxpr's inputs its own.
        called = ("jaxpr" in eqn.params or "call_jaxpr" in eqn.params) and len(nested) == 1
        if called and len(nested[0].invars) == len(eqn.invars):
            reads = _live_inputs(nested[0], used)
        else:
            reads = [not nested] * len(eqn.invars)
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
        if not hit or equation.primitive.name == "stop_gradient":
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
        if any(jax_core.primitives.custom_vjp_call_p.name in _primitives(program) for program in inner):
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
    (`dew.nn.programs.ENTRYWISE`, a scalar's broadcast, a nested call's own operations so), each value it
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
        name = eqn.primitive.name
        nested = list(jax_core.jaxprs_in_params(eqn.params))
        if nested:
            # A call or a custom derivative's primal (whose rule `layout.transposed` checks).
            if name not in CALLS or not all(
                    _entrywise(inner, set(read), tile) for inner in nested):
                return False
        elif not layout.entrywise(eqn):
            return False
        tainted.update(eqn.outvars)
    return True


def _primitives(jaxpr: jax_core.Jaxpr) -> set[str]:
    """The names of the operations of `jaxpr` and of every jaxpr inside it (`dew.nn.programs.equations`)."""
    return {eqn.primitive.name for eqn in equations(jaxpr)}


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
    return _Hooked(call.score_params, output, initial, hooks.drop)


class _Guard(NamedTuple):
    """A guarded launch's arrays (`algebra.walked_launches`): the rows that shift, `live` `[B, H, Qp]` (int32,
    nonzero where live); its keys' value magnitudes, which a step's `algebra.room` reads, come from the value
    tiles it walks (`Family.key_magnitudes`)."""

    live: jax.Array


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

    def split(self):
        """This program's slice of its query tile's keys (`splits`), 0 of one."""
        return lax.rem(lax.axis_index("q_tiles"), self.splits) if self.splits > 1 else jnp.int32(0)

    def padded(self, first) -> jax.Array | bool:
        """Whether the query tile from `first` holds padding rows past the call's queries, some or all of
        them (a call padded to whole programs)."""
        if self.queries is None:
            return False
        return first + self.tiles.block_q > self.queries

    def query_tile(self, count: int):
        """This program's query tile of the grid's `count`: a causal call's longest first (its last query
        tiles see the most keys), so the grid's tail holds its shortest programs, as FlashAttention-3's
        scheduler orders them; a full call's in order."""
        index = lax.axis_index("q_tiles")
        if self.splits > 1:
            index = lax.div(index, self.splits)
        return count - 1 - index if self.causal else index

    def __post_init__(self):
        if not whole_tiles(self.family, self.tiles):
            raise ValueError(f"{self.tiles}: query and key tiles of whole {self.family.rows} rows and "
                             f"{self.family.consumers} consumers at most ({self.family.name})")

    def walked(self, first):
        """The key tiles a query block from `first` reads: those holding the call's keys (none wholly past
        them, `algebra.seen`), up to its diagonal where causal."""
        held = min(-(-self.keys // self.tiles.block_k), self.steps)
        if not self.causal:
            return jnp.int32(held)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), held)

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
        full_lo, full_hi = spans.full_lo[row, block], spans.full_hi[row, block]

        def whole(index):
            inside = (index >= full_lo) & (index < full_hi)
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.query_doc[row, block]
            return inside & (doc != 0) & (doc == spans.key_doc[row, index])

        return *self._sliced(spans.lo[row, block], spans.hi[row, block]), whole

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
            found.append((position, spans.start[row, index], spans.end[row, index], holds))
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

        def seen_whole(index):
            inside = (block >= full_lo[row, index]) & (block < full_hi[row, index])
            if spans.query_doc is None or spans.key_doc is None:
                return inside
            doc = spans.key_doc[row, block]
            return inside & (doc != 0) & (doc == spans.query_doc[row, index])

        assert spans.q_lo is not None and spans.q_hi is not None, "a key block's walk reads its readers"
        return spans.q_lo[row, block], spans.q_hi[row, block], seen_whole

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
    if spans is None:
        return algebra.Coords(q_pos, k_pos, drop=draw, keys_first=keys_first, tile_keys=tile_keys,
                              key_start=key_start, key_offsets=key_offsets)
    keys = pl.ds(k_first, shape[key_axis])
    of_keys = family.rows_of if keys_first else family.columns_of
    bounds, query_ids = held if held is not None else _query_bounds(family, spans, row, first,
                                                                    shape[query_axis], keys_first=keys_first)
    ids = None
    if query_ids is not None and spans.key_ids is not None:
        ids = query_ids, of_keys(spans.key_ids.at[row, keys])
    return algebra.Coords(q_pos, k_pos, bounds, ids, draw, keys_first=keys_first, tile_keys=tile_keys)


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
    """The streams' views by name (`Family.opened`, `Family.stream`, `mosaic.viewed`): each `[B, S, H, D]`
    operand's `[S, D]` at batch row `row` and head `head`, step `j` its `j`-th block of rows."""
    return {name: (ref.at[row, :, head], lambda j: (j, 0)) for name, ref in refs.items()}
