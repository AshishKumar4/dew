"""What the attention template's Mosaic GPU backward launches share (`attention_mosaic_backward`): a launch's
static choices (`_Backward`), and a step's hook parameters and its weights and score cotangents under the
forward's statistics (`_scores`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.extend import core as jax_core

from dew.nn.kernels.templates import (
    attention_algebra as algebra,
    attention_gradient as gradient,
    mosaic,
    swizzling,
)
from dew.nn.kernels.templates.attention import DQ_GRADS, Kind, by_power
from dew.nn.kernels.templates.attention_backward_algebra import (
    derived_cotangents,
    derived_from_weights,
    derived_weights,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardRegisters,
    BackwardTiles,
    _param,
    _Program,
    _streamed,
    _views_of,
    head_slice,
    indexed_views,
    riding,
    score_stream,
    score_views,
)
from dew.nn.precision import scalar_power2


@dataclasses.dataclass(frozen=True)
class _Backward:
    """A backward launch's static choices: its program's geometry, its forward's (`spec`: the call's form, the
    logits' scale and units, its score hook and initial state), its form, its consumers, whether it is
    interpreted, the score hook's parameter kinds (`kinds`), and whether the dK/dV program adds each tile's
    dQ too (`fused`: the fast form's one launch, dQ by atomic adds, D the forward's anchored one
    unrefined)."""

    program: _Program[BackwardTiles]
    spec: algebra.Forward
    accurate: bool
    consumers: int
    interpret: bool
    fused: bool
    kinds: tuple[Kind, ...]
    approx_math: bool
    """Whether its launches take the hardware's approximate transcendentals (`approximate`): never in the
    accurate form."""
    hooked: tuple[bool, bool]
    """Whether its score hook reads its queries' places and its keys' (`score_places`)."""
    dropped: bool
    """Whether the call drops entries (a `Dropout`)."""
    reduces: bool
    """Whether its score hook may reduce across warps (`hook_reduces`), its launches' warpgroups reserving
    their scratch (`reducing`)."""
    rounded: bool = True
    """Whether its caller takes dK and dV rounded to the operands' dtypes (`MosaicBackward.rounded`); where
    not, the fused launch stores them fp32 (`attention_mosaic_fused.stores_typed`)."""
    registered: BackwardRegisters | None = None
    """The call's facts its launches' registers read (`BackwardRegisters`), none where None."""

    def registers(self, *, dq: bool) -> int | None:
        """A consumer's registers in its launch at its program's tiles, the dQ launch's (`dq`) or a dK/dV
        one's, as its derivation counts them (`BackwardRegisters.at`), which the launch states with its
        kernel (`Family.kernel`'s `registers`); None where it states none, or its family budgets none
        (`Family.budgets_registers`)."""
        if self.registered is None or not self.program.family.budgets_registers:
            return None
        return self.registered.at(self.program.tiles, dq=dq)

    def places_read(self, *, masked: bool | None = None) -> tuple[bool, bool]:
        """Whether a step surely reads its queries' places and its keys' (`algebra.places_read`), `masked` or
        not, or either (None: chosen at run time, or every entry's visibility read)."""
        return algebra.places_read(self.spec, masked=masked is not False, hooked=self.hooked,
                                   dropped=self.dropped,
                                   padded=self.spec.keys % self.program.tiles.block_k != 0)

    @property
    def how(self) -> algebra.Recompute:
        """How the backward recomputes the forward's weights."""
        return algebra.recompute(self.spec)

    @property
    def scale(self) -> float:
        """The logits' scale."""
        return self.spec.scale

    @property
    def grad_scale(self) -> float:
        """The scale the gradients' cotangents take: the call's whole scale, the logits' times the power its
        query took before its products (`algebra.Forward.prescale`), against the call's own keys for dQ and
        its own queries for dK, so their products stay within the call's own range."""
        return self.spec.scale * self.spec.prescale

    @property
    def scaled(self) -> bool:
        """Whether the call's queries take a power of two for their logits (`by_power`): the fused launch's
        in registers, the keys-first launch's a stream or a stage of its own (`_dkv_step`), dK the call's
        own queries'."""
        return self.spec.prescale != 1.0

    @property
    def summed(self) -> tuple[bool, ...]:
        """Which score parameters the dQ program sums per query (`DQ_GRADS`); the others' gradients the dK/dV
        program writes."""
        return tuple(kind in DQ_GRADS for kind in self.kinds)

def _relay_tiles(family: mosaic.Family, consumers: int, rows: int, depth: int, dtype) -> list:
    """A backward launch's relay tiles, one a consumer, where its register left operands of `rows` x `depth`
    (its score cotangents and weights, a step's accumulators) cross warps on their way to a product's layout
    (`Family.relay_bytes`: a family arranged by fewer rows than a role's, `bundle_of`); none elsewhere."""
    if not family.relay_bytes(rows, depth):
        return []
    return [family.staged_memory((consumers, rows, depth), dtype, swizzling.UNSWIZZLED)]


def _relayed(walk: _Backward, relay: list, index) -> _Backward:
    """`walk` whose family relays its register left operands through consumer `index`'s tile of `relay`
    (`_relay_tiles`' allocation, `Family.relaying`); itself where the launch allocates none."""
    if not relay:
        return walk
    family = walk.program.family.relaying(relay[0].at[index])
    return dataclasses.replace(walk, program=dataclasses.replace(walk.program, family=family))


def _score_params(family: mosaic.Family, refs, kinds: tuple[Kind, ...], row, head, first, like,
                  k_first=None, *, keys_first: bool = False, held: tuple | None = None, slots=None,
                  streamed: tuple[int, ...] = (), at: tuple = (0, 0)) -> tuple:
    """A tile's score hook parameters (`_param`), one per kind of `kinds`: those its walk holds its `held`
    values (`held_params`), those riding its stream (`streamed`, by index: `riding`) the step's `slots`'
    from `at`, every other read from its ref."""
    held = (None,) * len(kinds) if held is None else held
    laid = zip(refs, kinds, held, strict=True)
    return tuple(_param(family, ref, kind, row, head, first, like, k_first, keys_first=keys_first, held=kept,
                        slot=slots[score_stream(index)] if slots is not None and index in streamed else None,
                        at=at)
                 for index, (ref, kind, kept) in enumerate(laid))

def _read(value):
    """Resolve a deferred reader (`functools.partial`), or return an eager value unchanged."""
    return value() if isinstance(value, functools.partial) else value


def _splits_scores(walk: _Backward) -> bool:
    """Whether a backward tile's scores split into `_score_weights` (inside `product_pair`'s `between`) and
    `_score_cotangents` (rows 11707, 11715): undropped fast-form walks without score-hook parameters."""
    return not walk.accurate and not walk.dropped and not walk.kinds


def _tile_scores(walk: _Backward, slots: tuple, resident: tuple, rows: tuple,
                 coords: algebra.Coords | functools.partial[algebra.Coords], whole,
                 shape: tuple[int, int], *, masked: bool | None, whole_operands: bool = False
                 ) -> tuple[jax.Array, jax.Array]:
    """A queries-first step's tile of a fast-form call whose score hook, if any, takes no parameters
    (`_fused_step`'s, `_coop_step`'s): its `S` and `dP` products (issued through `product_pair`, overlapping
    `_score_weights` with `dP` where `_splits_scores` and `masked is not None`), masked where `masked`, or
    where it is None chosen at run time, unmasked where every query sees the tile `whole`. Returns its
    weights narrowed for dV (`algebra.value_weights`), a dropping call's dropped (`algebra.drawn`), and its
    score cotangents through the hook (`gradient.pulled`), one part in the operands' dtype (dK's and dQ's);
    with `whole_operands` (fp16's, `algebra.running_parts`), both fp32: the normalized weights, dropped, and
    the scaled cotangents."""
    family = walk.program.family
    (q_slot, do_slot), (keys, values) = slots, resident
    held, values_t = keys.transpose((1, 0)), values.transpose((1, 0))
    if not family.operands_in_registers:
        rows = tuple(_read(row) for row in rows)
        coords = _read(coords)
    tile_coords = _read(coords)
    if not walk.scaled and masked is not None and _splits_scores(walk):
        def between(s):
            return _score_weights(walk, tile_coords, (), s, rows[0], rows[1], masked=masked)

        (exponentials, weights, slopes, seen), dp = family.product_pair(
            q_slot, held, do_slot, values_t, shape, between=between)
        cotangents = _score_cotangents(tile_coords, weights, dp, rows[2], rows[3], seen)
        cotangents = gradient.pulled(cotangents, slopes, seen, ())[0]
    else:
        if walk.scaled:
            s = family.registers_product(by_power(family.tile_of(q_slot), walk.spec.prescale), held)
            dp = family.product(do_slot, values_t, shape)
        else:
            s, dp = family.product_pair(q_slot, held, do_slot, values_t, shape)

        def scores(*, is_masked: bool):
            # A tile every query sees whole (`is_masked` False) takes no mask (`_dq_step`'s choice).
            exps, w, cot, sl, sn = _scores(walk, tile_coords, (), s, dp, *rows, masked=is_masked)
            multiplier = algebra.drawn(tile_coords, tile_coords.q_pos, tile_coords.k_pos)
            if multiplier is not None:
                exps, w = exps * multiplier, w * multiplier
            return exps, w, gradient.pulled(cot, sl, sn, ())[0]

        if masked is None:
            exponentials, weights, cotangents = lax.cond(whole, functools.partial(scores, is_masked=False),
                                                         functools.partial(scores, is_masked=True))
        else:
            exponentials, weights, cotangents = scores(is_masked=masked)
    if whole_operands:
        return weights, cotangents * walk.grad_scale
    part = mosaic.unscaled_part(*algebra.cotangent_parts(cotangents * walk.grad_scale, q_slot.dtype,
                                                         accurate=False))
    return algebra.value_weights(exponentials, weights, do_slot.dtype, walk.dropped), part


def _score_logits(walk: _Backward, coords: algebra.Coords, params: tuple, s, *, masked: bool):
    """Shared mask and hooked-logits prologue for `_score_weights` and `_scores`."""
    seen = algebra.seen(walk.spec, coords, s) if masked else None
    how = walk.how
    logits, slopes = gradient.tile_logits(coords.hooked(walk.spec.score_mod), s, seen, coords.q_pos,
                                         coords.k_pos, params,
                                         scale=walk.spec.scale if how.factor is None else 1.0, slopes=True)
    return logits, slopes, seen


def _score_weights(walk: _Backward, coords: algebra.Coords, params: tuple, s, peak, total, *,
                   masked: bool):
    """First half of `_scores` (row 11659): mask, hooked logits + slopes, and derived unnormalized and
    normalized weights from `s`, `peak`, and `total` (runnable while `dP` is in flight)."""
    logits, slopes, seen = _score_logits(walk, coords, params, s, masked=masked)
    exponentials, weights = derived_weights(
        keys_first=bool(coords.keys_first), how=walk.how, accurate=walk.accurate).mapped(
            {"logits": logits, "peak": _read(peak), "total": _read(total)})
    return exponentials, weights, slopes, seen


def _score_cotangents(coords: algebra.Coords, weights, dp, anchor, delta, seen):
    """Second half of `_scores` for undropped fast-form walks (row 11659): derived score cotangents from
    normalized `weights` and `dp`."""
    tiles = {"weights": weights, "products": dp, "anchor": _read(anchor), "delta": _read(delta)}
    if seen is not None:
        tiles["seen"] = seen
    return derived_from_weights(keys_first=bool(coords.keys_first), masked=seen is not None).mapped(tiles)


def _scores(walk: _Backward, coords: algebra.Coords, params: tuple, s, dp, peak, total, anchor, delta,
            chosen, *, masked: bool):
    """A backward tile's weights unnormalized and over `total`, its score cotangents
    (`attention_backward_algebra.softmax_cotangents`) and the score hook's slopes (`gradient.tile_logits`),
    from its products `s` and `dp`, its coordinates, its hook parameters `params`, its mask (`algebra.seen`
    where `masked`) and its queries' statistics, each a vector or a function reading it where it is used (a
    stream's stage, `_dkv_step`)."""
    multiplier = algebra.drawn(coords, coords.q_pos, coords.k_pos)
    if not walk.accurate and multiplier is None:
        exponentials, weights, slopes, seen = _score_weights(walk, coords, params, s, peak, total,
                                                             masked=masked)
        cotangents = _score_cotangents(coords, weights, dp, anchor, delta, seen)
        return exponentials, weights, cotangents, slopes, seen
    logits, slopes, seen = _score_logits(walk, coords, params, s, masked=masked)
    # A dropping anchor's centre can survive where its tile's product rounds to zero before the
    # multiplier joins it. Take its self-difference as 0, as the accurate form does. The undropped
    # fast form keeps FlashAttention 3's math (the select took 10.8% of fused dK/dV's instructions).
    at_anchor = coords.k_pos == coords.queries(_read(chosen), coords.k_pos)
    tiles = {"logits": logits, "products": dp, "peak": _read(peak), "total": _read(total),
             "anchor": _read(anchor), "delta": _read(delta), "at_anchor": at_anchor}
    if multiplier is not None:
        tiles["keep"] = multiplier
    if seen is not None:
        tiles["seen"] = seen
    forms = derived_cotangents(bool(coords.keys_first), multiplier is not None, seen is not None, walk.how,
                               walk.accurate, anchored=True)
    exponentials, weights, cotangents = forms.mapped(tiles)
    return exponentials, weights, cotangents, slopes, seen
def _rescaled_parts(walk: _Backward, acc, shift, rhs, refs: tuple, staged: tuple = ()):
    """An fp16 step's accumulator `acc` (dV's or dK's) taken to its walk's new scale (`shift`,
    `precision.scalar_power2`, a power of two at most 1) before its operand's two fp16 parts' products over
    `rhs` issue into it from their staged `refs`; `staged` the parts each staged into its ref just before its
    product (the fused launch's order, `attention_mosaic_fused._sixteen_step`), none where they are staged
    already (the cooperative launch's exchange, `attention_mosaic_coop._sixteen_step`)."""
    family = walk.program.family
    acc = family.rescaled_products(acc, lambda value: scalar_power2(value, shift), (), rhs,
                                   walk.interpret)
    for index, ref in enumerate(refs):
        if staged:
            family.staged(staged[index], ref)
        acc = family.products(acc, ref.transpose((1, 0)), rhs)
    return acc


def _parts_dq(walk: _Backward, refs: tuple, keys, shape: tuple[int, int], exponent) -> jax.Array:
    """An fp16 step's dQ: its score cotangents' two staged fp16 parts (`refs`) times `keys`, summed and taken
    back by their scale `exponent` (`algebra.running_restored`)."""
    family = walk.program.family
    low, high = refs
    return algebra.running_restored(family.product(low, keys, shape) + family.product(high, keys, shape),
                                    exponent)


def _dq_added(walk: _Backward, dq_ref, out_smem, place: tuple, index, q_first, dq) -> None:
    """A fused dK/dV consumer's tile `dq` added into dQ at its queries from `q_first` (`place` its row and
    head), through its output stage, free until its dK and dV are stored (None where the family's adds read
    no stage, `Transport.adds_through_shared`); interpreted, into a copy of the consumer's own
    (`Family.added`). Where the stage holds several tiles'
    rows (its keys' rows twice a query tile's), a query tile's add takes the next slot of them in turn, so
    it waits for none of the earlier adds' reads of their own (`added`'s `pending`): FlashAttention 3's
    dQ reduce runs off the consumers' path."""
    row, head = place
    rows = dq.shape[0]
    view = dq_ref.at[index if walk.interpret else 0,
                     *head_slice(walk.program.family, row, head, pl.ds(q_first, rows))]
    if out_smem is None:
        walk.program.family.added(dq, view, walk.interpret, smem=None)
        return
    slots = out_smem.shape[1] // rows
    slot = lax.rem(lax.div(q_first, rows), slots) if slots > 1 else 0
    walk.program.family.added(dq, view, walk.interpret, smem=out_smem.at[index, pl.ds(slot * rows, rows)],
                              pending=slots - 1)


def _query_program(program: _Program, streams, k_ref, v_ref, spans, *, score: tuple = ((), ()), heads: int,
                   kv_heads: int, queries: int, consumers: int) -> tuple:
    """A program over query tiles' place and key streams (the dQ launch's and dropout's first pass's): its
    batch row and query head, its key and value views (`_views_of`) and those of its score hook's parameters
    that ride them (`score`, their refs and kinds: `score_views`), their streams opened, its consumers'
    first query (`_Program.query_tile`, a tile each) and the key tiles any of them reads (`_Program.reach`,
    `_streamed`)."""
    block_q, family = program.tiles.block_q, program.family
    row, head = family.program_index("batch"), family.program_index("heads")
    pair = consumers * block_q
    refs, kinds = score
    views = _views_of(family, {"keys": k_ref, "values": v_ref}, row, lax.div(head, heads // kv_heads))
    if riding(family, kinds, "keys"):
        views |= score_views(family, refs, kinds, "keys", row, head, program.query_tile(queries // pair))
    if spans is not None and spans.walk_count is not None:
        # Each step's key tiles, read through its pair's ranges where its walk is indexed
        # (`_Program.tile_of`).
        views = indexed_views(views, program.tile_of(program.query_tile(queries // pair) * pair, spans, row))
    streams = family.opened(streams, views)
    pair_first = program.query_tile(queries // pair) * pair
    streamed = _streamed([program.reach(pair_first + index * block_q, spans, row)[:2]
                          for index in range(consumers)])
    return row, head, views, streams, pair_first, streamed


def _query_tile_loaded(family: mosaic.Family, refs: tuple, smem: tuple, loaded: tuple, place: tuple, first,
                       index) -> None:
    """Consumer `index`'s query tile from `first` and its output cotangents, each `(ref, smem)` of `refs`
    and `smem` loaded whole against its `loaded` barrier at the program's batch row and head (`place`)."""
    row, head = place
    rows = head_slice(family, row, head, pl.ds(first, smem[0].shape[1]))
    family.loaded(tuple((ref.at[rows], stage.at[index], barrier.at[index])
                        for ref, stage, barrier in zip(refs, smem, loaded, strict=True)))


@functools.cache
def _traced_step_tile_count(*, backward: bool, natural_form: bool, scaled: bool,
                            accurate: bool = False, dropped: bool = False) -> int:
    """Peak live 2D score-tile arrays in the traced attention algebra step (`jax.make_jaxpr`, row 10965)."""
    tile = (8, 8)
    s_in, row = jax.ShapeDtypeStruct(tile, jnp.float32), jax.ShapeDtypeStruct((tile[0],), jnp.float32)
    q_pos = jax.ShapeDtypeStruct((tile[0], 1), jnp.int32)
    k_pos = jax.ShapeDtypeStruct((1, tile[1]), jnp.int32)
    spec = algebra.Forward("causal", 64, ids=False, scale=1.0 / 8.0, natural=natural_form, headroom=scaled)
    if backward:
        anchor_tile = jax.ShapeDtypeStruct(tile, jnp.bool_)
        def backward_step(s, dp, peak, total, anchor, delta, q, k, at_anchor):
            seen = algebra.seen(spec, algebra.Coords(q, k), s)
            anchored = accurate or dropped
            forms = derived_cotangents(keys_first=False, dropped=dropped, masked=True,
                                       accurate=accurate, anchored=anchored)
            tiles = {"logits": s, "products": dp, "peak": peak, "total": total, "anchor": anchor,
                     "delta": delta, "seen": seen}
            if dropped:
                tiles["keep"] = s
            if anchored:
                tiles["at_anchor"] = at_anchor
            exps, weights, ds = forms.mapped(tiles)
            return algebra.value_weights(exps, weights, jnp.bfloat16, dropped), ds.astype(jnp.bfloat16)
        jaxpr = jax.make_jaxpr(backward_step)(
            s_in, s_in, row, row, row, row, q_pos, k_pos, anchor_tile).jaxpr
    else:
        def forward_step(s, m, l_val, q, k):
            mod = algebra.modify(spec, s, algebra.Coords(q, k), (), masked=True)
            new_m, alpha = algebra.peak(spec, m, mod)
            w = algebra.weights(spec, mod, new_m)
            if scaled:
                exp = jnp.ones((tile[0],), jnp.int32)
                held = algebra.Scaled(exp, exp, algebra.raised_weights(spec, mod, lambda: new_m))
                return (*algebra.at_scale(w, held), new_m, alpha)
            return w.astype(jnp.bfloat16), algebra.total(l_val, alpha, w), new_m
        jaxpr = jax.make_jaxpr(forward_step)(s_in, row, row, q_pos, k_pos).jaxpr

    def is_tile(v):
        return not isinstance(v, jax_core.Literal) and getattr(v.aval, "shape", None) == tile

    last_use = {v: pos for pos, eqn in enumerate(jaxpr.eqns) for v in eqn.invars if is_tile(v)}
    last_use.update({v: len(jaxpr.eqns) for v in jaxpr.outvars if is_tile(v)})
    live = {v for v in jaxpr.invars if is_tile(v)}
    peak = len(live)
    for pos, eqn in enumerate(jaxpr.eqns):
        live.update(v for v in eqn.outvars if is_tile(v))
        peak = max(peak, len(live))
        live = {v for v in live if last_use.get(v, -1) > pos}
    return peak


def step_live_tiles(block_q: int, block_k: int, *, backward: bool = False, natural_form: bool = False,
                    scaled: bool = False, accurate: bool = False, dropped: bool = False) -> tuple:
    """Live `(block_q, block_k)` fp32 score tiles in TPU VMEM derived from the traced step (row 10965)."""
    count = _traced_step_tile_count(backward=backward, natural_form=natural_form, scaled=scaled,
                                    accurate=accurate, dropped=dropped)
    return (((block_q, block_k), jnp.float32),) * count
