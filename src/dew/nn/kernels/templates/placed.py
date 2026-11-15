"""tcgen05's placed mainloops (row 5368): a walk whose declared products (`mosaic.Product`) a role of their
own issues, a step ahead of the consumers that read their results and give them their left operands, their
accumulators in tensor memory (FlashAttention 4's structure), on the toolkit's tcgen05 rings
(`mosaic_sm100.Tcgen05Rings`): the attention forward's rows (`ROWS`)."""

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import handoffs, mosaic, resources, swizzling
from dew.nn.kernels.templates.mosaic_sm100 import (
    TMEM,
    Accumulator,
    Signal,
    Tcgen05ChunksIssuer,
    Tcgen05Rings,
    column_span,
)


@dataclasses.dataclass(frozen=True)
class _PlacedSlots:
    """A consumer step's place where its family issues its products on a role of their own
    (`Tcgen05Placed.walk`): its ordinal `n` in its walk (its signals' turn), the `turn` of its two tiles of
    logits and weights it reads and writes (a constant: its walk's steps two a trip, a traced turn picks no
    allocation), whether it is its walk's `first` (a constant: its first step peeled from the rest) and
    whether `more` follow. It reads no ring slot (the issuing role does, `Tcgen05Placed.issuing`): a slot's
    release is nothing and its value tile none, whose products the issuer issues."""
    n: Any
    turn: int
    first: bool
    more: Any

    def release(self, name: str) -> None:
        del name

    def __getitem__(self, name: str) -> None:
        del name


class Tcgen05Placed(Tcgen05Rings):
    """tcgen05's rows (row 5368): a walk on tcgen05's 128-row products issued by a warp of their own
    (`issuing`) a step ahead of the consumer warpgroup that reads them, a query row a thread in tensor
    memory's own layout (`TMEM`): FlashAttention 4's structure, a walk's declared products
    (`mosaic.Product`) placed on the issuing role. Each step the issuer issues the next key tile's logits
    into one of two accumulators and, once the consumer gives the step's weights (written narrowed into one
    of two tiles, tcgen05's TS operand), the weighted values onto the sums, which stay in tensor memory
    across the walk (`accumulated`), rescaled there by the consumer a block of columns at a time; each
    hand-off a signal of its own (the logits' `scored`, the weights' `weighed`, the sums' `summed`), waited
    at its turn's phase, no commit of one behind a wait on another's barrier. Its products fill tcgen05's
    128 lanes, where the chunks' 64-row products fill half (`CHUNKS`). Its consumer serves one walk
    (`consumers`), its loads the producer warp's and its products the issuer warp's (`roles`)."""

    name = "tcgen05.rows"
    places_on_role = True
    issues_ahead = True
    rescales_kept = True
    rewalks = False
    consumers = 1

    products_in = Tcgen05ChunksIssuer.products_in
    _scoped = Tcgen05ChunksIssuer._scoped

    @property
    def walked_rows(self) -> int:
        # A swizzle tile's rows of a 16-bit stream: its weights' depth in tensor memory, its logits' columns.
        return 64

    @property
    def layout(self) -> plgpu.Layout:
        return TMEM

    @property
    def row(self):
        return TMEM.reduce(1)

    @property
    def column(self):
        return TMEM.reduce(0)

    @property
    def lane_columns(self) -> int:
        # A thread holds its row's every column: no bit of a column's index is its lane's.
        return 0

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return self.full(shape, 0.0)

    def full(self, shape: tuple[int, ...], value: float, dtype=jnp.float32) -> jax.Array:
        # A row's in tensor memory's own layout (a row a thread), as the walk carries it; a tile's a splat
        # its uses lay out.
        found = jnp.full(shape, value, dtype)
        return plgpu.layout_cast(found, self.native_row) if len(shape) == 1 else found

    def iota(self, shape: tuple[int, ...], dimension: int) -> jax.Array:
        # A tile's indices in tensor memory's own layout, as its logits are read; a vector's by inference.
        if len(shape) == 2:
            return plgpu.broadcasted_iota(jnp.int32, shape, dimension, layout=TMEM)
        return super().iota(shape, dimension)

    def keeps_sums(self, width: int) -> bool:
        # Every walk's, in tensor memory (`accumulated`), within the multiprocessor's where its launch's
        # products' columns are (`product_columns`, `MosaicForward._launch_legal`).
        del width
        return True

    def roles(self, consumers: int, producer: int | None = None) -> handoffs.Schedule:
        # Its consumer warpgroups, then its last warpgroup's warps: the producer's loads and the issuer's
        # products (`mosaic.Family.scheduled`'s `issuer`).
        return handoffs.Schedule((handoffs.Role("consumer", consumers),
                                  handoffs.Role("producer", warp=True, weight=0),
                                  handoffs.Role("issuer", warp=True, weight=0)), floor=producer)

    def walk_own[R](self, streams, views: dict, own: tuple, streamed: tuple, wg,
                    walk: Callable[..., R], *, between=None) -> R:
        # Its consumer reads no ring (its issuer does): no step it passes.
        del views, wg, between
        return walk(streams, *own, streamed[0])

    def product_scratch(self, consumers: int, widths: tuple[int, ...], lefts: tuple = (),
                        ahead: tuple[int, ...] = (), sums: tuple[int, ...] = (),
                        together: tuple[int, ...] = ()) -> dict:
        # A role's two accumulators of the one product it issues ahead (the logits' turns, `LOGITS`), two
        # tiles of the one left operand it is given (the weights' turns, `WEIGHTS`, 16-bit two to a column),
        # its one walk's sums kept across it (`SUMS`), and its hand-offs: the logits' (`scored`, two turns,
        # the issuer's commit), the weights' (`weighed`, two, the consumer's threads, their tensor-memory
        # stores ordered before the products that read them) and the sums' (`summed`, one, the commit). The
        # products it places are those alone (`placed_products`).
        del widths, together
        ((logits,), (left,), (kept,)) = ahead, tuple(dict.fromkeys(lefts)), sums
        rows, (depth, dtype) = self.rows, left
        return {"scored": tuple(self.signals(2).scratch() for _ in range(consumers)),
                "weighed": tuple(self.signals(2).scratch() for _ in range(consumers)),
                "summed": tuple(self.signals(1).scratch() for _ in range(consumers)),
                LOGITS: tuple(tuple(self.tensor_memory((rows, logits)) for _ in range(2))
                              for _ in range(consumers)),
                WEIGHTS: tuple(tuple(self.tensor_memory((rows, depth), dtype) for _ in range(2))
                               for _ in range(consumers)),
                SUMS: tuple(self.tensor_memory((rows, kept)) for _ in range(consumers))}

    def product_columns(self, products: tuple | None) -> int:
        if products is None:
            return 0
        scratch = self.product_scratch(*products)
        held = tuple(scratch[name] for name in (LOGITS, WEIGHTS, SUMS))
        return resources.tensor_memory_columns(jax.tree.leaves(held))

    def product_shared(self, scratch) -> tuple:
        # Its accumulators and tiles are tensor memory; its hand-offs' barriers shared memory.
        return tuple(barrier for name in ("scored", "weighed", "summed") for barrier in scratch[name])

    def product_addresses(self, products: tuple | None) -> int:
        return 0 if products is None else resources.TMEM_ADDRESS_BYTES

    def _signals(self) -> tuple[Signal, Signal, Signal]:
        """The building role's hand-offs (`product_scratch`'s): the logits', the weights', the sums'."""
        found, role = self._scoped(), mosaic.ROLE.get()
        return (self.signals(2).opened(found["scored"][role]), self.signals(2).opened(found["weighed"][role]),
                self.signals(1).opened(found["summed"][role]))

    def _turns(self, name: str) -> tuple[Accumulator, ...]:
        """The building role's two turns of the `name` tiles (`product_scratch`'s `LOGITS` or `WEIGHTS`), each
        its scope's shape and dtype."""
        return tuple(Accumulator(ref, tuple(ref.shape), dtype=ref.dtype)
                     for ref in self._scoped()[name][mosaic.ROLE.get()])

    def scored(self, declared: mosaic.Product, lhs, index, slots, shape: tuple[int, int]) -> jax.Array:
        # The issuer's logits of the step (`issuing`), awaited at its turn and read in tensor memory's own
        # layout from its turn's accumulator.
        del lhs, index, shape
        scored, _, _ = self._signals()
        scored.wait(scored.turn(slots.n))
        return self.read(self._turns(declared.into)[slots.turn], layout=TMEM)

    def accumulating(self, declared: mosaic.Product | None, slots, inline: Callable) -> Callable:
        # The step's weights given the issuer: once the step before's weighted values are done (its weights'
        # turn free, the sums settled), written narrowed into their turn's tile, the sums taken by the step's
        # factor in place a block of columns at a time where its product adds onto them (`declared`'s
        # `accumulate`), both published together (`weighed`); the walk's first step rescales nothing (its sums
        # are the issuer's first product's, fresh).
        del inline
        _, weighed, summed = self._signals()
        adds = declared is not None and declared.accumulate

        def accumulate(acc, scale, parts, value):
            del value
            (weights,) = parts
            if not slots.first:
                summed.wait(summed.turn(slots.n - 1))
            self.write(self._turns(WEIGHTS)[slots.turn], weights)
            if scale is not None and adds and not slots.first:
                _rescaled(self, acc, scale)
            weighed.published(weighed.turn(slots.n))
            return acc

        return accumulate

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool,
                       initial: tuple | None = None) -> tuple[C, tuple]:
        # The walk's sums kept in the role's tensor memory across it (`product_scratch`'s `sums`), the
        # issuer's products onto them, its first fresh (`issuing`): no initial value but the walk's empty
        # start (its launch's legality, `MosaicForward.refusal`); read past the walk a block of columns at a
        # time where stored (`mosaic.Kept`, `stored`).
        del interpret, initial
        found, role = self._scoped(), mosaic.ROLE.get()
        (shape,) = shapes
        sums = (Accumulator(found[SUMS][role], shape),)
        return mosaic.looped(segments, start, sums), tuple(mosaic.Kept(acc) for acc in sums)

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, products: Sequence[mosaic.Product] = ()
                ) -> tuple[C, tuple]:
        # The consumer's walk of its one segment: no ring of its own (the issuer reads them, `issuing`), each
        # step at its ordinal (its signals' turn), its first peeled from the rest (it awaits no sums before it
        # and rescales none), the rest two a trip (its tiles' turns constants), an even count's last alone;
        # each marked (`mosaic.stepped`); then the last step's weighted values awaited.
        del streams, first, in_flight, unrolled, products
        ((step, lo, hi),) = mosaic.stepped(segments)
        count = hi - lo

        def placed(index, carry, accumulators, *, turn: int, opening: bool = False):
            return step(index, _PlacedSlots(index - lo, turn, opening, index + 1 < hi), carry, accumulators)

        def trip(pair, carry, accumulators):
            carry, accumulators = placed(lo + 1 + 2 * pair, carry, accumulators, turn=1)
            return placed(lo + 2 + 2 * pair, carry, accumulators, turn=0)

        def last(_, carry, accumulators):
            return placed(hi - 1, carry, accumulators, turn=1)

        loops = (mosaic.Loop(functools.partial(placed, turn=0, opening=True), lo, jnp.minimum(lo + 1, hi)),
                 mosaic.Loop(trip, 0, jnp.maximum(count - 1, 0) // 2),
                 mosaic.Loop(last, 0, jnp.where((count > 0) & (count % 2 == 0), 1, 0)))
        carry, kept = self.accumulated(loops, start, shapes, interpret, initial)
        _, _, summed = self._signals()
        pl.when(hi > lo)(lambda: summed.wait(summed.turn(hi - lo - 1)))
        return carry, kept

    def issuing(self, streams, products: Sequence[mosaic.Product], lo, hi, *, first, lefts: Mapping,
                ready) -> None:
        # Once the query is in (`ready`), the walk's steps as `issued` runs them, each product's operands'
        # slots given back by its commit (`CommittedRing`): a step's logits fresh into their turn's
        # accumulator, its weighted values onto the sums once the consumer gives the step's weights (the
        # logits' turn is free then: the consumer read it before giving the step before's weights), but at
        # the walk's first step, whose sums are its product's alone.
        scores, weighted = placed_products(products)
        scored, weighed, summed = self._signals()
        keys, values = streams[scores.right], streams[weighted.right]
        lhs = lefts[scores.left]
        logits, weights = self._turns(scores.into), self._turns(WEIGHTS)
        held = self._scoped()[weighted.into][mosaic.ROLE.get()]
        sums = Accumulator(held, tuple(held.shape))

        def score(index, turn: int) -> None:
            ring = keys.turn(index - first)
            right = keys.acquire(ring)
            self.products(logits[turn], lhs, right.T if scores.transposed else right, accumulate=False)
            scored.committed(scored.turn(index - lo))
            keys.release(ring)

        def weigh(index, turn: int) -> None:
            n, ring = index - lo, values.turn(index - first)
            weighed.wait(weighed.turn(n))
            self.products(dataclasses.replace(sums, accumulate=n > 0), weights[turn], values.acquire(ring))
            summed.committed(summed.turn(n))
            values.release(ring)

        plgpu.barrier_wait(ready)
        mosaic.looped(issued(score, weigh, lo, hi), None, ())

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # Sums kept past the walk (`mosaic.Kept`), each in turn, taken by their rows' factor in place in
        # tensor memory's own layout (a row a thread; zero where the factor is, a row that saw no key, whose
        # sums its walk never wrote), then read out a swizzle tile of columns at a time in the epilogue's
        # layout (`STORED`) into `smem`, and stored a stage of columns at a time (the whole where `smem`
        # holds them, CUTLASS's sub-tiled epilogue where it is narrower); every other part as a register
        # family's.
        if not any(isinstance(part, mosaic.Kept) for part in parts):
            return super().stored(parts, views, smem=smem, wait=wait)
        dtype, stage = smem.dtype, smem.shape[1]
        read = min(stage, swizzling.swizzle_columns(dtype, swizzling.swizzle_for(dtype, stage)))
        for index, (kept, view) in enumerate(zip(parts, views, strict=True)):
            acc, factor = kept.acc, kept.factor
            if factor is not None:
                _rescaled(self, acc, lambda block, factor=factor: jnp.where(factor[:, None] != 0,
                                                                            block * factor[:, None], 0.0))
                plgpu.commit_tmem()
            for column in range(0, acc.shape[1], stage):
                if wait or index or column:
                    # The stage's last copy out read before it is written again.
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                for part in range(0, min(stage, acc.shape[1] - column), read):
                    value = self.read(acc, first=column + part, columns=read)
                    smem[:, pl.ds(part, read)] = value.astype(dtype)
                plgpu.commit_smem()
                whole = stage >= acc.shape[1]
                plgpu.copy_smem_to_gmem(smem, view if whole else view.at[:, pl.ds(column, stage)])
        if wait:
            plgpu.wait_smem_to_gmem(0)
        return None


RESCALED_COLUMNS = 32
"""The sums' columns a consumer of tcgen05's rows takes by a factor at once in tensor memory (a row's 32 a
thread, beside its step's logits and weights)."""

LOGITS, WEIGHTS, SUMS = "logits", "weights", "sums"
"""The scratch of tcgen05's rows' products (`Tcgen05Placed.product_scratch`): the logits' two accumulators,
the weights' two tiles, the sums kept across the walk; a declared product's by its accumulator
(`placed_products`)."""


def placed_products(products: Sequence[mosaic.Product]) -> tuple[mosaic.Product, mosaic.Product]:
    """The two products of a walk tcgen05's rows place (`Tcgen05Placed.issuing`), FlashAttention 4's: the
    logits, fresh each step from a tile of the program's and issued a step ahead (`mosaic.Product.ahead`),
    and the weighted values onto the sums across the walk, their left the weights the consumer gives;
    a walk declaring any other products raises."""
    ahead = [product for product in products if product.ahead]
    given = [product for product in products if product.left is None and not product.ahead]
    if not (len(products) == 2 and len(ahead) == len(given) == 1 and ahead[0].left is not None
            and (ahead[0].into, ahead[0].accumulate) == (LOGITS, False)
            and (given[0].into, given[0].accumulate) == (SUMS, True)):
        raise ValueError(f"tcgen05's rows place a walk's logits issued ahead into {LOGITS!r} and its "
                         f"weighted values onto its {SUMS!r} alone, not {products}")
    return ahead[0], given[0]


def _rescaled(family: Tcgen05Placed, acc: Accumulator, scale: Callable[[jax.Array], jax.Array]) -> None:
    """The sums `acc` taken by `scale` in place in tensor memory, a block of `RESCALED_COLUMNS` at a time
    (a narrower sums' whole)."""
    for column in range(0, acc.shape[1], RESCALED_COLUMNS):
        family.update(acc, scale, first=column, columns=min(RESCALED_COLUMNS, acc.shape[1] - column))


def issued(score: Callable, weigh: Callable, lo, hi) -> tuple[mosaic.Loop, ...]:
    """The issuer's walk of steps `lo` to `hi` (`Tcgen05Placed.issuing`) as the loops it runs in turn
    (`mosaic.Loop`), each step's next logits `score(index + 1, turn)` before its weighted values
    `weigh(index, turn)`, each at its turn (a step's place among its two tiles, its parity) a constant: the
    first step's with its own logits too, then two steps a trip, an odd count's step before its last, and
    the last's weighted values alone. Each step's weighted values marked (`mosaic.stepped`): a count of its
    work by emission reads a step a mark, the first step's two logits and the last's none its steady
    step's (`emission.Role.step`, the mean of its loops)."""
    count = hi - lo
    ((weigh, _, _),) = mosaic.stepped(((weigh, lo, hi),))

    def opening(index, carry, accumulators):
        score(index, 0)
        pl.when(index + 1 < hi)(lambda: score(index + 1, 1))
        weigh(index, 0)
        return carry, accumulators

    def trip(pair, carry, accumulators):
        index = lo + 1 + 2 * pair
        score(index + 1, 0)
        weigh(index, 1)
        score(index + 2, 1)
        weigh(index + 1, 0)
        return carry, accumulators

    def odd(_, carry, accumulators):
        score(hi - 1, 0)
        weigh(hi - 2, 1)
        return carry, accumulators

    def last(_, carry, accumulators):
        lax.cond(count % 2 == 0, lambda: weigh(hi - 1, 1), lambda: weigh(hi - 1, 0))
        return carry, accumulators

    return (mosaic.Loop(opening, lo, jnp.minimum(lo + 1, hi)),
            mosaic.Loop(trip, 0, jnp.maximum(count - 2, 0) // 2),
            mosaic.Loop(odd, 0, jnp.where((count > 2) & (count % 2 == 1), 1, 0)),
            mosaic.Loop(last, 0, jnp.where(count > 1, 1, 0)))


ROWS = Tcgen05Placed()
"""tcgen05's rows: the attention mainloop's walk on 128-row products a warp issues (row 5368)."""


COMPUTE = 2
"""The compute warpgroups of tcgen05's keys first (`Tcgen05KeysFirst`), taking a walk's steps in turn
(FlashAttention 3's ping-pong): one warpgroup is a warp on each of a multiprocessor's four schedulers, whose
latencies no other warp hides, and it waits for each step's products (a B200's dK/dV at Qwen3-8B's 2 x 4096
under ncu with one: 30.7% of its issue slots busy, its polls of the products' barriers 8% of the kernel's
stall samples; et4839n)."""

TURNS = 2
"""The turns of a walked placement's fresh products (`Tcgen05Walked`), a step's parity: the step its compute
warpgroup reads and the next, which the issuer issues a step ahead."""

SCORES, COTANGENTS, DK, DV, DQ = "logits", "cotangents", "dk", "dv", "dq"
"""The scratch of tcgen05's walked placements (`Tcgen05Walked.product_scratch`): the scores' and their
cotangents' two accumulators each (keys first `Sᵀ` and `dPᵀ`, queries first `S` and `dP`), a turn a step's
parity, the lefts their kept products read written narrowed over each turn's first half, a fused walk's
step's dQ over the turns once their products are read (a half of its columns each, `DQ`); the sums kept
across the walk (keys first dK and dV, queries first dQ)."""


@dataclasses.dataclass(frozen=True)
class WalkedProducts:
    """The products of an attention backward's walk tcgen05 places (`walked_products`), FlashAttention 4's
    backward's: the scores and their cotangents (`ahead`), fresh each step from the program's resident tiles
    and issued a step ahead; the products onto sums kept across the walk (`kept`), each its left the
    narrowed turn of an ahead product its compute warpgroup gives (tcgen05's TS form: keys first dV `+= Pᵀ
    dO` and dK `+= dSᵀ Q`, queries first dQ `+= dS K`); and a fused keys-first walk's dQ `= dS K`
    (`fused`), fresh each step, its left staged in shared memory, its right the resident keys."""

    ahead: tuple[mosaic.Product, ...]
    kept: tuple[mosaic.Product, ...]
    fused: mosaic.Product | None = None

    @property
    def given(self) -> tuple[str, ...]:
        """Each kept product's left: the ahead product whose narrowed turn holds it."""
        return tuple(product.left or "" for product in self.kept)


@dataclasses.dataclass(frozen=True)
class HandOffs:
    """A walked placement's signals (`Tcgen05Walked.product_scratch`): the scores' (`scored`), the given
    lefts' (`formed`), the walk's end (`finished`), and a fused walk's dQ's and its turns' freeing
    (`dq_ready`, `freed`; None where the walk is not fused, `fused`)."""

    scored: Signal
    formed: Signal
    finished: Signal
    dq_ready: Signal | None = None
    freed: Signal | None = None

    def fused(self) -> tuple[Signal, Signal]:
        """A fused walk's dQ's and its turns' freeing; raises where the walk is not fused."""
        if self.dq_ready is None or self.freed is None:
            raise ValueError("a walk that is not fused hands off no dQ")
        return self.dq_ready, self.freed


def walked_products(products: Sequence[mosaic.Product], kept: tuple[str, ...]) -> WalkedProducts:
    """`products` as a walk tcgen05 places (`WalkedProducts`) whose sums are `kept`; raises for any other set:
    two fresh products issued ahead from tiles of the program's into `SCORES` and `COTANGENTS`, a product
    onto each of `kept`'s sums from an ahead product's narrowed turn, at most one fused `DQ`."""
    ahead = tuple(product for product in products if product.ahead)
    summed = tuple(product for product in products if product.accumulate)
    fused = [product for product in products if not product.ahead and not product.accumulate]
    intos = tuple(product.into for product in ahead)
    if not (intos == (SCORES, COTANGENTS)
            and all(not product.accumulate and product.left is not None for product in ahead)
            and sorted(product.into for product in summed) == sorted(kept)
            and all(product.left in intos for product in summed)
            and len(fused) <= 1
            and all(product.into == DQ and product.left is not None for product in fused)):
        raise ValueError(f"tcgen05's walked placements issue {SCORES!r} and {COTANGENTS!r} ahead from the "
                         f"program's tiles, a product onto each of {kept} from their narrowed turns and at "
                         f"most a fused {DQ!r}, not {products}")
    return WalkedProducts(ahead, summed, fused[0] if fused else None)


def walked_swizzles(products: Sequence[mosaic.Product], dtype, walked: int) -> dict[str, int]:
    """The swizzles of the streams a walk tcgen05 places reads as its kept products' right operands, the
    walked tile their depth (`walked_products`): the widest whose MMA depth group (Mosaic GPU's `K` of `8 x
    swizzle bytes` bits) divides the walked tile (32 bf16 keys or queries a step at 64 bytes, where 128
    took 64; a 64-row step at width 256 passed a B200's block, et-walked-d256)."""
    return dict.fromkeys((product.right for product in products if product.accumulate),
                         swizzling.swizzle_for(dtype, walked))


def ahead_alone(products: Sequence[mosaic.Product]) -> tuple[str, ...]:
    """The streams of a walk tcgen05 places (`walked_products`) that its products issued ahead alone read:
    each step's slot given back by their commit, a step before the next step's are issued (`issued`), so one
    slot holds the stream (keys first the logits' queries, queries first the values: at width 256 a second
    slot of the keys-first walk's logits passed a B200's block, et-walked-d256)."""
    read = {product.right for product in products if not product.ahead}
    return tuple(dict.fromkeys(product.right for product in products
                               if product.ahead and product.right not in read))


class Tcgen05Walked(Tcgen05Placed):
    """tcgen05's walked placements (row 4839): an attention backward launch's walk on tcgen05's 128-row
    products, its resident side a row each (keys first a key, queries first a query), issued by a warp of
    their own (`issuing`), FlashAttention 4's backward's structure. Each step the issuer issues the next
    step's scores and their cotangents (`WalkedProducts.ahead`) into the step's turn (`TURNS`) and, once the
    compute warpgroup whose step it is (`consumers` of them, taking the steps in turn) gives their kept
    products' lefts (written narrowed over the turn's own columns, FlashAttention 4's `P` over `S`), those
    products onto the sums that stay in tensor memory across the walk (`kept_sums`, `accumulated`); each
    stream's slot given back by the commit of the step's last product that reads it. Each hand-off a signal
    of its own (the scores' `scored`, the given lefts' `formed`, the walk's end `finished`)."""

    rescales_kept = False
    kept_sums: tuple[str, ...] = ()
    """The sums the walk keeps in tensor memory across it, in its shapes' order (`accumulated`)."""

    @property
    def walked_rows(self) -> int:
        # Its walked tile its kept products' depth, their right operands staged at the swizzle whose depth
        # group divides it (`walked_swizzles`): 32 16-bit rows at 64 bytes.
        return 32

    def product_scratch(self, consumers: int, widths: tuple[int, ...], lefts: tuple = (),
                        ahead: tuple[int, ...] = (), sums: tuple[int, ...] = (),
                        together: tuple[int, ...] = ()) -> dict:
        # The program's two turns of the scores and of their cotangents (`ahead`'s widths, a step's walked
        # tile), a packed 16-bit view over each (`lefts`' dtype: the narrowed lefts over the turn's first
        # half), the kept sums (`sums`, `kept_sums`'), and the hand-offs: the scores' (`scored`, two turns,
        # the issuer's commit), the given lefts' (`formed`, two, the compute warpgroups' threads, their
        # tensor-memory stores ordered before the products that read them) and the walk's end (`finished`,
        # the issuer's last commit). A fused walk's (`widths`: its dQ's halves' columns) each step's dQ over
        # its turns, a half of its columns over each (a 64-row accumulator of them, half a column's lanes
        # each, `Tcgen05ChunksIssuer`'s), and its hand-offs: the issuer's commit (`dq_ready`) and the
        # compute warpgroup's read of it, its turns free (`freed`).
        del consumers, together
        ((scores, cotangents), (left,)) = ahead, tuple(dict.fromkeys(lefts))
        rows, (_, dtype) = self.rows, left

        def turns(width: int, halves: tuple[int, ...]):
            return tuple(plgpu.RefUnion(self.tensor_memory((rows, width)),
                                        self.tensor_memory((rows, 2 * width), dtype),
                                        *(self.tensor_memory((scores, half)) for half in halves))
                         for _ in range(TURNS))

        fused = {"dq_ready": self.signals(TURNS).scratch(), "freed": self.signals(TURNS).scratch()}
        return {"scored": self.signals(TURNS).scratch(), "formed": self.signals(TURNS).scratch(),
                "finished": self.signals(1).scratch(), **(fused if widths else {}),
                SCORES: turns(scores, widths[:1]), COTANGENTS: turns(cotangents, widths[1:2]),
                **{name: self.tensor_memory((rows, width))
                   for name, width in zip(self.kept_sums, sums, strict=True)}}

    def product_columns(self, products: tuple | None) -> int:
        # Its scratch's tensor memory as it allocates it (`product_scratch`): each turn's union the widest of
        # what it aliases, beside its kept sums.
        if products is None:
            return 0
        scratch = self.product_scratch(*products)
        held = (scratch[SCORES], scratch[COTANGENTS], *(scratch[name] for name in self.kept_sums))
        return resources.tensor_memory_columns(jax.tree.leaves(held))

    def product_shared(self, scratch) -> tuple:
        return tuple(scratch[name] for name in ("scored", "formed", "finished", "dq_ready", "freed")
                     if name in scratch)

    def _hand_offs(self) -> HandOffs:
        """The program's hand-offs (`product_scratch`'s, `HandOffs`)."""
        found = self._scoped()
        fused = {name: self.signals(TURNS).opened(found[name]) for name in ("dq_ready", "freed")
                 if name in found}
        turns = self.signals(TURNS)
        return HandOffs(turns.opened(found["scored"]), turns.opened(found["formed"]),
                        self.signals(1).opened(found["finished"]), **fused)

    def turn_of(self, name: str, turn: int, *, narrowed: bool = False) -> Accumulator:
        """Turn `turn`'s accumulator of `SCORES` or `COTANGENTS` (`product_scratch`'s), or its narrowed left
        over the turn's first half (`narrowed`, two 16-bit values to a column)."""
        wide, packed, *_ = self._scoped()[name][turn]
        columns = int(wide.shape[1])
        if not narrowed:
            return Accumulator(wide, (self.rows, columns))
        return Accumulator(column_span(packed, 0, columns), (self.rows, columns), dtype=packed.dtype)

    def dq_of(self, turn: int) -> tuple[Accumulator, ...]:
        """A fused walk's step's dQ over turn `turn`, its halves of columns in order (`product_scratch`'s,
        `DQ`): over the scores' turn, then the cotangents'."""
        return tuple(Accumulator(half, tuple(half.shape))
                     for name in (SCORES, COTANGENTS) for half in self._scoped()[name][turn][2:])

    def dq_read(self, slots) -> tuple[jax.Array, ...]:
        """A compute warpgroup's step's dQ (`dq_of` its turn's halves), read once the issuer's products of it
        are done (`dq_ready`), in registers in the layout a 64-row accumulator loads into; its turns freed
        for the issuer's next products into them (`freed`) once read."""
        dq_ready, freed = self._hand_offs().fused()
        dq_ready.wait(dq_ready.turn(slots.n))
        halves = tuple(self.read(acc, layout=plgpu.Layout.WGMMA) for acc in self.dq_of(slots.turn))
        plgpu.wait_load_tmem()
        freed.published(freed.turn(slots.n))
        return halves

    def awaited(self, slots) -> tuple[Accumulator, Accumulator]:
        """A compute warpgroup's step's scores and their cotangents (its turn's), once the issuer's products
        of them are done (`scored` at the step's ordinal)."""
        scored = self._hand_offs().scored
        scored.wait(scored.turn(slots.n))
        return self.turn_of(SCORES, slots.turn), self.turn_of(COTANGENTS, slots.turn)

    def given(self, slots) -> None:
        """A compute warpgroup's step's lefts written over its turn (`turn_of(..., narrowed=True)`) given to
        the issuer, their stores visible to its products first."""
        formed = self._hand_offs().formed
        formed.published(formed.turn(slots.n))

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, products: Sequence[mosaic.Product] = ()
                ) -> tuple[C, tuple]:
        # A compute warpgroup's steps of the walk's one segment, those whose ordinal is its own among the
        # `consumers` (its role `mosaic.ROLE`), a trip the steps of `TURNS` in turn that are its (each turn a
        # constant: a traced turn picks no allocation), each at its ordinal (its signals' turn), past the
        # walk's end not taken (`mosaic.taken`), and marked (`mosaic.stepped`); then the issuer's last
        # products awaited (`finished`), whose sums the epilogue reads, zeros where the walk has no step.
        del streams, first, in_flight, unrolled, products
        ((step, lo, hi),) = mosaic.stepped(segments)
        own, count = mosaic.ROLE.get(), hi - lo

        def placed(trip, carry, accumulators):
            for turn in range(own, TURNS, self.consumers):
                n = TURNS * trip + turn

                def stepped(carry, n=n, turn=turn):
                    # No step's first is a constant here (the walk's trips are its compute warpgroups').
                    slots = _PlacedSlots(n, turn, first=False, more=n + 1 < count)
                    return step(lo + n, slots, carry, accumulators)[0]

                if turn == own:
                    carry = stepped(carry)
                else:
                    (carry,) = mosaic.taken(n < count, lambda held, stepped=stepped: (stepped(held),),
                                            carry)
            return carry, accumulators

        loops = (mosaic.Loop(placed, 0, lax.div(jnp.maximum(count - own, 0) + TURNS - 1, TURNS)),)
        carry, kept = self.accumulated(loops, start, shapes, interpret, initial)
        finished = self._hand_offs().finished
        finished.wait(finished.turn(0))
        if own == 0:
            # A block no step reads: its sums no product's, zeros (the first warpgroup's to store).
            @pl.when(hi <= lo)
            def _():
                for sums in kept:
                    for column in range(0, sums.acc.shape[1], RESCALED_COLUMNS):
                        zeros = jnp.zeros((self.rows, min(RESCALED_COLUMNS, sums.acc.shape[1] - column)))
                        self.write(sums.acc, plgpu.layout_cast(zeros, TMEM), first=column)
                plgpu.commit_tmem()

        return carry, kept

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool,
                       initial: tuple | None = None) -> tuple[C, tuple]:
        # The kept sums in the program's tensor memory across the walk (`product_scratch`'s), the issuer's
        # products onto them, its first fresh (`issuing`); read past the walk a block of columns at a time
        # where stored (`mosaic.Kept`, `stored`).
        del interpret, initial
        found = self._scoped()
        sums = tuple(Accumulator(found[name], shape)
                     for name, shape in zip(self.kept_sums, shapes, strict=True))
        return mosaic.looped(segments, start, sums), tuple(mosaic.Kept(acc) for acc in sums)

    def issuing(self, streams, products: Sequence[mosaic.Product], lo, hi, *, first, lefts: Mapping,
                ready, shares: Mapping | None = None) -> None:
        # Once the resident tiles are in (`ready`, their barriers), the walk's steps as `issued` runs them: a
        # step's scores and their cotangents fresh into its turn, then the kept products onto their sums
        # (`WalkedProducts.kept`, in their declared order) once its compute warpgroup gives their lefts, but
        # at the walk's first step, whose sums are its products' alone; each stream's slot given back by
        # the commit of the step's last product that reads it (`CommittedRing`); the walk's end committed
        # once its last are issued. A kept product reads its program's share of its right's columns
        # (`shares[into]`, `(first, count)`; the whole where it has none). A fused walk's dQ over the step's
        # turns once their compute warpgroup read the step before last's.
        placed = walked_products(products, self.kept_sums)
        hand_offs = self._hand_offs()
        scored, formed, finished = hand_offs.scored, hand_offs.formed, hand_offs.finished
        held = self._scoped()
        sums = {name: Accumulator(held[name], tuple(held[name].shape)) for name in self.kept_sums}
        shares = {} if shares is None else shares
        last_reader = {product.right: index for index, product in enumerate((*placed.ahead, *placed.kept))}

        def shared(slot, product: mosaic.Product):
            first_column, count = shares.get(product.into, (0, None))
            return slot if count is None else slot.at[:, pl.ds(first_column, count)]

        def read(position: int, product: mosaic.Product, index):
            # The product's right's slot of the step, given back past the step's last product that reads it.
            ring = streams[product.right]
            turn = ring.turn(index - first)
            return ring.acquire(turn), lambda: (ring.release(turn) if last_reader[product.right] == position
                                                else None)

        def score(index, turn: int) -> None:
            if placed.fused is not None:
                _, freed = hand_offs.fused()
                pl.when(index - lo >= TURNS)(lambda: freed.wait(freed.turn(index - lo - TURNS)))
            released = []
            for position, product in enumerate(placed.ahead):
                slot, release = read(position, product, index)
                self.products(self.turn_of(product.into, turn), lefts[product.left],
                              slot.T if product.transposed else slot, accumulate=False)
                released.append(release)
            scored.committed(scored.turn(index - lo))
            for release in released:
                release()

        def weigh(index, turn: int) -> None:
            n = index - lo
            formed.wait(formed.turn(n))
            for position, (product, left) in enumerate(zip(placed.kept, placed.given, strict=True),
                                                       len(placed.ahead)):
                slot, release = read(position, product, index)
                self.products(dataclasses.replace(sums[product.into], accumulate=n > 0),
                              self.turn_of(left, turn, narrowed=True), shared(slot, product))
                release()
            if placed.fused is not None:
                # dQ `= dS K` over the step's turns once their products are read (tcgen05's products from one
                # thread run in order), its left the score cotangents staged, its right the resident keys a
                # half of their columns over each turn.
                staged, keys = lefts[placed.fused.left].at[turn].T, lefts[placed.fused.right]
                for half, acc in enumerate(self.dq_of(turn)):
                    columns = acc.shape[1]
                    self.products(acc, staged, keys.at[:, pl.ds(half * columns, columns)], accumulate=False)
                dq_ready, _ = hand_offs.fused()
                dq_ready.committed(dq_ready.turn(n))

        for barrier in ready:
            plgpu.barrier_wait(barrier)
        mosaic.looped(issued(score, weigh, lo, hi), None, ())
        finished.committed(finished.turn(0))


def walked_launch(family: mosaic.Family) -> str:
    """The backward launch a walked placement is written for: keys first the dK/dV launch's (`"dkv"`),
    queries first the dQ launch's (`"dq"`); raises for any other family."""
    found = walked(family)
    return "dkv" if isinstance(found, Tcgen05KeysFirst) else "dq"


def walked(family: mosaic.Family) -> "Tcgen05Walked":
    """`family` as a walk tcgen05 places (`Tcgen05Walked`), which a launch that places its products on a
    role is; raises for any other."""
    if not isinstance(family, Tcgen05Walked):
        raise TypeError(f"{family.name} places no backward walk on a role")
    return family


class Tcgen05KeysFirst(Tcgen05Walked):
    """tcgen05's keys first (row 4839): the attention backward's dK/dV walk placed (`Tcgen05Walked`), a key a
    row, its steps a tile of queries each, its two compute warpgroups (`COMPUTE`) taking them in turn, dK and
    dV kept (keys first `Sᵀ = K Qᵀ` and `dPᵀ = V dOᵀ` ahead, dV `+= Pᵀ dO` and dK `+= dSᵀ Q`)."""

    name = "tcgen05.keys"
    consumers = COMPUTE
    kept_sums = (DK, DV)


class Tcgen05QueriesFirst(Tcgen05Walked):
    """tcgen05's queries first (row 4839): the attention backward's dQ walk placed (`Tcgen05Walked`), a query
    a row, its steps a tile of keys each, one compute warpgroup, whose rows' sums its registers carry across
    the walk, dQ kept (`S = Q Kᵀ` and `dP = dO Vᵀ` ahead, dQ `+= dS K`): where a fused keys-first walk's
    resident keys and values and its query streams pass a block's shared memory (width 256)."""

    name = "tcgen05.queries"
    consumers = 1
    kept_sums = (DQ,)


KEYS = Tcgen05KeysFirst()
"""tcgen05's keys first: the attention backward's dK/dV walk on 128-row products a warp issues (row 4839)."""

QUERIES = Tcgen05QueriesFirst()
"""tcgen05's queries first: the attention backward's dQ walk on 128-row products a warp issues (row 4839)."""
