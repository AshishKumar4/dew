"""The recurrence template's Mosaic GPU mainloop as a template implementation (`templates.base.Template`), the
delta rule's selector's 'mosaic_gpu' kernel (`recurrence_chunk.GATED_DELTA`).

Its config is the forward's tiles (`mainloop.Tiles`), derived from the call's shape and the device's and
its bundle's facts (`Mosaic.derived`): the widest value block whose carried memory and step tiles a thread's
registers hold (`resources.live_registers`) and whose states launch fills the device's multiprocessors,
with as many chunks in flight as every launch's shared memory
allows a block of the device (`staged_bytes`: each launch's streams and staging tiles as its bundle declares
them, `planned`, the fast backward's with the forward's where a call takes it). Nothing is
measured, tabled, swept or ranked: a GPU no row names derives its tiles from its own facts, and a measured
gap is fixed in the derivation or the kernel.
"""

import collections
import contextlib
import dataclasses
import functools
import math
import threading
from collections.abc import Hashable, Iterable, Iterator, Sequence, Set
from typing import ClassVar

import jax
import jax.numpy as jnp
import tokamax
from jax.ad_checkpoint import checkpoint_name

from dew import _cache
from dew.nn import derivatives
from dew.nn.kernels import chunked_launch, generation
from dew.nn.kernels.templates import base, bundles, cost, mosaic, report, resources, sass
from dew.nn.kernels.templates.recurrence import chunk_forward, mainloop, mainloop_backward
from dew.nn.residuals import Residual


def _widths(ba: tokamax.BoundArguments, family: mosaic.Family, *, training: bool = True
            ) -> tuple[int, int, jnp.dtype]:
    """The call's keys' and values' widths padded as its launches pad them (`mainloop.padded_widths`), a
    training call's values as its fast backward reads them, and the dtype its tiles hold (the operands',
    fp32 in an fp32 call)."""
    arguments = ba.arguments
    dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
    widths = mainloop.padded_widths(family, arguments["key"].shape[-1], arguments["value"].shape[-1],
                                    kept=training, per_key=arguments["g"].ndim == 4)
    return (*widths, jnp.dtype(dtype))


def chunk_of(ba: tokamax.BoundArguments, family: mosaic.Family, target: generation.GpuTarget) -> int:
    """The tokens a chunk (`Mosaic.derived`'s first rule), the work-span model's optimum solved for it: a
    token's triangular work grows with its chunk `C`, `C (2 (3 Dk + 2 Dv) + 4 R (log2 R - 1))` tensor FLOPs a
    head (its pairs against the keys, its system, the inverse applied to its keys and values, its output, and
    its share of the doubling inverse at a role's rows `R`), where its memory's traffic between launches
    shrinks with it, `2 Dk Dv` held words a chunk (the states launch's store, the output launch's read): their
    sum is least at `sqrt((traffic / memory rate) / (work / tensor rate))`, the device's rates
    (`cost.rates`), rounded to a power of two times the bundle's role rows (`Family.role_rows`), no fewer; the
    role's rows where the device states neither rate."""
    rows = family.role_rows
    key_width, value_width, dtype = _widths(ba, family)
    tensor, memory = cost.rates(target, device=True).tensor, target.memory_rate
    if not tensor or not memory:
        return rows
    work = 2 * (3 * key_width + 2 * value_width) + 4 * rows * (math.log2(rows) - 1)
    traffic = 2 * key_width * value_width * jnp.dtype(dtype).itemsize
    best = math.sqrt((traffic / memory) / (work / tensor))
    return rows * 2 ** max(0, round(math.log2(best / rows)))


def _dtype(ba: tokamax.BoundArguments) -> jnp.dtype:
    """The call's operands' common dtype."""
    return jnp.dtype(jnp.result_type(*(ba.arguments[name].dtype for name in ("query", "key", "value"))))


def bundle(target: generation.GpuTarget, dtype=jnp.bfloat16) -> mosaic.Family | None:
    """The mainloop's bundle for `target`'s instruction family and a call of `dtype` (`bundles.of`), None
    where none claims it."""
    return bundles.of(target.generation, mainloop.bundle_dtype(dtype), scoped=True)



def planned(q, k, v, g, beta, h0=None, *, tiles: mainloop.Tiles, family: mosaic.Family, starts=None,
            max_segments: int | None = None, delta: bool = True, backward: bool = False,
            training_forward: bool = False,
            target: generation.GpuTarget | None = None,
            rounded: bool = False, apart: int = 0, values_gated: bool = False, unit_pairs: bool = False,
            decay_bound: float | None = None,
                wide: bool = False, ranged: bool = False,
            skip=None, forked: chunk_forward.ForkedTable | None = None) -> tuple[mainloop.Launch, ...]:
    """The launches a call over operands of these shapes (`jax.ShapeDtypeStruct`s, `chunk_forward.forward`'s)
    runs at `tiles` by `family`'s bundle on `target` (the device's, or the assumed one's, where None), in
    order, as planned (`mainloop.Launch`): its arrays abstract (`jax.eval_shape`), its kernels declared and
    not built (`mainloop.launch_kernel`): a forward without a gradient's (`chunk_forward.forward`'s); with
    `backward`, the forward a fast backward takes (`chunk_forward.kept_forward`'s) and that backward's
    (`mainloop_backward.gradients`); `training_forward`, that forward alone, a training call's. A `wide`
    call's output lands in fp32 (`chunk_forward.forward`'s): planned in the operands' dtype, the SSD's output
    launch asked 16 KB past its plan (204800 bytes against 188544 on an H100's compile, the resource gate's).
    A `skip` (`[HV]`, its shape) fused into the output and reverse walk is planned with them: its reverse walk
    holds a tile more. A forked call's table (`forked`, `chunk_forward.ForkedTable`) is planned with its
    walks: its landing tiles theirs."""
    found: list[mainloop.Launch] = []

    def run(q, k, v, g, beta, h0, starts, skip, forked):
        token = mainloop.PLANNED.set(found)
        try:
            out, final, (inverse, kept) = chunk_forward.kept_forward(
                q, k, v, g, beta, h0, tiles=tiles, starts=starts, max_segments=max_segments, delta=delta,
                family=family, rounded=rounded, apart=apart, values_gated=values_gated, unit_pairs=unit_pairs,
                decay_bound=decay_bound,
                    wide=wide, ranged=ranged, skip=skip,
                keeps=backward or training_forward, forked=forked)
            if backward:
                mainloop_backward.gradients(q, k, v, g, beta, h0, out, final, tiles=tiles, starts=starts,
                                            max_segments=max_segments, family=family, inverse=inverse,
                                            kept=kept, delta=delta, rounded=rounded, apart=apart,
                                            values_gated=values_gated, unit_pairs=unit_pairs,
                                            decay_bound=decay_bound,
                                            ranged=ranged, skip=skip, forked=forked)
        finally:
            mainloop.PLANNED.reset(token)

    with contextlib.nullcontext() if target is None else generation.assumed(target):
        jax.eval_shape(run, q, k, v, g, beta, h0, starts, skip, forked)
    return tuple(found)


def _launches_held(launches: tuple[mainloop.Launch, ...], family: mosaic.Family) -> bool:
    """Whether each of `launches`' consumer threads hold their registers (`Launch.registers` within
    `Family.registers` at its consumer roles beside a producer at setmaxnreg's least, of the bundle it issues
    by, `Launch.family`, `family` where it names none) and a multiprocessor holds a program of it
    (`Launch.resident`). The model proposes at the least; a producer raised past it is the assembler's
    (`Mosaic._raised`): a compile that spills the raised launch, consumers or producer, refuses it."""
    return all(launch.resident > 0
               and launch.registers <= (launch.family or family).registers(launch.consumers)
               for launch in launches)


def holds(launches: tuple[mainloop.Launch, ...], family: mosaic.Family, shared: int) -> bool:
    """Whether each of `launches` fits a block of `shared` bytes, its consumer threads hold their registers
    (`Launch.registers` within `Family.registers` at its consumer roles, of the bundle it issues by,
    `Launch.family`, `family` where it names none) and a multiprocessor holds a program of it
    (`Launch.resident`): a generated launch spills nothing."""
    return all(launch.shared <= shared for launch in launches) and _launches_held(launches, family)


def backward_holds(q, k, v, g, beta, h0=None, *, tiles: mainloop.Tiles, family: mosaic.Family,
                   target: generation.GpuTarget, starts=None, max_segments: int | None = None,
                   delta: bool = True, rounded: bool = False, apart: int = 0, values_gated: bool = False,
                   unit_pairs: bool = False, decay_bound: float | None = None, wide: bool = False,
                   ranged: bool = False, skip=None,
                   forked: chunk_forward.ForkedTable | None = None) -> bool:
    """Whether the launches a fast backward runs (`mainloop_backward.gradients`'s and the forward it takes,
    `chunk_forward.kept_forward`'s, its output fp32 where `wide`) hold on `family`'s bundle at `tiles` on
    `target`, planned there and within
    its block's shared memory (`holds`); where they do not, the rule's gradient is the portable math's.
    Planned on the device's own target off a GPU, an interpreted call's states launch held no program at the
    interpreted target's shared budget (`Launch.resident`). Each plan is the call's cached one (`_planned`):
    planned afresh at each of a Qwen3.5-4B-width step's 168 traces of its GDN layers, it took 11.37 s of its
    182.9 s compile_ahead on sm90, its 425 plans 13.34 s (cProfile, armada job 20261105140541-69454bb5 at
    base/kernels 384436fbe8, row 4260)."""
    *shapes, skip = (None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype)
                     for x in (q, k, v, g, beta, h0, starts, skip))
    both = _planned(tuple(shapes), tiles, family, target, max_segments, delta, backward=True,
                    rounded=rounded, apart=apart, values_gated=values_gated, unit_pairs=unit_pairs,
                    decay_bound=decay_bound, wide=wide,
                    ranged=ranged, skip=skip, forked=_shaped(forked))
    return holds(both, family, generation.shared_budget(target))

PLANS_KEPT = 1024
"""The plans `_planned` keeps, the least recently read given way first: a Qwen3.5-4B-width step's
compile_ahead planned 425 (row 4260's profile), and every plan kept for the process's life held each
geometry a derivation ever tried (row 4390)."""

_PLANS: collections.OrderedDict = collections.OrderedDict()
"""`_planned`'s plans by their call, tiles, bundle and target's facts, the most recently read last: each the
call's launches by the depth they were planned at, its trace's at one stage."""
_PLANNING = threading.Lock()

_SPILLED: collections.OrderedDict = collections.OrderedDict()
"""`Mosaic._spilled`'s launches by what a derivation reads (`report.derivation_inputs`), the most recently
made last."""


def _planned(shapes: tuple, tiles: mainloop.Tiles, family: mosaic.Family, target: generation.GpuTarget,
             max_segments: int | None, delta: bool, backward: bool,
             rounded: bool = False, wide: bool = False,
             ranged: bool = False, skip: jax.ShapeDtypeStruct | None = None,
             forked: chunk_forward.ForkedTable | None = None, training_forward: bool = False,
             apart: int = 0, values_gated: bool = False, unit_pairs: bool = False,
             decay_bound: float | None = None) -> tuple[mainloop.Launch,
                                                                                           ...]:
    """`planned` of a call over `shapes` (`jax.ShapeDtypeStruct`s or None) by `family`'s
    bundle on `target`, cached by the facts a plan reads of it (two targets of one family share no plan: an
    sm87's and an sm80's multiprocessors and shared memory differ), `PLANS_KEPT` of them. A ring's depth
    (`Tiles.stages`) moves only the launches whose ring is the tiles' (`mainloop.Launch.ring`): the call is
    traced once at one stage and those launches planned at the tiles' depth (`Launch.deepened`). Traced at
    each depth, the derivation took 2.0 s of a CPU's time for Kimi Linear's delta attention on sm100, 30
    traces of 66 ms (jsf-dtime at 0cbc7f8f1d9); each depth's launches are kept beside the trace's, as a
    derivation asks one tiles' plan again for every rule it reads (replanned at each ask, 1.6 million rings
    deepened over a refusal walk's 296 derivations, 461 of its 842 profiled seconds, row 6551)."""
    facts = (target.generation, target.device_kind, target.shared_limit, target.multiprocessors)
    # A backward's plan holds the training call's forward: one key for it.
    training_forward = training_forward and not backward
    traced = dataclasses.replace(tiles, stages=1)
    key = (shapes, traced, family, facts, max_segments, delta, backward, training_forward, rounded, apart,
           values_gated, unit_pairs, decay_bound,
           wide, ranged, skip, forked)
    with _PLANNING:
        depths = _PLANS.get(key)
        if depths is not None:
            _PLANS.move_to_end(key)
            if (found := depths.get(tiles.stages)) is not None:
                return found
    if depths is None:
        *operands, starts = shapes
        depths = {1: planned(*operands, tiles=traced, family=family, starts=starts, max_segments=max_segments,
                             delta=delta, backward=backward, training_forward=training_forward, target=target,
                             rounded=rounded, apart=apart, values_gated=values_gated, unit_pairs=unit_pairs,
                             decay_bound=decay_bound,
                                 wide=wide, ranged=ranged, skip=skip,
                             forked=forked)}
    found = tuple(launch.deepened(tiles.stages, target) for launch in depths[1])
    with _PLANNING:
        _PLANS[key] = {**_PLANS.get(key, depths), tiles.stages: found}
        while len(_PLANS) > PLANS_KEPT:
            _PLANS.popitem(last=False)
    return found


def _shaped(forked: chunk_forward.ForkedTable | None) -> chunk_forward.ForkedTable | None:
    """A forked call's table (`chunk_forward.ForkedTable`) as its arrays' shapes, a plan's key."""
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), forked)

def ranged(dtype, delta: bool, bound: float | None) -> bool:
    """Whether a call's solve holds its chunk inverse at fp32's range (`mainloop.Call.ranged`): an fp16
    delta rule whose form proves no bound on the inverse's entries (bf16 already merges at fp32's range,
    `mainloop.merge_parts`, row 9522)."""
    return delta and jnp.dtype(dtype) == jnp.float16 and bound is None


def mosaic_rule(query, key, value, g, beta, state=None, *, tiles, starts=None,
                max_segments: int | None = None, scale: float | None = None, delta: bool = True,
                wide: bool = False, family=None, bound: float | None = None,
                firsts=None, rounded: bool = False, apart: int = 0, values_gated: bool = False,
                unit_pairs: bool = False,
                decay_bound: float | None = None, skip=None, recomputed: bool = False,
                forked: chunked_launch.Forked | None = None):
    """The chunked rule, a decay a head (`g` `[B, S, Hv]`) or a key dimension (`[B, S, Hv, Dk]`), with or
    without the `delta` rule, on the recurrence template's Mosaic GPU mainloop (`chunk_forward.forward`): the
    port's operands and results (`recurrence_chunk.fla_rule`'s), the operands in their common dtype and beta
    with them, the decay and the memory fp32, on one device, by the bundle of its instruction family
    (`mainloop.family_of`, `bundles.of`), interpreted off a GPU (Mosaic GPU's interpreter, the interpreted
    target's bundle, `mainloop.issued`). Its gradient: the fast backward's on the mainloop
    (`mainloop_backward.gradients`, fla's DAG) where it takes the call's form
    (`mainloop_backward.fast_backward_form`) and its launches hold on the device (`backward_holds`); it takes
    no other call's, which raises NotImplementedError as the gradient is traced:
    the delta rule's selector gives a training call it refuses another kernel (`Mosaic.trains`), and a call
    that names the mainloop insists on it (ledger 1926: no kernel falls back inside itself). Its states and
    output launches take `tiles` (`mainloop.Tiles`, the template's config); a `family` given is the bundle it
    issues by, interpreted off a GPU of that family (a CPU's). A 16-bit delta rule takes the chunk inverse's
    `bound` its form proves, its solve at fp32's range where it proves none
    (`ranged`); a `rounded` call takes the run's fast forms' numbers (`recurrence_chunk.fast_rounding`,
    `chunk_forward.forward`). Where `recomputed`, its fast backward forms the forward's memories again rather
    than keeping them (a remat rung recomputing 'delta_memory', `residuals.RECOMPUTED`, resolved at its
    route). A `forked` call's documents (`recurrence_chunk.Chunk.loads`), `state` its table's given memories
    `[B, S', HV, Dk, Dv]`, are laid as a packed call's are (`forked_table`), each row's walk taking each
    document's memory up from the table at its first chunk (`chunk_forward.ForkedTable`): a forked row costs
    its packed row at its documents' bound; the entry write is differentiated by JAX, the table's memories by
    the walks."""

    dtype = jnp.result_type(query, key, value)
    device, interpret = mainloop.issued(dtype)
    issuing = family or device
    query, key, value = (x.astype(dtype) for x in (query, key, value))
    # The delta rule's write gate in the operands' dtype, as fla's layer holds it; without the delta rule the
    # call's own, as the portable math takes it (Mamba-2's fp32 `dt`).
    beta = beta.astype(dtype) if delta else beta
    g = g.astype(jnp.float32)
    # Interpreted off a GPU (an assumed one's calls lower for it); a GPU no bundle claims runs no launch.
    if family is None and device is None and not interpret:
        raise NotImplementedError(f"the recurrence template has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    run = {"family": issuing, "interpret": interpret, "tiles": tiles,
           "ranged": ranged(dtype, delta, bound)}
    table = None
    if forked is not None:
        batch, length = query.shape[:2]
        table, starts, firsts, max_segments = forked_table(state, forked, batch, length, tiles.chunk)
        state = None
    # The call's forms every launch it plans or runs reads, beside its geometry (a forked call's documents'
    # bound its table's).
    forms = {"max_segments": max_segments, "delta": delta, "wide": wide, "rounded": rounded, "apart": apart,
             "values_gated": values_gated, "unit_pairs": unit_pairs, "decay_bound": decay_bound}
    # The fast backward where its launches hold on the device (a derived launch spills nothing), else none.
    fast = (mainloop_backward.fast_backward_form(delta, rounded, dtype, per_key=g.ndim == 4, wide=wide)
            and backward_holds(query, key, value, g, beta, state, tiles=tiles, family=issuing,
                               target=Mosaic.launched(), starts=starts, ranged=run["ranged"], skip=skip,
                               forked=table, **forms))

    # The launch tracing the forward (`report.launch_digest`), whose name the fast backward's kernels carry
    # too (`report.relaunched`): JAX traces the backward after that launch's scope ended.
    launch = report.launch_digest()

    # The documents' starts and firsts operands, not closed over (a custom VJP's function closes over no
    # tracer).
    def primal(query, key, value, g, beta, state, starts, firsts, skip, table):
        # A forward without a gradient at its own tiles, no fast backward narrowing them (`Tiles.forward`).
        alone = {**run, "tiles": tiles.forward or tiles}
        return chunk_forward.forward(query, key, value, g, beta, state, starts=starts, firsts=firsts,
                                     scale=scale, skip=skip, forked=table, **forms, **alone)

    def forward(*operands):
        if not fast:
            return primal(*operands), (operands, (None, None))
        # The chunks' inverses kept for the fast backward, as fla's keeps its `A`, and the launches' memories,
        # weighted keys and corrected values, as the port keeps its memory, but where the block's remat rung
        # recomputes them (`recomputed`). Each is named as the port names its own, so a rung that saves names
        # keeps them rather than replaying the forward's launches for them (`residuals.DELTA_RESIDUALS`).
        out, final, (inverse, kept) = chunk_forward.kept_forward(
            *operands[:6], starts=operands[6], firsts=operands[7], scale=scale, skip=operands[8],
            forked=operands[9], **forms, **run)
        kept = None if recomputed else checkpoint_name(kept, Residual.DELTA_MEMORY)
        return (out, final), (operands, (checkpoint_name(inverse, Residual.DELTA_INVERSE), kept))

    def backward(saved, grads):
        operands, (inverse, kept) = saved
        out_grad, final_grad = grads
        query, key, value, g, beta, state, starts, firsts, skip, table = operands
        if not fast:
            form = (f"{dtype.name}, {'the delta rule' if delta else 'no delta rule'}, a decay a "
                    f"{'head' if g.ndim == 3 else 'key dimension'}")
            raise NotImplementedError(
                f"the recurrence template's fast backward takes no gradient of this call ({form}, or its "
                "launches past the device's at these tiles): the delta rule's selector gives a training call "
                "it refuses the port or the portable math (`MOSAIC_SUPPORTS`' 'backward')")
        with report.relaunched(launch):
            *grads, dh0, dskip, dentries = mainloop_backward.gradients(
                query, key, value, g, beta, state, out_grad, final_grad, starts=starts, firsts=firsts,
                scale=scale, inverse=inverse, kept=kept, skip=skip, forked=table,
                **{name: value for name, value in forms.items() if name != "wide"}, **run)
        # A forked call's given memories' gradient its table's first, its slots' none.
        tabled_grads = None if table is None else dataclasses.replace(table, given=dh0, loads=None, rows=None,
                                                                       stores=None, entries=dentries)
        # The skip's gradient formed in its dtype before the inputs' cotangents leave (row 6027).
        (grads, dh0, tabled_grads), dskip = derivatives.with_gradients_formed(
            (grads, dh0, tabled_grads), None if dskip is None else dskip.astype(skip.dtype))
        return (*grads, None if state is None else dh0, None, None, dskip, tabled_grads)

    rule = derivatives.contracted("test_the_template_rule_holds_its_orders")(
        derivatives.declared_derivative(derivatives.KernelGlue(), primal, forward, backward))
    return rule(query, key, value, g, beta, state, starts, firsts, skip, table)


def forked_table(given, forked: chunked_launch.Forked, batch: int, length: int, chunk: int) -> tuple:
    """A forked call (`mosaic_rule`'s) as a packed call of its documents: its table as the walks take it,
    `(given, loads, stores, entries, turning)` on its documents' chunks with its memories fp32
    (`chunk_forward.ForkedTable`, `chunked_launch.ForkedChunks`), its documents' first tokens, their firsts
    and its documents' bound, the packed call's `starts`, `firsts` and `max_segments`."""
    chunks = chunked_launch.ForkedChunks.of(forked, batch, length, chunk)
    entries = chunks.entries(forked)
    table = chunk_forward.ForkedTable(given.astype(jnp.float32), chunks.loads, chunks.stores,
                                      None if entries is None else entries.astype(jnp.float32), chunks.rows,
                                      forked.held, adds=chunks.added(forked),
                                      turning=chunks.turning(forked))
    return table, forked.tokens(length)[0] >= 0, forked.firsts, forked.loads.shape[1]


def walk_roles(family: mosaic.Family) -> int:
    """The consumer roles a walk program of `family` takes (`Tiles.walkers`): its consumers beside a producer
    (`Family.warpgroups`), each its own value blocks, else one, where a family without one runs its
    consumers over one pipeline together, whose scoped allocations the walks do not make collective
    (mma.sync's two consumers, d1a42595a22, failed to lower the walks on an A100: test_lowering and the
    compile gate red at base 14e38ca857); at most two, a third (wgmma's attention forward's) unmeasured on a
    walk."""
    return min(family.consumers, 2) if family.warpgroups(1) > 1 else 1


def _spread(held: int, programs: int, target: generation.GpuTarget) -> int:
    """The programs of a launch holding `held` a multiprocessor of `target` runs at once: no more than its
    `programs` spread over the device's multiprocessors where it knows them, as a launch's blocks go to its
    multiprocessors in turn (`_running`'s states launch, `_output_hides`' output)."""
    if target.multiprocessors is None:
        return held
    return min(held, -(-programs // target.multiprocessors))


def _states_launch(launches: tuple[mainloop.Launch, ...]) -> mainloop.Launch:
    """The forward's states launch of one product's `launches`: the one whose ring its tiles' stages are
    (`mainloop.Launch.ring`)."""
    return next(launch for launch in launches if launch.ring is not None)


def _running(states: mainloop.Launch, target: generation.GpuTarget) -> int:
    """The programs of a states launch `states` a multiprocessor of `target` runs at once: those it holds
    (`Launch.resident`), but no more than its programs spread over the device's multiprocessors
    (`_spread`). A residency past them hides nothing: on an A100 Qwen3-Next's 128 programs 32 wide held three
    a multiprocessor one deep and two two deep, and ran two a multiprocessor either way, the forward 0.749 ms
    one deep against 0.727 two deep (the 27B's 0.943 against 0.916), where the one deep ring narrowed the
    block to 16 at 0.752 (1.193, gdn-pins-a100-3b0d2267d)."""
    return _spread(states.resident, states.programs, target)


def _output_hides(launches: tuple[mainloop.Launch, ...], target: generation.GpuTarget) -> bool:
    """Whether the output launch of one product's forward `launches` (`Mosaic._product_launches`) hides any
    latency on `target`: more than one program a multiprocessor running at once (`_spread`) or a ring more
    than one deep. One program one deep, the widest legal block held Kimi Linear's output on an A100
    latency-bound: 375.0 us of its training against 287.6 at a block of 64, two programs (the 27B gated
    delta net's 257.1 against 205.6); on a GH200 the 27B's one program two deep at 128 ran 84.2 us against
    89.9 at 64's two one deep, a narrower block hiding no more (M-JS-219)."""
    [output] = [launch for launch in launches if launch.name.split("__")[0].endswith("output")]
    return _spread(output.resident, output.programs, target) > 1 or output.stages > 1


@dataclasses.dataclass(frozen=True)
class Mosaic(base.Template[mainloop.Tiles]):
    """The chunked rule on the recurrence template's mainloop (`mosaic_rule`), its config
    the forward's tiles. Its operands are the rule's: queries and keys `[B, T, H, Dk]`, values
    `[B, T, HV, Dv]`, the log decay a head's `[B, T, HV]` or a key dimension's `[B, T, HV, Dk]`, the write
    gate `[B, T, HV]`, each row's initial memory and packed documents' first tokens, and their rows'
    documents' first slots as the batch built them (`firsts`, `dew.batch.Coordinates.firsts`); a forked
    call's documents (`forked`, `chunked_launch.Forked`), its initial memories its table's given ones."""

    name: ClassVar[str] = "mosaic_gpu"
    config_cls: ClassVar[type[mainloop.Tiles]] = mainloop.Tiles
    CARRIES_DERIVATIVE: ClassVar[bool] = True

    def _fwd(self, query: jax.Array, key: jax.Array, value: jax.Array, g: jax.Array, beta: jax.Array,
             state: jax.Array | None = None, starts: jax.Array | None = None, *,
             firsts: jax.Array | None = None, max_segments: int | None = None, scale: float | None = None,
             delta: bool = True, wide: bool = False, bound: float | None = None,
             rounded: bool = False, apart: int = 0, values_gated: bool = False, unit_pairs: bool = False,
             decay_bound: float | None = None,
                 skip: jax.Array | None = None,
             recomputed: bool = False,
             forked: chunked_launch.Forked | None = None, config: mainloop.Tiles,
             return_residuals: bool = False):
        del return_residuals
        return mosaic_rule(query, key, value, g, beta, state, tiles=config, starts=starts, firsts=firsts,
                           max_segments=max_segments, scale=scale, delta=delta, wide=wide, bound=bound,
                           rounded=rounded, apart=apart, values_gated=values_gated, unit_pairs=unit_pairs,
                           decay_bound=decay_bound,
                               skip=skip, recomputed=recomputed,
                           forked=forked), None

    @staticmethod
    def launched() -> generation.GpuTarget:
        """The target a call's launches run for: this device's where a bundle claims its family, else the
        interpreted target's (`generation.interpreted_target`): an assumed device's as it states it, so an
        interpreted check of its route derives its tiles; with none assumed, its multiprocessors unstated:
        an interpreter runs a launch's programs one after another, no wave of them to fill, so the
        derivation takes the widest tiles that hold (`derived`; packed documents' interpreted gradient took
        21 s a draw at the one-wave rule's 16-wide value block and 14 at 32, row 1850)."""
        target = generation.gpu_target()
        if bundle(target) is not None:
            return target
        return dataclasses.replace(generation.interpreted_target(), multiprocessors=None)

    def launches(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, target: generation.GpuTarget, *,
                  backward: bool, training_forward: bool = False) -> tuple[mainloop.Launch, ...]:
        """The launches the call `ba` runs at `config` by `target`'s bundle (`planned`): a forward without a
        gradient's; with `backward`, its fast backward's and the forward it takes where it takes the call's
        form (`mainloop_backward.fast_backward_form`); with `training_forward`, that forward alone, a training
        call's (the forward without a gradient's where the fast backward takes no such form)."""
        arguments = ba.arguments
        names = ("query", "key", "value", "g", "beta", "state", "starts")
        given = [arguments.get(name) for name in names]
        shapes = tuple(None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype) for x in given)
        dtype = jnp.result_type(*(arguments[name].dtype for name in names[:3]))
        family = bundle(target, dtype)
        assert family is not None, f"no bundle of the mainloop claims {target.generation}"
        delta = arguments.get("delta", True)
        fast = (backward or training_forward) and mainloop_backward.fast_backward_form(
            delta, arguments.get("rounded", False), dtype, per_key=arguments["g"].ndim == 4,
            wide=arguments.get("wide", False))
        # The skip fused into the output and reverse walk where given (its tiles their registers').
        skip = arguments.get("skip")
        forked, table, max_segments = arguments.get("forked"), None, arguments.get("max_segments")
        if forked is not None:
            # A forked call's walks over its documents laid as a packed call's at the config's chunk, and
            # its table.
            batch, length = arguments["query"].shape[:2]
            table, starts, _, _ = jax.eval_shape(
                functools.partial(forked_table, batch=batch, length=length, chunk=config.chunk), given[5],
                forked)
            shapes, max_segments = (*shapes[:5], None, starts), forked.loads.shape[1]
        return _planned(shapes, config, family, target, max_segments, delta, bool(fast) and backward,
                        arguments.get("rounded", False), arguments.get("wide", False),
                        ranged(dtype, delta, arguments.get("bound")),
                        None if skip is None else jax.ShapeDtypeStruct(skip.shape, skip.dtype), forked=table,
                        training_forward=bool(fast), apart=arguments.get("apart", 0),
                        values_gated=arguments.get("values_gated", False),
                        unit_pairs=arguments.get("unit_pairs", False),
                        decay_bound=arguments.get("decay_bound"))

    GENERATES: ClassVar[bool] = True

    def registers(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> int | None:
        """The most registers a consumer thread of any of the forwards' launches holds (`Launch.registers`,
        each launch's stated tiles, `resources.live_registers`; `_forwards`): a legal config's launches spill
        nothing. The fast backward's launches are held to the same budget where its rule takes them
        (`backward_holds`), and a call they do not hold takes no gradient of it (`trains`)."""
        target = self.launch_target()
        if bundle(target) is None:
            return None
        return max(launch.registers for launch in self._forwards(ba, config, target))

    def _forwards(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, target: generation.GpuTarget
                  ) -> tuple[mainloop.Launch, ...]:
        """The forwards' launches a call runs at `config`: a forward without a gradient's at its own tiles
        (`Tiles.forward`, `config` where None) and a training call's at `config` (`launches`'
        `training_forward`), each
        product's launches held to the tiles it runs at, not the other's (`_product_holds`)."""
        return (*self.launches(ba, config.forward or config, target, backward=False),
                *self.launches(ba, config, target, backward=False, training_forward=True))

    def register_budget(self, config: mainloop.Tiles) -> int:
        family = bundle(self.launch_target())
        return resources.facts().thread_registers if family is None else family.registers(1)

    def _product_holds(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                       *, training: bool, launches: tuple[mainloop.Launch, ...] | None = None) -> bool:
        """Whether one product's forward holds at `tiles` (`legal`'s rules on its launches alone: a training
        call's where `training`, `launches`' `training_forward`, else a forward without a gradient's): the
        tiles a rule stages for one product are held to the launches that product runs: a forward alone keeps
        nothing for a backward (`chunk_forward.kept_forward`'s `keeps`), a training call's forward keeps its
        memories and corrected values, each its own tiles (`Tiles.forward`, row 5668); `launches` that
        product's at `tiles` where the caller planned them (`_product_launches`)."""
        family = bundle(target, _dtype(ba))
        if family is None:
            return self.takes(ba, tiles, target)
        return (self._product_fits(ba, tiles, target, training=training, launches=launches)
                and not self._refused(ba, tiles, target, training=training))

    def _product_launches(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                          target: generation.GpuTarget, *, training: bool) -> tuple[mainloop.Launch, ...]:
        """One product's forward's launches at `tiles`: a training call's where `training`, else a forward
        without a gradient's (`launches`' `training_forward`)."""
        return self.launches(ba, tiles, target, backward=False, training_forward=training)

    def _product_fits(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                      *, training: bool, launches: tuple[mainloop.Launch, ...] | None = None) -> bool:
        """Whether one product's forward's launches fit `target` at `tiles` (`_product_holds` but a compile's
        refusal): each program's shared memory within a block, its threads' registers within their roles'
        budgets and a multiprocessor holding a program."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        if launches is None:
            launches = self._product_launches(ba, tiles, target, training=training)
        return (max(launch.shared for launch in launches) <= self.program_shared(tiles, target)
                and _launches_held(launches, family))

    def _refused(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                 *, training: bool) -> bool:
        """Whether a compile refused `tiles` past what its derivation's raise answers (`_exhausted`), or a
        launch it spilled so at other tiles plans at `tiles` as it did there but its grid (`_spilled`): the
        same kernel at the same registers, shared memory, ring, roles and residency, which a compile spills
        again along the same ladder. Read at its ladder's last rung, a launch that moves registers was never
        read as spilled, its base's raise answering it (train 929's F4). Asked a tiles at a time, lightning
        attention's reverse walk on an H100 took every rung of its ladder again at each of its tiles' eight
        ring depths, whose ring it does not read, and its training step raised Unfitted after 3,107 s (row
        6816); on an A100 its prefill certified its output walk fifteen times, fourteen spilling 336 bytes at
        255 registers stated alike, and the gated delta net's training call its scores' backward 32 times,
        each 8 bytes at 255 stated alike. Asked for one product (`training`: a training call's, else a
        forward alone's), its own kernels' refusals alone (`mainloop.ran_alone`), so a spill of one leaves
        the other's tiles as they are (row 10559)."""
        if self._exhausted(ba, tiles, target, training=training):
            return True
        # Planned only against a spilled launch's plans: where no compile spilled one, every first derivation
        # and every identity unit, its plans' traces read nothing (row 9697: a third of a derivation's plan
        # traces, jsf-holds).
        spilled = {(name, plan) for name, plan in self._spilled(ba, target) if _of(name, training)}
        return bool(spilled) and not spilled.isdisjoint(self._plans(ba, tiles, target).items())

    def _spilled(self, ba: tokamax.BoundArguments, target: generation.GpuTarget
                 ) -> frozenset[tuple[str, frozenset[tuple[bool, mainloop.Launch]]]]:
        """Each launch a compile spilled at every rung of its ladder from its base (`_spent`), by name with
        its plans at the ladder's base tiles but their grids in each product that runs it (`_plans`): kept by
        what its derivation reads (`report.derivation_inputs`: the call's identity, the target and its
        refusals), so a derivation's checks of its tiles read it once a refusal (planned at each, a refusal
        walk's 355 derivations took 148 s)."""
        key = report.derivation_inputs(self, ba, target)
        _, _, refusals, _ = key

        def made() -> frozenset[tuple[str, frozenset[tuple[bool, mainloop.Launch]]]]:
            # A spent launch's plans are its base tiles' (`_base`), where tiles planning it alike start the
            # same ladder.
            return frozenset(
                (name, plan) for config, kernels in refusals
                if isinstance(config, mainloop.Tiles)
                and (spent := self._spent(ba, config, report.names(kernels), target))
                for planned in [self._plans(ba, _base(config), target)]
                for owned in [_owners(spent, (launch for plan in planned.values() for _, launch in plan))]
                for name, plan in planned.items() if name in owned)

        return _cache.holding(_SPILLED, key, made, PLANS_KEPT)

    def _spent(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, kernels: frozenset[str],
               target: generation.GpuTarget) -> frozenset[str]:
        """The kernels of `kernels`, spilled at `config`, whose ladders a compile spent there: each stepping
        no further from `config` alone (`_stepped`: its producer at its foot and one program a
        multiprocessor), `config` on the walk its base takes (`rungs` from `_base(config)`), and each refused
        at every rung of it up to `config`. Another launch at its foot stops the walk short of this one's; a
        config pinned or inherited part way down a ladder (`neighbours`) leaves its base and upper rungs
        unasked; a rung that held this kernel and refused another is no evidence against it."""
        terminal = [kernel for kernel in kernels
                    if self._stepped(ba, config, frozenset({kernel}), target) is None]
        if not terminal:
            return frozenset()
        base = _base(config)
        walk = [base, *self.rungs(ba, base)]
        if config not in walk:
            return frozenset()
        refused = [report.refused_at(self, ba, rung, target) for rung in walk[:walk.index(config) + 1]]
        return frozenset(kernel for kernel in terminal if all(kernel in at for at in refused))

    def _exhausted(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                   *, training: bool) -> bool:
        """Whether a compile refused `tiles` past what its derivation's raise answers: a kernel spilled
        (`report.refused_at`), of the product `training` names (`_refused`), whose launch's producer can rise
        no further, or would leave its tiles unheld (`_raised`)."""
        spilled = frozenset(kernel for kernel in report.refused_at(self, ba, tiles, target)
                            if _of(kernel, training))
        return bool(spilled) and self._raised(ba, tiles, spilled, target) is None

    def _plans(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
               ) -> dict[str, frozenset[tuple[bool, mainloop.Launch]]]:
        """Each launch the call runs at `tiles` by name, as planned but its grid (`_ungridded`) in each
        product that runs it: a training call's, and a forward alone's at its own tiles (`Tiles.forward`). A
        refusal names a kernel, not the product whose compile spilled it, so a launch is alike where every
        product's plan of it is."""
        plans: dict[str, set[tuple[bool, mainloop.Launch]]] = collections.defaultdict(set)
        for backward in (True, False):
            for launch in self.launches(ba, tiles if backward else tiles.forward or tiles, target,
                                        backward=backward):
                plans[launch.name].add((backward, _ungridded(launch)))
        return {name: frozenset(plan) for name, plan in plans.items()}

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mainloop.Tiles | None:
        """The call's tiles on `target`, from its shape and the device's and its bundle's facts
        (docs/design/kernel-templates.md §7; no candidate space, no ranking: each rule gives way from the
        widest value block to the next narrower where it does not hold), each rule naming the fact it reads:
        - chunk, the work-span model's optimum (`chunk_of`), halved while the rules below find no tiles at
          it or the fast backward does not hold at them, down to a role's rows (`Tiles.chunk`);
        - value block, in steps of the narrowest a staged operand block takes (`Family.narrowest_columns`)
          dividing the padded values: the widest legal (`legal`: every launch's shared memory within a block
          of the device and its registers within a thread's, `registers`), then the widest at which the
          call's fast backward holds where it holds at any narrower (`_backward_holds`: its launches take the
          states launch's block, an H100's reverse launch 328 registers 64 wide), then the narrowest whose
          states launch, a program a value block, sequence and value head walking its sequence's chunks in
          turn, still runs in one wave of the programs the device's multiprocessors hold at once
          (`states_held`), else that widest one with walk roles (`_walked`): a wave's time is its walk's,
          whose steps wait on their loads, about as long a program 16 wide as 32 (an H100's 114 us a wave 16
          wide against 95 32 wide, the 27B's and Qwen3-Next's, SE-family-h100-b0980aaff), so one wave of the
          most programs (fla's chunk_fwd_h runs the 27B gated delta net's 192 programs a 32-wide block on an
          A100's 108, where 64 wide its 96 left twelve idle the whole walk, SE-rx-ncu-b106b18d8), and past one
          wave the fewest programs; the widest where the device's multiprocessors are unknown. Narrower in
          one wave only while the states ring holds one stage: a narrower program reloads the chunk's keys
          and weights whole, which a ring two deep hides at the wider block, so of one wave the widest two
          deep (keys and values 256 wide over 8 heads, one row: the states launch 0.200 ms 32 wide two deep
          against 0.212 16 wide on an H100, 0.285 against 0.367 to 0.415 on an A100, and 0.332 and 0.636 at
          64 wide one deep, batch-window-{h100,a100}-b2ad84c61);
        - output block: the widest legal beside it (`_staged`), the output launch's registers apart from the
          states launch's: on an A100 the output launch holds 264 registers 32 wide, where the states
          launch's 384 programs 16 wide took two waves of 593 us (SE-family-a100-0c8018352);
        - stages: the ring of the states launch's steps whose plan hides the most latency, its programs a
          multiprocessor running at once (`_running`) and its depth one choice (`_staged`,
          `resources.hidden`).
        - a forward without a gradient: the same rules without the fast backward's (`Tiles.forward`), where
          they run the states launch in one wave and the training call's do not: on an H100 the 27B gated
          delta net's forward took 0.513 ms at the training call's 32-wide block, its states launch's 192
          programs two waves, and 0.433 at 64 wide, one wave of 96, where fla's port took 0.440
          (JS-fwdtiles2-h100-a7ec9a134); past one wave either way the training call's, as keys 128 and
          values 256 wide over 16 heads of 8 rows took the states launch 0.114 ms 64 wide three deep against
          0.095 32 wide five deep (batch-window-h100-0b202f7cc).
        None where no block is legal: the launch refuses the call.

        A compile that refused the joined tiles (`report.refused_at`: kernels of the launch spilled) refuses
        the tiles each spilling kernel's product ran at, the training call's or the forward alone's
        (`Tiles.forward`), and that product's rules take the next they give: its ring a stage shallower, else
        the next narrower value block, as a refusal of a product's own tiles steps it (`legal`)."""
        family = bundle(target, _dtype(ba))
        if family is None:
            return None
        # Each step down a halving of the derived chunk, for legality alone: where no tiles hold, or the fast
        # backward does not hold at them, down to a role's rows.
        chunk = chunk_of(ba, family, target)
        while True:
            tiles = self._chunked(ba, target, chunk)
            if chunk == family.role_rows or (tiles is not None and self._backward_holds(ba, tiles, target)):
                return tiles
            chunk //= 2

    def _chunked(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, chunk: int
                 ) -> mainloop.Tiles | None:
        """`derived`'s tiles at `chunk`, past each product's tiles a compile refused, its launches' producers
        raised where a compile found them spilling (`_raised`)."""
        refused: dict[bool, set[mainloop.Tiles]] = {True: set(), False: set()}
        while True:
            trained = self._derive(ba, target, chunk, training=True, refused=refused[True])
            if trained is None:
                return None
            alone = self._derive(ba, target, chunk, training=False, refused=refused[False])
            family = bundle(target, _dtype(ba))
            # A forward alone may pad its values narrower than a training call does (`_widths`): its own
            # tiles.
            narrower = family is not None and (_widths(ba, family, training=False)[1]
                                               < _widths(ba, family)[1])
            # The training call's tiles serve a forward alone only where no compile refused them for it.
            serves = (trained not in refused[False]
                      and self._product_holds(ba, trained, target, training=False))
            if alone is not None and target.multiprocessors is not None and not narrower and (
                    not self._one_wave(ba, alone, target)
                    or self._one_wave(ba, trained, target, training=True)) and serves:
                alone = None
            elif alone is None and not serves:
                # No tiles at this chunk hold a forward alone, the training call's neither: none serve the
                # call (a forked Nemotron-H 4B SSD forward on sm100 at the training call's 96-wide block asked
                # 304,316 bytes of shared memory, its scan holding no program).
                return None
            forward = trained if alone in (None, trained) else alone
            joined = self._settled(ba, trained if forward is trained
                                   else dataclasses.replace(trained, forward=alone), target)
            if not _refused_kernels(self, ba, joined, target):
                return joined
            # Each product's tiles refused where a kernel it runs spilled past the last raise: the training
            # call's launches at its tiles, a forward alone's at its own (`Tiles.forward`, the training call's
            # where None). A refusal of the joined tiles is no refusal of either's (`legal`'s), so a forked
            # forward's spilling scan once refused its whole value block where a shallower ring held
            # (Nemotron-H 4B's on sm100).
            # A forward alone's launches are named apart from a training call's (`mainloop.Call.alone`), so a
            # refused kernel's name says which product ran it: each product steps its own tiles.
            spilled = report.refused_at(self, ba, joined, target)
            products = ((refused[True], trained, self.launches(ba, joined, target, backward=True), spilled),
                        (refused[False], forward,
                         self.launches(ba, joined.forward or joined, target, backward=False),
                         _refused_kernels(self, ba, joined, target)))
            ran = [(tiles_refused, tiles) for tiles_refused, tiles, launches, kernels in products
                   if _owners(kernels, launches)]
            refused_here = sorted(_refused_kernels(self, ba, joined, target))
            assert ran, f"the spilled kernels {refused_here} are none of the call's launches"
            for tiles_refused, tiles in ran:
                tiles_refused.add(tiles)

    def _settled(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
                 ) -> mainloop.Tiles:
        """`tiles` with their spilling launches' producers stepped along their ladders for as long as a
        compile refused them (`report.refused_at`: a training call's at its tiles, a forward alone's at its
        own, `Tiles.forward`) and a rung answers it (`_raised`): first each one's consumers at their
        statement's share (`resources.stated_producer`), then its producer's top, the program's entry share,
        which every role holds with no register moved, then a setmaxnreg step lower at a time, each rung
        giving its consumers a step more: where the statement falls short, the assembler finds a producer's
        registers. A spill at the top is the consumers', the producer holding there all it may; a rung below
        holds wherever the producer's need and the consumers' meet. Tiles still spilling at the ladder's foot
        their derivation refuses (`_chunked`). The 4B training forward's output walk took ten
        certifications of an H100's start to its share from the top down, two from the least through its
        statement's share (row 6743). Past its ladder's foot, a launch takes a program fewer a
        multiprocessor, its producer at the least and its ladder run again there (`_stepped`)."""
        while (spilled := _refused_kernels(self, ba, tiles, target)) and (
                stepped := self._stepped(ba, tiles, spilled, target)) is not None:
            tiles = stepped
        return tiles

    def emitted(self, config: Hashable, kernel: str) -> Hashable:
        """What `kernel` reads of `config` as it is emitted: all of it but the other launches' producers and
        residents (`Tiles.producers`, each launch's own setmaxnreg cap, and `Tiles.residents`, the call's
        tiles' and its forward's apart, each a level a launch reads), so a producer's raise leaves every other
        kernel of the launch the module it was, which a rung's certification then reads from the lowering
        cache and Mosaic's compile (`rungs`): a rung re-lowered and recompiled every kernel of the 4B's
        training launch, 7 and 5 s of each 12."""
        if not self._typed(config):
            return config
        # The tiles its launch is given (`Tiles.at`), its cap and residents by its name, which the launch
        # passes to `report.named` and reads of `Call.producers` and `Call.residents` alike.
        return config.at((kernel,))

    def rungs(self, ba: tokamax.BoundArguments, config: Hashable) -> Iterator[mainloop.Tiles]:
        """The tiles a refused `config`'s spilling launches step through (`_stepped`): each one's consumers'
        statement's share first, then its producer's top, then a setmaxnreg step lower a rung past the stated
        share, and past its foot a program fewer a multiprocessor
        with its producer's ladder run again, each planned as it is asked for: the ladder `_settled` takes a
        compile at a time, which certification asks in turn, the first that holds taken
        (`tuning.KernelOp._climbed`)."""
        if not self._typed(config):
            return
        target = generation.gpu_target()
        spilled, rung = _refused_kernels(self, ba, config, target), config
        while spilled and (stepped := self._stepped(ba, rung, spilled, target)) is not None:
            rung = stepped
            yield rung

    def _stepped(self, ba: tokamax.BoundArguments, joined: mainloop.Tiles, spilled: frozenset[str],
                 target: generation.GpuTarget) -> mainloop.Tiles | None:
        """`joined`'s next rung for its `spilled` launches: their producers a rung down their ladders
        (`_raised`), or, past a ladder's foot or where every rung left gives their consumers less
        (`_consumers_spilled`), those launches a program fewer a multiprocessor with their producers at the
        least again (`_fewer`), whose ladders then run at that residency; None past one program."""
        if not self._consumers_spilled(ba, joined, spilled, target) and (
                raised := self._raised(ba, joined, spilled, target)) is not None:
            return _limited(joined, producers=raised, residents=joined.residents)
        return self._fewer(ba, joined, spilled, target)

    def _consumers_spilled(self, ba: tokamax.BoundArguments, joined: mainloop.Tiles, spilled: frozenset[str],
                           target: generation.GpuTarget) -> bool:
        """Whether `joined`'s `spilled` kernels each spilled in its consumers' code alone
        (`report.spilled_in`, the training call's and its forward alone's) with its launch's producer at its
        consumers' statement's share, or at its least where it states none (`_stated_caps`): every rung its
        ladder has left raises the producer, giving the consumers less, which a program fewer gives more (row
        6744: a write-gated H100 scores' backward spilled 16 bytes in its consumers at the least, and its
        ladder compiled a hundred rungs from the top down, each spilling more, before the program fewer)."""
        roles = report.spilled_in(self, ba, joined, target) | (
            report.spilled_in(self, ba, joined.forward, target) if joined.forward else {})
        if any(roles.get(kernel) != (sass.CONSUMERS,) for kernel in spilled):
            return False
        family = bundle(target, _dtype(ba))
        assert family is not None
        moving = self._moving(ba, joined, spilled, target)
        stated = _stated_caps(moving, family)
        return all(joined.producers.of(launch.name) == stated.get(launch.name) for launch in moving)

    def _fewer(self, ba: tokamax.BoundArguments, joined: mainloop.Tiles, spilled: frozenset[str],
               target: generation.GpuTarget) -> mainloop.Tiles | None:
        """`joined` with its `spilled` launches whose producers' ladders ran out (`_moving`: a launch that
        moves no registers has no ladder, and its tiles are refused as before) a program fewer a
        multiprocessor than they plan now (`resources.Residents.fewer`), each the fewest its instances plan
        (the training call's and the forward alone's, each at its own tiles), their producers at the least;
        None where one plans one program already. Every role of a launch takes a larger share there, where
        the assembler may hold what no rung of its producer's ladder gave its consumers (row 6718)."""
        planned: dict[str, int] = {}
        for launch in self._moving(ba, joined, spilled, target):
            planned[launch.name] = min(planned.get(launch.name, launch.resident), launch.resident)
        residents = joined.residents.fewer(planned) if planned else None
        if residents is None:
            return None
        return _limited(joined, producers=joined.producers.without(planned), residents=residents)

    def _raised(self, ba: tokamax.BoundArguments, joined: mainloop.Tiles, spilled: frozenset[str],
                target: generation.GpuTarget) -> resources.Producers | None:
        """`joined`'s producers with those of its `spilled` launches (each kernel's own, `_owners`) at the
        next rung of their ladders (`resources.Producers.stepped`): the top, the program's entry share at its
        planned residency, which bounds a producer (past it the roles' moves invert and the assembler rejects
        the kernel), then a setmaxnreg step lower; each a launch whose program moves registers from its
        producer (`resources.moves_registers`, as its schedule does). None where none does or one is at its
        ladder's foot (its tiles then refused, the next derived). A rung moves registers within a program's
        pool, which its launch plans at the least (`mainloop.launch_kernel`): whether its roles hold is the
        assembler's, a compile that spills them refusing it."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        moving = self._moving(ba, joined, spilled, target)
        tops = {launch.name: resources.entry_registers(
            (launch.family or family).warpgroups(launch.consumers), launch.resident) for launch in moving}
        return joined.producers.stepped(tops, _stated_caps(moving, family)) if moving else None

    def _moving(self, ba: tokamax.BoundArguments, joined: mainloop.Tiles, spilled: frozenset[str],
                target: generation.GpuTarget) -> list[mainloop.Launch]:
        """Every instance of `joined`'s `spilled` launches (each kernel's own, `_owners`: a name the training
        call's backward and the forward alone can each plan at their own tiles) whose program moves registers
        from its producer (`resources.moves_registers`), each judged at its own schedule and residency."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        launches = (*self.launches(ba, joined, target, backward=True),
                    *self.launches(ba, joined.forward or joined, target, backward=False))
        owned = _owners(spilled, launches)
        return [launch for launch in launches if launch.name in owned
                and resources.moves_registers(launch.registers, (launch.family or family).warpgroups(
                    launch.consumers), launch.consumers, launch.resident)]

    def _derive(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, chunk: int, *,
                training: bool, refused: Set[mainloop.Tiles] = frozenset()) -> mainloop.Tiles | None:
        """`derived`'s rules at `chunk`, the fast backward's with them where `training`, past the tiles a
        compile `refused` (`_staged`)."""
        family = bundle(target, _dtype(ba))
        if family is None:
            return None
        blocks = self._blocks(ba, family, training=training)
        # Each rule gives way from the widest block to the next narrower where it does not hold, a block's
        # tiles staged as a rule first reads them: the widest legal, then the widest at which the fast
        # backward holds where it holds at any narrower, then the narrowest still running the states
        # launch in one wave.
        staged = functools.cache(lambda index: self._staged(ba, blocks[index], target, chunk,
                                                            training=training, refused=refused))
        start = next((index for index in range(len(blocks)) if staged(index) is not None), None)
        if start is None:
            return None
        def trains(index: int) -> bool:
            tiles = staged(index)
            return training and tiles is not None and self._backward_holds(ba, tiles, target)

        trained = next((index for index in range(start, len(blocks)) if trains(index)), start)
        chosen = staged(trained)
        assert chosen is not None
        if target.multiprocessors is None:
            return chosen
        def one_wave(tiles: mainloop.Tiles) -> bool:
            return self._one_wave(ba, tiles, target, training=training)

        if not one_wave(chosen):
            held = self.states_held(ba, chosen, target, training=training)
            return self._walked(ba, chosen, target,
                                self._programs(ba, chosen, target, training=training)
                                // (target.multiprocessors * held), training=training, refused=refused)
        # Narrower only while the fast backward holds where it held at the block given way from, and only
        # while the block's states ring holds one stage: of one wave, the widest whose ring holds two.
        held = trains(trained)
        for index in range(trained + 1, len(blocks)):
            if chosen.stages > 1:
                break
            tiles = staged(index)
            if tiles is None or not one_wave(tiles) or (held and not trains(index)):
                break
            chosen = tiles
        return chosen

    def _programs(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                  target: generation.GpuTarget, *, training: bool = True) -> int:
        """The walk's value blocks at `tiles`, its launch's grid as planned (`_states`): a block of each laid
        sequence and head, `tiles.walkers` of them a program, a forward's values padded as it pads them;
        a forked call's sequences its rows, each walking its documents in turn (`forked_table`)."""
        return self._states(ba, tiles, target, training=training).programs * tiles.walkers

    def _one_wave(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                  target: generation.GpuTarget, *, training: bool = False) -> bool:
        """Whether the walk's programs at `tiles` run in one wave of `target`'s multiprocessors, as many a
        multiprocessor as it holds (`states_held`)."""
        assert target.multiprocessors is not None, "a wave counts the device's multiprocessors"
        held = target.multiprocessors * self.states_held(ba, tiles, target, training=training)
        return self._programs(ba, tiles, target, training=training) <= held

    def _walked(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                waves: int, *, training: bool = True, refused: Set[mainloop.Tiles] = frozenset()
                ) -> mainloop.Tiles:
        """`tiles` with the value blocks of one sequence and head a walk program takes (`Tiles.walkers`),
        one a consumer role beside one producer, where its states launch's programs fill `waves` waves of
        the device's: the most roles the family's program takes (`Family.consumers`) that divide the head's
        value blocks, no more than its waves (a grouped grid still fills a wave), whose launches hold their
        registers at that many roles' budget (`Family.registers`) and their shared memory, at its stage
        rule (`_staged`); one role else. A walk's chain waits a chunk on its products: the states launch
        took 1.04 ms over 66 waves of programs of about 6.6 chunks in a Qwen3.5-4B training step at 4x6,144
        tokens a device on an H100, one 256-thread program a multiprocessor (two roles hold its 128
        registers' need and the reverse launch's 184 within their 232, and read the head's keys once for
        both)."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        _, columns, _ = _widths(ba, family, training=training)
        blocks = columns // tiles.block_v
        for walkers in range(min(walk_roles(family), waves), 1, -1):
            if blocks % walkers:
                continue
            grouped = self._staged(ba, tiles.block_v, target, tiles.chunk, walkers=walkers,
                                   wide=tiles.block_o or tiles.block_v, training=training, refused=refused)
            if grouped is not None and (not training or self._backward_holds(ba, grouped, target)
                                        == self._backward_holds(ba, tiles, target)):
                return grouped
        return tiles

    def trains(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the call's gradient runs on `target`: its fast backward takes its form
        (`mainloop_backward.fast_backward_form`: the delta rule with a decay a head or a key dimension, a bf16
        call without it and a decay a head or none, its output's gradient bf16 where it is not rounded,
        either in fp32 under the fast forms) and its launches
        hold at the call's derived tiles (`_backward_holds`), the rule's gradient (`mosaic_rule`) refusing
        every other call."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launch_target()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        forms = mainloop_backward.fast_backward_form(arguments.get("delta", True),
                                                     arguments.get("rounded", False), dtype,
                                                     per_key=arguments["g"].ndim == 4,
                                                     wide=arguments.get("wide", False))
        if not forms:
            return False
        fits = self.fits(ba, target)
        return bool(fits) and self._backward_holds(ba, fits[0], target)

    def _backward_holds(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                        target: generation.GpuTarget) -> bool:
        """Whether the launches the call's fast backward runs, its own and the forward it takes, hold at
        `tiles` (`holds`; a call taking none, `launches`', holds its forward's)."""
        both = self.launches(ba, tiles, target, backward=True)
        family = bundle(target, _dtype(ba))
        assert family is not None
        return holds(both, family, generation.shared_budget(target))

    def states_held(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                    *, training: bool = False) -> int:
        """The programs of the walk at `tiles` a multiprocessor of `target` holds at once (`resources.plan`,
        as its persistent launches'), a training call's where `training` (`_states`)."""
        return self._states(ba, tiles, target, training=training).resident

    def _states(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget, *,
                training: bool = False) -> mainloop.Launch:
        """The forward's walk over each sequence's chunks, its states launch: a forward without a gradient's,
        or where `training` the forward its fast backward takes (`chunk_forward.kept_forward`)."""
        return _states_launch(self.launches(ba, tiles, target, backward=False, training_forward=training))

    def _blocks(self, ba: tokamax.BoundArguments, family: mosaic.Family, *, training: bool = True
                ) -> list[int]:
        """The value blocks a call takes on `family`'s bundle, widest first (`derived`'s first rule before
        legality): those dividing its padded values (a forward's as it pads them, `_widths`) in steps of the
        narrowest a staged operand block takes (`Family.narrowest_columns`, at the 16-bit operands its
        products take)."""
        _, columns, _ = _widths(ba, family, training=training)
        unit = family.narrowest_columns(jnp.bfloat16)
        return [block for block in range(columns - columns % unit, 0, -unit) if columns % block == 0]

    def _staged(self, ba: tokamax.BoundArguments, block: int, target: generation.GpuTarget, chunk: int, *,
                walkers: int = 1, wide: int | None = None, training: bool = False,
                refused: Set[mainloop.Tiles] = frozenset()) -> mainloop.Tiles | None:
        """`block`'s tiles at `chunk` and `walkers` walk roles (`Tiles.walkers`) with the output block legal
        beside it whose output launch hides any latency, the widest where none does (`Tiles.block_o`,
        `_output_hides`; `wide` where given), at the ring of the states launch's steps,
        within the GPU's block (`generation.shared_budget`) over a step's bytes, whose plan hides the most
        latency, its programs a multiprocessor running at once (`_running`) and its depth one choice
        (`resources.hidden`), past the tiles a compile `refused` (`_chunked`'s); None where no output block is
        legal beside it."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        widths = self._blocks(ba, family, training=training) if wide is None else [wide]

        # Widest first: the first legal width whose output launch hides latency, else the first legal, each
        # width's launches planned once, for its fit, its output's latency and its rings alike.
        widest = shallowest = None
        for width in widths:
            tiles = mainloop.Tiles(block, 1, None if width == block else width, walkers, chunk=chunk)
            if tiles in refused:
                continue
            launches = self._product_launches(ba, tiles, target, training=training)
            if not self._product_holds(ba, tiles, target, training=training, launches=launches):
                continue
            widest = widest or (tiles, launches)
            if _output_hides(launches, target):
                shallowest = (tiles, launches)
                break
        shallowest = shallowest or widest
        if shallowest is None:
            return None
        tiles, launches = shallowest
        states = _states_launch(launches)
        order = resources.hidden(family, states.consumers)
        # Each deeper ring's launches the shallowest's deepened (`Launch.deepened`, as `_planned` deepens its
        # one trace): planned again, each depth traced the call's layout again, a forked call's table
        # (`forked_table`) among it (row 12086).
        rings = [(tiles, launches)]
        for stages in range(2, min(resources.MOST_STAGES,
                                   generation.shared_budget(target) // max(states.step_bytes, 1)) + 1):
            deep = dataclasses.replace(tiles, stages=stages)
            deepened = tuple(launch.deepened(stages, target) for launch in launches)
            if deep not in refused and self._product_holds(ba, deep, target, training=training,
                                                           launches=deepened):
                rings.append((deep, deepened))
        def hidden(ring) -> tuple:
            deep, deepened = ring
            return order(_running(_states_launch(deepened), target), deep.stages)

        return max(rings, key=hidden)[0]

    def neighbours(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> list[mainloop.Tiles]:
        """The tiles the opt-in tuner times beside the derived `config` (`variants`; the gate holds at each):
        the next value block either side of it at its stage rule (`_staged`), and its block one stage
        shallower (or two deep where it takes one), their launches' producers raised as the derived config's
        are (`_settled`)."""
        target = self.launch_target()
        family = bundle(target, _dtype(ba))
        assert family is not None
        blocks = self._blocks(ba, family)
        at = blocks.index(config.block_v)
        beside = [self._staged(ba, blocks[index], target, config.chunk) for index in (at - 1, at + 1)
                  if 0 <= index < len(blocks)]
        other = dataclasses.replace(config, stages=config.stages - 1 if config.stages > 1 else 2)
        # Each at the derived config's producers and residents, stepped past its own compiles' spills
        # (`_settled`).
        return [self._settled(ba, _limited(tiles, producers=config.producers, residents=config.residents),
                              target)
                for tiles in (*beside, other) if tiles is not None]

    def staged_bytes(self, ba: tokamax.BoundArguments, config: mainloop.Tiles,
                     target: generation.GpuTarget) -> int:
        # The forwards' launches at their own tiles (`_forwards`): the fast backward's are held where its rule
        # takes them.
        return max(launch.shared for launch in self._forwards(ba, config, target))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the mainloop runs the call on `target`: 16-bit or fp32 operands, a decay a key dimension's
        in a 16-bit call or one under the run's fast forms (`mainloop.per_key_refusal`), a
        16-bit delta rule's solve at fp32's range where its form proves no bound (`ranged`), a bundle for the
        target's family, the tensor rate its cost model reads, and a config whose launches the device holds
        at the call's widths (`fits`: their shared memory and registers; for a training call's launch where a
        device certifies, `holds`, the config its certification takes, none where every config its
        derivation offers spilled: the gated delta net's
        scores' backward on an A100 spilled 8 bytes at 255 registers at each, and its training step raised
        Unfitted from inside the call, row 6816). Off a GPU (a CPU's), where the rule
        runs its bundle interpreted, the target is the interpreted bundle's generation (`launched`)."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launch_target()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        if dtype not in (jnp.bfloat16, jnp.float16, jnp.float32):
            return False
        taken = (arguments["g"].ndim == 3
                 or mainloop.per_key_refusal(dtype, rounded=arguments.get("rounded", False)) is None)
        family = bundle(target, dtype)
        if not (taken and family is not None and target.tensor_rate("16bit") is not None):
            return False
        # A readout apart from a group of tokens' writes (`mainloop.Call.apart`) the template forms where
        # every chunk it derives holds whole groups: its chunks are its role's rows times a power of two
        # (`chunk_of`), so where the group divides the role's rows (a rank of three refused, train 1000's F1).
        if family.role_rows % max(arguments.get("apart", 0), 1):
            return False
        # A training call's launch, selected as reverse mode runs it (`recurrence_chunk._per_product`), holds
        # where its certification takes a config (`holds`), as the call's own certification then reads it
        # back. A forward alone is certified as its call runs it: selected beside a training call's, which a
        # step runs instead, it asked a compile of a launch no step ran (train 929's F5). A gradient the fast
        # backward does not take (`trains`) is the route's backward requirement's to refuse
        # (`recurrence_chunk.TEMPLATE_SUPPORTS`): a certification of its pullback raises in the rule's
        # gradient (`mosaic_rule`).
        if report.pulling() and self.trains(ba, target):
            return self.holds(ba)
        return bool(self.fits(ba, target))

    def launch_target(self) -> generation.GpuTarget:
        # Its configs derived for the target it launches for (`launched`), the interpreted one's off a GPU.
        return self.launched()



def _refused_kernels(op, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
                     ) -> frozenset[str]:
    """The kernels a compile refused at `tiles` on `target` (`report.refused_at`): its training call's and its
    forward alone's, at its own tiles where it has them (`Tiles.forward`)."""
    return report.refused_at(op, ba, tiles, target) | (report.refused_at(op, ba, tiles.forward, target)
                                                       if tiles.forward else frozenset())


def _of(kernel: str, training: bool) -> bool:
    """Whether `kernel` is the product's `training` names (a training call's, else a forward alone's,
    `mainloop.ran_alone`)."""
    return mainloop.ran_alone(kernel) != training


def _owner(kernel: str, names: Iterable[str]) -> str | None:
    """The launch of `names` that ran `kernel`: the longest it starts with, a name another launch's extends
    (`recurrence_intra` beside `recurrence_intra_solve`) owning only its own."""
    return next((name for name in sorted(names, key=len, reverse=True) if kernel.startswith(name)), None)


def _base(config: mainloop.Tiles) -> mainloop.Tiles:
    """`config` before any rung of its launches' ladders (`rungs`): no producer raised, no program held
    fewer, its forward's neither."""
    return _limited(config, producers=resources.Producers(), residents=resources.Residents())


def _ungridded(launch: mainloop.Launch) -> mainloop.Launch:
    """`launch` as planned but its grid (`Launch.programs`), which no thread's registers or shared memory
    read: what a compile of its kernel spills at."""
    return dataclasses.replace(launch, programs=0)


def _owners(kernels: Iterable[str], launches: Iterable[mainloop.Launch]) -> frozenset[str]:
    """The launches that ran `kernels`, by name (`_owner`)."""
    names = {launch.name for launch in launches}
    return frozenset(owner for kernel in kernels if (owner := _owner(kernel, names)))


def _stated_caps(launches: Sequence[mainloop.Launch], family: mosaic.Family) -> dict[str, int]:
    """Each of `launches`' names whose every instance's consumers' statement leaves its producer more than the
    least (`resources.stated_producer`), at the fewest its instances leave it, so each holds its consumers'
    statement."""
    caps: dict[str, int | None] = {}
    for launch in launches:
        cap = resources.stated_producer(launch.registers, (launch.family or family).warpgroups(
            launch.consumers), launch.consumers, launch.resident)
        held = caps.get(launch.name, cap)
        caps[launch.name] = None if cap is None or held is None else min(cap, held)
    return {name: cap for name, cap in caps.items() if cap is not None}


def _limited(tiles: mainloop.Tiles, *, producers: resources.Producers,
             residents: resources.Residents) -> mainloop.Tiles:
    """`tiles`, its forward's too, its launches' producers lowered to `producers` and held to `residents`'
    programs a multiprocessor."""
    def held(each: mainloop.Tiles) -> mainloop.Tiles:
        return dataclasses.replace(each, producers=producers, residents=residents)

    return dataclasses.replace(held(tiles), forward=None if tiles.forward is None else held(tiles.forward))


MOSAIC = Mosaic()
"""The implementation a call runs (`recurrence_chunk.chunk`'s 'mosaic' mainloop, `GATED_DELTA`'s kernel)."""
