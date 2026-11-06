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
import threading
from collections.abc import Set
from typing import ClassVar

import jax
import jax.numpy as jnp
import tokamax
from jax.ad_checkpoint import checkpoint_name

from dew.nn import derivatives
from dew.nn.kernels import generation
from dew.nn.kernels.templates import base, bundles, mosaic, report, resources
from dew.nn.kernels.templates.recurrence import mainloop, mainloop_backward


def _widths(ba: tokamax.BoundArguments, family: mosaic.Family, *, training: bool = True
            ) -> tuple[int, int, jnp.dtype]:
    """The call's keys' and values' widths padded to whole tiles of `family`'s bundle (`mainloop.block_of`),
    a forward's values (not `training`) to whole value tiles (`mainloop.value_unit`), and the dtype its
    tiles hold (the operands', fp32 in an fp32 call)."""
    arguments = ba.arguments
    dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
    pad = mainloop.block_of(family)
    unit = pad if training else mainloop.value_unit(family)
    return (-(-arguments["key"].shape[-1] // pad) * pad, -(-arguments["value"].shape[-1] // unit) * unit,
            jnp.dtype(dtype))


def _dtype(ba: tokamax.BoundArguments) -> jnp.dtype:
    """The call's operands' common dtype."""
    return jnp.dtype(jnp.result_type(*(ba.arguments[name].dtype for name in ("query", "key", "value"))))


def bundle(target: generation.GpuTarget, dtype=jnp.bfloat16) -> mosaic.Family | None:
    """The mainloop's bundle for `target`'s instruction family and a call of `dtype` (`bundles.of`), None
    where none claims it."""
    return bundles.of(target.generation, mainloop.bundle_dtype(dtype))



def planned(q, k, v, g, beta, h0=None, *, tiles: mainloop.Tiles, family: mosaic.Family, starts=None,
            max_segments: int | None = None, delta: bool = True, backward: bool = False,
            floor: float | None = None, target: generation.GpuTarget | None = None,
            rounded: bool = False, wide: bool = False, ranged: bool = False,
            skip=None) -> tuple[mainloop.Launch, ...]:
    """The launches a call over operands of these shapes (`jax.ShapeDtypeStruct`s, `mainloop.forward`'s) runs
    at `tiles` by `family`'s bundle on `target` (the device's, or the assumed one's, where None), in order, as
    planned (`mainloop.Launch`): its arrays abstract (`jax.eval_shape`), its kernels declared and not built
    (`mainloop.launch_kernel`): a forward without a gradient's (`mainloop.forward`'s); with `backward`, the
    forward a fast backward takes (`mainloop.kept_forward`'s) and that backward's
    (`mainloop_backward.gradients`). A `wide` call's output lands in fp32
    (`mainloop.forward`'s): planned in the operands' dtype, the SSD's output launch asked 16 KB past its plan
    (204800 bytes against 188544 on an H100's compile, the resource gate's). A `skip` (`[HV]`, its shape)
    fused into the output and reverse walk is planned with them: its reverse walk holds a tile more."""
    found: list[mainloop.Launch] = []

    def run(q, k, v, g, beta, h0, starts, skip):
        token = mainloop.PLANNED.set(found)
        try:
            out, final, (inverse, held) = mainloop.kept_forward(
                q, k, v, g, beta, h0, tiles=tiles, starts=starts, max_segments=max_segments, delta=delta,
                family=family, floor=floor, rounded=rounded, wide=wide, ranged=ranged, skip=skip,
                keeps=backward)
            if backward:
                mainloop_backward.gradients(q, k, v, g, beta, h0, out, final, tiles=tiles, starts=starts,
                                            max_segments=max_segments, family=family, inverse=inverse,
                                            kept=held, delta=delta, floor=floor, rounded=rounded,
                                            ranged=ranged, skip=skip)
        finally:
            mainloop.PLANNED.reset(token)

    with contextlib.nullcontext() if target is None else generation.assumed(target):
        jax.eval_shape(run, q, k, v, g, beta, h0, starts, skip)
    return tuple(found)


def holds(launches: tuple[mainloop.Launch, ...], family: mosaic.Family, shared: int) -> bool:
    """Whether each of `launches` fits a block of `shared` bytes, its consumer threads hold their registers
    (`Launch.registers` within `Family.registers` at its consumer roles, of the bundle it issues by,
    `Launch.family`, `family` where it names none) and a multiprocessor holds a program of it
    (`Launch.resident`): a generated launch spills nothing."""
    return all(launch.shared <= shared and launch.resident > 0
               and launch.registers <= (launch.family or family).registers(launch.consumers)
               for launch in launches)


def backward_holds(q, k, v, g, beta, h0=None, *, tiles: mainloop.Tiles, family: mosaic.Family,
                   target: generation.GpuTarget, starts=None, max_segments: int | None = None,
                   delta: bool = True, floor: float | None = None, rounded: bool = False,
                   ranged: bool = False, skip=None) -> bool:
    """Whether the launches a fast backward runs (`mainloop_backward.gradients`'s and the forward it takes,
    `mainloop.kept_forward`'s) hold on `family`'s bundle at `tiles` on `target`, planned there and within its
    block's shared memory (`holds`); where they do not, the rule's gradient is the portable math's. Planned on
    the device's own target off a GPU, an interpreted call's states launch held no program at the interpreted
    target's shared budget (`Launch.resident`). Each plan is the call's cached one (`_planned`): planned
    afresh at each of a Qwen3.5-4B-width step's 168 traces of its GDN layers, it took 11.37 s of its 182.9 s
    compile_ahead on sm90, its 425 plans 13.34 s (cProfile, armada job 20261105140541-69454bb5 at
    base/kernels 384436fbe8, row 4260)."""
    *shapes, skip = (None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype)
                     for x in (q, k, v, g, beta, h0, starts, skip))
    both = _planned(tuple(shapes), tiles, family, target, max_segments, delta, backward=True, floor=floor,
                    rounded=rounded, ranged=ranged, skip=skip)
    return holds(both, family, generation.shared_budget(target))

PLANS_KEPT = 1024
"""The plans `_planned` keeps, the least recently read given way first: a Qwen3.5-4B-width step's
compile_ahead planned 425 (row 4260's profile), and every plan kept for the process's life held each
geometry a derivation ever tried (row 4390)."""

_PLANS: collections.OrderedDict = collections.OrderedDict()
"""`_planned`'s plans by their call, tiles, bundle and target's facts, the most recently read last."""
_PLANNING = threading.Lock()


def _planned(shapes: tuple, tiles: mainloop.Tiles, family: mosaic.Family, target: generation.GpuTarget,
             max_segments: int | None, delta: bool, backward: bool,
             floor: float | None, rounded: bool = False, wide: bool = False,
             ranged: bool = False, skip: jax.ShapeDtypeStruct | None = None) -> tuple[mainloop.Launch, ...]:
    """`planned` of a call over `shapes` (`jax.ShapeDtypeStruct`s or None) by `family`'s
    bundle on `target`, cached by the facts a plan reads of it (two targets of one family share no plan: an
    sm87's and an sm80's multiprocessors and shared memory differ), `PLANS_KEPT` of them."""
    facts = (target.generation, target.device_kind, target.shared_limit, target.multiprocessors)
    key = (shapes, tiles, family, facts, max_segments, delta, backward, floor, rounded, wide, ranged, skip)
    with _PLANNING:
        if key in _PLANS:
            _PLANS.move_to_end(key)
            return _PLANS[key]
    *operands, starts = shapes
    found = planned(*operands, tiles=tiles, family=family, starts=starts, max_segments=max_segments,
                    delta=delta, backward=backward, floor=floor, target=target, rounded=rounded, wide=wide,
                    ranged=ranged, skip=skip)
    with _PLANNING:
        _PLANS[key] = found
        while len(_PLANS) > PLANS_KEPT:
            _PLANS.popitem(last=False)
    return found


def ranged(dtype, delta: bool, bound: float | None) -> bool:
    """Whether a call's solve holds its chunk inverse at fp32's range (`mainloop.Call.ranged`): a 16-bit
    delta rule whose form proves no bound on the inverse's entries."""
    return delta and jnp.dtype(dtype) != jnp.float32 and bound is None


def mosaic_rule(query, key, value, g, beta, state=None, *, tiles, starts=None,
                max_segments: int | None = None, scale: float | None = None, delta: bool = True,
                wide: bool = False, family=None, floor: float | None = None, bound: float | None = None,
                firsts=None, rounded: bool = False, skip=None, recomputed: bool = False, every: bool = False):
    """The chunked rule, a decay a head (`g` `[B, S, Hv]`) or a key dimension (`[B, S, Hv, Dk]`), with or
    without the `delta` rule, on the recurrence template's Mosaic GPU mainloop (`mainloop.forward`): the
    port's operands and results (`recurrence_chunk.fla_rule`'s), the operands in their common dtype and beta
    with them, the decay and the memory fp32, on one device, by the bundle of its instruction family
    (`mainloop.family_of`, `bundles.of`), interpreted off a GPU (Mosaic GPU's interpreter,
    `bundles.INTERPRETED`'s). Its gradient: the gated delta rule's in a 16-bit call on the mainloop
    (`mainloop_backward.gradients`, fla's DAG, fla's rounding) where its launches hold on the device
    (`backward_holds`); it takes no other call's, which raises NotImplementedError as the gradient is traced:
    the delta rule's selector gives a training call it refuses another kernel (`Mosaic.trains`), and a call
    that names the mainloop insists on it (ledger 1926: no kernel falls back inside itself). Its states and
    output launches take `tiles` (`mainloop.Tiles`, the template's config); a `family` given is the bundle
    it issues by, interpreted off a GPU of that family (a CPU's). A decay a key dimension takes the layer's
    declared `floor`, which the mainloop's intra launch reads (`mainloop.per_key_refusal`), and a 16-bit
    delta rule the chunk inverse's `bound` its form proves, its solve at fp32's range where it proves none
    (`ranged`); a `rounded` call takes the run's fast forms' numbers (`recurrence_chunk.fast_rounding`,
    `mainloop.forward`). Where `recomputed`, its fast backward forms the forward's memories again rather than
    keeping them (a remat rung recomputing 'delta_memory', `recurrence_chunk.RECOMPUTED`, resolved at its
    route). Where `every` packed document (`starts`) enters the row's `state`, not its first alone
    (`recurrence_chunk.Chunk.entered`), the state's gradient sums every document's."""

    dtype = jnp.result_type(query, key, value)
    device, interpret = mainloop.family_of(dtype), generation.backend() != 'gpu'
    issuing = family or device or bundles.INTERPRETED
    query, key, value = (x.astype(dtype) for x in (query, key, value))
    # The delta rule's write gate in the operands' dtype, as fla's layer holds it; without the delta rule the
    # call's own, as the portable math takes it (Mamba-2's fp32 `dt`).
    beta = beta.astype(dtype) if delta else beta
    g = g.astype(jnp.float32)
    # Interpreted off a GPU (an assumed one's calls lower for it); a GPU no bundle claims runs no launch.
    if family is None and device is None and not interpret:
        raise NotImplementedError(f"the recurrence template has no bundle for the "
                                  f"{generation.family(generation.device_generation())!r} family")
    run = {"family": issuing, "interpret": interpret, "tiles": tiles, "floor": floor,
           "ranged": ranged(dtype, delta, bound)}
    # The fast backward where its launches hold on the device (a derived launch spills nothing), else none.
    fast = (mainloop_backward.fast_backward_form(delta, rounded, dtype, per_key=g.ndim == 4)
            and backward_holds(
                query, key, value, g, beta, state, tiles=tiles, family=issuing,
                target=Mosaic.launched(), starts=starts,
                max_segments=max_segments, delta=delta, floor=floor, rounded=rounded, ranged=run["ranged"],
                skip=skip))

    # The launch tracing the forward (`report.launch_digest`), whose name the fast backward's kernels carry
    # too (`report.relaunched`): JAX traces the backward after that launch's scope ended.
    launch = report.launch_digest()

    # The documents' starts and firsts operands, not closed over (a custom VJP's function closes over no
    # tracer).
    def primal(query, key, value, g, beta, state, starts, firsts, skip):
        # A forward without a gradient at its own tiles, no fast backward narrowing them (`Tiles.forward`).
        alone = {**run, "tiles": tiles.forward or tiles}
        return mainloop.forward(query, key, value, g, beta, state, starts=starts, firsts=firsts,
                                max_segments=max_segments, scale=scale, delta=delta, wide=wide,
                                rounded=rounded, skip=skip, every=every, **alone)

    def forward(*operands):
        if not fast:
            return primal(*operands), (operands, (None, None))
        # The chunks' inverses kept for the fast backward, as fla's keeps its `A`, and the launches' memories,
        # weighted keys and corrected values, as the port keeps its memory, but where the block's remat rung
        # recomputes them (`recomputed`). Each is named as the port names its own, so a rung that saves names
        # keeps them rather than replaying the forward's launches for them (`mixer_base.DELTA_RESIDUALS`).
        out, final, (inverse, kept) = mainloop.kept_forward(*operands[:6], starts=operands[6],
                                                            firsts=operands[7], max_segments=max_segments,
                                                            scale=scale, delta=delta, wide=wide,
                                                            rounded=rounded, skip=operands[8], every=every,
                                                            **run)
        kept = None if recomputed else checkpoint_name(kept, 'delta_memory')
        return (out, final), (operands, (checkpoint_name(inverse, 'delta_inverse'), kept))

    def backward(saved, grads):
        operands, (inverse, kept) = saved
        out_grad, final_grad = grads
        query, key, value, g, beta, state, starts, firsts, skip = operands
        if not fast:
            form = (f"{dtype.name}, {'the delta rule' if delta else 'no delta rule'}, a decay a "
                    f"{'head' if g.ndim == 3 else 'key dimension'}")
            raise NotImplementedError(
                f"the recurrence template's fast backward takes no gradient of this call ({form}, or its "
                "launches past the device's at these tiles): the delta rule's selector gives a training call "
                "it refuses the port or the portable math (`MOSAIC_SUPPORTS`' 'backward')")
        with report.relaunched(launch):
            *grads, dh0, dskip = mainloop_backward.gradients(
                query, key, value, g, beta, state, out_grad, final_grad, starts=starts, firsts=firsts,
                max_segments=max_segments, scale=scale, inverse=inverse, kept=kept, delta=delta,
                rounded=rounded, skip=skip, every=every, **run)
        return (*grads, None if state is None else dh0, None, None,
                None if dskip is None else dskip.astype(skip.dtype))

    rule = derivatives.declared_derivative(derivatives.KernelGlue(), primal, forward, backward)
    return rule(query, key, value, g, beta, state, starts, firsts, skip)


@dataclasses.dataclass(frozen=True)
class Mosaic(base.Template[mainloop.Tiles]):
    """The chunked rule on the recurrence template's mainloop (`mosaic_rule`), its config
    the forward's tiles. Its operands are the rule's: queries and keys `[B, T, H, Dk]`, values
    `[B, T, HV, Dv]`, the log decay a head's `[B, T, HV]` or a key dimension's `[B, T, HV, Dk]`, the write
    gate `[B, T, HV]`, each row's initial memory and packed documents' first tokens, and their rows'
    documents' first slots as the batch built them (`firsts`, `dew.batch.Coordinates.firsts`)."""

    name: ClassVar[str] = "mosaic_gpu"
    config_cls: ClassVar[type[mainloop.Tiles]] = mainloop.Tiles

    def _fwd(self, query: jax.Array, key: jax.Array, value: jax.Array, g: jax.Array, beta: jax.Array,
             state: jax.Array | None = None, starts: jax.Array | None = None, *,
             firsts: jax.Array | None = None, max_segments: int | None = None, scale: float | None = None,
             delta: bool = True, wide: bool = False, floor: float | None = None, bound: float | None = None,
             rounded: bool = False, skip: jax.Array | None = None, recomputed: bool = False,
             every: bool = False, config: mainloop.Tiles, return_residuals: bool = False):
        del return_residuals
        return mosaic_rule(query, key, value, g, beta, state, tiles=config, starts=starts, firsts=firsts,
                           max_segments=max_segments, scale=scale, delta=delta, wide=wide, floor=floor,
                           bound=bound, rounded=rounded, skip=skip, recomputed=recomputed, every=every), None

    @staticmethod
    def launched() -> generation.GpuTarget:
        """The target a call's launches run for: this device's where a bundle claims its family, else the
        interpreted bundle's generation's (`generation.INTERPRETED_GENERATION`), its multiprocessors
        unstated: an interpreter runs a launch's programs one after another, no wave of them to fill, so the
        derivation takes the widest tiles that hold (`derived`; packed documents' interpreted gradient took
        21 s a draw at the one-wave rule's 16-wide value block and 14 at 32, row 1850)."""
        target = generation.gpu_target()
        if bundle(target) is not None:
            return target
        interpreted = generation.gpu_target(generation.INTERPRETED_GENERATION)
        return dataclasses.replace(interpreted, multiprocessors=None)

    def _launches(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, target: generation.GpuTarget, *,
                  backward: bool) -> tuple[mainloop.Launch, ...]:
        """The launches the call `ba` runs at `config` by `target`'s bundle (`planned`);
        with `backward`, its fast backward's too where it takes it (a decay a head, 16-bit operands)."""
        arguments = ba.arguments
        names = ("query", "key", "value", "g", "beta", "state", "starts")
        given = [arguments.get(name) for name in names]
        shapes = tuple(None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype) for x in given)
        dtype = jnp.result_type(*(arguments[name].dtype for name in names[:3]))
        family = bundle(target, dtype)
        assert family is not None, f"no bundle of the mainloop claims {target.generation}"
        delta = arguments.get("delta", True)
        fast = backward and mainloop_backward.fast_backward_form(
            delta, arguments.get("rounded", False), dtype, per_key=arguments["g"].ndim == 4)
        # The skip fused into the output and reverse walk where given (its tiles their registers').
        skip = arguments.get("skip")
        return _planned(shapes, config, family, target, arguments.get("max_segments"), delta, bool(fast),
                        arguments.get("floor"), arguments.get("rounded", False), arguments.get("wide", False),
                        ranged(dtype, delta, arguments.get("bound")),
                        None if skip is None else jax.ShapeDtypeStruct(skip.shape, skip.dtype))

    GENERATES: ClassVar[bool] = True

    def registers(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> int | None:
        """The most registers a consumer thread of any of the forward's launches holds (`Launch.registers`,
        each launch's stated tiles, `resources.live_registers`): a legal config's launches spill nothing. The
        fast backward's launches are held to the same budget where its rule takes them
        (`backward_holds`), and a call they do not hold takes no gradient of it
        (`trains`)."""
        target = self.launched()
        if bundle(target) is None:
            return None
        return max(launch.registers for launch in self._launches(ba, config, target, backward=False))

    def register_budget(self, config: mainloop.Tiles) -> int:
        family = bundle(self.launched())
        return resources.facts().thread_registers if family is None else family.registers(1)

    def _held(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> bool:
        """Whether every forward launch's consumer threads hold their registers at its own roles' budget
        (`Family.registers` at `Launch.consumers`: a walk of `Tiles.walkers` roles holds less a role than a
        launch of one) and a multiprocessor holds a program of it (`Launch.resident`)."""
        target = self.launched()
        family = bundle(target, _dtype(ba))
        if family is None:
            return super()._held(ba, config)
        return all(launch.resident > 0
                   and launch.registers <= (launch.family or family).registers(launch.consumers)
                   for launch in self._launches(ba, config, target, backward=False))

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mainloop.Tiles | None:
        """The call's tiles on `target`, from its shape and the device's and its bundle's facts
        (docs/design/kernel-templates.md §7; no candidate space, no ranking: each rule gives way from the
        widest value block to the next narrower where it does not hold), each rule naming the fact it reads:
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

        A compile that refused the joined tiles (`report.spilled`: kernels of the launch spilled) steps the
        tiles the spilling kernels ran at down to their next narrower value block: the training call's where
        they are the fast backward's alone, else both, a forward launch running at the training call's tiles
        and, where they differ, at the forward's (`Tiles.forward`)."""
        refused: dict[bool, set[int]] = {True: set(), False: set()}
        while True:
            trained = self._derive(ba, target, training=True, refused=refused[True])
            if trained is None:
                return None
            alone = self._derive(ba, target, training=False, refused=refused[False])
            family = bundle(target, _dtype(ba))
            # A forward alone pads its values narrower than a training call does (`_widths`): its own tiles.
            narrower = family is not None and (_widths(ba, family, training=False)[1]
                                               < _widths(ba, family)[1])
            if alone is not None and target.multiprocessors is not None and not narrower and (
                    not self._one_wave(ba, alone, target)
                    or self._one_wave(ba, trained, target, training=True)):
                alone = None
            joined = trained if alone in (None, trained) else dataclasses.replace(trained, forward=alone)
            spilled = report.spilled(self, ba, joined)
            if not spilled:
                return joined
            refused[True].add(trained.block_v)
            forward = {launch.name for launch in self._launches(ba, joined, target, backward=False)}
            if alone is not None and any(kernel.startswith(name) for kernel in spilled for name in forward):
                refused[False].add(alone.block_v)

    def _derive(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, *, training: bool,
                refused: Set[int] = frozenset()) -> mainloop.Tiles | None:
        """`derived`'s rules, the fast backward's with them where `training`, past the value blocks a compile
        `refused`."""
        family = bundle(target, _dtype(ba))
        if family is None:
            return None
        blocks = self._blocks(ba, family, training=training)
        # Each rule gives way from the widest block to the next narrower where it does not hold, a block's
        # tiles staged as a rule first reads them: the widest legal, then the widest at which the fast
        # backward holds where it holds at any narrower, then the narrowest still running the states
        # launch in one wave.
        staged = functools.cache(lambda index: None if blocks[index] in refused
                                 else self._staged(ba, blocks[index], target, training=training))
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
                                // (target.multiprocessors * held), training=training)
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
        """The states launch's programs at `tiles`: a value block of each sequence and head, a forward's
        values padded as it pads them (`_widths`)."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        _, columns, _ = _widths(ba, family, training=training)
        arguments = ba.arguments
        batch, length, heads, _ = arguments["value"].shape
        sequences = batch * (1 if arguments.get("starts") is None
                             else arguments.get("max_segments") or length)
        return columns // tiles.block_v * sequences * heads

    def _one_wave(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                  target: generation.GpuTarget, *, training: bool = False) -> bool:
        """Whether the walk's programs at `tiles` run in one wave of `target`'s multiprocessors, as many a
        multiprocessor as it holds (`states_held`)."""
        assert target.multiprocessors is not None, "a wave counts the device's multiprocessors"
        held = target.multiprocessors * self.states_held(ba, tiles, target, training=training)
        return self._programs(ba, tiles, target, training=training) <= held

    def _walked(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget,
                waves: int, *, training: bool = True) -> mainloop.Tiles:
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
        for walkers in range(min(family.consumers, waves), 1, -1):
            if blocks % walkers:
                continue
            grouped = self._staged(ba, tiles.block_v, target, walkers=walkers,
                                   wide=tiles.block_o or tiles.block_v, training=training)
            if grouped is not None and (not training or self._backward_holds(ba, grouped, target)
                                        == self._backward_holds(ba, tiles, target)):
                return grouped
        return tiles

    def trains(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the call's gradient runs on `target`: its fast backward takes its form
        (`mainloop_backward.fast_backward_form`: the delta rule with a decay a head or a key dimension, the
        SSD's rated scan in bf16 under the fast forms, either in fp32 under the fast forms) and its launches
        hold at the call's derived tiles (`_backward_holds`), the rule's gradient (`mosaic_rule`) refusing
        every other call."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launched()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        forms = mainloop_backward.fast_backward_form(arguments.get("delta", True),
                                                     arguments.get("rounded", False), dtype,
                                                     per_key=arguments["g"].ndim == 4)
        if not forms:
            return False
        fits = self.fits(ba, target)
        return bool(fits) and self._backward_holds(ba, fits[0], target)

    def _backward_holds(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                        target: generation.GpuTarget) -> bool:
        """Whether the launches the call's fast backward runs, its own and the forward it takes, hold at
        `tiles` (`holds`; a call taking none, `_launches`', holds its forward's)."""
        both = self._launches(ba, tiles, target, backward=True)
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
        """The forward's walk over each sequence's chunks: a forward without a gradient's scan, forming the
        output in the walk (`mainloop._scan`), or where `training` the states launch of the forward its fast
        backward takes (`mainloop.kept_forward`), a call without one its scan."""
        return next(launch for launch in self._launches(ba, tiles, target, backward=training)
                    if launch.name.startswith(("recurrence_states", "recurrence_scan")))

    def _blocks(self, ba: tokamax.BoundArguments, family: mosaic.Family, *, training: bool = True
                ) -> list[int]:
        """The value blocks a call takes on `family`'s bundle, widest first (`derived`'s first rule before
        legality): those dividing its padded values (a forward's as it pads them, `_widths`) in steps of the
        narrowest a staged operand block takes (`Family.narrowest_columns`, at the 16-bit operands its
        products take)."""
        _, columns, _ = _widths(ba, family, training=training)
        unit = family.narrowest_columns(jnp.bfloat16)
        return [block for block in range(columns - columns % unit, 0, -unit) if columns % block == 0]

    def _staged(self, ba: tokamax.BoundArguments, block: int, target: generation.GpuTarget, *,
                walkers: int = 1, wide: int | None = None, training: bool = False) -> mainloop.Tiles | None:
        """`block`'s tiles at `walkers` walk roles (`Tiles.walkers`) with the widest output block legal beside
        it (`Tiles.block_o`; `wide` where given), at the ring of the states launch's steps, within the
        GPU's block (`generation.shared_budget`) over a step's bytes, whose plan hides the most latency,
        its programs a multiprocessor running at once (`_running`) and its depth one choice
        (`resources.hidden`); None where no output block is legal beside it."""
        family = bundle(target, _dtype(ba))
        assert family is not None
        widths = self._blocks(ba, family, training=training) if wide is None else [wide]
        shallowest = next((tiles for tiles in (mainloop.Tiles(block, 1, None if width == block else width,
                                                              walkers)
                                               for width in widths)
                           if self.legal(ba, tiles, target)), None)
        if shallowest is None:
            return None
        states = self._states(ba, shallowest, target, training=training)
        order = resources.hidden(family, states.consumers)
        deeper = (dataclasses.replace(shallowest, stages=stages)
                  for stages in range(2, min(resources.MOST_STAGES,
                                             generation.shared_budget(target)
                                             // max(states.step_bytes, 1)) + 1))
        rings = [shallowest, *(tiles for tiles in deeper if self.legal(ba, tiles, target))]
        return max(rings, key=lambda tiles: order(self._running(ba, tiles, target, training=training),
                                                  tiles.stages))

    def _running(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget, *,
                 training: bool = False) -> int:
        """The programs of the states launch at `tiles` a multiprocessor of `target` runs at once: those it
        holds (`states_held`), but no more than its programs spread over the device's multiprocessors where
        it knows them, as a launch's blocks go to its multiprocessors in turn. A residency past them hides
        nothing: on an A100 Qwen3-Next's 128 programs 32 wide held three a multiprocessor one deep and two
        two deep, and ran two a multiprocessor either way, the forward 0.749 ms one deep against 0.727 two
        deep (the 27B's 0.943 against 0.916), where the one deep ring narrowed the block to 16 at 0.752
        (1.193, gdn-pins-a100-3b0d2267d)."""
        held = self.states_held(ba, tiles, target, training=training)
        if target.multiprocessors is None:
            return held
        programs = self._programs(ba, tiles, target, training=training) // tiles.walkers
        return min(held, -(-programs // target.multiprocessors))

    def neighbours(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> list[mainloop.Tiles]:
        """The tiles the opt-in tuner times beside the derived `config` (`variants`; the gate holds at each):
        the next value block either side of it at its stage rule (`_staged`), and its block one stage
        shallower (or two deep where it takes one)."""
        target = self.launched()
        family = bundle(target, _dtype(ba))
        assert family is not None
        blocks = self._blocks(ba, family)
        at = blocks.index(config.block_v)
        beside = [self._staged(ba, blocks[index], target) for index in (at - 1, at + 1)
                  if 0 <= index < len(blocks)]
        other = dataclasses.replace(config, stages=config.stages - 1 if config.stages > 1 else 2)
        return [tiles for tiles in (*beside, other) if tiles is not None]

    def staged_bytes(self, ba: tokamax.BoundArguments, config: mainloop.Tiles,
                     target: generation.GpuTarget) -> int:
        # The forward's launches: the fast backward's are held where its rule takes them.
        return max(launch.shared for launch in self._launches(ba, config, target, backward=False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the mainloop runs the call on `target`: 16-bit or fp32 operands, a decay a key dimension's
        in a 16-bit call under a declared floor its sub-chunks factor under (`mainloop.per_key_refusal`), a
        16-bit delta rule's solve at fp32's range where its form proves no bound (`ranged`), a bundle for the
        target's family, the tensor rate its cost model reads, and a config whose launches the device holds
        at the call's widths (`fits`: their shared memory and registers). Off a GPU (a CPU's), where the rule
        runs its bundle interpreted, the target is the interpreted bundle's generation (`launched`)."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launched()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        if dtype not in (jnp.bfloat16, jnp.float16, jnp.float32):
            return False
        taken = arguments["g"].ndim == 3 or mainloop.per_key_refusal(dtype) is None
        if not (taken and bundle(target, dtype) is not None and target.tensor_rate("16bit") is not None):
            return False
        return bool(self.fits(ba, target))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> mainloop.Tiles:
        # Its derived config on the target it launches for (`Template.estimate` and its ranking are gone).
        return self.fitting(ba, self.launched())[0]


MOSAIC = Mosaic()
"""The implementation a call runs (`recurrence_chunk.chunk`'s 'mosaic' mainloop, `GATED_DELTA`'s kernel)."""
