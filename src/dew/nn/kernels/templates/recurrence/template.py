"""The recurrence template's Mosaic GPU mainloop as a template implementation (`templates.base.Template`), the
delta rule's selector's 'mosaic_gpu' kernel (`recurrence_chunk.GATED_DELTA`).

Its config is the forward's tiles (`mainloop.Tiles`), derived from the call's shape and the device's and
its bundle's facts (`Mosaic.derived`): the widest value block whose carried memory and step tiles a thread's
registers hold (`resources.live_registers`) and whose states launch fills the device's multiprocessors,
with as many chunks in flight as every launch's shared memory
allows a block of the device (`staged_bytes`: each launch's streams and staging tiles as its bundle declares
them, `mainloop.planned`, the fast backward's with the forward's where a call takes it). Nothing is
measured, tabled, swept or ranked: a GPU no row names derives its tiles from its own facts, and a measured
gap is fixed in the derivation or the kernel. The cost model's estimate (`cost`) reports the derived
launches' seconds.
"""

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
import tokamax

from dew.nn.kernels import generation
from dew.nn.kernels.templates import base, bundles, cost, mosaic, resources
from dew.nn.kernels.templates.recurrence import mainloop

INTERPRETED_GENERATION = "sm90"
"""The generation whose facts rank a call's configs off a GPU of a family a bundle claims (a CPU): the
interpreted bundle's (`mainloop.INTERPRETED`, wgmma's)."""


def _widths(ba: tokamax.BoundArguments, family: mosaic.Family) -> tuple[int, int, jnp.dtype]:
    """The call's keys' and values' widths padded to whole tiles of `family`'s bundle (`mainloop.block_of`),
    and the dtype its tiles hold (the operands', fp32 in an fp32 call)."""
    arguments = ba.arguments
    dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
    pad = mainloop.block_of(family)
    return (-(-arguments["key"].shape[-1] // pad) * pad, -(-arguments["value"].shape[-1] // pad) * pad,
            jnp.dtype(dtype))


def _dtype(ba: tokamax.BoundArguments) -> jnp.dtype:
    """The call's operands' common dtype."""
    return jnp.dtype(jnp.result_type(*(ba.arguments[name].dtype for name in ("query", "key", "value"))))


def bundle(target: generation.GpuTarget, dtype=jnp.bfloat16, tiles: mainloop.Tiles | None = None
           ) -> mosaic.Family | None:
    """The mainloop's bundle for `target`'s instruction family and a call of `dtype` (`bundles.of`), the one
    `tiles` name where they name one (`Tiles.layout`), None where none claims it."""
    device = bundles.of(target.generation, mainloop.bundle_dtype(dtype))
    if device is None or tiles is None or tiles.layout is None:
        return device
    return bundles.named(tiles.layout, mainloop.bundle_dtype(dtype))


def forward_layout(target: generation.GpuTarget, dtype) -> str | None:
    """The bundle a forward without a gradient issues by where not the device's: warpgroup MMA's layout
    without a producer (`bundles.layouts`' last), whose consumers hold a program's entry registers and two
    programs a multiprocessor (`ConsumerFed.programs`): its launches walk a chunk at a time on their products'
    latency, one program a multiprocessor beside a producer leaving its tensor cores idle. On an H100 PCIe
    the 27B gated delta net's forward took 0.520 ms on it against 0.597 beside a producer and the fla port's
    0.691, Qwen3-Next's 0.387 against 0.435 and 0.555 (JS-fed3-h100-9d3a38de3). The training call's launches
    keep the producer's layout, its fast backward's tiles derived on it."""
    layouts = bundles.layouts(target.generation, mainloop.bundle_dtype(dtype))
    return layouts[-1].name if len(layouts) > 1 else None


_PLANS: dict = {}
"""`_planned`'s plans by their call, tiles, bundle and target's facts."""


def _planned(shapes: tuple, tiles: mainloop.Tiles, family: mosaic.Family, target: generation.GpuTarget,
             max_segments: int | None, delta: bool, backward: bool,
             floor: float | None) -> tuple[mainloop.Launch, ...]:
    """`mainloop.planned` of a call over `shapes` (`jax.ShapeDtypeStruct`s or None) by `family`'s bundle on
    `target`, cached by the facts a plan reads of it (two targets of one family share no plan: an sm87's and
    an sm80's multiprocessors and shared memory differ)."""
    facts = (target.generation, target.device_kind, target.shared_limit, target.multiprocessors)
    key = (shapes, tiles, family, facts, max_segments, delta, backward, floor)
    if key not in _PLANS:
        *operands, starts = shapes
        _PLANS[key] = mainloop.planned(*operands, tiles=tiles, family=family, starts=starts,
                                       max_segments=max_segments, delta=delta, backward=backward, floor=floor,
                                       target=target)
    return _PLANS[key]


@dataclasses.dataclass(frozen=True)
class Mosaic(base.Template[mainloop.Tiles]):
    """The chunked rule on the recurrence template's mainloop (`recurrence_chunk.mosaic_rule`), its config
    the forward's tiles. Its operands are the rule's: queries and keys `[B, T, H, Dk]`, values
    `[B, T, HV, Dv]`, the log decay a head's `[B, T, HV]` or a key dimension's `[B, T, HV, Dk]`, the write
    gate `[B, T, HV]`, each row's initial memory and packed documents' first tokens, and their rows'
    documents' first slots as the batch built them (`firsts`, `dew.batch.Coordinates.firsts`)."""

    name: ClassVar[str] = "mosaic_gpu"
    config_cls: ClassVar[type[mainloop.Tiles]] = mainloop.Tiles

    def _fwd(self, query: jax.Array, key: jax.Array, value: jax.Array, g: jax.Array, beta: jax.Array,
             state: jax.Array | None = None, starts: jax.Array | None = None, *,
             firsts: jax.Array | None = None, max_segments: int | None = None, scale: float | None = None,
             delta: bool = True, wide: bool = False, floor: float | None = None, config: mainloop.Tiles,
             return_residuals: bool = False):
        from dew.nn.kernels.recurrence_chunk import mosaic_rule

        del return_residuals
        return mosaic_rule(query, key, value, g, beta, state, tiles=config, starts=starts, firsts=firsts,
                           max_segments=max_segments, scale=scale, delta=delta, wide=wide, floor=floor), None

    @staticmethod
    def launched() -> generation.GpuTarget:
        """The target a call's launches run for: this device's where a bundle claims its family, else the
        interpreted bundle's generation's (`INTERPRETED_GENERATION`), its multiprocessors unstated: an
        interpreter runs a launch's programs one after another, no wave of them to fill, so the derivation
        takes the widest tiles that hold (`derived`; packed documents' interpreted gradient took 21 s a draw
        at the one-wave rule's 16-wide value block and 14 at 32, row 1850)."""
        target = generation.gpu_target()
        if bundle(target) is not None:
            return target
        return dataclasses.replace(generation.gpu_target(INTERPRETED_GENERATION), multiprocessors=None)

    def _launches(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, target: generation.GpuTarget, *,
                  backward: bool) -> tuple[mainloop.Launch, ...]:
        """The launches the call `ba` runs at `config` by `target`'s bundle (`mainloop.planned`); with
        `backward`, its fast backward's too where it takes it (a decay a head, 16-bit operands)."""
        arguments = ba.arguments
        names = ("query", "key", "value", "g", "beta", "state", "starts")
        given = [arguments.get(name) for name in names]
        shapes = tuple(None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype) for x in given)
        dtype = jnp.result_type(*(arguments[name].dtype for name in names[:3]))
        per_head = arguments["g"].ndim == 3
        fast = backward and arguments.get("delta", True) and per_head and dtype != jnp.float32
        family = bundle(target, dtype, config)
        assert family is not None, f"no bundle of the mainloop claims {target.generation}"
        return _planned(shapes, config, family, target, arguments.get("max_segments"),
                        arguments.get("delta", True), bool(fast), arguments.get("floor"))

    GENERATES: ClassVar[bool] = True

    def registers(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> int | None:
        """The most registers a consumer thread of any of the forward's launches holds (`Launch.registers`,
        each launch's stated tiles, `resources.live_registers`): a legal config's launches spill nothing. The
        fast backward's launches are held to the same budget where its rule takes them
        (`mainloop.backward_holds`), and a call they do not hold takes no gradient of it (`trains`)."""
        target = self.launched()
        if bundle(target) is None:
            return None
        return max(launch.registers for launch in self._launches(ba, config, target, backward=False))

    def register_budget(self, config: mainloop.Tiles) -> int:
        family = bundle(self.launched())
        return cost.THREAD_REGISTERS if family is None else family.registers(1)

    def _held(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> bool:
        """Whether every forward launch's consumer threads hold their registers at its own roles' budget
        (`Family.registers` at `Launch.consumers`: a walk of `Tiles.walkers` roles holds less a role than a
        launch of one)."""
        target = self.launched()
        family = bundle(target, _dtype(ba), config)
        if family is None:
            return super()._held(ba, config)
        return all(launch.registers <= family.registers(launch.consumers)
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
          wave the fewest programs; the widest where the device's multiprocessors are unknown;
        - output block: the widest legal beside it (`_staged`), the output launch's registers apart from the
          states launch's: on an A100 the output launch holds 264 registers 32 wide, where the states
          launch's 384 programs 16 wide took two waves of 593 us (SE-family-a100-0c8018352);
        - stages: the deepest ring of the states launch's steps of which a multiprocessor holds as many
          programs as one stage's (`states_held`, `_staged`), one where none of two or more is legal.
        - a forward without a gradient: the same rules without the fast backward's (`Tiles.forward`, where
          they differ): on an H100 the 27B gated delta net's forward took 0.513 ms at the training call's
          32-wide block, its states launch's 192 programs two waves, and 0.433 at 64 wide, one wave of 96,
          where fla's port took 0.440 (JS-fwdtiles2-h100-a7ec9a134).
        None where no block is legal: the launch refuses the call."""
        trained = self._derive(ba, target, training=True)
        if trained is None:
            return None
        alone = self._derive(ba, target, training=False, layout=forward_layout(target, _dtype(ba)))
        return trained if alone in (None, trained) else dataclasses.replace(trained, forward=alone)

    def _derive(self, ba: tokamax.BoundArguments, target: generation.GpuTarget, *, training: bool,
                layout: str | None = None) -> mainloop.Tiles | None:
        """`derived`'s rules, the fast backward's with them where `training`, on the bundle `layout` names
        (`Tiles.layout`, the device's where None)."""
        family = bundle(target, _dtype(ba), mainloop.Tiles(0, 0, layout=layout))
        if family is None:
            return None
        _, columns, _ = _widths(ba, family)
        blocks = self._blocks(ba, family)
        # Each rule gives way from the widest block to the next narrower where it does not hold, a block's
        # tiles staged as a rule first reads them: the widest legal, then the widest at which the fast
        # backward holds where it holds at any narrower, then the narrowest still running the states
        # launch in one wave.
        staged = functools.cache(lambda index: self._staged(ba, blocks[index], target, layout=layout))
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
        arguments = ba.arguments
        batch, length, heads, _ = arguments["value"].shape
        sequences = batch * (1 if arguments.get("starts") is None
                             else arguments.get("max_segments") or length)
        multiprocessors = target.multiprocessors

        def one_wave(tiles: mainloop.Tiles) -> bool:
            programs = columns // tiles.block_v * sequences * heads
            return programs <= multiprocessors * self.states_held(ba, tiles, target)

        if not one_wave(chosen):
            programs = columns // chosen.block_v * sequences * heads
            return self._walked(ba, chosen, target,
                                programs // (multiprocessors * self.states_held(ba, chosen, target)),
                                training=training)
        # Narrower only while the fast backward holds where it held at the block given way from.
        held = trains(trained)
        for index in range(trained + 1, len(blocks)):
            tiles = staged(index)
            if tiles is None or not one_wave(tiles) or (held and not trains(index)):
                break
            chosen = tiles
        return chosen

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
        family = bundle(target, _dtype(ba), tiles)
        assert family is not None
        _, columns, _ = _widths(ba, family)
        blocks = columns // tiles.block_v
        for walkers in range(min(family.consumers, waves), 1, -1):
            if blocks % walkers:
                continue
            grouped = self._staged(ba, tiles.block_v, target, walkers=walkers,
                                   wide=tiles.block_o or tiles.block_v, layout=tiles.layout)
            if grouped is not None and (not training or self._backward_holds(ba, grouped, target)
                                        == self._backward_holds(ba, tiles, target)):
                return grouped
        return tiles

    def trains(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the call's gradient runs on `target`: its fast backward takes it (the delta rule, a decay a
        head, 16-bit operands) and its launches hold at the call's derived tiles (`_backward_holds`), the
        rule's gradient (`mosaic_rule`) refusing every other call."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launched()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        sixteen = dtype in (jnp.bfloat16, jnp.float16)
        if not (arguments.get("delta", True) and arguments["g"].ndim == 3 and sixteen):
            return False
        fits = self.fits(ba, target)
        return bool(fits) and self._backward_holds(ba, fits[0], target)

    def _backward_holds(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                        target: generation.GpuTarget) -> bool:
        """Whether the call's fast backward's launches, past the forward's, hold at `tiles` (`mainloop.holds`;
        a call taking none, `_launches`', holds)."""
        forward = self._launches(ba, tiles, target, backward=False)
        both = self._launches(ba, tiles, target, backward=True)
        family = bundle(target, _dtype(ba), tiles)
        assert family is not None
        return mainloop.holds(both[len(forward):], family, generation.shared_budget(target))

    def states_held(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
                    ) -> int:
        """The programs of the states launch at `tiles` a multiprocessor of `target` holds at once
        (`resources.plan`, as its persistent launches')."""
        return self._states(ba, tiles, target).resident

    def _states(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
                ) -> mainloop.Launch:
        return next(launch for launch in self._launches(ba, tiles, target, backward=False)
                    if launch.name.startswith("recurrence_states"))

    def _blocks(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> list[int]:
        """The value blocks a call takes on `family`'s bundle, widest first (`derived`'s first rule before
        legality): those dividing its padded values in steps of the narrowest a staged operand block takes
        (`Family.narrowest_columns`, at the 16-bit operands its products take)."""
        _, columns, _ = _widths(ba, family)
        unit = family.narrowest_columns(jnp.bfloat16)
        return [block for block in range(columns - columns % unit, 0, -unit) if columns % block == 0]

    def _staged(self, ba: tokamax.BoundArguments, block: int, target: generation.GpuTarget, *,
                walkers: int = 1, wide: int | None = None,
                layout: str | None = None) -> mainloop.Tiles | None:
        """`block`'s tiles at `walkers` walk roles (`Tiles.walkers`) with the widest output block legal beside
        it (`Tiles.block_o`; `wide` where given), at the deepest ring of the states launch's steps of which
        a multiprocessor holds as many programs as one stage's (`states_held`; `resources.deepest`, within
        `Family.most_shared` over a step's bytes), else one stage; None where no output block is legal
        beside it. The ring deepest alone held an A100's states launch to one program a multiprocessor,
        where its registers held three (SE-family-a100-8c5b5b16a). The tiles name `layout`'s bundle where
        given (`Tiles.layout`)."""
        family = bundle(target, _dtype(ba), mainloop.Tiles(0, 0, layout=layout))
        assert family is not None
        widths = self._blocks(ba, family) if wide is None else [wide]
        shallowest = next((tiles for tiles in (mainloop.Tiles(block, 1, None if width == block else width,
                                                              walkers, layout=layout)
                                               for width in widths)
                           if self.legal(ba, tiles, target)), None)
        if shallowest is None:
            return None
        programs = self.states_held(ba, shallowest, target)
        step = self._states(ba, shallowest, target).step_bytes
        deepest = resources.deepest(lambda stages: dataclasses.replace(shallowest, stages=stages),
                                 lambda tiles: (self.legal(ba, tiles, target)
                                                and self.states_held(ba, tiles, target) >= programs),
                                 family.most_shared // max(step, 1))
        return deepest or shallowest

    def neighbours(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> list[mainloop.Tiles]:
        """The tiles the opt-in tuner times beside the derived `config` (`variants`; the gate holds at each):
        the next value block either side of it at its stage rule (`_staged`), and its block one stage
        shallower (or two deep where it takes one)."""
        target = self.launched()
        family = bundle(target, _dtype(ba), config)
        assert family is not None
        blocks = self._blocks(ba, family)
        at = blocks.index(config.block_v)
        beside = [self._staged(ba, blocks[index], target, layout=config.layout)
                  for index in (at - 1, at + 1) if 0 <= index < len(blocks)]
        other = dataclasses.replace(config, stages=config.stages - 1 if config.stages > 1 else 2)
        return [tiles for tiles in (*beside, other) if tiles is not None]

    def staged_bytes(self, ba: tokamax.BoundArguments, config: mainloop.Tiles,
                     target: generation.GpuTarget) -> int:
        # The forward's launches: the fast backward's are held where its rule takes them.
        return max(launch.shared for launch in self._launches(ba, config, target, backward=False))

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the mainloop runs the call on `target`: 16-bit or fp32 operands, a decay a key dimension's
        in a 16-bit call under a declared floor its sub-chunks factor under (`mainloop.per_key_refusal`), a
        bundle for the target's
        family, the tensor rate its cost model reads, and a config whose launches the device holds at the
        call's widths (`fits`: their shared memory and registers). Off a GPU (a CPU's), where the rule runs
        its bundle interpreted, the target is the interpreted bundle's generation (`launched`)."""
        if generation.backend() == "cpu" and bundle(target) is None:
            target = self.launched()
        arguments = ba.arguments
        dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
        if dtype not in (jnp.bfloat16, jnp.float16, jnp.float32):
            return False
        factored = arguments["g"].ndim == 3 or mainloop.per_key_refusal(dtype, arguments.get("floor")) is None
        if not (factored and bundle(target, dtype) is not None and target.tensor_rate("16bit") is not None):
            return False
        return bool(self.fits(ba, target))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> mainloop.Tiles:
        # Its derived config on the target it launches for (`Template.estimate` and its ranking are gone).
        return self.fitting(ba, self.launched())[0]


MOSAIC = Mosaic()
"""The implementation a call runs (`recurrence_chunk.chunk`'s 'mosaic' mainloop, `GATED_DELTA`'s kernel)."""
