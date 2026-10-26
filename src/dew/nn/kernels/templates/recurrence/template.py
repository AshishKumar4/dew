"""The recurrence template's Mosaic GPU mainloop as a template implementation (`templates.base.Template`), the
delta rule's selector's 'mosaic_gpu' kernel (`recurrence_chunk.GATED_DELTA`).

Its config is the forward's tiles (`mainloop.Tiles`), derived from the call's shape and the device's and
its bundle's facts (`Mosaic.derived`): the widest value block whose carried memory and step tiles a thread's
registers hold (`mosaic.live_registers`) and whose states launch fills the device's multiprocessors, with as
many chunks in flight as every launch's shared memory
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
from dew.nn.kernels.templates import base, cost, mosaic
from dew.nn.kernels.templates.recurrence import mainloop

INTERPRETED_GENERATION = "sm90"
"""The generation whose facts rank a call's configs off a GPU of a family a bundle claims (a CPU): the
interpreted bundle's (`mainloop.INTERPRETED`, wgmma's)."""


def claimed(target: generation.GpuTarget) -> str | None:
    """The instruction family of `target` a bundle of the mainloop claims (`mainloop.FAMILIES`), or None."""
    name = generation.family(target.generation)
    return name if name in mainloop.FAMILIES else None


def _widths(ba: tokamax.BoundArguments, family: mosaic.Family) -> tuple[int, int, jnp.dtype]:
    """The call's keys' and values' widths padded to whole tiles of `family`'s bundle (`mainloop.block_of`),
    and the dtype its tiles hold (the operands', fp32 in an fp32 call)."""
    arguments = ba.arguments
    dtype = jnp.result_type(*(arguments[name].dtype for name in ("query", "key", "value")))
    pad = mainloop.block_of(family)
    return (-(-arguments["key"].shape[-1] // pad) * pad, -(-arguments["value"].shape[-1] // pad) * pad,
            jnp.dtype(dtype))


def bundle(target: generation.GpuTarget) -> mosaic.Family | None:
    """The mainloop's bundle for `target`'s instruction family, None where none claims it."""
    name = claimed(target)
    return None if name is None else mainloop.FAMILIES[name]


@functools.cache
def _planned(shapes: tuple, tiles: mainloop.Tiles, family: str, max_segments: int | None, delta: bool,
             backward: bool, floor: float | None) -> tuple[mainloop.Launch, ...]:
    """`mainloop.planned` of a call over `shapes` (`jax.ShapeDtypeStruct`s or None) by `family`'s bundle."""
    *operands, starts = shapes
    return mainloop.planned(*operands, tiles=tiles, family=mainloop.FAMILIES[family], starts=starts,
                            max_segments=max_segments, delta=delta, backward=backward, floor=floor)


def _launch_cost(launch: mainloop.Launch, target: generation.GpuTarget) -> cost.Cost:
    """The cost model's reading of a planned launch on `target`: its programs, as many on a multiprocessor
    at once as its shared memory and its threads' registers allow (`cost.resident`, each thread at the
    registers a thread has: the mainloop's launches hold 158 to 255, ptxas's count on an A100), each
    streaming its steps' tiles through shared memory with its chunks in flight, issuing its tensor products
    (16-bit: an fp32 call's are its operands' bf16 parts') and its exponentials, and its bytes in global
    memory."""
    return cost.Cost(programs=launch.programs, device_bytes=launch.device_bytes, program_flops=launch.flops,
                     program_staged=launch.steps * launch.step_bytes,
                     program_exponentials=launch.exponentials, steps=launch.steps, stages=launch.stages,
                     resident=cost.resident(target, launch.shared, launch.threads, cost.THREAD_REGISTERS))


@dataclasses.dataclass(frozen=True)
class Mosaic(base.Template[mainloop.Tiles]):
    """The chunked rule on the recurrence template's mainloop (`recurrence_chunk.mosaic_rule`), its config
    the forward's tiles. Its operands are the rule's: queries and keys `[B, T, H, Dk]`, values
    `[B, T, HV, Dv]`, the log decay a head's `[B, T, HV]` or a key dimension's `[B, T, HV, Dk]`, the write
    gate `[B, T, HV]`, each row's initial memory and packed documents' first tokens."""

    name: ClassVar[str] = "mosaic_gpu"
    config_cls: ClassVar[type[mainloop.Tiles]] = mainloop.Tiles

    def _fwd(self, query: jax.Array, key: jax.Array, value: jax.Array, g: jax.Array, beta: jax.Array,
             state: jax.Array | None = None, starts: jax.Array | None = None, *,
             max_segments: int | None = None, scale: float | None = None, delta: bool = True,
             wide: bool = False, floor: float | None = None, config: mainloop.Tiles,
             return_residuals: bool = False):
        from dew.nn.kernels.recurrence_chunk import mosaic_rule

        del return_residuals
        return mosaic_rule(query, key, value, g, beta, state, tiles=config, starts=starts,
                           max_segments=max_segments, scale=scale, delta=delta, wide=wide, floor=floor), None

    @staticmethod
    def launched() -> generation.GpuTarget:
        """The target a call's launches run for: this device's where a bundle claims its family, else the
        interpreted bundle's generation's (`INTERPRETED_GENERATION`)."""
        target = generation.gpu_target()
        return target if bundle(target) is not None else generation.gpu_target(INTERPRETED_GENERATION)

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
        family = claimed(target)
        assert family is not None, f"no bundle of the mainloop claims {target.generation}"
        return _planned(shapes, config, family, arguments.get("max_segments"), arguments.get("delta", True),
                        bool(fast), arguments.get("floor"))

    GENERATES: ClassVar[bool] = True

    def registers(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> int | None:
        """The most registers a consumer thread of any of the forward's launches holds (`Launch.registers`,
        each launch's stated tiles, `mosaic.live_registers`): a legal config's launches spill nothing. The
        fast backward's launches are held to the same budget where its rule takes them
        (`mainloop.backward_holds`), the portable math's gradient where they are not."""
        target = self.launched()
        if bundle(target) is None:
            return None
        return max(launch.registers for launch in self._launches(ba, config, target, backward=False))

    def register_budget(self, config: mainloop.Tiles) -> int:
        family = bundle(self.launched())
        return cost.THREAD_REGISTERS if family is None else family.registers(1)

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mainloop.Tiles | None:
        """The call's tiles on `target`, from its shape and the device's and its bundle's facts
        (docs/design/kernel-templates.md §7), each rule naming the fact it reads:
        - value block: the widest dividing the padded values in steps of the narrowest a staged operand
          block takes (`Family.narrowest_columns`) that is legal (`legal`: every launch's shared memory
          within a block of the device and its registers within a thread's, `registers`) and whose states
          launch, a program a value block, sequence and value head walking its sequence's chunks in turn,
          has as many programs as the device has multiprocessors (fla's chunk_fwd_h runs the 27B gated
          delta net's 192 programs a 32-wide block on an A100's 108, where 64 wide its 96 left twelve
          idle the whole walk, SE-rx-ncu-b106b18d8); the narrowest legal where none fills them; of
          those at which the call's fast backward holds where it holds at any (`_backward_holds`: its
          launches take the states launch's block, an H100's reverse launch 328 registers 64 wide);
        - output block: the widest legal beside it (`_staged`), the output launch's registers apart from the
          states launch's: on an A100 the output launch holds 264 registers 32 wide, where the states
          launch's 384 programs 16 wide took two waves of 593 us (SE-family-a100-0c8018352);
        - stages: the deepest ring of the states launch's steps of which a multiprocessor holds as many
          programs as one stage's (`states_held`, `_staged`), one where none of two or more is legal.
        None where no block is legal: the launch refuses the call."""
        family = bundle(target)
        if family is None:
            return None
        _, columns, _ = _widths(ba, family)
        legal = [tiles for tiles in (self._staged(ba, block, target) for block in self._blocks(ba, family))
                 if tiles is not None]
        if not legal:
            return None
        legal = [tiles for tiles in legal if self._backward_holds(ba, tiles, target)] or legal
        arguments = ba.arguments
        batch, length, heads, _ = arguments["value"].shape
        sequences = batch * (1 if arguments.get("starts") is None
                             else arguments.get("max_segments") or length)
        filling = [tiles for tiles in legal if target.multiprocessors is None
                   or columns // tiles.block_v * sequences * heads >= target.multiprocessors]
        return filling[0] if filling else legal[-1]

    def _backward_holds(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles,
                        target: generation.GpuTarget) -> bool:
        """Whether the call's fast backward's launches, past the forward's, hold at `tiles` (`mainloop.holds`;
        a call taking none, `_launches`', holds)."""
        forward = self._launches(ba, tiles, target, backward=False)
        both = self._launches(ba, tiles, target, backward=True)
        family = bundle(target)
        assert family is not None
        return mainloop.holds(both[len(forward):], family, generation.shared_budget(target))

    def states_held(self, ba: tokamax.BoundArguments, tiles: mainloop.Tiles, target: generation.GpuTarget
                    ) -> int:
        """The programs of the states launch at `tiles` a multiprocessor of `target` holds at once
        (`mainloop.held_programs`, as its persistent launches')."""
        states = self._states(ba, tiles, target)
        family = bundle(target)
        assert family is not None
        return mainloop.held_programs(family, target, states.shared)

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

    def _staged(self, ba: tokamax.BoundArguments, block: int, target: generation.GpuTarget
                ) -> mainloop.Tiles | None:
        """`block`'s tiles with the widest output block legal beside it (`Tiles.block_o`), at the deepest
        ring of the states launch's steps of which a multiprocessor holds as many programs as one stage's
        (`states_held`; `mosaic.deepest`, within `Family.most_shared` over a step's bytes), else one stage;
        None where no output block is legal beside it. The ring deepest alone held an A100's states launch
        to one program a multiprocessor, where its registers held three (SE-family-a100-8c5b5b16a)."""
        family = bundle(target)
        assert family is not None
        shallowest = next((tiles for tiles in (mainloop.Tiles(block, 1, None if wide == block else wide)
                                               for wide in self._blocks(ba, family))
                           if self.legal(ba, tiles, target)), None)
        if shallowest is None:
            return None
        programs = self.states_held(ba, shallowest, target)
        step = self._states(ba, shallowest, target).step_bytes
        deepest = mosaic.deepest(lambda stages: dataclasses.replace(shallowest, stages=stages),
                                 lambda tiles: (self.legal(ba, tiles, target)
                                                and self.states_held(ba, tiles, target) >= programs),
                                 family.most_shared // max(step, 1))
        return deepest or shallowest

    def neighbours(self, ba: tokamax.BoundArguments, config: mainloop.Tiles) -> list[mainloop.Tiles]:
        """The tiles the opt-in tuner times beside the derived `config` (`variants`; the gate holds at each):
        the next value block either side of it at its stage rule (`_staged`), and its block one stage
        shallower (or two deep where it takes one)."""
        target = self.launched()
        family = bundle(target)
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

    def cost(self, ba: tokamax.BoundArguments, config: mainloop.Tiles, target: generation.GpuTarget) -> float:
        """The forward's launches' seconds at `config` on `target`, summed (each `cost.seconds`). No one
        record holds them all, so the template states no `costed` (`templates.features` reads that as the
        whole operation's); its launches' records are `_launches`'."""
        return sum(cost.seconds(_launch_cost(launch, target), target)
                   for launch in self._launches(ba, config, target, backward=False))

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
        factored = arguments["g"].ndim == 3 or mainloop.per_key_refusal(dtype, arguments.get("floor")) is None
        family = bundle(target)
        if not (dtype in (jnp.bfloat16, jnp.float16, jnp.float32) and factored and family is not None
                and target.tensor_rate("16bit") is not None):
            return False
        return bool(self.fits(ba, target))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> mainloop.Tiles:
        return self.estimate(ba, self.launched())[1]


MOSAIC = Mosaic()
"""The implementation a call runs (`recurrence_chunk.chunk`'s 'mosaic' mainloop, `GATED_DELTA`'s kernel)."""
