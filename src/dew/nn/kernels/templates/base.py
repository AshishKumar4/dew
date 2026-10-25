"""What every algorithm template shares: a launch whose configs fit the device and whose heuristic is
its cost model's fastest fitting config (`Template`), and one call of a template's implementations
(`Call`), whose eligibility and estimates a selector over them reads (`dew.nn.kernels.selection`).

A template's launch declares the configs it is proven over (`space`), the shared memory a program
stages at each (`staged_bytes`), and the seconds it estimates at each on a target (`cost`,
`dew.nn.kernels.templates.cost`). The tuner measures among the fitting configs only where a run asks
it to (`tuning.tune`); without a measurement the launch takes its heuristic, the same on every run.
"""

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import ClassVar

import tokamax

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import cost


class Template[Config: Hashable](tuning.KernelOp):
    """A template's launch, its configs all of one `Config` type: `space`, `staged_bytes` and `cost`
    give its configs, the ones that fit a block of the device (`fitting`), and its heuristic, the
    fitting config `cost` estimates fastest, ties going to the first in `space`'s order. An
    implementation that does not run a call on a target says so in `eligible`; its operands' token
    counts are `ROWS`, as for any `KernelOp`, and every other argument is keyed as it is. A hook is a
    plain JAX callable the kernel's body traces, and the ops beside the launch fuse into it through
    `jax.experimental.pallas.fuser`: neither is the base's."""

    name: ClassVar[str] = "template"
    """The implementation's name, as a route reports it and `generation.chosen` may name it."""

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Config]:
        """Every config the launch is proven over for `ba`, in a fixed order."""
        raise NotImplementedError(f"{type(self).__name__} declares no space")

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> int:
        """The shared memory, in bytes, a program of the launch of `ba` stages at `config` on `target`, whose
        family sets how a compiler lays it out (a warpgroup MMA's operands, a pipeline's stages)."""
        raise NotImplementedError(f"{type(self).__name__} declares no staged bytes")

    def registers(self, ba: tokamax.BoundArguments, config: Config) -> int | None:
        """The 32-bit registers a thread of the launch's program holds live at `config`, from its tiles'
        sizes; None where the launch states none. Past `register_budget` the program spills."""
        return None

    def register_budget(self, config: Config) -> int:
        """The registers a thread of the launch's program may hold at `config`: a thread's own
        (`cost.THREAD_REGISTERS`) unless the launch's roles share the file otherwise (a warp-specialized
        program's consumers, `mosaic.consumer_registers`)."""
        del config
        return cost.THREAD_REGISTERS

    def costed(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> cost.Cost:
        """The work, bytes and geometry the estimate of the launch of `ba` at `config` on `target` reads
        (`cost.Cost`): the analytic estimate's inputs as they are, for the choice layer's timing log."""
        raise NotImplementedError(f"{type(self).__name__} declares no cost")

    def cost(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> float:
        """The seconds the launch of `ba` is estimated to take at `config` on `target`: its `costed`
        record's (`cost.seconds`)."""
        return cost.seconds(self.costed(ba, config, target), target)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the implementation runs the call `ba` on `target` (its dtypes, its widths, the
        target's facts): every one does unless it says otherwise."""
        return True

    def fits(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs of `space` whose programs fit a block of `target` (`generation.shared_budget`) and
        a thread's registers (`registers`, `register_budget`), in `space`'s order; none where none does,
        which an implementation's eligibility reads."""
        budget = generation.shared_budget(target)
        return tuple(config for config in self.space(ba) if self.staged_bytes(ba, config, target) <= budget
                     and (self.registers(ba, config) or 0) <= self.register_budget(config))

    def fitting(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs that fit (`fits`). A call none fits raises, naming the least any config stages and
        the budget: the template's own requirements refuse the call before it reaches a launch."""
        fits = self.fits(ba, target)
        if not fits:
            # The least any config whose threads hold their registers stages; None where none does.
            held = (config for config in self.space(ba)
                    if (self.registers(ba, config) or 0) <= self.register_budget(config))
            least = min((self.staged_bytes(ba, config, target) for config in held), default=None)
            raise Unfitted(type(self).__name__, least, generation.shared_budget(target), target.generation)
        return fits

    def calibrated(self, config: Config) -> bool:
        """Whether `config` lies in the region its cost model is calibrated over (the extents its sweeps
        measured): a measured search restriction, apart from what is legal (`space`, `fitting`). Every
        config does unless the launch states one."""
        return True

    def searched(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The fitting configs (`fitting`) the estimate ranks and a measurement times: those its cost model
        is calibrated over (`calibrated`), or every fitting one where none is, in `space`'s order."""
        fits = self.fitting(ba, target)
        calibrated = tuple(config for config in fits if self.calibrated(config))
        return calibrated if calibrated else fits

    def pinnable(self, ba: tokamax.BoundArguments) -> set[Config]:
        # Every legal config, calibrated or not: a pin is the caller's measurement.
        return set(self.fitting(ba, generation.gpu_target()))

    def unpinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> str:
        """Why `config` is not among `ba`'s legal configs: outside `space`, holding more registers than a
        thread does, or staging more shared memory than a block of the GPU holds."""
        held = [candidate for candidate in self.space(ba) if candidate == config]
        if not held:
            return "not in its space at this launch's key"
        registers, budget = self.registers(ba, held[0]) or 0, self.register_budget(held[0])
        if registers > budget:
            return f"its threads hold {registers} registers live, past the {budget} a thread has"
        target = generation.gpu_target()
        staged = self.staged_bytes(ba, held[0], target)
        return (f"its programs stage {staged} bytes of shared memory, past the "
                f"{generation.shared_budget(target)} a block of {target.generation} holds")

    def estimate(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[float, Config]:
        """The seconds `cost` estimates at the launch's fastest searched config on `target` (`searched`),
        and that config, the first of `space`'s order where several tie."""
        fits = self.searched(ba, target)
        costs = [self.cost(ba, config, target) for config in fits]
        fastest = min(range(len(fits)), key=lambda index: (costs[index], index))
        return costs[fastest], fits[fastest]

    def _get_autotuning_configs(self, ba: tokamax.BoundArguments) -> set[Config]:
        return set(self.searched(ba, generation.gpu_target()))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        return self.estimate(ba, generation.gpu_target())[1]


def dividing[C](configs: Sequence[C], extents: tuple[int, ...],
                blocks: Callable[[C], tuple[int, ...]]) -> tuple[C, ...]:
    """The configs of `configs` whose blocks (`blocks(config)`) divide the launch's `extents`, place by
    place, in `configs`' order: a launch's space at its padded shapes."""
    return tuple(config for config in configs
                 if all(extent % block == 0 for extent, block in zip(extents, blocks(config), strict=True)))


class Unfitted(ValueError):
    """A call no config of a launch holds: its programs stage at least `least` bytes of shared memory,
    past the `budget` a block of `generation` holds, or (`least` None) no config's threads hold their
    registers (`Template.register_budget`), or the launch declares none."""

    def __init__(self, launch: str, least: int | None, budget: int, generation: str):
        super().__init__(f"{launch}'s programs stage at least {least} bytes of shared memory for this call, "
                         f"past the {budget} a block of {generation} holds" if least is not None else
                         f"no config of {launch} holds this call within a thread's registers or a block of "
                         f"{generation}")
        self.launch, self.least, self.budget, self.generation = launch, least, budget, generation

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.launch, self.least, self.budget, self.generation)


@dataclasses.dataclass(frozen=True)
class Call:
    """One call of a template's implementations on `target`, each bound to the call's own arguments by
    its `name` (every implementation shares the template's signature): what a selector over them reads,
    whether each runs the call (`Template.eligible`) and the seconds its fastest fitting config is
    estimated to take (`Template.estimate`), each asked once."""

    bound: Mapping[str, tuple[Template, tokamax.BoundArguments]]
    target: generation.GpuTarget
    _eligible: dict[str, bool] = dataclasses.field(default_factory=dict, compare=False, repr=False)

    def implementation(self, name: str) -> Template:
        return self.bound[name][0]

    def eligible(self, name: str) -> bool:
        if name not in self._eligible:
            implementation, ba = self.bound[name]
            self._eligible[name] = implementation.eligible(ba, self.target)
        return self._eligible[name]

    def seconds(self, name: str) -> float:
        implementation, ba = self.bound[name]
        return implementation.estimate(ba, self.target)[0]


def call(implementations: Sequence[Template], *args, **kwargs) -> Call:
    """The call of `implementations` with `args` and `kwargs` on this device's target."""
    return Call({implementation.name: (implementation, implementation.bind(*args, **kwargs))
                 for implementation in implementations}, generation.gpu_target())
