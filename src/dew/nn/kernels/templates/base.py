"""What every algorithm template shares: a launch whose configs fit the device and whose heuristic is
its cost model's fastest fitting config (`Template`), and one call of a template's implementations
(`Call`), whose eligibility a selector over them reads (`dew.nn.kernels.selection`).

A template's launch declares the configs it is proven over (`space`), the shared memory a program
stages at each (`staged_bytes`), and the seconds it estimates at each on a target (`cost`,
`dew.nn.kernels.templates.cost`). The launch takes its heuristic, the same on every run.
"""

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import ClassVar, TypeGuard

import tokamax

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import cost


class Template[Config: Hashable](tuning.KernelOp):
    """A template's launch, its configs all of one `Config` type. A template that generates its kernel
    (`derived`, docs/design/kernel-templates.md §0) derives its one config from the device's and its
    family's facts and the call's extents; one that does not yet names a fixed set (`space`) the cost
    model picks its heuristic from, ties going to the first in `space`'s order. `staged_bytes` and
    `registers` say whether a config fits a block of the device (`legal`, `fitting`), which a pin of any
    legal config is checked by. An implementation that does not run a call on a target says so in
    `eligible`. A hook is a plain JAX callable the kernel's body traces, and the ops beside the launch
    fuse into it through `jax.experimental.pallas.fuser`: neither is the base's."""

    name: ClassVar[str] = "template"
    """The implementation's name, as a route reports it and `generation.chosen` may name it."""

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Config | None:
        """The one config the launch generates for `ba` on `target`, from the device's and its family's
        facts and the call's extents (docs/design/kernel-templates.md §7), None where it generates none
        for the call; a launch that does not generate (`GENERATES`) names a fixed `space` instead."""
        raise NotImplementedError(f"{type(self).__name__} generates no config")

    GENERATES: ClassVar[bool] = False
    """Whether the launch generates its config (`derived`); else it names a fixed `space`."""

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Config]:
        """Every config the launch takes for `ba`: its generated one on this device (`derived`), or the
        fixed set it names, in a fixed order."""
        if self.GENERATES:
            config = self.derived(ba, generation.gpu_target())
            return () if config is None else (config,)
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
        (`cost.Cost`)."""
        raise NotImplementedError(f"{type(self).__name__} declares no cost")

    def cost(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> float:
        """The seconds the launch of `ba` is estimated to take at `config` on `target`: its `costed`
        record's (`cost.seconds`)."""
        return cost.seconds(self.costed(ba, config, target), target)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the implementation runs the call `ba` on `target` (its dtypes, its widths, the
        target's facts): every one does unless it says otherwise."""
        return True

    def legal(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether a program of the launch of `ba` at `config` fits a block of `target`
        (`generation.shared_budget`) and its threads hold their registers (`registers`,
        `register_budget`): a generated kernel spills nothing."""
        return self.staged_bytes(ba, config, target) <= self.program_shared(config, target) and self._held(
            ba, config)

    def program_shared(self, config: Config, target: generation.GpuTarget) -> int:
        """The shared memory a program of the launch at `config` may stage on `target`: a block's
        (`generation.shared_budget`), where one program runs on a multiprocessor at a time."""
        del config
        return generation.shared_budget(target)

    def fits(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs the launch takes on `target` that are legal there (`legal`): its generated one
        (`derived`), or those of its `space`, in `space`'s order; none where none is, which an
        implementation's eligibility reads."""
        if self.GENERATES:
            config = self.derived(ba, target)
            return () if config is None or not self.legal(ba, config, target) else (config,)
        return tuple(config for config in self.space(ba) if self.legal(ba, config, target))

    def pinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> bool:
        # A generating launch's pin may name any legal config of its type; a fixed space's, one of the
        # configs it keeps (`fitting`), whose space and fitting state what its legality alone does not.
        target = generation.gpu_target()
        if not self._typed(config):
            return False
        if self.GENERATES:
            return self.legal(ba, config, target)
        try:
            return config in self.fitting(ba, target)
        except Unfitted:
            return False

    def _typed(self, config: Hashable) -> TypeGuard[Config]:
        """Whether `config` is of the launch's config type (`config_cls`)."""
        return isinstance(config, self.config_cls)

    def _held(self, ba: tokamax.BoundArguments, config: Config) -> bool:
        """Whether `config`'s threads hold their registers (`registers` within `register_budget`)."""
        return (self.registers(ba, config) or 0) <= self.register_budget(config)

    def fitting(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs that fit (`fits`). A call none fits raises, naming the least any config stages and
        the budget: the template's own requirements refuse the call before it reaches a launch."""
        fits = self.fits(ba, target)
        if not fits:
            # The least any config whose threads hold their registers stages; None where none does (a
            # generating launch that generates none).
            named = () if self.GENERATES else self.space(ba)
            held = [config for config in named if self._held(ba, config)]
            least = min((self.staged_bytes(ba, config, target) for config in held), default=None)
            raise Unfitted(type(self).__name__, least, generation.shared_budget(target), target.generation)
        return fits

    def unpinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> str:
        """Why `config` is no pin for `ba` (`pinnable`): no config of the launch's type, holding more
        registers than a thread does, staging more shared memory than a block of the GPU holds, or
        outside what the launch's kernel takes (its legality's other rules, or a fixed space)."""
        if not self._typed(config):
            return f"it is no {self.config_cls.__name__}"
        registers, budget = self.registers(ba, config) or 0, self.register_budget(config)
        if not self._held(ba, config):
            return f"its threads hold {registers} registers live, past the {budget} a thread has"
        target = generation.gpu_target()
        staged, shared = self.staged_bytes(ba, config, target), self.program_shared(config, target)
        if staged > shared:
            return (f"its programs stage {staged} bytes of shared memory, past the {shared} a program of it "
                    f"on {target.generation} holds")
        return "its blocks, ring or schedule are not ones the launch's kernel takes for this call"

    def estimate(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[float, Config]:
        """The seconds `cost` estimates for the launch on `target` and its config: its generated one
        (`derived`), or the fitting config of its `space` `cost` estimates fastest, the first of
        `space`'s order where several tie."""
        fits = self.fitting(ba, target)
        costs = [self.cost(ba, config, target) for config in fits]
        fastest = min(range(len(fits)), key=lambda index: (costs[index], index))
        return costs[fastest], fits[fastest]

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        return self.estimate(ba, generation.gpu_target())[1]

    def neighbours(self, ba: tokamax.BoundArguments, config: Config) -> Sequence[Config]:
        """The configs next to a generating launch's derived `config` an opt-in tuning run may time beside
        it (`variants`): none unless the launch states them; the legal ones are kept."""
        del ba, config
        return ()

    def variants(self, ba: tokamax.BoundArguments) -> tuple[Config, ...]:
        # A generating launch's derived config, then its legal neighbours; a fixed space's (the frozen
        # Triton launches') heuristic alone, which no tuning extends.
        first = ba.heuristics_config
        if not self.GENERATES:
            return (first,)
        target = generation.gpu_target()
        return tuple(dict.fromkeys((first, *(config for config in self.neighbours(ba, first)
                                             if self.legal(ba, config, target)))))


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
    whether each runs the call (`Template.eligible`), asked once."""

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


def call(implementations: Sequence[Template], *args, **kwargs) -> Call:
    """The call of `implementations` with `args` and `kwargs` on this device's target."""
    return Call({implementation.name: (implementation, implementation.bind(*args, **kwargs))
                 for implementation in implementations}, generation.gpu_target())
