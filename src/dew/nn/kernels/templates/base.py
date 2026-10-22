"""What every algorithm template shares: a launch whose configs fit the device and whose heuristic is
its cost model's fastest fitting config (`Template`), and one call's choice among a template's
implementations by the same estimates (`choose`).

A template's launch declares the configs it is proven over (`space`), the shared memory a program
stages at each (`staged_bytes`), and the seconds it estimates at each on a target (`cost`,
`dew.nn.kernels.templates.cost`). The tuner measures among the fitting configs only where a run asks
it to (`tuning.tune`); without a measurement the launch takes its heuristic, the same on every run.
"""

from collections.abc import Hashable, Sequence
from typing import ClassVar

import tokamax

from dew.nn.kernels import generation, selection, tuning
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

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Config) -> int:
        """The shared memory, in bytes, a program of the launch of `ba` stages at `config`."""
        raise NotImplementedError(f"{type(self).__name__} declares no staged bytes")

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

    def fitting(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs of `space` whose programs fit a block of `target` (`generation.shared_budget`),
        in `space`'s order. A call none fits raises, naming the least any config stages and the
        budget: the template's own requirements refuse the call before it reaches a launch."""
        budget = generation.shared_budget(target)
        space = tuple(self.space(ba))
        fits = tuple(config for config in space if self.staged_bytes(ba, config) <= budget)
        if not fits:
            raise Unfitted(type(self).__name__, min((self.staged_bytes(ba, config) for config in space),
                                                     default=None), budget, target.generation)
        return fits

    def estimate(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[float, Config]:
        """The seconds `cost` estimates at the launch's fastest fitting config on `target`, and that
        config, the first of `space`'s order where several tie."""
        fits = self.fitting(ba, target)
        costs = [self.cost(ba, config, target) for config in fits]
        fastest = min(range(len(fits)), key=lambda index: (costs[index], index))
        return costs[fastest], fits[fastest]

    def _get_autotuning_configs(self, ba: tokamax.BoundArguments) -> set[Config]:
        return set(self.fitting(ba, generation.gpu_target()))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        return self.estimate(ba, generation.gpu_target())[1]


class Unfitted(ValueError):
    """A call no config of a launch holds: its programs stage at least `least` bytes of shared memory
    (None where the launch declares no config), past the `budget` a block of `generation` holds."""

    def __init__(self, launch: str, least: int | None, budget: int, generation: str):
        super().__init__(f"{launch}'s programs stage at least {least} bytes of shared memory for this "
                         f"call, past the {budget} a block of {generation} holds")
        self.launch, self.least, self.budget, self.generation = launch, least, budget, generation

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.launch, self.least, self.budget, self.generation)


def choose(choice: str, implementations: Sequence[Template], *args, **kwargs) -> Template:
    """The implementation of `implementations` one call of `choice` runs, reported as its route: the
    one `generation.chosen` names, else the one whose fastest fitting config `estimate`s fewest
    seconds among those `eligible` for the call on this device, the first of the order given where
    several tie. The choice is the cost model's: a measured pick chooses a launch's config, never its
    implementation. Every implementation shares the template's signature, so each binds the call's
    own arguments. A named implementation that does not run the call here raises."""
    target = generation.gpu_target()
    bound = {implementation.name: (implementation, implementation.bind(*args, **kwargs))
             for implementation in implementations}
    eligible = [(implementation, ba) for implementation, ba in bound.values()
                if implementation.eligible(ba, target)]
    if not eligible:
        raise ValueError(f"no implementation of {choice} runs this call on {target.generation}: "
                         f"{', '.join(bound)}")
    named = selection.forced(choice)
    if named is not None:
        # A chosen implementation is the call's without estimating the others, whose costs may need
        # facts the target does not state.
        if named not in bound or not bound[named][0].eligible(bound[named][1], target):
            raise ValueError(f"{choice}'s implementation {named!r} is chosen and does not run this call on "
                             f"{target.generation}; the ones that do are "
                             f"{', '.join(implementation.name for implementation, _ in eligible)}")
        picked, ba = bound[named]
        picked.fitting(ba, target)
        generation.route(choice, kernel=picked.name)
        return picked
    # One eligible implementation is the call's without an estimate, so a target with no cost facts
    # (a TPU, where XLA's alone runs) needs none.
    costs = [0.0] if len(eligible) == 1 else [implementation.estimate(ba, target)[0]
                                              for implementation, ba in eligible]
    picked = eligible[min(range(len(eligible)), key=lambda index: (costs[index], index))][0]
    generation.route(choice, kernel=picked.name, preferred=picked.name)
    return picked
