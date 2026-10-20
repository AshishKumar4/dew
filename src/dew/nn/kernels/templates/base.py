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


class Template(tuning.KernelOp):
    """A template's launch: `space`, `staged_bytes` and `cost` give its configs, the ones that fit a
    block of the device (`fitting`), and its heuristic, the fitting config `cost` estimates fastest,
    ties going to the first in `space`'s order. An implementation that does not run on a target says
    so in `eligible`; its operands' token counts are `ROWS`, as for any `KernelOp`, and every other
    argument is keyed as it is. A hook is a plain JAX callable the kernel's body traces, and the ops
    beside the launch fuse into it through `jax.experimental.pallas.fuser`: neither is the base's."""

    name: ClassVar[str] = "template"
    """The implementation's name, as a route reports it and `generation.chosen` may name it."""

    def space(self, ba: tokamax.BoundArguments) -> Sequence[Hashable]:
        """Every config the launch is proven over for `ba`, in a fixed order."""
        raise NotImplementedError(f"{type(self).__name__} declares no space")

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Hashable) -> int:
        """The shared memory, in bytes, a program of the launch of `ba` stages at `config`."""
        raise NotImplementedError(f"{type(self).__name__} declares no staged bytes")

    def cost(self, ba: tokamax.BoundArguments, config: Hashable, target: generation.GpuTarget) -> float:
        """The seconds the launch of `ba` is estimated to take at `config` on `target`."""
        raise NotImplementedError(f"{type(self).__name__} declares no cost")

    def eligible(self, target: generation.GpuTarget) -> bool:
        """Whether the implementation runs on `target`: every one does unless it says otherwise."""
        return True

    def fitting(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Hashable, ...]:
        """The configs of `space` whose programs fit a block of `target` (`generation.shared_budget`),
        in `space`'s order. A call none fits raises, naming the least any config stages and the
        budget: the template's own requirements refuse the call before it reaches a launch."""
        budget = generation.shared_budget(target)
        space = tuple(self.space(ba))
        fits = tuple(config for config in space if self.staged_bytes(ba, config) <= budget)
        if not fits:
            least = min((self.staged_bytes(ba, config) for config in space), default=None)
            raise ValueError(f"{type(self).__name__}'s programs stage at least {least} bytes of shared "
                             f"memory for this call, past the {budget} a block of {target.generation} "
                             "holds")
        return fits

    def estimate(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[float, Hashable]:
        """The seconds `cost` estimates at the launch's fastest fitting config on `target`, and that
        config, the first of `space`'s order where several tie."""
        fits = self.fitting(ba, target)
        costs = [self.cost(ba, config, target) for config in fits]
        fastest = min(range(len(fits)), key=lambda index: (costs[index], index))
        return costs[fastest], fits[fastest]

    def _get_autotuning_configs(self, ba: tokamax.BoundArguments) -> set[Hashable]:
        return set(self.fitting(ba, generation.gpu_target()))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Hashable:
        return self.estimate(ba, generation.gpu_target())[1]


def choose(choice: str, implementations: Sequence[Template], *args, **kwargs) -> Template:
    """The implementation of `implementations` one call of `choice` runs, reported as its route: the
    one `generation.chosen` names, else the eligible one whose fastest fitting config `estimate`s
    fewest seconds on this device, the first of the order given where several tie. Every one shares
    the template's signature, so each binds the call's own arguments. A named implementation that
    does not run here or holds no config for the call raises."""
    target = generation.gpu_target()
    by_name = {implementation.name: implementation for implementation in implementations}
    eligible = [implementation for implementation in implementations if implementation.eligible(target)]
    if not eligible:
        raise ValueError(f"no implementation of {choice} runs on {target.generation}: "
                         f"{', '.join(by_name)}")
    estimated = min(eligible, key=lambda implementation: (
        implementation.estimate(implementation.bind(*args, **kwargs), target)[0],
        eligible.index(implementation)))
    named = selection.forced(choice)
    if named is None:
        picked = estimated
    elif named not in by_name or not by_name[named].eligible(target):
        raise ValueError(f"{choice}'s implementation {named!r} is chosen and does not run on "
                         f"{target.generation}; the ones that do are "
                         f"{', '.join(implementation.name for implementation in eligible)}")
    else:
        picked = by_name[named]
        picked.fitting(picked.bind(*args, **kwargs), target)
    generation.route(choice, kernel=picked.name, preferred=estimated.name)
    return picked
