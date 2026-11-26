"""What every algorithm template shares: a launch whose config fits the device, derived from its facts
(`Template.derived`) or a fixed space's first that fits, nothing ranked (bible §6), and one call of a
template's implementations (`Call`), whose eligibility a selector over them reads
(`dew.nn.kernels.selection`).

A template's launch declares the config it derives (`derived`) or the fixed set it names (`space`), and
the shared memory a program stages at each (`staged_bytes`). The launch takes its heuristic, the same on
every run.
"""

import collections
import dataclasses
import functools
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import ClassVar, TypeGuard

import tokamax

from dew import _cache
from dew._facts import Held
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import report, resources

DERIVATIONS_HELD = 4096
"""The launches' derived configs a process keeps (`_derived_once`), the least recently read dropped past
them: a model's distinct launches (each call's shapes and dtypes on its target), 182 for a Qwen3.5-4B decode
step's trace (FS-WALKELIG)."""

_DERIVED: collections.OrderedDict[Hashable, Held[Hashable]] = collections.OrderedDict()
"""Each launch's derived config by what its derivation reads (`_derived_once`, held through
`_cache.holding`), the least recently read first."""


def _derived_once(derive: Callable) -> Callable:
    """A launch's `derived`, run once for what it reads: the call, the target and the configs its compiles
    refused (`report.derivation_inputs`), and the run's numerics, its algebra pass and its kernel choices,
    the contexts a derivation's forms and costs read (`dew._cache`); held through `_cache.holding`, the
    latest `DERIVATIONS_HELD`; every eligibility check
    and planner candidate after reads that once (row 4259: the GEMM's tile, derived at every check, took 36
    s of a Qwen3.5-4B step's compile_ahead)."""
    @functools.wraps(derive)
    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget):
        key = (derive, report.derivation_inputs(self, ba, target), _cache.NUMERICS.value,
               _cache.ALGEBRA.value, _cache.KERNEL_CHOICES.value)
        return _cache.holding(_DERIVED, key, lambda: derive(self, ba, target), DERIVATIONS_HELD)

    return derived


class Template[Config: Hashable](tuning.KernelOp):
    """A template's launch, its configs all of one `Config` type. A template that generates its kernel
    (`derived`, docs/design/kernel-templates.md §0) derives its one config from the device's and its
    family's facts and the call's extents; one that does not yet names a fixed set (`space`), whose first
    that fits is its heuristic. `staged_bytes` and
    `registers` say whether a config fits a block of the device (`legal`, `fitting`), which a pin of any
    legal config is checked by. An implementation that does not run a call on a target says so in
    `eligible`. A hook is a plain JAX callable the kernel's body traces (`templates.hooks.Hook`), and the ops
    beside the launch fuse into it through `jax.experimental.pallas.fuser`: neither is the base's."""

    name: ClassVar[str] = "template"
    """The implementation's name, as a route reports it and `generation.chosen` may name it."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each launch's own derivation runs once for what it reads (`_derived_once`).
        if "derived" in vars(cls):
            cls.derived = _derived_once(vars(cls)["derived"])

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
            config = self.derived(ba, self.launch_target())
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
        (`Multiprocessor.thread_registers`) unless the launch's roles share the file otherwise (a
        warp-specialized program's consumers, `resources.consumer_registers`)."""
        del config
        return resources.facts().thread_registers


    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the implementation runs the call `ba` on `target` (its dtypes, its widths, the
        target's facts): every one does unless it says otherwise."""
        return True

    def legal(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether a program of the launch of `ba` at `config` fits a block of `target`
        (`generation.shared_budget`), a compile of this process having found it spill nothing and its
        warpgroup MMAs unserialized (`report.refused`): its registers and its products' issue are the
        assembler's verdict (`tuning`'s certification, a program's compile, `report.settled`), not a
        model's, and a generated kernel spills nothing."""
        return (self.staged_bytes(ba, config, target) <= self.program_shared(config, target)
                and not report.refused(self, ba, config, target))

    def ranks_alone(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        """Whether the launch's derivation takes its ranking's first legal config for `ba` on `target`
        (`takes`): where a receipt measured the ranking's first faster than the config its register model
        proposes (M-FS-15). Else the model proposes (`held`): today's pick, which no receipt has moved. A
        launch says where it moves."""
        del ba, target
        return False

    def takes(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether the launch's derivation takes `config` for `ba` on `target`: a legal config (`legal`) it
        proposes (`proposes`). The model proposes and the assembler decides: a config it proposes that spills
        or whose warpgroup MMAs ptxas serializes is refused by the compile that finds it, and the
        derivation steps past it (`report.refused`)."""
        return self.legal(ba, config, target) and self.proposes(ba, config, target)

    def proposes(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget, *,
                 held: Callable[[], bool] | None = None) -> bool:
        """Whether the derivation offers `config` for `ba` on `target` to its legality: any config where the
        call ranks alone (`ranks_alone`), else one its register model holds on `target` (`held`, or the
        launch's own proposal, `held`: evaluated lazily so a rank-only launch computes no register estimate,
        row 9477; under `target` whatever GPU is in effect, row 12000)."""
        if self.ranks_alone(ba, target):
            return True
        with generation.as_target(target):
            return self.held(ba, config) if held is None else held()

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
        return tuple(config for config in self.space(ba) if self.takes(ba, config, target))

    def pinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> bool:
        # A generating launch's pin may name any legal config of its type; a fixed space's, one of the
        # configs it keeps (`fitting`), whose space and fitting state what its legality alone does not.
        target = self.launch_target()
        if not self._typed(config):
            return False
        if self.GENERATES:
            return self.legal(ba, config, target)
        return config in self.space(ba) and self.legal(ba, config, target)

    def _typed(self, config: Hashable) -> TypeGuard[Config]:
        """Whether `config` is of the launch's config type (`config_cls`)."""
        return isinstance(config, self.config_cls)

    def held(self, ba: tokamax.BoundArguments, config: Config) -> bool:
        """Whether the launch's register model proposes `config` (`takes`): its threads' modeled registers
        (`registers`) within `register_budget`; a launch stating none (a TPU's, the XLA program's) proposes
        any. A proposal, not legality: the assembler's compile decides (`legal`)."""
        return self.within(self.registers(ba, config), config)

    def within(self, registers: int | None, config: Config) -> bool:
        """Whether register estimate `registers` fits `config`'s `register_budget` (None fits any)."""
        return registers is None or registers <= self.register_budget(config)

    def offering(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs the launch offers the call `ba` on `target`, in its order, its refusals read: those
        that fit (`fits`), unless the launch states its own."""
        return self.fits(ba, target)

    def fitting(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> tuple[Config, ...]:
        """The configs the launch offers (`offering`). A call it offers none raises (`unfitted`): the
        template's own requirements refuse the call before it reaches a launch."""
        fits = self.offering(ba, target)
        if not fits:
            raise self.unfitted(ba, target)
        return fits

    def unfitted(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> "Unfitted":
        """Why the launch offers the call `ba` no config on `target`: the least any config whose threads hold
        their registers stages, against the budget; None where none does (a generating launch that
        generates none)."""
        named = () if self.GENERATES else self.space(ba)
        with generation.as_target(target):
            held = [config for config in named if self.held(ba, config)]
        least = min((self.staged_bytes(ba, config, target) for config in held), default=None)
        return Unfitted(type(self).__name__, least, generation.shared_budget(target), target.generation)

    def launch_target(self) -> generation.GpuTarget:
        """The target the launch derives its configs for: the GPU in effect, unless the launch states its
        own."""
        return generation.gpu_target()

    def unpinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> str:
        """Why `config` is no pin for `ba` (`pinnable`): no config of the launch's type, staging more shared
        memory than a block of the GPU holds, refused where a compile of this process assembled it
        (`report.refused`: spilling, or its warpgroup MMAs serialized), or outside what the launch's kernel
        takes (its legality's other rules, or a fixed space)."""
        if not self._typed(config):
            return f"it is no {self.config_cls.__name__}"
        target = self.launch_target()
        if report.refused(self, ba, config, target):
            return ("its kernel spilled or serialized its warpgroup MMAs where a compile of this process "
                    "assembled it")
        staged, shared = self.staged_bytes(ba, config, target), self.program_shared(config, target)
        if staged > shared:
            return (f"its programs stage {staged} bytes of shared memory, past the {shared} a program of it "
                    f"on {target.generation} holds")
        return "its blocks, ring or schedule are not ones the launch's kernel takes for this call"


    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        # The config the launch offers now (`remaining`), the one entry its derivation has; a call it offers
        # none raises `unfitted`'s refusal.
        config = self.remaining(ba)
        if config is None:
            raise self.unfitted(ba, self.launch_target())
        return config

    def remaining(self, ba: tokamax.BoundArguments) -> Config | None:
        # The derived config, or a fixed space's first that fits: nothing ranks configs (bible §6). The first
        # config the launch offers (`offering`, its refusals read), none where it offers none.
        offered = self.offering(ba, self.launch_target())
        return offered[0] if offered else None

    def exhausted(self, ba: tokamax.BoundArguments) -> report.Exhausted:
        # A call no config fits refuses as fitting does (`unfitted`), naming what it stages; one
        # certification refused every config of, as any launch's does.
        if report.refusals_at(self, ba, self.launch_target()):
            return super().exhausted(ba)
        return self.unfitted(ba, self.launch_target())

    def holds(self, ba: tokamax.BoundArguments) -> bool:
        """Whether a config of the launch holds the call `ba`, which its implementation's eligibility reads:
        the call's own taking (`taken`) finds one, a pin, a tuned pick or the derived config certification
        holds where a device certifies launches (`report.certified`), so a call certification refuses every
        config of takes another route before its call, its misses in the report."""
        return self.taken(ba) is not None

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
        target = self.launch_target()
        return tuple(dict.fromkeys((first, *(config for config in self.neighbours(ba, first)
                                             if self.takes(ba, config, target)))))


class Unfitted(report.Exhausted):
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
