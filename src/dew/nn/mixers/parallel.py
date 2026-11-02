"""The parallel mixer kind: several token mixers reading one normed input side by side, their outputs
summed into the layer's one residual add, as a parallel hybrid's layer runs them (Falcon-H1's Mamba-2
beside its attention, FalconH1DecoderLayer.forward in modeling_falcon_h1.py)."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping

from flax import linen as nn

from dew.nn.backbones.decoder_block import ReadsMixing
from dew.nn.flops import counted_flops, parameter_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, PredictionPhase
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import scaled
from dew.nn.protocols import ProjectionGroup, ReadsTrain, Routes, RoutesExperts, declared_groups


@dataclasses.dataclass(frozen=True)
class ParallelBranch:
    """One mixer of a parallel layer: its module's `name` under the layer's mixer, the mixer kind it
    runs, and the factors its input and its output are scaled by (Falcon-H1's
    `attention_in_multiplier` and `ssm_in_multiplier` on the inputs, `attention_out_multiplier` and
    `ssm_out_multiplier` on the outputs), each a product in at least float32 rounded once to its
    operand's dtype (`dew.nn.precision.scaled`), as torch's product of a tensor and a Python float."""

    name: str
    mixer: MixerBase
    input_multiplier: float = 1.0
    output_multiplier: float = 1.0


class ParallelMixers(nn.Module):
    """The mixers of a parallel layer over one input: each reads the input times its branch's input
    multiplier, under the layer's call, and their outputs, each times its output multiplier, are summed
    in the branches' order (`ParallelMixer`). Every branch holds its own parameters, its own decode cache
    and its own routers under its name. A prediction depth's phase reaches the branches that read it
    (`MixerBase.reads_phase`), the rest running their ordinary call; a routing replay reaches the branches
    that route (`RoutesExperts`), each its own record by its name, or the one routed branch a layer's
    single record."""

    branches: tuple[tuple[str, Callable[..., nn.Module], float, float, bool], ...]
    """Each branch's name, its mixer's factory, its input and output multipliers, and whether it reads a
    prediction phase."""

    def setup(self):
        self.mixers = tuple(factory(name=name) for name, factory, _, _, _ in self.branches)

    @property
    def reads(self) -> frozenset[str]:
        """The `MIXING` fields every branch reads (`dew.nn.backbones.decoder_block.ReadsMixing`; one that
        declares none reads none): the call carries each to every branch, so a field one branch would not
        read is one the block refuses, rather than a restriction that branch would drop."""
        return frozenset.intersection(*(mixer.reads if isinstance(mixer, ReadsMixing) else frozenset()
                                        for mixer in self.mixers))

    @property
    def routes_experts(self) -> bool:
        """Whether a branch routes over experts (`dew.nn.protocols.RoutesExperts`), so the layer's routing
        replay reaches it."""
        return bool(self._routed())

    @nn.nowrap
    def _routed(self) -> frozenset[str]:
        """The branches that route over experts, by name."""
        return frozenset(name for name, factory, *_ in self.branches
                         if isinstance(built := factory(parent=None), RoutesExperts) and built.routes_experts)

    @property
    def reads_train(self) -> bool:
        """Whether a branch takes the train flag (`dew.nn.protocols.ReadsTrain`)."""
        return any(isinstance(mixer, ReadsTrain) and mixer.reads_train for mixer in self.mixers)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Its branches' work; the scalings and the sum are elementwise."""
        return sum(counted_flops(factory(parent=None), lengths) for _, factory, _, _, _ in self.branches)

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Its branches' learned maps."""
        return sum(parameter_flops(factory(parent=None), lengths) for _, factory, _, _, _ in self.branches)

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """The packed groups its branches declare (`dew.nn.protocols.ProjectionSites`)."""
        return declared_groups(*self.mixers)

    def __call__(self, x, call: MixerCall = PLAIN_CALL, *, train: bool = False,
                 prediction_phase: PredictionPhase = 'ordinary',
                 routes: Routes | Mapping[str, Routes] | None = None):
        routed = self._routed()
        if routes is not None and not isinstance(routes, Mapping):
            if len(routed) != 1:
                raise ValueError(f"a layer's one routing record replays one routed site, and branches "
                                 f"{sorted(routed)} route; key the record by branch name")
            routes = {next(iter(routed)): routes}
        if routes is not None and (missing := sorted(routed - set(routes))):
            raise ValueError(f"the routing record names {sorted(routes)}, not the routed branches {missing}")
        total = None
        for mixer, (name, _, into, out, phased) in zip(self.mixers, self.branches, strict=True):
            options = {
                **({'train': train} if isinstance(mixer, ReadsTrain) and mixer.reads_train else {}),
                **({'prediction_phase': prediction_phase} if phased and prediction_phase != 'ordinary'
                   else {}),
                **({'routes': routes.get(name)} if routes is not None and name in routed else {})}
            output = scaled(mixer(scaled(x, into), call, **options), out)
            total = output if total is None else total + output
        return total


@dataclasses.dataclass(frozen=True)
class ParallelMixer(MixerBase):
    """Several mixer kinds side by side in one layer (`ParallelMixers`), each `ParallelBranch` its own
    mixer under its name, at the layer's geometry: a parallel hybrid, Falcon-H1's Mamba-2 beside its
    attention. What a branch keeps, reads or refuses, the layer keeps, reads or refuses: its cache, its
    layer's index (`reads_depth`, which the call hands every branch), its remat marks, its mixed serving
    step, its generation, a prediction depth's phase (`reads_phase`), and its routing replay, by its
    name."""

    branches: tuple[ParallelBranch, ...] = ()

    def __post_init__(self):
        names = [branch.name for branch in self.branches]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError(f"a parallel mixer runs two or more branches of distinct names, got {names}")

    @property
    def reads_phase(self) -> bool:
        return any(branch.mixer.reads_phase for branch in self.branches)

    @property
    def reads_depth(self) -> bool:
        return any(branch.mixer.reads_depth for branch in self.branches)

    @property
    def keeps_triton_gemm(self) -> bool:
        return any(branch.mixer.keeps_triton_gemm for branch in self.branches)

    @property
    def keeps_cache(self) -> bool:
        return any(branch.mixer.keeps_cache for branch in self.branches)

    @property
    def reads_blocks(self) -> bool:
        return any(branch.mixer.reads_blocks for branch in self.branches)

    def marks(self) -> tuple[str, ...] | None:
        """The names any branch marks; every name where one marks every name."""
        marked = [branch.mixer.marks() for branch in self.branches]
        if any(names is None for names in marked):
            return None
        return tuple(dict.fromkeys(name for names in marked if names is not None for name in names))

    def mixed_refusal(self, ctx: MixerContext) -> str | None:
        return next((f"branch {branch.name!r}: {refusal}" for branch in self.branches
                     if (refusal := branch.mixer.mixed_refusal(ctx)) is not None), None)

    def generation_refusal(self) -> str | None:
        return next((refusal for branch in self.branches
                     if (refusal := branch.mixer.generation_refusal()) is not None), None)

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        factories = tuple((branch.name, branch.mixer.build(ctx), float(branch.input_multiplier),
                           float(branch.output_multiplier), branch.mixer.reads_phase)
                          for branch in self.branches)
        return functools.partial(ParallelMixers, branches=factories)

