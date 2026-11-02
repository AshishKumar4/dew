"""The parallel mixer kind: several token mixers reading one normed input side by side, their outputs
summed into the layer's one residual add, as a parallel hybrid's layer runs them (Falcon-H1's Mamba-2
beside its attention, FalconH1DecoderLayer.forward in modeling_falcon_h1.py)."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

from flax import linen as nn

from dew.nn.backbones.decoder_block import ReadsMixing
from dew.nn.flops import counted_flops, parameter_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, PredictionPhase
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import scaled
from dew.nn.protocols import ProjectionGroup, ReadsTrain, RoutesExperts, declared_groups


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
    in the branches' order (`ParallelMixer`). Every branch holds its own parameters and its own decode
    cache under its name."""

    branches: tuple[tuple[str, Callable[..., nn.Module], float, float], ...]
    """Each branch's name, its mixer's factory, and its input and output multipliers."""

    def setup(self):
        self.mixers = tuple(factory(name=name) for name, factory, _, _ in self.branches)

    @property
    def reads(self) -> frozenset[str]:
        """The `MIXING` fields any branch reads (`dew.nn.backbones.decoder_block.ReadsMixing`): the call
        carries each of them to every branch, and a branch reads only its own."""
        return frozenset().union(*(mixer.reads for mixer in self.mixers if isinstance(mixer, ReadsMixing)))

    @property
    def reads_train(self) -> bool:
        """Whether a branch takes the train flag (`dew.nn.protocols.ReadsTrain`)."""
        return any(isinstance(mixer, ReadsTrain) and mixer.reads_train for mixer in self.mixers)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Its branches' work; the scalings and the sum are elementwise."""
        return sum(counted_flops(factory(parent=None), lengths) for _, factory, _, _ in self.branches)

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Its branches' learned maps."""
        return sum(parameter_flops(factory(parent=None), lengths) for _, factory, _, _ in self.branches)

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """The packed groups its branches declare (`dew.nn.protocols.ProjectionSites`)."""
        return declared_groups(*self.mixers)

    def __call__(self, x, call: MixerCall = PLAIN_CALL, *, train: bool = False,
                 prediction_phase: PredictionPhase = 'ordinary'):
        if prediction_phase != 'ordinary':
            raise ValueError(f"a parallel layer runs its branches in the ordinary phase alone, not a "
                             f"prediction depth's {prediction_phase!r}; give the prediction depth a mixer "
                             "of one kind")
        total = None
        for mixer, (_, _, into, out) in zip(self.mixers, self.branches, strict=True):
            reads_train = isinstance(mixer, ReadsTrain) and mixer.reads_train
            output = scaled(mixer(scaled(x, into), call, **({'train': train} if reads_train else {})), out)
            total = output if total is None else total + output
        return total


@dataclasses.dataclass(frozen=True)
class ParallelMixer(MixerBase):
    """Several mixer kinds side by side in one layer (`ParallelMixers`), each `ParallelBranch` its own
    mixer under its name, at the layer's geometry: a parallel hybrid, Falcon-H1's Mamba-2 beside its
    attention. What a branch keeps or refuses, the layer keeps or refuses: its cache, its remat marks,
    its mixed serving step, its generation. A branch that routes experts or computes by its layer's
    index is refused, as the layer replays one routing record and hands its depth to one mixer."""

    branches: tuple[ParallelBranch, ...] = ()

    def __post_init__(self):
        names = [branch.name for branch in self.branches]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError(f"a parallel mixer runs two or more branches of distinct names, got {names}")
        for branch in self.branches:
            if isinstance(branch.mixer, ParallelMixer) or branch.mixer.reads_depth:
                raise ValueError(f"branch {branch.name!r} runs {type(branch.mixer).__name__}, which nests "
                                 "branches or computes by its layer's index; give each branch a mixer of one "
                                 "kind that reads no depth")

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
                           float(branch.output_multiplier)) for branch in self.branches)
        for name, factory, _, _ in factories:
            built = factory(parent=None)
            if isinstance(built, RoutesExperts) and built.routes_experts:
                raise ValueError(f"branch {name!r} routes experts, and a layer replays one routing record "
                                 "for its feed-forward; route in the feed-forward")
        return functools.partial(ParallelMixers, branches=factories)

