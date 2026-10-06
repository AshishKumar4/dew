"""Published diffusion scheduler files, resolved once and sampled on their native grids.

Class defaults and solver controls live in `source_policy`; `source_grids`
builds the numerical tables without re-reading the config. `SourceSchedule`
keeps the public training and sampling contract and owns its grid cache.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax
import numpy as np

from dew.diffusion.process import Process
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.schedules.flow import FlowMatchingScheduler
from dew.diffusion.schedules.karras import EDMNoiseScheduler
from dew.diffusion.schedules.source_grids import empirical_mu as empirical_mu, published_betas, sampling_grid
from dew.diffusion.schedules.source_policy import Origin as Origin, _Policy, resolve_config
from dew.diffusion.transforms import PredictionTransform
from dew.sampling.solvers import Solver


@dataclass(frozen=True, eq=False)
class SourceSchedule:
    """Reproduces a published scheduler file with Dew's native solver and time grids.

    `from_config` reads the file's class and controls once and resolves them
    into a prediction transform, a native `solver` and the policy that
    builds grids. It raises `ValueError` for an unsupported class or for an
    active control this module does not reconstruct.
    """

    config: Mapping[str, object]
    betas: np.ndarray
    prediction: PredictionTransform
    policy: _Policy
    # The native solver this file's class and controls name, resolved once
    # when the file was read.
    solver: Solver
    # The grids one schedule has already built, keyed by the call that built
    # them. Held here rather than in an lru_cache over the method, whose keys
    # are the schedules themselves: those outlive every pipeline that asks.
    _grids: dict[tuple[int, int | None, Origin], tuple[Process, jax.Array]] = field(
        default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> SourceSchedule:
        betas, prediction, policy, solver = resolve_config(config)
        return cls(MappingProxyType(dict(config)), betas, prediction, policy, solver)

    @property
    def train_steps(self) -> int:
        """The training step count the class declares, which is the beta
        table's length wherever the class tabulates one."""
        return self.policy.train_steps

    def training_process(self, tokens: int | None = None) -> Process:
        """Return the process Dew fine-tunes the checkpoint on, at `tokens` latent tokens.

        A scheduler file states how its checkpoint samples, so the training
        process uses the convention its sampler reads. That is the VP beta
        table, EDM's log-normal sigma draw, or the flow path at the shift
        its sampler uses. A file with dynamic shifting needs `tokens`. The
        terminal stretch applies to sampling only.
        """
        if self.policy.family == "flow":
            flow = self.policy.flow
            shift = 1.0 if flow is None else flow.base(tokens)
            return Process(FlowMatchingScheduler(shift=shift), self.prediction)
        if self.policy.flow_shift is not None:
            return Process(FlowMatchingScheduler(shift=self.policy.flow_shift), self.prediction)
        if self.policy.family == "edm":
            schedule = EDMNoiseScheduler(sigma_min=self.policy.sigma_min or 0.002,
                                         sigma_max=self.policy.sigma_max or 80.0,
                                         sigma_data=self.policy.sigma_data)
            return Process(schedule, self.prediction)
        return Process(DiscreteNoiseScheduler(self.betas, p2_loss_weight_gamma=0), self.prediction)

    def sampling(self, steps: int, *, tokens: int | None = None,
                 origin: Origin = "scheduler") -> tuple[Process, jax.Array]:
        """Return the process and the explicit descending grid for sampling in `steps` steps.

        `tokens` is the latent token count that a resolution-dependent flow
        shift reads, and `origin` is where a flow file's sigmas start. The
        calling pipeline supplies both through the task's grid callable.
        Raises `ValueError` when `steps` is not a positive integer, or when
        it is larger than the training table of a `tabulated` or `lambda`
        class.
        """
        held = self._grids.get((steps, tokens, origin))
        if held is None:
            held = self._grids[(steps, tokens, origin)] = sampling_grid(
                self.policy, self.betas, self.prediction, steps, tokens, origin)
        return held


__all__ = ["SourceSchedule", "published_betas"]
