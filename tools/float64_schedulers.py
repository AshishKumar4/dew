"""Independent oracles of Diffusers' published schedulers (0.39.0): each class's tables, grid and update
re-derived from its public config and its published formulas, read off the scheduler's own source and never
by widening the library itself (testing.md: a reference generator reads the reference's public interface
alone), at a precision of the caller's (`dtype`): float64, the truth a reference's walk is held to, or
float32, its tables built as the library builds them, the reference arm check 1 holds the unmodified
library's float32 walk to against that truth (tests/test_float64_schedulers.py). Each oracle exposes the
surface a reference's walk drives (`timesteps`, `sigmas`, `init_noise_sigma`, `scale_model_input`,
`step(...).prev_sample`) in torch, so a walk's gradient runs through it.

Noise an ancestral step draws is the walk's own: a step takes it from `draws` where the library would draw
it (`randn_tensor`), so both arms integrate one draw without the library's module patched.

The oracles live by family (tools/float64_tables.py, float64_sigmas.py, float64_multistep.py) over what
they share (tools/float64_oracle.py); this module names each class's (`ORACLES`) and builds a built
scheduler's (`oracle_of`)."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Protocol

import torch
from float64_multistep import DEIS, EDM, DPMMultistep, DPMSinglestep, UniPC
from float64_oracle import Oracle
from float64_sigmas import DPMSDE, KDPM2, LMS, Euler, EulerAncestral, FlowMatch, Heun, KDPM2Ancestral
from float64_tables import DDIM, DDPM, LCM, PNDM, TCD

ORACLES: Mapping[str, type[Oracle]] = {
    "DDIMScheduler": DDIM, "DDPMScheduler": DDPM, "EulerDiscreteScheduler": Euler,
    "EulerAncestralDiscreteScheduler": EulerAncestral, "HeunDiscreteScheduler": Heun,
    "LMSDiscreteScheduler": LMS, "KDPM2DiscreteScheduler": KDPM2,
    "KDPM2AncestralDiscreteScheduler": KDPM2Ancestral, "PNDMScheduler": PNDM,
    "DPMSolverMultistepScheduler": DPMMultistep, "DEISMultistepScheduler": DEIS,
    "UniPCMultistepScheduler": UniPC, "DPMSolverSinglestepScheduler": DPMSinglestep,
    "EDMDPMSolverMultistepScheduler": EDM, "LCMScheduler": LCM, "TCDScheduler": TCD,
    "FlowMatchEulerDiscreteScheduler": FlowMatch, "DPMSolverSDEScheduler": DPMSDE}
"""Each published scheduler class an oracle re-derives, by the library's name."""


class Configured(Protocol):
    """A built library scheduler, as `oracle_of` reads it: its config (`ConfigMixin.config`)."""

    @property
    def config(self) -> Mapping[str, object]: ...


def oracle_of(scheduler: Configured, dtype: torch.dtype = torch.float64) -> Oracle:
    """The oracle of a built library `scheduler` at `dtype`, from its config: every option its class's
    constructor takes (a config converted from another class's keeps that class's options, which this one
    ignores: `ConfigMixin.from_config`), so an option the oracle does not state still refuses it."""
    taken = inspect.signature(type(scheduler).__init__).parameters
    return ORACLES[type(scheduler).__name__](dtype, **{name: value for name, value in scheduler.config.items()
                                                       if name in taken and name != "self"})
