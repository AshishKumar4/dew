"""The reverse process for diffusion, and decoding for language models.

Each name is imported only when it is used (`dew._exports`), so a module that reads one of the
package's modules imports that module alone."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.sampling.decoding import LogitsTransform, StepState, Stopping
    from dew.sampling.flow import FlowSDE, FlowTrajectory, GaussianTransition
    from dew.sampling.guidance import APG, CFG, Autoguidance, CFGPlusPlus, Guidance
    from dew.sampling.guided import Grammar
    from dew.sampling.sample import sample
    from dew.sampling.solvers import (
        DDIM,
        DDPM,
        DEIS,
        KDPM2,
        LMS,
        PNDM,
        RK4,
        TCD,
        Consistency,
        DPMSolverMultistep,
        DPMSolverSDE,
        DPMSolverSinglestep,
        Euler,
        EulerAncestral,
        Heun,
        MultiStepDPM,
        Solver,
        UniPC,
    )
    from dew.sampling.strategies import Beam, Sample, Speculative, Strategy
    from dew.sampling.text import Generation, Sampling, generate, with_ids_of

_EXPORTS = {
    "LogitsTransform": "dew.sampling.decoding",
    "StepState": "dew.sampling.decoding",
    "Stopping": "dew.sampling.decoding",
    "FlowSDE": "dew.sampling.flow",
    "FlowTrajectory": "dew.sampling.flow",
    "GaussianTransition": "dew.sampling.flow",
    "APG": "dew.sampling.guidance",
    "CFG": "dew.sampling.guidance",
    "Autoguidance": "dew.sampling.guidance",
    "CFGPlusPlus": "dew.sampling.guidance",
    "Guidance": "dew.sampling.guidance",
    "Grammar": "dew.sampling.guided",
    "sample": "dew.sampling.sample",
    "DDIM": "dew.sampling.solvers",
    "DDPM": "dew.sampling.solvers",
    "DEIS": "dew.sampling.solvers",
    "KDPM2": "dew.sampling.solvers",
    "LMS": "dew.sampling.solvers",
    "PNDM": "dew.sampling.solvers",
    "RK4": "dew.sampling.solvers",
    "TCD": "dew.sampling.solvers",
    "Consistency": "dew.sampling.solvers",
    "DPMSolverMultistep": "dew.sampling.solvers",
    "DPMSolverSDE": "dew.sampling.solvers",
    "DPMSolverSinglestep": "dew.sampling.solvers",
    "Euler": "dew.sampling.solvers",
    "EulerAncestral": "dew.sampling.solvers",
    "Heun": "dew.sampling.solvers",
    "MultiStepDPM": "dew.sampling.solvers",
    "Solver": "dew.sampling.solvers",
    "UniPC": "dew.sampling.solvers",
    "Beam": "dew.sampling.strategies",
    "Sample": "dew.sampling.strategies",
    "Speculative": "dew.sampling.strategies",
    "Strategy": "dew.sampling.strategies",
    "Generation": "dew.sampling.text",
    "Sampling": "dew.sampling.text",
    "generate": "dew.sampling.text",
    "with_ids_of": "dew.sampling.text",
}


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)


__all__ = [
    "APG",
    "CFG",
    "DDIM",
    "DDPM",
    "DEIS",
    "KDPM2",
    "LMS",
    "PNDM",
    "RK4",
    "TCD",
    "Autoguidance",
    "Beam",
    "CFGPlusPlus",
    "Consistency",
    "DPMSolverMultistep",
    "DPMSolverSDE",
    "DPMSolverSinglestep",
    "Euler",
    "EulerAncestral",
    "FlowSDE",
    "FlowTrajectory",
    "GaussianTransition",
    "Generation",
    "Grammar",
    "Guidance",
    "Heun",
    "LogitsTransform",
    "MultiStepDPM",
    "Sample",
    "Sampling",
    "Solver",
    "Speculative",
    "StepState",
    "Stopping",
    "Strategy",
    "UniPC",
    "generate",
    "sample",
    "with_ids_of",
]
