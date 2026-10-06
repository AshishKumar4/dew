"""Diffusion solvers, each taking one reverse step from t to t_next with the model's prediction at t.

A solver is a value. Whatever it needs between steps is kept in its state,
which `init` builds and `step` carries through `sample`'s scan. The signal and
noise rates come from `process`. A solver that needs another model evaluation
(Heun's corrector, RK4's stages, KDPM2's midpoint) calls `denoise`. A solver
that integrates dx / dsigma = eps refuses a schedule whose alpha is not one.

The solvers named after Diffusers 0.34.0 schedulers reproduce their
arithmetic. Before the compiled scan, `init` checks each algorithm's endpoint
domain on the concrete grid; finite endpoint limits are tested, and endpoints
where the update is undefined raise an error instead of substituting another
update.
"""

from .brownian import DPMSolverSDE
from .common import Solver
from .dpm import DEIS, Algorithm as Algorithm, DPMSolverMultistep, DPMSolverSinglestep
from .gaussian import DDIM, DDPM, PNDM, TCD, Consistency
from .sigma import KDPM2, LMS, RK4, Euler, EulerAncestral, Heun, MultiStepDPM
from .unipc import UniPC

__all__ = ["DDIM", "DDPM", "DEIS", "KDPM2", "LMS", "PNDM", "RK4", "TCD", "Consistency", "DPMSolverMultistep",
           "DPMSolverSDE", "DPMSolverSinglestep", "Euler", "EulerAncestral", "Heun", "MultiStepDPM", "Solver",
           "UniPC"]
