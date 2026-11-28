"""Diffusion solvers, each taking one reverse step from t to t_next with the model's prediction at t.

A solver is a value, and a pytree: its settings are static, so they shape
the compiled walk, and only a number a walk varies by request (DPMSolverSDE's
seed) is a leaf. Whatever it needs between steps is kept in its state,
which `init` builds and `step` carries through `sample`'s scan. The signal and
noise rates come from `process`. A solver that needs another model evaluation
(Heun's corrector, RK4's stages, KDPM2's midpoint) calls `denoise`. A solver
that integrates dx / dsigma = eps refuses a schedule whose alpha is not one.

The solvers named after Diffusers 0.34.0 schedulers take their updates. The
exponential integrators among them (DPM-Solver, DEIS, UniPC), whose weights
depend on the grid alone, solve each step's weights once in float64 in
`init`, from the rates their model reads, so their walks round no log or
exponential of their own, only the weighted sums of the sample and the
model's outputs. Before the compiled scan, `init` checks each algorithm's
endpoint domain on the concrete grid; finite endpoint limits are tested,
and endpoints where the update is undefined raise an error instead of
substituting another update. A multistep update, which divides by the gaps
between its history's levels, refuses a grid with two equal levels before
its last interval.
"""

from .brownian import DPMSolverSDE
from .common import RowKeys, RowSolver, Solver
from .dpm import DEIS, Algorithm as Algorithm, DPMSolverMultistep, DPMSolverSinglestep
from .gaussian import DDIM, DDPM, PNDM, TCD, Consistency
from .sigma import KDPM2, LMS, RK4, Euler, EulerAncestral, Heun, MultiStepDPM
from .unipc import UniPC

__all__ = ["DDIM", "DDPM", "DEIS", "KDPM2", "LMS", "PNDM", "RK4", "TCD", "Consistency", "DPMSolverMultistep",
           "DPMSolverSDE", "DPMSolverSinglestep", "Euler", "EulerAncestral", "Heun", "MultiStepDPM",
           "RowKeys", "RowSolver", "Solver", "UniPC"]
