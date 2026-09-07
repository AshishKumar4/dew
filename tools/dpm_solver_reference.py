"""DPM-Solver++ (2M) reference trajectories from Diffusers, for tests/fixtures/dpm_solver.

Runs `DPMSolverMultistepScheduler` (pinned 0.34.0, `algorithm_type="dpmsolver++"`,
`solver_order=2`, `solver_type="midpoint"`, no noise) over the exact beta table
Dew's `LinearNoiseScheduler(1000)` tabulates, on the integer time grid Dew's
`Process.times(steps)` walks, with `final_sigmas_type="sigma_min"` so the last
sigma is the table's entry at index 0, which is where Dew's grid ends. The
model is the optimal epsilon predictor for x0 ~ N(0, 0.3^2) in closed form,
so both sides are deterministic and nothing is trained.

What lands: the x_T draw, every latent after each solver step, and the
vector-Jacobian product of the final latent against a fixed cotangent through
the whole trajectory, in float64 on the torch side.

    PYTHONPATH=src python tools/dpm_solver_reference.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from dew.diffusion import EpsilonPredictionTransform, LinearNoiseScheduler, Process
from dew.diffusion.schedules.linear import linear_beta_schedule

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dpm_solver"
DATA_STD = 0.3
TRAIN_STEPS = 1000
STEPS = 21
SHAPE = (4, 3, 4, 4)


def dew_grid(steps: int) -> np.ndarray:
    """The integer indices `Process.times(steps)` reaches on the tabulated
    schedule, read from the process itself: jax's float32 linspace lands
    549.99994 where numpy lands 550, and the index truncates."""
    process = Process(LinearNoiseScheduler(TRAIN_STEPS), EpsilonPredictionTransform())
    times = np.asarray(process.times(steps))
    return np.clip(times.astype(np.int32), 0, TRAIN_STEPS - 1)


def oracle_epsilon(x: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E[eps | x_t] for x0 ~ N(0, DATA_STD^2): x sigma / (alpha^2 s^2 + sigma^2)."""
    return x * sigma / (alpha ** 2 * DATA_STD ** 2 + sigma ** 2)


def run(x_T: torch.Tensor):
    betas = linear_beta_schedule(TRAIN_STEPS)
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=TRAIN_STEPS, trained_betas=betas,
        prediction_type="epsilon", algorithm_type="dpmsolver++", solver_order=2,
        solver_type="midpoint", final_sigmas_type="sigma_min",
        lower_order_final=True, thresholding=False)
    grid = dew_grid(STEPS)
    scheduler.set_timesteps(timesteps=grid[:-1].tolist())
    alphas_cumprod = torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), dim=0)
    x = x_T
    latents = []
    for timestep in scheduler.timesteps:
        index = int(timestep)
        alpha = alphas_cumprod[index].sqrt().to(x.dtype)
        sigma = (1 - alphas_cumprod[index]).sqrt().to(x.dtype)
        eps = oracle_epsilon(x, alpha, sigma)
        x = scheduler.step(eps, timestep, x).prev_sample
        latents.append(x)
    return latents


def main() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    x_T = torch.randn(SHAPE, generator=generator, dtype=torch.float64)
    cotangent = torch.randn(SHAPE, generator=generator, dtype=torch.float64)
    x_T.requires_grad_(True)
    latents = run(x_T)
    final = latents[-1]
    (grad,) = torch.autograd.grad((final * cotangent).sum(), x_T)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURES / "linear_vp_2m.npz",
        x_T=x_T.detach().numpy().astype(np.float32),
        latents=np.stack([latent.detach().numpy() for latent in latents]).astype(np.float32),
        cotangent=cotangent.numpy().astype(np.float32),
        grad=grad.numpy().astype(np.float32),
        grid=dew_grid(STEPS))
    (FIXTURES / "linear_vp_2m.json").write_text(json.dumps({
        "diffusers": __import__("diffusers").__version__,
        "scheduler": "DPMSolverMultistepScheduler",
        "algorithm_type": "dpmsolver++", "solver_order": 2, "solver_type": "midpoint",
        "final_sigmas_type": "sigma_min", "prediction_type": "epsilon",
        "train_steps": TRAIN_STEPS, "steps": STEPS, "data_std": DATA_STD,
        "shape": list(SHAPE)}, indent=1) + "\n")
    size = sum(path.stat().st_size for path in FIXTURES.iterdir())
    print(f"{FIXTURES}: {size / 1e3:.0f} kB, {sorted(p.name for p in FIXTURES.iterdir())}")


if __name__ == "__main__":
    main()
