"""Endpoint limits of the Diffusers 0.34.0 update equations, CPU only.

Positive-rate scheduler trajectories are in diffusers_reference.py. Here
alpha=0 source limits use mpmath 1.3.0 at 80 digits with lambda_source=-1e12.
The actual pinned DPM/DEIS update methods run unchanged with scalar exp/log
replaced by mpmath functions. Rates are supplied separately to retain tiny
alpha without computing it as 1-sigma. Their first-order deterministic
update is evaluated as alpha_next*x0 + sigma_next*epsilon: this is the same
DDIM equation, with its removable source cancellation eliminated before
rounding. Subsequent stages run the reference methods. Values at lambda
-5e11 and -1e12 must agree before a fixture is written.

UniPC uses its actual torch step and linear solve at alpha_source=0; its
first predictor is evaluated in the same DDIM form. Target-zero limits use
actual torch scheduler trajectories with sigma_target=1e-12 and 1e-14.
They must agree. KDPM2 fixtures run its actual zero-sigma terminal call.
The nonlinear model and cotangent exercise both coefficients and VJPs.

PYTHONPATH=src python tools/diffusers_limits_reference.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"

import diffusers
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import torch
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.schedulers import (
    scheduling_deis_multistep, scheduling_dpmsolver_multistep, scheduling_dpmsolver_singlestep,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "diffusers"
START = np.array([[1.2, -0.7]], np.float32)
COTANGENT = np.array([[0.7, -1.3]], np.float32)
NOISE = [np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(0), i), START.shape))
         for i in range(3)]
SOURCE_CASES = {
    "multi_pp3": ("multi", dict(solver_order=3, algorithm_type="dpmsolver++")),
    "multi_eps3": ("multi", dict(solver_order=3, algorithm_type="dpmsolver")),
    "multi_sdepp3": ("multi", dict(solver_order=3, algorithm_type="sde-dpmsolver++")),
    "single_pp2m": ("single", dict(solver_order=2, solver_type="midpoint")),
    "single_pp2h": ("single", dict(solver_order=2, solver_type="heun")),
    "single_pp3m": ("single", dict(solver_order=3, solver_type="midpoint")),
    "single_pp3h": ("single", dict(solver_order=3, solver_type="heun")),
    "single_sdepp2h": ("single", dict(solver_order=2, algorithm_type="sde-dpmsolver++", solver_type="heun")),
    "single_sdepp3m": ("single", dict(solver_order=3, algorithm_type="sde-dpmsolver++")),
    "single_sdepp3h_final_zero": ("single", dict(solver_order=3, algorithm_type="sde-dpmsolver++",
                                               solver_type="heun", final_sigmas_type="zero")),
    "deis2": ("deis", dict(solver_order=2)),
    "deis3": ("deis", dict(solver_order=3)),
}
UNIPC_CASES = {
    "x0_bh2": dict(predict_x0=True, solver_type="bh2"),
    "x0_bh1_disabled": dict(predict_x0=True, solver_type="bh1", disable_corrector=[0]),
    "eps_bh2_disabled": dict(predict_x0=False, solver_type="bh2", disable_corrector=[0]),
    "eps_bh1_disabled": dict(predict_x0=False, solver_type="bh1", disable_corrector=[0]),
}
TARGET_CASES = {
    "deis2": (DEISMultistepScheduler, dict(solver_order=2, lower_order_final=False)),
    "deis3": (DEISMultistepScheduler, dict(solver_order=3, lower_order_final=False)),
    "unipc_eps3": (UniPCMultistepScheduler, dict(solver_order=3, predict_x0=False)),
    "unipc_x03": (UniPCMultistepScheduler, dict(solver_order=3, predict_x0=True)),
    "sde_eps1": (DPMSolverMultistepScheduler, dict(solver_order=1, algorithm_type="sde-dpmsolver")),
    "sde_eps2_lowered": (DPMSolverMultistepScheduler, dict(solver_order=2, algorithm_type="sde-dpmsolver")),
}


def scalar_source(kind, config, initial, noise, length):
    classes = {"multi": DPMSolverMultistepScheduler, "single": DPMSolverSinglestepScheduler,
               "deis": DEISMultistepScheduler}
    modules = {"multi": scheduling_dpmsolver_multistep, "single": scheduling_dpmsolver_singlestep,
               "deis": scheduling_deis_multistep}
    extras = {} if kind == "deis" else dict(final_sigmas_type="sigma_min")
    settings = dict(prediction_type="sample", lower_order_final=False, **extras)
    settings.update(config)
    scheduler = classes[kind](**settings)
    # Match the actual order list for the three-interval grid, including the
    # automatic incomplete-group lowering of set_timesteps.
    scheduler.set_timesteps(3)
    orders = scheduler.order_list if kind == "single" else [1, 2, config["solver_order"]]
    alpha = [mp.exp(-length), mp.mpf(".25"), mp.mpf(".5"), mp.mpf(".75")]
    sigma = [mp.mpf(1), mp.mpf(".75"), mp.mpf(".5"), mp.mpf(".25")]
    if config.get("final_sigmas_type") == "zero":
        alpha[-1], sigma[-1] = mp.mpf(1), mp.mpf(0)
    scheduler.sigmas = [0, 1, 2, 3]
    scheduler._sigma_to_alpha_sigma_t = lambda index: (alpha[index], sigma[index])
    algorithm = config.get("algorithm_type", "dpmsolver++")
    outputs, latents = [], []
    x, anchor = initial, initial
    arithmetic = SimpleNamespace(log=mp.log, exp=mp.exp, sqrt=mp.sqrt)
    with patch.object(modules[kind], "torch", arithmetic):
        with patch.object(modules[kind], "np", SimpleNamespace(log=mp.log)):
            for i in range(3):
                scheduler._step_index = i
                velocity = mp.mpf(".7") * mp.tanh(x) + mp.mpf(".15") * sigma[i]
                clean, eps = x - sigma[i] * velocity, x + alpha[i] * velocity
                converted = eps if algorithm == "dpmsolver" or kind == "deis" else clean
                outputs.append(converted)
                if i == 0:
                    if algorithm.startswith("sde"):
                        x = alpha[1] * clean + sigma[1] * noise[0]
                    else:
                        x = alpha[1] * clean + sigma[1] * eps
                elif kind == "single":
                    order = orders[i]
                    if order == 1:
                        anchor = x
                    x = scheduler.singlestep_dpm_solver_update(outputs, sample=anchor, order=order, noise=noise[i])
                elif kind == "multi":
                    if orders[i] == 2:
                        x = scheduler.multistep_dpm_solver_second_order_update(outputs, sample=x, noise=noise[i])
                    else:
                        x = scheduler.multistep_dpm_solver_third_order_update(outputs, sample=x, noise=noise[i])
                elif orders[i] == 2:
                    x = scheduler.multistep_deis_second_order_update(outputs, sample=x)
                else:
                    x = scheduler.multistep_deis_third_order_update(outputs, sample=x)
                latents.append(x)
    return latents


def mp_record(kind, config, length):
    latents, gradients = [], []
    with mp.workdps(80):
        for j in range(START.shape[-1]):
            x = mp.mpf(float(START[0, j]))
            noise = [mp.mpf(float(z[0, j])) for z in NOISE]
            run = lambda initial: scalar_source(kind, config, initial, noise, length)
            latents.append([float(v) for v in run(x)])
            gradients.append(float(mp.diff(lambda initial: run(initial)[-1], x)) * COTANGENT[0, j])
    return np.array(latents, np.float64).T[:, None, :], np.array([gradients], np.float64)


def torch_record(scheduler, grid, *, source=False):
    scheduler.set_timesteps(3)
    scheduler.sigmas = torch.tensor(grid, dtype=torch.float64)
    x = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    initial, latents = x, []
    for i, timestep in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        velocity = 0.7 * x.tanh() + 0.15 * sigma
        clean = x - sigma * velocity
        eps = x + (1 - sigma) * velocity
        noise = torch.tensor(NOISE[i], dtype=torch.float64)
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            stepped = scheduler.step(clean, timestep, x, variance_noise=noise, return_dict=False)[0]
        else:
            stepped = scheduler.step(clean, timestep, x, return_dict=False)[0]
        if source and i == 0:
            following = scheduler.sigmas[i + 1]
            stepped = (1 - following) * clean + following * eps
        x = stepped
        latents.append(x)
    gradient, = torch.autograd.grad((x * torch.tensor(COTANGENT)).sum(), initial)
    return torch.stack(latents).detach().numpy(), gradient.numpy()


def main():
    if diffusers.__version__ != "0.34.0" or mp.__version__ != "1.3.0":
        raise RuntimeError("Requires diffusers==0.34.0 and mpmath==1.3.0")
    arrays = {}
    configs = {}

    def save(name, config, grid, result):
        latents, grad = result
        if not np.isfinite(latents).all() or not np.isfinite(grad).all():
            raise ArithmeticError(name)
        for field, value in dict(x_T=START, cotangent=COTANGENT, latents=latents, grad=grad, grid=grid).items():
            arrays[f"{name}.{field}"] = np.asarray(value, dtype=np.float32)
        configs[name] = config

    for name, (kind, config) in SOURCE_CASES.items():
        first, final = mp_record(kind, config, mp.mpf("5e11")), mp_record(kind, config, mp.mpf("1e12"))
        for a, b in zip(first, final):
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
        last_time = 0.0 if config.get("final_sigmas_type") == "zero" else 0.25
        save("source." + name, config, [1, .75, .5, last_time], final)
    for name, config in UNIPC_CASES.items():
        scheduler = UniPCMultistepScheduler(use_flow_sigmas=True, prediction_type="sample", solver_order=3,
                                           final_sigmas_type="sigma_min", lower_order_final=False, **config)
        save("source.unipc_" + name, config, [1, .75, .5, .25],
             torch_record(scheduler, [1, .75, .5, .25], source=True))
    for name, (cls, config) in TARGET_CASES.items():
        extras = {} if cls is DEISMultistepScheduler else dict(final_sigmas_type="sigma_min")
        results = [torch_record(cls(use_flow_sigmas=True, prediction_type="sample", **extras, **config),
                                [.9, .6, .3, delta]) for delta in (1e-12, 1e-14)]
        for a, b in zip(*results):
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
        save("target." + name, config, [.9, .6, .3, 0], results[-1])
    for ancestral, cls in [(False, KDPM2DiscreteScheduler), (True, KDPM2AncestralDiscreteScheduler)]:
        scheduler = cls()
        scheduler.set_timesteps(4)
        scheduler.set_begin_index(len(scheduler.timesteps) - 1)
        sigma = float(scheduler.sigmas[len(scheduler.timesteps) - 1])
        x = torch.tensor(START, requires_grad=True)
        latent = scheduler.step(0.2 * x.tanh(), scheduler.timesteps[-1], x).prev_sample
        gradient, = torch.autograd.grad((latent * torch.tensor(COTANGENT)).sum(), x)
        save("target.kdpm2_" + str(ancestral).lower(), dict(sigma=sigma, ancestral=ancestral), [1, 0],
             (latent.detach().numpy()[None], gradient.numpy()))
    np.savez(FIXTURES / "limits.npz", **arrays)
    (FIXTURES / "limits.json").write_text(json.dumps(dict(diffusers="0.34.0", mpmath="1.3.0", cases=configs), indent=1) + "\n")
    print(f"{len(configs)} endpoint trajectories and VJPs written")


if __name__ == "__main__":
    main()
