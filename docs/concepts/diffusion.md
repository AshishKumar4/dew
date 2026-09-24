# Diffusion processes and solvers

This page explains how Dew describes a diffusion model's noise and how it samples. It assumes you have trained one, for example in [Train a diffusion model](../guides/diffusion.md) or the [tutorials](../tutorials.md).

A diffusion model learns to undo noise. Training takes a clean sample $x_0$, draws a time $t$, and makes a noisy one, $x_t = \alpha_t x_0 + \sigma_t \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. The network sees $x_t$ and $t$ and predicts something from which the clean sample can be recovered. Sampling starts from pure noise and walks the times back to zero.

## The process

A `Process` holds the one convention a model is trained and sampled with:

| Part | What it decides |
|---|---|
| Schedule | $\alpha_t$ and $\sigma_t$ at each time, and how training draws times |
| Prediction transform | What the network outputs: the noise, the clean sample, a velocity, a flow, or the Karras-preconditioned form, and how that converts to $x_0$ and $\epsilon$ |
| Weighting | How the loss weighs each time: the schedule's own weight, a P2-style weight, or Min-SNR |
| Sampling schedule | The time grid inference walks, when it differs from training |

The process gives the rest of the code three things. `process.noise(key, shape)` draws the starting noise at the highest level. `process.times(steps)` is the descending grid a sampler walks. `process.denoiser(model, params, conditions)` wraps a model and its weights into the function a solver calls, which maps a noisy sample and its time to the model's estimates of the clean sample and of the noise.

## Presets

A preset is a frozen dataclass of the numbers that define a published convention, and calling it builds the process: `EDM()` is the configuration and `EDM()()` the `Process`. A run's `run.json` stores the preset's fields, so sampling always rebuilds the convention the model was trained with.

| Preset | Convention |
|---|---|
| `EDM` | Karras et al. (2022): log-normal training noise, the EDM preconditioning and weighting, sampled on the rho-spaced Karras grid |
| `Karras` | The EDM preconditioning, trained on noise levels drawn uniformly along the grid it samples on |
| `Cosine` | The cosine beta table with v-prediction; its default P2 weight makes the loss an unweighted $x_0$ loss |
| `Flow` | Rectified flow on the linear path: velocity prediction, logit-normal times and SD3's resolution shift |
| `Sqrt` | Diffusion-LM (Li et al., 2022): the square-root schedule with the plain $x_0$ loss |
| `MDLM` | Masked diffusion over tokens (Sahoo et al., 2024) on the log-linear schedule, from `dew.diffusion.discrete`; it takes the vocabulary's `mask_id` |

The presets live in `dew.diffusion.presets` and in the `dew.presets` registry. Training and inference can use different schedules: EDM trains on log-normal noise levels and samples on the Karras grid. `MinSNR(gamma)` in `dew.diffusion` replaces a process's weighting with min-SNR-$\gamma$ (Hang et al., 2023).

## Solvers

`sample(denoise, x_T, steps, solver=..., key=...)` runs a solver over `steps` points from the highest time to zero, then returns the model's clean prediction at the last point. The whole walk is one `jax.lax.scan`, so it compiles once, and every step's noise comes from `key` folded with the step index. Changing the solver changes nothing about the trained weights.

The first group of solvers are the classic integrators: `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, and `MultiStepDPM`, a third-order finite-difference integrator in sigma space that keeps the last three noise estimates. The second group follows the Diffusers schedulers and reproduces the Diffusers 0.34.0 trajectories recorded by `tools/diffusers_reference.py`:

- `DPMSolverMultistep` covers every algorithm, order and second-order form of `DPMSolverMultistepScheduler`, and the EDM scheduler's update over the EDM process.
- `DPMSolverSinglestep`, `DPMSolverSDE`, `DEIS`, `UniPC`, `PNDM`, `LMS`, `KDPM2` (plain and ancestral) and `TCD`.
- `Consistency`, used with the `ConsistencyBoundary` prediction transform, for latent consistency models.

`FlowSDE` is Flow-GRPO's Euler-Maruyama solver on a rectified-flow process.

`MultiStepDPM` and `DPMSolverMultistep` are different things despite the names. `DPMSolverSDE` is the solver of `DPMSolverSDEScheduler`, not one of the SDE algorithms of `DPMSolverMultistep`: each interval takes two ancestral steps, and both draw noise from one keyed Brownian bridge over the schedule's positive sigma range, so the two draws are nested increments of a single path.

`DDPM(variance="large")` uses the wider published posterior variance, the beta of the variance-preserving forward step. That beta is zero wherever alpha is one, so DDPM refuses a variance-exploding grid instead of sampling it without noise. Neither variance adds noise on the step whose own time is the schedule's zero.

DPM-Solver++ 2M without any lowering of order at the end is `DPMSolverMultistep(order=2, algorithm="dpmsolver++", solver_type="midpoint", lower_order_final=False, euler_at_final=False)`. By default `lower_order_final=True`, which follows Diffusers: in a walk of fewer than 15 steps, the last step is first order and the one before it at most second order. A solver's `init` takes `(x_T, times, process, key=key)` with a concrete time grid, so an invalid pair of endpoints fails before the compiled loop starts.

Source clipping and dynamic thresholding live in `SourceLimitedPrediction`. They belong to the process's prediction conversion, not to a solver, so a solver that reads the clean prediction twice sees the limited value both times.

## Guidance

`CFG(scale, interval=(0.0, 1.0), rescale=0.0)` is classifier-free guidance. At each step the model predicts once with the condition and once without it, and the guided prediction is `uncond + scale * (cond - uncond)`. Outside `interval`, measured in trajectory progress from 0 at pure noise to 1 at the clean sample, the scale drops to 1 (Kynkäänniemi et al., 2024). `rescale` is Diffusers' `guidance_rescale` (Lin et al., 2023), which pulls the guided output toward the conditional output's standard deviation; `rescale=0` leaves it unchanged.

Guidance is applied to the model's raw outputs, and the process converts the guided output once, as published pipelines order it, so a nonlinear conversion never sees the two branches separately. For one model to answer both questions, train it with some conditions blanked: `DiffusionObjective(unconditional_prob=...)` replaces the condition with its empty value on that fraction of rows, 12% by default.

## Text to image and published pipelines

`TextToImage` in `dew.inference` combines the text encoder, the denoising loop and, for latent models, the decoder. `objective.pipeline(state)` builds one from a trained objective, and `TextToImage.from_run(directory)` rebuilds one from a saved run.

For Stable Diffusion and SDXL checkpoints, `dew.pipeline(source)` rebuilds the source's own scheduler instead of picking a solver by name. It supports the source's clipping, thresholding and timestep spacing, and the Karras, exponential and beta grids where the corresponding scheduler has them. An unsupported combination raises an error instead of falling back to different defaults. The source-scheduler checks use tiny synthetic trajectories, not released-model quality benchmarks. [Supported models](../models.md) lists the pipelines that load.
