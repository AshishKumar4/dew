# Diffusion processes and solvers

A diffusion model learns to undo noise. Training takes a clean sample $x_0$, draws a time $t$ and a Gaussian $\epsilon \sim \mathcal{N}(0, I)$, and forms the noisy sample

$$x_t = \alpha_t x_0 + \sigma_t \epsilon.$$

The network sees $x_t$ and $t$ and predicts a quantity from which $x_0$ and $\epsilon$ can be recovered. Sampling starts from noise at the highest time and walks a grid of times down to zero with a solver. In Dew, a `Process` holds everything about this convention that training and sampling must agree on, a preset builds a published `Process`, and `dew.sampling.sample` runs a solver over it.

![The same image noised at six times under the Cosine and Flow presets, with the alpha and sigma curves of both.](../assets/diffusion-forward-light.svg)
![The same image noised at six times under the Cosine and Flow presets, with the alpha and sigma curves of both.](../assets/diffusion-forward-dark.svg)

## Example

```python
import jax
import jax.numpy as jnp

from dew.diffusion.presets import Cosine, Flow

for preset in (Cosine(), Flow()):
    process = preset()
    schedule = process.schedule
    t = jnp.linspace(0, schedule.T, 5)
    alpha, sigma = schedule.rates(t)
    times = schedule.sample_t(jax.random.key(0), 4)
    print(type(preset).__name__, "T =", schedule.T)
    print("  alpha", " ".join(f"{float(v):.3f}" for v in alpha))
    print("  sigma", " ".join(f"{float(v):.3f}" for v in sigma))
    print("  training times", " ".join(f"{float(v):g}" for v in times))
```

```text
Cosine T = 1000
  alpha 1.000 0.920 0.702 0.378 0.000
  sigma 0.006 0.393 0.713 0.926 1.000
  training times 789 0 712 373
Flow T = 1.0
  alpha 1.000 0.750 0.500 0.250 0.000
  sigma 0.000 0.250 0.500 0.750 1.000
  training times 0.835159 0.883424 0.393268 0.480356
```

`Cosine` is a table over 1000 discrete steps and draws integer training times uniformly. `Flow` is continuous on `[0, 1]`, with $\alpha_t = 1 - t$ and $\sigma_t = t$, and draws training times from a logit-normal distribution.

## Process

A `Process` is a frozen dataclass of four parts:

| Part | What it decides |
|---|---|
| Schedule | $\alpha_t$ and $\sigma_t$ at each time, and how training draws times |
| Prediction transform | What the network outputs: the noise, the clean sample, a velocity, a flow, or the Karras-preconditioned form, and how that converts to $x_0$ and $\epsilon$ |
| Weighting | How the loss weighs each time: the schedule's own weight, a P2-style weight, or Min-SNR |
| Sampling schedule | The time grid inference walks, when it differs from training |

Three methods serve sampling. `process.noise(key, shape)` draws the starting noise at the highest level. `process.times(steps)` is the descending grid a sampler walks. `process.denoiser(model, params, conditions)` wraps a model and its weights into the function a solver calls, which maps a noisy sample and its time to the model's estimates of the clean sample and of the noise.

## Presets

A preset is a frozen dataclass of the numbers that define a published convention. Pass `EDM(regime="pixel")` or `Flow()` directly to `DiffusionObjective` or `DiffusionRunConfig(preset=...)`; they build the process, and the objective's image task keeps that same process. Calling a preset yourself builds a `Process` for direct schedule inspection or low-level sampling. EDM's regime chooses the training noise levels: Karras et al. 2022's for pixels or EDM2's for latents. Explicit `P_mean` and `P_std` values override the regime; supplying both also works without a regime, so old run records retain their training distribution. Otherwise a preset without a regime refuses to build, and a run config fills it from whether the run has an autoencoder. A run's `run.json` stores the preset's fields, so sampling rebuilds the convention the model was trained with.

| Preset | Convention |
|---|---|
| `EDM` | Log-normal training noise (Karras et al. 2022 for pixels, EDM2 for latents), the EDM preconditioning and weighting, sampled on the rho-spaced Karras grid |
| `Karras` | The EDM preconditioning, trained on noise levels drawn uniformly along the grid it samples on |
| `Cosine` | The cosine beta table with v-prediction; its default P2 weight makes the loss an unweighted $x_0$ loss |
| `Flow` | Rectified flow on the linear path: velocity prediction, logit-normal times and SD3's resolution shift |
| `Sqrt` | Diffusion-LM (Li et al., 2022): the square-root schedule with the plain $x_0$ loss |
| `MDLM` | Masked diffusion over tokens (Sahoo et al., 2024) on the log-linear schedule, from `dew.diffusion.discrete`; it takes the vocabulary's `mask_id` |

The presets are in `dew.diffusion.presets` and in the `dew.presets` registry. Training and inference can use different schedules: EDM trains on log-normal noise levels and samples on the Karras grid. `MinSNR(gamma)` in `dew.diffusion` replaces a process's weighting with min-SNR-$\gamma$ (Hang et al., 2023).

## Solvers

`sample(denoise, x_T, steps, solver=..., key=...)` runs a solver from the highest time to zero. `denoise` is `process.denoiser(model, params, conditions)`, which calls the model and converts its output into estimates of $x_0$ and $\epsilon$. This samples from an untrained DiT with two solvers; only the shapes are meaningful:

```python
from dew.nn.backbones.dit import SimpleDiT
from dew.sampling import DDIM, Euler, sample

process = Flow()()
model = SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                  mlp_ratio=2, dtype=jnp.float32, attention_impl="xla")
params = model.init(jax.random.key(0), jnp.zeros((1, 8, 8, 3)), jnp.zeros((1,)))
denoise = process.denoiser(model, params, conditions={})
x_T = process.noise(jax.random.key(1), (2, 8, 8, 3))
print(process.times(5))
for solver in (Euler(), DDIM()):
    x_0 = sample(denoise, x_T, 5, solver=solver, key=jax.random.key(2))
    print(type(solver).__name__, x_0.shape, x_0.dtype)
```

```text
[1.   0.75 0.5  0.25 0.  ]
Euler (2, 8, 8, 3) float32
DDIM (2, 8, 8, 3) float32
```

`sample` walks `steps` points, `process.times(steps)`, and returns the model's clean prediction at the last point. The whole walk is one `jax.lax.scan`, so it compiles once, and every step's noise comes from `key` folded with the step index. Changing the solver changes nothing about the trained weights.

The classic integrators are `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, and `MultiStepDPM`, a third-order finite-difference integrator in sigma space that keeps the last three noise estimates. A second group follows the Diffusers schedulers and reproduces the Diffusers 0.34.0 trajectories recorded by `tools/diffusers_reference.py`:

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

## Text to image

`TextToImage` in `dew.inference` combines the text encoder, the denoising loop and, for latent models, the decoder. `objective.pipeline(state)` builds one from a trained objective, and `TextToImage.from_run(directory)` rebuilds one from a saved run.

For Stable Diffusion and SDXL checkpoints, `dew.pipeline(source)` rebuilds the source's own scheduler instead of picking a solver by name. It supports the source's clipping, thresholding and timestep spacing, and the Karras, exponential and beta grids where the corresponding scheduler has them. An unsupported combination raises an error instead of falling back to different defaults. The scheduler checks compare tiny synthetic trajectories against Diffusers; they are not image-quality benchmarks. [Supported models](../models.md) lists the pipelines that load.
