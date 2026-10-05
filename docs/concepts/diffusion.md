# Diffusion processes and solvers

A diffusion model learns to undo noise. Training takes a clean sample $x_0$, draws a time $t$ and a Gaussian $\epsilon \sim \mathcal{N}(0, I)$, and forms the noisy sample

$$x_t = \alpha_t x_0 + \sigma_t \epsilon.$$

The network takes $x_t$ and $t$ as inputs. It predicts a quantity that you can convert to $x_0$ and $\epsilon$. To sample, a solver starts from noise at the highest time and steps through a time grid down to zero.

In Dew, a `Process` defines the choices that training and sampling must share. A preset builds a `Process` from a published method, and `dew.sampling.sample` runs a solver with it.

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

For sampling, `process.noise(key, shape)` draws the starting noise at the highest level. `process.times(steps)` returns the descending time grid. `process.rates(t, like=x)` returns `(alpha, sigma)` at `t`, shaped to broadcast against `x`.

`process.denoiser(model, params, conditions)` builds the function a solver calls. Given a noisy sample and its time, it uses the model and weights to estimate the clean sample and the noise.

## Presets

A preset is a frozen dataclass with the numbers that define a published method. Pass `EDM(regime="pixel")` or `Flow()` directly to `DiffusionObjective` or `DiffusionRunConfig(preset=...)`. They build the process, which the objective's image task also uses. Call a preset yourself to build a `Process` for inspecting schedules or sampling at the lower level.

EDM's regime selects the training noise levels: Karras et al. 2022's for pixels or EDM2's for latents. Explicit `P_mean` and `P_std` values override the regime. If you supply both, you can omit the regime; old run records therefore keep their training distribution. Otherwise the EDM preset needs a regime before it can build. A run config selects it based on whether the run has an autoencoder. `run.json` stores the preset's fields so sampling can rebuild the process used in training.

| Preset | Convention |
|---|---|
| `EDM` | Log-normal training noise (Karras et al. 2022 for pixels, EDM2 for latents), the EDM preconditioning and weighting, sampled on the rho-spaced Karras grid |
| `Karras` | The EDM preconditioning, trained on noise levels drawn uniformly along the grid it samples on |
| `Cosine` | The cosine beta table with v-prediction; its default P2 weight makes the loss an unweighted $x_0$ loss |
| `Flow` | Rectified flow on the linear path with velocity prediction. Training times follow one of SD3's densities (Esser et al., 2024): logit-normal (the default), the heavy-tailed `mode` density, `cosmap` or uniform. The time is the noise level, as in the paper; Diffusers' SD3 and Flux training scripts index their timesteps in descending order, so their `--logit_mean` is the negative of `logit_mean` here. `shift` is SD3's static resolution shift; `resolution_shift` sets it from the image size instead, as Flux's exp(mu) of a token count. A run config fills that count from its data on a 16-pixel grid, Flux's own; a model that tokenizes the image otherwise sets `tokens` to its count |
| `JiT` | JiT (Li & He, 2025): rectified flow on the linear path where the model predicts the clean sample and the loss scores it in velocity space (`VelocityLoss`), with sigma floored at `t_eps`. Training times are the reference's logit-normal. `simple_dit`'s `patch_bottleneck` is JiT's bottleneck patch embedding for large pixel patches |
| `MeanFlow` | MeanFlow (Geng et al., 2025): rectified flow whose model predicts the average velocity over an interval, reading the interval's `duration` beside the time (`Process.interval`, `simple_dit(interval=True)`). It trains under `MeanFlowObjective` (`mean_flow` on the run config), and one Euler step over the whole grid samples it |
| `Sqrt` | Diffusion-LM (Li et al., 2022): the square-root schedule with the plain $x_0$ loss |
| `MDLM` | Masked diffusion over tokens (Sahoo et al., 2024) on the log-linear schedule, from `dew.diffusion.discrete`; it takes the vocabulary's `mask_id` |

The presets are classes in `dew.diffusion.presets`. Training and inference can use different schedules. For example, EDM trains on log-normal noise levels and samples on the Karras grid.

`MinSNR(gamma)` in `dew.diffusion` replaces a process's weighting with min-SNR-$\gamma$ (Hang et al., 2023). It weights the $x_0$ loss by $\min(\mathrm{SNR}, \gamma)$. For an $\epsilon$ prediction, it divides that weight by SNR; for a $v$ prediction, by SNR + 1. EDM preconditioning computes its loss on $x_0$, so it uses the cap without division.

EDM2 (Karras et al., 2024) uses the `EDM` preset's latent regime. Its `edm2_unet` backbone uses the magnitude-preserving layers in `dew.nn.mp`. It matches NVlabs' `UNet` (`tools/edm2_reference.py`), with a text condition in place of the class label.

`--optim.forced-weight-normalization` renormalizes those layers' weights after every update, as the paper specifies. Set `uncertainty=128` on the run config to learn the paper's loss weighting. This trains a head u(sigma) beside the model with the loss w / e^u ||D - y||^2 + u. The published task omits that head.

## Training aids

To add representation alignment, set `alignment=RepresentationAlignment(...)` on `DiffusionRunConfig`. It builds `DiffusionObjective(alignment=Alignment(...))`.

REPA (Yu et al., 2025) uses an MLP to project the model's hidden tokens after one layer. The loss compares them with a frozen DINOv2 encoder's patch features of the clean image. It uses negative cosine similarity, weighted by `weight` (REPA's `proj_coeff`). Dew halves the L2 denoising error and the alignment loss together.

iREPA (Singh et al., 2026) uses `projector="conv"`, a single 3x3 convolution over the token grid. `spatial_norm=gamma` subtracts gamma times the encoder features' spatial mean, then z-scores them over space. The training script uses gamma 0.6.

`tools/repa_reference.py` compares both losses with the official code. REPA's complete training loss, as `train.py` composes `SILoss`, and its gradient in every weight are twice Dew's. Encoder preprocessing matches REPA and iREPA: divide by 255, apply ImageNet normalization, then resize bicubically to 224 for a 256-pixel image with `resolution=224`. The projector trains with the model, while the encoder's weights stay frozen. The published task omits the projector and encoder.

The aligned tokens must match the encoder's patch grid in raster order. In REPA's 256-pixel setup, a latent DiT with patch size 2 has 16x16 tokens. The default `encoder="facebook/dinov2-base"` uses `resolution=224` to produce 16x16 patches of 14 pixels. `layer="dit_block_7"` aligns after the eighth block of `simple_dit`. In Python, load this encoder with `dew.nn.autoencoders.rae.load_dinov2` and use `module.clone(input_size=224)` with its parameters.

`RepresentationAlignment(end_to_end=EndToEnd())` adds REPA-E (Leng et al., 2025), which tunes the run's KL autoencoder with the model. The autoencoder loss includes L1 reconstruction, KL and 1.5 times the alignment loss on its latent through the frozen model. The model trains on the detached latent after affine-free batch normalization.

The batch norm's running statistics replace the autoencoder's fixed latent scale. A saved run decodes with the tuned autoencoder and these statistics. As in REPA-E's `extract_latents_stats`, it scales by the running variance's reciprocal square root without the batch norm's epsilon.

Dew omits REPA-E's LPIPS and PatchGAN terms. `tools/repae_reference.py` compares a complete step with the published `train_repae.py` loop, with those two terms at weight 0. The regularizer, batch norm, autoencoder gradient and running statistics match. The model's gradient is half the reference's because Dew halves its L2 loss.

`simple_dit`'s `routes` implements TREAD's token routing (Krause et al., 2025) during training only. Each `(ratio, start, end)` makes a random `ratio` of the tokens skip blocks `start` to `end`. These tokens keep their values from before `start`, then rejoin the others after `end`. The skipped blocks therefore process fewer tokens. The gather and scatter match CompVis/tread's `Router` (`tools/tread_reference.py`). Sampling runs every token through every block.

## Few-step generators

`MeanFlowObjective` trains the average velocity u(z_t, r, t) using the MeanFlow identity u = v - (t - r) du/dt. One `jax.jvp` through the model computes the derivative along the flow. Training-time guidance mixes the sample's velocity with the model's unconditional and conditional velocities (`omega`, `kappa`). `norm_p` and `norm_eps` control adaptive weighting of each row's squared error.

As in the reference, a draw at `unconditional_prob` determines how many rows have their condition dropped. These are the first rows in the batch, the instantaneous rows. The loss and gradient match Gsunshine/meanflow's `forward` on the reference's draws (`tools/meanflow_reference.py`). For an interval process, `sample` passes the model the interval to the next grid point at each step. `steps=2` therefore makes one step from noise to data.

MeanFlow's and sCM's losses differentiate the model in time, so its time embedding must be smooth in time. `simple_dit`'s default Fourier scale of 16, applied to a flow's model time (sigma times 1000), is not.

A 2-D two-class toy compared the scales on an RTX 4080. It used a ring of eight Gaussians and a one-token `simple_dit` trained with these objectives. At scale 16, MeanFlow diverged after 8,000 steps, and the sCM student reached 11% class accuracy in one step. With `time_scale=0.002`, MeanFlow reached 99% in one step and 100% in two. The rCM student reached 98.6% in one step. Its teacher needs about 32 Euler steps for 99.9%.

`ShortcutObjective` (Frans et al., 2025) trains a velocity conditioned on its step size. Select `shortcut` on the run config with the `Shortcut` preset. Most rows use flow matching at the finest step, 1 / `sections`. One row in `bootstrap_every` trains self-consistency at dyadic levels: one step of size 2d should match two steps of size d from the EMA weights.

The path is `(1 - (1 - 1e-5) t) * noise + t * data`, as in the reference. It keeps a noise factor of 1e-5 at the data end, and the loss is mean squared error. The loss and gradient match kvfrans/shortcut-models' `get_targets` and `update` loss (`tools/shortcut_reference.py`). The reference reuses the batch's first images for both flow and self-consistency rows. Dew gives each row its own image; each row's loss has the same distribution as the reference's.

`ConsistencyDistillationObjective` implements rCM (Zheng et al., 2025). Set `distill=ConsistencyDistillation(teacher=<run directory>)` on the run config to distill a saved flow run into a few-step student on TrigFlow.

The sCM loss (Lu & Song, 2025) trains the student toward the teacher ODE's tangent using one `jax.jvp` through the student. The DMD2 loss (Yin et al., 2024) moves the student's samples along the difference between a fake score, trained on those samples, and the teacher's score. Set `consistency_weight=0` for DMD2 alone or `dmd_weight=0` for sCM alone. Both losses match NVlabs/rcm's methods on their draws (`tools/rcm_reference.py`).

As in rCM, the student and fake score use separate copies of the optimizer, each stepped on its own updates ([Several networks, several optimizers](objectives.md#several-networks-several-optimizers)). The EMA averages student updates using the student's update count. `ema_decay=power_decay(0.1)` (`dew.training.posthoc`) selects rCM's power EMA. After ten `Trainer` updates, both networks, the EMA and both Adam optimizers' moments match rCM's training loop. The phases alternate on each update, so `accumulation` must be 1. The student samples with `Consistency`.

cuDNN and TPU fused attention kernels provide only reverse-mode derivatives. Where the JVP runs, `dew.nn.attention.forward_mode_attention()` keeps the fused kernel's value but computes its tangent with the reference path. On an RTX 4080, the cuDNN tangent matches the XLA kernel's within bf16 rounding.

`GuidanceDistillationObjective` distills a saved run's classifier-free guidance. Set `guidance_distill=GuidanceDistillation(teacher=<run directory>)` on the run config. The student takes the guidance scale as a conditioning input, as FLUX.1 [dev] does. This is stage one of Meng et al. (2023).

Each row draws a scale w. On the same noised sample, the student's raw output regresses onto the teacher's u + w (c - u). FLUX.1 [dev] has no published training code for its guidance embedding, so the tests compare the loss with the paper's equation. A saved student samples one branch at its conditioner's guidance value.

`AdversarialDistillationObjective` trains a few-step student with LADD's projected discriminator (Sauer et al., 2024). Configure it with `adversarial=AdversarialDistillation(teacher=<run directory>, feature_layers=...)`.

It renoises both clean predictions and reference samples. The frozen teacher's token grids are inputs to StyleGAN-T heads with spectral normalization, local batch normalization and projection conditioning. Hinge losses train each side while stopping gradients through the other. ADD's R1 penalty regularizes the heads. `distillation_weight` adds ADD's alpha-weighted, summed squared distance toward the teacher's denoising (Sauer et al., 2023). LADD omits that term for synthetic data. Dew does not implement ADD's DINOv2 discriminator.

Neither paper publishes training code. `tools/ladd_reference.py` implements a torch oracle for one step using the papers' equations, StyleGAN-T's published `DiscHead` and DiT's timestep embedding. Dew's loss, every gradient and the heads' spectral state match the oracle on Dew's draws.

The papers leave two choices unspecified. Dew trains the student and heads in the same step; StyleGAN-T alternates generator and discriminator steps. Dew also updates spectral norms once per step on the real pass. Other passes use that iteration without storing their own.

The CIFAR-10 runs below use 32-pixel images, a flow `simple_dit` teacher (width 256, six blocks, 3,000 training steps), and FID-5k. Teacher columns give one-/four-/25-step Euler FID with CFG 1.5. Student columns use `Consistency`, without sampling-time CFG.

| Setup | Student steps | Seeds | One-step FID | Four-step FID | Teacher FID (1 / 4 / 25 steps) |
|---|---:|---|---:|---:|---|
| RTX 4080, prototype MLP heads, real data, several learning rates and renoising levels | 1,500–3,000 | 0 | 319–365 | — | 316 / 103 / 90 |
| RTX 4080, StyleGAN-T heads + R1, real data, lr 1e-5, batch 64, no distillation | 4,000 | 0 | 268 | 237 | 316 / 103 / 90 |
| Same | 4,000 | 1 | 283 | 292 | 316 / 103 / 90 |
| Same | 4,000 | 2 | 319 | 283 | 316 / 103 / 90 |
| RTX 4080, StyleGAN-T heads + R1, ADD distillation weight 2.5 | 2,500–4,000 | 0 | 430–437 | — | 316 / 103 / 90 |
| A100, StyleGAN-T heads + R1, synthetic teacher samples, lr 1e-5, batch 128, no distillation | 20,000 | 0 | 243.01 | 168.87 | 310.34 / 102.85 / 85.54 |
| Same | 20,000 | 1 | 216.91 | 160.52 | 310.34 / 102.85 / 85.54 |
| Same | 20,000 | 2 | 254.73 | 143.26 | 310.34 / 102.85 / 85.54 |

Every A100 seed improved one-step FID by 18 to 30% over that run's one-step teacher. None reached the teacher's four-step FID. These runs used the paper's high-noise renoising and resumed between 4,000-step segments. The ADD distance dominated the short real-data runs. Omitting it follows LADD's synthetic-data recipe and leaves ADD's equation unchanged.

A separate two-class 2-D toy exposed the renoising sensitivity in the prototype heads (RTX 4080, one seed, 3,000 steps):

| Setup | One-step class accuracy | Log-density | Mode coverage |
|---|---:|---:|---|
| Paper renoising mean 1, std 1 | 53% | — | Discriminator near chance |
| Renoising mean -2, std 1, distillation weight 2.5, 256-wide prototype heads | 99.6% | 0.49 | All modes |
| Same lower renoising, no distillation or 64-wide prototype heads | — | — | One mode per class |

These runs test training at small scale. They do not reproduce the papers' image-quality results.

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

`sample` visits the `steps` points returned by `process.times(steps)`. It returns the model's clean prediction at the last point. The loop is one `jax.lax.scan`, so it compiles once. Each step draws noise from `key` folded with the step index. You can change the solver without changing the trained weights.

The classic integrators are `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, and `MultiStepDPM`. `MultiStepDPM` is a third-order finite-difference integrator in sigma space that keeps the last three noise estimates. A second group follows the Diffusers schedulers and reproduces the Diffusers 0.34.0 trajectories recorded by `tools/diffusers_reference.py`:

- `DPMSolverMultistep` covers every algorithm, order and second-order form of `DPMSolverMultistepScheduler`, and the EDM scheduler's update over the EDM process.
- `DPMSolverSinglestep`, `DPMSolverSDE`, `DEIS`, `UniPC`, `PNDM`, `LMS`, `KDPM2` (plain and ancestral) and `TCD`.
- `Consistency`, used with the `ConsistencyBoundary` prediction transform, for latent consistency models.

`FlowSDE` is Flow-GRPO's Euler-Maruyama solver on a rectified-flow process.

`Heun` implements Algorithm 2 of Karras et al. (2022). `s_churn`, `s_tmin`, `s_tmax` and `s_noise` control that algorithm's stochasticity. Inside `[s_tmin, s_tmax]`, each step adds fresh noise to raise sigma by a factor of 1 + min(s_churn / N, sqrt(2) - 1). It then takes the Heun step from that sigma. This matches NVlabs' `edm_sampler` (`tools/edm_reference.py`). Churn changes sigma, so it needs a variance-exploding process. Dew rejects a step that would raise sigma above the schedule's highest value.

`MultiStepDPM` and `DPMSolverMultistep` are different solvers. `DPMSolverSDE` implements `DPMSolverSDEScheduler`; it is separate from `DPMSolverMultistep`'s SDE algorithms. Each interval takes two ancestral steps. Both draw noise from one keyed Brownian bridge over the schedule's positive sigma range, so the draws are nested increments of a single path.

`DDPM(variance="large")` uses the wider published posterior variance: the beta of the variance-preserving forward step. Beta is zero wherever alpha is one, so DDPM rejects a variance-exploding grid. With either variance, the step at the schedule's zero time adds no noise.

For DPM-Solver++ 2M without lowering the order at the end, use `DPMSolverMultistep(order=2, algorithm="dpmsolver++", solver_type="midpoint", lower_order_final=False, euler_at_final=False)`. The default `lower_order_final=True` follows Diffusers. With fewer than 15 steps, the last step is first order and the preceding step is at most second order. A solver's `init` takes `(x_T, times, process, key=key)` with a concrete time grid. Invalid endpoints therefore raise an error before the compiled loop starts.

`SourceLimitedPrediction` applies source clipping and dynamic thresholding during the process's prediction conversion. A solver that reads the clean prediction twice therefore gets the limited value both times.

## Guidance

`CFG(scale, interval=(0.0, 1.0), rescale=0.0)` applies classifier-free guidance. At each step, the model predicts once with the condition and once without it. The guided prediction is `uncond + scale * (cond - uncond)`. Outside `interval`, the scale drops to 1 (Kynkäänniemi et al., 2024).

The interval uses fractions of the sampling trajectory and includes both endpoints. Step `i` of `N` is guided when `start <= i / N <= stop`; the final denoise counts as step `N`. The guidance decision applies to every evaluation within that step, including Heun's corrector or a midpoint.

This matches the paper's sampler, checked in `tests/test_guidance.py`. Its `guidance_interval=[a, b]` over `N` steps is `interval=(a / N, b / N)` here. A Diffusers guider applies `start` and `stop` to steps `[int(start * N), int(stop * N))`. To match it, use `interval=(int(start * N) / N, (int(stop * N) - 1) / N)`. The guidance rules below use `interval` in the same way.

`rescale` matches Diffusers' `guidance_rescale` (Lin et al., 2023). It adjusts the guided output toward the conditional output's standard deviation. `rescale=0` leaves the output unchanged.

As in published pipelines, guidance applies to the model's raw outputs before the process converts the result. A nonlinear conversion therefore sees the combined output, without converting the branches separately. To train one model for conditional and unconditional predictions, set `DiffusionObjective(unconditional_prob=...)`. This replaces the condition with its empty value on that fraction of rows, 12% by default.

You can also pass these guidance rules to `sample` and `TextToImage`:

- `APG(scale, eta=1.0, norm_threshold=15.0, momentum=0.0)` is adaptive projected guidance (Sadat et al., 2025). It averages the direction `cond - uncond` over sampling steps with `momentum`, clips its norm at `norm_threshold`, and scales its component parallel to the conditional output by `eta`. With `eta=1`, no clipping and no momentum, it equals `CFG`. It matches Diffusers' `AdaptiveProjectedGuidance` over a complete trajectory, including the interval mapping above. When guidance is off, the momentum average stays unchanged, as in Diffusers.
- `CFGPlusPlus(scale)` is CFG++ (Chung et al., 2025), with `scale` in [0, 1]. It guides the clean prediction and uses the unconditional noise prediction to renoise each step. It requires a solver that steps from the `(x_0, epsilon)` pair, such as `DDIM`.
- `Autoguidance(scale, model)` uses a weaker model of the same task under the same condition (Karras et al., 2024). The guided output is `guide + scale * (model - guide)`. Store the guide's variables under `guide` in the denoiser's variables. This matches NVlabs/edm2's `edm_sampler(gnet=...)`.

## Text to image

`TextToImage` in `dew.inference` combines the text encoder, the denoising loop and, for latent models, the decoder. `objective.pipeline(state)` builds one from a trained objective, and `TextToImage.from_run(directory)` rebuilds one from a saved run.

For Stable Diffusion and SDXL checkpoints, `dew.pipeline(source)` rebuilds the source scheduler. It supports the source's clipping, thresholding and timestep spacing. Where that scheduler supports them, it also handles Karras, exponential and beta grids.

DPM-Solver multistep and UniPC scheduler files with `use_flow_sigmas` use the shifted rectified-flow path and velocity prediction. Flow pipelines such as SANA and Wan ship these files. Unsupported combinations raise an error; Dew does not substitute other defaults. Scheduler checks compare tiny synthetic trajectories with Diffusers. They do not measure image quality. [Supported models](../models.md) lists the pipelines that load.
