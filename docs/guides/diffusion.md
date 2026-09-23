# Train an image diffusion model

This guide assumes you have done the [first training run](../getting-started.md), know how image tensors are laid out, and know the idea of predicting a clean signal from a noisy input. It walks through Dew's diffusion configuration with a small flow-matching run. You do not need to download any data or model.

## Train on synthetic images

The example makes eight striped images, trains a small DiT on them, and saves one generated image per batch row, eight in all, to a NumPy file. It shows the API and the output range. It says nothing about image quality.

```python
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Field, InputSpec, Trainer
from dew.data import Dataset
from dew.diffusion.presets import Flow
from dew.nn.backbones.dit import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import Euler

images = np.zeros((8, 8, 8, 3), dtype=np.uint8)
images[:, :, ::2, :] = 255
batch = {"image": images}
data = Dataset(train=lambda: itertools.repeat(batch), val=None, records=8, batch=8)
model = SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                  mlp_ratio=2, dtype=jnp.float32, attention_impl="xla")
objective = DiffusionObjective(model, Flow()(), InputSpec(Field("image", (8, 8, 3))),
                               sampler=Euler(), steps=4)
trainer = Trainer(objective, optax.adam(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
info = Step(step=state.step, key=jax.random.key(1), ema=state.averaged)
preview = objective.evaluate(state.params, batch, info)
output = np.asarray(preview.images)
assert output.shape == (8, 8, 8, 3)
assert np.all(np.isfinite(output))
assert output.min() >= -1 and output.max() <= 1
np.save("preview.npy", output)
assert Path("preview.npy").is_file()
print("Saved preview.npy:", output.shape)
```

The image batch is NHWC: batch, height, width, channels. The source images are uint8 pixels in `[0, 255]`, and the objective converts them to the model's normalized range. `Field("image", (8, 8, 3))` describes one sample, without the batch dimension.

The DiT cuts each 8×8 image into 4×4 patches, which gives four spatial tokens. The sizes are this small so the example runs in seconds. Real training needs more data, a larger model and many more optimizer steps.

## Understand the process and solver

`Flow()` is a configuration value. Calling it builds the `Process`, which samples noise, builds the training target and converts the model's prediction. That is why the example writes `Flow()()`: the first call makes the preset and the second makes its process.

The objective samples noise and noise levels and computes the flow-matching loss. Its `steps=4` sets the number of sampling steps for evaluation previews, not the number of optimizer updates. `trainer.fit(..., steps=3)` sets the training target.

`Euler()` is the numerical solver that generates the preview. A different solver, noise schedule or prediction transform samples differently, so pick a process and solver that work together. Dew also has DDPM, DDIM, Heun, RK4 and Euler ancestral, plus solvers that follow the Diffusers schedulers:

- `DPMSolverMultistep` covers every algorithm, order and second-order form of `DPMSolverMultistepScheduler`, and the EDM scheduler's update over the EDM process.
- `DPMSolverSinglestep`, `DPMSolverSDE`, `DEIS`, `UniPC`, `PNDM`, `LMS`, `KDPM2` (plain and ancestral) and `TCD`.
- `Consistency`, used with the `ConsistencyBoundary` prediction transform, for latent consistency models.

Each of these reproduces the Diffusers 0.34.0 trajectories recorded by `tools/diffusers_reference.py`. `MultiStepDPM` has a similar name but is a different thing: a finite-difference integrator in sigma, not a Diffusers scheduler.

`DPMSolverSDE` is the solver of `DPMSolverSDEScheduler`. It is not one of the SDE algorithms of `DPMSolverMultistep`. Each interval takes two ancestral steps from its own start. Both steps draw noise from one keyed dyadic Brownian bridge over the schedule's positive sigma range, so the two draws are correlated as nested increments of a single path.

`DDPM(variance="large")` uses the wider published posterior variance, which is the beta of the variance-preserving forward step. That beta is zero wherever alpha is one, so DDPM refuses a variance-exploding grid instead of sampling it without noise. Neither variance adds noise on the step whose own time is the schedule's zero.

Source clipping and dynamic thresholding live in `SourceLimitedPrediction`. They are part of the process's prediction conversion, not part of a solver, so a solver that reads the clean prediction twice sees the limited value both times.

DPM-Solver++ 2M without any lowering of order at the end is `DPMSolverMultistep(order=2, algorithm="dpmsolver++", solver_type="midpoint", lower_order_final=False, euler_at_final=False)`. By default `lower_order_final=True`, which follows Diffusers: in a walk of fewer than 15 steps, the last step is first order and the one before it at most second order. A solver's `init` takes `(x_T, times, process, key=key)` with a concrete time grid, so an invalid pair of endpoints fails before the compiled loop starts. `key` is the root key of the walk; only `DPMSolverSDE` reads it, for its Brownian state.

`CFG(scale, interval=..., rescale=...)` applies guidance to the model's raw outputs, and then the process converts the guided output once. Published pipelines run their scheduler in the same order, so a nonlinear conversion never sees the two branches separately. `rescale` is Diffusers' `guidance_rescale`, the standard-deviation correction of Lin et al. 2023. `rescale=0` gives the plain guided output.

## Inspect the preview

The script writes `preview.npy` to its working directory. It holds eight float32 images with values in `[-1, 1]`: `evaluate` draws one sample for every real row of the batch it is given. To display them, compute `(output + 1) / 2` and clip to `[0, 1]`. Three updates on striped images do not give you a useful generative model.

The call to `evaluate` passes its own independent key. When `Trainer.fit` schedules evaluation, it derives the keys from the run key and the step instead; [evaluation and tracking](evaluation.md) describes this. Eight samples are not a dataset-level measure of generative quality; that guide also covers the metrics.

## Add conditions or a latent encoder

For text conditioning, `InputSpec.conditions` maps a model keyword argument to a condition encoder and the batch field it reads. Tokenize captions with the same encoder and tokenizer settings you train with. A pretrained text tower may need a model download and a lot of memory.

With an autoencoder configured, training runs on latent tensors instead of pixels. Set the denoising model's channel count and spatial shape to match the encoder output, and keep the encoder's scaling convention. Loading a VAE does not load the weights of an external diffusion transformer.

Use [training recipes](../recipes.md) for runs on real datasets. The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) names the checkpoints that load.
