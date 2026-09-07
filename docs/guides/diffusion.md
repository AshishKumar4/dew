# Train an image diffusion model

This guide assumes the [first training run](../getting-started.md), image tensors, and the idea of predicting a clean signal from a noisy input. It introduces Dew's diffusion configuration through a small flow-matching run. No data or model download is required.

## Train on synthetic images

The example creates eight stripe images, trains a small DiT, and writes four generated preview images to a NumPy file. It demonstrates the API and output range, not image quality.

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
assert output.shape == (4, 8, 8, 3)
assert np.all(np.isfinite(output))
assert output.min() >= -1 and output.max() <= 1
np.save("preview.npy", output)
assert Path("preview.npy").is_file()
print("Saved preview.npy:", output.shape)
```

The image batch is NHWC: batch, height, width, channels. The source uses uint8 pixels in `[0, 255]`; the objective converts them to the model's normalized range. `Field("image", (8, 8, 3))` describes one sample, excluding the batch dimension.

The DiT divides each 8×8 image into 4×4 patches, producing four spatial tokens. These tiny dimensions keep the example suitable for a CPU smoke run. Real training requires more data, capacity, and optimization steps.

## Understand the process and solver

`Flow()` is a configuration value. Calling it builds the `Process` used for noise sampling, target construction, and prediction conversion. That explains the two calls in `Flow()()`: construct the preset, then construct its process.

The objective samples noise and noise levels and computes the flow-matching loss. Its `steps=4` setting controls evaluation sampling, not the number of optimizer updates. `trainer.fit(..., steps=3)` controls the training target.

`Euler()` is the numerical solver used for preview generation. Changing a solver, noise schedule, or prediction transform changes sampling semantics; choose a compatible process and solver. Other supplied solvers include DDPM, DDIM, Heun, RK4 and Euler ancestral, and the Diffusers scheduler updates: `DPMSolverMultistep` (every algorithm, order and second-order form of `DPMSolverMultistepScheduler`, and the EDM scheduler's update over the EDM process), `DPMSolverSinglestep`, `DEIS`, `UniPC`, `PNDM`, `LMS`, `KDPM2` (plain and ancestral), `TCD`, and `Consistency` with the `ConsistencyBoundary` prediction transform for latent consistency models. Each reproduces Diffusers 0.34.0's trajectories on the fixtures `tools/diffusers_reference.py` records. The named `MultiStepDPM` implementation is a finite-difference sigma integrator, not a Diffusers scheduler.

## Inspect the preview

The script writes `preview.npy` in its working directory. It contains four float32-compatible image arrays in `[-1, 1]`. For display, convert with `(output + 1) / 2` and clip to `[0, 1]`. Three updates on stripe images will not produce a useful generative model.

The manual call to `evaluate` uses an explicit independent key. For evaluation scheduled by `Trainer.fit`, current RNG reuse and preview/scoring limitations apply. A four-image preview is not a dataset-level generative-quality metric; see [evaluation and tracking](evaluation.md).

## Add conditions or a latent encoder

For text conditioning, `InputSpec.conditions` maps model keyword arguments to condition encoders and their batch fields. Tokenize captions with the same encoder/tokenizer configuration used for training. A pretrained text tower can require a model download and significant memory.

A configured autoencoder changes training from pixels to latent tensors. Match the denoising model's channel count and spatial shape to the encoder output and retain the correct scaling convention. Loading a VAE does not by itself load an external diffusion transformer's weights.

Use [training recipes](../recipes.md) for dataset-backed runs, and consult [capability limits](../reference/support.md) before selecting external checkpoints or a distributed deployment.
