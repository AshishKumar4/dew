# Diffusion training

`DiffusionObjective` trains a denoising model under a diffusion `Process`: it noises each image to a random time, runs the model and computes the process's loss. Its `evaluate` samples images with a solver. This page trains a small flow-matching DiT on synthetic images; the data, the model and the steps are tiny so the example runs on a CPU in seconds, and the samples say nothing about image quality. [Diffusion processes and solvers](../concepts/diffusion.md) describes the processes, presets and solvers.

## Example

The example makes eight striped 8×8 images, trains for three steps and samples one image per batch row.

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
data = Dataset(train=lambda partition: itertools.repeat(batch), val=None, records=8, batch=8)
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

The image batch is NHWC (batch, height, width, channels) uint8 in `[0, 255]`; the objective converts it to the model's range `[-1, 1]`. `InputSpec(Field("image", (8, 8, 3)))` describes one sample, without the batch dimension. The DiT cuts each 8×8 image into 4×4 patches, four tokens per image.

## Process and solver

`Flow()` is a preset, a frozen dataclass of the numbers that define rectified flow. Calling it builds the `Process`, which draws noise and times, builds the training target and converts the model's output; hence `Flow()()`.

`DiffusionObjective(steps=4)` is the number of solver steps `evaluate` samples with; `trainer.fit(..., steps=3)` is the number of training steps.

`Euler()` is the solver `evaluate` uses. The solver can change without retraining; the process cannot, because it defines what the model learned to predict.

## Samples

`evaluate` returns an image artifact whose `images` are float32 in `[-1, 1]`, one sample per row of the batch it is given; `(output + 1) / 2` maps them to `[0, 1]` for display.

The example passes `evaluate` its own key. When `Trainer.fit` schedules evaluation, the keys come from the run key and the step. [Evaluation and tracking](evaluation.md) covers scheduled evaluation and the image metrics (FID, CLIP score).

## Conditions and latent encoders

For text conditioning, `InputSpec.conditions` maps a model keyword argument to a condition encoder and the batch field it reads. Captions must be tokenized with the encoder and tokenizer settings used in training. A pretrained text tower is downloaded on first use.

With an autoencoder configured, training runs on latents instead of pixels. The denoising model's channel count and spatial shape must match the encoder's output, and the encoder's scaling convention must be kept.

[Recipes](../recipes.md) runs diffusion training on real datasets from the command line. [Supported models](../models.md) lists the published diffusion checkpoints that load.
