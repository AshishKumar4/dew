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

The image batch is NHWC: batch, height, width, channels. The source images are uint8 pixels in `[0, 255]`, and the objective converts them to the model's normalized range. `Field("image", (8, 8, 3))` describes one sample, without the batch dimension.

The DiT cuts each 8×8 image into 4×4 patches, which gives four spatial tokens. The sizes are this small so the example runs in seconds. Real training needs more data, a larger model and many more optimizer steps.

## Understand the process and solver

`Flow()` is a configuration value. Calling it builds the `Process`, which samples noise, builds the training target and converts the model's prediction. That is why the example writes `Flow()()`: the first call makes the preset and the second makes its process.

The objective samples noise and noise levels and computes the flow-matching loss. Its `steps=4` sets the number of sampling steps for evaluation previews, not the number of optimizer updates. `trainer.fit(..., steps=3)` sets the training target.

`Euler()` is the numerical solver that generates the preview. A different solver, noise schedule or prediction transform samples differently, so pick a process and solver that work together. [Diffusion processes and solvers](../concepts/diffusion.md) lists the presets and solvers and explains classifier-free guidance.

## Inspect the preview

The script writes `preview.npy` to its working directory. It holds eight float32 images with values in `[-1, 1]`: `evaluate` draws one sample for every real row of the batch it is given. To display them, compute `(output + 1) / 2` and clip to `[0, 1]`. Three updates on striped images do not give you a useful generative model.

The call to `evaluate` passes its own independent key. When `Trainer.fit` schedules evaluation, it derives the keys from the run key and the step instead; [evaluation and tracking](evaluation.md) describes this. Eight samples are not a dataset-level measure of generative quality; that guide also covers the metrics.

## Add conditions or a latent encoder

For text conditioning, `InputSpec.conditions` maps a model keyword argument to a condition encoder and the batch field it reads. Tokenize captions with the same encoder and tokenizer settings you train with. A pretrained text tower may need a model download and a lot of memory.

With an autoencoder configured, training runs on latent tensors instead of pixels. Set the denoising model's channel count and spatial shape to match the encoder output, and keep the encoder's scaling convention. Loading a VAE does not load the weights of an external diffusion transformer.

Use [training recipes](../recipes.md) for runs on real datasets. [Supported models](../models.md) lists the published checkpoints that load.
