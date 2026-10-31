# Dew documentation

Dew is a library for training neural networks with JAX and Flax. A training run is a Flax model, an `Objective` that defines the variables and the loss, a `Dataset` that yields batches, and a `Trainer` that runs the optimization on one device or a mesh of devices. Dew ships objectives for language models, image and video diffusion, diffusion language models, JEPA encoders and post-training (SFT, DPO, GRPO, PPO).

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data.dataset import Loader
from dew.objectives.base import Aux, Objective


class Regression(Objective):
    def __init__(self, model):
        self.model = model

    def init(self, key, variables=None):
        return self.model.init(key, np.zeros((1, 1), np.float32))

    def loss(self, variables, batch, step):
        errors = (self.model.apply(variables, batch["x"]) - batch["y"]) ** 2
        return self.row_mean(errors, batch), Aux(metrics={})


x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Loader(train=lambda partition: itertools.repeat(batch), val=None, records=32, batch=32)
trainer = Trainer(Regression(nn.Dense(1)), optax.sgd(0.1), key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

The pages assume you know Python, NumPy and the basics of training with gradients. Pages that also need Flax or sharding say so at the top.

## Getting started

1. [Installation](installation.md): install Dew and check which devices JAX sees.
2. [Quickstart](getting-started.md): the example above, one part at a time.
3. [Key concepts](key-concepts.md): the model, the objective, the dataset and the trainer.
4. [Tutorials](tutorials.md): notebooks that train diffusion models, a language model and a JEPA encoder on real data.

## Topics

| Topic | Page |
|---|---|
| Batches, dataset specifications, TFDS and Hugging Face data, resuming the data stream | [Training data](concepts/data.md) |
| Writing an `Objective` | [Custom objectives](concepts/objectives.md) |
| Validation metrics, previews and experiment trackers | [Evaluation and tracking](guides/evaluation.md) |
| Saving, resuming and exporting | [Checkpoints](guides/checkpoints.md) |
| Diffusion processes, presets, solvers and guidance | [Diffusion processes and solvers](concepts/diffusion.md) |
| Training an image diffusion model | [Diffusion training](guides/diffusion.md) |
| Language models: training, loading published checkpoints | [Language models](concepts/language_models.md) |
| Diffusion language models: masked diffusion, DiffusionGemma | [Diffusion language models](concepts/diffusion_language_models.md) |
| Mixture-of-experts layers | [Mixture of experts](concepts/moe.md) |
| SFT, DPO, GRPO and PPO | [Post-training](concepts/post_training.md) |
| Text generation and serving | [Generation and serving](concepts/inference.md) |
| JEPA encoders | [JEPA](guides/representation-learning.md) |
| Meshes, sharding rules and several hosts | [Distributed training](concepts/distributed.md), [Multiple hosts](guides/multi-node.md), [Cloud TPUs](tpu.md) |
| Checkpoints Dew can load | [Supported models](models.md) |
| Every public class and function | [API reference](reference/core-api.md) |

## Status

Dew is research software before version 1.0, so the API and checkpoint formats can change between versions. I have run it on CPU, on pools of local processes, on single GPUs, on one host with four GPUs and on one TPU v6e chip, but not on two physical nodes. [Multiple hosts](guides/multi-node.md) lists what has and has not been run. The [roadmap](roadmap.md) lists work still to complete or qualify.

[Papers and attribution](references.md) lists the papers and upstream code behind the models and methods. The binding [architecture](design/architecture.md), [training](design/training.md), [data](design/data.md), [post-training](design/post-training.md) and [testing](design/testing.md) specs stay in the repository under `docs/design`. Research notes stay under `docs/research`.
