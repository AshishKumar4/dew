<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="Dew" width="360">
</picture>

<h1>Dew</h1>

<a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT license"></a>
</div>

Dew is a Python framework for training machine learning models with JAX and Flax Linen. It provides training objectives, data loading, optimization, evaluation, and checkpointing for language models, diffusion models, and JEPA representation learning.

Use a supplied objective or define initialization and a loss for your own Flax model. `Trainer` computes gradients and updates parameters with an Optax optimizer. It also manages device sharding and optional evaluation, tracking, and checkpoints.

## Capabilities and status

- Autoregressive and masked-diffusion language modeling, image and video diffusion, flow matching, and JEPA objectives.
- SFT, DPO, and GRPO with local generation and reward functions.
- Grain-based data pipelines, token packing, and checkpointable training streams.
- Data, parameter, expert, sequence, and pipeline parallelism; optional mixed-precision and quantized computation.
- Hugging Face checkpoint translation for selected decoder and vision families, with small reference fixtures.

Dew is under active development. APIs and checkpoint layouts can change before 1.0. CPU tests, local multiprocess tests, and single-GPU checks cover parts of the implementation; they do not establish production readiness on a multi-host GPU or TPU system. See [capabilities and limitations](docs/reference/support.md) for the scope of each claim. Dew does not currently provide an inference server or continuous request batching.

## Installation

Use Python 3.11 or newer in a virtual environment. These POSIX-shell commands require [uv](https://docs.astral.sh/uv/getting-started/installation/) on your PATH. Install from the repository:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

The distribution is `dew-ml`; Python imports use `dew`. The quickstart below needs no dataset download, model weights, account, or accelerator. See [installation](docs/installation.md) for optional dependencies and the official [JAX hardware installation instructions](https://docs.jax.dev/en/latest/installation.html) before configuring CUDA or TPU support.

## Quickstart

This example learns the relation `y = 2x + 1` with a one-output linear layer. Save the complete block as `train.py` and run `JAX_PLATFORMS=cpu python train.py`.

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Dataset
from dew.objectives.base import Aux, Mean, Objective, mean_loss


class Regression(Objective):
    def __init__(self, model):
        self.model = model

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, 1), dtype=jnp.float32))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        loss = Mean(jnp.sum(errors), jnp.asarray(errors.size))
        mse, _ = mean_loss(loss)
        return loss, Aux(metrics={"mse": mse})


x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
batch = {"x": x, "y": 2 * x + 1}
data = Dataset(train=lambda: itertools.repeat(batch), val=None,
               records=32, batch=32)
model = nn.Dense(features=1)
trainer = Trainer(Regression(model), optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
prediction = model.apply(state.params, x)
mse = float(jnp.mean((prediction - batch["y"]) ** 2))
print(f"Final mean squared error: {mse:.6f}")
assert mse < 1e-4
```

`x` and `y` are float32 arrays with shape `(32, 1)`: 32 examples with one feature or target each. `Regression.loss` returns the squared-error sum and its element count as `Mean`, plus training metrics. `Trainer.fit` consumes 100 batches and returns a `TrainState`; `state.params` contains the Flax variables used by `model.apply`.

The loss should decrease. The CPU validation run printed `Final mean squared error: 0.000000`; the assertion allows small floating-point differences. This example fits a synthetic relation, not a held-out dataset. It writes no checkpoints or run configuration and performs no evaluation pass.

Continue with [your first training run](docs/getting-started.md) to understand initialization, randomness, and how to change the model or batch.

## Documentation and examples

- [Learning paths](docs/index.md): choose a tutorial or a task-oriented guide.
- [Writing a custom objective](docs/concepts/objectives.md): define variables, loss, and optional evaluation.
- [Supplying training data](docs/concepts/data.md): batch conventions, data sources, and reproducibility.
- [Resuming training](docs/guides/checkpoints.md): save model and optimizer state together with data position.
- [Language models](docs/concepts/language_models.md) and [post-training](docs/concepts/post_training.md): token batches, generation, SFT, DPO, and GRPO.
- [Distributed training](docs/concepts/distributed.md): meshes, sharding, and validation limits.
- [Recipes](docs/recipes.md) and [API reference](docs/api.md): entry points and exported interfaces.

The [`examples/`](examples/) directory contains training scripts. Some examples and recipes need external datasets or model downloads. The older [`tutorials/`](tutorials/) notebooks are being revised and are not the starting point for the current API.

## Contributing, license, and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) for development, testing, reference parity, and writing requirements. Dew uses the [MIT license](LICENSE).

For research attribution, cite the model or method you use and identify the Dew commit that produced your results. [References](docs/references.md) lists the underlying work. Dew developed from [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff) and builds on JAX, Flax, Optax, Orbax, and Grain. Earlier FlaxDiff experiments received support from Google TPU Research Cloud; this does not imply current hardware qualification or available cloud resources.
