# Train models with Dew

Dew provides Python interfaces for training Flax Linen models with JAX. An `Objective` defines initialization, a loss function, and optional evaluation. `Trainer` runs optimization and manages device sharding and checkpoints when configured.

These pages assume Python and basic machine learning: batches, loss functions, gradients, and train/validation splits. You can start without experience running JAX on multiple devices. Pages that require Flax or sharding knowledge state that prerequisite.

## Start here

1. [Install Dew](installation.md) in a virtual environment and check which JAX devices are available.
2. [Run a small training example](getting-started.md) without downloading data or weights.
3. [Write a custom objective](concepts/objectives.md) and adapt the model and data to your experiment.

The first example teaches the full sequence: create batches, initialize a Flax model, compute a loss, optimize it, and inspect the returned variables. It uses a linear regression problem so you can check the result directly.

## Follow a task

| Task | Guide |
|---|---|
| Read and batch your own data | [Supplying training data](concepts/data.md) |
| Save a run and continue it | [Resuming training](guides/checkpoints.md) |
| Measure validation behavior | [Evaluation and tracking](guides/evaluation.md) |
| Train or load a language model | [Language models](concepts/language_models.md) |
| Fine-tune with SFT, DPO, or GRPO | [Post-training](concepts/post_training.md) |
| Train an image diffusion model | [Image diffusion](guides/diffusion.md) |
| Train a JEPA encoder | [Representation learning](guides/representation-learning.md) |
| Place a model on multiple devices | [Distributed training](concepts/distributed.md) |
| Work with expert routing | [Mixture of experts](concepts/moe.md) |

## Understand the interfaces

[Objectives and state](concepts/objectives.md) explains what the loss receives and returns. [Data](concepts/data.md) explains how dataset specifications produce iterators. [Distributed training](concepts/distributed.md) explains meshes and parameter placement after you have run a single-device example.

Use the [core API reference](reference/core-api.md) for signatures and defaults and the [module index](api.md) to locate other interfaces. Consult [capabilities and limitations](reference/support.md) before choosing a model, precision mode, or deployment setup. A translated model configuration, a small numerical parity fixture, and a full checkpoint run are different levels of validation.

## Project status

Dew is pre-1.0 research software. Checkpoint and API compatibility can change. Current verification includes CPU, local process-pool, and single-GPU work; multi-host GPU and TPU deployment requires separate validation. Known training and evaluation limitations appear beside the affected workflow.

[References](references.md) links to JAX, Flax, and the model and method papers. Internal design history and research notes remain in the repository for contributors; they are not prerequisites for using Dew.
