# Train models with Dew

> An AI assistant maintains this document. It is presented as-is.

Dew is a Python library for training Flax Linen models with JAX. You write an `Objective` that sets up the model's variables, computes a loss, and optionally evaluates. `Trainer` runs the optimization. It also places the model on your devices and writes checkpoints if you ask it to.

I assume you know Python and the basics of machine learning: batches, loss functions, gradients, and train/validation splits. You do not need to have run JAX on more than one device. A page that needs Flax or sharding knowledge says so at the top.

## Start here

1. [Install Dew](installation.md) in a virtual environment and check which JAX devices it can see.
2. [Run a small training example](getting-started.md). It downloads no data and no weights.
3. [Write a custom objective](concepts/objectives.md) and change the model and data for your own experiment.

The first example goes through the whole sequence once: make a batch, initialize a Flax model, compute a loss, optimize it, and look at the trained variables. It fits a straight line, so you can check the answer yourself.

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

[Objectives and state](concepts/objectives.md) explains what the loss function receives and what it returns. [Data](concepts/data.md) explains how a dataset specification turns into iterators. [Distributed training](concepts/distributed.md) explains meshes and parameter placement; read it after you have run something on one device.

The [core API reference](reference/core-api.md) lists the signatures and defaults of the documented interfaces. The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) names the model configurations whose whole workflow runs, and for each unfinished one, the piece it is missing.

## Project status

Dew is research software and has not reached 1.0. Checkpoint formats and the API can change between versions. I have tested it on CPU, on a local pool of processes, and on a single GPU. Multi-host GPU and TPU deployments need their own testing. Each guide lists the known training and evaluation limitations next to the workflow they affect.

[References](references.md) links to JAX, Flax, and the papers behind the models and methods. The repository also holds design history and research notes for contributors. You do not need to read them to use Dew.
