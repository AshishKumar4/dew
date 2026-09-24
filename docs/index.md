# Dew documentation

Dew is a Python library for training models with JAX and Flax. You describe what to learn with an `Objective`: how to initialize a model's variables and how to compute its loss. `Trainer` runs the optimization, places the model on your devices, keeps a moving average of the weights and writes checkpoints if you ask it to. Built-in objectives cover language models, image and video diffusion, diffusion language models, JEPA and post-training.

I assume you know Python and the basics of machine learning: batches, loss functions, gradients, and train and validation splits. You do not need to have run JAX on more than one device. A page that needs Flax or sharding knowledge says so at the top.

## Start here

1. [Install Dew](installation.md) and check which JAX devices it can see.
2. [Run the quickstart](getting-started.md). It fits a small model in seconds on a CPU and downloads nothing.
3. Read [Key concepts](key-concepts.md) for the four objects every run is made of.
4. Work through the [tutorials](tutorials.md), notebooks that train real models on real data and run on Colab.

## Find what you need

The pages come in four kinds. Tutorials teach a complete workflow from start to finish. How-to guides answer one task each and assume you know the basics. Concept pages explain how a part of Dew works and why. The reference lists every public name.

| I want to | Read |
|---|---|
| Feed my own data to a run | [Supply training data](concepts/data.md) |
| Train with a loss Dew does not have | [Write a custom objective](concepts/objectives.md) |
| Measure validation behavior and log a run | [Evaluate and track runs](guides/evaluation.md) |
| Save a run and continue it | [Save and resume](guides/checkpoints.md) |
| Train a language model, or load a published one | [Train language models](concepts/language_models.md) |
| Fine-tune with SFT, DPO, GRPO or PPO | [Post-train with SFT, DPO and RL](concepts/post_training.md) |
| Generate from a trained model or serve it | [Generate and serve](concepts/inference.md) |
| Train an image diffusion model | [Train a diffusion model](guides/diffusion.md) |
| Train a JEPA encoder | [Train a JEPA encoder](guides/representation-learning.md) |
| Place a model on several devices or hosts | [Distributed training](concepts/distributed.md), [Train on several nodes](guides/multi-node.md) |
| Run on Cloud TPUs | [Run on Cloud TPUs](tpu.md) |
| Look up a class or function | [API reference](reference/core-api.md) |
| Check whether a checkpoint loads | [Supported models](models.md) |

## Project status

Dew is research software and has not reached 1.0. Checkpoint formats and the API can change between versions. I have tested it on CPU, on pools of local processes, on single GPUs, on one host with four GPUs and on one TPU v6e chip. It has not run on two physical nodes. Each guide lists the known limits next to the workflow they affect, and [Train on several nodes](guides/multi-node.md#what-has-and-has-not-been-run) keeps the full list of what has and has not been run.

[Papers and attribution](references.md) links the papers and upstream code behind the models and methods. The repository also holds design history and research notes for contributors; you do not need them to use Dew.
