# References and attribution

> An AI assistant maintains this document. It is presented as-is.

The papers below explain methods that Dew uses. The projects below are Dew's dependencies, reference implementations it is checked against, or sources of code it adapts. Citing a paper or project does not mean Dew reproduces all of its results or supports every model it covers. For Dew's own interfaces, use the [core API reference](reference/core-api.md). For measurements with their hardware and revision, see [benchmarks](benchmarks.md).

## Diffusion and flow models

If you are new to diffusion, read DDPM first for the denoising objective. Then read EDM, which explains how noise levels, the network's parameterization, training, and sampling relate to each other. Flow Matching describes a related way to learn continuous paths between two distributions. The other papers each cover a particular sampler, weighting rule, architecture, or guidance method.

| Topic | Paper |
| --- | --- |
| Denoising diffusion | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| Implicit sampling | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) |
| Learned variances and training choices | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) |
| Image synthesis and classifier guidance | [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) |
| Continuous-time score models | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) |
| Parameterization and sampling design | [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) |
| P2 loss weighting | [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227) |
| Pseudo numerical sampling methods | [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778) |
| Differential-equation solvers | [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) |
| Transformer diffusion backbones | [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) |
| Rectified-flow transformers | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) |
| Continuous flow objectives | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) |
| Min-SNR loss weighting | [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) |
| Guidance over part of the sampling path | [Applying Guidance in a Limited Interval Improves Sample and Distribution Quality](https://arxiv.org/abs/2404.07724) |
| Diffusion language modeling and a square-root schedule | [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) |

Dew's `simple_dit` embeds the time step with EDM-style random Fourier features, where the original DiT uses sinusoidal embeddings. So `simple_dit` is a variant of the DiT architecture. How the time value is scaled belongs to the model and training convention you pick. The flow preset multiplies time by 1000, but Fourier features do not require that factor in general. The [diffusion guide](guides/diffusion.md) shows how a preset and a model fit together.

## Representation learning and sequence models

- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)](https://arxiv.org/abs/2301.08243) predicts the representations of hidden image regions from the visible context.
- [Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)](https://arxiv.org/abs/2404.08471) applies the same feature prediction to video.
- [Simplified State Space Layers for Sequence Modeling (S5)](https://arxiv.org/abs/2208.04933) describes the state-space layer that Dew's SSM components follow.

The [representation-learning guide](guides/representation-learning.md) explains the encoder, the predictor, the target encoder, and the masks before it gets to recipe settings.

## Language-model post-training

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) derives DPO from preference data and a reference policy.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) introduces GRPO.
- [TRL](https://github.com/huggingface/trl) and [verl](https://github.com/verl-project/verl) are the reference implementations for the small numerical comparisons recorded in [post-training](concepts/post_training.md).

Those comparisons check fixed losses, gradients, and one chat-rendering case. They do not show that Dew learns end to end like TRL or verl, and they say nothing about agentic RL or production deployment.

## Dependencies and adapted code

Dew uses [JAX](https://github.com/jax-ml/jax) for arrays and transformations, [Flax Linen](https://github.com/google/flax) for neural-network modules, [Optax](https://github.com/google-deepmind/optax) for optimizers, [Orbax](https://github.com/google/orbax) for checkpoints, and [Grain](https://github.com/google/grain) for data loading. [tyro](https://github.com/brentyi/tyro) turns the recipe dataclasses into command-line interfaces. [Weights & Biases](https://github.com/wandb/wandb) is an optional run tracker.

Depending on the loader you choose, the image and public-dataset paths use [Albumentations](https://github.com/albumentations-team/albumentations), [OpenCV](https://github.com/opencv/opencv-python), and [TensorFlow Datasets](https://github.com/tensorflow/datasets). [Transformers](https://github.com/huggingface/transformers) provides tokenizers and encoders, and [safetensors](https://github.com/huggingface/safetensors) provides the tensor file format. The offline byte-token and in-memory examples do not need all of these; [installation](installation.md) lists the extras each workflow needs.

The Stable Diffusion Flax VAE is adapted from [Hugging Face Diffusers](https://github.com/huggingface/diffusers) v0.29.2, which is Apache-2.0. Parts of Dew's attention blocks are also adapted from Diffusers' Flax attention code. The InceptionV3 model used for FID is adapted mainly from [jax-fid](https://github.com/matthias-wright/jax-fid), which itself descends from the PyTorch/torchvision model. If you redistribute adapted code, keep the upstream attribution and license notices. The licenses of dependencies, checkpoints, and datasets are separate from Dew's license.

[facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) and [facebookresearch/jepa](https://github.com/facebookresearch/jepa) are the reference code for JEPA masking and probes. [Katherine Crowson's k-diffusion](https://github.com/crowsonkb/k-diffusion/) and [NVIDIA's EDM implementation](https://github.com/NVlabs/edm) are the references for diffusion parameterizations, schedules, and solvers.

## Tutorials and further reading

[Sander Dieleman's posts](https://sander.ai/posts/) cover [diffusion](https://sander.ai/2022/01/31/diffusion.html), [typicality](https://sander.ai/2020/09/01/typicality.html), [guidance geometry](https://sander.ai/2023/08/28/geometry.html#warning), and [noise schedules](https://sander.ai/2024/06/14/noise-schedules.html). [Tony Duan's Diffusion Models from Scratch](https://www.tonyduan.com/diffusion/index.html) works through the mathematics with small MNIST implementations and [accompanying code](https://github.com/tonyduan/diffusion).

I started my original FlaxDiff experiments from the Keras tutorials for [DDPM by A_K Nain](https://keras.io/examples/generative/ddpm/) and [DDIM by András Béres](https://keras.io/examples/generative/ddim/). They are still good introductions, though their APIs are not Dew's. The [FlaxDiff history page](from-flaxdiff.md) keeps those older experiments apart from current runs.

## Related projects and interoperability

[MaxText](https://github.com/AI-Hypercomputer/maxtext) and [Levanter](https://github.com/stanford-crfm/levanter) are JAX projects for training language models. [verl](https://github.com/verl-project/verl) is for RL post-training and [vLLM](https://github.com/vllm-project/vllm) is for inference and serving. Each project documents its own supported models and deployment requirements.

Dew's `save_hf_layout` writes `model.safetensors` and `config.json` into a directory in the Hugging Face layout. It does not translate tensor names or the model configuration. The separate `save_pretrained_decoder` API does translate for the decoder families it accepts, refuses model features it does not support, and writes the files of the tokenizer you give it next to the weights. Neither API guarantees that a given serving engine can run the export. Check the family-specific limits in [language models](concepts/language_models.md), pass the tokenizer you trained with, and test the export in the program that will load it.
