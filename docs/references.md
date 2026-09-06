# References and attribution

These papers explain methods used in Dew, and these projects provide dependencies, reference implementations, or code that Dew adapts. A citation is not a claim that Dew reproduces every result or supports every model in the cited project. Use the [API overview](api.md) for Dew's interfaces and [benchmarks](benchmarks.md) for measurements with their hardware and revision context.

## Diffusion and flow models

For an introduction, start with DDPM for the denoising objective, then EDM for the relationship between noise levels, network parameterization, training, and sampling. Flow Matching describes a related way to learn continuous paths between distributions. The remaining papers cover particular samplers, weighting rules, architectures, or guidance choices.

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

Dew's `simple_dit` is not an exact copy of the DiT architecture. In particular, its time embedding uses EDM-style random Fourier features rather than DiT's sinusoidal embedding. Time scaling belongs to the chosen model/training convention; a flow preset's factor of 1000 is not a universal requirement of Fourier features. See the [diffusion guide](guides/diffusion.md) for how a preset and model fit together.

## Representation learning and sequence models

- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)](https://arxiv.org/abs/2301.08243) explains predicting hidden image representations from visible context.
- [Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)](https://arxiv.org/abs/2404.08471) extends feature prediction to video.
- [Simplified State Space Layers for Sequence Modeling (S5)](https://arxiv.org/abs/2208.04933) describes the state-space layer used as a reference for Dew's SSM components.

The [representation-learning guide](guides/representation-learning.md) explains the encoder, predictor, target encoder, and masks before introducing recipe settings.

## Language-model post-training

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) derives DPO from preference data and a reference policy.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) introduces GRPO.
- [TRL](https://github.com/huggingface/trl) and [verl](https://github.com/verl-project/verl) provide reference implementations for the narrow numerical comparisons recorded in [post-training](concepts/post_training.md).

Those comparisons concern fixed losses, gradients, and one chat rendering case. They do not establish end-to-end learning parity, agentic RL, or production deployment support.

## Dependencies and adapted code

Dew uses [JAX](https://github.com/jax-ml/jax) for array computation and transformations, [Flax Linen](https://github.com/google/flax) for neural-network modules, [Optax](https://github.com/google-deepmind/optax) for optimization, [Orbax](https://github.com/google/orbax) for checkpointing, and [Grain](https://github.com/google/grain) for data loading. [tyro](https://github.com/brentyi/tyro) turns recipe dataclasses into command-line interfaces. [Weights & Biases](https://github.com/wandb/wandb) provides optional run tracking.

Image and public-dataset paths use [Albumentations](https://github.com/albumentations-team/albumentations), [OpenCV](https://github.com/opencv/opencv-python), and [TensorFlow Datasets](https://github.com/tensorflow/datasets), depending on the chosen loader. [Transformers](https://github.com/huggingface/transformers) supplies tokenizer and encoder integrations, and [safetensors](https://github.com/huggingface/safetensors) supplies tensor-file serialization. These are not all required for the offline byte-token and in-memory examples; follow [installation](installation.md) for the extras your workflow needs.

The Stable Diffusion Flax VAE derives from [Hugging Face Diffusers](https://github.com/huggingface/diffusers) v0.29.2, under Apache-2.0. Parts of Dew's attention blocks also derive from Diffusers' Flax attention implementation. The InceptionV3 implementation used for FID derives primarily from [jax-fid](https://github.com/matthias-wright/jax-fid), with the model's documented PyTorch/torchvision ancestry. Keep upstream attribution and applicable license notices when redistributing adapted code; the dependency, checkpoint, and dataset licenses remain separate from Dew's license.

[facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) and [facebookresearch/jepa](https://github.com/facebookresearch/jepa) are the reference codebases for JEPA masking and probe behavior. [Katherine Crowson's k-diffusion](https://github.com/crowsonkb/k-diffusion/) and [NVIDIA's EDM implementation](https://github.com/NVlabs/edm) are references for diffusion parameterizations, schedules, and solvers.

## Tutorials and further reading

[Sander Dieleman's posts](https://sander.ai/posts/) cover [diffusion](https://sander.ai/2022/01/31/diffusion.html), [typicality](https://sander.ai/2020/09/01/typicality.html), [guidance geometry](https://sander.ai/2023/08/28/geometry.html#warning), and [noise schedules](https://sander.ai/2024/06/14/noise-schedules.html). [Tony Duan's Diffusion Models from Scratch](https://www.tonyduan.com/diffusion/index.html) presents the mathematics with small MNIST implementations and [accompanying code](https://github.com/tonyduan/diffusion).

The Keras tutorials for [DDPM by A_K Nain](https://keras.io/examples/generative/ddpm/) and [DDIM by András Béres](https://keras.io/examples/generative/ddim/) were starting points for the author's original FlaxDiff experiments. They remain useful introductions, though their APIs are not Dew's. The [FlaxDiff history page](from-flaxdiff.md) separates those older experiments from current runs.

## Related projects and interoperability

[MaxText](https://github.com/AI-Hypercomputer/maxtext) and [Levanter](https://github.com/stanford-crfm/levanter) are JAX language-model training projects. [verl](https://github.com/verl-project/verl) focuses on RL post-training and [vLLM](https://github.com/vllm-project/vllm) on inference and serving. Their support matrices and deployment requirements are their own.

Dew's `save_hf_layout` writes filenames in a Hugging Face-style directory without translating tensor names or model configuration. The separate `save_pretrained_decoder` API performs family-aware translation for the decoder cases it accepts and refuses unsupported model features. Neither API alone proves that a particular serving engine can run the export. Check family-specific boundaries in [language models](concepts/language_models.md), keep the matching tokenizer assets, and validate the export in the intended consumer.
