# References

## Research papers and preprints
- The Original Denoising Diffusion Probabilistic Models (DDPM) [paper](https://arxiv.org/abs/2006.11239)
- Denoising Diffusion Implicit Models (DDIM) [paper](https://arxiv.org/abs/2010.02502)
- Improved Denoising Diffusion Probabilistic Models [paper](https://arxiv.org/abs/2102.09672)
- Diffusion Models beat GANs on image synthesis [paper](https://arxiv.org/pdf/2105.05233)
- Score-Based Generative Modeling through Stochastic Differential Equations [paper](https://arxiv.org/pdf/2011.13456)
- Elucidating the design space of Diffusion-based generative models (EDM) [paper](https://arxiv.org/abs/2206.00364)
- Perception Prioritized Training of Diffusion Models (P2 Weighting) [paper](https://arxiv.org/abs/2204.00227)
- Pseudo Numerical Methods for Diffusion Models on Manifolds (PNMDM) [paper](https://arxiv.org/abs/2202.09778)
- The DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps [paper](https://arxiv.org/pdf/2206.00927)
- Scalable Diffusion Models with Transformers (DiT) [paper](https://arxiv.org/abs/2212.09748)
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3) [paper](https://arxiv.org/abs/2403.03206)
- Flow Matching for Generative Modeling [paper](https://arxiv.org/abs/2210.02747)
- Efficient Diffusion Training via Min-SNR Weighting Strategy [paper](https://arxiv.org/abs/2303.09556)
- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) [paper](https://arxiv.org/abs/2301.08243)
- Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA) [paper](https://arxiv.org/abs/2404.08471)
- Simplified State Space Layers for Sequence Modeling (S5) [paper](https://arxiv.org/abs/2208.04933)
- Applying Guidance in a Limited Interval Improves Sample and Distribution Quality (interval-limited CFG) [paper](https://arxiv.org/abs/2404.07724)
- Diffusion-LM Improves Controllable Text Generation (sqrt schedule) [paper](https://arxiv.org/abs/2205.14217)

## Libraries this is built on
- [JAX](https://github.com/jax-ml/jax) and [Flax](https://github.com/google/flax) (Linen). All of the models and the trainer are written in these.
- [Optax](https://github.com/google-deepmind/optax) for the optimizers and schedules, [Orbax](https://github.com/google/orbax) for checkpoints, and [Grain](https://github.com/google/grain) for the data pipeline.
- [tyro](https://github.com/brentyi/tyro), which turns the config dataclasses into the command line, and [Weights & Biases](https://github.com/wandb/wandb) for tracking runs.
- [albumentations](https://github.com/albumentations-team/albumentations) for augmentation, [OpenCV](https://github.com/opencv/opencv-python) for decoding, and [TensorFlow Datasets](https://github.com/tensorflow/datasets) for the public datasets.
- [transformers](https://github.com/huggingface/transformers) for the CLIP and audio encoders, and [safetensors](https://github.com/huggingface/safetensors) for interop.

## Useful blogs and codebases
- An incredible series of blogs on various diffusion related topics by [Sander Dieleman](https://sander.ai/posts/). The posts particularly on [diffusion models](https://sander.ai/2022/01/31/diffusion.html), [Typicality](https://sander.ai/2020/09/01/typicality.html), [Geometry of Diffusion Guidance](https://sander.ai/2023/08/28/geometry.html#warning) and [Noise Schedules](https://sander.ai/2024/06/14/noise-schedules.html) are a must read
- An awesome blog series by Tony Duan on [Diffusion models from scratch](https://www.tonyduan.com/diffusion/index.html). Although it trains models for MNIST and the implementations are a bit basic, the maths is explained in a very nice way. The codebase is [here](https://github.com/tonyduan/diffusion)
- The [k-diffusion](https://github.com/crowsonkb/k-diffusion/) codebase by Katherine Crowson, which hosts an exhaustive implementation of the EDM paper (Karras et al) along with the DPM-Solver, DPM-Solver++ (both 2S and 2M) in pytorch. Most other diffusion libraries borrow from this.
- The [Official EDM implementation](https://github.com/NVlabs/edm) by Tero Karras, in pytorch. Really neat code and the reference implementation for all the karras based samplers/schedules.
- The [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers). The vendored Flax VAE and parts of the attention module derive from it (Apache-2.0, attribution headers preserved).
- [jax-fid](https://github.com/matthias-wright/jax-fid), the origin of the vendored InceptionV3 used for FID.
- [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) and [facebookresearch/jepa](https://github.com/facebookresearch/jepa), the reference I-JEPA and V-JEPA code which the masking and the probes follow.
- The [Keras DDPM Tutorial](https://keras.io/examples/generative/ddpm/) by A_K Nain, and the [Keras DDIM implementation](https://keras.io/examples/generative/ddim/) by András Béres, which are great starting points for beginners to understand the basics of diffusion models. I started my journey by trying to implement the concepts introduced in these tutorials from scratch.

## Related projects
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [Levanter](https://github.com/stanford-crfm/levanter) are mature JAX trainers for language models and worth looking at if language models are all you need. Dew is a lot smaller, and also covers diffusion and JEPA.
- [verl](https://github.com/verl-project/verl) (RL post-training) and [vLLM](https://github.com/vllm-project/vllm) (serving) both read Hugging Face model directories, and `dew.interop.save_hf_layout` writes that pair of files. It renames nothing: the names on disk are the module names in the tree and the config is dew's own, so reading an export in either of them still needs a per-family translation.
