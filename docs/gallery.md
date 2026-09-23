# Gallery: historical FlaxDiff experiments

These images come from my earlier experiments with FlaxDiff, the project Dew grew out of. The settings below copy the run descriptions I recorded at the time, including the old scheduler and model names. You cannot run them as configurations with today's Dew API, and they do not show that a current checkout reproduces these images.

For each run the gallery records the training data, the image size, the sampling settings, and some model fields. It has no complete environment, checkpoint, seed record, or quality evaluation for any of them. For a current workflow, start with [recipes](recipes.md) and the [diffusion guide](guides/diffusion.md). For timed measurements with their revision and hardware, see [benchmarks](benchmarks.md).

## Text-to-image on a mixed captioned dataset

This model trained on LAION-Aesthetics 12M, CC12M, MS COCO, and a one-million-image subset of COYO-700M with aesthetic score 6 or higher, on a TPU-v4-32 slice. Sampling used Euler ancestral sampling for 200 steps with classifier-free guidance (CFG). CFG mixes the model's conditional and unconditional predictions to control how closely sampling follows the text.

Every image in the grid used the same prompt, "a beautiful landscape with a river with mountains." My record does not give the guidance scale for this grid.

| Setting | Recorded value |
| --- | --- |
| Batch size | 256 |
| Image size | 128 × 128 |
| Training epochs | 5 |
| Steps per epoch | 74,573 |
| Feature depths | `[128, 256, 512, 1024]` |
| Training noise schedule | `EDMNoiseScheduler` |
| Inference noise schedule | `KarrasVENoiseScheduler` |

![Historical text-to-image landscape grid using Euler ancestral sampling and CFG](assets/gallery/medium_epoch5.png)

## Text-to-image on Oxford Flowers

This run used Oxford Flowers 102, Euler ancestral sampling for 200 steps, and CFG scale 2. The prompts, in grid order, were:

> water tulip; a water lily; a water lily; a water lily; a photo of a marigold; a water lily; a water lily; a photo of a lotus; a photo of a lotus; a photo of a lotus; a photo of a rose; a photo of a rose; a photo of a rose; a photo of a rose; a photo of a rose

| Setting | Recorded value |
| --- | --- |
| Batch size | 16 |
| Image size | 128 × 128 |
| Training epochs | 1,000 |
| Steps per epoch | 511 |
| Training noise schedule | `EDMNoiseScheduler` |
| Inference noise schedule | `KarrasVENoiseScheduler` |

![Historical Oxford Flowers text-to-image grid using Euler ancestral sampling at guidance scale 2](assets/gallery/text2img_euler_ancestral_1.png)

## Unconditional Oxford Flowers with DDPM

An unconditional model generates images without a text prompt. This grid used DDPM sampling for 1,000 steps, with `CosineNoiseScheduler` for both training and inference.

| Setting | Recorded value |
| --- | --- |
| Dataset | Oxford Flowers 102 |
| Batch size | 16 |
| Image size | 64 × 64 |
| Training epochs | 1,000 |
| Steps per epoch | 511 |
| Embedding features | 256 |
| Feature depths | `[64, 128, 256, 512]` |
| Attention configuration | Five entries, each `{"heads": 4}` |
| Residual blocks | 2 |
| Middle residual blocks | 1 |

The attention list and the feature-depth list copy the old record. They are not arguments you can pass to the current UNet.

![Historical unconditional Oxford Flowers grid using 1000-step DDPM sampling](assets/gallery/ddpm2.png)

## Unconditional Oxford Flowers with Heun

This grid used a 10-step Heun sampler. Heun takes a prediction step and then a correction on each sampling interval, so the exact number of network evaluations depends on how the solver handles the last step. The old caption said 20 model evaluations, but I kept no trace that confirms that count.

| Setting | Recorded value |
| --- | --- |
| Dataset | Oxford Flowers 102 |
| Batch size | 16 |
| Image size | 64 × 64 |
| Training epochs | 1,000 |
| Steps per epoch | 511 |
| Training noise schedule | `EDMNoiseScheduler` |
| Inference noise schedule | `KarrasVENoiseScheduler` |

![Historical unconditional Oxford Flowers grid using 10-step Heun sampling](assets/gallery/heun.png)

Do not read these grids as a controlled comparison of samplers. The records do not show that they used the same checkpoints, seeds, or training settings. If you prepare a new run with any of these datasets, check its license and access conditions; the gallery does not redistribute the datasets. [References and attribution](references.md) lists the research and the upstream implementations behind these methods.
