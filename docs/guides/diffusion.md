# Diffusion training

`DiffusionObjective` trains a denoising model under a diffusion `Process`: it noises each image to a random time, runs the model and computes the process's loss. Its `evaluate` samples images with a solver. This page trains a small flow-matching DiT on synthetic images; the data, the model and the steps are tiny so the example runs on a CPU in seconds, and the samples say nothing about image quality. [Diffusion processes and solvers](../concepts/diffusion.md) describes the processes, presets and solvers.

## Example

The example makes eight striped 8×8 images, trains for three steps and samples one image per batch row.

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
objective = DiffusionObjective(model, Flow(), InputSpec(Field("image", (8, 8, 3))),
                               solver=Euler(), steps=4)
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

The image batch is NHWC (batch, height, width, channels) uint8 in `[0, 255]`; the objective converts it to the model's range `[-1, 1]`. `InputSpec(Field("image", (8, 8, 3)))` describes one sample, without the batch dimension. The DiT cuts each 8×8 image into 4×4 patches, four tokens per image.

## Process and solver

`Flow()` is a preset, a frozen dataclass of the numbers that define rectified flow. The objective builds its `Process` once, which draws noise and times, builds the training target and converts the model's output. `objective.process` exposes it for direct sampling or schedule inspection. A custom `Process` can be passed in the same position.

`DiffusionObjective(steps=4)` is the number of solver steps `evaluate` samples with; `trainer.fit(..., steps=3)` is the number of training steps.

`Euler()` is the solver `evaluate` uses. The solver can change without retraining; the process cannot, because it defines what the model learned to predict.

## Samples

`evaluate` returns an image artifact whose `images` are float32 in `[-1, 1]`, one sample per row of the batch it is given; `(output + 1) / 2` maps them to `[0, 1]` for display.

The example passes `evaluate` its own key. When `Trainer.fit` schedules evaluation, the keys come from the run key and the step. [Evaluation and tracking](evaluation.md) covers scheduled evaluation and the image metrics (FID, CLIP score).

## Conditions and latent encoders

For text conditioning, `InputSpec.conditions` maps a model keyword argument to a condition encoder and the batch field it reads. Captions must be tokenized with the encoder and tokenizer settings used in training. A pretrained text tower is downloaded on first use.

With an autoencoder configured, training runs on latents instead of pixels. The denoising model's channel count and spatial shape must match the encoder's output, and the encoder's scaling convention must be kept.

`DiffusionRunConfig.autoencoder` is a `PretrainedAutoencoder`, which builds the autoencoder its checkpoint's config names: a Stable Diffusion `AutoencoderKL`, SANA's deep compression autoencoder (`AutoencoderDC`; the f32c32 checkpoints downsample 32 times into 32 channels), Wan 2.1's causal video VAE (`AutoencoderKLWan`, read from a pipeline's `vae/`), or a representation autoencoder (`AutoencoderRAE`). A DC-AE, Wan or RAE repository takes `revision="main"` or a commit; `bf16` and `flax` name the SD1-era Flax layouts:

```python
from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder

config = DiffusionRunConfig(autoencoder=PretrainedAutoencoder(
    "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", revision="main"))
```

On the published SANA 1.1 weights and a 256x384 batch, the DC-AE port matches diffusers 0.34.0's `AutoencoderDC` to 9e-6 of the largest latent value and 3e-6 of the largest decoded pixel.

The Wan VAE compresses time as well as space: a clip of 1 + 4k frames encodes to 1 + k latent frames, each 8 times smaller on a side with 16 channels, and a `VideoDataset` run's clips need that length. Each frame reads only the frames before it, so the first frame encodes alone and an image is a one-frame clip. On `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and a 9-frame 128x192 clip, the port matches diffusers 0.34.0's `AutoencoderKLWan`: the largest difference, divided by the larger of 1 and the largest reference value, is 1.5e-6 for the latent and 8.8e-6 for the decoded pixels (`tests/test_wan_vae.py`, a `network` test).

A representation autoencoder (RAE) encodes with a frozen pretrained vision encoder, DINOv2 with registers, SigLIP or ViT-MAE, and decodes with a ViT trained to paint the image back. Its latent is the encoder's patch tokens: the `nyu-visionx` RAEs turn any image, resized to the encoder's input, into a 16x16 grid of 768 channels, normalized per position, and decode it to 256x256 pixels. Their encoders stay frozen in a run like any autoencoder here. The dataset's image size only sets what the encoder resizes from; train at the size the decoder paints. On the three published checkpoints (`RAE-dinov2-wReg-base-ViTXL-n08`, `RAE-siglip2-base-p16-i256-ViTXL-n08`, `RAE-mae-base-p16-ViTXL-n08`), a 256x256 image and a standard normal latent, compared over the first 32 of the latent's 768 channels and a 64x64 corner of the decode, the port is held to diffusers 0.40.0's `AutoencoderRAE` (with transformers 4.57.1) run in float64. The largest difference, divided by the larger of 1 and the largest reference value, is 9.8e-6, 1.8e-5 and 1.1e-6 for the DINOv2, SigLIP and MAE latents, where the source's own float32 runs land 7.9e-6, 2.8e-5 and 1.9e-6 away, and at most 4e-6 for the decoded pixels (`tests/test_rae.py`, a `network` test). SigLIP's residual stream reaches several hundred, so float32 rounding alone moves its latent that far.

To condition on sound, `HFAudio` (`hf_audio`) runs a transformers audio model, such as wav2vec2 or Whisper's encoder, through torchax (the `torchax` extra) and hands the model its last hidden states under the same `textcontext` keyword. A run selects it with `DiffusionRunConfig(data=LocalVideos(...), text=None, audio=AudioCondition())`, or `text:None audio:audio-condition` on the command line. The video dataset's `audio_model` names the tower, its clips' `audio` field carries the extractor's input, and the clip length sets the waveform length every clip and the silent unconditional input are encoded at. Sampling takes one `{"audio": waveform}` record per sample, mono at the extractor's rate.

## Fine-tuning a published pipeline

`PretrainedPipeline.load(name_or_dir)` reads a published pipeline in the diffusers layout (SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image, Z-Image, or Wan 2.1 for text to video) into a bundle: the denoiser as `model`, its process, its text conditioning and its autoencoder. `pipe.diffusion_objective(**options)` builds a `DiffusionObjective` that starts from the pipeline's weights and trains the whole denoiser. The text encoders and the autoencoder stay frozen. Evaluation samples the way the pipeline does, with its own solver, step count and guidance, unless you pass `solver=`, `steps=` or `guidance=`. Training batches carry uint8 NHWC images at the pipeline's resolution, or a video pipeline's uint8 `[N, frames, H, W, 3]` clips under `video`, and `objective.inputs.tokenize(captions)`. The recipe's `--pretrained` flag runs the same full fine-tuning from the command line.

`pipe.lora(rank=, modules=, key=)` returns the same kind of bundle with a fresh low-rank adapter (LoRA) on the denoiser projections that `modules` names, the way Diffusers' `target_modules` does. `to_q`, `to_k`, `to_v` and `to_out.0` are the attention projections of each family's transformer and of the SD UNet. Names match the denoiser alone, so a text encoder is never adapted. B starts at zero, so the adapted pipeline samples exactly what the source does until it trains. Its `diffusion_objective` trains the adapter's factors and keeps every other weight frozen:

<!-- not run: downloads FLUX.1-schnell and needs an image dataset -->
```python
import jax
import optax

from dew.interop import PretrainedPipeline
from dew.training import Trainer

pipe = PretrainedPipeline.load("black-forest-labs/FLUX.1-schnell", dtype="bfloat16")
key = jax.random.key(0)
tuned = pipe.lora(rank=16, modules=("to_q", "to_k", "to_v", "to_out.0"), key=key)
objective = tuned.diffusion_objective()
state = Trainer(objective, optax.adamw(1e-4), key=key).fit(data, steps=1000)
tuned.adapter.save(state.params, "flux-adapter")
tuned.save("flux-merged", variables=state.params)
images = objective.pipeline(state)(["a red bird"], key=0).host().images
```

`tuned.adapter.save` writes `pytorch_lora_weights.safetensors`, the denoiser's PEFT config in its header, which Diffusers' `load_lora_weights` reads for that family. `tuned.save` writes the whole pipeline in the diffusers layout with the factors merged into the kernels. Both take the trainer's `state.params` as it comes back. `LoRA.load(pipe.model, pipe.variables, pipe.layouts, path)` reads such a file back, or one Diffusers or a PEFT trainer wrote.

<!-- not run: needs torch, diffusers and peft -->
```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
pipe.load_lora_weights("flux-adapter")
```

`DiffusionObjective(..., trainable=)` takes the same filter for code that builds the objective itself. It chooses among the denoiser's own leaves, so it trains the plain denoising loss: a loss head (`uncertainty`, `alignment`, `end_to_end`) or an objective with a loss of its own (MeanFlow, shortcut, distillation, Flow-GRPO) refuses one.

[Recipes](../recipes.md) runs diffusion training on real datasets from the command line. [Supported models](../models.md) lists the published diffusion checkpoints that load.
