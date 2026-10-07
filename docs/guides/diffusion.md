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

from dew import Field, InputSpec, Trainer
from dew.config import OptimConfig
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
trainer = Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
info = Step(step=state.step, key=jax.random.key(1), ema=state.averaged)
preview = objective.evaluate(state.variables, batch, info)
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

`DiffusionRunConfig.autoencoder` is a `PretrainedAutoencoder`, which builds the autoencoder its checkpoint's config names:

- a Stable Diffusion `AutoencoderKL`;
- SANA's deep compression autoencoder, `AutoencoderDC` (the f32c32 checkpoints downsample 32 times into 32 channels);
- Wan 2.1's causal video VAE, `AutoencoderKLWan`, read from a pipeline's `vae/`;
- a representation autoencoder, `AutoencoderRAE`.

A DC-AE, Wan or RAE repository takes `revision="main"` or a commit; `bf16` and `flax` name the SD1-era Flax layouts:

```python
from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder

config = DiffusionRunConfig(autoencoder=PretrainedAutoencoder(
    "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", revision="main"))
```

On the published SANA 1.1 weights and a 256x384 batch, the DC-AE port matches diffusers 0.34.0's `AutoencoderDC` to 9e-6 of the largest latent value and 3e-6 of the largest decoded pixel.

The Wan VAE compresses time as well as space: a clip of 1 + 4k frames encodes to 1 + k latent frames, each 8 times smaller on a side with 16 channels, and a `VideoDataset` run's clips need that length. Each frame reads only the frames before it, so the first frame encodes alone and an image is a one-frame clip. On `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and a 9-frame 128x192 clip, the port matches diffusers 0.34.0's `AutoencoderKLWan`: the largest difference, divided by the larger of 1 and the largest reference value, is 1.5e-6 for the latent and 8.8e-6 for the decoded pixels (`tests/test_wan_vae.py`, a `network` test).

A representation autoencoder (RAE) encodes with a frozen pretrained vision encoder, DINOv2 with registers, SigLIP or ViT-MAE, and decodes with a ViT trained to reconstruct the image. Its latent is the encoder's patch tokens. The `nyu-visionx` RAEs resize any image to the encoder's input, encode it into a 16x16 grid of 768 channels normalized per position, and decode that back to 256x256 pixels. Their encoders stay frozen in a run, like every autoencoder here. The dataset's image size only sets what the encoder resizes from, so train at the decoder's output size.

`tests/test_rae.py` (a `network` test) compares the port with diffusers 0.40.0's `AutoencoderRAE` (with transformers 4.57.1) run in float64, on the three published checkpoints (`RAE-dinov2-wReg-base-ViTXL-n08`, `RAE-siglip2-base-p16-i256-ViTXL-n08`, `RAE-mae-base-p16-ViTXL-n08`), a 256x256 image and a standard normal latent, over the first 32 of the latent's 768 channels and a 64x64 corner of the decode. The largest difference, divided by the larger of 1 and the largest reference value, is 9.8e-6, 1.8e-5 and 1.1e-6 for the DINOv2, SigLIP and MAE latents, against 7.9e-6, 2.8e-5 and 1.9e-6 for the source's own float32 runs, and at most 4e-6 for the decoded pixels. SigLIP's residual stream reaches several hundred, so float32 rounding alone moves its latent that far.

To condition on sound, `HFAudio` (`hf_audio`) runs a transformers audio model, such as wav2vec2 or Whisper's encoder, through torchax (the `torchax` extra) and passes the model's last hidden states to the denoiser under the same `textcontext` keyword. A run selects it with `DiffusionRunConfig(data=LocalVideos(...), text=None, audio=AudioCondition())`, or `text:None audio:audio-condition` on the command line. The video dataset's `audio_model` names the tower, its clips' `audio` field carries the extractor's input, and the clip length sets the waveform length every clip and the silent unconditional input are encoded at. Sampling takes one `{"audio": waveform}` record per sample, mono at the extractor's rate.

## Fine-tuning a published pipeline

`PretrainedPipeline.load(name_or_dir)` reads a published pipeline in the diffusers layout (SD 1.x/2.x/XL, SD3, Flux, FLUX.2, Qwen-Image, Z-Image, or Wan 2.1 for text to video) into a bundle that holds the denoiser as `model`, its process, its text conditioning and its autoencoder. `DiffusionObjective(pipe, **options)` trains the pipeline from its weights. It reads the denoiser, the process, the conditions, the autoencoder and the sampling policy from the bundle, and any keyword you pass overrides the bundle's value. It trains the whole denoiser, while the text encoders and the autoencoder stay frozen. Evaluation samples the way the pipeline does, with its own solver, step count and guidance, unless you pass `solver=`, `steps=` or `guidance=`. Training batches hold uint8 NHWC images at the pipeline's resolution, or a video pipeline's uint8 `[N, frames, H, W, 3]` clips under `video`, and `objective.inputs.tokenize(captions)`. The recipe's `--pretrained` flag runs the same full fine-tuning from the command line.

A flow pipeline (SD3, Flux, FLUX.2, Qwen-Image, Z-Image, Wan) trains on the convention of the SD3 paper (Esser et al., 2024), not on the defaults of Diffusers' training scripts. Training times are drawn from a logit-normal distribution (mean 0, std 1), continuous on [0, 1]. The shift in the pipeline's sampler maps each time to a noise level: SD3's static 3.0, or Flux's exp(mu) at the data's token count. The loss is the velocity error at unit weight. Diffusers 0.34's DreamBooth scripts differ in three ways:

- They index a 1000-entry table of noise levels instead of drawing a continuous time.
- `train_dreambooth_flux.py` defaults to `--weighting_scheme none`. That draws uniformly over Flux's training table, which stays unshifted because its scheduler shifts only at sampling time.
- `train_dreambooth_sd3.py` defaults to `--precondition_outputs 1`. It scores the clean latent it recovers from the predicted velocity, which weights the velocity error by sigma squared. The paper and Dew score the velocity itself, which is the script's `--precondition_outputs 0`.

To train on a script's default draw instead, pass the preset that reproduces it: `Flow(shift=1.0, density="uniform")` for the Flux script, or `Flow(shift=3.0)` (Dew's default for SD3) for the SD3 script. Either one reaches every table index the script draws, so the script's noise levels are the table at the preset's times, within one table step of the preset's own, and its loss weights are equal (`tests/test_diffusion_run_sources.py`).

To fine-tune with a low-rank adapter (LoRA) instead, `pipe.adapt(LoRA(rank=, modules=), key=)` returns the same kind of bundle with a fresh adapter on the denoiser projections that `modules` names, the way Diffusers' `target_modules` does. `to_q`, `to_k`, `to_v` and `to_out.0` are the attention projections of each family's transformer and of the SD UNet. The names are matched against the denoiser only, so a text encoder is never adapted. B starts at zero, so the adapted pipeline samples exactly what the source does until it trains. An objective over the adapted bundle trains the adapter's factors and keeps every other weight frozen:

<!-- not run: downloads FLUX.1-schnell and needs an image dataset -->
```python
import jax

from dew.config import OptimConfig
from dew.interop import PretrainedPipeline
from dew.lora import LoRA
from dew.training import Trainer

pipe = PretrainedPipeline.load("black-forest-labs/FLUX.1-schnell", dtype="bfloat16")
tuned = pipe.adapt(LoRA(rank=16, modules=("to_q", "to_k", "to_v", "to_out.0")), key=0)
objective = DiffusionObjective(tuned)
state = Trainer(objective, OptimConfig(learning_rate=1e-4), key=0).fit(data, steps=1000)
tuned.adapter.save(state.variables, "flux-adapter")
tuned.save("flux-merged", variables=state.variables)
images = objective.pipeline(state)(["a red bird"], key=0).host().images
```

`tuned.adapter.save` writes `pytorch_lora_weights.safetensors`, with the denoiser's PEFT config in its header, which Diffusers' `load_lora_weights` reads for that family. `tuned.save` writes the whole pipeline in the diffusers layout with the factors merged into the kernels. Both take the trainer's `state.variables` as they come back from `fit`. `LoRA.load(pipe.model, pipe.variables, path, layouts=pipe.layouts)` reads such a file back, as well as one that Diffusers or a PEFT trainer wrote.

A recipe run takes the same spec as `--lora.rank 16 --lora.modules to_q to_k to_v to_out.0` next to `--pretrained`. Without `--pretrained`, a run from scratch puts the adapter on a fresh draw of the denoiser from the run's key. `Adapter.from_run(run)` rebuilds a run's adapter from the run alone, so `adapter.save(adapter.variables, path)` writes the same file.

<!-- not run: needs torch, diffusers and peft -->
```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
pipe.load_lora_weights("flux-adapter")
```

A loss head that the objective adds (`uncertainty`, `alignment`, `end_to_end`) trains alongside the factors. To train part of a denoiser without an adapter, split its starting variables with `dew.objectives.base.freeze(variables, filter)` and pass them as `variables=`. The leaves the filter keeps are trained and the rest stay frozen.

[Recipes](../recipes.md) runs diffusion training on real datasets from the command line. [Supported models](../models.md) lists the published diffusion checkpoints that load.
