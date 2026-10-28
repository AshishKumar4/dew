# Coming from FlaxDiff

Dew started as a fork of FlaxDiff, my earlier framework built around diffusion. Dew separates model construction, data loading, objectives and training, so the same trainer also runs language models and representation learning. This page maps FlaxDiff's modules and capabilities to Dew's and describes the one FlaxDiff checkpoint format Dew loads. Dew's names are not import aliases; code does not migrate by search and replace.

## Module map

| FlaxDiff area | Dew area | Change to account for |
| --- | --- | --- |
| `flaxdiff.models` | `dew.nn.backbones` | Build the model from its class with its current fields; old constructor fields need review. |
| `flaxdiff.schedulers` and `flaxdiff.predictors` | `dew.diffusion.schedules` and `dew.diffusion.transforms` | A diffusion preset composes schedule, target, weighting, and preconditioning choices. A schedule alone does not describe the whole training convention. |
| The diffusion trainer in `flaxdiff.trainer` | `dew.Trainer` plus `dew.objectives.diffusion.DiffusionObjective` | The objective computes the diffusion-specific parts; the trainer runs the updates, places the state, logs and manages checkpoints. |
| `flaxdiff.jepa` | `dew.objectives.jepa` and `dew.nn.backbones.jepa` | The objective and its modules implement the encoder, the predictor and the target. |
| `flaxdiff.samplers` and `flaxdiff.inference` | `dew.sampling` | Construct sampling with the model's training process and compatible conditions. |
| `flaxdiff.metrics` | `dew.eval` | Choose metrics that consume the current objective's evaluation outputs. |
| `training.py` and `training_jepa.py` | `recipes/diffusion/train.py` and `recipes/jepa/train.py` | Recipes use dataclass configurations and generated CLI flags instead of the old argparse interface. |

[Custom objectives](concepts/objectives.md) describes the contract every objective follows, [Diffusion training](guides/diffusion.md) the diffusion pieces, and [Recipes](recipes.md) the command-line configuration.

## What FlaxDiff did, and where it is in Dew

Each row names what FlaxDiff provided, Dew's version of it, and the test that holds it. Where Dew's version is only a new name for the same thing, the old spelling was dropped on purpose; where Dew has no version yet, the row says so.

| FlaxDiff | Dew | Held by |
|---|---|---|
| Linear, cosine, square-root, Karras VE and EDM schedules | `dew.diffusion.schedules`, and the `EDM`, `Karras`, `Cosine` and `Flow` presets in place of `get_diffusion_preset` | `tests/test_schedulers.py` (published beta tables, sigma grids, inversions) |
| Flow matching, logit-normal draws, resolution shift | `FlowMatchingScheduler`, `Flow`, `ResolutionShift` | `tests/test_flow_matching.py` (SD3 and Flux losses and gradients) |
| The exponential schedule | `ExpNoiseScheduler` and `exp_beta_schedule`, its table clipped at its own `beta_end`; Diffusers' exponential table stays `published_betas(schedule="exp")` | `tests/test_schedulers.py`: the betas bit for bit, the rates and P2 weights on par with FlaxDiff's (`tools/flaxdiff_schedules_reference.py`) |
| The continuous VP cosine and the tangent-grid cosine schedules | `CosineContinuousNoiseScheduler`; `CosineGeneralNoiseScheduler`, with FlaxDiff's weight, 1 + sigma^2 / sigma_max^2, its conditioning on whole thousandths of t and its sigma_data of 1. With a kappa other than 1 the grid ends at sigma_max / kappa^2, as FlaxDiff's does: `sigma_range` names those ends, which solvers keep to, and `sigma_min` and `sigma_max` hold the arguments | `tests/test_schedulers.py`: rates, sigmas and weights on par with FlaxDiff's, its conditioning exactly its own |
| Epsilon, x0, v, velocity and EDM-preconditioned targets | `dew.diffusion.transforms` | `tests/test_predictors.py` |
| DDPM, DDIM, Euler, Euler ancestral, Heun, RK4 and multistep DPM samplers | `dew.sampling.solvers` | `tests/test_samplers.py`; RK4 and multistep DPM against FlaxDiff's own latents and gradients |
| Classifier-free guidance, also over an interval | `dew.sampling.guidance.CFG` | `tests/test_guidance.py` |
| UNet, UNet3D, SimpleDiT, SimpleUDiT, SimpleMMDiT, HierarchicalMMDiT, VideoDiT | `dew.nn.backbones`, with Hilbert and zigzag scans as a `scan_order` field in place of name suffixes | `tests/test_flaxdiff_family_source.py`: outputs and every gradient against FlaxDiff's source |
| The SSM-attention hybrid DiT | `HybridSSMAttentionDiT` | `tests/test_flaxdiff.py`, on an old run's weights |
| UViT | `UViT`, now following published U-ViT | `tests/test_uvit_source.py`; the old UViT's own options are not carried over |
| SimpleAutoEncoder, the learned-variance head (`learn_sigma`), the `diffusers_unet_simple` wrapper | Not in Dew yet | |
| The Stable Diffusion VAE, latent scale and shift, and a VAE trained from scratch | `StableDiffusionVAE`, `AutoencoderKL` | `tests/test_autoencoder.py`, `tests/test_vae.py` |
| I-JEPA and V-JEPA with linear and kNN probes | `dew.objectives.jepa`, `dew.nn.backbones.jepa` | `tests/test_jepa.py`, `tests/test_jepa_source.py` |
| Grain, TFDS, ArrayRecord, URL and video sources, augmentation and caption dropout | `dew.data` chains and presets, in place of the named dataset maps and `FLAXDIFF_AUGMENT_MODE` | `tests/test_data.py`, `tests/test_online_loader.py`, `tests/test_augmenters.py` |
| CLIP text and audio encoders | `dew.inputs.encoders.CLIPText`, `HFAudio` | `tests/test_text_encoders.py`, `tests/test_audio_condition.py` |
| The diffusion trainer: data parallel and FSDP, accumulation, EMA, mixed precision, checkpoints | `dew.Trainer` with `DiffusionObjective` | `tests/test_parallelism.py`, `tests/test_training_references.py`, `tests/test_tree_checkpoints.py` |
| Weights & Biases logging and registry publishing | `WandbTracker`, `dew.io.publish` | `tests/test_reporting.py`, `tests/test_publish.py` |
| Ranking a run against its sweep's or project's best and promoting it to the registry | `dew.io.promote`, `trainer.promote`; the trainer still never deletes a checkpoint | `tests/test_publish.py` |
| `from_wandb_run` and `from_wandb_registry` | `dew.pipeline("wandb:entity/project/run_id")`, `dew.pipeline("wandb:wandb-registry-model/name:best")` | `tests/test_publish.py` |
| Ranking a sweep's runs and promoting the best, and loading by W&B run name | Not in Dew yet: publish a chosen checkpoint and load it with `TextToImage.from_run` | |
| FID, CLIP score, PSNR, SSIM | `dew.eval` | `tests/test_fid.py`, `tests/test_clip_metrics.py`, `tests/test_image_metrics.py` |
| TPU create, setup and launch | `dew.cli.tpu`, `dew launch` | `tests/test_tpu_cli.py`, `tests/test_launch.py` (command contracts, not a TPU pod run) |

## Loading an older text-to-image checkpoint

`TextToImage.from_flaxdiff` loads `simple_udit` and `hybrid_dit` latent text-to-image models from the older checkpoint format, trained on the Stable Diffusion VAE with a CLIP text encoder. It returns a `TextToImage`, the same task `dew.pipeline` returns for a Dew run.

It takes three things from the old run:

| Argument | Where it comes from |
|---|---|
| Checkpoint directory | One step directory written by the older trainer, with a `default/` folder inside |
| `config` | The run's saved training config, as JSON; it names the architecture and sizes, which the checkpoint does not |
| `jax_version` | The JAX version in the run's `requirements.txt`. The older code drew the random frequencies of its time embedding from `jax.random`, whose stream changed in JAX 0.5.0, and did not save them, so the loader draws them the way that version did. |

<!-- not run: needs an older checkpoint and its training config -->
```python
import json

from dew.inference import TextToImage
from dew.sampling import Heun

config = json.load(open("run_config.json"))  # the run's saved training config
pipe = TextToImage.from_flaxdiff("checkpoints/350000", config, jax_version="0.5.3")
images = pipe(["a lighthouse on a rocky coast"], key=0, steps=25, solver=Heun()).host().images
```

`images` is a float array in `[-1, 1]`, `[prompts, 256, 256, 3]` for a 256px run. The text tower (CLIP ViT-L/14) and the VAE download from the Hugging Face Hub under the names the config records. By default a call samples the way the older trainer previewed its runs, with Euler ancestral over 200 steps of that trainer's own time grid and classifier-free guidance 3. Passing another `steps` samples the same grid at that count. `tests/test_flaxdiff.py` compares a call with the older package's own preview sampling of the same run, run from its source at the pinned commit. The loader reads the averaged (EMA) weights of the last state; `ema=False` and `best=True` choose the others. It builds the matching Dew architecture with `adaln_silu=False` and `text_pooling="all"`, the two places where the blocks of the `flaxdiff` 0.2 package differ from Dew's defaults, and `tests/test_flaxdiff.py` checks its output against that package's own code.

Anything else, including the older UNets and the 2024 checkpoints, has no loader. Keep each of those runs together with the source revision, environment, data and encoder files that produced it.

## FlaxDiff's named datasets

FlaxDiff named its corpora in three maps (`datasetMap`, `onlineDatasetMap`, `mediaDatasetMap`). Dew keeps no names. Each kept corpus is a chain.

| FlaxDiff name | Dew |
|---|---|
| `oxford_flowers102` | `TFDS("oxford_flowers102", split="all", version="2.1.1").pipe(presets.images, 256, batch=64, caption=images.caption(templates=FLOWER_TEMPLATES))` |
| `voxceleb2` | `Files(f"{root}/train/**/*.mp4", format="files").map(speaking).pipe(presets.clips, 16 / 25, frames=16, fps=25, size=256, batch=8, sample_rate=16000, margin=3 / 25, context=3 / 25, missing="reject", rejects=False)` |

`FLOWER_TEMPLATES` are FlaxDiff's five: "a photo of a {}", "a photo of a {} flower", "This is a photo of a {}", "This is a photo of a {} flower", "A photo of a {} flower". `speaking` gives each clip FlaxDiff's caption:

```python
def speaking(row):
    return {"mp4": row["path"], "caption": "a video of a person speaking"}
```

The other 16 names read buckets that no longer exist (`gs://msml612-diffusion-data`, `gs://flaxdiff-datasets-regional`, `gs://flaxdiff-datasets`). Copies of them read with these chains:

- ArrayRecord corpora (`laion12m_coco`, `laion2b_aesthetic`, `diffusiondb`, `cc3m`, `cc12m`, `laiona_coco`, `aesthetic_coyo`): `Files("<dir>/*.array_record").filter_map(images.DecodeImage()).map(images.Resize(256))`.
- Combined corpora (`combined_msml612`, `combined_aesthetic`, `laiona_coco_coyo`, `combined_30m`): `concat` of those stores, in FlaxDiff's order, a store listed twice read twice.
- `combined_online`: `concat` of the `url` and `caption` catalogs with FlaxDiff's repeats, then `.shuffle(0).filter_map(images.fetch(timeout=15)).filter_map(images.DecodeImage()).filter(images.MinSide(128)).filter(images.MaxAspect(2.4)).filter(images.NotFlat()).map(images.Fit(256))`. A fetch is tried once.

## New runs

A Dew run starts from a current configuration in a new output directory. A recipe writes `run.json` next to its checkpoints. The checkpoint state holds the live variables, the optimizer state, the EMA or reference tree when there is one, the step counters, and the random key. The parameter names and the state structure must match the model you rebuild. Renaming a checkpoint directory or changing the package you import does not convert what is inside it. [Checkpoints](guides/checkpoints.md) describes saving and restoring.

## Older results

The [Gallery](gallery.md) shows samples from models trained with Dew, with their recorded settings. The [benchmark page](benchmarks.md) lists the revision, environment, and hardware for each measured Dew run.

Dew still contains code adapted from the earlier project and ideas taken from other research. [Papers and attribution](references.md) lists Diffusers, jax-fid, JEPA, and the other upstream sources. Their attribution and license terms still apply after the code moved into different modules.
