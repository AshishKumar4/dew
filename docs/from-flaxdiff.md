# Coming from FlaxDiff

Dew started as a fork of FlaxDiff, my earlier diffusion-only framework. Dew separates model construction, data loading, objectives and training. That separation lets the same trainer run language models and representation learning. Below is a map of FlaxDiff's modules to Dew's and the one FlaxDiff checkpoint format Dew can load. The new names are not import aliases. You need to review the calling code when migrating; search and replace is not enough.

## Module map

| FlaxDiff area | Dew area | Change to account for |
| --- | --- | --- |
| `flaxdiff.models` | `dew.nn.backbones` | Build the model from its class with its current fields; old constructor fields need review. |
| `flaxdiff.schedulers` and `flaxdiff.predictors` | `dew.diffusion.schedules` and `dew.diffusion.transforms` | A diffusion preset composes schedule, target, weighting, and preconditioning choices. A schedule alone does not describe the whole training convention. |
| The diffusion trainer in `flaxdiff.trainer` | `dew.Trainer` plus `dew.objectives.diffusion.DiffusionObjective` | The objective runs diffusion-specific computation. The trainer runs updates, places state, logs progress and manages checkpoints. |
| `flaxdiff.jepa` | `dew.objectives.jepa` and `dew.nn.backbones.jepa` | The objective and its modules implement the encoder, predictor and target behavior. |
| `flaxdiff.samplers` and `flaxdiff.inference` | `dew.sampling` | Construct sampling with the model's training process and compatible conditions. |
| `flaxdiff.metrics` | `dew.eval` | Choose metrics that consume the current objective's evaluation outputs. |
| `training.py` and `training_jepa.py` | `recipes/diffusion/train.py` and `recipes/jepa/train.py` | Recipes use dataclass configurations and generated CLI flags instead of the old argparse interface. |

See [Custom objectives](concepts/objectives.md) for the objective contract, [Diffusion training](guides/diffusion.md) for the diffusion components, and [Recipes](recipes.md) for command-line configuration.

## Loading an older text-to-image checkpoint

`TextToImage.from_flaxdiff` loads `simple_udit` and `hybrid_dit` latent text-to-image models from the older checkpoint format. These models trained with the Stable Diffusion VAE and a CLIP text encoder. The loader returns a `TextToImage`, the same task `dew.pipeline` returns for a Dew run.

You need these from the old run:

| Argument | Where it comes from |
|---|---|
| Checkpoint directory | One step directory written by the older trainer, with a `default/` folder inside |
| `config` | The run's saved training config, as JSON; it names the architecture and sizes, which the checkpoint does not |
| `jax_version` | The JAX version in the run's `requirements.txt`. The older code used `jax.random` for the time embedding's random frequencies without saving them. That random stream changed in JAX 0.5.0, so the loader regenerates the frequencies using the old version's stream. |

<!-- not run: needs an older checkpoint and its training config -->
```python
import json

from dew.sampling import Heun, TextToImage

config = json.load(open("run_config.json"))  # the run's saved training config
pipe = TextToImage.from_flaxdiff("checkpoints/350000", config, jax_version="0.5.3")
images = pipe(["a lighthouse on a rocky coast"], key=0, steps=25, solver=Heun()).host().images
```

`images` is a float array in `[-1, 1]`. For a 256px run, its shape is `[prompts, 256, 256, 3]`. The loader downloads the text tower (CLIP ViT-L/14) and VAE from the Hugging Face Hub using the names in the config.

By default, sampling matches the older trainer's previews: Euler ancestral, 200 steps on its time grid, and classifier-free guidance 3. Set `steps` to use a different number of steps on that grid. `tests/test_flaxdiff.py` compares sampling against the older package's preview of the same run, using its source at the pinned commit.

The loader reads the last state's averaged (EMA) weights. Use `ema=False` or `best=True` to select the other weights. It builds the Dew architecture with `adaln_silu=False` and `text_pooling="all"`. These settings match the two places where `flaxdiff` 0.2's blocks differ from Dew's defaults. `tests/test_flaxdiff.py` checks the output against that package's code.

There is no loader for other formats, including the older UNets and the 2024 checkpoints. Keep those runs with the source revision, environment, data and encoder files that produced them.

## New runs

A Dew run starts with a current configuration and a new output directory. A recipe writes `run.json` next to its checkpoints. Checkpoint state includes live variables, optimizer state, step counters and the random key. It also includes an EMA or reference tree when the run has one. Parameter names and state structure must match the rebuilt model. Renaming the checkpoint directory or importing a different package does not convert the checkpoint. See [Checkpoints](guides/checkpoints.md) for saving and restoring.

## Older results

The [Gallery](gallery.md) shows samples from models trained with Dew, with their recorded settings. The [benchmark page](benchmarks.md) lists the revision, environment, and hardware for each measured Dew run.

Dew includes code adapted from the earlier project and ideas from other research. [Papers and attribution](references.md) lists Diffusers, jax-fid, JEPA and the other upstream sources. Moving code between modules does not change its attribution or license terms.
