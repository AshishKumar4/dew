# Coming from FlaxDiff

FlaxDiff was an earlier name for Dew, when it was built around diffusion. Dew separates model construction, data loading, objectives and training, so the same trainer also runs language models and representation learning. This page maps FlaxDiff's modules to Dew's and describes the one FlaxDiff checkpoint format Dew loads. Dew's names are not import aliases; code does not migrate by search and replace.

## Module map

| FlaxDiff area | Dew area | Change to account for |
| --- | --- | --- |
| `flaxdiff.models` | `dew.nn.backbones` | Build the model from its class with its current fields; old constructor fields need review. |
| `flaxdiff.schedulers` and `flaxdiff.predictors` | `dew.diffusion.schedules` and `dew.diffusion.transforms` | A diffusion preset composes schedule, target, weighting, and preconditioning choices. A schedule alone does not describe the whole training convention. |
| The diffusion trainer in `flaxdiff.trainer` | `dew.Trainer` plus `dew.objectives.diffusion.DiffusionObjective` | The objective owns diffusion-specific computation; the trainer owns updates, state placement, logging, and checkpoint orchestration. |
| `flaxdiff.jepa` | `dew.objectives.jepa` and `dew.nn.backbones.jepa` | Encoder/predictor/target behavior belongs to the objective and its modules. |
| `flaxdiff.samplers` and `flaxdiff.inference` | `dew.sampling` | Construct sampling with the model's training process and compatible conditions. |
| `flaxdiff.metrics` | `dew.eval` | Choose metrics that consume the current objective's evaluation outputs. |
| `training.py` and `training_jepa.py` | `recipes/diffusion/train.py` and `recipes/jepa/train.py` | Recipes use dataclass configurations and generated CLI flags instead of the old argparse interface. |

[Custom objectives](concepts/objectives.md) describes the contract every objective follows, [Diffusion training](guides/diffusion.md) the diffusion pieces, and [Recipes](recipes.md) the command-line configuration.

## Loading an older text-to-image checkpoint

`dew.interop.flaxdiff.load_flaxdiff` loads `simple_udit` and `hybrid_dit` latent text-to-image models from the older checkpoint format, trained on the Stable Diffusion VAE with a CLIP text encoder. It returns a `TextToImage`, the same task `dew.pipeline` returns for a Dew run.

It takes three things from the old run:

| Argument | Where it comes from |
|---|---|
| Checkpoint directory | One step directory written by the older trainer, with a `default/` folder inside |
| `config` | The run's saved training config, as JSON; it names the architecture and sizes, which the checkpoint does not |
| `jax_version` | The JAX version in the run's `requirements.txt`. The older code drew the random frequencies of its time embedding from `jax.random`, whose stream changed in JAX 0.5.0, and did not save them, so the loader draws them the way that version did. |

<!-- not run: needs an older checkpoint and its training config -->
```python
import json

from dew.interop.flaxdiff import load_flaxdiff
from dew.sampling import Heun

config = json.load(open("run_config.json"))  # the run's saved training config
pipe = load_flaxdiff("checkpoints/350000", config, jax_version="0.5.3")
images = pipe(["a lighthouse on a rocky coast"], key=0, steps=25, sampler=Heun()).host().images
```

`images` is a float array in `[-1, 1]`, `[prompts, 256, 256, 3]` for a 256px run. The text tower (CLIP ViT-L/14) and the VAE download from the Hugging Face Hub under the names the config records. By default a call samples the way the older trainer previewed its runs: Euler ancestral over 200 steps, classifier-free guidance 3. The loader reads the averaged (EMA) weights of the last state; `ema=False` and `best=True` choose the others. It builds the matching Dew architecture with `adaln_silu=False` and `text_pooling="all"`, the two places where the blocks of the `flaxdiff` 0.2 package differ from Dew's defaults, and `tests/test_flaxdiff.py` checks its output against that package's own code.

Anything else, including the older UNets and the 2024 checkpoints, has no loader. Keep each of those runs together with the source revision, environment, data and encoder files that produced it.

## New runs

A Dew run starts from a current configuration in a new output directory. A recipe writes `run.json` next to its checkpoints. The checkpoint state holds the live variables, the optimizer state, the EMA or reference tree when there is one, the step counters, and the random key. The parameter names and the state structure must match the model you rebuild. Renaming a checkpoint directory or changing the package you import does not convert what is inside it. [Checkpoints](guides/checkpoints.md) describes saving and restoring.

## Older results

The [Gallery](gallery.md) shows samples from models trained with Dew, with their recorded settings. The [benchmark page](benchmarks.md) lists the revision, environment, and hardware for each measured Dew run.

Dew still contains code adapted from the earlier project and ideas taken from other research. [Papers and attribution](references.md) lists Diffusers, jax-fid, JEPA, and the other upstream sources. Their attribution and license terms still apply after the code moved into different modules.
