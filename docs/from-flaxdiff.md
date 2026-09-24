# Coming from FlaxDiff

FlaxDiff was this project's earlier API, built around diffusion. Dew keeps model construction, data loading, task objectives, and training apart, so the same trainer also runs language models and representation learning. The table below tells you where each familiar piece lives now. The Dew names are not import aliases, and you cannot migrate code by search and replace.

If you are new to the project, start with [getting started](getting-started.md). You do not need to learn FlaxDiff first.

## Map the responsibilities

| FlaxDiff area | Dew area | Change to account for |
| --- | --- | --- |
| `flaxdiff.models` | `dew.nn` and the `dew.models` registry | Build a registered model from its current fields; old constructor fields need review. |
| `flaxdiff.schedulers` and `flaxdiff.predictors` | `dew.diffusion.schedules` and `dew.diffusion.transforms` | A diffusion preset composes schedule, target, weighting, and preconditioning choices. A schedule alone does not describe the whole training convention. |
| The diffusion trainer in `flaxdiff.trainer` | `dew.Trainer` plus `dew.objectives.diffusion.DiffusionObjective` | The objective owns diffusion-specific computation; the trainer owns updates, state placement, logging, and checkpoint orchestration. |
| `flaxdiff.jepa` | `dew.objectives.jepa` and `dew.nn.backbones.jepa` | Encoder/predictor/target behavior belongs to the objective and its modules. |
| `flaxdiff.samplers` and `flaxdiff.inference` | `dew.sampling` | Construct sampling with the model's training process and compatible conditions. |
| `flaxdiff.metrics` | `dew.eval` | Choose metrics that consume the current objective's evaluation outputs. |
| `training.py` and `training_jepa.py` | `recipes/diffusion/train.py` and `recipes/jepa/train.py` | Recipes use dataclass configurations and generated CLI flags instead of the old argparse interface. |

[Objectives](concepts/objectives.md) describes the training contract all objectives share. [The diffusion guide](guides/diffusion.md) covers the diffusion pieces, and [recipes](recipes.md) covers command-line configuration. The [core API reference](reference/core-api.md) documents the current interfaces.

## Load a FlaxDiff text-to-image checkpoint

`dew.interop.flaxdiff.load_flaxdiff` loads one kind of FlaxDiff run: a `simple_udit` latent text-to-image model from FlaxDiff 0.2 (the code flaxdiff 0.2.8 shipped), trained on the Stable Diffusion VAE with a CLIP text encoder. It returns a `TextToImage`, the same task a Dew run's `dew.pipeline` gives you.

It needs three things from the old run. The first is one checkpoint step, the directory FlaxDiff's trainer wrote with a `default/` folder inside. The second is the run config the trainer logged to wandb, which names the architecture and its sizes; the checkpoint does not. The third is the jax version the run trained under, from the run's `requirements.txt`. FlaxDiff drew the random frequencies of its time embedding from `jax.random`, whose stream changed in jax 0.5.0, and never saved them, so the loader has to draw them the way that jax did.

```python
import json

from dew.interop.flaxdiff import load_flaxdiff
from dew.sampling import Heun

config = json.load(open("run_config.json"))  # wandb.Api().run(path).config
pipe = load_flaxdiff("checkpoints/350000", config, jax_version="0.5.3")
images = pipe(["a lighthouse on a rocky coast"], seed=0, steps=25, sampler=Heun()).host().images
```

`images` is a float array in `[-1, 1]`, `[prompts, 256, 256, 3]` for a 256px run. The text tower (CLIP ViT-L/14) and the VAE download from the Hugging Face Hub under the names the config records. By default a call samples the way FlaxDiff's trainer previewed the run: Euler ancestral over 200 steps, classifier-free guidance 3. The loader reads the averaged (EMA) weights of the last state; `ema=False` and `best=True` choose the others. It builds Dew's own `simple_udit` with `adaln_silu=False` and `text_pooling="all"`, the two places where FlaxDiff 0.2's blocks differ from Dew's defaults, and `tests/test_flaxdiff.py` checks its output against FlaxDiff's own code.

Anything else, including FlaxDiff's UNets and its 2024 checkpoints, has no loader. Keep each of those runs together with the source revision, environment, data and encoder files that produced it.

## Start in a new run directory

For a Dew run, write a current configuration and use a new output directory. A recipe writes `run.json` next to its checkpoints. The checkpoint state holds the live variables, the optimizer state, the EMA or reference tree when there is one, the step counters, and the random key. The parameter names and the state structure must match the model you rebuild. Renaming a checkpoint directory or changing the package you import does not convert what is inside it. Read [checkpoints](guides/checkpoints.md) for how saving and restoring work today.

## Interpret the older results

The [gallery](gallery.md) keeps the earlier image grids and the settings I recorded, with the old API names marked as historical. Those runs show nothing about whether the current API reproduces them, or about current distributed-training support. The [benchmark page](benchmarks.md) lists the revision, environment, and hardware for each measured Dew run.

Dew still contains code adapted from the earlier project and ideas taken from other research. [References and attribution](references.md) lists Diffusers, jax-fid, JEPA, and the other upstream sources. Their attribution and license terms still apply after the code moved into different modules.
