# Coming from FlaxDiff

> An AI assistant maintains this document. It is presented as-is.

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

## Start in a new run directory

Dew has no converter for FlaxDiff checkpoints and no FlaxDiff import paths. Keep each old run together with the source revision, environment, data, and tokenizer or encoder files that produced it. If you want to inspect or sample an old checkpoint, keep that environment too.

For a Dew run, write a current configuration and use a new output directory. A recipe writes `run.json` next to its checkpoints. The checkpoint state holds the live variables, the optimizer state, the EMA or reference tree when there is one, the step counters, and the random key. The parameter names and the state structure must match the model you rebuild. Renaming a checkpoint directory or changing the package you import does not convert what is inside it.

If you want to move only the parameters across, you need an explicit mapping of names and shapes and a comparison of the two models' outputs. This page does not give you that mapping. Read [checkpoints](guides/checkpoints.md) for how saving and restoring work today. Both projects use Flax, but that does not make their checkpoints compatible.

## Interpret the older results

The [gallery](gallery.md) keeps the earlier image grids and the settings I recorded, with the old API names marked as historical. Those runs show nothing about whether the current API reproduces them, or about current distributed-training support. The [benchmark page](benchmarks.md) lists the revision, environment, and hardware for each measured Dew run.

Dew still contains code adapted from the earlier project and ideas taken from other research. [References and attribution](references.md) lists Diffusers, jax-fid, JEPA, and the other upstream sources. Their attribution and license terms still apply after the code moved into different modules.
