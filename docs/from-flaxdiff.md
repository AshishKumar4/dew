# Coming from FlaxDiff

FlaxDiff was the project's earlier diffusion-focused API. Dew separates model construction, data loading, task objectives, and training so the trainer can also run language-model and representation-learning workloads. The names below help you locate familiar responsibilities; they are not import aliases or a mechanical search-and-replace migration.

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

Read [objectives](concepts/objectives.md) for the shared training contract, [the diffusion guide](guides/diffusion.md) for the diffusion pieces, and [recipes](recipes.md) for command-line configuration. The [API overview](api.md) lists current modules.

## Start in a new run directory

Dew does not provide a FlaxDiff checkpoint converter or deprecated FlaxDiff import paths. Keep an older run with the source revision, environment, data, and tokenizer or encoder assets that created it. Preserve that environment if you need to inspect or sample its checkpoint.

For a Dew run, build a current configuration in a new output directory. A recipe writes `run.json` alongside checkpoints, and the checkpoint state includes the live variables, optimizer state, EMA/reference tree when used, step, and random key. Parameter names and state structure must match the model you rebuild. Renaming a checkpoint directory or changing its package import does not translate its contents.

A parameter-only transfer, if you undertake one, needs an explicit name/shape mapping and output comparisons for the model involved. This page does not supply that mapping. Use [checkpoints](guides/checkpoints.md) for the current save/restore contract; do not infer compatibility from the fact that both versions use Flax.

## Interpret the older results

The [gallery](gallery.md) retains the earlier image grids and their recorded settings, with old API names labeled as historical. Those runs do not establish current-API reproducibility or current distributed-training support. The [benchmark page](benchmarks.md) keeps separate revision, environment, and hardware information for measured Dew runs.

Dew retains adapted code and research influences from the earlier project. See [references and attribution](references.md) for Diffusers, jax-fid, JEPA, and other upstream sources; their attribution and licensing obligations do not disappear when modules move.
