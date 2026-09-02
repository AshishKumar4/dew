# Coming from FlaxDiff

If you are coming from FlaxDiff, most of the code is the same, only moved around:

- `flaxdiff.models` became `dew.nn`, with the registry at `dew.registry`.
- `flaxdiff.schedulers` and `flaxdiff.predictors` became `dew.diffusion.schedules` and `dew.diffusion.transforms`.
- `flaxdiff.trainer.GeneralDiffusionTrainer` became `dew.training.ObjectiveTrainer`, and `flaxdiff.jepa` became `dew.objectives.jepa` plus `dew.nn.backbones.jepa`.
- `flaxdiff.samplers` and `flaxdiff.inference` became `dew.sampling`, and `flaxdiff.metrics` became `dew.eval`.
- `training.py` and `training_jepa.py` became `recipes/diffusion/train.py` and `recipes/jepa/train.py`, with typed configs instead of argparse.
- Parameter trees and checkpoint layouts did not change, so old checkpoints load. `tools/convert_legacy_checkpoint.py` handles the ones from before the DiT consolidation.
