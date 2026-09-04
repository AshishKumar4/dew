# Coming from FlaxDiff

If you are coming from FlaxDiff, most of the code is the same, only moved around:

- `flaxdiff.models` became `dew.nn`, with the registry at `dew.registry`.
- `flaxdiff.schedulers` and `flaxdiff.predictors` became `dew.diffusion.schedules` and `dew.diffusion.transforms`.
- `flaxdiff.trainer`'s diffusion trainer became `dew.training.Trainer`, which knows no diffusion: the diffusion loss is `dew.objectives.diffusion.DiffusionObjective`, and `flaxdiff.jepa` became `dew.objectives.jepa` plus `dew.nn.backbones.jepa`.
- `flaxdiff.samplers` and `flaxdiff.inference` became `dew.sampling`, and `flaxdiff.metrics` became `dew.eval`.
- `training.py` and `training_jepa.py` became `recipes/diffusion/train.py` and `recipes/jepa/train.py`, with typed configs instead of argparse.
- Dew is unpublished and the API design is a clean cutover. There is no checkpoint converter or deprecated path: a run created before this design stays with the commit that created it, and a run on this API writes the six-leaf state and `run.json` it can rebuild from.
