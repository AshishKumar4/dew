"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

    python recipes/jepa/train.py --trainer.horizon.epochs 300 --probe-classes 102

Oxford flowers is the default, its validation split held to score the probes; its data is a chain
(`dew.data.presets.images`, its size and batch in `default`). The encoder is the run's model, which
`default` states, the predictor takes the encoder's width and heads plus --predictor, and the probes score
the frozen encoder at every validation (`dew.config.jepa.JepaRunConfig`).
"""

import dataclasses

from dew.config.jepa import JepaRunConfig
from dew.data import presets
from dew.data.sources import source


def flowers():
    """Oxford flowers, its validation split held out."""
    return source("tfds://oxford_flowers102").holdout(split="validation")


def default(train, held) -> JepaRunConfig:
    """The recipe's run over `train`'s and `held`'s images, `held`'s labels scoring the probes: an image
    that does not decode is dropped and counted (`data/rejected/`), its record still read by the pass its
    epochs count."""
    run = JepaRunConfig(data=train.shuffle(seed=0).repeat().pipe(presets.images, 224, batch=64),
                        validation=held.pipe(presets.images, 224, batch=64))
    encoder = {"patch_size": 16, "emb_features": 384, "num_layers": 12, "num_heads": 6}
    return dataclasses.replace(run, model=dataclasses.replace(run.model,
                                                              fields={**run.model.fields, **encoder}))


def main(argv: list[str] | None = None, images=None):
    """Run the recipe on `argv`, the process's own command line where None, over `images`' training and
    held-out records, Oxford flowers' where None, and return the run's last state."""
    return JepaRunConfig.cli(argv, default=default(*(flowers() if images is None else images))).run()


if __name__ == "__main__":
    main()
