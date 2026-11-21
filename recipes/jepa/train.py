"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

    python recipes/jepa/train.py --set data.images.size=224 --set data.images.batch=64 \\
        --trainer.epochs 300 --probe-classes 102 \\
        --model.patch-size 16 --model.emb-features 384 --model.num-layers 12 --model.num-heads 6

Oxford flowers is the default, its validation split held to score the probes; its
data is a chain (`dew.data.presets.images`) whose settings `--set data.images.<setting>=`
changes. The encoder is --model, the predictor takes the encoder's width and heads plus
--predictor, and the probes score the frozen encoder at every validation
(`dew.config.jepa.JepaRunConfig`).
"""

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
    return JepaRunConfig(data=train.shuffle(seed=0).repeat().pipe(presets.images, 224, batch=64),
                         validation=held.pipe(presets.images, 224, batch=64))


def main(argv: list[str] | None = None, images=None):
    """Run the recipe on `argv`, the process's own command line where None, over `images`' training and
    held-out records, Oxford flowers' where None, and return the run's last state."""
    return JepaRunConfig.cli(argv, default=default(*(flowers() if images is None else images))).run()


if __name__ == "__main__":
    main()
