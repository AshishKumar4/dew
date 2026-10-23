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

TRAIN, HELD = source("tfds://oxford_flowers102").holdout(split="validation")
FLOWERS = TRAIN.shuffle(seed=0).repeat().pipe(presets.images, 224, batch=64)
VALIDATION = HELD.pipe(presets.images, 224, batch=64)

if __name__ == "__main__":
    JepaRunConfig.cli(default=JepaRunConfig(data=FLOWERS, validation=VALIDATION)).run()
