"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

    python recipes/jepa/train.py --data.source tfds://oxford_flowers102 \\
        --data.image-size 224 --trainer.batch-size 64 --trainer.epochs 300 --probe-classes 102 \\
        --model.patch-size 16 --model.emb-features 384 --model.num-layers 12 --model.num-heads 6

The encoder is --model, the predictor takes the encoder's width and heads plus
--predictor, and the probes score the frozen encoder at every validation
(`dew.objectives.jepa.JepaRunConfig`).
"""

from dew.objectives.jepa import JepaRunConfig

if __name__ == "__main__":
    JepaRunConfig.cli().run()
