"""Train an I-JEPA encoder on Oxford Flowers, probe it, save the encoder.

    python examples/train_jepa.py --epochs 300 --image-size 224
    python examples/train_jepa.py --steps 20 --image-size 32 --patch-size 4   # smoke run
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import optax
import tyro

from dew.data import OxfordFlowers
from dew.inputs import Field
from dew.interop import save_params
from dew.objectives.jepa import JepaObjective, multi_block_mask
import dew.nn.backbones  # registers the models
from dew.registry import metrics, models
from dew.training import Checkpoints, Trainer


@dataclass
class Config:
    classes: int = 102
    image_size: int = 224
    patch_size: int = 16
    batch_size: int = 64
    epochs: int = 300
    steps: int | None = None
    """Run length in steps; unset trains for `epochs` passes over the data."""
    learning_rate: float = 1e-3
    model: dict = field(default_factory=lambda: dict(emb_features=384, num_layers=12, num_heads=6))
    out: Path = Path("runs/ijepa-flowers")


def main(config: Config, data=None):
    data = data or OxfordFlowers(image_size=config.image_size).load(batch=config.batch_size)
    steps = config.steps or config.epochs * data.steps_per_epoch
    grid = (config.image_size // config.patch_size,) * 2
    encoder = models.build("jepa_encoder", **config.model, patch_size=config.patch_size, dtype="bfloat16")
    predictor = models.build(
        "jepa_predictor", grid=grid, emb_features=config.model["emb_features"],
        num_heads=config.model["num_heads"], predictor_features=config.model["emb_features"] // 2,
        num_layers=max(1, config.model["num_layers"] // 2), dtype="bfloat16")
    objective = JepaObjective(encoder, predictor, mask=multi_block_mask(grid),
                              sample=Field("image", (config.image_size, config.image_size, 3)),
                              momentum_steps=steps)

    trainer = Trainer(objective, optax.adamw(config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50, eval_every=steps,
                        metrics=(metrics.linear_probe(config.classes), metrics.knn_probe(config.classes)))

    config.out.mkdir(parents=True, exist_ok=True)
    save_params(state.ema["params"]["context_encoder"], config.out / "encoder.safetensors")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
