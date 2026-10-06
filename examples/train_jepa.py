"""Train an I-JEPA encoder on Oxford Flowers, probe it, save the encoder.

python examples/train_jepa.py --data-path /data/oxford_flowers102/2.1.1 --epochs 300
python examples/train_jepa.py --data-path /data/oxford_flowers102/2.1.1 \
    --steps 20 --image-size 32 --patch-size 4
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import OptimConfig
from dew.data import TFDSImages
from dew.inputs import Field
from dew.interop import save_params
from dew.objectives.jepa import (
    JepaEncoder,
    JepaObjective,
    JepaPredictor,
    KnnProbe,
    LinearProbe,
    MultiBlockMask,
)
from dew.training import Checkpoints, Trainer


@dataclass
class Config:
    data_path: Path | None = None
    classes: int = 102
    image_size: int = 224
    patch_size: int = 16
    batch_size: int = 64
    epochs: int = 300
    steps: int | None = None
    """Run length in steps; unset trains for `epochs` passes over the data."""
    learning_rate: float = 1e-3
    model: dict = field(default_factory=lambda: {
        "emb_features": 384, "num_layers": 12, "num_heads": 6})
    out: Path = Path("runs/ijepa-flowers")


def main(config: Config, data=None):
    data = data or TFDSImages(
        path=None if config.data_path is None else str(config.data_path.expanduser()),
        name="oxford_flowers102",
        image_size=config.image_size,
    ).load(batch=config.batch_size)
    steps = config.steps or data.epoch_steps(config.epochs)
    side = config.image_size // config.patch_size
    grid = (side, side)
    encoder = JepaEncoder(**config.model, patch_size=config.patch_size, dtype=jnp.bfloat16)
    predictor = JepaPredictor(
        grid=grid, emb_features=config.model["emb_features"],
        num_heads=config.model["num_heads"], predictor_features=config.model["emb_features"] // 2,
        num_layers=max(1, config.model["num_layers"] // 2), dtype=jnp.bfloat16)
    objective = JepaObjective(encoder, predictor, mask=MultiBlockMask.for_grid(grid),
                              sample=Field("image", (config.image_size, config.image_size, 3)),
                              momentum_steps=steps)

    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50, eval_every=steps,
                        metrics=(LinearProbe(config.classes), KnnProbe(config.classes)))

    config.out.mkdir(parents=True, exist_ok=True)
    save_params(state.averaged["params"]["context_encoder"], config.out / "encoder.safetensors")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
