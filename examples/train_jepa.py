"""Train an I-JEPA encoder on Oxford Flowers and probe its embeddings with a linear and a kNN classifier.

    python examples/train_jepa.py --epochs 300 --image-size 224
    python examples/train_jepa.py --epochs 1 --steps-per-epoch 20 --image-size 32 --patch-size 4   # smoke run
"""
from dataclasses import dataclass
from pathlib import Path

import jax
import optax
import tyro

from dew.data.dataloaders import get_dataset_grain
from dew.inputs import DiffusionInputConfig
from dew.interop import save_params
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.jepa.probes import get_knn_probe_metric, get_linear_probe_metric
from dew.registry import apply_precision_policy, build_model
from dew.training import ObjectiveTrainer


@dataclass
class Config:
    dataset: str = "oxford_flowers102"
    classes: int = 102
    image_size: int = 224
    patch_size: int = 16
    batch_size: int = 64
    epochs: int = 300
    steps_per_epoch: int | None = None
    learning_rate: float = 1e-3
    emb_features: int = 384
    num_layers: int = 12
    num_heads: int = 6
    out: Path = Path("runs/ijepa-flowers")


def main(config: Config, data=None):
    data = data or get_dataset_grain(config.dataset, batch_size=config.batch_size, image_scale=config.image_size)
    grid = (config.image_size // config.patch_size,) * 2

    # The context encoder is the model being trained. The predictor maps its embeddings of the
    # visible patches to the target encoder's embeddings of the masked blocks.
    encoder_config = apply_precision_policy("jepa_encoder", dict(
        patch_size=config.patch_size, emb_features=config.emb_features,
        num_layers=config.num_layers, num_heads=config.num_heads,
    ), dtype="bfloat16", attention_impl="auto")
    encoder = build_model("jepa_encoder", encoder_config)
    predictor = build_model("jepa_predictor", dict(
        grid=grid, emb_features=config.emb_features, predictor_features=config.emb_features // 2,
        num_layers=max(1, config.num_layers // 2), num_heads=config.num_heads,
        dtype=encoder_config["dtype"], attention_impl=encoder_config["attention_impl"],
    ))
    objective = JepaObjective(encoder, predictor, mask=multi_block_mask(grid),
                              sample_data_key="image", sample_data_shape=(config.image_size, config.image_size, 3))

    trainer = ObjectiveTrainer(
        encoder, optax.adamw(config.learning_rate), objective=objective,
        input_config=DiffusionInputConfig(sample_data_key="image",
                                          sample_data_shape=(config.image_size, config.image_size, 3), conditions=[]),
        eval_metrics=[get_linear_probe_metric(config.classes), get_knn_probe_metric(config.classes)],
        rngs=jax.random.PRNGKey(0), name=config.out.name, checkpoint_base_path=str(config.out / "checkpoints"),
    )
    steps = config.steps_per_epoch or data["train_len"] // config.batch_size
    state = trainer.fit(data, training_steps_per_epoch=steps, epochs=config.epochs, val_steps_per_epoch=1)

    # The EMA of the context encoder is the encoder to keep.
    save_params(state.ema_params["params"]["context_encoder"], config.out / "encoder.safetensors")
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
