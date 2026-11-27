"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --trainer.epochs 2000

Oxford flowers with flower captions is the default; docs/recipes.md lists the
other corpora. The run is `dew.config.diffusion.DiffusionRunConfig`, its model and
its data a chain (`dew.data.presets.images`, its size and batch) as `default` states them.

`--pretrained stabilityai/stable-diffusion-3.5-medium preset:none` fine-tunes a
published pipeline (SD3, Flux, Qwen-Image or a UNet) on its own conditioning,
autoencoder and convention; a run whose objective is `ObjectiveConfig(functools.partial(FlowGRPOObjective))` with
`--rollout.reward clip_score` trains the model with Flow-GRPO on that reward instead of the
denoising loss.
"""

import functools

from dew.config.diffusion import DiffusionRunConfig
from dew.data import images, presets
from dew.data.sources import source
from dew.nn.backbones.dit import SimpleDiT
from dew.registry import ModelConfig

TEMPLATES = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}",
             "This is a photo of a {} flower", "A photo of a {} flower")
VALIDATION_TEMPLATE = "a photo of a {label}"
"""The held-out images' one caption template, the first of `TEMPLATES`."""


def flowers():
    """Oxford flowers, its last 128 records held out."""
    return source("tfds://oxford_flowers102").holdout(records=128)


def default(train, held) -> DiffusionRunConfig:
    """The recipe's run over labelled images, `train`'s and `held`'s, each captioned by its class: a
    training image by a template drawn for it, a held-out one by the first, so every validation reads the
    same captions (a drawn caption in validation is refused, its draws having no record index to follow).
    An image that does not decode is dropped and counted (`data/rejected/`), its record still read by the
    pass its epochs count."""
    return DiffusionRunConfig(
        model=ModelConfig(functools.partial(SimpleDiT, patch_size=4, emb_features=512,
                                            num_layers=12, num_heads=8)),
        data=train.shuffle(seed=0).repeat().pipe(presets.images, 128, batch=32, augment=True,
                                                 caption=images.caption(templates=TEMPLATES)),
        validation=held.pipe(presets.images, 128, batch=32, caption=images.caption(VALIDATION_TEMPLATE)))


def main(argv: list[str] | None = None, images=None):
    """Run the recipe on `argv`, the process's own command line where None, over `images`' training and
    held-out records, Oxford flowers' where None, and return the run's last state."""
    return DiffusionRunConfig.cli(argv, default=default(*(flowers() if images is None else images))).run()


if __name__ == "__main__":
    main()
