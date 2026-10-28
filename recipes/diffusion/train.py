"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --set data.images.size=128 --set data.images.batch=32 \\
        --trainer.epochs 2000 --model simple_dit --model.patch-size 4 --model.emb-features 512 \\
        --model.num-layers 12 --model.num-heads 8

Oxford flowers with flower captions is the default; docs/recipes.md lists the
other corpora. The run is `dew.config.diffusion.DiffusionRunConfig`, its
data a chain (`dew.data.presets.images`) whose settings `--set data.images.<setting>=`
changes.

`--pretrained stabilityai/stable-diffusion-3.5-medium preset:none` fine-tunes a
published pipeline (SD3, Flux, Qwen-Image or a UNet) on its own conditioning,
autoencoder and convention; `--objective flow_grpo --rollout.reward clip_score`
trains the model with Flow-GRPO on that reward instead of the denoising loss.
"""

from dew.config.diffusion import DiffusionRunConfig
from dew.data import images, presets
from dew.data.sources import source

TEMPLATES = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}",
             "This is a photo of a {} flower", "A photo of a {} flower")
TRAIN, HELD = source("tfds://oxford_flowers102").holdout(records=128)
FLOWERS = TRAIN.shuffle(seed=0).repeat().pipe(presets.images, 128, batch=32, augment=True,
                                              caption=images.caption(templates=TEMPLATES))
VALIDATION = HELD.pipe(presets.images, 128, batch=32, caption=images.caption(templates=TEMPLATES))

if __name__ == "__main__":
    DiffusionRunConfig.cli(default=DiffusionRunConfig(data=FLOWERS, validation=VALIDATION)).run()
