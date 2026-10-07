"""Train a diffusion model over images or latents.

    python recipes/diffusion/train.py --data.path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \\
        --data.image-size 128 --trainer.batch-size 32 --trainer.epochs 2000 \\
        --model simple_dit --model.patch-size 4 --model.emb-features 512 \\
        --model.num-layers 12 --model.num-heads 8

Oxford flowers with flower captions is the default; docs/recipes.md lists the
other corpora. The run is `dew.objectives.diffusion.DiffusionRunConfig`.

`--pretrained stabilityai/stable-diffusion-3.5-medium preset:none` fine-tunes a
published pipeline (SD3, Flux, Qwen-Image or a UNet) on its own conditioning,
autoencoder and convention; `--objective flow_grpo --objective.reward clip_score`
trains the model with Flow-GRPO on that reward instead of the denoising loss.
"""

from dew.data import TFDSImages
from dew.objectives.diffusion import DiffusionRunConfig

FLOWERS = TFDSImages(name="oxford_flowers102", caption_templates=(
    "a photo of a {}", "a photo of a {} flower", "This is a photo of a {}", "This is a photo of a {} flower",
    "A photo of a {} flower"))

if __name__ == "__main__":
    DiffusionRunConfig.cli(default=DiffusionRunConfig(data=FLOWERS)).run()
