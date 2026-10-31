"""A diffusion run over an inpainting pipeline through `dew train` (ledger 667, train 370's 2890): the
pipeline's mask is the chain's `mask` field, which `dew.data.images.RandomMask` draws for each image as
Diffusers' inpainting trainer draws one, and the denoising and Flow-GRPO objectives each train the
pipeline's denoiser through its masked-image inputs. A chain without a mask is refused, naming the step
that draws one."""

import dataclasses
import json
import shutil
import tarfile
from pathlib import Path

import jax
import numpy as np
import pytest
from objective_data import IMAGES

from dew.checkpoints import Checkpoints
from dew.cli.main import main
from dew.config import ObjectiveConfig
from dew.config.diffusion import DiffusionRunConfig, FlowRolloutConfig
from dew.data import TFDS, images as image_steps, presets
from dew.data.dataset import DataPartition
from dew.interop.pretrained import load_diffusion_source
from dew.registry import ModelConfig
from dew.sampling.solvers import Euler

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SIZE = 16
"""The tiny inpainting UNet's images: its 8-latent sample at the VAE's downscale of 2."""

FLOW_SCHEDULER = {"_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.34.0",
                  "num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False}
"""Diffusers' rectified-flow scheduler file, as an SD3 release states it at no shift."""


@pytest.fixture(scope="module")
def inpainting(tmp_path_factory) -> dict[str, Path]:
    """The tiny inpainting pipeline (a UNet of 9 input channels, its latents, the masked image's and the
    mask), as published with its DDIM scheduler (`ddim`) and with Diffusers' rectified-flow scheduler
    file in its place (`flow`), which Flow-GRPO walks."""
    root = tmp_path_factory.mktemp("inpainting")
    with tarfile.open(FIXTURES / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(root, members=[member for member in archive.getmembers()
                                          if member.name.split("/")[0] == "inpaint"], filter="data")
    flow = root / "flow"
    shutil.copytree(root / "inpaint", flow)
    (flow / "scheduler" / "scheduler_config.json").write_text(json.dumps(FLOW_SCHEDULER))
    index = json.loads((flow / "model_index.json").read_text())
    (flow / "model_index.json").write_text(json.dumps({**index, "scheduler": ["diffusers",
                                                                              "FlowMatchEulerDiscreteScheduler"]}))
    return {"ddim": root / "inpaint", "flow": flow}


def pictures():
    """The suite's own images (tests/fixtures/tfds/dew_images), which a run reads offline."""
    return TFDS("dew_images", path=str(IMAGES), split="all")


def masked_images(batch: int = 8, *, rejects: bool = True):
    """The run's captioned images at SIZE with a mask drawn for each (`RandomMask`): a validation split's
    keeps every image (`rejects=False`), so its masks are drawn by each record's own index."""
    return pictures().pipe(presets.images, SIZE, batch=batch, caption=image_steps.caption(),
                           augment=image_steps.RandomMask(), rejects=rejects)


def precision() -> ModelConfig:
    model = DiffusionRunConfig().model
    return dataclasses.replace(model, fields={**model.fields, "dtype": "float32", "attention_impl": "xla"})


RUNS = {
    "diffusion": ("ddim", {"objective": ObjectiveConfig("diffusion", {"solver": Euler(), "guidance": None,
                                                                      "steps": 2, "ema_decay": None})}),
    "flow_grpo": ("flow", {"objective": ObjectiveConfig("flow_grpo", {"solver": Euler(), "guidance": None,
                                                                      "steps": 2, "clip_range": 0.2}),
                           "rollout": FlowRolloutConfig(reward="psnr", groups=2, steps=3, train_steps=2)}),
}


def random_mask(random, im_shape, ratio=1, mask_full_image=False):
    """Diffusers' `random_mask` (examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py
    at 86ef420b1, Apache-2.0), verbatim but for `random`, its module, passed in, and its boxes' lines
    wrapped."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2,
             center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2,
             center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


class Scripted:
    """The integer draws a script states, in turn, each within the bounds its caller asks: Python's
    `randint(a, b)` (inclusive) for the reference, NumPy's `integers(low, high)` (exclusive) for Dew."""

    def __init__(self, draws):
        self.draws = iter(draws)

    def randint(self, low, high):
        drawn = next(self.draws)
        assert low <= drawn <= high, (low, drawn, high)
        return drawn

    def integers(self, low, high):
        return self.randint(low, high - 1)


DRAWS = {
    "rectangle": [7, 11, 5, 9, 0],
    "ellipse": [7, 11, 5, 9, 1],
    "whole image": [12, 20, 6, 10, 0],
    "at the far corner": [5, 3, 10, 19, 1],
    "empty": [0, 0, 0, 0, 0],
}
"""Draws over a 12-wide, 20-high image: size (width, height), centre (x, y), then the shape, at its
bounds' ends among them."""


@pytest.mark.parametrize("draws", list(DRAWS))
def test_a_random_mask_draws_as_diffusers_inpainting_trainer_draws(draws):
    """`RandomMask` over the same integer draws makes Diffusers' mask (`random_mask`, then its
    `prepare_mask_and_masked_image` binarizing at a half) bit for bit, over a non-square image, either
    shape and its draws' bounds, written as [height, width, 1] float32 (1 the region painted)."""
    width, height = 12, 20
    image = np.zeros((height, width, 3), np.uint8)
    drawn = image_steps.RandomMask().random_map({"image": image}, Scripted(DRAWS[draws]))["mask"]
    mask = np.asarray(random_mask(Scripted(DRAWS[draws]), (width, height)).convert("L"), np.float32) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    np.testing.assert_array_equal(drawn, mask[..., None], strict=True)
    facts = image_steps.RandomMask().declared({"fields": {"image": [height, width, 3]}})
    assert facts["fields"]["mask"] == [height, width, 1]


@pytest.mark.parametrize("objective", list(RUNS))
def test_dew_train_trains_an_inpainting_pipeline_through_its_masks(inpainting, objective, tmp_path):
    """`dew train` over the inpainting pipeline, its images' masks drawn by RandomMask, commits a step
    that moves the denoiser's weights from the pipeline's, the objective reading the chain's mask; the
    chain's masks are each image's own, marking some of it and not all."""
    scheduler, fields = RUNS[objective]
    pipeline = inpainting[scheduler]
    run = DiffusionRunConfig(pretrained=str(pipeline), preset=None, model=precision(), data=masked_images(),
                             val_metrics=(), **fields)
    built = run.build()
    assert built.inputs.mask is not None and built.inputs.mask.shape == (SIZE, SIZE, 1)
    loader, _ = run.compiled(built.data(), run.chain(), None)
    masks = np.asarray(next(iter(loader.train(DataPartition())))["mask"])
    assert masks.shape == (8, SIZE, SIZE, 1) and set(np.unique(masks)) <= {0.0, 1.0}
    assert len({mask.tobytes() for mask in masks}) > 1, "each image draws its own mask"

    record = tmp_path / "record"
    run.save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "inpainting", "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_final": "True",
                "trainer.checkpoint_every": "None", "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": "0.05"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0

    trained = Checkpoints(str(runs / "inpainting")).variables(ema=False)
    published = load_diffusion_source(str(pipeline), dtype="float32", attention_impl="xla",
                                      size=(SIZE, SIZE)).variables
    moved = [not np.array_equal(np.asarray(after), np.asarray(before)) for after, before in zip(
        jax.tree.leaves(trained["params"]), jax.tree.leaves(published["params"]), strict=True)]
    assert any(moved), "the step moves the denoiser"


def test_an_inpainting_run_over_a_chain_without_masks_is_refused(inpainting):
    """An inpainting pipeline paints the region its mask marks: over images that carry no mask the run
    is refused as it builds."""
    run = DiffusionRunConfig(pretrained=str(inpainting["ddim"]), preset=None, model=precision(),
                             data=pictures().pipe(presets.images, SIZE, batch=8,
                                                  caption=image_steps.caption()), val_metrics=(),
                             objective=RUNS["diffusion"][1]["objective"])
    with pytest.raises(ValueError):
        run.build()


def test_an_inpainting_run_validates_through_its_held_out_images_masks(inpainting, tmp_path):
    """An inpainting run's validation chain carries its images' masks as its training chain does
    (`RandomMask`), and `dew train` scores the held-out loss through them at each evaluation, its
    validation logged beside its training (ledger 2900). A validation chain without masks is refused as
    the run is prepared, naming the step that draws them, where an evaluation would meet no mask."""
    _, fields = RUNS["diffusion"]
    run = DiffusionRunConfig(pretrained=str(inpainting["ddim"]), preset=None, model=precision(),
                             data=masked_images(), validation=masked_images(rejects=False), val_metrics=(),
                             **fields)
    record = tmp_path / "record"
    run.save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "validated", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.log_every": "1", "trainer.eval_every": "1", "trainer.checkpoint_final": "False",
                "trainer.checkpoint_every": "None", "trainer.multi_host": "False",
                "trainer.compilation_cache_dir": "None", "optim.learning_rate": "0.05"}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    rows = (runs / "validated" / "tracking" / "scalars.jsonl").read_text().splitlines()
    held_out = [value for row in rows for name, value in json.loads(row)["scalars"].items()
                if name.startswith("val/") and name.endswith("loss")]
    assert held_out and all(np.isfinite(held_out))

    unmasked = dataclasses.replace(run, validation=pictures().pipe(presets.images, SIZE, batch=8,
                                                                     caption=image_steps.caption()))
    with pytest.raises(ValueError, match="RandomMask"):
        unmasked.prepare()

