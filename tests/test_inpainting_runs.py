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
from diffusion_support import images_of

from dew.checkpoints import Checkpoints
from dew.cli.main import main
from dew.config import ObjectiveConfig
from dew.config.diffusion import DiffusionRunConfig, FlowRolloutConfig
from dew.data import images as image_steps, presets
from dew.data.sources import source
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


def masked_images(batch: int = 4):
    """The run's images at SIZE with a mask drawn for each (`RandomMask`)."""
    return source("tfds://oxford_flowers102").pipe(presets.images, SIZE, batch=batch,
                                                   augment=image_steps.RandomMask())


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
    masks = np.asarray(next(iter(run.chain().read()))["mask"])
    assert masks.shape == (4, SIZE, SIZE, 1) and set(np.unique(masks)) <= {0.0, 1.0}
    assert len({mask.tobytes() for mask in masks}) > 1, "each image draws its own mask"

    record = tmp_path / "record"
    run.save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "inpainting", "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
                "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_final": "True",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
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
                             data=images_of("tfds://oxford_flowers102", SIZE), val_metrics=(),
                             objective=RUNS["diffusion"][1]["objective"])
    with pytest.raises(ValueError):
        run.build()
