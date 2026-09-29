"""Numerical regressions for nondefault published diffusion controls."""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import unet_fields

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def edges(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-diffusion-edges")
    with tarfile.open(ROOT / "tests/fixtures/native_diffusion_edges.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.mark.parametrize("case", ["ddim-linspace", "pndm-linspace", "ddim-clip", "euler-zero-snr", "dpm-karras", "norm", "odd"])
def test_native_source_control_and_gradient_parity(edges, case):
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_native_diffusion_edges.py"), str(edges), case],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu", "USE_TF": "0",
             "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case,control,value", [
    ("dpm-karras", "use_lu_lambdas", True),
    ("dpm-karras", "use_flow_sigmas", True),
    ("dpm-karras", "variance_type", "learned_range"),
    ("dpm-karras", "prediction_type", "flow_prediction"),
    ("euler-zero-snr", "interpolation_type", "log_linear"),
    ("euler-zero-snr", "timestep_type", "continuous"),
])
def test_unimplemented_active_scheduler_controls_do_not_change_meaning(edges, case, control, value):
    """A control the pinned class declares and this reconstruction does not
    read is refused at load, so a checkpoint whose meaning it changes cannot
    be sampled as if it were absent."""
    with np.load(edges / "schedulers.npz") as reference:
        config = json.loads(str(reference[case + ".config"]))
    with pytest.raises(ValueError):
        SourceSchedule.from_config({**config, control: value})


@pytest.mark.parametrize("scheduler,controls", [
    # The published TCD step reads none of the three limits its config carries.
    ("TCDScheduler", {"thresholding": True}),
    ("TCDScheduler", {"clip_sample": True}),
    # Its epsilon branch ignores thresholding, and DDPM's log-space wide
    # variance returns a logarithm the step then takes the square root of.
    ("UniPCMultistepScheduler", {"predict_x0": False, "thresholding": True}),
    ("DDPMScheduler", {"variance_type": "fixed_large_log"}),
    ("DDPMScheduler", {"variance_type": "learned"}),
    ("DPMSolverSDEScheduler", {"use_karras_sigmas": True, "use_beta_sigmas": True}),
])
def test_controls_the_published_step_does_not_read_are_refused(scheduler, controls):
    config = {"_class_name": scheduler, "num_train_timesteps": 20, "beta_start": 0.00085,
              "beta_end": 0.012, "beta_schedule": "scaled_linear", **controls}
    with pytest.raises(ValueError):
        SourceSchedule.from_config(config)


def test_source_pndm_does_not_substitute_epsilon_history_for_velocity(edges):
    with np.load(edges / "schedulers.npz") as reference:
        config = json.loads(str(reference["pndm-linspace.config"]))
    with pytest.raises(ValueError):
        SourceSchedule.from_config({**config, "prediction_type": "v_prediction"})


@pytest.mark.parametrize("control,value", [("center_input_sample", True), ("resnet_time_scale_shift", "scale_shift"),
                                           ("act_fn", "mish"), ("conv_in_kernel", 1)])
def test_unimplemented_unet_operations_are_explicit(edges, control, value):
    config = json.loads((edges / "unet/config.json").read_text())
    with pytest.raises(ValueError):
        unet_fields({**config, control: value})


SDXL = ROOT / "tests/fixtures/hf/sdxl-base-1.0"


def test_the_published_sdxl_unet_config_builds_diffusers_own_parameter_count():
    """stabilityai/stable-diffusion-xl-base-1.0's UNet config (at the pinned
    revision, committed) stores `upcast_attention: null`, which diffusers
    reads as False, like the refiner's and sdxl-turbo's. It builds, and its
    parameter count is diffusers' own UNet2DConditionModel's on that config."""
    import jax
    import jax.numpy as jnp

    from dew.diffusion.process import DenoisingCondition
    from dew.nn.backbones import UNet2DCondition

    config = json.loads((SDXL / "unet/config.json").read_text())
    assert config["upcast_attention"] is None
    model = UNet2DCondition(**unet_fields(config))
    size, time_ids = config["sample_size"], 6
    pooled = config["projection_class_embeddings_input_dim"] - time_ids * config["addition_time_embed_dim"]
    condition = DenoisingCondition(jnp.zeros((1, 77, config["cross_attention_dim"])), jnp.zeros((1, pooled)),
                                   jnp.zeros((1, time_ids)))
    shapes = jax.eval_shape(lambda: model.init(
        jax.random.key(0), jnp.zeros((1, size, size, config["in_channels"])), jnp.zeros((1,)),
        conditioning=condition))
    assert sum(leaf.size for leaf in jax.tree.leaves(shapes)) == 2_567_463_684


@pytest.mark.parametrize("control, expected", [
    ("flip_sin_to_cos", False), ("use_linear_projection", False)])
def test_a_null_unet_flag_reads_as_false_as_diffusers_reads_it(edges, control, expected):
    """diffusers tests these flags' truth, so null is False even where the
    default is True (flip_sin_to_cos); a null fixed flag is its False."""
    config = json.loads((edges / "unet/config.json").read_text())
    fields = unet_fields({**config, control: None, "upcast_attention": None, "center_input_sample": None})
    assert {"flip_sin_to_cos": fields["cosine_first"],
            "use_linear_projection": fields["linear_projection"]}[control] is expected
