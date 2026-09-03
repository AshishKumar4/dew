"""Inference pipeline tests: a checkpoint on disk through to a sample.

Everything below the artifact download works offline, so these drive the real
pipeline from a checkpoint the trainer has just written (the same path a
wandb run takes once the artifact has landed), and cover the wandb entry
points only where they report a failed lookup.

parse_config is exercised against both generations of logged config: the
current one, which carries a serialized input_config, and the older one, which
has none and has to have its text conditioning reconstructed from `arguments`.
"""

from pathlib import Path
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import orbax.checkpoint as ocp

from dew import sampling as inference
from dew.sampling import (
    DiffusionInferencePipeline, InferencePipeline, load_from_checkpoint, parse_config,
)
from dew.sampling import loading as inference_utils
from dew.inputs import (
    CONDITIONAL_ENCODERS_REGISTRY, ConditioningEncoder, DiffusionInputConfig,
)
from dew.nn.backbones.dit import SimpleDiT
from dew.diffusion.transforms import get_diffusion_preset
from dew.sampling.euler import EulerAncestralSampler
from dew.training import ObjectiveTrainer
from dew.checkpoints.utils import get_latest_checkpoint

RES = 8
MODEL_KWARGS = dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)


def make_trainer(tmp_path, name="inference", **kwargs):
    """The tiny unconditional DiT of tests/test_trainer.py, on one device."""
    train_schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(**MODEL_KWARGS),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image",
            sample_data_shape=(RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def current_config():
    """A config in the shape training.py logs today, for the model above."""
    return {
        "model": {
            "emb_features": 16,
            "dtype": "None",
            "precision": "default",
            "output_channels": 3,
            "attention_impl": None,
            "remat": False,
            "patch_size": 4,
            "num_layers": 1,
            "num_heads": 2,
            "dropout_rate": 0.0,
            "mlp_ratio": 1,
            "use_hilbert": False,
            "use_zigzag": False,
        },
        "architecture": "simple_dit",
        "arguments": {
            "architecture": "simple_dit",
            "image_size": RES,
            "noise_schedule": "edm",
        },
        "input_config": {
            "sample_data_key": "image",
            "sample_data_shape": (RES, RES, 3),
            "conditions": [],
        },
    }


def legacy_config():
    """A run from before input_config was logged.

    No 'input_config' and no top-level 'architecture': parse_config has to
    read the architecture out of `arguments` and synthesize the text
    conditioning that every run of that era used.
    """
    return {
        "model": {
            "emb_features": 64,
            "dtype": "bfloat16",
            "precision": "high",
            "output_channels": 3,
            "feature_depths": [16, 32],
            "attention_configs": [None, None],
            "num_res_blocks": 1,
            "num_middle_res_blocks": 1,
            "activation": "swish",
            "norm_groups": 4,
            "named_norms": False,
        },
        "arguments": {
            "architecture": "unet",
            "image_size": 128,
            "noise_schedule": "cosine",
            "autoencoder": None,
            "autoencoder_opts": "{}",
        },
    }


class StubTextEncoder(ConditioningEncoder):
    """Stands in for CLIP-L/14, which parse_config would otherwise download."""

    @property
    def key(self):
        return "text"

    def tokenize(self, data):
        return {"tokens": np.zeros((len(data), 77), np.int32)}

    def encode_from_tokens(self, tokens):
        return np.zeros((len(tokens["tokens"]), 77, 768), np.float32)

    def serialize(self):
        return {"modelname": self.model}

    @staticmethod
    def deserialize(serialized_config):
        return StubTextEncoder(model=serialized_config["modelname"], tokenizer=None)


# --------------------------------------------------------------------------
# Checkpoint -> pipeline -> samples
# --------------------------------------------------------------------------

def test_pipeline_generates_from_a_local_checkpoint(tmp_path):
    """The full offline path: a trainer checkpoint, the config that run logged,
    and a sample out of the reconstructed model."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    state = load_from_checkpoint(get_latest_checkpoint(trainer.checkpoint_path()))
    assert state.params is not None and state.ema_params is not None

    pipeline = DiffusionInferencePipeline.create(
        config=parse_config(current_config()), state=state)
    samples = pipeline.generate_samples(
        num_samples=2,
        resolution=RES,
        diffusion_steps=3,
        guidance_scale=0.0,
        sampler_class=EulerAncestralSampler,
        seed=0,
    )

    assert samples.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(samples))
    # post_process clips into the image range
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_pipeline_can_sample_from_the_ema_parameters(tmp_path):
    """The EMA copy is what a real run publishes, so it has to be reachable
    through the same call."""
    trainer = make_trainer(tmp_path, name="inference-ema")
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    state = load_from_checkpoint(get_latest_checkpoint(trainer.checkpoint_path()))
    pipeline = DiffusionInferencePipeline.create(
        config=parse_config(current_config()), state=state)
    samples = pipeline.generate_samples(
        num_samples=2, resolution=RES, diffusion_steps=3, guidance_scale=0.0,
        sampler_class=EulerAncestralSampler, seed=0, use_ema=True)

    assert samples.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(samples))


def test_missing_checkpoint_is_reported(tmp_path):
    """A load that failed must say so rather than hand back an empty state that
    reads as a successful one."""
    with pytest.raises(FileNotFoundError):
        load_from_checkpoint(str(tmp_path / "nothing-here"))


def test_load_from_checkpoint_picks_the_best_step(tmp_path):
    """'best' is the step whose epoch loss was lowest, which is not the newest
    one as soon as a run gets worse."""
    trainer = make_trainer(tmp_path, name="inference-best")
    trainer.save(epoch=0, step=1, metrics={"loss": 0.2})
    trainer.state = trainer.state.apply_gradients(
        grads=jax.tree.map(jnp.ones_like, trainer.state.params))
    trainer.save(epoch=1, step=2, metrics={"loss": 0.8})
    trainer.wait_for_checkpoints()

    best = load_from_checkpoint(trainer.checkpoint_path(), step="best")
    latest = load_from_checkpoint(trainer.checkpoint_path())
    assert int(best.step) == 0, "step 2 came back as the best"
    assert int(latest.step) == 1
    assert int(load_from_checkpoint(trainer.checkpoint_path(), step=1).step) == 0


def test_a_single_step_directory_is_its_own_best_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path, name="single-best")
    trainer.save(epoch=0, step=1, metrics={"loss": 0.2})
    trainer.wait_for_checkpoints()
    step_dir = get_latest_checkpoint(trainer.checkpoint_path())

    state = load_from_checkpoint(step_dir, step="best")
    assert int(state.step) == 0
def test_legacy_converter_preserves_the_best_state(tmp_path):
    source = tmp_path / "legacy"
    output = tmp_path / "converted"

    def state(step, marker):
        params = {
            "params": {"patch_embed": {"kernel": np.array([marker], np.float32)}}}
        return {"step": np.array(step, np.int32), "params": params,
                "ema_params": params}

    manager = ocp.CheckpointManager(
        source, options=ocp.CheckpointManagerOptions(create=True))
    manager.save(3, args=ocp.args.PyTreeSave({
        "state": state(3, 3.0),
        "best_state": state(1, 1.0),
        "best_loss": np.array(0.25),
        "rngs": np.array([0, 1], np.uint32),
        "epoch": 2,
    }), force=True)
    manager.wait_until_finished()

    tool = Path(__file__).parents[1] / "tools" / "convert_legacy_checkpoint.py"
    subprocess.run([sys.executable, str(tool), str(source), str(output)], check=True)

    latest = load_from_checkpoint(str(output), step="latest")
    best = load_from_checkpoint(str(output), step="best")
    assert float(latest.params["params"]["embed"]["patch_embed"]["kernel"][0]) == 3.0
    assert float(best.params["params"]["embed"]["patch_embed"]["kernel"][0]) == 1.0


def test_load_from_checkpoint_returns_the_saved_train_state(tmp_path):
    """The restored object carries the saved state's fields under the
    TrainState's own attribute names, with the saved values."""
    trainer = make_trainer(tmp_path, name="restored-fields")
    # One real update, so params, the EMA copy, the adam moments and the
    # step counter are all non-trivial and the fields can be told apart.
    trainer.state = trainer.state.apply_gradients(
        grads=jax.tree.map(jnp.ones_like, trainer.state.params)).apply_ema(0.99)
    trainer.save(epoch=0, step=1, metrics={"loss": 0.2})
    trainer.wait_for_checkpoints()

    restored = load_from_checkpoint(trainer.checkpoint_path())
    assert int(restored.step) == int(trainer.state.step)
    for field in ("params", "ema_params", "opt_state", "rngs", "metrics"):
        saved = jax.tree.leaves(getattr(trainer.state, field))
        loaded = jax.tree.leaves(getattr(restored, field))
        assert len(loaded) == len(saved) != 0, field
        for before, after in zip(saved, loaded, strict=True):
            np.testing.assert_allclose(np.asarray(before), np.asarray(after))
    assert restored.dynamic_scale == trainer.state.dynamic_scale is None


def test_a_resume_from_the_restored_state_trains(tmp_path):
    """The restored state goes back into a trainer as `train_state` and the
    run continues from the restored step, keeping the restored optimizer."""
    trainer = make_trainer(tmp_path, name="resume-source")
    trainer.state = trainer.state.apply_gradients(
        grads=jax.tree.map(jnp.ones_like, trainer.state.params)).apply_ema(0.99)
    trainer.save(epoch=0, step=1, metrics={"loss": 0.2})
    trainer.wait_for_checkpoints()
    restored = load_from_checkpoint(trainer.checkpoint_path())
    resumed = make_trainer(tmp_path, name="resumed", train_state=restored)
    resumed_step = int(restored.step)
    assert int(resumed.state.step) == resumed_step
    images = np.tile(
        np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
        (8, 1, RES, 3))

    def batches():
        while True:
            yield {"image": jnp.asarray(images)}

    # The resume lands mid-epoch, so one step completes epoch 0.
    data = {"train": batches, "train_len": 32, "local_batch_size": 8}
    state = resumed.fit(data, training_steps_per_epoch=2, epochs=1,
                        val_steps_per_epoch=0)
    assert int(state.step) == resumed_step + 1, "the resumed run trained nothing"
    assert all(np.all(np.isfinite(np.asarray(leaf)))
               for leaf in jax.tree.leaves(state.params)), "the resumed run diverged"

# --------------------------------------------------------------------------
# parse_config across config generations
# --------------------------------------------------------------------------

def test_parse_config_current_generation():
    parsed = parse_config(current_config())

    assert type(parsed["model"]).__name__ == "SimpleDiT"
    assert parsed["model"].emb_features == 16
    assert parsed["architecture"] == "simple_dit"
    assert parsed["autoencoder"] is None
    assert isinstance(parsed["input_config"], DiffusionInputConfig)
    assert parsed["input_config"].sample_data_shape == (RES, RES, 3)
    assert parsed["input_config"].conditions == []

    _, sampling_schedule, transform = get_diffusion_preset("edm")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert type(parsed["prediction_transform"]) is type(transform)


def test_parse_config_current_generation_rebuilds_conditions(monkeypatch):
    """A serialized condition has to come back with its pretokenization flag,
    which decides whether inference tokenizes or embeds directly."""
    monkeypatch.setitem(CONDITIONAL_ENCODERS_REGISTRY, "text", StubTextEncoder)
    config = current_config()
    config["input_config"]["conditions"] = [{
        "encoder": {"modelname": "openai/clip-vit-large-patch14"},
        "encoder_key": "text",
        "conditioning_data_key": "text",
        "pretokenized": True,
        "unconditional_input": "",
        "model_key_override": "textcontext",
    }]

    condition = parse_config(config)["input_config"].conditions[0]
    assert condition.pretokenized is True
    assert condition.model_key_override == "textcontext"
    assert condition.get_unconditional().shape == (1, 77, 768)


def test_parse_config_legacy_generation(monkeypatch):
    """No input_config at all: the text conditioning is reconstructed from the
    image size and the era's fixed conventions."""
    monkeypatch.setattr(inference_utils, "defaultTextEncodeModel",
                        lambda: StubTextEncoder(model="clip", tokenizer=None))
    parsed = parse_config(legacy_config())

    assert type(parsed["model"]).__name__ == "Unet"
    assert parsed["architecture"] == "unet"
    assert parsed["autoencoder"] is None

    input_config = parsed["input_config"]
    assert input_config.sample_data_key == "image"
    assert input_config.sample_data_shape == (128, 128, 3)

    condition, = input_config.conditions
    assert condition.conditioning_data_key == "text"
    assert condition.pretokenized is True
    assert condition.model_key_override == "textcontext"
    assert condition.unconditional_input == ""

    _, sampling_schedule, transform = get_diffusion_preset("cosine")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert type(parsed["prediction_transform"]) is type(transform)


def test_parse_config_overrides_reach_both_levels():
    parsed = parse_config(current_config(), overrides={"noise_schedule": "cosine"})

    _, sampling_schedule, _ = get_diffusion_preset("cosine")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert parsed["raw_config"]["arguments"]["noise_schedule"] == "cosine"


# --------------------------------------------------------------------------
# Public surface and the wandb entry points
# --------------------------------------------------------------------------

def test_package_exports_the_public_names():
    for name in ("InferencePipeline", "DiffusionInferencePipeline", "parse_config",
                 "load_from_checkpoint", "load_from_wandb_run",
                 "load_from_wandb_registry", "get_wandb_run", "RestoredState"):
        assert hasattr(inference, name), name


def test_base_pipeline_advertises_no_loader():
    """Loading is the concrete pipeline's business, so the base class declares
    no from_wandb of its own."""
    assert not hasattr(InferencePipeline, "from_wandb")
    assert hasattr(DiffusionInferencePipeline, "from_wandb_run")
    assert hasattr(DiffusionInferencePipeline, "from_wandb_registry")


def test_wandb_run_loader_reports_a_miss(monkeypatch):
    """A run that cannot be found returns Nones, rather than raising on an
    artifact name that was never bound."""
    monkeypatch.setattr(inference_utils, "get_wandb_run", lambda *a, **k: None)

    assert inference_utils.load_from_wandb_run(
        "no-such-run", project="p", entity="e") == (None, None, None, None)


def test_wandb_registry_loader_reports_a_miss(monkeypatch):
    def unreachable(*args, **kwargs):
        raise RuntimeError("wandb is unreachable")

    monkeypatch.setattr(inference_utils.wandb, "Api", unreachable)

    assert inference_utils.load_from_wandb_registry(
        "no-such-model", project="p") == (None, None, None, None)
