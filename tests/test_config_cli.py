"""The typed run config: serialization, CLI parsing, and what reaches the trainer."""

import collections
import importlib.util
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
import tyro

from dew.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data.dataloaders import load_data
from dew.inputs import ConditionalInputConfig, DiffusionInputConfig
from dew.inputs.encoders import ConditioningEncoder
from dew.registry import build_model
from dew.training import build_optimizer, prepare_process

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
RES = 32
PATCH = 4
TOKENS = 5


def load_recipe(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_train_recipe", RECIPES / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def populated_config(cls=RunConfig, **objective_knobs):
    """A config with nothing left at its default, so a round-trip has work to do."""
    return cls(
        model=ModelConfig("simple_dit+hilbert", {
            "patch_size": PATCH, "emb_features": 32, "num_layers": 2, "num_heads": 2,
        }),
        data=DataConfig(dataset="oxford_flowers102", batch_size=8, image_size=64,
                        loader="grain", augmentation_mode="flip_only", worker_count=2),
        optim=OptimConfig(optimizer="lamb", optimizer_opts={"b1": 0.95},
                          learning_rate=1e-3, learning_rate_schedule="cosine",
                          weight_decay=0.01, clip_grads=1.0, grad_accum_steps=3,
                          use_dynamic_scale=True),
        trainer=TrainerConfig(name="run-{dataset}", epochs=3, steps_per_epoch=7,
                              checkpoint_every_steps=11,
                              checkpoint_fs="gcs", fsdp_size=2,
                              logical_axis_rules={"mlp": "fsdp"},
                              sharding_tolerance=0.1, wandb_offline=True),
        **objective_knobs,
    )


def fake_dataset(*args, **kwargs):
    def batches():
        rs = np.random.RandomState(0)
        while True:
            yield {"image": jnp.asarray(rs.uniform(0, 255, (4, RES, RES, 3))),
                   "label": jnp.asarray(rs.randint(0, 4, 4))}
    return {"train": batches, "val": batches, "train_len": 16, "local_batch_size": 4}


def test_run_config_round_trips_through_json():
    config = populated_config()
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_recipe_configs_round_trip_too():
    diffusion = load_recipe("diffusion")
    config = populated_config(diffusion.DiffusionRunConfig, noise_schedule="flow",
                              flow_shift=3.0, min_snr_gamma=5.0,
                              val_metrics=["clip", "fid"], autoencoder="stable_diffusion")
    assert diffusion.DiffusionRunConfig.from_dict(config.to_dict()) == config


def test_tyro_parses_the_flags_into_the_same_config():
    parsed = tyro.cli(RunConfig, args=[
        "--model.architecture", "simple_dit+hilbert",
        "--model.config",
        f'{{"patch_size": {PATCH}, "emb_features": 32, "num_layers": 2, "num_heads": 2}}',
        "--data.dataset", "oxford_flowers102", "--data.batch-size", "8",
        "--data.image-size", "64", "--data.loader", "grain",
        "--data.augmentation-mode", "flip_only", "--data.worker-count", "2",
        "--optim.optimizer", "lamb", "--optim.optimizer-opts", '{"b1": 0.95}',
        "--optim.learning-rate", "1e-3", "--optim.learning-rate-schedule", "cosine",
        "--optim.weight-decay", "0.01", "--optim.clip-grads", "1.0",
        "--optim.grad-accum-steps", "3", "--optim.use-dynamic-scale",
        "--trainer.name", "run-{dataset}", "--trainer.epochs", "3",
        "--trainer.steps-per-epoch", "7", "--trainer.checkpoint-every-steps", "11",
        "--trainer.checkpoint-fs", "gcs", "--trainer.fsdp-size", "2",
        "--trainer.logical-axis-rules", '{"mlp": "fsdp"}',
        "--trainer.sharding-tolerance", "0.1", "--trainer.wandb-offline",
    ])
    assert parsed == populated_config()


def test_model_config_passes_through_to_the_registry():
    diffusion = load_recipe("diffusion")
    config = populated_config(diffusion.DiffusionRunConfig,
                              autoencoder="stable_diffusion")
    architecture, kwargs = diffusion.model_kwargs(config, channels=4, sample_size=8)

    assert architecture == 'simple_dit'
    # The suffix and the latent channels are the recipe's to add, the rest is
    # whatever --model.config said
    assert kwargs['use_hilbert'] is True
    assert kwargs['output_channels'] == 4
    assert kwargs['emb_features'] == 32

    model = build_model(architecture, kwargs)
    assert model.patch_size == PATCH
    assert model.emb_features == 32
    assert model.output_channels == 4
    assert model.scan_order == 'hilbert'


def test_grad_accum_steps_wraps_the_optimizer_and_reaches_the_trainer(tmp_path, monkeypatch):
    diffusion = load_recipe("diffusion")
    assert not isinstance(diffusion.build_optimizer(OptimConfig(), 10), optax.MultiSteps)
    assert isinstance(
        diffusion.build_optimizer(OptimConfig(grad_accum_steps=3), 10), optax.MultiSteps)
    jepa = load_recipe("jepa")
    monkeypatch.setattr(jepa, "load_data", fake_dataset)
    trainer = jepa.main(jepa.JepaRunConfig(
        model=ModelConfig("jepa_encoder", {
            "patch_size": PATCH, "emb_features": 16, "num_layers": 1, "num_heads": 2,
            "mlp_ratio": 2,
        }),
        data=DataConfig(image_size=RES, batch_size=4, val_steps_per_epoch=1),
        optim=OptimConfig(learning_rate=1e-3, grad_accum_steps=2),
        trainer=TrainerConfig(epochs=1, steps_per_epoch=2, distributed_training=False,
                              checkpoint_dir=str(tmp_path), compilation_cache_dir=None, multi_host=False),
        predictor={"predictor_features": 8, "num_layers": 1, "num_heads": 2},
    ))

    assert trainer.grad_accum_steps == 2
    assert isinstance(trainer.state.tx, optax.MultiSteps)


def reference_optimizer(config: OptimConfig, steps_per_epoch: int):
    """The recipes' old inline construction, verbatim, for equivalence."""
    learning_rate = config.learning_rate
    if config.learning_rate_schedule == 'cosine':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=config.learning_rate_peak,
            warmup_steps=config.learning_rate_warmup_steps,
            decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
            end_value=config.learning_rate_end,
        )
    opts = dict(config.optimizer_opts)
    if config.weight_decay is not None:
        opts['weight_decay'] = config.weight_decay
    solver = {'adam': optax.adam, 'adamw': optax.adamw, 'lamb': optax.lamb}[
        config.optimizer](learning_rate, **opts)
    if config.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(config.clip_grads), solver)
    if config.grad_accum_steps > 1:
        solver = optax.MultiSteps(solver, every_k_schedule=config.grad_accum_steps)
    return solver


def test_library_build_optimizer_accepts_muon():
    """The orthogonalized update flows to a matrix and the AdamW update to a
    vector, on a tree with no declared axes at all, which is what a library
    caller passes. Which group each parameter of a Dew model lands in is
    tests/test_optim.py's subject."""
    solver = build_optimizer(OptimConfig(optimizer="muon", learning_rate=1e-3), 10)
    params = {"w": jnp.ones((4, 6)), "b": jnp.zeros(6)}
    state = solver.init(params)
    updates, state = solver.update(jax.tree.map(jnp.ones_like, params), state, params)
    params = optax.apply_updates(params, updates)
    assert not jnp.array_equal(params["w"], jnp.ones((4, 6)))
    assert not jnp.array_equal(params["b"], jnp.zeros(6))
    # The same wiring under jit and gradient accumulation, which is how a run
    # actually wires it.
    solver = build_optimizer(OptimConfig(optimizer="muon", learning_rate=1e-3,
                                         grad_accum_steps=2), 10)
    run = jax.jit(lambda g, s, p: solver.update(g, s, p))
    _, _ = run(jax.tree.map(jnp.ones_like, params), solver.init(params), params)
    # The recipe config accepts the literal end to end.
    tyro.cli(OptimConfig, args=["--optimizer", "muon", "--learning-rate", "1e-3"])


def run_steps(solver, steps=9):
    """The updates a solver emits on a fixed gradient stream."""
    params = {"w": jnp.asarray([0.5, -0.25])}
    state = solver.init(params)
    out = []
    rng = np.random.RandomState(0)
    for _ in range(steps):
        grads = {"w": jnp.asarray(rng.randn(2))}
        updates, state = solver.update(grads, state, params)
        out.append(np.asarray(updates["w"]))
    return out


def old_inline_schedule(config: OptimConfig, steps_per_epoch: int):
    """The warmup-cosine schedule the config layer has to reproduce."""
    return optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate, peak_value=config.learning_rate_peak,
        warmup_steps=config.learning_rate_warmup_steps,
        decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
        end_value=config.learning_rate_end,
    )


def test_library_build_optimizer_runs_the_old_inline_cosine_schedule():
    config = OptimConfig(
        optimizer="adamw", optimizer_opts={"b1": 0.0, "b2": 0.0, "eps": 0.0},
        learning_rate=1e-3, learning_rate_schedule="cosine",
        learning_rate_peak=3e-3, learning_rate_end=1e-5,
        learning_rate_warmup_steps=3, learning_rate_decay_epochs=2,
    )
    steps_per_epoch = 5  # decay_steps = 5 * 2

    # b1=b2=eps=0 makes adamw emit -schedule(step) * sign(g), so the update
    # magnitudes are the learning rates the solver actually runs.
    grads = {"w": jnp.asarray([1.0, -1.0])}
    solver = build_optimizer(config, steps_per_epoch)
    state = solver.init(grads)
    for step in range(13):  # the warmup, the peak, the decay, the flat tail
        updates, state = solver.update(grads, state, grads)
        assert np.isclose(-float(updates["w"][0]),
                          float(old_inline_schedule(config, steps_per_epoch)(step)),
                          rtol=1e-4)


def test_library_build_optimizer_clips_the_global_norm():
    """Clip caps the gradient tree before the solver sees it. Adam's own
    normalization hides pure rescaling, so a large eps makes the cap show."""
    solver = build_optimizer(
        OptimConfig(learning_rate=1e-3, clip_grads=0.5, optimizer_opts={"eps": 1.0}), 10)
    grads = {"w": jnp.asarray([3.0, 4.0])}  # global norm 5, over the 0.5 cap
    a, _ = solver.update(grads, solver.init(grads), grads)

    # The same first stage optax would build by hand.
    reference = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(1e-3, eps=1.0))
    r, _ = reference.update(grads, reference.init(grads), grads)
    assert jnp.allclose(a["w"], r["w"])

    # A 10x twin caps to the same tree, so the same update.
    b, _ = solver.update(jax.tree.map(lambda g: g * 10, grads),
                         solver.init(grads), grads)
    assert jnp.allclose(a["w"], b["w"])

    # Without the cap the two differ, so it is the clip that made them equal.
    unclipped = build_optimizer(OptimConfig(learning_rate=1e-3,
                                            optimizer_opts={"eps": 1.0}), 10)
    c, _ = unclipped.update(grads, unclipped.init(grads), grads)
    d, _ = unclipped.update(jax.tree.map(lambda g: g * 10, grads),
                            unclipped.init(grads), grads)
    assert not jnp.allclose(c["w"], d["w"])


def test_load_data_dispatches_over_the_registries(monkeypatch):
    """The loader picks the factory from registry membership, not name spelling."""
    from dew.data import dataloaders

    calls = []
    sentinel = object()

    def record(name):
        def factory(*args, **kwargs):
            calls.append((name, args, kwargs))
            return sentinel
        return factory

    monkeypatch.setattr(dataloaders, "get_dataset_grain", record("grain"))
    monkeypatch.setattr(dataloaders, "get_media_dataset_grain", record("media"))
    monkeypatch.setattr(dataloaders, "get_dataset_online", record("online"))

    # A datasetMap name with loader='grain' goes to the legacy image factory.
    config = DataConfig(dataset="oxford_flowers102", loader="grain")
    assert load_data(config) is sentinel
    assert calls[0][0] == "grain"
    assert calls[0][2]["dataset_source"] == config.dataset_path

    # A mediaDatasetMap-only name (no datasetMap entry) goes to the media factory.
    calls.clear()
    load_data(DataConfig(dataset="voxceleb2", loader="grain"))
    assert calls[0][0] == "media"

    # An onlineDatasetMap name, forced online: the read thread/buffer scaling.
    calls.clear()
    config = DataConfig(dataset="combined_online", loader="online",
                        read_thread_count=10, worker_buffer_size=20)
    load_data(config)
    assert calls[0][0] == "online"
    assert calls[0][2]["read_thread_count"] == 40
    assert calls[0][2]["worker_buffer_size"] == 100

    # Auto mode streams only when the name is registered solely online.
    calls.clear()
    load_data(DataConfig(dataset="combined_online", loader="auto"))
    assert calls[0][0] == "online"

    calls.clear()
    load_data(DataConfig(dataset="oxford_flowers102", loader="auto"))
    assert calls[0][0] == "grain"

    calls.clear()
    load_data(DataConfig(dataset="voxceleb2", loader="auto"))
    assert calls[0][0] == "media"


# --- what a run does when nothing is configured ---------------------------

def test_compilation_cache_dir_defaults_into_the_cache_home(tmp_path, monkeypatch):
    """The cache is worth ~50s of the ~55s time-to-first-step on a DiT-B and
    changes nothing about the step, so it is on unless a run says otherwise."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert TrainerConfig().compilation_cache_dir == str(tmp_path / "dew" / "xla")

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert TrainerConfig().compilation_cache_dir == str(Path.home() / ".cache/dew/xla")

    off = tyro.cli(RunConfig, args=["--trainer.compilation-cache-dir", "None"])
    assert off.trainer.compilation_cache_dir is None


def test_defaults_name_no_person_and_no_course_project(monkeypatch):
    """No default carries a personal path or a course project's wandb team, so a
    fresh clone trains on neither."""
    # The cache dir is the one default read from the environment, and every
    # path this machine offers has the username in it; pin it to a neutral one
    # so the check is about the config rather than about who is logged in.
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-cache-home")
    configs = [RunConfig(),
               load_recipe("diffusion").DiffusionRunConfig(),
               load_recipe("jepa").JepaRunConfig()]

    for config in configs:
        recorded = json.dumps(config.to_dict())
        for personal in ("mrwhite0racle", "umd-projects", "msml605"):
            assert personal not in recorded

    assert DataConfig().dataset == "oxford_flowers102"
    assert DataConfig().dataset_path is None
    assert TrainerConfig().wandb_project is None
    assert TrainerConfig().wandb_entity is None


def test_resume_last_run_without_a_project_is_an_error():
    """It is a wandb run id, and with no project there is nothing to resume, so
    the run stops instead of starting over from step 0."""
    with pytest.raises(ValueError, match="wandb_project"):
        TrainerConfig(resume_last_run="9xk2p1")

    resumed = TrainerConfig(resume_last_run="9xk2p1", wandb_project="dew")
    assert resumed.resume_last_run == "9xk2p1"


def test_prepare_process_joins_the_pool_when_a_cluster_is_configured(monkeypatch):
    """A pod run whose initialize() fails stops, and so does a pod run that forgot
    a flag. The default asks the environment; only the single-host signature is
    allowed to continue."""
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "unset")
    joins = []
    monkeypatch.setattr(jax.distributed, "initialize", lambda *a, **k: joins.append(a))

    prepare_process("flip_only")
    assert len(joins) == 1
    assert os.environ["FLAXDIFF_AUGMENT_MODE"] == "flip_only"

    prepare_process("flip_only", multi_host=False)
    assert len(joins) == 1

    def single_host(*args, **kwargs):
        raise ValueError("coordinator_address should be defined.")

    monkeypatch.setattr(jax.distributed, "initialize", single_host)
    prepare_process("flip_only")
    with pytest.raises(ValueError, match="coordinator_address"):
        prepare_process("flip_only", multi_host=True)

    def half_configured(*args, **kwargs):
        raise ValueError("num_processes must be defined")

    monkeypatch.setattr(jax.distributed, "initialize", half_configured)
    with pytest.raises(ValueError, match="num_processes"):
        prepare_process("flip_only")


def test_xla_flags_reach_the_environment_before_the_pool_is_joined(monkeypatch):
    """XLA reads XLA_FLAGS when it opens a backend, so the flags have to be in
    the environment before the first JAX call, which is the pool join."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    seen = []
    monkeypatch.setattr(jax.distributed, "initialize",
                        lambda *a, **k: seen.append(os.environ["XLA_FLAGS"]))

    prepare_process("flip_only", xla_flags="--xla_gpu_triton_gemm_any=true")
    assert seen == ["--xla_force_host_platform_device_count=8 "
                    "--xla_gpu_triton_gemm_any=true"]

    prepare_process("flip_only")
    assert seen[-1] == os.environ["XLA_FLAGS"]


TOKENIZED = []


class StubTextEncoder(ConditioningEncoder):
    """A text encoder shaped like the CLIP one, with nothing to download."""

    @property
    def key(self):
        return "text"

    def tokenize(self, prompts):
        TOKENIZED.append(list(prompts))
        # A transformers tokenizer hands back a UserDict, which jax.tree counts
        # as a single leaf
        return collections.UserDict({
            "input_ids": np.zeros((len(prompts), TOKENS), np.int32),
            "attention_mask": np.ones((len(prompts), TOKENS), np.int32),
        })

    def encode_from_tokens(self, tokens):
        return jnp.zeros((len(tokens["input_ids"]), TOKENS, 8), jnp.float32)

    def serialize(self):
        return {"modelname": self.model}

    @staticmethod
    def deserialize(serialized_config):
        return StubTextEncoder(model=serialized_config["modelname"], tokenizer=None)


def stub_input_config(config):
    """build_input_config without CLIP-L/14's weights."""
    return DiffusionInputConfig(
        sample_data_key='image',
        sample_data_shape=(config.data.image_size, config.data.image_size, 3),
        conditions=[ConditionalInputConfig(
            encoder=StubTextEncoder(model="stub", tokenizer=None),
            conditioning_data_key='text',
            pretokenized=True,
            unconditional_input="",
            model_key_override="textcontext",
        )],
    )


def fake_captioned_dataset(*args, **kwargs):
    def batches():
        rs = np.random.RandomState(0)
        while True:
            yield {"image": jnp.asarray(rs.uniform(0, 255, (4, RES, RES, 3))),
                   "text": {"input_ids": np.zeros((4, TOKENS), np.int32),
                            "attention_mask": np.ones((4, TOKENS), np.int32)}}
    return {"train": batches, "val": batches, "train_len": 16,
            "local_batch_size": 4, "global_batch_size": 4}


def test_validation_prompts_feed_fixed_caption_batches(tmp_path):
    diffusion = load_recipe("diffusion")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a water lily\n\n a photo of a rose\n")

    batch_source, count = diffusion.validation_prompt_batches(
        str(prompts), StubTextEncoder(model="stub", tokenizer=None),
        batch_size=4, steps=2)
    batches = list(batch_source())

    assert count == 2
    # A short list wraps: val_steps_per_epoch always has batches to score
    assert len(batches) == 2
    assert batches[0]["text"]["input_ids"].shape == (4, TOKENS)
    # Tokens have to arrive as arrays, or shard_batch cannot place them
    assert len(jax.tree.leaves(batches[0])) == 2

    empty = tmp_path / "empty.txt"
    empty.write_text("\n \n")
    with pytest.raises(ValueError, match="No prompts"):
        diffusion.validation_prompt_batches(
            str(empty), StubTextEncoder(model="stub", tokenizer=None), 4, 2)


def test_diffusion_entrypoint_runs_without_wandb(tmp_path, monkeypatch):
    """The recipe end to end with no wandb project: nothing is logged, and
    validation samples the prompt file rather than a machine-local pickle."""
    diffusion = load_recipe("diffusion")
    monkeypatch.setattr(diffusion, "load_data", fake_captioned_dataset)
    monkeypatch.setattr(diffusion, "build_input_config", stub_input_config)
    TOKENIZED.clear()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a water lily\na photo of a rose\n")

    trainer = diffusion.main(diffusion.DiffusionRunConfig(
        model=ModelConfig("simple_dit", {
            "patch_size": PATCH, "emb_features": 32, "num_layers": 1, "num_heads": 2,
            "mlp_ratio": 1,
        }),
        data=DataConfig(image_size=RES, batch_size=4, val_steps_per_epoch=1),
        trainer=TrainerConfig(epochs=1, steps_per_epoch=2, distributed_training=False,
                              checkpoint_dir=str(tmp_path), compilation_cache_dir=None, multi_host=False),
        val_metrics=[],
        validation_prompts=str(prompts),
    ))

    assert trainer.wandb is None
    assert trainer.state.step == 2
    # The prompt file, wrapped to the batch size, reached the sampler on both
    # the sanity pass and the epoch-end one
    assert TOKENIZED.count(["a water lily", "a photo of a rose"] * 2) == 2
