"""Every registry architecture, trained through the real trainer, on both meshes.

test_models.py proves each architecture has a working forward pass, and
test_parallelism.py proves the trainer's sharding on one tiny DiT. Between
them nothing ever put the other architectures' real parameter trees through
ObjectiveTrainer.fit, so an architecture could be unshardable, or silently
never sharded, and only a production run would find out.

Each case trains two steps on the simulated 8-device CPU mesh, once as pure
data parallelism (8x1) and once as data x fsdp (2x4), and checks what only a
real fit can check: finite losses out of the compiled step, parameters and
their optimizer moments genuinely split over the fsdp axis, the objective's
validation step running against the sharded EMA copy, and a checkpoint on
disk afterwards.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.diffusion.transforms import get_diffusion_preset
from dew.eval.common import EvaluationMetric
from dew.inputs import ConditionalInputConfig, DiffusionInputConfig
from dew.inputs.encoders import ConditioningEncoder
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.registry import MODEL_REGISTRY, build_model
from dew.training import ObjectiveTrainer

RES = 16
FRAMES = 2
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
# One batch element per simulated device, so the 8x1 and 2x4 meshes see the
# same global batch.
BATCH = 8
# These models hold thousands of parameters, orders below the production shard
# threshold, so lower it or "fsdp on" would mean "everything replicated".
TINY_SHARD = 256
# Enough to run the sampler loop end to end and nothing more: sample quality
# is not what a two-step run can be about.
SAMPLER_STEPS = 2
# 2x2 target blocks on the 4x4 grid, which leaves 8 context tokens.
MASK = multi_block_mask(GRID, num_targets=2, scale=(0.2, 0.3))

TEXT_TOKENS = 8
TEXT_FEATURES = 32


@dataclass
class StubTextEncoder(ConditioningEncoder):
    """Stands in for CLIP-L/14, whose weights would put a download in every
    case here. Pretokenized text of a fixed width, embedded by broadcast."""

    @property
    def key(self):
        return "text"

    def tokenize(self, data):
        return np.zeros((len(data), TEXT_TOKENS), np.int32)

    def encode_from_tokens(self, tokens):
        embedded = jnp.asarray(tokens, jnp.float32)[..., None] / TEXT_TOKENS
        return jnp.broadcast_to(embedded, (*embedded.shape[:2], TEXT_FEATURES))

    def serialize(self):
        return {}

    @staticmethod
    def deserialize(serialized_config):
        raise NotImplementedError("test-only encoder")


@dataclass(frozen=True)
class Case:
    """One architecture at the smallest size that still exercises it."""

    architecture: str
    config: dict
    frames: int = 0
    """Video architectures take (frames, H, W, C) samples; 0 means images."""
    predictor: Optional[dict] = None
    """Set for JEPA: `architecture` is the encoder and this builds its predictor."""
    fsdp_xfail: Optional[str] = None
    """Non-None records a real defect the fsdp leg hits, with its error."""

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def sample_shape(self) -> tuple:
        square = (RES, RES, 3)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def sample_key(self) -> str:
        return "video" if self.frames else "image"


DIT = {"patch_size": PATCH, "emb_features": 64, "num_layers": 2, "num_heads": 2,
       "mlp_ratio": 2}
UNET = {"emb_features": 64, "feature_depths": [16, 32],
        "attention_configs": [None, {"heads": 2, "use_projection": False,
                                     "use_self_and_cross": False}],
        "num_res_blocks": 1, "num_middle_res_blocks": 1}
ENCODER = {"patch_size": PATCH, "emb_features": 32, "num_layers": 2, "num_heads": 2,
           "mlp_ratio": 2}
PREDICTOR = {"grid": GRID, "emb_features": 32, "predictor_features": 16,
             "num_layers": 1, "num_heads": 2, "mlp_ratio": 2}

CASES = [
    Case("unet", UNET),
    # Both U-shaped stacks split their layers into a down and an up half
    Case("uvit", {"patch_size": PATCH, "emb_features": 64, "num_layers": 4,
                  "num_heads": 2}),
    Case("simple_udit", {**DIT, "num_layers": 2}),
    Case("simple_dit", DIT),
    Case("simple_mmdit", DIT),
    Case("hierarchical_mmdit", {"base_patch_size": 2, "emb_features": (32, 64, 96),
                                "num_layers": (1, 1, 1), "num_heads": (2, 2, 2),
                                "mlp_ratio": 2}),
    Case("hybrid_dit", {**DIT, "num_layers": 4, "ssm_state_dim": 8,
                        "ssm_attention_ratio": "3:1"}),
    Case("video_dit", {**DIT, "num_layers": 1}, frames=FRAMES),
    Case("unet_3d", {**UNET, "attention_configs": [None, None], "temporal_heads": 2},
         frames=FRAMES),
    Case("jepa_encoder", ENCODER, predictor=PREDICTOR),
    Case("jepa_video_encoder", {**ENCODER, "num_layers": 1},
         predictor={**PREDICTOR, "factorized": True}, frames=FRAMES),
]

# jepa_predictor has no training step of its own: it is built through the
# registry and trained inside the two JEPA cases.
COVERED = {case.architecture for case in CASES} | {"jepa_predictor"}

IDS = [case.architecture for case in CASES]


def test_every_registry_architecture_is_trained_here():
    """The point of the file: a new architecture must arrive with a trained case."""
    assert COVERED == set(MODEL_REGISTRY)


def benchmark_tool():
    """tools/ is a directory of scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_step",
        Path(__file__).resolve().parents[1] / "tools" / "benchmark_step.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_step_tool_measures_a_real_step(tmp_path):
    """tools/benchmark_step.py drives the same trainer internals this file
    does, so it rots the moment they move. One cpu-smoke case keeps it honest."""
    tool = benchmark_tool()
    rows = tool.run(tool.BenchmarkConfig(
        preset='cpu-smoke', architectures=['simple_dit'], warmup=1, steps=2,
        checkpoint_dir=str(tmp_path)))

    (row,) = rows
    assert row['ms_per_step'] > 0 and row['samples_per_sec'] > 0
    assert row['flops_per_step'] > 0
    assert row['params'] > 0
    assert row['finite'] and np.isfinite(row['loss'])
    assert row['device_kind'] == jax.devices()[0].device_kind
    if row['device_kind'] == 'cpu':
        # No published peak FLOPs for a CPU and no allocator stats behind it,
        # so these are honest Nones rather than invented numbers.
        assert row['utilization'] is None and row['peak_device_bytes'] is None
    assert tool.format_table(rows).count("\n") == 2


def test_benchmark_small_preset_profiles_every_architecture():
    """--preset small is the GPU sweep, so an architecture missing from it is
    an architecture nobody has ever profiled."""
    tool = benchmark_tool()
    covered = ({case.architecture for case in tool.small_cases('bfloat16')}
               | {"jepa_predictor"})
    assert covered == set(MODEL_REGISTRY)


def text_condition() -> ConditionalInputConfig:
    return ConditionalInputConfig(
        encoder=StubTextEncoder(model=None, tokenizer=None),
        conditioning_data_key="text",
        pretokenized=True,
        unconditional_input="",
        model_key_override="textcontext",
    )


def batches(case: Case):
    """uint8-range samples, as the data pipeline delivers them."""
    rng = np.random.default_rng(0)
    batch = {case.sample_key: rng.integers(
        0, 256, size=(BATCH, *case.sample_shape)).astype(np.float32)}
    if not case.is_jepa:
        batch["text"] = np.ones((BATCH, TEXT_TOKENS), np.int32)

    def source():
        while True:
            yield batch

    return source


def shape_metric(seen):
    """A real EvaluationMetric over the objective's validation artifacts.

    The validation loop reports exceptions rather than raising them, so an
    assertion inside a metric would be swallowed. Recording here and asserting
    afterwards is what makes a validation step that never ran a failure.
    """
    def record(artifacts, batch):
        artifacts = np.asarray(artifacts)
        seen.append(artifacts.shape)
        return float(artifacts.std())

    return EvaluationMetric(function=record, name="artifact_spread")


def record_step_losses(trainer, losses):
    """Per-step losses. fit() only returns the epoch mean, and one non-finite
    step out of two is exactly what this file is looking for."""
    compile_step = trainer._compiled_step

    def compiled(train_step_fn, *args):
        step = compile_step(train_step_fn, *args)

        def recording(state, rng, batch):
            result = step(state, rng, batch)
            losses.append(float(result[1]))
            return result

        return recording

    trainer._compiled_step = compiled


def make_trainer(case: Case, tmp_path, fsdp_size, seen):
    """The trainer a recipe would build for this case: a registry model, the
    real objective, the real optimizer, no wandb."""
    model = build_model(case.architecture, case.config)
    common = dict(
        model=model,
        optimizer=optax.adam(1e-3),
        rngs=jax.random.PRNGKey(0),
        name=f"{case.architecture}-fsdp{fsdp_size}",
        wandb_config=None,
        distributed_training=True,
        fsdp_size=fsdp_size,
        fsdp_min_param_size=TINY_SHARD,
        checkpoint_base_path=str(tmp_path),
        eval_metrics=[shape_metric(seen)],
    )

    if case.is_jepa:
        objective = JepaObjective(
            model,
            build_model("jepa_predictor", case.predictor),
            MASK,
            case.sample_key,
            case.sample_shape,
        )
        return ObjectiveTrainer(
            input_config=DiffusionInputConfig(
                sample_data_key=case.sample_key,
                sample_data_shape=case.sample_shape,
                conditions=[]),
            objective=objective,
            **common,
        )

    train_schedule, _, transform = get_diffusion_preset("edm")
    trainer = ObjectiveTrainer(
        input_config=DiffusionInputConfig(
            sample_data_key=case.sample_key,
            sample_data_shape=case.sample_shape,
            conditions=[text_condition()]),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        **common,
    )
    # 200 sampler steps is the production default and pure overhead here
    trainer.objective.diffusion_steps = SAMPLER_STEPS
    return trainer


def fsdp_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if 'fsdp' in str(leaf.sharding.spec)]


def run_case(case: Case, tmp_path, fsdp_size):
    """One two-step run on the mesh, with everything the assertions read."""
    seen, losses = [], []
    trainer = make_trainer(case, tmp_path, fsdp_size, seen)
    assert trainer.mesh.shape["data"] == jax.device_count() // fsdp_size
    assert trainer.mesh.shape["fsdp"] == fsdp_size
    record_step_losses(trainer, losses)

    source = batches(case)
    data = {"train": source, "val": source, "train_len": BATCH * 8,
            "local_batch_size": BATCH, "global_batch_size": BATCH}
    state = trainer.fit(data, training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=1)

    assert_run_landed(trainer, state, case, seen, losses)
    return trainer, state


def assert_run_landed(trainer, state, case, seen, losses):
    """Two real steps, a validation pass over the EMA copy, a checkpoint."""
    assert int(state.step) == 2
    assert len(losses) == 2 and all(np.isfinite(loss) for loss in losses), losses

    # fit() validates once as a pre-training sanity check and once after the
    # epoch, both from the EMA parameters as they sit on the mesh.
    if case.is_jepa:
        expected = (BATCH, case.config["emb_features"])
    else:
        expected = (BATCH, *case.sample_shape)
    assert seen == [expected] * 2, seen
    assert np.isfinite(trainer.best_val_metrics["val/artifact_spread"])

    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == 2
    assert os.path.isdir(os.path.join(trainer.checkpoint_path(), "2"))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_architecture_trains_data_parallel(case, tmp_path):
    """8x1: every parameter replicated, the batch split across every device."""
    _, state = run_case(case, tmp_path, fsdp_size=1)
    assert not fsdp_leaves(state.params), "nothing may shard on a 1-wide fsdp axis"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_architecture_trains_under_fsdp(case, tmp_path):
    """2x4: the parameter tree really split four ways, moments and EMA with it."""
    if case.fsdp_xfail:
        pytest.xfail(case.fsdp_xfail)
    _, state = run_case(case, tmp_path, fsdp_size=4)

    sharded = fsdp_leaves(state.params)
    assert sharded, "no parameter was sharded over the fsdp axis"
    for param in sharded:
        assert param.addressable_shards[0].data.size == param.size // 4, \
            "shard is not a quarter of the global parameter"

    # Adam's moments and the EMA copy follow the params they track, without the
    # optimizer or the model ever describing a layout.
    param_specs = [leaf.sharding.spec for leaf in jax.tree.leaves(state.params)]
    assert param_specs == [leaf.sharding.spec
                           for leaf in jax.tree.leaves(state.opt_state[0].mu)]
    assert param_specs == [leaf.sharding.spec
                           for leaf in jax.tree.leaves(state.ema_params)]
    assert fsdp_leaves(state.opt_state[0].mu), "no optimizer moment was sharded"
