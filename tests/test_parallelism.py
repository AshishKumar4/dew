"""Sharding, FSDP and data-pipeline tests on a simulated 8-device CPU mesh.

The parity tests are the safety net for the shard_map -> jit + NamedSharding
migration: a partitioned run has to produce the same numbers as a single-device
one, otherwise the collectives GSPMD derived are not the ones we meant.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from dew.inputs import DiffusionInputConfig
from dew.eval.common import EvaluationMetric
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
from dew.diffusion.transforms import get_diffusion_preset
from dew.training import ObjectiveTrainer, SimpleTrainer
from dew.objectives.base import EMASpec, Objective
from dew.training.distributed import (
    DEFAULT_LOGICAL_AXIS_RULES, DevicePrefetchIterator, batch_sharding, build_mesh,
    parameter_spec, shard_batch, state_sharding_tree,
)

RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# lower it or "FSDP on" would silently mean "everything replicated".
TINY = 256


def make_trainer(tmp_path, name, distributed_training, fsdp_size=1,
                 optimizer=None, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3) if optimizer is None else optimizer,
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=distributed_training,
        fsdp_size=fsdp_size,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batches():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(BATCH, RES, RES, 3)).astype(np.float32)
    while True:
        yield {"image": images}


def run_losses(trainer, steps):
    """Per-step losses from the real compiled training step."""
    train_step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate
    losses = []
    for _ in range(steps):
        state, loss, _, rng, is_finite = train_step(state, rng, next(source))
        assert bool(is_finite)
        losses.append(float(loss))
    return losses


# --------------------------------------------------------------------------
# Sharding heuristic
# --------------------------------------------------------------------------

def test_parameter_spec_replicates_without_fsdp():
    assert parameter_spec((1024, 1024), fsdp_size=1, min_shard_size=16) == P()


def test_parameter_spec_replicates_small_params():
    assert parameter_spec((8, 8), fsdp_size=2, min_shard_size=2 ** 16) == P()


def test_parameter_spec_shards_largest_divisible_axis():
    assert parameter_spec((64, 1024), fsdp_size=2, min_shard_size=16) == P(None, 'fsdp')
    assert parameter_spec((1024, 64), fsdp_size=2, min_shard_size=16) == P('fsdp')


def test_parameter_spec_falls_back_to_replication_when_indivisible():
    assert parameter_spec((15, 15), fsdp_size=2, min_shard_size=16) == P()


def declared_specs(variables, rules):
    """Parameter specs on a two-way fsdp mesh under one rule table."""
    shardings = state_sharding_tree(
        build_mesh(fsdp_size=2), variables, min_shard_size=1,
        logical_axis_rules=rules)
    return jax.tree.map(lambda sharding: sharding.spec, shardings)["params"]


def test_causal_transformer_axes_land_on_the_dimensions_they_name():
    """One rule at a time: the dimension that moves is the one the table names,
    which is what a declared axis has to mean."""
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=1, num_heads=2,
        num_kv_heads=1, mlp_ratio=2, max_seq_len=8, tie_embeddings=False)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))

    vocab = declared_specs(variables, {"vocab": "fsdp"})
    assert vocab["embed_tokens"]["embedding"] == P("fsdp")
    assert vocab["lm_head"]["kernel"] == P(None, "fsdp")

    # Grouped-query k and v are the 'kv' axis, q is 'heads': a rule for one
    # must leave the other whole.
    kv = declared_specs(variables, {"kv": "fsdp"})
    attention = kv["layers_0"]["self_attn"]
    assert attention["k_proj"]["kernel"] == P(None, "fsdp")
    assert attention["v_proj"]["kernel"] == P(None, "fsdp")
    assert attention["q_proj"]["kernel"] == P()

    heads = declared_specs(variables, {"heads": "fsdp"})["layers_0"]["self_attn"]
    assert heads["q_proj"]["kernel"] == P(None, "fsdp")
    assert heads["k_proj"]["kernel"] == P()

    out = declared_specs(variables, {"attention": "fsdp"})["layers_0"]["self_attn"]
    assert out["o_proj"]["kernel"] == P("fsdp")

    mlp = declared_specs(variables, {"mlp": "fsdp"})["layers_0"]["mlp"]
    assert mlp["gate_proj"]["kernel"] == P(None, "fsdp")
    assert mlp["up_proj"]["kernel"] == P(None, "fsdp")
    assert mlp["down_proj"]["kernel"] == P("fsdp")

    embed = declared_specs(variables, {"embed": "fsdp"})
    assert embed["embed_tokens"]["embedding"] == P(None, "fsdp")
    assert embed["lm_head"]["kernel"] == P("fsdp")


def test_dit_axes_land_on_the_dimensions_they_name():
    """The same for the DiT stack, whose attention kernels carry three axes."""
    model = SimpleDiT(
        patch_size=4, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=2)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), None)

    heads = declared_specs(variables, {"heads": "fsdp"})["dit_block_0"]["attention"]
    assert heads["to_q"]["kernel"] == P(None, "fsdp")
    assert heads["to_out_0"]["kernel"] == P("fsdp")

    head_dim = declared_specs(
        variables, {"head_dim": "fsdp"})["dit_block_0"]["attention"]
    assert head_dim["to_q"]["kernel"] == P(None, None, "fsdp")
    assert head_dim["to_out_0"]["kernel"] == P(None, "fsdp")

    embed = declared_specs(variables, {"embed": "fsdp"})
    assert embed["embed"]["patch_embed"]["Conv_0"]["kernel"] == P(None, None, None, "fsdp")
    assert embed["conditioning"]["time_embed"]["layers_2"]["kernel"] == P(None, "fsdp")

    modulation = declared_specs(variables, {"modulation": "fsdp"})
    assert (modulation["dit_block_0"]["ada_params_module"]["ada_proj"]["kernel"]
            == P(None, "fsdp"))

    output = declared_specs(variables, {"output": "fsdp"})
    assert output["output"]["final_proj"]["kernel"] == P(None, "fsdp")

    mlp = declared_specs(variables, {"mlp": "fsdp"})
    assert mlp["conditioning"]["time_embed"]["layers_2"]["kernel"] == P("fsdp")


def test_rule_override_changes_only_declared_axes():
    model = SimpleDiT(
        patch_size=4, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=2)
    abstract_variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), None)
    specs = declared_specs(abstract_variables, {"mlp": "fsdp"})

    assert specs["dit_block_0"]["mlp"]["layers_0"]["kernel"] == P(None, "fsdp")
    assert specs["dit_block_0"]["mlp"]["layers_2"]["kernel"] == P("fsdp")
    assert specs["dit_block_0"]["attention"]["to_q"]["kernel"] == P()
    assert specs["embed"]["patch_embed"]["Conv_0"]["kernel"] == P()


def test_default_logical_rules_keep_the_shape_heuristic():
    model = SimpleDiT(
        patch_size=4, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=2)
    abstract_variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), None)
    shardings = state_sharding_tree(
        build_mesh(fsdp_size=2), abstract_variables, min_shard_size=1,
        logical_axis_rules=DEFAULT_LOGICAL_AXIS_RULES)
    expected = jax.tree.map(
        lambda leaf: parameter_spec(leaf.shape, fsdp_size=2, min_shard_size=1),
        abstract_variables)
    actual = jax.tree.map(lambda sharding: sharding.spec, shardings)
    assert actual == expected


def test_rules_may_name_a_mesh_axis_this_mesh_does_not_have():
    """A table can carry the future tensor axis; today's mesh drops it."""
    model = SimpleDiT(
        patch_size=4, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=2)
    abstract_variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), None)
    shardings = state_sharding_tree(
        build_mesh(fsdp_size=2), abstract_variables, min_shard_size=1,
        logical_axis_rules={"mlp": ["tensor", "fsdp"], "heads": "tensor"})["params"]

    assert (shardings["dit_block_0"]["mlp"]["layers_0"]["kernel"].spec
            == P(None, "fsdp"))
    assert shardings["dit_block_0"]["attention"]["to_q"]["kernel"].spec == P()


class IndivisibleModel(nn.Module):
    """A parameter no mesh axis of size two can split."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(15, use_bias=False, name="indivisible")(x)


def make_indivisible_trainer(tmp_path, tolerance):
    return SimpleTrainer(
        model=IndivisibleModel(),
        input_shapes={"x": (15,)},
        optimizer=optax.adam(1e-3),
        rngs=jax.random.key(0),
        distributed_training=True,
        fsdp_size=2,
        fsdp_min_param_size=1,
        sharding_tolerance=tolerance,
        checkpoint_base_path=str(tmp_path),
        name=f"indivisible-{tolerance}",
    )


def test_sharding_tolerance_names_the_largest_replicated_parameter(tmp_path):
    with pytest.raises(ValueError) as error:
        make_indivisible_trainer(tmp_path, tolerance=0.02)

    message = str(error.value)
    assert "100.00%" in message and "2.00%" in message
    assert "['params']['indivisible']['kernel']" in message
    assert "(15, 15)" in message


def test_sharding_tolerance_can_allow_intentional_replication(tmp_path):
    trainer = make_indivisible_trainer(tmp_path, tolerance=1.0)
    kernel = trainer.state.params["params"]["indivisible"]["kernel"]
    assert kernel.sharding.spec == P()


def test_build_mesh_rejects_bad_fsdp_size():
    with pytest.raises(ValueError):
        build_mesh(fsdp_size=3)


def test_build_mesh_axes():
    mesh = build_mesh(fsdp_size=2)
    assert mesh.shape['data'] == jax.device_count() // 2
    assert mesh.shape['fsdp'] == 2


# --------------------------------------------------------------------------
# Batch placement and prefetch
# --------------------------------------------------------------------------

def test_shard_batch_splits_across_all_devices():
    mesh = build_mesh(fsdp_size=2)
    batch = {"image": np.zeros((jax.device_count(), 4), np.float32)}
    sharded = shard_batch(batch_sharding(mesh), batch)["image"]
    assert len(sharded.addressable_shards) == jax.device_count()
    assert sharded.addressable_shards[0].data.shape == (1, 4)


def test_prefetch_iterator_preserves_order_and_terminates():
    mesh = build_mesh()
    source = ({"x": np.full((jax.device_count(), 2), i, np.float32)} for i in range(5))
    it = DevicePrefetchIterator(source, batch_sharding(mesh), depth=2)
    seen = [float(np.asarray(b["x"])[0, 0]) for b in it]
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]
    with pytest.raises(StopIteration):
        next(it)


def test_prefetch_iterator_surfaces_source_errors():
    mesh = build_mesh()

    def broken():
        yield {"x": np.zeros((jax.device_count(), 2), np.float32)}
        raise ValueError("source exploded")

    it = DevicePrefetchIterator(broken(), batch_sharding(mesh), depth=2)
    next(it)
    with pytest.raises(ValueError, match="source exploded"):
        next(it)


def test_prefetch_iterator_tracks_checkpointable_source_state():
    """The position handed out must be the consumed batch's, not the thread's."""
    import grain.python as pygrain

    loader = pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, 64, 1),
        sampler=pygrain.IndexSampler(num_records=64, shuffle=False, seed=0, num_epochs=1,
                                     shard_options=pygrain.NoSharding()),
        operations=[pygrain.Batch(jax.device_count(), drop_remainder=True)],
        worker_count=0,
    )
    mesh = build_mesh()
    it = DevicePrefetchIterator(iter(loader), batch_sharding(mesh), depth=2)
    next(it)
    next(it)
    state = it.source_state
    expected = np.asarray(next(it))

    resumed = DevicePrefetchIterator(iter(loader), batch_sharding(mesh), depth=2,
                                     source_state=state)
    assert np.array_equal(np.asarray(next(resumed)), expected)


# --------------------------------------------------------------------------
# Numerical parity
# --------------------------------------------------------------------------

def test_single_and_multi_device_losses_agree(tmp_path):
    """The whole point of the migration: partitioning must not change the maths."""
    steps = 20
    single = run_losses(make_trainer(tmp_path / "one", "one", distributed_training=False), steps)
    multi = run_losses(make_trainer(tmp_path / "many", "many", distributed_training=True), steps)
    assert jax.device_count() > 1
    np.testing.assert_allclose(single, multi, rtol=2e-4, atol=2e-5)


def test_fsdp_losses_match_replicated(tmp_path):
    """Sharding the parameters must not change the loss trajectory."""
    steps = 20
    replicated = run_losses(
        make_trainer(tmp_path / "dp", "dp", distributed_training=True, fsdp_size=1), steps)
    fsdp = make_trainer(tmp_path / "fsdp", "fsdp", distributed_training=True,
                        fsdp_size=2, fsdp_min_param_size=TINY)
    assert any('fsdp' in str(x.sharding.spec) for x in jax.tree.leaves(fsdp.state.params))
    np.testing.assert_allclose(replicated, run_losses(fsdp, steps), rtol=2e-4, atol=2e-5)


# --------------------------------------------------------------------------
# FSDP actually shards
# --------------------------------------------------------------------------

def test_fsdp_shards_parameters_and_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path, "fsdp-shapes", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    leaves = jax.tree.leaves(trainer.state.params)
    sharded = [x for x in leaves if 'fsdp' in str(x.sharding.spec)]
    assert sharded, "no parameter was sharded over the fsdp axis"

    for param in sharded:
        local = param.addressable_shards[0].data
        assert local.size == param.size // 2, "shard is not half the global param"
        # Exactly the dimension the spec names is halved. Which dimension that
        # is belongs to the rules table, not to this test.
        split = [axis for axis, (whole, part) in enumerate(
            zip(param.shape, local.shape)) if whole != part]
        assert len(split) == 1, f"{param.shape} -> {local.shape}"
        assert param.shape[split[0]] // 2 == local.shape[split[0]]
        assert param.sharding.spec[split[0]] == 'fsdp'

    # Adam moments and the EMA copy must follow the params they track, without
    # the optimizer or the model ever describing a layout.
    mu = trainer.state.opt_state[0].mu
    param_specs = [x.sharding.spec for x in jax.tree.leaves(trainer.state.params)]
    mu_specs = [x.sharding.spec for x in jax.tree.leaves(mu)]
    assert param_specs == mu_specs

    ema_specs = [x.sharding.spec for x in jax.tree.leaves(trainer.state.ema_params)]
    assert param_specs == ema_specs


def test_replicated_run_shards_nothing(tmp_path):
    trainer = make_trainer(tmp_path, "dp-shapes", distributed_training=True, fsdp_size=1)
    for leaf in jax.tree.leaves(trainer.state.params):
        assert leaf.sharding.spec == P()


# --------------------------------------------------------------------------
# Checkpointing under sharding
# --------------------------------------------------------------------------

def test_sharded_checkpoint_roundtrips(tmp_path):
    trainer = make_trainer(tmp_path, "ckpt", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads).apply_ema(0.99)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "ckpt", distributed_training=True, fsdp_size=2,
                            fsdp_min_param_size=TINY,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1
    for before, after in zip(jax.tree.leaves(trainer.state.params),
                             jax.tree.leaves(restored.state.params)):
        assert before.sharding.spec == after.sharding.spec
        np.testing.assert_allclose(np.asarray(before), np.asarray(after))


# The last commit before parameters carried logical axes. A checkpoint written
# by that code has to keep restoring: unboxing before the save is what keeps
# the leaf names and the on-disk layout out of the annotations' reach.
PRE_LOGICAL_AXES_REV = "139d241"

PRE_LOGICAL_AXES_SAVE = '''
"""Save a checkpoint with whatever dew is first on sys.path."""
import sys

import jax
import numpy as np
import optax

import dew
from dew.diffusion.transforms import get_diffusion_preset
from dew.inputs import DiffusionInputConfig
from dew.nn.backbones.dit import SimpleDiT
from dew.training import ObjectiveTrainer

run_dir, params_dump = sys.argv[1:]
print(dew.__file__)
schedule, _, transform = get_diffusion_preset("edm")
trainer = ObjectiveTrainer(
    model=SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                    mlp_ratio=1),
    optimizer=optax.adam(1e-3),
    noise_schedule=schedule,
    model_output_transform=transform,
    input_config=DiffusionInputConfig(
        sample_data_key="image", sample_data_shape=(8, 8, 3), conditions=[]),
    rngs=jax.random.PRNGKey(0),
    name="pre-logical-axes",
    wandb_config=None,
    distributed_training=True,
    fsdp_size=2,
    fsdp_min_param_size=256,
    checkpoint_base_path=run_dir,
)
# Off the initial values, so a restore that quietly re-initialised instead of
# reading the file would show up as a mismatch rather than as a pass.
trainer.state = trainer.state.replace(
    params=jax.tree.map(lambda p: p + 0.25, trainer.state.params))
trainer.save(epoch=0, step=3)
trainer.wait_for_checkpoints()
flat, _ = jax.tree_util.tree_flatten_with_path(trainer.state.params)
np.savez(params_dump, **{jax.tree_util.keystr(path): np.asarray(value)
                         for path, value in flat})
'''


def test_checkpoint_written_before_logical_axes_still_restores(tmp_path):
    """Frozen checkpoint layout: the annotations must not reach the file."""
    repository = Path(__file__).resolve().parents[1]
    worktree = tmp_path / "pre-logical-axes-worktree"
    subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "--detach", "--quiet",
         str(worktree), PRE_LOGICAL_AXES_REV],
        check=True, capture_output=True, text=True)
    try:
        script = tmp_path / "save_pre_logical_axes.py"
        script.write_text(PRE_LOGICAL_AXES_SAVE)
        saved = subprocess.run(
            [sys.executable, str(script), str(tmp_path / "run"),
             str(tmp_path / "params.npz")],
            check=True, capture_output=True, text=True, cwd=str(worktree),
            env={**os.environ, "PYTHONPATH": str(worktree / "src"),
                 "JAX_PLATFORMS": "cpu"})
    finally:
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "remove", "--force",
             str(worktree)], capture_output=True, text=True)

    # Without this the test could quietly compare this code against itself.
    assert str(worktree) in saved.stdout, saved.stdout

    written = np.load(tmp_path / "params.npz")
    restored = make_trainer(
        tmp_path, "pre-logical-axes", distributed_training=True, fsdp_size=2,
        fsdp_min_param_size=TINY,
        load_from_checkpoint=str(tmp_path / "run" / "pre-logical-axes"))
    leaves = {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(
            restored.state.params)[0]}

    assert set(leaves) == set(written.files)
    for name, value in leaves.items():
        np.testing.assert_array_equal(value, written[name], err_msg=name)


def test_gradient_accumulation_updates_only_on_the_boundary(tmp_path):
    """MultiSteps must hold the params still until k micro-batches have run.

    Also covers the accumulator surviving the sharding heuristic: its buffers
    are param-shaped, so they pick up the param specs.
    """
    accum = 3
    trainer = make_trainer(tmp_path, "accum", distributed_training=True, fsdp_size=2,
                           fsdp_min_param_size=TINY, grad_accum_steps=accum,
                           optimizer=optax.MultiSteps(optax.sgd(0.5), every_k_schedule=accum))

    train_step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate

    def snapshot(s):
        return [np.asarray(x).copy() for x in jax.tree.leaves(s.params)]

    reference = snapshot(state)
    for micro in range(1, accum * 2 + 1):
        state, _, _, rng, _ = train_step(state, rng, next(source))
        moved = any(not np.array_equal(a, b) for a, b in zip(reference, snapshot(state)))
        at_boundary = micro % accum == 0
        assert moved == at_boundary, f"micro-step {micro}: moved={moved}"
        if at_boundary:
            reference = snapshot(state)


class DeterministicObjective(Objective):
    """Squared error against the input, straight through the real model.

    No noise level, no dropout, no unconditional mask: the loss depends only
    on the parameters and the batch. That is what makes two accumulation
    regimes comparable - every micro-gradient in a window is identical, which
    is exactly the "one big batch" an accumulated update stands in for, so a
    k-accumulated run and a plain run must trace the same parameters.
    """

    tag = "deterministic"

    def __init__(self, model, input_shapes, decay):
        self.model = model
        self.input_shapes = input_shapes
        self.ema = EMASpec(decay=decay)

    def init_params(self, rng):
        return self.model.init(
            rng, **{k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()})

    def loss(self, params, ema_params, batch, rng, step):
        data = (jnp.asarray(batch["image"], jnp.float32) - 127.5) / 127.5
        preds = self.model.apply(params, data, jnp.zeros((data.shape[0],), jnp.float32))
        return jnp.mean((preds - data) ** 2), {}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: None


def make_deterministic_trainer(tmp_path, name, grad_accum_steps):
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    input_config = DiffusionInputConfig(
        sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[])
    optimizer = optax.sgd(0.5)
    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=grad_accum_steps)
    return ObjectiveTrainer(
        model=model,
        optimizer=optimizer,
        input_config=input_config,
        # A ramp rather than a constant, so indexing the schedule by micro-step
        # instead of by update is visible in the result.
        objective=DeterministicObjective(
            model, input_config.get_input_shapes(),
            optax.linear_schedule(0.9, 1.0, transition_steps=8)),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=True,
        fsdp_size=2,
        fsdp_min_param_size=TINY,
        checkpoint_base_path=str(tmp_path),
        grad_accum_steps=grad_accum_steps,
    )


def ema_snapshot(state):
    return [np.asarray(x).copy() for x in jax.tree.leaves(state.ema_params)]


def test_ema_moves_only_on_accumulation_boundaries(tmp_path):
    """The EMA has to run on the optimizer's clock, not the micro-batch's.

    Between boundaries MultiSteps holds the params still, so an average taken
    every micro-step would blend in parameters that never changed and pull the
    EMA k times as far per update.
    """
    accum = 4
    trainer = make_deterministic_trainer(tmp_path, "ema-accum", accum)
    train_step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate

    reference = ema_snapshot(state)
    for micro in range(1, accum * 2 + 1):
        state, _, _, rng, _ = train_step(state, rng, next(source))
        moved = any(not np.array_equal(a, b)
                    for a, b in zip(reference, ema_snapshot(state)))
        at_boundary = micro % accum == 0
        assert moved == at_boundary, f"micro-step {micro}: ema moved={moved}"
        if at_boundary:
            reference = ema_snapshot(state)


def test_accumulated_ema_matches_a_plain_run_at_equal_update_counts(tmp_path):
    """k micro-batches of the same data must land where one big batch would,
    EMA included - same params, same decay index, same average."""
    accum, updates = 4, 3

    def run(name, k):
        trainer = make_deterministic_trainer(tmp_path / name, name, k)
        train_step = trainer._define_train_step(batch_size=BATCH)
        source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
        state, rng = trainer.state, trainer.rngstate
        for _ in range(updates * k):
            state, _, _, rng, _ = train_step(state, rng, next(source))
        return state

    plain = run("plain", 1)
    accumulated = run("accumulated", accum)

    # the comparison is only meaningful if the EMA left its starting point
    assert any(not np.allclose(np.asarray(p), np.asarray(e))
               for p, e in zip(jax.tree.leaves(plain.params),
                               jax.tree.leaves(plain.ema_params)))
    for a, b in zip(jax.tree.leaves(plain.params), jax.tree.leaves(accumulated.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    for a, b in zip(jax.tree.leaves(plain.ema_params),
                    jax.tree.leaves(accumulated.ema_params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_grad_accum_steps_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        make_deterministic_trainer(tmp_path, "bad-accum", 0)


def grain_image_loader(num_records=256):
    """A checkpointable source that yields distinguishable image batches."""
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.float32)}

    return pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, num_records, 1),
        sampler=pygrain.IndexSampler(num_records=num_records, shuffle=False, seed=0,
                                     num_epochs=4, shard_options=pygrain.NoSharding()),
        operations=[ToImage(), pygrain.Batch(BATCH, drop_remainder=True)],
        worker_count=0,
    )


def test_resume_continues_mid_epoch(tmp_path):
    """A resumed run must not replay the batches it already trained on."""
    steps = 3
    data = {"train": grain_image_loader, "train_len": BATCH * 64,
            "local_batch_size": BATCH, "global_batch_size": BATCH}

    trainer = make_trainer(tmp_path, "resume", distributed_training=True)
    trainer.fit(data, training_steps_per_epoch=steps, epochs=1, val_steps_per_epoch=0)
    assert trainer.dataset_state is not None, "iterator position was never captured"
    trainer.wait_for_checkpoints()

    resumed = make_trainer(tmp_path, "resume", distributed_training=True,
                           load_from_checkpoint=trainer.checkpoint_path())
    assert resumed.dataset_state == trainer.dataset_state

    # The next batch after resuming is the one the first run would have seen next
    reference = DevicePrefetchIterator(
        iter(grain_image_loader()), trainer.batch_sharding,
        source_state=trainer.dataset_state)
    resumed_iter = DevicePrefetchIterator(
        iter(grain_image_loader()), resumed.batch_sharding,
        source_state=resumed.dataset_state)
    np.testing.assert_array_equal(np.asarray(next(resumed_iter)["image"]),
                                  np.asarray(next(reference)["image"]))

    # and it is past the batches already consumed
    fresh = DevicePrefetchIterator(iter(grain_image_loader()), resumed.batch_sharding)
    first = np.asarray(next(fresh)["image"])
    assert not np.array_equal(np.asarray(next(resumed_iter)["image"]), first)
def test_fit_resumes_only_the_unfinished_part_of_an_epoch(tmp_path):
    data = {"train": grain_image_loader, "train_len": BATCH * 64,
            "local_batch_size": BATCH, "global_batch_size": BATCH}
    trainer = make_trainer(tmp_path, "resume-partial", distributed_training=True)
    source = DevicePrefetchIterator(data["train"](), trainer.batch_sharding)
    train_step = trainer._define_train_step(batch_size=BATCH)
    _, current_step, state, rng_state = trainer.train_loop(
        trainer.state, train_step, source, 2, 0, trainer.rngstate)
    trainer.latest_step = current_step
    trainer.state, trainer.rngstate = state, rng_state
    trainer.save(epoch=0, step=current_step)
    trainer.wait_for_checkpoints()

    resumed = make_trainer(
        tmp_path, "resume-partial", distributed_training=True,
        load_from_checkpoint=trainer.checkpoint_path())
    state = resumed.fit(
        data, training_steps_per_epoch=6, epochs=1, val_steps_per_epoch=0)

    assert int(state.step) == 6
    assert resumed.latest_step == 6


def test_load_tolerates_checkpoints_without_iterator_state(tmp_path):
    """Checkpoints written before iterator tracking must still restore."""
    trainer = make_trainer(tmp_path, "legacy", distributed_training=True)
    assert trainer.dataset_state is None
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "legacy", distributed_training=True,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 0
    assert restored.dataset_state is None


def test_checkpoint_restores_onto_a_different_mesh(tmp_path):
    """A run saved with FSDP must be resumable on a replicated mesh."""
    trainer = make_trainer(tmp_path, "mesh", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "mesh", distributed_training=True, fsdp_size=1,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1
    for leaf in jax.tree.leaves(restored.state.params):
        assert leaf.sharding.spec == P()
    for before, after in zip(jax.tree.leaves(trainer.state.params),
                             jax.tree.leaves(restored.state.params)):
        np.testing.assert_allclose(np.asarray(before), np.asarray(after))


# --------------------------------------------------------------------------
# The whole run, on a real mesh
# --------------------------------------------------------------------------

def peak_to_peak_metric(seen):
    """A real EvaluationMetric over the sampler's artifacts.

    Deliberately not CLIP or FID: those download pretrained weights. What is
    under test is the metric plumbing - artifacts in, score out, best tracked,
    logged - which is identical whatever the score means.
    """
    def peak_to_peak(generated, batch):
        seen.append((np.asarray(generated).shape,
                     None if batch is None else np.asarray(batch["image"]).shape))
        return jnp.max(generated) - jnp.min(generated)

    return EvaluationMetric(function=peak_to_peak, name="sample_range")


@pytest.mark.parametrize("fsdp_size", [1, 4])
def test_fit_end_to_end_with_validation_and_metrics(tmp_path, fsdp_size):
    """A full run on the simulated 8-device mesh: training, the validation
    loop with its sampler, an evaluation metric, and the final checkpoint.

    Parametrized over pure data parallelism and a 2x4 data/fsdp mesh, because
    validation samples from the *sharded* EMA parameters - a layout the
    training step alone never exercises.
    """
    seen = []
    trainer = make_trainer(tmp_path, f"e2e-fsdp{fsdp_size}", distributed_training=True,
                           fsdp_size=fsdp_size, fsdp_min_param_size=TINY,
                           eval_metrics=[peak_to_peak_metric(seen)])
    assert trainer.mesh.shape["data"] == jax.device_count() // fsdp_size
    assert trainer.mesh.shape["fsdp"] == fsdp_size
    sharded = [x for x in jax.tree.leaves(trainer.state.ema_params)
               if 'fsdp' in str(x.sharding.spec)]
    assert bool(sharded) == (fsdp_size > 1)
    # 200 sampler steps per validation batch is the production default and pure
    # overhead here; the loop is what is under test, not the sample quality.
    trainer.objective.diffusion_steps = 4

    data = {"train": batches, "val": batches, "train_len": BATCH * 8,
            "local_batch_size": BATCH, "global_batch_size": BATCH}
    state = trainer.fit(data, training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=1)

    assert int(state.step) == 2
    # the sanity validation before training, then one after the epoch
    assert seen == [((4, RES, RES, 3), (BATCH, RES, RES, 3))] * 2

    score = trainer.best_val_metrics["val/sample_range"]
    assert np.isfinite(score) and score > 0

    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == 2
