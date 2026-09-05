"""Sharding, FSDP and data-pipeline tests on a simulated 8-device CPU mesh.

The parity tests are the safety net for the partitioned step: a run over eight
devices, and over a sharded parameter tree, has to produce the numbers a plain
single-device optax loop produces, otherwise the collectives GSPMD derived are
not the ones we meant.
"""

import json

import jax
from dew.data import Loading
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from dew.artifacts import Representations
from dew.inputs import unit_range
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.sharding import DECLARED, logical_axes
from dew.objectives.base import Aux, EMASpec, Objective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import (
    DEFAULT_RULES, DevicePrefetchIterator, parameter_spec, shard_batch,
)
from dew.training.optim import OPTIMIZER_MAP

RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# lower it or "FSDP on" would silently mean "everything replicated".
TINY = 256


class DeterministicObjective(Objective):
    """Squared error against the input, straight through the real DiT.

    No noise level, no dropout, no unconditional mask: the loss depends only
    on the parameters and the batch. That is what makes two accumulation
    regimes comparable, and two topologies: every micro-gradient in a window
    is identical, which is exactly the "one big batch" an accumulated update
    stands in for, so a k-accumulated run and a plain run must trace the same
    parameters, and a partitioned run the same losses as a single-device one.
    """

    artifact = Representations

    def __init__(self, decay=optax.constant_schedule(0.999), emb_features=32):
        self.model = SimpleDiT(patch_size=4, emb_features=emb_features, num_layers=1,
                               num_heads=2, mlp_ratio=1)
        self.ema = EMASpec(decay=decay)

    def init(self, key):
        return self.model.init(key, jnp.ones((1, RES, RES, 3)), jnp.zeros((1,)))

    def loss(self, params, batch, step):
        data = unit_range(batch["image"])
        preds = self.model.apply(params, data, jnp.zeros((data.shape[0],), jnp.float32))
        return jnp.mean((preds - data) ** 2), Aux({})

    def evaluate(self, params, batch, step):
        """The EMA copy's reconstruction, pooled per example: reads the
        sharded EMA parameters, which the training step alone never does."""
        data = unit_range(batch["image"])
        preds = self.model.apply(step.ema, data, jnp.zeros((data.shape[0],), jnp.float32))
        return Representations(features=jnp.mean(preds, axis=(1, 2)),
                               labels=jnp.zeros((data.shape[0],), jnp.int32))


class Data:
    def __init__(self, train, val=None, batch=BATCH, records=None):
        self._train, self.val = train, val
        self.batch, self.records = batch, records

    def train(self):
        return self._train()

    @property
    def steps_per_epoch(self):
        return None if self.records is None else self.records // self.batch


def images():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(BATCH, RES, RES, 3)).astype(np.uint8)


def batches():
    batch = {"image": images()}
    while True:
        yield batch


class RecordingTracker:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))

    def losses(self):
        return [s["train/loss"] for _, s in self.scalars if "train/loss" in s]


def make_trainer(tmp_path=None, fsdp=1, optimizer=None, objective=None, tracker=None,
                 **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path), keep=4)
    return Trainer(
        DeterministicObjective() if objective is None else objective,
        optax.adam(1e-3) if optimizer is None else optimizer,
        key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp),
        layout=Layout(min_shard=TINY),
        checkpoints=checkpoints,
        tracker=tracker,
        **kwargs,
    )


def run_losses(steps, **kwargs):
    """Per-step losses of a fit, as the tracker receives them."""
    tracker = RecordingTracker()
    make_trainer(tracker=tracker, **kwargs).fit(Data(batches), steps=steps, log_every=1)
    return tracker.losses()


def reference_losses(trainer, steps):
    """The same run as a plain optax loop on one device: the yardstick a
    partitioned step has to reproduce."""
    objective, optimizer = trainer.objective, trainer.optimizer
    state = trainer.initial_state()
    device = jax.devices()[0]
    params = jax.device_put(state.params, device)
    opt_state = jax.device_put(state.opt_state, device)

    @jax.jit
    def step(params, opt_state, batch, key):
        info = type(state)(step=jnp.zeros((), jnp.int32), params=params, opt_state=opt_state,
                           ema=None, key=key)
        del info

        def loss_fn(trainable):
            from dew.objectives.base import Step
            return objective.loss({**params, "params": trainable}, batch,
                                  Step(step=jnp.zeros((), jnp.int32), key=key, ema=None))

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params["params"])
        updates, opt_state = optimizer.update(grads, opt_state, params["params"])
        return {**params, "params": optax.apply_updates(params["params"], updates)}, opt_state, loss

    losses = []
    source = batches()
    for index in range(steps):
        batch = jax.device_put(next(source), device)
        params, opt_state, loss = step(params, opt_state, batch,
                                       jax.random.fold_in(state.key, index))
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
    shardings = Layout(rules=rules, min_shard=1).shardings(
        build_mesh(MeshSpec(fsdp=2)), variables)
    return jax.tree.map(lambda sharding: sharding.spec, shardings)["params"]


def dit_variables(**overrides):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=2,
                      **overrides)
    return jax.eval_shape(model.init, jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))


def test_causal_transformer_axes_land_on_the_dimensions_they_name():
    """One rule at a time: the dimension that moves is the one the module
    declares, which is what a declared axis has to mean."""
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=1, num_heads=2,
        num_kv_heads=1, mlp_features=64, max_seq_len=8, tie_embeddings=False)
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
    variables = dit_variables()

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


def test_a_declared_axis_that_cannot_name_a_parameter_is_an_error():
    """A module that keeps its name while its parameter gains a dimension has
    to stop the derivation, not shard whichever dimensions the short name
    happens to reach."""
    variables = {"params": {"q_proj": {
        "kernel": jax.ShapeDtypeStruct((8, 8, 8), jnp.float32)}}}
    with pytest.raises(ValueError, match="q_proj"):
        Layout(min_shard=1).shardings(build_mesh(MeshSpec(fsdp=2)), variables)


def test_a_declaration_that_names_one_axis_twice_is_refused_where_it_is_written():
    """flax assigns a mesh axis to a logical name at most once per array, so
    a kernel declared ("embed", "embed") cannot be placed: the refusal has to
    come at the decorator, not from Layout.shardings on the first sharded
    mesh a run builds (the eh_proj declaration shipped that way)."""
    with pytest.raises(ValueError, match="twice"):
        @logical_axes({("fused_proj",): ("embed", "embed")})
        class Fused(nn.Module):
            pass
    assert ("fused_proj",) not in DECLARED


def test_rule_override_changes_only_declared_axes():
    specs = declared_specs(dit_variables(), {"mlp": "fsdp"})

    assert specs["dit_block_0"]["mlp"]["layers_0"]["kernel"] == P(None, "fsdp")
    assert specs["dit_block_0"]["mlp"]["layers_2"]["kernel"] == P("fsdp")
    assert specs["dit_block_0"]["attention"]["to_q"]["kernel"] == P()
    assert specs["embed"]["patch_embed"]["Conv_0"]["kernel"] == P()


def test_default_rules_keep_the_shape_heuristic_for_the_dit():
    """The declared table reproduces the largest-divisible-axis choice on the
    DiT's shapes, so declaring the model moved none of its leaves."""
    variables = dit_variables()
    shardings = Layout(rules=DEFAULT_RULES, min_shard=1).shardings(
        build_mesh(MeshSpec(fsdp=2)), variables)
    expected = jax.tree.map(
        lambda leaf: parameter_spec(leaf.shape, fsdp_size=2, min_shard_size=1), variables)
    assert jax.tree.map(lambda sharding: sharding.spec, shardings) == expected


def test_rules_may_name_a_mesh_axis_this_mesh_does_not_have():
    """A table can carry the future tensor axis; today's mesh drops it."""
    shardings = Layout(rules={"mlp": ["tensor", "fsdp"], "heads": "tensor"},
                       min_shard=1).shardings(build_mesh(MeshSpec(fsdp=2)),
                                              dit_variables())["params"]

    assert (shardings["dit_block_0"]["mlp"]["layers_0"]["kernel"].spec
            == P(None, "fsdp"))
    assert shardings["dit_block_0"]["attention"]["to_q"]["kernel"].spec == P()


class IndivisibleModel(nn.Module):
    """A parameter no mesh axis of size two can split."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(15, use_bias=False, name="indivisible")(x)


class Indivisible(Objective):
    ema = None

    def init(self, key):
        return IndivisibleModel().init(key, jnp.ones((1, 15)))

    def loss(self, params, batch, step):
        return jnp.sum(IndivisibleModel().apply(params, batch["image"][:, 0, 0, :]) ** 2), Aux({})


def build_indivisible(tolerance):
    return Trainer(Indivisible(), optax.adam(1e-3), key=jax.random.key(0),
                   mesh=MeshSpec(fsdp=2),
                   layout=Layout(min_shard=1, tolerance=tolerance)).fit(Data(batches), steps=0)


def test_sharding_tolerance_names_the_largest_replicated_parameter():
    with pytest.raises(ValueError) as error:
        build_indivisible(tolerance=0.02)

    message = str(error.value)
    assert "100.00%" in message and "2.00%" in message
    assert "['params']['indivisible']['kernel']" in message
    assert "(15, 15)" in message


def test_sharding_tolerance_can_allow_intentional_replication():
    state = build_indivisible(tolerance=1.0)
    assert state.params["params"]["indivisible"]["kernel"].sharding.spec == P()


def test_the_layout_default_tolerance_is_two_percent():
    """A layout that names no tolerance carries the library's 2%, without
    the config repeating the number."""
    assert Layout().tolerance == 0.02
    with pytest.raises(ValueError, match="2.00%"):
        Trainer(Indivisible(), optax.adam(1e-3), key=jax.random.key(0),
                mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1)).fit(Data(batches), steps=0)


def test_sharding_tolerance_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Layout(tolerance=1.5)


@pytest.mark.parametrize("fsdp_size", [2, 4, 8])
def test_odd_vocabulary_shards_the_embedding_on_its_other_axis(fsdp_size):
    """GPT-2's 50257 rows divide by nothing, so the rule that wins the
    embedding cannot be taken: the width has to carry the shard, or a real run
    stops on the tolerance check with 98% of the model replicated. Nothing
    about that changes as the fsdp axis widens, which is where a fallback that
    only ever divided by two would show up."""
    model = CausalTransformer(
        vocab_size=50257, emb_features=64, num_layers=1, num_heads=2,
        num_kv_heads=1, mlp_features=128, max_seq_len=8)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size))
    layout = Layout(min_shard=TINY)
    shardings = layout.shardings(mesh, variables)

    assert shardings["params"]["embed_tokens"]["embedding"].spec == P(None, "fsdp")
    layout.check(variables["params"], shardings["params"], mesh)


def test_build_mesh_rejects_bad_fsdp_size():
    with pytest.raises(ValueError):
        build_mesh(MeshSpec(fsdp=3))


def test_build_mesh_axes():
    mesh = build_mesh(MeshSpec(fsdp=2))
    assert mesh.shape['data'] == jax.device_count() // 2
    assert mesh.shape['fsdp'] == 2


# --------------------------------------------------------------------------
# Batch placement and prefetch
# --------------------------------------------------------------------------

def test_shard_batch_splits_across_all_devices():
    mesh = build_mesh(MeshSpec(fsdp=2))
    batch = {"image": np.zeros((jax.device_count(), 4), np.float32)}
    sharded = shard_batch(mesh, batch)["image"]
    assert len(sharded.addressable_shards) == jax.device_count()
    assert sharded.addressable_shards[0].data.shape == (1, 4)


def test_prefetch_iterator_preserves_order_and_terminates():
    mesh = build_mesh()
    source = ({"x": np.full((jax.device_count(), 2), i, np.float32)} for i in range(5))
    it = DevicePrefetchIterator(source, mesh, depth=2)
    seen = [float(np.asarray(b["x"])[0, 0]) for b in it]
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]
    with pytest.raises(StopIteration):
        next(it)


def test_prefetch_iterator_surfaces_source_errors():
    mesh = build_mesh()

    def broken():
        yield {"x": np.zeros((jax.device_count(), 2), np.float32)}
        raise ValueError("source exploded")

    it = DevicePrefetchIterator(broken(), mesh, depth=2)
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
    it = DevicePrefetchIterator(iter(loader), mesh, depth=2)
    next(it)
    next(it)
    state = it.source_state
    expected = np.asarray(next(it))

    resumed = DevicePrefetchIterator(iter(loader), mesh, depth=2,
                                     source_state=state)
    assert np.array_equal(np.asarray(next(resumed)), expected)


def test_prefetch_iterator_resumes_a_packed_dataset_iterator(tmp_path):
    """The packed loader is grain's Dataset API, whose iterator reports its
    position as a dict where the DataLoader reports JSON bytes. A checkpoint
    holds bytes, so the position has to arrive as bytes and go back as a
    dict."""
    from dew.data import PackedTokens

    documents = np.concatenate([np.arange(1, 9), [0]] * 40).astype(np.uint16)
    (tmp_path / "train.bin").write_bytes(documents.tobytes())
    (tmp_path / "val.bin").write_bytes(documents.tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint16", "eos_id": 0}))
    data = PackedTokens(path=str(tmp_path), seq_len=8, seed=0, loading=Loading(workers=0),
                        packing_bins=2).load(batch=jax.device_count())

    mesh = build_mesh()
    it = DevicePrefetchIterator(data.train(), mesh, depth=2)
    next(it)
    state = it.source_state
    expected = np.asarray(next(it)["text"])
    assert isinstance(state, bytes), "a checkpoint carries the position as bytes"

    resumed = DevicePrefetchIterator(data.train(), mesh, depth=2,
                                     source_state=state)
    assert np.array_equal(np.asarray(next(resumed)["text"]), expected)


# --------------------------------------------------------------------------
# Numerical parity
# --------------------------------------------------------------------------

def test_the_partitioned_step_reproduces_a_single_device_loop():
    """Partitioning over eight devices must not change the maths."""
    steps = 20
    trainer = make_trainer(optimizer=optax.sgd(0.5))
    reference = reference_losses(trainer, steps)
    assert jax.device_count() > 1
    np.testing.assert_allclose(reference, run_losses(steps, optimizer=optax.sgd(0.5)),
                               rtol=2e-4, atol=2e-5)


def test_fsdp_losses_match_replicated():
    """Sharding the parameters must not change the loss trajectory."""
    steps = 20
    replicated = run_losses(steps, fsdp=1)
    fsdp = run_losses(steps, fsdp=2)
    assert np.isfinite(replicated).all()
    np.testing.assert_allclose(replicated, fsdp, rtol=2e-4, atol=2e-5)


# --------------------------------------------------------------------------
# FSDP actually shards
# --------------------------------------------------------------------------

def test_fsdp_shards_parameters_and_optimizer_state():
    state = make_trainer(fsdp=2).fit(Data(batches), steps=1, log_every=1)
    leaves = jax.tree.leaves(state.params)
    sharded = [x for x in leaves if 'fsdp' in str(x.sharding.spec)]
    assert sharded, "no parameter was sharded over the fsdp axis"

    for param in sharded:
        local = param.addressable_shards[0].data
        assert local.size == param.size // 2, "shard is not half the global param"
        # Exactly the dimension the spec names is halved. Which dimension that
        # is belongs to the declarations, not to this test.
        split = [axis for axis, (whole, part) in enumerate(
            zip(param.shape, local.shape)) if whole != part]
        assert len(split) == 1, f"{param.shape} -> {local.shape}"
        assert param.shape[split[0]] // 2 == local.shape[split[0]]
        assert param.sharding.spec[split[0]] == 'fsdp'

    # Adam moments and the EMA copy must follow the params they track, without
    # the optimizer or the model ever describing a layout.
    param_specs = [x.sharding.spec for x in jax.tree.leaves(state.params)]
    assert param_specs == [x.sharding.spec for x in jax.tree.leaves(state.opt_state[0].mu)]
    assert param_specs == [x.sharding.spec for x in jax.tree.leaves(state.ema)]


def test_muon_masked_optimizer_state_shards_with_its_parameters():
    """A masked optax transform leaves a MaskedNode where its group does not
    apply, and muon's adam branch is always masked, so the derivation has to
    carry a leaf that is not an array at all."""
    state = make_trainer(fsdp=2, optimizer=optax.contrib.muon(1e-3)).fit(
        Data(batches), steps=0)

    def is_placeholder(leaf):
        return isinstance(leaf, optax.MaskedNode)

    placeholders = [leaf for leaf in jax.tree.leaves(
        state.opt_state, is_leaf=is_placeholder) if is_placeholder(leaf)]
    assert placeholders, "muon left no masked placeholder for the derivation to carry"

    kernel = state.params["params"]["dit_block_0"]["mlp"]["layers_0"]["kernel"]
    assert kernel.sharding.spec == P(None, "fsdp")
    moment_specs = {leaf.sharding.spec for leaf in jax.tree.leaves(state.opt_state)}
    assert kernel.sharding.spec in moment_specs


def test_replicated_run_shards_nothing():
    state = make_trainer(fsdp=1).fit(Data(batches), steps=0)
    for leaf in jax.tree.leaves(state.params):
        assert leaf.sharding.spec == P()


# --------------------------------------------------------------------------
# Checkpointing under sharding
# --------------------------------------------------------------------------

def test_sharded_checkpoint_roundtrips(tmp_path):
    trained = make_trainer(tmp_path, fsdp=2).fit(Data(batches), steps=1, log_every=1)
    restored = make_trainer(tmp_path, fsdp=2).fit(Data(batches), steps=1)

    assert int(restored.step) == 1
    for before, after in zip(jax.tree.leaves(trained.params),
                             jax.tree.leaves(restored.params), strict=True):
        assert before.sharding.spec == after.sharding.spec
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_checkpoint_restores_onto_a_different_mesh(tmp_path):
    """A run saved with FSDP must be resumable on a replicated mesh."""
    trained = make_trainer(tmp_path, fsdp=2).fit(Data(batches), steps=1, log_every=1)
    restored = make_trainer(tmp_path, fsdp=1).fit(Data(batches), steps=1)

    assert int(restored.step) == 1
    for leaf in jax.tree.leaves(restored.params):
        assert leaf.sharding.spec == P()
    for before, after in zip(jax.tree.leaves(trained.params),
                             jax.tree.leaves(restored.params), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


@pytest.mark.parametrize("written,restored", [(1, 8), (8, 1)])
def test_a_checkpoint_restores_across_the_whole_fsdp_range(tmp_path, written, restored):
    """A run resumes on the hardware it gets, not the hardware it left.

    Both directions of the widest change the simulated mesh allows: every
    parameter replicated, and every parameter split eight ways. A checkpoint
    is bytes rather than arithmetic, so the values have to come back equal.
    """
    trained = make_trainer(tmp_path, fsdp=written).fit(Data(batches), steps=1, log_every=1)
    before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(trained.params)]

    reopened = make_trainer(tmp_path, fsdp=restored).fit(Data(batches), steps=1)
    assert int(reopened.step) == 1
    leaves = jax.tree.leaves(reopened.params)
    for saved, leaf in zip(before, leaves, strict=True):
        np.testing.assert_array_equal(saved, np.asarray(leaf))

    sharded = [leaf for leaf in leaves if 'fsdp' in str(leaf.sharding.spec)]
    assert bool(sharded) == (restored > 1), "the restored layout is not this run's"
    for leaf in sharded:
        assert leaf.addressable_shards[0].data.size == leaf.size // restored


def test_a_checkpoint_without_a_position_resumes_from_the_top_of_the_stream(tmp_path):
    """A stream without get_state writes no position; a resume from that
    checkpoint restores the state and reads the stream from its start."""
    make_trainer(tmp_path).fit(Data(batches), steps=1, log_every=1)
    _, position = Checkpoints(str(tmp_path)).restore()
    assert position is None
    resumed = make_trainer(tmp_path).fit(Data(batches), steps=2, log_every=1)
    assert int(resumed.step) == 2


# --------------------------------------------------------------------------
# Gradient accumulation and the EMA clock
# --------------------------------------------------------------------------

def make_accumulating(tmp_path=None, accumulation=1):
    return make_trainer(
        tmp_path, fsdp=2, optimizer=optax.sgd(0.5), accumulation=accumulation,
        # A ramp rather than a constant, so indexing the schedule by micro-step
        # instead of by update is visible in the result.
        objective=DeterministicObjective(optax.linear_schedule(0.9, 1.0, transition_steps=8)))


def micro_steps(trainer, count, state=None):
    """`count` micro-steps of the trainer's own step body, on the host."""
    step = trainer._default_step()
    state = trainer.initial_state() if state is None else state
    source = batches()
    for _ in range(count):
        state, _, _, _ = step(state, None, next(source))
    return state


def snapshot(tree):
    return [np.asarray(x).copy() for x in jax.tree.leaves(tree)]


def moved(before, after):
    return any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_gradient_accumulation_updates_only_on_the_boundary():
    """MultiSteps must hold the params still until k micro-batches have run."""
    accum = 3
    trainer = make_accumulating(accumulation=accum)
    state = trainer.initial_state()
    reference = snapshot(state.params)
    for micro in range(1, accum * 2 + 1):
        state = micro_steps(trainer, 1, state)
        at_boundary = micro % accum == 0
        assert moved(reference, snapshot(state.params)) == at_boundary, f"micro-step {micro}"
        if at_boundary:
            reference = snapshot(state.params)


def test_ema_moves_only_on_accumulation_boundaries():
    """The EMA has to run on the optimizer's clock, not the micro-batch's.

    Between boundaries MultiSteps holds the params still, so an average taken
    every micro-step would blend in parameters that never changed and pull the
    EMA k times as far per update.
    """
    accum = 4
    trainer = make_accumulating(accumulation=accum)
    state = trainer.initial_state()
    reference = snapshot(state.ema)
    for micro in range(1, accum * 2 + 1):
        state = micro_steps(trainer, 1, state)
        at_boundary = micro % accum == 0
        assert moved(reference, snapshot(state.ema)) == at_boundary, f"micro-step {micro}"
        if at_boundary:
            reference = snapshot(state.ema)


def test_accumulated_ema_matches_a_plain_run_at_equal_update_counts():
    """k micro-batches of the same data must land where one big batch would,
    EMA included: same params, same decay index, same average."""
    accum, updates = 4, 3
    plain = micro_steps(make_accumulating(accumulation=1), updates)
    accumulated = micro_steps(make_accumulating(accumulation=accum), updates * accum)

    # the comparison is only meaningful if the EMA left its starting point
    assert moved(snapshot(plain.params), snapshot(plain.ema))
    for a, b in zip(jax.tree.leaves(plain.params), jax.tree.leaves(accumulated.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    for a, b in zip(jax.tree.leaves(plain.ema), jax.tree.leaves(accumulated.ema)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_a_resume_inside_an_accumulation_window_keeps_the_ema_clock(tmp_path):
    """An accumulated run killed mid-window has to average like one run whole.

    The checkpoint is taken one micro-batch into a window, so it carries the
    half-filled accumulator, the MultiSteps counters and a step count that is
    not a multiple of the window. The EMA decay is a ramp indexed by completed
    updates, so a resume that took the micro-batch counter for the update
    counter, or started either counter over, moves the average a different
    distance from where the uninterrupted run put it.
    """
    accum, updates = 2, 3
    total = accum * updates * 2
    cut = accum * updates + 1

    whole = make_accumulating(tmp_path / "whole", accum).fit(Data(batches), steps=total)
    interrupted = make_accumulating(tmp_path / "split", accum).fit(Data(batches), steps=cut)
    assert int(np.asarray(interrupted.opt_state.mini_step)) == 1, "the cut is not mid-window"
    resumed = make_accumulating(tmp_path / "split", accum).fit(Data(batches), steps=total)

    assert int(resumed.step) == int(whole.step) == total
    assert moved(snapshot(whole.params), snapshot(whole.ema))
    for expected, actual in zip(jax.tree.leaves(whole.params),
                                jax.tree.leaves(resumed.params), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual),
                                   rtol=1e-6, atol=1e-7)
    for expected, actual in zip(jax.tree.leaves(whole.ema),
                                jax.tree.leaves(resumed.ema), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual),
                                   rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------
# Resuming a grain loader mid-epoch
# --------------------------------------------------------------------------

def grain_image_loader(num_records=256):
    """A checkpointable source that yields distinguishable image batches."""
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.uint8)}

    return iter(pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, num_records, 1),
        sampler=pygrain.IndexSampler(num_records=num_records, shuffle=False, seed=0,
                                     num_epochs=4, shard_options=pygrain.NoSharding()),
        operations=[ToImage(), pygrain.Batch(BATCH, drop_remainder=True)],
        worker_count=0,
    ))


def test_a_resumed_run_reads_the_batch_after_its_checkpoint(tmp_path):
    """A resumed run must not replay the batches it already trained on."""
    data = Data(grain_image_loader, records=BATCH * 64)
    make_trainer(tmp_path).fit(data, steps=3, log_every=1)
    _, position = Checkpoints(str(tmp_path)).restore()
    assert position is not None, "iterator position was never captured"

    mesh = build_mesh()
    resumed = DevicePrefetchIterator(grain_image_loader(), mesh,
                                     source_state=position)
    fresh = DevicePrefetchIterator(grain_image_loader(), mesh)
    for _ in range(3):
        next(fresh)
    np.testing.assert_array_equal(np.asarray(next(resumed)["image"]),
                                  np.asarray(next(fresh)["image"]))


def test_fit_resumes_the_unfinished_part_of_a_run(tmp_path):
    data = Data(grain_image_loader, records=BATCH * 64)
    make_trainer(tmp_path).fit(data, steps=2, log_every=1)
    state = make_trainer(tmp_path).fit(data, steps=6, log_every=1)
    assert int(state.step) == 6
    assert Checkpoints(str(tmp_path)).latest == 6


# --------------------------------------------------------------------------
# The whole run, on a real mesh
# --------------------------------------------------------------------------

class PeakToPeak:
    """A real metric over the objective's artifacts.

    Deliberately not CLIP or FID: those download pretrained weights. What is
    under test is the metric plumbing (artifacts in, score out, logged),
    which is identical whatever the score means.
    """
    name = "sample_range"
    reads = Representations

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, artifact, batch):
        self.seen.append((np.asarray(artifact.features).shape, np.asarray(batch["image"]).shape))
        return float(jnp.max(artifact.features) - jnp.min(artifact.features))

    def reduce(self, values):
        return float(np.mean(values))


def val_stream():
    source = batches()
    for _ in range(2):
        yield next(source)


@pytest.mark.parametrize("fsdp_size", [1, 4])
def test_fit_end_to_end_with_validation_and_metrics(tmp_path, fsdp_size):
    """A full run on the simulated 8-device mesh: training, the validation
    pass, a metric, the tracker, and the final checkpoint.

    Parametrized over pure data parallelism and a 2x4 data/fsdp mesh, because
    validation reads the *sharded* EMA parameters, a layout the training step
    alone never exercises.
    """
    seen = []
    tracker = RecordingTracker()
    trainer = make_trainer(tmp_path, fsdp=fsdp_size, tracker=tracker)
    state = trainer.fit(Data(batches, val=val_stream), steps=2, log_every=1, eval_every=1,
                        metrics=(PeakToPeak(seen),))

    assert int(state.step) == 2
    sharded = [x for x in jax.tree.leaves(state.ema) if 'fsdp' in str(x.sharding.spec)]
    assert bool(sharded) == (fsdp_size > 1)
    # A pass after step 1 and one at the end, two batches each.
    assert seen == [((BATCH, 3), (BATCH, RES, RES, 3))] * 4
    scores = [s["val/sample_range"] for _, s in tracker.scalars if "val/sample_range" in s]
    assert len(scores) == 2 and all(np.isfinite(score) and score > 0 for score in scores)
    assert [step for step, _ in tracker.artifacts] == [1, 2]
    assert Checkpoints(str(tmp_path)).latest == 2


# --------------------------------------------------------------------------
# The optimizer parameter groups under sharding
# --------------------------------------------------------------------------

def muon_state_specs(variables, rules):
    """Params and the moments of both optimizer groups, on a two-way mesh."""
    solver = OPTIMIZER_MAP["muon"](1e-3)
    opt_state = jax.eval_shape(solver.init, variables)
    shardings = Layout(rules=rules, min_shard=1).shardings(
        build_mesh(MeshSpec(fsdp=2)), (variables, opt_state))
    return jax.tree.map(lambda sharding: sharding.spec, shardings)


def muon_moment_specs(opt_state_specs, param_specs):
    """Per parameter, its own spec and the specs its moments were given.

    A parameter is identified the way the derivation identifies it, by the
    trailing run of its path, because each group nests a copy of the
    parameter tree inside its own state. A group holds no leaf at all for a
    parameter it does not step, so the moments found are the membership.
    """
    found = {tuple(entry.key for entry in path): (spec, [])
             for path, spec in jax.tree_util.tree_flatten_with_path(param_specs)[0]}
    for path, spec in jax.tree_util.tree_flatten_with_path(opt_state_specs)[0]:
        names = []
        for entry in reversed(path):
            if not isinstance(entry, jax.tree_util.DictKey):
                break
            names.append(entry.key)
        candidate = tuple(reversed(names))
        if candidate in found:
            found[candidate][1].append(spec)
    return found


def test_muon_group_moments_take_the_spec_their_parameter_declared():
    """The split nests the parameter tree in two masked states, one per group.
    Under a rule table that names one axis, a moment that fell back to the
    shape heuristic would shard a parameter the table leaves whole, so the
    assertion is that every moment of every parameter carries that
    parameter's own spec, in both groups. The moment count says which group
    stepped it: AdamW keeps two, Muon one.
    """
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=1, num_heads=2,
        num_kv_heads=1, mlp_features=64, max_seq_len=8, tie_embeddings=False)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    param_specs, opt_state_specs = muon_state_specs(variables, {"mlp": "fsdp"})

    params = param_specs["params"]
    assert params["embed_tokens"]["embedding"] == P()
    assert params["layers_0"]["mlp"]["down_proj"]["kernel"] == P("fsdp")
    assert params["layers_0"]["mlp"]["up_proj"]["kernel"] == P(None, "fsdp")

    moments = muon_moment_specs(opt_state_specs, param_specs)
    for name, (declared, specs) in moments.items():
        assert specs, f"{name} has no optimizer moment in either group"
        assert set(specs) == {declared}, name

    embedding = moments[("params", "embed_tokens", "embedding")]
    kernel = moments[("params", "layers_0", "mlp", "down_proj", "kernel")]
    assert len(embedding[1]) == 2 and embedding[0] == P()
    assert len(kernel[1]) == 1 and kernel[0] == P("fsdp")


# --------------------------------------------------------------------------
# Sharding invariants at every fsdp width
# --------------------------------------------------------------------------

def assert_specs_can_split(variables, shardings, fsdp_size):
    """Every dimension a spec names has to be splittable that many ways.

    A spec that names a dimension the mesh axis does not divide, or one of
    size 1, is a layout jit rejects or a collective that moves nothing. The
    rules are supposed to drop those names and hand the axis to another
    dimension.
    """
    leaves = jax.tree_util.tree_flatten_with_path(variables)[0]
    for (path, value), sharding in zip(leaves, jax.tree.leaves(shardings), strict=True):
        for dimension, entry in enumerate(sharding.spec):
            if not entry:
                continue
            size = value.shape[dimension]
            where = f"{jax.tree_util.keystr(path)} {value.shape} -> {sharding.spec}"
            assert size % fsdp_size == 0, f"{where} cannot split {fsdp_size} ways"
            assert size > 1, f"{where} shards a dimension of one"


def single_head_variables(num_heads=1):
    """A DiT whose attention kernels carry a heads dimension of one."""
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1,
                      num_heads=num_heads, mlp_ratio=1)
    return jax.eval_shape(model.init, jax.random.key(0),
                          jnp.ones((1, RES, RES, 3)), jnp.ones((1,)))


@pytest.mark.slow
@pytest.mark.parametrize("fsdp_size", [2, 4, 8])
def test_a_single_head_model_shards_a_dimension_it_can_split(fsdp_size):
    """One head makes the heads dimension 1, which shards nothing.

    The DiT's attention kernels are (embed, heads, head_dim) and the
    declaration names all three, so the rules have to leave the single head
    alone and shard a dimension that exists. Every model in the suite has two
    heads, where a spec naming the heads axis happens to work.
    """
    variables = single_head_variables()
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size))
    layout = Layout(min_shard=TINY)
    shardings = layout.shardings(mesh, variables)
    attention = shardings["params"]["dit_block_0"]["attention"]

    assert attention["to_q"]["kernel"].spec == P("fsdp")
    assert_specs_can_split(variables["params"], shardings["params"], fsdp_size)
    layout.check(variables["params"], shardings["params"], mesh)


class OneLongVector(nn.Module):
    """A model whose only parameter worth sharding has a single dimension."""
    length: int = 4096

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (self.length,))
        return nn.Dense(4)(x) * jnp.sum(scale)


@pytest.mark.slow
@pytest.mark.parametrize("fsdp_size", [2, 4, 8])
def test_a_one_dimensional_parameter_shards_on_its_only_axis(fsdp_size):
    """A vector has one axis to give, and giving it is not optional.

    Every declared parameter is a matrix or a stack of them, so a 1-D
    parameter falls to the shape heuristic. Leaving it replicated because it
    has no second dimension would put the whole model on every device and
    pass every other sharding test here.
    """
    variables = jax.eval_shape(
        OneLongVector().init, jax.random.key(0), jnp.ones((1, 4)))
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size))
    layout = Layout(min_shard=TINY)
    shardings = layout.shardings(mesh, variables)

    assert shardings["params"]["scale"].spec == P("fsdp")
    layout.check(variables["params"], shardings["params"], mesh)


@pytest.mark.slow
@pytest.mark.parametrize("fsdp_size,replicated_fraction", [(2, 0.0), (4, 0.36), (8, 0.90)])
def test_a_width_the_mesh_cannot_divide_stops_the_run_rather_than_replicating_it(
        fsdp_size, replicated_fraction):
    """62 features divide by two and by nothing else the mesh offers.

    The rules drop a name they cannot use, so the wider the fsdp axis the more
    of this model stays whole: nothing at two, a third at four, nine tenths at
    eight. What must not happen is a run that trains anyway with the model
    replicated on every device, which is what the tolerance check is for.
    """
    model = CausalTransformer(
        vocab_size=64, emb_features=62, num_layers=1, num_heads=1, num_kv_heads=1,
        mlp_features=124, max_seq_len=8)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size))
    layout = Layout(min_shard=TINY)
    shardings = layout.shardings(mesh, variables)

    # Whatever the rules could not place, they left alone rather than named.
    assert_specs_can_split(variables["params"], shardings["params"], fsdp_size)

    if not replicated_fraction:
        layout.check(variables["params"], shardings["params"], mesh)
        return
    with pytest.raises(ValueError) as error:
        layout.check(variables["params"], shardings["params"], mesh)
    reported = float(str(error.value).split("%")[0]) / 100
    assert abs(reported - replicated_fraction) < 0.01, str(error.value)
