"""DiLoCo (row 876): replica groups each training an inner copy of the model, every few updates averaged
into the outer model by an outer optimizer. What is asserted is the algorithm's arithmetic against an
independent float64 reference, its limits (one replica, one inner update), where the replicas and their
rows live, and that a resumed run restores each replica exactly."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import BATCH, FEATURES, Counting, Regression, dataset
from etils import epath
from flax import linen as nn
from model_support import largest_difference
from recording import bits
from reference_error import assert_computes_the_oracle

from dew.averaging import power_decay
from dew.checkpoints import frozen_entries
from dew.objectives.base import Aux, EMASpec, Objective, Ratio, Weighted
from dew.optim import power_profiles
from dew.training import Checkpoints, DiLoCo, Layout, MeshSpec, Trainer

RATE, OUTER_RATE, MOMENTUM = 0.05, 0.7, 0.9


class Affine64(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2, dtype=jnp.float64, param_dtype=jnp.float64)(x.astype(jnp.float64))


class Regression64(Objective):
    """Squared error of a float64 affine map against `2 * x[:, :2]`; `noisy`, its inputs perturbed by the
    step's draw, so a step's key shows in its gradient; `scaled`, its error times a batch scalar;
    `weighted`, its error a `Weighted` of one term, statistics a window replays rather than pools;
    `rowwise`, its statistics each row's error and count, which add over a window and its own reduction
    divides; `drawn`, beside them the update's draws of a value an example (`Step.sample_update`), one an
    example of the whole update, which its reduction leaves aside."""

    def __init__(self, *, noisy: bool = False, scaled: bool = False, weighted: bool = False,
                 rowwise: bool = False, drawn: bool = False):
        self.model, self.noisy, self.scaled, self.weighted = Affine64(), noisy, scaled, weighted
        self.rowwise, self.drawn = rowwise or drawn, drawn

    def reduce_loss(self, stats):
        if self.rowwise:
            return jnp.sum(stats[0]) / jnp.sum(stats[1]), jnp.asarray(a=True)
        return super().reduce_loss(stats)

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, variables, batch, step):
        x = batch["x"]
        if self.noisy:
            x = x + 0.1 * jax.random.normal(step.key, x.shape, x.dtype)
        squares = (self.model.apply(variables, x) - batch["y"]) ** 2
        if self.rowwise:
            count = jnp.ones(squares.shape[:1], squares.dtype)
            if not self.drawn:
                return (jnp.mean(squares, axis=-1), count), Aux({})
            update = step.sample_update("weight", jax.random.uniform, (), dtype=squares.dtype,
                                        rows=squares.shape[0])
            return (jnp.mean(squares, axis=-1), count, update), Aux({})
        if self.weighted:
            term = Ratio(jnp.sum(squares), jnp.asarray(squares.size, squares.dtype))
            return Weighted((term,), (1.0,)), Aux({})
        return jnp.mean(squares) * (batch["weight"] if self.scaled else 1.0), Aux({})


def trainer(replicas: int, *, diloco: DiLoCo | None = None, optimizer=None, path=None, objective=None,
            **options) -> Trainer:
    return Trainer(Regression64() if objective is None else objective, optax.sgd(RATE) if optimizer is None
                   else optimizer, key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0),
                   mesh=MeshSpec(diloco=replicas).build(jax.devices()[:replicas]), diloco=diloco,
                   checkpoints=None if path is None else Checkpoints(str(path)), **options)


def params(tree) -> np.ndarray:
    """A Dense's kernel and bias as one float64 vector, in that order."""
    dense = tree["Dense_0"]
    return np.concatenate([np.ravel(np.asarray(dense[name], np.float64)) for name in ("kernel", "bias")])


def reference(initial: dict, steps: int, replicas: int, every: int, inner_momentum: float,
              accumulation: int = 1) -> np.ndarray:
    """DiLoCo in NumPy float64 (Douillard et al., 2023, Algorithm 1): each replica takes `every` steps of
    SGD with momentum on its contiguous share of each batch's rows from the outer parameters, the outer
    gradient is the outer parameters less the replicas' mean, and the outer step is Nesterov SGD; with
    `accumulation`, each inner update the mean of that many steps' gradients, and `every` counts updates."""
    kernel, bias = np.asarray(initial["kernel"], np.float64), np.asarray(initial["bias"], np.float64)
    outer = np.concatenate([kernel.ravel(), bias.ravel()])
    outer_momentum = np.zeros_like(outer)
    inner = [outer.copy() for _ in range(replicas)]
    inner_velocity = [np.zeros_like(outer) for _ in range(replicas)]
    source = Counting()
    pooled = [np.zeros_like(outer) for _ in range(replicas)]
    for step in range(steps):
        batch = next(source)
        x, y = batch["x"].astype(np.float64), batch["y"].astype(np.float64)
        share = BATCH // replicas
        for replica in range(replicas):
            rows = slice(replica * share, (replica + 1) * share)
            w = inner[replica][:FEATURES * 2].reshape(FEATURES, 2)
            b = inner[replica][FEATURES * 2:]
            residual = x[rows] @ w + b - y[rows]
            scale = 2 / residual.size
            pooled[replica] += np.concatenate([(scale * x[rows].T @ residual).ravel(),
                                               scale * residual.sum(0)])
            if (step + 1) % accumulation:
                continue
            gradient, pooled[replica] = pooled[replica] / accumulation, np.zeros_like(outer)
            inner_velocity[replica] = inner_momentum * inner_velocity[replica] + gradient
            inner[replica] = inner[replica] - RATE * inner_velocity[replica]
        if (step + 1) % (every * accumulation) == 0:
            delta = outer - np.mean(inner, axis=0)
            outer_momentum = MOMENTUM * outer_momentum + delta
            outer = outer - OUTER_RATE * (delta + MOMENTUM * outer_momentum)
            inner = [outer.copy() for _ in range(replicas)]
    return outer


@pytest.mark.mesh(devices=2)
def test_a_two_replica_diloco_run_is_the_algorithms_float64_reference():
    """Two replicas, each two inner steps of SGD with momentum between syncs, and a Nesterov outer step:
    the outer parameters after four steps, two syncs, are the NumPy float64 reference's."""
    with jax.enable_x64(new_val=True):
        run = trainer(2, diloco=DiLoCo(every=2, rate=OUTER_RATE, momentum=MOMENTUM),
                      optimizer=optax.sgd(RATE, momentum=0.5))
        initial = jax.device_get(run.initial_state().variables["params"]["Dense_0"])
        state = run.fit(dataset(), steps=4, log_every=1)
        found = params(state.variables["params"])
    # Each parameter sums its replica's 4 rows' gradient terms over 4 inner steps, and 2 replicas' means.
    assert_computes_the_oracle(found, reference(initial, 4, 2, 2, 0.5), "DiLoCo's outer parameters",
                               roundings=4 * 4 * 2 + 2 * 2)


@pytest.mark.mesh(devices=2)
def test_a_two_replica_diloco_run_pooling_two_steps_an_update_is_the_float64_reference():
    """Each replica pooling two steps' gradients into each inner update, two updates between meetings:
    the outer parameters after eight steps, two meetings, are the reference's, each update the mean of
    its window's gradients (row 876 review)."""
    with jax.enable_x64(new_val=True):
        run = trainer(2, diloco=DiLoCo(every=2, rate=OUTER_RATE, momentum=MOMENTUM),
                      optimizer=optax.sgd(RATE, momentum=0.5), accumulation=2)
        initial = jax.device_get(run.initial_state().variables["params"]["Dense_0"])
        state = run.fit(dataset(), steps=8, log_every=1)
        found = params(state.variables["params"])
    # Each parameter sums its replica's 4 rows' terms over 8 steps, and 2 replicas' means.
    assert_computes_the_oracle(found, reference(initial, 8, 2, 2, 0.5, accumulation=2),
                               "a pooling DiLoCo run's outer parameters", roundings=4 * 8 * 2 + 2 * 2)


@pytest.mark.mesh(devices=2)
def test_a_diloco_run_reads_a_batchs_scalar_input_in_every_group():
    """A batch input of no rows, a scalar every group reads whole, reaches each group's step: a run whose
    loss weighs its error by it trains as it does with the weight folded into its rate."""
    def weighted(rows):
        return {**rows, "weight": np.asarray(0.5, np.float32)}

    with jax.enable_x64(new_val=True):
        scaled = trainer(2, diloco=DiLoCo(every=1, rate=1.0, momentum=0.0),
                         objective=Regression64(scaled=True)).fit(
            dataset(train=lambda: map(weighted, Counting())), steps=2, log_every=1)
        halved = trainer(2, diloco=DiLoCo(every=1, rate=1.0, momentum=0.0),
                         optimizer=optax.sgd(RATE / 2)).fit(dataset(), steps=2, log_every=1)
    assert_computes_the_oracle(params(scaled.variables["params"]), params(halved.variables["params"]),
                               "a scalar-weighted DiLoCo run's parameters", roundings=8 * 2 + 4)
    """Two replicas that sync after every inner update with plain SGD inside and an outer step of rate 1
    and no momentum train as one data-parallel run over the whole batch: the replicas' mean step is the
    whole batch's."""
    with jax.enable_x64(new_val=True):
        synced = DiLoCo(every=1, rate=1.0, momentum=0.0)
        diloco = trainer(2, diloco=synced).fit(dataset(), steps=3, log_every=1)
        plain = trainer(1).fit(dataset(), steps=3, log_every=1)
    # Each parameter sums its 8 rows' gradient terms over 3 steps.
    assert_computes_the_oracle(params(diloco.variables["params"]), params(plain.variables["params"]),
                               "a synchronous DiLoCo run's parameters", roundings=8 * 3 + 2 * 3)


def test_a_single_replica_diloco_run_between_syncs_trains_as_a_plain_run():
    """One replica between syncs takes the plain run's steps, its draws among them: its inner parameters
    and optimizer state after three steps of a loss its key perturbs, no sync among them, are the plain
    run's to the bit."""
    with jax.enable_x64(new_val=True):
        noisy = Regression64(noisy=True)
        diloco = trainer(1, diloco=DiLoCo(every=8), optimizer=optax.adam(RATE), objective=noisy).fit(
            dataset(), steps=3)
        plain = trainer(1, optimizer=optax.adam(RATE), objective=Regression64(noisy=True)).fit(dataset(),
                                                                                              steps=3)
    for field in ("variables", "opt_state"):
        for mine, theirs in zip(jax.tree.leaves(getattr(diloco.inner, field)),
                                jax.tree.leaves(getattr(plain, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(mine)[0], np.asarray(theirs), err_msg=field)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("accumulation", [1, 2])
def test_a_resumed_diloco_run_restores_each_replica_and_ends_where_an_unbroken_one_does(tmp_path,
                                                                                          accumulation):
    """A DiLoCo run stopped between syncs and resumed restores each replica's inner state and the outer
    state, and ends where an unbroken run does, to the bit; with two steps an update, stopped inside a
    window, each replica's part-filled window too."""
    def run(path=None):
        return trainer(2, diloco=DiLoCo(every=2, rate=OUTER_RATE, momentum=MOMENTUM),
                       optimizer=optax.adam(RATE), path=path, accumulation=accumulation)

    with jax.enable_x64(new_val=True):
        unbroken = run().fit(dataset(), steps=5, log_every=1)
        stopped = run(tmp_path / "run").fit(dataset(), steps=3, checkpoint_every=1, log_every=1)
        assert (stopped.inner.accumulation is not None) == (accumulation > 1)
        restored, _, _ = run(tmp_path / "run").place()
        for mine, theirs in zip(bits(stopped), bits(restored), strict=True):
            np.testing.assert_array_equal(mine, theirs)
        resumed = run(tmp_path / "run").fit(dataset(), steps=5, checkpoint_every=1, log_every=1)
    for mine, theirs in zip(bits(resumed), bits(unbroken), strict=True):
        np.testing.assert_array_equal(mine, theirs)


@pytest.mark.mesh(devices=2)
def test_a_diloco_mesh_holds_each_replicas_rows_and_inner_copy_on_its_own_devices():
    """A mesh of two DiLoCo replicas splits a batch's rows over them, holds each replica's inner copy on
    its own devices alone, and splits the outer parameters over the replicas where a dimension divides,
    each holding a share of the model only their meeting reads."""
    from dew.distributed.batches import batch_shardings
    from dew.mesh import DILOCO_AXIS

    with jax.enable_x64(new_val=True):
        run = trainer(2, diloco=DiLoCo(every=2))
        state, _, _ = run.place()
        mesh = run.device_mesh
        rows = batch_shardings(mesh, {"x": np.zeros((BATCH, FEATURES))})["x"]
    assert rows.spec[0] == DILOCO_AXIS or DILOCO_AXIS in rows.spec[0]
    kernel = state.variables["params"]["Dense_0"]["kernel"]
    assert DILOCO_AXIS in jax.tree.leaves(tuple(kernel.sharding.spec))
    assert {shard.data.shape for shard in kernel.addressable_shards} == {(FEATURES, 1)}
    for leaf in jax.tree.leaves(state.inner.variables["params"]):
        assert leaf.shape[0] == 2 and leaf.sharding.spec[0] == DILOCO_AXIS
        assert {shard.device for shard in leaf.addressable_shards if shard.index[0] == slice(0, 1)} == {
            mesh.devices.ravel()[0]}


@pytest.mark.mesh(devices=2)
def test_a_configured_run_trains_two_diloco_groups_from_its_command_line(tmp_path):
    """`dew train` with `--set trainer.mesh.diloco=2` and a DiLoCo of one window between meetings trains
    two groups: its checkpoint holds each group's inner copy, stacked over the groups, beside the outer
    model they met in."""
    from run_kinds import lm, trained_by_cli

    from dew.checkpoints import Checkpoints as Saved

    run = trained_by_cli(lm(tmp_path, drawn=False), tmp_path, "diloco", {
        "trainer.mesh.diloco": "2", "trainer.checkpoint_every": "1",
        "trainer.diloco": '{"class": "dew.training.diloco:DiLoCo", "fields": {"every": 1}}'})
    stored = Saved(str(run)).stored()
    inner, outer = stored["inner"]["variables"]["params"], stored["variables"]["params"]
    assert jax.tree.structure(inner) == jax.tree.structure(outer)
    assert all(mine.shape == (2, *theirs.shape)
               for mine, theirs in zip(jax.tree.leaves(inner), jax.tree.leaves(outer), strict=True))


@pytest.mark.mesh(devices=2)
def test_a_supplied_diloco_state_trains_on_from_each_groups_own_state():
    """A DiLoCo state supplied to a fit (`fit(state=...)`), its optimizer a schedule's, trains on from each
    group's inner state and the outer model it holds: two more steps, each a meeting, move the outer
    parameters, all finite."""
    from dew.optim import Cosine, OptimConfig

    def run():
        return trainer(2, diloco=DiLoCo(every=1, rate=1.0, momentum=0.0),
                       optimizer=OptimConfig(learning_rate=Cosine(peak_value=RATE, warmup_steps=1,
                                                                  decay_steps=8)))

    with jax.enable_x64(new_val=True):
        first = run().fit(dataset(), steps=2, log_every=1)
        before = params(first.variables["params"])
        resumed = run().fit(dataset(), steps=4, log_every=1, state=first)
    assert int(resumed.step) == 4
    after = params(resumed.variables["params"])
    assert np.isfinite(after).all() and not np.array_equal(after, before)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("statistics", ["weighted", "rowwise", "drawn"])
def test_a_diloco_run_of_replayed_windows_trains_as_one_pooling_its_gradients(statistics):
    """A loss whose statistics a window replays rather than pools, a `Weighted`, each row's error, or those
    beside the update's draws of every example (row 876 review): the groups' windows are shaped and
    allocated as one group's step reads them, its examples among the whole update's, and with every inner
    update's window of two steps the run trains as one whose loss pools them."""
    with jax.enable_x64(new_val=True):
        def run(objective):
            return trainer(2, diloco=DiLoCo(every=2, rate=OUTER_RATE, momentum=MOMENTUM), objective=objective,
                           accumulation=2).fit(dataset(), steps=4, log_every=1)
        replayed, pooled = run(Regression64(**{statistics: True})), run(Regression64())
    # Each parameter sums its replica's rows' terms over 4 steps, and 2 replicas' means.
    assert_computes_the_oracle(params(replayed.variables["params"]), params(pooled.variables["params"]),
                               "a replaying DiLoCo run's outer parameters", roundings=4 * 4 * 2 + 4)


def test_a_diloco_trainer_refuses_what_it_cannot_express_as_it_is_made():
    """A DiLoCo trainer whose layout keeps state on the host, or which records each example's draws, is
    refused as it is made, before any state is supplied or placed (row 876 review): its host step and its
    offload would place and step the outer model alone, and no one record holds every group's draws."""
    for options in ({"layout": Layout(min_shard=1, tolerance=1.0, host=("opt_state",))}, {"record": True}):
        with pytest.raises(ValueError):
            Trainer(Regression64(), optax.adam(RATE), key=jax.random.key(0), diloco=DiLoCo(every=2),
                    **options)


def test_a_meeting_and_a_new_stage_drop_each_groups_part_filled_window():
    """At a meeting, and as a ramp's stage begins, each group's part-filled window is dropped as a rejected
    attempt drops one, its microstep back at the window's start, and the run's microstep the first
    group's: no update pools gradients of parameters the meeting replaced, or of an earlier stage."""
    from dew.state import Accumulation, Replicas, TrainState
    from dew.training.diloco import dropped

    def stacked(*values):
        return jnp.asarray(values)

    window = Accumulation(gradient=None, mass=stacked(2.0, 1.0), statistics=(stacked(3.0, 5.0),),
                          effects=(), qk_stats=(), batches=None, variables=None, attempts=None, tokens=None)
    inner = Replicas(microstep=stacked(6, 5), updates=stacked(3, 2),
                     variables={"params": {"w": jnp.ones((2, 3))}}, opt_state=(), ema=None, scale=None,
                     accumulation=window, deferred=None)
    state = TrainState(step=jnp.asarray(5), microstep=jnp.asarray(6), updates=jnp.asarray(3),
                       variables={"params": {"w": jnp.ones(3)}}, opt_state=(), ema=None,
                       key=jax.random.key(0), scale=None, window_size=jnp.asarray(2), inner=inner)
    diloco = DiLoCo(every=1, rate=1.0, momentum=0.0)
    met = diloco.synced(dataclasses.replace(state, opt_state=diloco.outer.init(state.variables["params"])))
    for after in (met, dropped(state, 2)):
        assert after.inner.microstep.tolist() == [6, 4] and int(after.microstep) == 6
        assert after.inner.accumulation.mass.tolist() == [0.0, 0.0]
        assert after.inner.accumulation.statistics[0].tolist() == [0.0, 0.0]


def test_a_replicas_fields_are_the_train_states_marked_ones_and_its_typed_view_names_them():
    """A DiLoCo replica holds its own of exactly the `TrainState` fields marked `REPLICA`, in their order
    and with their types (row 4139: `Replicas` redeclared eight of them), and `ReplicaState`, the view
    DiLoCo's readers are typed through, names the same fields with the same types."""
    from dew.state import REPLICA, REPLICA_FIELDS, Replicas, ReplicaState, TrainState

    marked = [(field.name, field.type) for field in dataclasses.fields(TrainState)
              if all(field.metadata.get(key) == value for key, value in REPLICA.items())]
    assert len(marked) == 8
    assert [(field.name, field.type) for field in dataclasses.fields(Replicas)] == marked
    assert tuple(name for name, _ in marked) == REPLICA_FIELDS
    view = {name: kind for name, kind in ReplicaState.__annotations__.items() if not name.startswith("__")}
    assert view == dict(marked)


@pytest.mark.mesh(devices=2)
def test_a_diloco_runs_profiles_are_kept_snapshots_that_rebuild_its_groups_mean_average(tmp_path):
    """Under DiLoCo the configured optimizer, and the power profiles it keeps, are the groups' inner ones
    (`DiLoCo.started`). Each checkpoint is a snapshot of them, kept past `keep` (row 4317), and the post-hoc
    EMA rebuilt from them is the groups' mean of each group's own reconstruction: within fp32 rounding at
    the weights' scale of the mean of the EMAs the groups track directly, as a run of one copy's is, each
    stored profile's mean over a hundred times that bound from it."""
    objective = Regression()
    objective.ema = EMASpec(decay=power_decay(0.07))
    run = Trainer(objective, power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0),
                  layout=Layout(min_shard=1, tolerance=1.0), mesh=MeshSpec(diloco=2).build(jax.devices()[:2]),
                  diloco=DiLoCo(every=2), checkpoints=Checkpoints(str(tmp_path / "run"), keep=1))
    state = run.fit(dataset(), steps=96, log_every=96, checkpoint_every=4)
    run.checkpoints.wait()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    assert checkpoints.profile_steps() == list(range(4, 97, 4))
    groups_mean = jax.tree.map(lambda leaf: np.mean(np.asarray(leaf), axis=0), state.inner.ema["params"])
    rebuilt = checkpoints.posthoc_ema(0.07)
    scale = max(float(np.max(np.abs(leaf))) for leaf in jax.tree.leaves(groups_mean))
    bound = 2 * np.finfo(np.float32).eps * scale
    assert largest_difference(rebuilt, groups_mean) <= bound
    for average in checkpoints.restore_profiles(96):
        mean = jax.tree.map(lambda leaf: np.mean(leaf, axis=0), average)
        assert largest_difference(mean, groups_mean) > 100 * bound


class Tabled(Objective):
    """A regression whose `count` collection moves every step, beside a `constants` table no step moves,
    with an EMA of its weights."""

    def __init__(self):
        self.ema = EMASpec(decay=optax.constant_schedule(0.5))

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.zeros((FEATURES,))}, "count": {"n": jnp.zeros(())},
                "constants": {"table": jnp.arange(1000.0)}}

    def loss(self, variables, batch, step):
        error = jnp.square(batch["x"] @ variables["params"]["w"] - batch["y"][:, 0])
        return (Ratio(jnp.sum(error), jnp.asarray(error.size, jnp.float32)),
                Aux(metrics={}, variables={"count": {"n": variables["count"]["n"] + 1}}))


@pytest.mark.mesh(devices=2)
def test_a_diloco_step_stores_its_groups_collections_once_and_their_ema_against_their_weights(tmp_path):
    """A DiLoCo run's groups hold their own copies of the model's collections and of its EMA (row 4318). A
    step writes the groups' collections but `params` to the run's content-addressed store, as it writes the
    outer model's (docs/guides/checkpoints.md), so the table no step moves is one entry of the groups' and
    one of the outer model's, which every step records; and it stores the groups' EMA as byte planes of its
    difference from their weights, which Orbax alone decodes to the groups' EMA bit for bit. The run resumes
    from those steps and ends where an unbroken one does, to the bit."""
    import orbax.checkpoint as ocp

    def run(path=None):
        return trainer(2, diloco=DiLoCo(every=2, rate=OUTER_RATE, momentum=MOMENTUM), objective=Tabled(),
                       path=path)

    unbroken = run().fit(dataset(), steps=5, log_every=1)
    stopped = run(tmp_path / "run").fit(dataset(), steps=3, checkpoint_every=1, log_every=1)

    manager = ocp.CheckpointManager(str(tmp_path / "run"), item_handlers=ocp.PyTreeCheckpointHandler())
    assert set(manager.metadata(3).item_metadata.tree["inner"]["variables"]) == {"params"}
    recorded = set.intersection(*(set(frozen_entries(str(tmp_path / "run" / str(step))))
                                  for step in (1, 2, 3)))
    with ocp.PyTreeCheckpointer() as checkpointer:
        shapes = sorted(tuple(leaf.shape) for entry in recorded for leaf in jax.tree.leaves(
            checkpointer.metadata(epath.Path(tmp_path / "run") / entry).item_metadata.tree))
    assert shapes == [(2, 1000), (1000,)]

    tree = manager.restore(3, args=ocp.args.PyTreeRestore())
    deltas = manager.metadata(3).custom_metadata["deltas"]["inner/ema"]
    weights = dict(jax.tree_util.tree_flatten_with_path(tree["inner"]["variables"])[0])
    averages = dict(jax.tree_util.tree_flatten_with_path(stopped.inner.ema)[0])
    assert deltas
    for path, stored in jax.tree_util.tree_flatten_with_path(tree["inner"]["ema"])[0]:
        want = np.asarray(averages[path])
        if jax.tree_util.keystr(path) in deltas:
            live = np.asarray(weights[path])
            unsigned = np.dtype(f"uint{8 * live.dtype.itemsize}")
            planes = np.asarray(stored)
            bits_of = sum(planes[k].astype(unsigned) << unsigned.type(8 * k) for k in range(len(planes)))
            stored = (live.view(unsigned) ^ bits_of).view(live.dtype)
        np.testing.assert_array_equal(np.asarray(stored), want)

    resumed = run(tmp_path / "run").fit(dataset(), steps=5, checkpoint_every=1, log_every=1)
    for mine, theirs in zip(bits(resumed), bits(unbroken), strict=True):
        np.testing.assert_array_equal(mine, theirs)
