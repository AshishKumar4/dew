"""DiLoCo (row 876): replica groups each training an inner copy of the model, every few updates averaged
into the outer model by an outer optimizer. What is asserted is the algorithm's arithmetic against an
independent float64 reference, its limits (one replica, one inner update), where the replicas and their
rows live, and that a resumed run restores each replica exactly."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import BATCH, FEATURES, Counting, dataset
from flax import linen as nn
from reference_error import assert_computes_the_oracle

from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.diloco import DiLoCo

RATE, OUTER_RATE, MOMENTUM = 0.05, 0.7, 0.9


class Affine64(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2, dtype=jnp.float64, param_dtype=jnp.float64)(x.astype(jnp.float64))


class Regression64(Objective):
    """Squared error of a float64 affine map against `2 * x[:, :2]`."""

    def __init__(self):
        self.model = Affine64()

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        return jnp.mean((prediction - batch["y"]) ** 2), Aux({})


def trainer(replicas: int, *, diloco: DiLoCo | None = None, optimizer=None, path=None, **options) -> Trainer:
    return Trainer(Regression64(), optax.sgd(RATE) if optimizer is None else optimizer, key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0),
                   mesh=MeshSpec(diloco=replicas).build(jax.devices()[:replicas]), diloco=diloco,
                   checkpoints=None if path is None else Checkpoints(str(path)), **options)


def values(state) -> list[np.ndarray]:
    """Every leaf of `state` as NumPy, a key as its data."""
    return [np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)
            for leaf in jax.tree.leaves(state)]


def params(tree) -> np.ndarray:
    """A Dense's kernel and bias as one float64 vector, in that order."""
    dense = tree["Dense_0"]
    return np.concatenate([np.ravel(np.asarray(dense[name], np.float64)) for name in ("kernel", "bias")])


def reference(initial: dict, steps: int, replicas: int, every: int, inner_momentum: float) -> np.ndarray:
    """DiLoCo in NumPy float64 (Douillard et al., 2023, Algorithm 1): each replica takes `every` steps of
    SGD with momentum on its contiguous share of each batch's rows from the outer parameters, the outer
    gradient is the outer parameters less the replicas' mean, and the outer step is Nesterov SGD."""
    kernel, bias = np.asarray(initial["kernel"], np.float64), np.asarray(initial["bias"], np.float64)
    outer = np.concatenate([kernel.ravel(), bias.ravel()])
    outer_momentum = np.zeros_like(outer)
    inner = [outer.copy() for _ in range(replicas)]
    inner_velocity = [np.zeros_like(outer) for _ in range(replicas)]
    source = Counting()
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
            gradient = np.concatenate([(scale * x[rows].T @ residual).ravel(), scale * residual.sum(0)])
            inner_velocity[replica] = inner_momentum * inner_velocity[replica] + gradient
            inner[replica] = inner[replica] - RATE * inner_velocity[replica]
        if (step + 1) % every == 0:
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
def test_diloco_of_one_inner_update_a_sync_and_plain_sgd_is_data_parallel_training():
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
    """One replica between syncs takes the plain run's steps: its inner parameters and optimizer state
    after three steps, no sync among them, are the plain run's to the bit."""
    with jax.enable_x64(new_val=True):
        diloco = trainer(1, diloco=DiLoCo(every=8), optimizer=optax.adam(RATE)).fit(dataset(), steps=3)
        plain = trainer(1, optimizer=optax.adam(RATE)).fit(dataset(), steps=3)
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
        restored, _, _ = run(tmp_path / "run").place()
        for mine, theirs in zip(values(stopped), values(restored), strict=True):
            np.testing.assert_array_equal(mine, theirs)
        resumed = run(tmp_path / "run").fit(dataset(), steps=5, checkpoint_every=1, log_every=1)
    for mine, theirs in zip(values(resumed), values(unbroken), strict=True):
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
