"""Post-hoc EMA: the Algorithm 3 solve, power-function averages kept by the
solver, their snapshots, and averages reconstructed from them."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Data, Regression

from dew.checkpoints import Checkpoints, Ranking
from dew.config import OptimConfig, RunConfig
from dew.objectives.base import EMASpec
from dew.training import Layout, Trainer
from dew.training.optim import PowerProfilesState, power_profiles
from dew.training.posthoc import coefficients, exponent, power_decay

# Outputs of NVlabs/edm2 training/phema.py (std_to_exp, power_function_beta,
# solve_posthoc_coefficients) on the same inputs, float64.
REFERENCE_EXPONENTS = {0.05: 16.972198602303447, 0.10: 6.937203937601809, 0.07: 11.244939447857924}
REFERENCE_BETA_AT_10 = {0.05: 0.15053493247717484, 0.10: 0.4333247206821348}
EDM2_SNAPSHOTS = [(4096 * i, std) for i in range(1, 9) for std in (0.05, 0.10)]
REFERENCE_AT_007 = [
    3.8923196884360404e-12, -9.5612913162106657e-11, 1.7380077549162091e-08, -1.1822569154039657e-07,
    2.2239824574642905e-06, -9.1411604368222978e-06, 6.6088575336683495e-05, -2.1443138302308976e-04,
    8.8659745140010418e-04, -2.5444840586677279e-03, 7.2521780242067900e-03, -1.9405441994355861e-02,
    4.2397913310664982e-02, -1.0864094698057378e-01, 3.8708213040015665e-01, 6.9312741477016915e-01]
REFERENCE_ONE_PROFILE = [-5.4318017037919378e-25, -3.1130114879646682e-17, -4.1687520707014240e-11,
                         -2.1418421161640175e-04, 1.0002141842533039e+00]


def test_the_solve_matches_the_reference_implementation():
    for std, gamma in REFERENCE_EXPONENTS.items():
        assert exponent(std) == pytest.approx(gamma, rel=1e-14)
    for std, beta in REFERENCE_BETA_AT_10.items():
        # After nine updates, the tenth keeps (1 - 1/10)^(γ + 1), in fp32.
        assert float(power_decay(std)(9)) == pytest.approx(beta, rel=1e-6)
    np.testing.assert_allclose(coefficients(EDM2_SNAPSHOTS, 32768, 0.07), REFERENCE_AT_007,
                               rtol=1e-9, atol=1e-15)
    one_profile = [(updates, 0.05) for updates in (1000, 2000, 3000, 5000, 8000)]
    np.testing.assert_allclose(coefficients(one_profile, 8000, 0.03), REFERENCE_ONE_PROFILE,
                               rtol=1e-9, atol=1e-15)


def test_an_average_the_snapshots_hold_is_reconstructed_as_itself():
    snapshots = EDM2_SNAPSHOTS[:EDM2_SNAPSHOTS.index((4096 * 5, 0.10)) + 1]
    expected = np.eye(len(snapshots))[-1]
    np.testing.assert_allclose(coefficients(snapshots, 4096 * 5, 0.10), expected, atol=1e-9)


def test_a_reconstructed_average_matches_one_tracked_directly(tmp_path):
    """A run tracks power EMAs of relative std 0.05 and 0.10 in its solver,
    snapshotted at every fourth step, and one of 0.07 as its own EMA. The
    average of 0.07 rebuilt from the snapshots lands on the tracked one to
    within fp32 rounding at the weights' scale; each stored average sits
    over a hundred times that bound from it."""
    objective = Regression()
    objective.ema = EMASpec(decay=power_decay(0.07))
    trainer = Trainer(objective, power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0),
                      checkpoints=Checkpoints(str(tmp_path / "run")))
    state = trainer.fit(Data(), steps=96, log_every=96, checkpoint_every=4)
    trainer.checkpoints.wait()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    assert checkpoints.profile_steps() == list(range(4, 97, 4))
    assert checkpoints.profile_metadata(96) == (96, (float(np.float32(0.05)), float(np.float32(0.1))))
    direct = jax.tree.map(np.asarray, state.ema["params"])
    rebuilt = Checkpoints(str(tmp_path / "run")).posthoc_ema(0.07)
    stored = checkpoints.restore_profiles(96)

    def distance(a, b):
        return max(float(np.max(np.abs(x - y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b),
                                                                 strict=True))
    scale = max(float(np.max(np.abs(leaf))) for leaf in jax.tree.leaves(direct))
    bound = 2 * np.finfo(np.float32).eps * scale
    assert distance(rebuilt, direct) <= bound
    assert all(distance(average, direct) > 100 * bound for average in stored)
    assert jax.tree.map(lambda leaf: leaf.dtype, rebuilt) == jax.tree.map(lambda leaf: leaf.dtype, direct)


def test_a_checkpoint_writes_the_averages_once_and_restores_them(tmp_path):
    """A snapshot is the ordinary checkpoint itself: averages are stored
    once in opt_state, retained, and every restore reads them bit for bit."""
    def trainer():
        return Trainer(Regression(), power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0),
                       checkpoints=Checkpoints(str(tmp_path / "run")))
    state = trainer().fit(Data(), steps=8, log_every=8, checkpoint_every=4)
    Checkpoints(str(tmp_path / "run")).wait()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    written = checkpoints._open().item_metadata(8)
    assert dict(written)["opt_state"]["averages"] is not None
    stored = tuple(checkpoints.stored(8)["opt_state"]["averages"])
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), stored) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), state.opt_state.averages)
    restored, _, _ = trainer().place()
    untyped, _ = checkpoints.restore(None, 8)
    for averages in (restored.opt_state.averages, tuple(untyped["opt_state"]["averages"])):
        for held, expected in zip(jax.tree.leaves(averages), jax.tree.leaves(state.opt_state.averages),
                                  strict=True):
            np.testing.assert_array_equal(np.asarray(held), np.asarray(expected))

def test_a_run_record_names_its_profiles_and_the_solver_keeps_them():
    config = RunConfig(optim=OptimConfig(ema_profiles=(0.05, 0.10)))
    loaded = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert loaded == config

    params = {"w": jnp.ones(3)}
    solver = loaded.optim.build(10)
    state = solver.init(params)
    assert isinstance(state, PowerProfilesState)
    np.testing.assert_array_equal(state.stds, np.float32([0.05, 0.10]))
    _, state = solver.update({"w": jnp.ones(3)}, state, params)
    assert int(state.updates) == 1
    # The first update takes the weights it made whole.
    for average in state.averages:
        np.testing.assert_array_equal(average["w"], params["w"] - 2.7e-4 * np.ones(3, np.float32))


def test_a_local_checkpoint_keeps_the_profiles_in_its_state(tmp_path):
    trainer = Trainer(Regression(), power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0))
    state = trainer.fit(Data(), steps=2, log_every=2)
    checkpoints = Checkpoints(str(tmp_path / 'run'), local_directory=str(tmp_path / 'local'), local_every=1)
    checkpoints.save_local(2, state, None)
    checkpoints.wait()
    template = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), state
    )
    restored, _ = checkpoints.restore(template, 2)
    for held, expected in zip(jax.tree.leaves(restored.opt_state.averages),
                              jax.tree.leaves(state.opt_state.averages), strict=True):
        np.testing.assert_array_equal(np.asarray(held), np.asarray(expected))
    assert checkpoints.profile_steps() == []


def test_an_interrupted_snapshot_restores_the_previous_step_and_is_never_reconstructed(tmp_path, monkeypatch):
    directory = str(tmp_path / 'run')
    trainer = Trainer(Regression(), power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0))
    state = trainer.fit(Data(), steps=2, log_every=2)
    checkpoints = Checkpoints(directory, keep=1)
    checkpoints.save(2, state, None, ranking=Ranking('train/loss', 0.2))
    checkpoints.wait()

    interrupted = Checkpoints(directory, keep=1)

    import orbax.checkpoint as ocp

    handler = ocp.PyTreeCheckpointHandler()

    def fail_before_commit(directory):
        raise OSError('interrupted before commit')

    monkeypatch.setattr(handler, 'finalize', fail_before_commit)
    interrupted._manager = ocp.CheckpointManager(
        directory, options=ocp.CheckpointManagerOptions(enable_async_checkpointing=True),
        item_handlers=handler)
    interrupted.save(4, state.replace(step=jnp.int32(4)), None, ranking=Ranking('train/loss', 0.1))
    with pytest.raises(OSError, match='interrupted before commit'):
        interrupted.wait()
    # The one write failed before its atomic commit. A fresh process must
    # not select that step for resume, best weights, or reconstruction.
    assert interrupted.latest == interrupted.best == 2
    fresh = Checkpoints(directory, keep=1)
    assert fresh.latest == fresh.best == 2
    assert fresh.profile_steps() == [2]
    template = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), state
    )
    restored, _ = fresh.restore(template)
    def bits(leaf):
        if isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf)

    for held, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(bits(held), bits(expected))

    fresh.save(6, state.replace(step=jnp.int32(6)), None, ranking=Ranking('train/loss', 0.05))
    fresh.wait()
    assert fresh.latest == fresh.best == 6
    assert fresh.profile_steps() == [2, 6]
    assert fresh._open().all_steps() == [2, 6]


@pytest.mark.parametrize('kind', ['array', 'list', 'frozen'])
def test_power_profiles_keeps_the_parameters_pytree(kind):
    from flax.core import freeze

    params = jnp.ones(3)
    if kind == 'list':
        params = [params, {'nested': jnp.ones(2)}]
    elif kind == 'frozen':
        params = freeze({'w': params, 'nested': {'b': jnp.ones(2)}})
    solver = power_profiles(optax.sgd(0.1), (0.05, 0.10))
    state = solver.init(params)
    updates, state = solver.update(jax.tree.map(jnp.ones_like, params), state, params)
    expected = optax.apply_updates(params, updates)
    for average in state.averages:
        assert jax.tree.structure(average) == jax.tree.structure(expected)
        for held, weight in zip(jax.tree.leaves(average), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(held), np.asarray(weight))


def test_reconstruction_accumulates_bfloat16_weights_in_float32(tmp_path):
    trainer = Trainer(Regression(), power_profiles(optax.sgd(0.1), (0.05, 0.10)), key=jax.random.key(0))
    state = trainer.fit(Data(), steps=2, log_every=2)
    averages = tuple(jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), average)
                     for average in state.opt_state.averages)
    state = state.replace(variables=jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), state.variables),
                          opt_state=state.opt_state._replace(averages=averages))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    checkpoints.save(2, state, None)
    checkpoints.wait()
    rebuilt = Checkpoints(str(tmp_path / 'run')).posthoc_ema(float(np.float32(0.05)))
    for held, expected in zip(jax.tree.leaves(rebuilt), jax.tree.leaves(averages[0]), strict=True):
        assert held.dtype == np.asarray(expected).dtype
        np.testing.assert_array_equal(held.view(np.uint16), np.asarray(expected).view(np.uint16))
