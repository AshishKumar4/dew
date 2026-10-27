"""The plan's host offload (`dew.training.planning.FitSearch`): where a step does not fit its devices, the
plan keeps the optimizer state and the EMA in its host's pinned memory before it recomputes anything,
and a remat rung that keeps its residuals there stands before the next that recomputes them. A plan
that offloads meets the first plan that keeps everything on the device: a run that times its plans
keeps the offload only where it runs clearly faster, and one that times none keeps the device-resident
plan wherever one fits.

A CPU device has the pinned host memory and no accounting of it apart from its own (tests/
test_host_placement.py), so the plan's choice is held here on budgets the devices report
(`test_plan.free`) and timings the test states; what an offloading remat rung keeps in host memory is
read off a step compiled for a compile-only H100, whose compiler accounts for it."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import compile_only_gpus
from reference_error import as_given, assert_as_exact
from test_host_placement import memory_kinds
from test_layout_validation import step_loss
from test_plan import BATCH, LIMIT, ROWS, SEQ, free, needs, one_device, token_files, trained, windows

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.lm import LMObjective
from dew.training import Trainer

OFFLOADED = ["opt_state", "ema"]


def offloading(split=None, seq_len=SEQ, checkpoints=None, **fields):
    """A decoder whose parameters outweigh its activations, two MLPs 1024 wide, trained by Adam with an
    EMA: its optimizer state and EMA, three times the parameters, are what the plan may keep in host
    memory."""
    model = CausalTransformer(vocab_size=64, emb_features=64, num_layers=2, num_heads=2, mlp_features=1024,
                              max_seq_len=SEQ, **fields)
    return one_device(Trainer(LMObjective(model, seq_len=seq_len, ema_decay=0.9), optax.adam(1e-3),
                              key=jax.random.key(0), split=split, checkpoints=checkpoints))


def searched(trainer):
    """The compiles the plan's search made, as (remat, host, fits)."""
    return [(probe["remat"], probe.get("host"), probe["fits"]) for probe in trainer.plan["probes"]]


def least_need(monkeypatch, remat=None, corpus=None) -> int:
    """The least a plan at `remat` with the state on the device needs to read BATCH, or the windows of
    `corpus` a run fits on (`windows`): its most microbatches, one a row, under whichever head its rung
    takes needs least, whole, tiled or tiled with its gradient in the backward (`recompute_more`)."""
    def at(steps: int) -> int:
        """The need with the head `steps` rungs along: whole, tiled, then its gradient in the backward."""
        trainer = (offloading(split=ROWS, remat=remat) if corpus is None
                   else offloading(split=8, remat=remat, seq_len=4))
        for step in (trainer.objective.tile_head, trainer.objective.unfuse_head)[:steps]:
            step()
        return needs(monkeypatch, trainer, None if corpus is None else (
            lambda: trainer.fit(windows(corpus), steps=1, checkpoint_every=None)))
    return min(at(steps) for steps in range(3))


def device_bound(monkeypatch, corpus=None) -> int:
    """Room for no plan that keeps the state on the device: a byte less than the least any needs, at
    the ladder's last remat rung."""
    return least_need(monkeypatch, "full", corpus) - 1


def test_a_step_that_fits_only_with_its_state_in_host_memory_keeps_it_there(monkeypatch):
    """With room for no plan that keeps the state on the device, the plan keeps the optimizer state and
    the EMA in host memory before it recomputes anything, after every rung of its device-resident search
    missed, and the run trains from there with them in pinned host memory and the parameters on the
    device."""
    free(monkeypatch, device_bound(monkeypatch))
    trainer = offloading()
    state = trained(trainer)
    assert (trainer.plan.get("host"), trainer.plan["remat"]) == (OFFLOADED, None)
    compiles = searched(trainer)
    assert not any(fits for _, host, fits in compiles if host == [])
    assert {remat for remat, host, _ in compiles if host == []} >= {None, "minimal", "full"}
    assert memory_kinds(state.opt_state) == memory_kinds(state.ema) == {"pinned_host"}
    assert memory_kinds(state.variables) == {"device"}


def tight(monkeypatch) -> int:
    """Room for no plan with the state on the device and no recomputation, and for the most
    microbatches at 'minimal': both an offload and a recomputation fit."""
    least, recomputing = least_need(monkeypatch), needs(monkeypatch, offloading(split=ROWS, remat="minimal"))
    assert recomputing < least, (recomputing, least)
    return least - 1


@pytest.mark.parametrize("seconds, host, remat", [
    ([[0.2, 0.21], [0.1, 0.12]], OFFLOADED, None), ([[0.1, 0.12], [0.2, 0.21]], [], "minimal"),
    ([[0.2, 0.3], [0.1, 0.25]], [], "minimal"), (None, [], "minimal")],
    ids=["offload-faster", "recompute-faster", "within-spread", "untimed"])
def test_a_plan_that_offloads_is_timed_against_the_first_that_keeps_the_state_on_the_device(
        monkeypatch, seconds, host, remat):
    """Where both an offload with no recomputation and a recomputation with the state on the device fit,
    the search compiles both, the device-resident plan first, and the run times them: the offload is kept
    only where its every sample beats the other's (`faster`). A compile that times nothing keeps the
    device-resident plan without timing."""
    free(monkeypatch, tight(monkeypatch))
    trainer = offloading()
    state, _, _ = trainer.place()
    timed = []
    step = trainer.compile(state, BATCH, None if seconds is None else (
        lambda steps: (timed.append(len(steps)), seconds)[1]))
    assert timed == ([] if seconds is None else [2])
    assert (trainer.plan.get("host"), trainer.plan["remat"]) == (host, remat)
    if seconds is not None:
        assert [(plan.get("host"), plan["remat"]) for plan in trainer.plan["timed"]] == [
            ([], "minimal"), (OFFLOADED, None)]
    state, loss, *_ = step(state, BATCH)
    assert np.isfinite(float(loss))
    assert memory_kinds(state.opt_state) == {"pinned_host" if host else "device"}


def device_bytes() -> int:
    """The bytes this process's live arrays hold in device memory."""
    return sum(array.nbytes for array in jax.live_arrays() if array.sharding.memory_kind == "device")


@pytest.mark.parametrize("shared", [False, True], ids=["own-buffers", "ema-is-the-parameters"])
def test_moving_the_state_to_host_memory_releases_the_device_bytes_the_plan_counted(shared):
    """The fit check counts a field the plan moves to host memory as the bytes the move releases
    (`rehomed`), and the move (`placed_as`) releases exactly those before the step runs: the device's
    live bytes fall by them, each moved source deleted. An EMA that is the parameters' own arrays shares
    their buffers, which the parameters still hold: it is copied to host memory, neither deleted nor
    counted."""
    from dew.training.distributed import placed_as, rehomed

    trainer = offloading()
    state, _, _ = trainer.place()
    if shared:
        state = dataclasses.replace(state, ema=state.variables["params"])
    trainer.layout = trainer.layout.keeping(OFFLOADED)
    shardings = trainer.shardings(state)
    moments, ema = jax.tree.leaves(state.opt_state), jax.tree.leaves(state.ema)
    counted = rehomed(state, shardings)
    assert counted == -sum(leaf.nbytes for leaf in moments + ([] if shared else ema))
    before = device_bytes()
    moved = placed_as(state, shardings)
    assert device_bytes() == before + counted
    assert all(leaf.is_deleted() for leaf in moments)
    assert all(leaf.is_deleted() is not shared for leaf in ema)
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(state.variables))
    assert memory_kinds(moved.opt_state) == memory_kinds(moved.ema) == {"pinned_host"}


def test_a_run_whose_plan_kept_its_state_in_host_memory_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    """A run whose plan kept the optimizer state and the EMA in host memory saves them there, and a run
    of the same configuration resumes from that checkpoint: it restores them where the run kept them, the
    devices holding them no more, and trains on at the rung the checkpoint names."""
    corpus = token_files(tmp_path)
    directory = str(tmp_path / "run")
    room = device_bound(monkeypatch, corpus)
    free(monkeypatch, room)
    first = offloading(seq_len=4, checkpoints=Checkpoints(directory))
    saved = first.fit(windows(corpus), steps=2, checkpoint_every=2)
    assert first.plan.get("host") == OFFLOADED
    # The resumed run's devices hold no moments and no EMA: what the first plan freed is free from the start.
    free(monkeypatch, room + sum(leaf.nbytes for leaf in jax.tree.leaves((saved.opt_state, saved.ema))))
    resumed = offloading(seq_len=4, checkpoints=Checkpoints(directory))
    state = resumed.fit(windows(corpus), steps=3, checkpoint_every=None)
    assert int(state.step) == 3
    assert (resumed.plan.get("host"), resumed.plan["remat"]) == (OFFLOADED, first.plan["remat"])
    assert memory_kinds(state.opt_state) == memory_kinds(state.ema) == {"pinned_host"}


def test_an_offloading_remat_rung_keeps_its_residuals_in_host_temporaries():
    """A decoder climbs from 'minimal' to 'minimal_offloaded', which keeps 'minimal''s residuals in
    pinned host memory. Compiled for a compile-only H100, its step keeps host temporaries where
    'minimal''s keeps none: what the device-resident search reads to refuse the rung
    (`dew.training.planning.keeps_host_temporaries`)."""
    model = CausalTransformer(vocab_size=64, emb_features=64, num_layers=2, num_heads=2, mlp_features=128,
                              max_seq_len=64, remat="minimal")
    assert model.recompute_more().remat == REMAT_POLICIES["minimal_offloaded"]
    mesh = Mesh(np.asarray(compile_only_gpus("sm90")), ("data",))
    whole = NamedSharding(mesh, P())

    def compiled(remat):
        objective = LMObjective(model.clone(remat=REMAT_POLICIES[remat]), 64)
        variables = jax.eval_shape(
            lambda: objective.model.init(jax.random.key(0), jnp.ones((1, 64), jnp.int32)))
        placed = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=whole),
                              variables)
        batch = {"text": jax.ShapeDtypeStruct((8, 65), jnp.int32, sharding=whole)}
        gradient = jax.jit(lambda params, batch: jax.grad(step_loss(objective, placed, batch))(params))
        return gradient.trace(placed["params"], batch).lower().compile()

    assert compiled("minimal").memory_analysis().host_temp_size_in_bytes == 0
    assert compiled("minimal_offloaded").memory_analysis().host_temp_size_in_bytes > 0


STEP_SEEDS = 8
"""The offloaded step's check-1 seeds, sized paired by `seeds_for` on armada."""


def stepping(trainer, initial, *, exact: bool = False):
    """One step of `trainer` from `initial` over a draw's token rows, the step compiled at its first call
    and every later call stepping a fresh copy of `initial`: the variables, the optimizer state and the
    EMA it leaves, on the host. `exact` steps the state widened to float64, under the caller's float64."""
    built = []

    @as_given
    def stepped(text):
        start = jax.tree.map(lambda leaf: leaf.astype(jnp.float64) if exact and jnp.issubdtype(
            leaf.dtype, jnp.floating) else jnp.copy(leaf), initial)
        state = jax.device_put(start, trainer.shardings(start))
        batch = {"text": jnp.asarray(text)}
        if not built:
            built.append(trainer.compile(state, batch))
        state, *_ = built[0](state, batch)
        return jax.device_get({name: getattr(state, name) for name in ("variables", "opt_state", "ema")})
    return stepped


def test_a_step_with_its_state_in_host_memory_trains_what_one_with_it_on_the_device_trains(monkeypatch):
    """Over drawn token rows, a step whose plan keeps the optimizer state and the EMA in host memory
    leaves the variables, the moments and the EMA on par with a step that keeps them on the device,
    both measured from the step in float64 (check 1): where the state sits is no arithmetic."""
    room = device_bound(monkeypatch)
    initial = offloading().initial_state()

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 64, (ROWS, SEQ + 1)).astype(np.int32),)

    offloads = offloading()
    host, device, truth = (stepping(offloads, initial), stepping(offloading(), initial),
                           stepping(offloading(), initial, exact=True))
    free(monkeypatch, room)
    host(*draw(0))
    assert offloads.plan.get("host") == OFFLOADED
    free(monkeypatch, LIMIT)
    device(*draw(0))
    with jax.enable_x64(new_val=True):
        truth(*draw(0))
    assert_as_exact(host, device, draw, "a step with its state in host memory", seeds=STEP_SEEDS,
                    truth_fn=truth)
