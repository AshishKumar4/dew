"""The plan's host offload (`dew.training.planning.FitSearch`): where a step does not fit its devices, the
plan keeps the optimizer state and the EMA in its host's pinned memory before it recomputes anything,
and a remat rung that keeps its residuals there stands before the next that recomputes them, each only
where the bytes it moves, over the host link, take no longer than the step's FLOPs at the device's peak
(`hides`).

A CPU device has the pinned host memory and no accounting of it apart from its own (tests/
test_host_placement.py), so the plan's choice is held here on budgets the devices report
(`test_plan.free`) and links the run states (`Trainer(links=...)`); what an offloading remat rung keeps
in host memory is read off a step compiled for a compile-only H100, whose compiler accounts for it."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import compile_only_gpus
from reference_error import as_given, assert_as_exact
from test_host_placement import memory_kinds
from test_layout_validation import step_loss
from test_plan import LIMIT, ROWS, SEQ, free, needs, one_device, trained

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.sharding import Link
from dew.objectives.lm import LMObjective
from dew.training import Trainer

HOST = "host"
"""The host link's name among a run's links (`dew.training.distributed.HOST_LINK`)."""
FAST = Link(bytes_per_second=1e18, flops_per_second=1e9)
"""A host link over which any step's state moves in less time than one FLOP takes at the stated peak."""
SLOW = Link(bytes_per_second=1.0, flops_per_second=1e18)
"""A host link over which no step's state moves in the time its FLOPs take."""
OFFLOADED = ["opt_state", "ema"]


def offloading(split=None, link=FAST, **fields):
    """A one-layer decoder trained by Adam with an EMA, both of which the plan may keep in host memory,
    over a host `link` the run states."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ, **fields)
    return one_device(Trainer(LMObjective(model, seq_len=SEQ, ema_decay=0.9), optax.adam(1e-3),
                              key=jax.random.key(0), split=split, links={HOST: link}))


def offload_probes(trainer):
    """The compiles the plan's search made, as (split, remat, host, fits, hidden)."""
    return [(probe["split"], probe["remat"], probe.get("host"), probe["fits"], probe.get("hidden"))
            for probe in trainer.plan["probes"]]


def tight(monkeypatch) -> int:
    """Room for no split of the step with its optimizer state and EMA on the device, the most
    microbatches taking the least, and for the most microbatches recomputing at 'minimal'."""
    least, recomputing = needs(monkeypatch, offloading(split=ROWS)), needs(
        monkeypatch, offloading(split=ROWS, remat="minimal"))
    assert recomputing < least, (recomputing, least)
    return least - 1


def test_a_step_that_fits_only_without_its_state_on_the_device_keeps_it_in_host_memory(monkeypatch):
    """With room for no split while the optimizer state and the EMA sit on the device, the plan keeps
    them in host memory before it recomputes anything, over a host link that moves them in less time
    than the step's FLOPs take, and the run trains from there with them in pinned host memory. Its
    checkpoint's rung says so, and a trainer that takes that rung places them there."""
    free(monkeypatch, tight(monkeypatch))
    trainer = offloading()
    state = trained(trainer)
    assert (trainer.plan.get("host"), trainer.plan["remat"]) == (OFFLOADED, None)
    searched = offload_probes(trainer)
    assert all(not fits for _, remat, host, fits, _ in searched if host == [])
    assert (trainer.plan["split"], None, OFFLOADED, True, True) in searched
    assert memory_kinds(state.opt_state) == memory_kinds(state.ema) == {"pinned_host"}
    assert memory_kinds(state.variables) == {"device"}

    resumed = offloading()
    resumed._resume_rung(trainer._checkpoint_rung())
    placed, _, _ = resumed.place()
    assert resumed.layout.host == tuple(OFFLOADED)
    assert memory_kinds(placed.opt_state) == memory_kinds(placed.ema) == {"pinned_host"}


def test_a_state_offload_the_host_link_cannot_hide_is_taken_back_and_the_step_recomputes(monkeypatch):
    """Over a host link too slow to move the optimizer state and the EMA in the time the step's FLOPs
    take, the plan compiles their offload, finds it exposed, takes it back and climbs the ladder: the
    run keeps its state on the device and recomputes."""
    free(monkeypatch, tight(monkeypatch))
    trainer = offloading(link=SLOW)
    state = trained(trainer)
    assert trainer.plan.get("host") == []
    assert (trainer.plan["remat"], trainer.plan["head_tile"]) != (None, None)
    assert any(host == OFFLOADED and hidden is False for _, _, host, _, hidden in offload_probes(trainer))
    assert memory_kinds(state.opt_state) == {"device"}


def test_an_offloading_remat_rung_is_priced_by_the_residuals_it_keeps_in_host_memory():
    """A decoder climbs from 'minimal' to 'minimal_offloaded', which keeps 'minimal''s residuals in
    pinned host memory. Compiled for a compile-only H100, its step keeps them in host temporaries where
    'minimal''s keeps none, and what it moves, written in the forward and read back in the backward,
    hides behind its FLOPs over a fast host link and not over a slow one (`hides`)."""
    model = CausalTransformer(vocab_size=64, emb_features=64, num_layers=2, num_heads=2, mlp_features=128,
                              max_seq_len=64, remat="minimal")
    assert model.recompute_more().remat == REMAT_POLICIES["minimal_offloaded"]
    from dew.training.planning import hides, offloaded

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

    kept, offloading_step = compiled("minimal"), compiled("minimal_offloaded")
    assert kept.memory_analysis().host_temp_size_in_bytes == 0
    temporaries = offloading_step.memory_analysis().host_temp_size_in_bytes
    assert temporaries > 0
    moved = offloaded(None, (), temporaries)
    assert moved == 2 * temporaries
    assert hides(moved, offloading_step, FAST) and not hides(moved, offloading_step, SLOW)


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
    room = tight(monkeypatch)
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
