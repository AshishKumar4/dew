"""The plan's activation offload (`dew.training.planning.FitSearch`): a remat rung that keeps its residuals in
pinned host memory (`minimal_offloaded`) stands before the next that recomputes them, and a plan at it meets
the first plan above it that keeps them on the device. A run that times its plans keeps the offload only
where it runs clearly faster, and one that times none keeps the device-resident plan wherever one fits;
where none fits, the offload stands.

A CPU's compiler keeps no host temporaries apart from its own, so the search's choice is held here on the
memory a test's analysis reports at each rung (`fitting`), with host temporaries at the offloading rung as
an A100's and a compile-only H100's compiler report them
(`test_an_offloading_remat_rung_keeps_its_residuals_in_host_temporaries`); the A100 runs it for real (node
receipts TL-a100-offload-1338-*)."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kernel_compile import compile_only_gpus
from reference_error import as_given, assert_as_exact
from test_layout_validation import step_loss
from test_plan import BATCH, LIMIT, ROWS, SEQ, free, one_device

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.lm import LMObjective
from dew.training import Trainer

OFFLOADED = "minimal_offloaded"


def decoder(remat=None, split=1, dtype=jnp.float32):
    """A one-layer decoder trained by Adam, its batch read in `split` microbatches."""
    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=SEQ, remat=remat, dtype=dtype)
    return one_device(Trainer(LMObjective(model, seq_len=SEQ), optax.adam(1e-3), key=jax.random.key(0),
                              split=split))


def fitting(monkeypatch, trainer, rungs):
    """Every device reports all of LIMIT free, and XLA's memory analysis reports, as a GPU's compiler does,
    temporaries no device holds at the remat rungs not named in `rungs`, and host temporaries at the
    offloading rung: the search's choice, not the CPU's memory, is what the test holds."""
    original = jax.stages.Compiled.memory_analysis

    def remat():
        model = trainer.objective.model
        return next((name for name, policy in REMAT_POLICIES.items() if policy == model.remat), None)

    def analysis(compiled):
        stats = original(compiled)
        reported = {name: getattr(stats, name) for name in dir(stats) if name.endswith("_in_bytes")}
        if remat() not in rungs:
            reported["temp_size_in_bytes"] += 2 * LIMIT
        if remat() == OFFLOADED:
            reported["host_temp_size_in_bytes"] += 1
        return types.SimpleNamespace(**reported)

    free(monkeypatch, LIMIT)
    monkeypatch.setattr(jax.stages.Compiled, "memory_analysis", analysis)


@pytest.mark.parametrize("seconds, remat", [
    ([[0.2, 0.21], [0.1, 0.12]], OFFLOADED), ([[0.1, 0.12], [0.2, 0.21]], "save_dot_except_mlp"),
    ([[0.2, 0.3], [0.1, 0.25]], "save_dot_except_mlp"), (None, "save_dot_except_mlp")],
    ids=["offload-faster", "recompute-faster", "within-spread", "untimed"])
def test_a_plan_whose_residuals_go_to_host_memory_is_timed_against_the_first_that_keeps_them(
        monkeypatch, seconds, remat):
    """Where no rung below 'minimal_offloaded' fits and the rungs above it do, the search compiles the
    offloading plan and then the first above it that keeps its residuals on the device, and the run times
    them, the device-resident plan first: the offload is kept only where its every sample beats the
    other's (`faster`). A compile that times nothing keeps the device-resident plan without timing."""
    trainer = decoder()
    fitting(monkeypatch, trainer, {OFFLOADED, "save_dot_except_mlp", "full"})
    state, _, _ = trainer.place()
    timed = []
    step = trainer.compile(state, BATCH, None if seconds is None else (
        lambda steps: (timed.append(len(steps)), seconds)[1]))
    assert OFFLOADED in {probe["remat"] for probe in trainer.plan["probes"]}
    assert timed == ([] if seconds is None else [2])
    assert trainer.plan["remat"] == remat
    if seconds is not None:
        assert [plan["remat"] for plan in trainer.plan["timed"]] == ["save_dot_except_mlp", OFFLOADED]
    state, loss, *_ = step(state, BATCH)
    assert np.isfinite(float(loss))


def test_a_plan_whose_residuals_go_to_host_memory_stands_where_no_plan_keeps_them_on_the_device(monkeypatch):
    """Where 'minimal_offloaded' alone fits, the search finds no plan above it that keeps the residuals on
    the device, the head whole, tiled or tiled with its gradient in the backward, and keeps the offload
    untimed."""
    trainer = decoder()
    fitting(monkeypatch, trainer, {OFFLOADED})
    state, _, _ = trainer.place()
    trainer.compile(state, BATCH, lambda steps: pytest.fail(f"{len(steps)} plans timed, one fits"))
    assert (trainer.plan["remat"], trainer.plan["timed"]) == (OFFLOADED, None)
    asked = {probe["remat"] for probe in trainer.plan["probes"]}
    assert {"save_dot_except_mlp", "full"} <= asked


def test_an_offloading_remat_rung_keeps_its_residuals_in_host_temporaries():
    """A decoder climbs from 'minimal' to 'minimal_offloaded', which keeps 'minimal''s residuals in
    pinned host memory. Compiled for a compile-only H100, its step keeps host temporaries where
    'minimal''s keeps none: what the device-resident search reads to refuse the rung
    (`dew.training.planning.keeps_host_temporaries`)."""
    model = CausalTransformer(vocab_size=64, emb_features=64, num_layers=2, num_heads=2, mlp_features=128,
                              max_seq_len=64, remat="minimal")
    assert model.recompute_more().remat == REMAT_POLICIES[OFFLOADED]
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
    assert compiled(OFFLOADED).memory_analysis().host_temp_size_in_bytes > 0


STEP_SEEDS = 8
"""The offloading rung's step's check-1 seeds, sized paired by `seeds_for` on armada (run-tests
rows-memory-probe11-85ab1203f); the site runs on no GPU tier."""


def stepping(trainer, initial, *, exact: bool = False):
    """One step of `trainer` from `initial` over a draw's token rows, the step compiled at its first call
    and every later call stepping a fresh copy of `initial`: the variables and the optimizer state it
    leaves, on the host. `exact` steps the state widened to float64, under the caller's float64."""
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
        return jax.device_get({name: getattr(state, name) for name in ("variables", "opt_state")})
    return stepped


def test_a_step_with_its_residuals_in_host_memory_trains_what_one_keeping_them_on_the_device_trains():
    """Over drawn token rows, a step at 'minimal_offloaded', its residuals in host memory, leaves the
    variables and the moments on par with a step that saves every residual on the device, both measured
    from the step in float64 (check 1): where a residual waits is no arithmetic."""
    initial = decoder().initial_state()

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 64, (ROWS, SEQ + 1)).astype(np.int32),)

    offloading = decoder(remat=OFFLOADED)
    host, device = stepping(offloading, initial), stepping(decoder(), initial)
    truth = stepping(decoder(dtype=jnp.float64), initial, exact=True)
    host(*draw(0))
    assert offloading.plan["remat"] == OFFLOADED
    device(*draw(0))
    with jax.enable_x64(new_val=True):
        truth(*draw(0))
    assert_as_exact(host, device, draw, "a step with its residuals in host memory", seeds=STEP_SEEDS,
                    truth_fn=truth)
