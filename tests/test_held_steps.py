"""What a run holds of its compiled steps, bounded by their device code (`dew.training.memory.CodeBounded`,
row 12385): the executables its searches hold for each other and its refits, and the runners its fit holds by
batch signature; and the limit a GPU compile stops past, which gives back the state a placed step reuses
(`compile_limit`, row 12465)."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Regression

from dew.distributed.placement import Layout, MeshSpec
from dew.training import FitPolicy, Trainer


def test_a_runs_held_steps_let_the_one_used_least_recently_go_first_past_their_budget():
    """What a run holds of its compiled steps (`CodeBounded`, row 12385: the executables its searches hold,
    the runners its fit holds) is bounded by their device code: holding one past the budget lets the least
    recently used go first, a step read since kept before one held earlier and not read again, each let go
    handed to `released`; and one alone past the budget is still held."""
    from dew.training.memory import CodeBounded

    released = []
    held = CodeBounded(budget=30, released=released.append)
    held.hold("small", "s", 10)
    held.hold("middle", "m", 10)
    assert held.get("small") == "s"
    held.hold("large", "l", 15)
    assert list(held) == ["small", "large"] and held.bytes == 25 and released == ["middle"]
    held.hold("larger", "L", 40)
    assert list(held) == ["larger"] and released == ["middle", "small", "large"]


def test_a_runs_held_steps_stay_whole_under_threads_holding_and_reading_at_once():
    """A later phase's search off the loop holds the run's executables beside the loop's own
    (`dew.training.ahead.LaterPlans`): threads holding and reading one table at once, each hold past its
    budget, leave it holding what its byte count says, every key let go once (review of row 12385: two
    threads let the same oldest go, a KeyError and a count off by its bytes)."""
    import sys
    import threading

    from dew.training.memory import CodeBounded

    released = []
    held = CodeBounded(budget=64, released=released.append)
    switching = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def holding(name: str):
            for index in range(2000):
                held.hold((name, index), index, 8)
                held.get((name, index - 3))
        threads = [threading.Thread(target=holding, args=(name,)) for name in "abcd"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(switching)
    assert held.bytes == 8 * len(held) <= held.budget, (held.bytes, len(held))
    assert len(released) == len(set(released)) == 4 * 2000 - len(held)


def _shaped(rows: tuple[int, ...]):
    """The affine run's stream, its batches `rows[step % len(rows)]` rows each."""
    from affine_run import Replaying

    from dew.data.dataset import Loader

    def batch_at(step: int) -> dict:
        x = np.random.default_rng(step).normal(size=(rows[step % len(rows)], FEATURES)).astype(np.float32)
        return {"x": x, "y": 2 * x[:, :2]}

    return Loader(train=lambda partition: Replaying(batch_at), val=None, batch=max(rows), records=None)


@pytest.mark.gpu(reason="device code's bytes, which a GPU's executables state and the CPU's do not")
@pytest.mark.compiles_at_run(reason="a fit's steps compiled as it meets each shape, no check-1 site")
def test_a_fit_meeting_shape_after_shape_holds_no_step_past_its_budget(monkeypatch):
    """A fit whose run holds no more device code than its latest step's (`HELD_CODE_BYTES`) meets three
    batch shapes and then each again: each shape it meets lets the steps of the one before go, the runner
    its fit held (`FitRun.compiled`) and the executable its compiler held (`FitSearch.executables`), so at
    no step does the backend hold more than the step it runs and the one before it, which the loop ran
    last; and it trains as a run that held every step, which holds all three at once (review of row
    12385: the fit's runners held every executable the compiler let go until the fit ended)."""
    from dew.training import memory
    from dew.training.observe import Observe

    def trained(budget: int | None):
        with monkeypatch.context() as patched:
            if budget is not None:
                patched.setattr(memory, "HELD_CODE_BYTES", budget)
            trainer = Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0))
            # Each executable this run's steps ran (`Trainer.executable`), and how many of them live at each
            # step, garbage collected.
            ran: dict[int, weakref.ref] = {}

            def live(state) -> int:
                ran.setdefault(id(trainer.executable), weakref.ref(trainer.executable))
                gc.collect()
                return sum(1 for held in ran.values() if held() is not None)

            observed = Observe(1, live)
            state = trainer.fit(_shaped((16, 8, 32)), steps=6, log_every=100, observe=observed)
        return [count for _, count in observed.values], jax.device_get(state.variables)

    bounded, held = trained(1)
    whole, kept = trained(None)
    assert max(bounded) <= 2 < max(whole), (bounded, whole)
    jax.tree.map(np.testing.assert_array_equal, held, kept)


@pytest.mark.gpu(reason="an allocator's live reading, which only an attached GPU states")
@pytest.mark.compiles_at_run(reason="its compile's limit read off the GPU's live room, no check-1 site")
def test_a_placed_steps_compile_limit_gives_back_the_state_it_reuses():
    """A placed run's step, compiled cold where its GPU has free memory short of its state and its
    temporaries together but not of its temporaries alone, compiles whole within its limit, is admitted at
    its first compile and runs: the limit gives back the bytes the placed state holds (`compile_limit`,
    `Candidate.resident`), which the step reuses where they lie (row 12465). Read off the shapes the step was
    lowered over, the state held nothing, and XLA stopped the step at its buffer assignment past a limit it
    fit within (review of row 12465). Its figures are read off the same step compiled ahead for a GPU of its
    generation, no binary of the step itself made before."""
    from token_support import token_rows

    from dew._cache import uncached
    from dew.distributed.batches import shard_batch
    from dew.nn.kernels.generation import compile_only_gpus, device_generation
    from dew.objectives.lm import LMObjective
    from dew.optim import Adam, OptimConfig
    from dew.registry import models
    from dew.training.ahead import compile_ahead
    from dew.training.memory import step_room

    device = jax.local_devices()[0]
    model = models.build("causal_transformer", vocab_size=32768, emb_features=1024, num_layers=1, num_heads=8,
                         num_kv_heads=2, head_dim=64, mlp_features=1024, max_seq_len=8, dtype="float32")
    rows = token_rows(32768, 1, 9)

    def built(mesh) -> Trainer:
        optimizer = OptimConfig(optimizer=Adam(), learning_rate=1e-3).build(100)
        return Trainer(LMObjective(model, 8, ema_decay=0.9), optimizer, key=jax.random.key(0), mesh=mesh,
                       layout=Layout(min_shard=2 ** 30), fit=FitPolicy(split=1, remat="none"))

    ahead = compile_ahead(built(MeshSpec().build(compile_only_gpus(device_generation(), 1, 2 ** 36))),
                          rows)
    argument, beside = ahead.argument_bytes, ahead.output_bytes - ahead.alias_bytes + (1 << 20)
    # Room for its temporaries twice over beside what it places, and half its state: short of the state
    # and temporaries together, so a limit that gives the state back nothing stops it.
    target = beside + 2 * ahead.temp_bytes + argument // 2
    assert target + (1 << 27) < argument + beside + ahead.temp_bytes, ahead
    # Compiled cold, so its limit stops a compile that misses (`uncached`).
    with uncached():
        trainer = built(MeshSpec())
        state, _, _ = trainer.place()
        batch = shard_batch(trainer.device_mesh, rows)
        room = step_room(device)
        assert room is not None and room > target, (room, target)
        filler = [jnp.zeros((1 << 24,), jnp.float32) for _ in range((room - target) >> 26)]
        filled = step_room(device)
        assert filled is not None and target <= filled < target + (1 << 27), (target, filled)
        step = trainer.compile(state, batch)
        probes = trainer.plan["probes"]
        assert probes[0]["fits"] and trainer.plan["split"] == 1, probes
        jax.block_until_ready(step(state, batch)[1])
        del filler
