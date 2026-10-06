"""A run writes each variables collection but `params` once, by content (`dew.checkpoints.FROZEN_STORE`)."""

from __future__ import annotations

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from etils import epath

from dew.checkpoints import FROZEN_STORE, Checkpoints
from dew.data import Dataset, Loading
from dew.lora import Adapter, LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import FROZEN, Aux, Objective, Ratio
from dew.objectives.lm import LMObjective
from dew.training import Trainer

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def model():
    return CausalTransformer(vocab_size=16, emb_features=32, num_layers=2, num_heads=2,
                             mlp_features=64, max_seq_len=16, dtype="float32", attention_impl="xla")


def size(path) -> int:
    """The bytes of every file under `path`."""
    return sum(entry.stat().st_size for entry in Path(str(path)).rglob("*") if entry.is_file())


def adapted_run(directory, steps: int, *, keep: int = 3) -> tuple[Trainer, Checkpoints, object]:
    """A LoRA run of `steps` steps, each checkpointed, and the state it ends with."""
    base = model()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    objective = LMObjective(adapter.model, seq_len=8, ema_decay=None, variables=adapter.variables)
    rows = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(16)]
    checkpoints = Checkpoints(str(directory), keep=keep)
    trainer = Trainer(objective, optax.sgd(1.0), key=0, checkpoints=checkpoints)
    state = trainer.fit(Dataset.from_records(rows, batch=8, loading=LOADING), steps=steps,
                        checkpoint_every=1, log_every=100)
    checkpoints.wait()
    return trainer, checkpoints, state


def recorded(checkpoints: Checkpoints, step: int) -> dict:
    """The digest of each collection the step at `step` holds in the store, by tree."""
    return checkpoints._open().metadata(step).custom_metadata["frozen"]


def test_a_lora_run_writes_its_frozen_base_once_and_each_step_holds_its_factors(tmp_path):
    """Three checkpoints of a LoRA run record one digest for the base under
    `frozen`, written once to the store, and each step holds the factors
    alone of the variables: with Orbax's own files, under a fifth of the base."""
    _, checkpoints, state = adapted_run(tmp_path / "run", 3)
    root = epath.Path(checkpoints.directory)
    store = sorted(path.name for path in (root / FROZEN_STORE).iterdir())
    steps = [entry.name for entry in root.iterdir() if entry.name.isdigit()]
    assert sorted(steps) == ["1", "2", "3"]
    assert [recorded(checkpoints, int(step)) for step in sorted(steps)] == [
        {"variables": {FROZEN: store[0]}}] * 3
    assert len(store) == 1
    base = size(root / FROZEN_STORE / store[0])
    sizes = [size(root / step) for step in steps]
    assert all(held < base / 5 for held in sizes), (base, sizes)
    held = checkpoints.stored(3)["variables"]
    assert set(held) == {"params", FROZEN}
    written = checkpoints._open().metadata(3).item_metadata.tree["variables"]
    names = {path[-1].key for path, _ in jax.tree_util.tree_flatten_with_path(written)[0]}
    assert names == {"lora_A", "lora_B"}
    restored, _ = checkpoints.restore()
    for saved, back in zip(jax.tree.leaves(state.variables), jax.tree.leaves(restored["variables"]),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(back))


def test_a_resume_from_the_store_trains_on_bitwise_as_an_uninterrupted_run(tmp_path):
    """Two steps, a resume and a third land where three uninterrupted steps
    do, every leaf of the state bitwise."""
    _, _, whole = adapted_run(tmp_path / "whole", 3)
    adapted_run(tmp_path / "resumed", 2)
    _, _, resumed = adapted_run(tmp_path / "resumed", 3)
    for left, right in zip(jax.tree.leaves((whole.variables, whole.opt_state)),
                           jax.tree.leaves((resumed.variables, resumed.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_stored_collection_whose_content_changed_is_refused(tmp_path):
    """The store's copy of the base, rewritten with one weight moved, no
    longer has the digest the step recorded, and the restore says so."""
    _, checkpoints, _ = adapted_run(tmp_path / "run", 1)
    digest = recorded(checkpoints, 1)["variables"][FROZEN]
    path = epath.Path(checkpoints.directory) / FROZEN_STORE / digest
    checkpointer = ocp.PyTreeCheckpointer()
    tree = checkpointer.restore(path)
    leaf = jax.tree.leaves(tree)[0]
    moved = jax.tree.unflatten(jax.tree.structure(tree),
                               [np.asarray(leaf) + 1, *map(np.asarray, jax.tree.leaves(tree)[1:])])
    shutil.rmtree(str(path))
    checkpointer.save(path, moved)
    with pytest.raises(ValueError, match=f"not the {digest} the step recorded for its frozen collection"):
        checkpoints.restore()


class Counting(Objective):
    """A regression whose `count` collection moves every step, beside a frozen `table`."""

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros((3,))}, "count": {"n": jnp.zeros(())},
                "constants": {"table": jnp.arange(1000.0)}}

    def loss(self, variables, batch, step):
        error = jnp.square(batch["x"] @ variables["params"]["w"] - batch["y"])
        return (Ratio(jnp.sum(error), jnp.asarray(error.size, jnp.float32)),
                Aux(metrics={}, variables={"count": {"n": variables["count"]["n"] + 1}}))


def test_a_stored_collection_stays_while_a_kept_step_records_it_and_goes_with_the_last(tmp_path):
    """Keeping one step, the store holds the frozen table and the moving
    count's digest of each step still kept or in flight: a superseded count
    is collected, the table never is, and a copy of the run directory
    restores the same state elsewhere."""
    rows = [{"x": np.full(3, index, np.float32), "y": np.float32(index)} for index in range(8)]
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=1)
    trainer = Trainer(Counting(), optax.sgd(0.01), key=0, checkpoints=checkpoints)
    trainer.fit(Dataset.from_records(rows, batch=8, loading=LOADING), steps=4, checkpoint_every=1,
                log_every=100)
    checkpoints.wait()
    store = epath.Path(checkpoints.directory) / FROZEN_STORE
    kept = [kept.step for kept in checkpoints.kept()]
    referenced = {digest for step in kept for names in recorded(checkpoints, step).values()
                  for digest in names.values()}
    # The last save collected what no kept step nor itself recorded; the step it superseded may remain.
    assert referenced <= {path.name for path in store.iterdir()}
    assert len(list(store.iterdir())) <= len(referenced) + 1
    restored, _ = checkpoints.restore()
    assert float(restored["variables"]["count"]["n"]) == 4
    np.testing.assert_array_equal(restored["variables"]["constants"]["table"], np.arange(1000.0))

    shutil.copytree(str(tmp_path / "run"), str(tmp_path / "copied"))
    shutil.rmtree(str(tmp_path / "run"))
    copied, _ = Checkpoints(str(tmp_path / "copied")).restore()
    for left, right in zip(jax.tree.leaves(restored), jax.tree.leaves(copied), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_the_adapter_and_its_task_load_from_a_run_whose_base_is_stored_once(tmp_path):
    """`Adapter.from_run` reads the factors and the stored base back, and
    computes what the trained adapter computes."""
    trainer, _, state = adapted_run(tmp_path / "run", 2)
    rebuilt = Adapter.from_run(tmp_path / "run")
    tokens = jnp.arange(1, 9)[None, :]
    np.testing.assert_array_equal(np.asarray(rebuilt.model.apply(rebuilt.variables, tokens)),
                                  np.asarray(trainer.objective.model.apply(state.variables, tokens)))


class Wide(Objective):
    """A regression beside a float64 table no step moves."""

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros((3,), jnp.float32)},
                "constants": {"table": jnp.linspace(0.0, 1.0, 1000, dtype=jnp.float64)}}

    def loss(self, variables, batch, step):
        error = jnp.square(batch["x"] @ variables["params"]["w"] - batch["y"])
        return Ratio(jnp.sum(error), jnp.asarray(error.size, jnp.float32)), Aux(metrics={})


def test_a_collection_saved_with_x64_on_restores_with_it_off(tmp_path):
    """The digest is of the stored bytes: a float64 table written with x64
    on restores with x64 off, as host float64 bitwise and as float32 onto a
    template, without the integrity check taking its float32 view for a
    different collection."""
    rows = [{"x": np.full(3, index, np.float32), "y": np.float32(index)} for index in range(8)]
    with jax.enable_x64(new_val=True):
        checkpoints = Checkpoints(str(tmp_path / "run"))
        Trainer(Wide(), optax.sgd(0.01), key=0, checkpoints=checkpoints).fit(
            Dataset.from_records(rows, batch=8, loading=LOADING), steps=1, checkpoint_every=1, log_every=100)
        checkpoints.wait()
        table = np.asarray(Wide().init(None)["constants"]["table"])
    assert not jax.config.jax_enable_x64
    restored, _ = Checkpoints(str(tmp_path / "run")).restore()
    held = restored["variables"]["constants"]["table"]
    assert held.dtype == np.float64
    np.testing.assert_array_equal(held, table)
    variables = Checkpoints(str(tmp_path / "run")).variables(ema=False)
    np.testing.assert_array_equal(np.asarray(variables["constants"]["table"]), table.astype(np.float32))
