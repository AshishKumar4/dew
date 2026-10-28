"""A run's checkpoint read back bank by bank: the store holds the trained layers, and loading it
again runs the programs the first load compiled."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from steady_state import steady_state

from dew.batch import FROZEN
from dew.checkpoints import Checkpoints
from dew.data import Loading, Memory
from dew.inference.banks import CheckpointBanks, bank_sites, banked, spread
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import group_layers
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

DEVICE = Layout(min_shard=1, tolerance=1.)
HOST = Layout(host=("variables",), min_shard=1, tolerance=1.)
ROWS = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(8)]


def decoder(**overrides):
    fields = {"vocab_size": 16, "emb_features": 8, "num_layers": 4, "num_heads": 2, "mlp_features": 16,
              "max_seq_len": 8, "scan_layers": True, "dtype": "float32"}
    return CausalTransformer(**{**fields, **overrides})


def trained(directory, objective, layout):
    """One checkpointed step of `objective` under `layout`; returns the state it ends with."""
    checkpoints = Checkpoints(str(directory))
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5), layout=layout,
                      checkpoints=checkpoints)
    data = Memory(ROWS).shuffle().repeat().batch(jax.device_count()).read("memory").loading(
        Loading(workers=0))
    state = trainer.fit(data, steps=1, checkpoint_every=1, log_every=100)
    checkpoints.wait()
    return state


def test_a_trees_banks_spread_back_into_its_rows():
    """`banked` moves each scanned run's leaves its rows all hold into one bank under the run's name,
    a leaf only some rows hold kept per layer, and tells `release` each row leaf it replaced; `spread`
    reads the tree back row for row: the one pair a host layout's frozen collection, its placement and
    a checkpoint's template take."""
    model = decoder(bank_layers=2)
    variables = jax.device_get(model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))["params"]
    tree = {name: dict(value) if name.startswith("layers_") else value for name, value in variables.items()}
    # A leaf one row holds alone, as a frozen collection under a partial freeze does.
    tree["layers_1"] = {**tree["layers_1"], "extra": np.ones(3, np.float32)}
    released = []
    packed = banked(tree, bank_sites(model, {"params": variables}), lambda rows, path: np.stack(rows),
                    lambda namespace, index, keys, bank: released.append((index, keys)))
    groups = {name: layers for name in packed
              if (layers := group_layers(name)) is not None and len(layers) > 1}
    assert groups and packed["layers_1"] == {"extra": tree["layers_1"]["extra"]}
    assert {index for index, _ in released} == {index for layers in groups.values() for index in layers}
    back = spread(packed)
    assert jax.tree.structure(back) == jax.tree.structure(tree)
    for actual, expected in zip(jax.tree.leaves(back), jax.tree.leaves(tree), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_a_run_places_bank_by_bank_and_again_on_the_programs_it_compiled(tmp_path):
    """A stack of two runs of two layers each: each bank the source places is its layers of the
    trained state stacked, and a second load stacks them with the program the first compiled."""
    model = decoder(bank_layers=2)
    state = trained(tmp_path / "run", LMObjective(model, 8, head_chunks=1), DEVICE)
    mesh = MeshSpec().build()
    first = CheckpointBanks(str(tmp_path / "run")).place(model, mesh=mesh, layout=DEVICE)
    with steady_state(allow=("host_to_device",)):
        second = CheckpointBanks(str(tmp_path / "run")).place(model, mesh=mesh, layout=DEVICE)
    params = state.variables["params"]
    assert sorted(name for name in first["params"] if group_layers(name)) == ["layers_0_1", "layers_2_3"]
    for name, layers in (("layers_0_1", (0, 1)), ("layers_2_3", (2, 3))):
        stacked = jax.tree.map(lambda *rows: np.stack(rows), *(params[f"layers_{index}"] for index in layers))
        for want, got, again in zip(jax.tree.leaves(stacked), jax.tree.leaves(first["params"][name]),
                                    jax.tree.leaves(second["params"][name]), strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)
            np.testing.assert_array_equal(np.asarray(again), want)


def test_a_cpu_owned_run_reads_its_frozen_banks_into_rows_again_on_the_programs_it_compiled(tmp_path):
    """A LoRA run whose frozen base the CPU owns stores that base banked. Reading the run's
    variables cuts each bank into the rows of its layers, those of the trained state, and a second
    read cuts them with the program the first compiled."""
    base = decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    objective = LMObjective(adapter.model, 8, head_chunks=1, ema_decay=None,
                            variables=jax.tree.map(np.asarray, adapter.variables))
    state = trained(tmp_path / "run", objective, HOST)
    assert any(len(group_layers(name) or ()) > 1 for name in state.variables[FROZEN])
    expected = spread(jax.device_get(state.variables[FROZEN]))
    first = Checkpoints(str(tmp_path / "run")).variables(ema=False)
    with steady_state(allow=("host_to_device",)):
        second = Checkpoints(str(tmp_path / "run")).variables(ema=False)
    assert sorted(first[FROZEN]) == sorted(expected)
    assert {f"layers_{index}" for index in range(4)} <= set(expected)
    for want, got, again in zip(jax.tree.leaves(expected), jax.tree.leaves(first[FROZEN]),
                                jax.tree.leaves(second[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)
        np.testing.assert_array_equal(np.asarray(again), want)
