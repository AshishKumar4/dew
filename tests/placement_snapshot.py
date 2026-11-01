"""Where Dew places every parameter of every model it builds (T69: rows 1153 and 3453): each case's leaves, by
path, with their shape, the logical axes their module declares and the partition spec the layout gives them
over `LAYOUT`, and a trainer's optimizer state and EMA beside its parameters. tools/placement_snapshot.py
writes the snapshot at a main (tests/fixtures/placement/snapshot.tsv, one line a leaf, so a change to how
axes are declared shows every placement it moves as its own line), and tests/test_placement_snapshot.py holds
each case to it.

The cases are the catalogue the precision and protocol tests build from (`tiny_models`: every registered
architecture and the decoder's mixtures and fixtures' mixers), a decoder over each registered mixer
(`test_mixers`'s fields), and declaring modules in forms no model there reaches: a flat and a headed
`NormalAttention`, and `GatedMLP` packed, ungated and biased."""

import functools

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from test_mixers import KIND_FIELDS, tiny
from tiny_models import DECODERS, build_model, tiny_inputs

from dew.distributed.placement import Layout, MeshSpec
from dew.nn.attention_blocks import NormalAttention
from dew.nn.mlp import GatedMLP
from dew.nn.sharding import boxed, boxed_axes, declarations, declared_axes, ruled_boxes
from dew.objectives.lm import LMObjective
from dew.registry import mixers, models
from dew.training import Trainer

LAYOUT = MeshSpec(fsdp=2, expert=2, tensor=2)
"""The layout every case is placed over, on eight devices: each parameter axis a declaration names has a
mesh axis of two to land on."""

MIN_SHARD = 16
"""The fewest elements a leaf shards at, so a tiny model's declarations decide its placement rather than
its size."""

COLUMNS = ("case", "path", "shape", "axes", "spec", "memory")
"""The snapshot's columns, tab-separated, one line a leaf."""


def _components() -> dict:
    """The declaring modules no catalogued model reaches in each of their forms, each with its inputs."""
    tokens = jnp.zeros((2, 5, 32))
    return {
        "NormalAttention+flat": lambda: (NormalAttention(32, heads=2, dim_head=8, flat=True,
                                                         attention_impl="reference"), (tokens,), {}),
        "NormalAttention+headed": lambda: (
            NormalAttention(32, heads=2, dim_head=8, attention_impl="reference"), (tokens,), {}),
        "GatedMLP+packed": lambda: (GatedMLP(16, 8, activation="geglu", packed=True,
                                             packed_order=("up_proj", "gate_proj")),
                                    (jnp.zeros((1, 8)),), {}),
        "GatedMLP+ungated": lambda: (GatedMLP(16, 8, activation="gelu"), (jnp.zeros((1, 8)),), {}),
        "GatedMLP+bias": lambda: (GatedMLP(16, 8, use_bias=True), (jnp.zeros((1, 8)),), {}),
    }


def cases() -> dict:
    """Each case's name, and how to build its module and the inputs it initializes on."""
    built = {}
    for architecture in [*sorted(models), *sorted(DECODERS)]:
        def model(architecture=architecture):
            inputs = tiny_inputs(architecture, jax.random.key(0))
            args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
            return build_model(architecture, "float32"), args, kwargs
        built[architecture] = model
    for kind in sorted(mixers.paths):
        def mixer(kind=kind):
            fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
            return (tiny(mixer=mixers.build(kind, **fields), max_seq_len=128, **overrides),
                    (jnp.zeros((1, 8), jnp.int32),), {})
        built[f"mixer:{kind}"] = mixer
    return {**built, **_components()}


TRAINED = ("causal_transformer", "causal_transformer+moe")
"""The cases whose trainer's state is recorded too: its Adam moments and EMA beside its parameters."""


def _axes(variables) -> dict[str, tuple]:
    """Each leaf's declared logical axes, as the layout reads them, by its key path."""
    rules = Layout().axis_rules
    declared = declarations(variables)
    with boxed(ruled_boxes(boxed_axes(variables), rules)):
        return {jax.tree_util.keystr(path): declared_axes(path, leaf.ndim, declared)
                for path, leaf in jax.tree_util.tree_leaves_with_path(nn.unbox(variables))}


def _lines(name: str, tree, placement) -> list[str]:
    """`tree`'s leaves placed as `placement` places them, a line each (`COLUMNS`)."""
    axes = _axes(tree)
    lines = []
    for (path, leaf), sharding in zip(jax.tree_util.tree_leaves_with_path(nn.unbox(tree)),
                                      jax.tree.leaves(placement), strict=True):
        named = axes[jax.tree_util.keystr(path)]
        lines.append("\t".join((name, jax.tree_util.keystr(path), "x".join(map(str, leaf.shape)) or "-",
                                "-" if named is None else ",".join(axis or "." for axis in named) or "()",
                                str(tuple(sharding.spec)), str(sharding.memory_kind))))
    return lines


def snapshot(name: str, build) -> list[str]:
    """`name`'s lines: its variables placed over `LAYOUT`, and, for a `TRAINED` case, its trainer's whole
    state."""
    mesh = LAYOUT.build(jax.devices()[:8])
    module, args, kwargs = build()
    variables = jax.eval_shape(functools.partial(module.init, **kwargs), jax.random.key(0), *args)
    lines = _lines(name, variables, Layout(min_shard=MIN_SHARD).shardings(mesh, variables))
    if name in TRAINED:
        trainer = Trainer(LMObjective(module, args[0].shape[1], ema_decay=0.9), optax.adamw(1e-3),
                          key=jax.random.key(0), mesh=LAYOUT, layout=Layout(min_shard=MIN_SHARD),
                          tracker=None)
        trainer.device_mesh = mesh
        state = jax.eval_shape(trainer.initial_state)
        placed = trainer.shardings(state)
        lines += _lines(f"{name}:state", {"opt_state": state.opt_state, "ema": state.ema},
                        {"opt_state": placed.opt_state, "ema": placed.ema})
    return lines
