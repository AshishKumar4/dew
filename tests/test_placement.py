"""Every parameter of every model Dew builds is placed by the axes its module declares, as the layout's rules
place them (`dew.distributed.placement.Layout.shardings`), held to the layout's typed invariants rather than
to a recorded rendering of its placement (ledger 4423): each split names only mesh axes its declared axes'
rules name, divides its dimension and claims a mesh axis once; a leaf below the layout's floor stays whole,
and one a rule can split is split; a leaf no module declares splits its largest dimension fsdp divides; a
trainer's optimizer state and EMA sit as their parameters do. A step so placed trains on par with the same
step on one device, and a pipeline's load on a mesh holds the values and the placement of its training
state's (sharded against unsharded, on the same flows)."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding
from placement_cases import LAYOUT, MIN_SHARD, TRAINED, cases, declared

from dew.batch import unboxed
from dew.distributed.placement import UNSPLIT_AXES, Layout
from dew.mesh import FSDP_AXIS, SEQUENCE_AXIS, mesh_axes
from dew.objectives.lm import LMObjective
from dew.training import Trainer

RULES = Layout().axis_rules


def splits(name: str) -> list[tuple[str, ...]]:
    """The mesh axes each of `name`'s rules splits a dimension over, in precedence order, fsdp's with the
    sequence axis it splits beside (`state_spec`)."""
    found = []
    for rule, assignment in RULES:
        if rule == name and assignment is not None:
            axes = mesh_axes(assignment)
            found.append((*axes, SEQUENCE_AXIS) if FSDP_AXIS in axes else axes)
    return found


def assert_placed(name: str, tree, placement, mesh, table=None) -> None:
    """Each leaf of `tree`, as `placement` places it on `mesh`, holds the layout's invariants for the axes its
    module declares (`declared`, read through `table` where the run's table names them), and a tree with a
    leaf past the floor splits at least one."""
    axes = declared(tree, table)
    leaves = jax.tree_util.tree_leaves_with_path(unboxed(tree))
    sharded = 0
    for (path, leaf), sharding in zip(leaves, jax.tree.leaves(placement), strict=True):
        where = f"{name} {jax.tree_util.keystr(path)} {tuple(leaf.shape)}"
        assert isinstance(sharding, NamedSharding) and sharding.mesh == mesh, where
        assert sharding.memory_kind == "device", where
        spec = tuple(sharding.spec)
        assert len(spec) <= leaf.ndim, (where, spec)
        per_dim = [mesh_axes(entry) for entry in spec]
        used = [axis for entry in per_dim for axis in entry]
        assert len(used) == len(set(used)), (where, spec)
        assert set(used) <= set(mesh.axis_names) and not set(used) & set(UNSPLIT_AXES), (where, spec)
        for size, entry in zip(leaf.shape, per_dim, strict=False):
            assert size % math.prod(mesh.shape[axis] for axis in entry) == 0, (where, spec)
        elements = math.prod(leaf.shape)
        named = axes[jax.tree_util.keystr(path)]
        if elements < MIN_SHARD:
            assert not used, (where, spec)
            continue
        if named is None:
            # The shape's: fsdp on its largest dimension fsdp divides, where one does.
            assert set(used) <= {FSDP_AXIS, SEQUENCE_AXIS}, (where, spec)
            divisible = [dim for dim, size in enumerate(leaf.shape) if size % mesh.shape[FSDP_AXIS] == 0]
            if divisible:
                largest = max(leaf.shape[dim] for dim in divisible)
                (dim,) = [dim for dim, entry in enumerate(per_dim) if FSDP_AXIS in entry]
                assert leaf.shape[dim] == largest, (where, spec)
            else:
                assert not used, (where, spec)
        else:
            # A split over an axis other than fsdp's group is one its dimension's declared axis's rules
            # name; fsdp's the rules' or the floor's fallback for a leaf they leave whole.
            for dim, entry in enumerate(per_dim):
                ruled = {axis for split in splits(named[dim]) for axis in split} if named[dim] else set()
                assert set(entry) - {FSDP_AXIS, SEQUENCE_AXIS} <= ruled, (where, named, spec)
        sharded += bool(used)
    assert sharded or not any(math.prod(leaf.shape) >= MIN_SHARD for _, leaf in leaves), name


def placed_case(name: str):
    """`name`'s variables, shapes alone, and their placement over `LAYOUT` on the suite's first eight
    devices."""
    mesh = LAYOUT.build(jax.devices()[:8])
    module, args, kwargs = cases()[name]()
    variables = jax.eval_shape(functools.partial(module.init, **kwargs), jax.random.key(0), *args)
    return mesh, module, args, variables, Layout(min_shard=MIN_SHARD).shardings(mesh, variables)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", sorted(cases()))
def test_each_parameter_is_placed_by_its_declared_axes(name):
    """Every leaf of the case's variables, over a layout of fsdp, expert and tensor of two each, holds the
    layout's invariants for the axes its module declares; a trainer's optimizer state and EMA, where the
    case is trained, sit as their parameters do, and its step count whole."""
    mesh, module, args, variables, placement = placed_case(name)
    assert_placed(name, variables, placement, mesh)
    if name not in TRAINED:
        return
    trainer = Trainer(LMObjective(module, args[0].shape[1], ema_decay=0.9), optax.adamw(1e-3),
                      key=jax.random.key(0), mesh=LAYOUT, layout=Layout(min_shard=MIN_SHARD), tracker=None)
    trainer.device_mesh = mesh
    state = jax.eval_shape(trainer.initial_state)
    placed = trainer.shardings(state)
    assert_placed(name, state.variables, placed.variables, mesh, trainer.objective.parameter_axes)
    adam = placed.opt_state[0]
    for moment in (adam.mu, adam.nu):
        assert jax.tree.leaves(moment) == jax.tree.leaves(placed.variables["params"]), name
    assert jax.tree.leaves(placed.ema) == jax.tree.leaves(placed.variables), name
    assert tuple(adam.count.spec) == (), name


def _lm_loss(module, params, tokens):
    """The next-token cross-entropy of `module` over `tokens`, its gradient beside it."""
    def loss(params):
        logits = module.apply(params, tokens)
        logits = logits.logits if hasattr(logits, "logits") else logits
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), tokens[:, 1:, None], axis=-1)
        return -jnp.mean(picked)
    return jax.value_and_grad(loss)(params)


def parity_arms(name: str):
    """The loss and gradients of `name`'s next-token step with its parameters placed by the layout, on one
    device, and in float64, and the draw of one seed's parameters and tokens they take."""
    from tiny_models import build_model

    mesh, _, _, _, placement = placed_case(name)
    module, wide = build_model(name, "float32"), build_model(name, "float64")
    # Eight rows of 64 tokens, which every split of the layout divides: enough positions that each
    # gradient's error resolves, the norms' scales among them.
    shape = (8, 64)

    def draw(seed):
        tokens = jax.random.randint(jax.random.key(seed + 1), shape, 0, 32, jnp.int32)
        return jax.device_get(module.init(jax.random.key(seed), tokens)), np.asarray(tokens)

    replicated = NamedSharding(mesh, jax.P())
    sharded = jax.jit(functools.partial(_lm_loss, module), in_shardings=(placement, replicated))
    alone = jax.jit(functools.partial(_lm_loss, module))
    return sharded, alone, jax.jit(functools.partial(_lm_loss, wide)), draw


PARITY_SEEDS = {"causal_transformer": 16, "causal_transformer+moe": 16}
"""Each trained case's seeds, as `seeds_for` sized them on its paired arms."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", TRAINED)
def test_a_step_placed_by_the_layout_trains_on_par_with_one_device(name):
    """A step whose parameters the layout places over fsdp, expert and tensor computes the loss and every
    gradient as exactly as the same step on one device, each against the step in float64 (check 1): the
    placement moves only where the work runs."""
    from reference_error import assert_as_exact

    sharded, alone, truth, draw = parity_arms(name)
    assert_as_exact(sharded, alone, draw, f"{name} placed over the layout", seeds=PARITY_SEEDS[name],
                    truth_fn=truth)


@pytest.mark.mesh(devices=8)
def test_a_pipelines_state_and_load_are_placed_by_their_declared_axes_and_load_alike(tmp_path):
    """The tiny Flux pipeline's training state, its frozen CLIP and T5 encoders by their own tables beside
    its denoiser's (`DiffusionObjective.tower_axes`), holds the layout's invariants, and its load placed on a
    mesh by the same tables (`pipeline_axes`) holds the values its load on one device holds, each leaf
    placed as the training state places it (ledger 3453)."""
    from pathlib import Path

    from diffusion_support import source_objective
    from interop_support import extract_fixture

    from dew.interop.pretrained import Pretrained

    source = extract_fixture(Path(__file__).with_name("fixtures") / "flux_source.tar.xz", tmp_path)
    _, objective = source_objective(source)
    mesh = LAYOUT.build(jax.devices()[:8])
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), mesh=LAYOUT,
                      layout=Layout(min_shard=MIN_SHARD), tracker=None)
    trainer.device_mesh = mesh
    state = jax.eval_shape(trainer.initial_state)
    placed = trainer.shardings(state)
    assert_placed("flux", state.variables, placed.variables, mesh, objective.parameter_axes)
    on_mesh = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla", mesh=LAYOUT,
                              layout=Layout(min_shard=MIN_SHARD))
    alone = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    trained = {jax.tree_util.keystr(path): sharding
               for path, sharding in jax.tree_util.tree_leaves_with_path(placed.variables)}
    loaded = jax.tree_util.tree_leaves_with_path(on_mesh.variables)
    assert loaded and len(loaded) == len(jax.tree.leaves(alone.variables))
    for (path, leaf), whole in zip(loaded, jax.tree.leaves(alone.variables), strict=True):
        key = jax.tree_util.keystr(path)
        assert leaf.sharding.spec == trained[key].spec and leaf.sharding.memory_kind == "device", key
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(whole), err_msg=key)
