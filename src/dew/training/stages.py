"""A stage-local train state: each run of layers a pipeline over the stage axis repeats, held stacked by
stage on its stage's own devices, its optimizer state and average with it.

A pipeline over the stage axis runs stage s on stage s's devices, the run's layers stacked by stage
(`dew.nn.backbones.decoder_stack.StackView`). A state that keeps them as `layers_N` keeps every layer
on every device, since one program cannot take a leaf that lives on some of its devices alone, and its
step restacks them. So the trainer keeps the run in the pipeline's own form, `stages/{name}`,
`[stages, ...]` with the leading dimension on the stage axis (`StackView.stage_form`), as MaxText's
pipeline keeps its stacked weights. The optimizer's moments, the EMA, the accumulated gradient and the
narrow copies mirror the parameters, so they follow, and a device holds 1/S of the run. The pipeline
reads that form as it is (`StackView.staged`).

Invariants:
- Where it lies. The objective says where each run lies and how it stacks (`Objective.stage_sites`,
  its models' `StagedStacks`): a decoder's run its stages repeat (`split_stages`), a stack of like
  blocks' (`block_sites`). Nothing else changes form: embeddings, a head and the layers before and
  after the run stay as the layout lays them.
- One form in memory, one on disk. A state the trainer builds, restores or is given holds every run
  stacked by stage, and a checkpoint holds every layer as `layers_N`, the tree every other reader (an
  export, a loader, an average, a resume onto another stage count or none) reads. The conversion is
  at the checkpoint's seam alone (`StageLocal.saved`, `StageLocal.restored`), through each stage's
  own devices: a stage's rows are arrays there, and no layer is gathered onto another stage's.
- Per layer, exactly. The optimizer steps the stacked leaves, so a rule that reads a leaf as one layer
  reads it as the stored layers it stacks (`dew.nn.sharding.stacked_layers`): Dew's
  weight-decay mask, Muon's matrix axes, a param group's patterns, int8 moments' size threshold,
  LAMB's trust ratio, LaProp's adaptive clip and QK-Clip's per-layer maxima each give what they give
  unstacked. An optax transformation given as it is runs over the stacked leaf, so the run keeps its runs
  stage-local where its update over the stacked leaves computes each layer as its update over the stored
  layers does, row by row the same expression of that layer's entries, a sum over every layer the sum of
  its layers' (`whole_reading`), and keeps every layer whole on every stage otherwise, the replicated
  layout's memory, which trains it exactly: a trust ratio over a leaf, or a build that decides by its
  leaves' shapes (rows 3934, 3995).

Rejected: the stage axis as one more axis the state splits over, as fsdp splits it, which keeps each
leaf one layer but moves the run's parameters to their stage and its gradients back every step, the
traffic a pipeline exists to avoid; and a checkpoint in the stacked form, which every reader of
`layers_N` would have to learn, for every stage count.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import math
import secrets
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, overload

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.algebra.terms import parameter_key, parameters_key
from dew.batch import unboxed
from dew.mesh import STAGE_AXIS, LayoutRefused
from dew.nn.kernels.options import owned_program, program
from dew.nn.kernels.templates.layout import ENTRYWISE, entrywise, jax_name
from dew.nn.sharding import abstract, held_by_stage

if TYPE_CHECKING:
    from dew.checkpoints import Checkpoints
    from dew.data.dataset import DataPartition
    from dew.distributed.placement import Layout
    from dew.nn.backbones.decoder_stack import DecoderBank, RowCut, RowStack
    from dew.nn.protocols import ProjectionGroup
    from dew.nn.sharding import Schedule
    from dew.state import TrainState

_log = logging.getLogger(__name__)


def _walked(tree, sites: Sequence[DecoderBank], convert: Callable[[DecoderBank, str, Mapping], Mapping],
            path: tuple[str, ...] = ()):
    """`tree` with every mapping that holds a site's run converted by `convert(site, where, node)`: a
    variables tree's collection's subtree at the site's namespace, and a mirror's of the parameters
    (an optimizer's moment, an accumulated gradient) whose root is the first mapping its containers
    reach. `path` is the node's keys from that first mapping. Every other mapping keeps its own kind
    (an optimizer's record that is a mapping of its own, a `FrozenDict`), rebuilt from its pytree
    definition, as the step that reads it takes it."""
    if not isinstance(tree, Mapping):
        return jax.tree.map(lambda node: _walked(node, sites, convert) if isinstance(node, Mapping) else node,
                            tree, is_leaf=lambda node: isinstance(node, Mapping))
    for site in sites:
        if site.namespace in (path, path[1:]) and site.view.holds(tree):
            return convert(site, "/".join(path), tree)
    children, definition = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda node: node is not tree)
    if definition.num_leaves == 1 and children[0][1] is tree:
        return tree
    return definition.unflatten([_walked(child, sites, convert, (*path, _key(where)))
                                 for where, child in children])


def _key(where: jax.tree_util.KeyPath) -> str:
    """The name of a mapping's child from its one-step key path: a dict's key, or a record's field."""
    return jax.tree_util.keystr(where, simple=True)


def stage_mesh(mesh: Mesh, stage: int) -> Mesh:
    """Stage `stage`'s devices of `mesh`, as a mesh of the same axes with a stage axis of one: a row of
    a stacked leaf sits there with the spec the layout gives its layer."""
    return Mesh(np.take(mesh.devices, [stage], axis=mesh.axis_names.index(STAGE_AXIS)), mesh.axis_names,
                axis_types=mesh.axis_types)


def _layer_rows(path, leaf: jax.Array, shape: tuple[int, ...]) -> list[jax.Array]:
    """`layer_names.array_rows`, imported where a run is cut, as `_stacked_rows` is."""
    from dew.nn.backbones.layer_names import array_rows

    return array_rows(path, leaf, shape)


def _stacked_rows(rows: Sequence[jax.Array], shape: tuple[int, ...]) -> jax.Array:
    """`decoder_stack.stacked_rows`, imported where a run is stacked: a run with no stage axis reads none of
    the decoder stack, as before stage-local storage (the axis registry its modules fill on import declares
    a parameter of another model by its name alone, ledger 1153)."""
    from dew.nn.backbones.decoder_stack import stacked_rows

    return stacked_rows(rows, shape)


def _sharded(sharding: NamedSharding, mesh: Mesh, spec: Sequence) -> NamedSharding:
    return NamedSharding(mesh, P(*spec), memory_kind=sharding.memory_kind)


@dataclasses.dataclass(frozen=True)
class StageLocal:
    """How a trainer over `mesh` keeps the runs `sites` declare stacked by stage (module docstring);
    with no site, every conversion leaves a tree as it is. `read_whole` names the transforms of an optimizer
    given built that kept every layer whole on every stage instead (`whole_reading`), which the run's plan
    reports (`Trainer.plan`); none otherwise."""

    sites: tuple[DecoderBank, ...]
    mesh: Mesh
    read_whole: tuple[str, ...] = ()

    @classmethod
    def of(cls, objective, mesh: Mesh, *, local: bool = True,
           optimizer: optax.GradientTransformation | None = None) -> StageLocal:
        """How a trainer over `mesh` keeps `objective`'s pipelined runs, from its tree's shapes, traced only
        where a run may keep them stage-local: no run where the mesh has no stage axis, the objective
        declares none (`Objective.stage_sites`, which a log names), the run keeps every layer whole on
        every stage (not `local`: `Layout.stage_local`, a CPU-owned state, a DiLoCo run), or `optimizer`,
        a transformation Dew does not build, holds a transform that may read a parameter whole
        (`whole_reading`): no per-layer view of a transform whose reductions only its own code knows is
        exact short of each layer a leaf of its own, which one program holds on every stage alone, so the
        run keeps every layer whole on every stage, as `Layout(stage_local=False)` does, said once and
        named in its plan (`read_whole`). Dew's own optimizers read a stacked leaf as its layers."""
        stages = mesh.shape.get(STAGE_AXIS, 1)
        if not local or stages == 1:
            return cls((), mesh)
        # The draw's shapes, its tree plain as a run holds it (`dew.batch.unboxed`).
        shapes = unboxed(abstract(objective.initializer, jax.random.key(0)))
        declared = objective.stage_sites(shapes, stages)
        if not declared:
            _log.info("%s declares no run a pipeline over %d stages stacks (Objective.stage_sites), so its "
                      "state keeps every layer whole on every stage", type(objective).__name__, stages)
        # A run a collection holds in part (a stack frozen by layer index) stays on its stored layers in
        # every collection, which the pipeline restacks as it would any stored layers.
        from dew.inference.banks import in_namespace

        staged = cls(tuple(site for site in declared if not any(
            site.view.splits(tree) for tree in in_namespace(shapes, site.namespace).values())), mesh)
        whole = whole_reading(optimizer, shapes, staged)
        if whole:
            _log.warning("the optimizer given built may read a stacked leaf otherwise than layer by layer "
                         "(%s), so each pipelined run keeps every layer whole on every stage, without "
                         "stage-local storage's 1/%d of the run a device; a named optimizer "
                         "(OptimConfig(optimizer=...)) reads each stored layer and keeps them stage-local",
                         "; ".join(whole), stages)
            return cls((), mesh, tuple(whole))
        return staged

    @property
    def stages(self) -> int:
        return self.mesh.shape[STAGE_AXIS]

    def staged[Leaf](self, tree, stack: RowStack[Leaf] = _stacked_rows):
        """`tree`, any tree that holds or mirrors variables, with each run stacked by stage, its rows
        made one leaf by `stack(rows, shape)` (`StackView.stage_form`): arrays, shapes or shardings. A
        run already stacked stays as it is."""
        return _walked(tree, self.sites, lambda site, where, node: site.view.stage_form(where, node, stack))

    def layered[Leaf](self, tree, cut: RowCut[Leaf] = _layer_rows):
        """`staged`'s inverse, `cut(path, leaf, shape)` giving a stacked leaf's rows
        (`StackView.layer_form`)."""
        return _walked(tree, self.sites, lambda site, where, node: site.view.layer_form(node, cut))

    def received(self, objective, state: TrainState) -> TrainState:
        """`state`, one a run handed on (`fit`'s), read as this run's step starts from it: a run stacked
        by stage over another count, or by a run where this one keeps every layer whole, back on its
        stored layers (`objective`'s stage sites at the count it holds), every mirror alike; a state on
        its layers, or stacked as this run stacks, as it is."""
        from dew.nn.backbones.layer_names import stage_count

        held = stage_count(state.variables)
        if held == 1 or (self.sites and held == self.stages):
            return state
        shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype), state.variables)
        sites = objective.stage_sites(shapes, held)
        return self.state(state, lambda tree: _walked(
            tree, sites, lambda site, where, node: site.view.layer_form(node, _layer_rows)))

    def holds(self, variables) -> bool:
        """Whether `variables` hold their runs stacked by stage."""
        staged: list[bool] = []
        _walked(variables, self.sites, lambda site, where, node: staged.append('stages' in node) or node)
        return any(staged)

    @overload
    @staticmethod
    def state(state: TrainState, convert: Callable) -> TrainState: ...

    @overload
    @staticmethod
    def state(state: Mapping[str, object], convert: Callable) -> Mapping[str, object]: ...

    @staticmethod
    def state(state: TrainState | Mapping[str, object], convert: Callable
              ) -> TrainState | Mapping[str, object]:
        """`state` (a `TrainState`, its shapes, its placement, or a mapping of some of its fields) with
        `convert` applied to every field that holds or mirrors the variables: the variables, the
        optimizer state, the EMA, the narrow copies and the accumulated gradient."""
        if isinstance(state, Mapping):
            return {name: convert(value) if name in _MIRRORS else value for name, value in state.items()}
        fields = {name: convert(getattr(state, name)) for name in _MIRRORS}
        if state.accumulation is not None:
            fields["accumulation"] = dataclasses.replace(state.accumulation,
                                                         gradient=convert(state.accumulation.gradient))
        return dataclasses.replace(state, **fields)

    def staged_shapes(self, tree):
        """`tree`'s shapes with each run's stacked by stage."""
        return self.staged(tree, lambda rows, shape: jax.ShapeDtypeStruct((*shape, *rows[0].shape),
                                                                          rows[0].dtype))

    def row_shapes(self, tree):
        """`tree`'s shapes with each stacked run's back on its stored layers, unplaced."""
        return self.layered(tree, lambda path, leaf, shape: [
            jax.ShapeDtypeStruct(leaf.shape[len(shape):], leaf.dtype)] * math.prod(shape))

    def placement(self, rows):
        """The placement of a stage-local tree from `rows`, its stored layers' placement: a run's leaf
        on the stage axis, then the spec of each row, which a like run's rows share."""
        def stacked(shardings: list[NamedSharding], shape: tuple[int, ...]) -> NamedSharding:
            unlike = next((sharding for sharding in shardings if sharding != shardings[0]), None)
            if unlike is not None:
                raise ValueError(f"a pipeline stacks like layers, and the layout places them apart "
                                 f"({shardings[0]} and {unlike}); place a run's layers alike")
            return self._stacked_sharding(shardings[0], len(shape))
        return self.staged(rows, stacked)

    def _stacked_sharding(self, row: NamedSharding, stacked: int) -> NamedSharding:
        return _sharded(row, self.mesh, (STAGE_AXIS, *[None] * (stacked - 1), *row.spec))

    def rows_placement(self, placement):
        """`placement`'s stored layers' own: each row the spec its stacked leaf gives it past the stack,
        on every stage."""
        return self.layered(placement, lambda path, sharding, shape: [
            _sharded(sharding, self.mesh, sharding.spec[len(shape):])] * math.prod(shape))

    @contextlib.contextmanager
    def held(self, *within: contextlib.AbstractContextManager) -> Iterator[None]:
        """The context a step traces and a layout reads the state in, every rule reading each stacked
        run's leaves as the stored layers they stack (`held_by_stage`), entered with `within`."""
        with held_by_stage(self.sites), contextlib.ExitStack() as entered:
            for context in within:
                entered.enter_context(context)
            yield

    def check(self, layout: Layout, variables, placement, mesh: Mesh) -> None:
        """`layout.check` of `variables` placed as `placement`, read as their stored layers: what it
        bounds is what the layout replicates of a layer."""
        layout.check(self.row_shapes(variables), self.rows_placement(placement), mesh)

    def placed(self, state: TrainState, placement_of: Callable[[TrainState], TrainState]) -> TrainState:
        """The placement of `state` (`Trainer.shardings`): `placement_of`'s of a state as it is, or of
        its stored layers' shapes, its stacked runs' placed on the stage axis (`placement`)."""
        if not self.holds(state.variables):
            return placement_of(state)
        return self.state(placement_of(self.state(state, self.row_shapes)), self.placement)

    def supplied(self, state: TrainState, validated: Callable[[TrainState], TrainState]
                 ) -> tuple[TrainState, TrainState]:
        """A state given whole, wherever its arrays are (one device's, another mesh's), its runs stacked by
        stage onto the placement `validated` gives the stacked state's shapes, and that placement: placed
        whole on this mesh first, as `validated` places it, and stacked there."""
        stack = functools.partial(self.state, convert=self.staged)
        placement = validated(jax.eval_shape(stack, state))
        if not self.sites:
            return jax.device_put(state, placement), placement
        whole = jax.device_put(state, validated(jax.eval_shape(lambda given: given, state)))
        return owned_program(stack, out_shardings=placement)(whole), placement

    @overload
    def restore(self, checkpoints: Checkpoints, template: TrainState, step: int | str | None,
                share: DataPartition | None = None, served: Sequence[ProjectionGroup] = ()
                ) -> tuple[TrainState, bytes | None]: ...

    @overload
    def restore(self, checkpoints: Checkpoints, template: Mapping[str, object], step: int | str | None,
                share: DataPartition | None = None, served: Sequence[ProjectionGroup] = ()
                ) -> tuple[Mapping[str, object], bytes | None]: ...

    def restore(self, checkpoints: Checkpoints, template: TrainState | Mapping[str, object],
                step: int | str | None, share: DataPartition | None = None,
                served: Sequence[ProjectionGroup] = ()
                ) -> tuple[TrainState | Mapping[str, object], bytes | None]:
        """`checkpoints.restore` of `template`, a state or some of its fields, each stacked run's
        layers read onto their own stage's devices (`checkpointed`) and stacked there (`restored`);
        `served` as the checkpoints read it."""
        restored, position = checkpoints.restore(self.state(template, self.checkpointed), step, share=share,
                                                 served=served)
        return self.state(restored, self.restored), position

    def stored(self, state: TrainState) -> TrainState:
        """`state` in its stored form, every field that holds or mirrors the variables with each layer whole
        on every stage (`whole`), as one device's run holds it (`Trainer.stored`). It holds what storing
        every layer on every stage would."""
        return self.state(state, self.whole)

    def refuse_unpipelined(self, schedule: Schedule, objective) -> None:
        """Refuse a stage axis under a model whose traced step ran no pipeline (`schedule`)."""
        stages = self.mesh.shape.get(STAGE_AXIS, 1)
        if stages > 1 and not schedule.pipelined:
            raise LayoutRefused(
                f"the stage axis of {stages} holds a pipeline's stages of a stack of like layers (a "
                f"decoder's, a DiT's or a ViT's blocks, a UNet level's transformers) that it splits "
                f"evenly, and {type(objective).__name__}'s model runs none, so every stage would "
                f"compute the whole step; give those devices to the data or fsdp axis")

    def checkpointed(self, shapes):
        """`shapes`, placed, as a checkpoint's rows: each stored layer's shape on its own stage's
        devices (`stage_mesh`), with the spec its stacked leaf gives it past the stack."""
        def cut(path, leaf: jax.ShapeDtypeStruct, shape: tuple[int, ...]) -> list[jax.ShapeDtypeStruct]:
            count = math.prod(shape) // self.stages
            return [jax.ShapeDtypeStruct(leaf.shape[len(shape):], leaf.dtype, sharding=_sharded(
                leaf.sharding, stage_mesh(self.mesh, row // count), leaf.sharding.spec[len(shape):]))
                for row in range(math.prod(shape))]
        return self.layered(shapes, cut)

    def saved(self, state: TrainState) -> TrainState:
        """`state` as a checkpoint holds it, every layer `layers_N`, each row an array on its own
        stage's devices made from that stage's shards: a device copies its own rows, and nothing
        moves between stages."""
        def cut(path, leaf: jax.Array, shape: tuple[int, ...]) -> list[jax.Array]:
            sharding = leaf.sharding
            assert isinstance(sharding, NamedSharding), "a stacked leaf is placed on the stage axis"
            rows = []
            for stage in range(self.stages):
                devices = stage_mesh(self.mesh, stage)
                held = set(devices.devices.flat)
                piece = jax.make_array_from_single_device_arrays(
                    (1, *leaf.shape[1:]), _sharded(sharding, devices, (None, *sharding.spec[1:])),
                    [shard.data for shard in leaf.addressable_shards if shard.device in held])
                row = _sharded(sharding, devices, sharding.spec[len(shape):])
                rows.extend(program(_cut_piece, static_argnames="stacked", out_shardings=(row,) * (
                    math.prod(shape) // self.stages))(piece, stacked=len(shape)))
            return rows
        return self.state(state, lambda tree: self.layered(tree, cut))

    def restored(self, tree):
        """`saved`'s inverse of one of a state's trees, read onto `checkpointed`'s placement: each
        stage's rows stacked on its own devices, their buffers given up to it, and the stages' pieces
        made one array, their buffers its shards."""
        def stack(rows: list[jax.Array], shape: tuple[int, ...]) -> jax.Array:
            count, pieces, row = math.prod(shape) // self.stages, [], rows[0].sharding
            assert isinstance(row, NamedSharding), "a checkpoint's row is read onto its stage's devices"
            for stage in range(self.stages):
                held = rows[stage * count:(stage + 1) * count]
                devices = held[0].sharding
                assert isinstance(devices, NamedSharding) and isinstance(devices.mesh, Mesh), (
                    "a stage's rows are read onto its own devices")
                # A stage's rows are the restore's own arrays, no other leaf's: its piece takes them over.
                piece = _sharded(row, devices.mesh, (None, *[None] * (len(shape) - 1), *row.spec))
                pieces.append(program(_stacked_piece, static_argnames="within", donate_argnums=0,
                                      out_shardings=piece)(held, within=shape[1:]))
            return jax.make_array_from_single_device_arrays(
                (*shape, *rows[0].shape), self._stacked_sharding(row, len(shape)),
                [shard.data for piece in pieces for shard in piece.addressable_shards])
        return self.staged(tree, stack)

    def whole(self, variables):
        """`variables` with each stacked run back on its stored layers, each layer on every stage
        (`Trainer.stored`'s). It costs what a state that held every layer on every device held
        throughout; variables that hold no stacked run are returned as they are."""
        if not self.holds(variables):
            return variables
        placement = self.rows_placement(jax.tree.map(lambda leaf: leaf.sharding, variables))
        return program(_layered_arrays, static_argnums=0, out_shardings=placement)(self, variables)


_MIRRORS = ("variables", "opt_state", "ema", "compute")
"""The train-state fields that hold or mirror the variables, which a stage-local state stacks alike."""


def _cut_piece(piece: jax.Array, *, stacked: int) -> tuple[jax.Array, ...]:
    """A stage's piece of a leaf stacked over `stacked` leading dimensions, as its rows (`array_rows`)."""
    return tuple(_layer_rows((), piece, piece.shape[:stacked]))


def _stacked_piece(rows: list[jax.Array], *, within: tuple[int, ...]) -> jax.Array:
    """A stage's rows stacked into its piece of the leaf, `[1, *within, ...]`."""
    return _stacked_rows(rows, (1, *within))


def _layered_arrays(local: StageLocal, tree):
    return local.layered(tree)


def whole_reading(optimizer: optax.GradientTransformation | None, shapes, local: StageLocal) -> list[str]:
    """Why `optimizer`'s update over a run's leaves stacked as `local` stacks them
    (`StageLocal.staged_shapes`) may compute otherwise than its update over the stored layers: none where,
    called as a commit calls it (`dew.training.transaction.optimizer_update`, a run's progress handed it
    or not), every leaf it returns, its update's and its state's, is of the stacked layers' shapes and, row
    by row, the same expression of its row's leaves over the stack as over each layer (`_expressions`), as
    optax's moments, decay, rates and schedules are, M3's products batched on the rows, and a clip by the
    global norm, its layers' sum (rows 3995, 4035, 4036). A reduction or a product across a stacked leaf's
    layers, a call it cannot see into, a QK-Clip's read of the forward's maxima, or a choice its build
    makes otherwise over a stacked tree (a mask by rank, a tree of the layers' paths) is named, and keeps
    every layer whole: LAMB's trust ratio, Adafactor's factored moments. None for Dew's own optimizers
    (None), which read a stacked leaf as its layers, and for a run with no stacked site."""
    from dew.gauss_newton import solves_equations
    from dew.optim import clips_qk
    from dew.training.transaction import optimizer_update

    if optimizer is None or not local.sites:
        return []
    with local.held():
        layer_params = shapes["params"]
        layer_state = jax.eval_shape(optimizer.init, layer_params)
        # The state as a run holds it: built over the layer tree, then stacked (`StageLocal.state`).
        state = local.staged_shapes(layer_state)
        if solves_equations(state):
            # Its update is a solve of the step's system over every parameter at once (`dew.gauss_newton`),
            # read from the statistics the step hands it, not its gradient a layer at a time.
            return ["a Gauss-Newton solve, over every parameter at once"]
        if clips_qk(state):
            return ["a QK-Clip, which reads the attention maxima its forward reports for each layer"]
        params = local.staged_shapes(shapes)["params"]
        try:
            built = jax.eval_shape(optimizer.init, params)
        except ValueError:
            # optax refuses a tree its build was given for another (a mask of the layers' paths).
            return ["a transformation built against the layer tree, which builds over no stacked one"]
        if _schema(built) != _schema(state):
            return ["a transformation whose build reads its tree's shapes, deciding otherwise when stacked"]
        held_leaves = len(jax.tree.leaves((layer_params, layer_state, layer_params)))
        places = jax.tree.unflatten(jax.tree.structure((layer_params, layer_state, layer_params)),
                                    range(held_leaves))
        inputs = jax.tree.leaves((local.staged({"params": places[0]}, _row_places)["params"],
                                  local.staged(places[1], _row_places),
                                  local.staged({"params": places[2]}, _row_places)["params"]))
        refusals: list[str] = []
        # A run's progress, where it has one, an input of both programs, handed as a commit hands it.
        clocks: list[tuple] = [()]
        if isinstance(optimizer, optax.GradientTransformationExtraArgs):
            clocks.append((jax.ShapeDtypeStruct((), jnp.float32),))
        for clock in clocks:
            def update(gradient, opt_state, held, *progress):
                # As a commit calls it, handed no attention maxima (no QK-Clip, above).
                return optimizer_update(optimizer, gradient, opt_state, held,
                                        progress=progress[0] if progress else None)

            returned = jax.eval_shape(update, layer_params, layer_state, layer_params, *clock)
            made = jax.tree.unflatten(jax.tree.structure(returned), range(len(jax.tree.leaves(returned))))
            outputs = jax.tree.leaves((local.staged({"params": made[0]}, _row_places)["params"],
                                       local.staged(made[1], _row_places)))
            staged = (local.staged_shapes({"params": returned[0]})["params"],
                      local.staged_shapes(returned[1]))
            if _schema(jax.eval_shape(update, params, state, params, *clock)) != _schema(staged):
                refusals.append("an update whose results over the stack are not the stacked layers' shapes")
                continue
            # The progress, the layer program's last input, is the same input of the stacked one.
            given = [*inputs, *range(held_leaves, held_leaves + len(clock))]
            refusals += _expressions(jax.make_jaxpr(update)(layer_params, layer_state, layer_params, *clock),
                                     jax.make_jaxpr(update)(params, state, params, *clock), given, outputs)
    return refusals


def _schema(tree) -> tuple:
    """`tree`'s structure and each leaf's shape and dtype."""
    return jax.tree.structure(tree), tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(tree))


@dataclasses.dataclass(frozen=True)
class _Rows:
    """A stacked leaf's rows, by their places among the layer tree's leaves, stage-major (`staged`), over
    its `lead` leading axes."""

    places: tuple[int, ...]
    lead: int


def _row_places(rows, shape) -> _Rows:
    return _Rows(tuple(rows), len(shape))


def _expressions(layered, stacked, inputs: Sequence, outputs: Sequence) -> list[str]:
    """Where the program `stacked`, over the leaves `inputs` (each a layer program input's place, or the
    `_Rows` it stacks), computes any of its `outputs` (each likewise) otherwise than `layered` computes
    each row: an output is the same expression of the inputs, row by row, where both programs, their
    calls inlined (`dew.training.staging.flattened`), compute it by the same equations of the same
    primitives, parameters and constants over the same inputs' rows (`_Numbering`). An equation of the
    stack reads each row as its layer's does where it holds the rows on its leading axes and its
    parameters are the layer's over them (`_row_form`): an entrywise one, a product, a reduction, a
    transpose, a reshape or a broadcast of each row. A sum over every axis of a stack is the sum of its
    rows', and a scalar's sums of non-negative floating terms are one sum whatever their order
    (`_Numbering.summed`), as a clip by the global norm forms it: the two programs' sums then round
    within one bound of the exact sum, though not to one value, so a comparison after them may decide
    otherwise at its threshold (the admission row 4036 asked for, a global-norm clip stage-local). Row
    r of a stack of n rows is read against row r of every stacked input of n rows; a stack of another
    count is no row's."""
    from dew.training.staging import flattened

    numbering = _Numbering()
    lay_equations, lay_results = flattened(layered.jaxpr, layered.jaxpr.invars, layered.consts)
    stack_equations, stack_results = flattened(stacked.jaxpr, stacked.jaxpr.invars, stacked.consts)
    lay = numbering.program(lay_equations, {var: numbering.number(("input", place))
                                            for place, var in enumerate(layered.jaxpr.invars)}, {})
    named = {var: tuple(numbering.number(("input", place)) for place in given.places)
             if isinstance(given, _Rows) else numbering.number(("input", given))
             for var, given in zip(stacked.jaxpr.invars, inputs, strict=True)}
    leads = {var: given.lead for var, given in zip(stacked.jaxpr.invars, inputs, strict=True)
             if isinstance(given, _Rows)}
    stack = numbering.program(stack_equations, named, leads)
    refusals = []
    for output, returned in zip(outputs, stack_results, strict=True):
        places = output.places if isinstance(output, _Rows) else (output,)
        rows = _rows_of(numbering.read(stack, returned), output, leads.get(returned), numbering.uniform)
        if rows is None or any(row != numbering.read(lay, lay_results[place])
                               for row, place in zip(rows, places, strict=True)):
            refusals.append(f"its output {places[0]} computes its layer otherwise over the stack, or by no "
                            "expression of its layer")
    return refusals


def _rows_of(number, output, lead: int | None, uniform: set[int]) -> tuple | None:
    """Each row's number of a result the stacked program numbered `number`, returned as `output`: a
    stack's rows on the leading axes `output` holds them on; one number of no stack, each row's only where
    it is one value at every entry (`uniform`, a scalar's broadcast); None where the two disagree."""
    if not isinstance(output, _Rows):
        return None if isinstance(number, tuple) or number is None else (number,)
    if not isinstance(number, tuple):
        return None if number not in uniform else (number,) * len(output.places)
    if len(number) != len(output.places) or lead != output.lead:
        return None
    return number


class _Numbering:
    """Numbers for expressions, one an expression however often it is formed: an input, a constant, or an
    equation's result over numbered operands (`_expressions`), keyed by the primitive and its parameters
    as the algebra pass keys them (`dew.algebra.terms.parameters_key`); a stack's value numbered row by
    row, a tuple of its rows' numbers."""

    def __init__(self):
        self.numbers: dict[Hashable, int] = {}
        self.sums: dict[int, tuple[int, int, int]] = {}
        self.words: dict[int, tuple[int, int]] = {}
        """Each summed term's two words, drawn at random for this numbering once its programs are traced
        (`summed`)."""
        self.nonnegative: set[int] = set()
        """The numbers of values no entry of which is negative: squares, magnitudes, their sums and roots,
        whose sums reassociated round within the same bound in any order (no cancellation), so their
        order is no part of their expression (`summed`)."""
        self.uniform: set[int] = set()
        """The numbers of a scalar's broadcasts, a value the same at every entry, so the same in each row of
        a stack of its shape."""

    def number(self, key: Hashable) -> int:
        return self.numbers.setdefault(key, len(self.numbers))

    def summed(self, terms: Sequence[int]) -> int:
        """The number of the sum of `terms`, each non-negative (`nonnegative`), each sum among them read as
        its own terms, in any order: the multiset of its terms, named by its count of terms and two exact
        sums of each term's two words times its multiplicity, a few integers a sum however many terms it
        holds, so no sum keeps its prefixes. The words are drawn at random, 64 bits each, once the programs
        are traced (`words`): two distinct multisets differ in some term's multiplicity, and the one value
        of that term's word that would make their sums agree, the rest drawn, is a 2**-64 chance a word,
        so whatever the programs compute, two sums are named alike unequal with a 2**-128 chance. A word
        fixed by its term, or a sum kept modulo a power of two, a program could reach a collision of
        (doubling a sum 64 times empties it modulo 2**64)."""
        first = second = count = 0
        for term in terms:
            held = self.sums.get(term)
            if held is None:
                held = (*self.words.setdefault(term, (secrets.randbits(64), secrets.randbits(64))), 1)
            first, second, count = first + held[0], second + held[1], count + held[2]
        total = self.number(("sum", first, second, count))
        self.sums[total] = (first, second, count)
        self.nonnegative.add(total)
        return total

    def read(self, numbers: dict, value) -> int | tuple[int, ...] | None:
        """`value`'s number among a program's `numbers` (`program`): a literal's of its own."""
        if isinstance(value, jcore.Literal):
            number = self.number(("literal", str(value.aval), parameter_key(value.val)))
            held = np.asarray(value.val)
            if held.dtype.kind == "f" and bool(np.all(held >= 0)):
                # A literal with no negative entry, a sum's starting zero among them (`summed`).
                self.nonnegative.add(number)
            return number
        return numbers.get(value)

    def equation(self, equation, params: Mapping, operands: Sequence, shapes: Sequence) -> list[int]:
        """The numbers of `equation`'s results over `operands` of `shapes` under `params`: an entry of a
        scalar's broadcast is the scalar, whatever shape it fills; an entrywise result's entry is its
        operands' entries'; any other's reads its operands' shapes too. A scalar's sum of scalars is one
        sum of their terms (`summed`)."""
        name = jax_name(equation.primitive)
        spread = name == "broadcast_in_dim" and tuple(shapes[0]) == ()
        if name == "add" and all(tuple(shape) == () for shape in shapes) and all(
                operand in self.nonnegative for operand in operands):
            return [self.summed(operands)]
        held = {key: setting for key, setting in params.items() if not (spread and key in _SHAPED)}
        _, keyed = parameters_key(equation.primitive, held)
        # An entrywise result over operands of one shape, or scalars, reads no shape; one that broadcasts
        # an operand's axes of one entry reads which.
        free = spread or (name in ENTRYWISE and len({tuple(shape) for shape in shapes} - {()}) <= 1)
        read = () if free else tuple(map(tuple, shapes))
        made = [self.number((equation.primitive, keyed, tuple(operands), read, index))
                for index in range(len(equation.outvars))]
        # A scalar's broadcast, and an entrywise result of such broadcasts and scalars, is one value at every
        # entry.
        uniform = all(operand in self.uniform or tuple(shape) == ()
                      for operand, shape in zip(operands, shapes, strict=True))
        if spread or (free and name in ENTRYWISE and uniform):
            self.uniform.update(made)
        if _keeps_sign(equation, name, params, operands, self.nonnegative):
            self.nonnegative.update(made)
        return made

    def program(self, equations, named: dict, leads: dict) -> dict:
        """Each value of `equations` over the inputs `named` to its expression's number, in program order, a
        stack's its rows' numbers, its leading axes of rows in `leads`, which this extends by each stack it
        makes; a value that reads a stack otherwise than row by row (`made`) has none."""
        numbers = dict(named)
        for equation in equations:
            operands = [self.read(numbers, operand) for operand in equation.invars]
            if any(operand is None for operand in operands):
                continue
            made = self.made(equation, operands, [leads.get(operand) if isinstance(number, tuple) else None
                                                   for operand, number in zip(equation.invars, operands,
                                                                              strict=True)])
            if made is None:
                continue
            for out, (number, lead) in zip(equation.outvars, made, strict=True):
                numbers[out] = number
                if lead is not None:
                    leads[out] = lead
        return numbers

    def made(self, equation, operands: list, leads: list) -> list[tuple] | None:
        """`equation`'s results over numbered `operands` (a stack's held on its `leads`), each its number
        and the leading axes of rows of a stack it makes: a loop or a branch read through its programs
        (`inlined`); a stack's rows moved whole (`_moved`); an equation of no stack as it is; and one over
        a stack row by row (`_row_form`), or a sum over every axis of one (`_full_sum`). None where a row
        reads another."""
        name = jax_name(equation.primitive)
        if name in ("scan", "cond"):
            return self.inlined(equation, operands, leads)
        moved = _moved(equation, operands, leads)
        if moved is not None:
            return [moved]
        stacked = [index for index, operand in enumerate(operands) if isinstance(operand, tuple)]
        shapes = [operand.aval.shape for operand in equation.invars]
        if not stacked:
            return [(number, None) for number in self.equation(equation, equation.params, operands, shapes)]
        if len({(len(operands[index]), leads[index]) for index in stacked}) != 1:
            return None
        count, held = len(operands[stacked[0]]), leads[stacked[0]]
        whole = tuple(shapes[stacked[0]])
        for index, operand in enumerate(operands):
            if index in stacked or not entrywise(equation) or tuple(shapes[index]) != whole:
                continue
            # An entrywise equation's operand of no stack of a stack's shape: each row's the same where it
            # is a scalar's broadcast; any other's is read whole beside each row, a key no layer's
            # expression holds (its shape among the key's).
            if operand in self.uniform:
                operands[index] = (operand,) * count
                stacked.append(index)
        row_shapes = [shape[held:] if index in stacked else shape for index, shape in enumerate(shapes)]
        every = _full_sum(equation, held)
        if every is not None and all(row in self.nonnegative for row in operands[0]):
            # A sum over every axis of a stack of no negative entry: the sum of each row's sum over its own
            # axes, in any order (`summed`).
            return [(self.summed([self.equation(equation, every, [operands[0][row]], row_shapes)[0]
                                  for row in range(count)]), None)]
        form = _row_form(equation, held, set(stacked), shapes)
        if form is None:
            return None
        by_row = [self.equation(equation, form, [operand[row] if isinstance(operand, tuple) else operand
                                                 for operand in operands], row_shapes)
                  for row in range(count)]
        return [(tuple(made[index] for made in by_row), held) for index in range(len(equation.outvars))]

    def inlined(self, equation, operands: list, leads: list) -> list[tuple] | None:
        """A loop's (`scan`) or a branch's (`cond`) results read through its programs over `operands`: a loop
        of at most `UNROLLED` steps unrolled, its carry through each step and each step's slice of what it
        scans, which a stack is none of; a branch's each program's results, one where they agree, else the
        branch's choice among them, row by row of a stack. None for a longer loop."""
        from dew.training.staging import flattened

        def run(closed, values: list, held: list) -> list[tuple] | None:
            body, returned = flattened(closed.jaxpr, closed.jaxpr.invars, closed.consts)
            inner = {var: lead for var, lead in zip(closed.jaxpr.invars, held, strict=True)
                     if lead is not None}
            numbers = self.program(body, dict(zip(closed.jaxpr.invars, values, strict=True)), inner)
            out = [self.read(numbers, value) for value in returned]
            if any(number is None for number in out):
                return None
            return [(number, inner.get(value) if isinstance(number, tuple) else None)
                    for number, value in zip(out, returned, strict=True)]

        params = equation.params
        if jax_name(equation.primitive) == "scan":
            counts = _scan_counts(equation)
            if counts is None:
                return None
            consts, carries = counts
            scanned = operands[consts + carries:]
            if params["length"] > UNROLLED or any(isinstance(value, tuple) for value in scanned):
                return None
            carry = list(zip(operands[consts:consts + carries], leads[consts:consts + carries], strict=True))
            steps: dict[int, list] = {}
            order = range(params["length"] - 1, -1, -1) if params["reverse"] else range(params["length"])
            for step in order:
                sliced = [self.number(("scanned", value, step)) for value in scanned]
                values = [*operands[:consts], *(number for number, _ in carry), *sliced]
                held = [*leads[:consts], *(lead for _, lead in carry), *([None] * len(sliced))]
                out = run(params["jaxpr"], values, held)
                if out is None or any(isinstance(number, tuple) for number, _ in out[carries:]):
                    return None
                carry, steps[step] = out[:carries], [number for number, _ in out[carries:]]
            # A loop's stacked results, its steps' on a new leading axis, keyed by their shapes too.
            stacked = [(self.number(("stacked steps", tuple(equation.outvars[carries + index].aval.shape),
                                     tuple(steps[step][index] for step in sorted(steps)))), None)
                       for index in range(len(equation.outvars) - carries)]
            return [*carry, *stacked]
        index, *given = operands
        if isinstance(index, tuple):
            return None
        branches = [run(branch, given, leads[1:]) for branch in params["branches"]]
        if any(branch is None for branch in branches):
            return None
        chosen = []
        for outs in zip(*branches, strict=True):
            if len(set(outs)) == 1:
                chosen.append(outs[0])
            elif all(not isinstance(number, tuple) for number, _ in outs):
                chosen.append((self.number(("branch", index, tuple(number for number, _ in outs))), None))
            else:
                # A branch's stack against another's value of no stack, the same each row (a scalar's
                # broadcast), row by row; stacks of other rows than each other's are no branch's choice.
                shapes = {(len(number), lead) for number, lead in outs if isinstance(number, tuple)}
                if len(shapes) != 1:
                    return None
                count, lead = shapes.pop()
                each = [number if isinstance(number, tuple) else (number,) * count for number, _ in outs]
                rows = zip(*each, strict=True)
                chosen.append((tuple(self.number(("branch", index, row)) for row in rows), lead))
        return chosen


def _scan_counts(equation) -> tuple[int, int] | None:
    """A loop's (`scan`) count of constants and of carried values among its operands: as its parameters
    state them where they do (`num_consts`, `num_carry`), else the one split its program's values allow,
    each constant and carry its program's own, each scanned value and each stacked result its program's
    a step, `length` steps of it; None where none or several do."""
    params = equation.params
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    body, length = params["jaxpr"].jaxpr, params["length"]
    given, made = [var.aval for var in equation.invars], [var.aval for var in equation.outvars]
    reads, writes = [var.aval for var in body.invars], [var.aval for var in body.outvars]

    def alike(one, other) -> bool:
        return tuple(one.shape) == tuple(other.shape) and one.dtype == other.dtype

    def stepped(whole, step) -> bool:
        return tuple(whole.shape) == (length, *step.shape) and whole.dtype == step.dtype

    def fits(consts: int, carries: int) -> bool:
        held = consts + carries
        return (held <= len(given)
                and all(map(alike, given[:held], reads[:held]))
                and all(map(alike, writes[:carries], reads[consts:held]))
                and all(map(stepped, given[held:], reads[held:]))
                and all(map(stepped, made[carries:], writes[carries:])))

    splits = [(consts, carries) for consts in range(len(given) + 1) for carries in range(len(made) + 1)
              if fits(consts, carries)]
    return splits[0] if len(splits) == 1 else None


UNROLLED = 256
"""The longest loop the stage-local check reads through step by step (`_Numbering.inlined`), its program
numbered once a step; a longer one keeps every layer whole."""


def _keeps_sign(equation, name: str | None, params: Mapping, operands: Sequence,
                nonnegative: set[int]) -> bool:
    """Whether an equation `name` under `params` over `operands` makes no negative entry, its operands and
    results all real floating point (an integer's magnitude or square can overflow negative, a
    complex square is no ordering at all): a square (`x * x`, `square`, an even integer power), a
    magnitude, an exponential, or a sum, product, maximum, quotient, root or floating conversion of
    values that make none."""
    values = [*equation.invars, *equation.outvars]
    if not all(jnp.issubdtype(value.aval.dtype, jnp.floating) for value in values):
        return False
    if name in ("abs", "exp", "square") or (name == "mul" and len(operands) == 2
                                             and operands[0] == operands[1]):
        return True
    if name == "integer_pow" and params.get("y", 1) % 2 == 0:
        return True
    return name in ("add", "mul", "div", "max", "sqrt", "rsqrt", "reduce_sum", "reduce_max",
                    "convert_element_type", "broadcast_in_dim", "reshape", "squeeze", "expand_dims") and all(
        operand in nonnegative for operand in operands)


def _moved(equation, operands: list, leads: list) -> tuple | None:
    """A stack's rows moved whole, each row as it was: a reshape that keeps every row's own shape after
    its rows' axes (merging, splitting or adding them, a value of no stack becoming a stack of one row,
    one row its value again), a squeeze of its rows' axes of one row, a slice of rows along its one axis
    of them, or a concatenation of stacks along it. None for any other equation."""
    name, params = jax_name(equation.primitive), equation.params
    if name not in ("reshape", "squeeze", "slice", "concatenate"):
        return None
    shape = tuple(equation.invars[0].aval.shape)
    number, lead = operands[0], leads[0]
    out = tuple(equation.outvars[0].aval.shape)
    if name in ("reshape", "squeeze"):
        if name == "reshape" and params.get("dimensions") is not None:
            return None
        held = 0 if lead is None else lead
        row = shape[held:]
        if len(out) < len(row) or (row and out[len(out) - len(row):] != row):
            return None
        rows = out[:len(out) - len(row)]
        count = 1 if lead is None else len(number)
        if math.prod(rows) != count or (name == "squeeze" and rows):
            return None
        if not rows:
            # One row, its value again.
            return (number[0], None) if isinstance(number, tuple) else None
        return ((number,) if lead is None else number), len(rows)
    if lead != 1:
        return None
    if name == "slice":
        strides = params["strides"]
        start, limit = tuple(params["start_indices"]), tuple(params["limit_indices"])
        whole = start[1:] == (0,) * (len(shape) - 1) and limit[1:] == shape[1:]
        if (strides is not None and any(stride != 1 for stride in strides)) or not whole:
            return None
        return number[start[0]:limit[0]], 1
    if name == "concatenate" and params["dimension"] == 0 and all(
            isinstance(operand, tuple) for operand in operands) and all(held == 1 for held in leads):
        return tuple(row for operand in operands for row in operand), 1
    return None


_SHAPED = frozenset({"shape", "broadcast_dimensions", "sharding"})
"""A scalar's broadcast's parameters that name the shape it fills, which a stacked leaf's holds more of."""


def _shifted(axes, held: int) -> tuple[int, ...] | None:
    """`axes` of a stack held on `held` leading axes as a row's, None where one is a row's own."""
    if any(axis < held for axis in axes):
        return None
    return tuple(int(axis) - held for axis in axes)


def _full_sum(equation, held: int) -> Mapping | None:
    """A sum of a stack over every axis (`reduce_sum`), as each row's sum over its own axes's parameters;
    None for any other equation."""
    if jax_name(equation.primitive) != "reduce_sum":
        return None
    rank = len(equation.invars[0].aval.shape)
    if tuple(sorted(equation.params["axes"])) != tuple(range(rank)):
        return None
    return {**equation.params, "axes": tuple(range(rank - held))}


def _row_form(equation, held: int, stacked: set[int], shapes: Sequence) -> Mapping | None:
    """`equation`'s parameters as one of its rows reads them, its stacked operands' (`stacked`) rows on their
    `held` leading axes and its results' too: an entrywise one over operands of its result's shape or
    scalars; a broadcast, a reshape, a transpose or a pad that keeps the rows' axes leading; a reduction,
    a cumulative one, a squeeze, an expansion or a concatenation along a row's own axes; a product of rows
    batched on their leading axes, or of a stack's rows and one operand of no stack. None where a row
    reads another, or the rows leave their leading axes."""
    name, params = jax_name(equation.primitive), dict(equation.params)
    out = equation.outvars[0].aval.shape
    rows = shapes[next(iter(stacked))][:held]
    if tuple(out[:held]) != tuple(rows):
        return None
    if entrywise(equation) and name != "broadcast_in_dim":
        # Each operand of its result's shape, a scalar, or of one entry along an axis it broadcasts.
        fits = all(tuple(shape) == () or (len(shape) == len(out) and all(
            size in (whole, 1) for size, whole in zip(shape, out, strict=True))) for shape in shapes)
        return params if fits else None
    if name == "broadcast_in_dim":
        dims = tuple(params["broadcast_dimensions"])
        if dims[:held] != tuple(range(held)):
            return None
        return {**params, "shape": tuple(params["shape"])[held:],
                "broadcast_dimensions": _shifted(dims[held:], held)}
    if name == "reshape":
        if params.get("dimensions") is not None:
            return None
        return {**params, "new_sizes": tuple(params["new_sizes"])[held:]}
    if name == "transpose":
        order = tuple(params["permutation"])
        if order[:held] != tuple(range(held)):
            return None
        return {**params, "permutation": _shifted(order[held:], held)}
    for key in ("axes", "dimensions"):
        if key in params and name in _ALONG:
            moved = _shifted(tuple(params[key]), held)
            return None if moved is None else {**params, key: moved}
    if name == "concatenate" and stacked != set(range(len(shapes))):
        return None
    for key in ("axis", "dimension"):
        if key in params and name in _ALONG:
            moved = _shifted((params[key],), held)
            return None if moved is None else {**params, key: moved[0]}
    if name == "pad":
        config = tuple(params["padding_config"])
        if stacked != {0} or any(tuple(int(edge) for edge in axis) != (0, 0, 0) for axis in config[:held]):
            return None
        return {**params, "padding_config": config[held:]}
    if name == "dot_general":
        (lhs_c, rhs_c), (lhs_b, rhs_b) = params["dimension_numbers"]
        if stacked == {0, 1}:
            if tuple(lhs_b[:held]) != tuple(range(held)) or tuple(rhs_b[:held]) != tuple(range(held)):
                return None
            numbers = ((_shifted(lhs_c, held), _shifted(rhs_c, held)),
                       (_shifted(lhs_b[held:], held), _shifted(rhs_b[held:], held)))
        elif stacked == {0} and not lhs_b and not rhs_b and _shifted(lhs_c, held) is not None:
            numbers = ((_shifted(lhs_c, held), tuple(rhs_c)), ((), ()))
        else:
            return None
        return {**params, "dimension_numbers": numbers}
    return None


_ALONG = frozenset({"reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or",
                    "argmax", "argmin", "squeeze", "expand_dims", "cumsum", "cumprod", "cummax", "cummin",
                    "cumlogsumexp", "concatenate"})
"""The equations along axes their parameters name, which a row reads where none is a row's own."""
