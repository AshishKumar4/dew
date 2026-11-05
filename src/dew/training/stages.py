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
  unstacked. An optax transformation given as it is runs over the stacked leaf, so unless every state it
  holds names a transform Dew knows to act a number at a time, the run keeps every layer whole on every
  stage, the replicated layout's memory, which trains it exactly (`StageLocal.of`): a stateless
  transform (`optax.EmptyState`) says nothing of what it reads, and optax's LAMB trust ratio, its
  adaptive clip and a constant rate's scale all hold one (row 3934).

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
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, overload

import jax
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.batch import unboxed
from dew.mesh import STAGE_AXIS, LayoutRefused
from dew.nn.kernels.options import owned_program, program
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
    """Why `optimizer`'s update may read a stacked run's leaf otherwise than layer by layer: none where,
    traced over `shapes`' parameters stacked as `local` stacks them (`StageLocal.staged`), each stored
    layer of every stacked leaf it returns, its update's and its state's, reads that layer alone of every
    stacked leaf it reads, the gradient's, the state's and the parameters' (row 3995): along the stacked
    layers none later and, the program over them reversed, none earlier
    (`dew.interop.sequence_causality.positional_refusal`), as optax's constant rate, its moments and its
    decay read; a trust ratio or a clip over a leaf's norm reads every layer, and is named. None for Dew's
    own optimizers (None), which read a stacked leaf as its layers, and for a run with no stacked site."""
    from dew.gauss_newton import solves_equations
    from dew.interop.sequence_causality import positional_refusal
    from dew.nn.sharding import layers_of

    if optimizer is None or not local.sites:
        return []
    with local.held():
        params = local.staged_shapes(shapes)["params"]
        state = jax.eval_shape(optimizer.init, params)
        if solves_equations(state):
            # Its update is a solve of the step's system over every parameter at once (`dew.gauss_newton`),
            # read from the statistics the step hands it, not its gradient a layer at a time.
            return ["a Gauss-Newton solve, over every parameter at once"]
        arguments = (params, state, params)
        given, structure = jax.tree_util.tree_flatten_with_path(arguments)
        stacked = [layers_of(path)[1] for path, _ in given]
        merged = [jax.ShapeDtypeStruct((math.prod(leaf.shape[:held]), *leaf.shape[held:]), leaf.dtype)
                  if held else leaf for (_, leaf), held in zip(given, stacked, strict=True)]
        returned, _ = jax.tree_util.tree_flatten_with_path(jax.eval_shape(optimizer.update, *arguments))
        made = [layers_of(path)[1] for path, _ in returned]

        def update(*leaves, flipped: frozenset[int] = frozenset(), flipping: frozenset[int] = frozenset()):
            # The stacked leaves' layers on one axis each, reversed where the walk reads them reversed.
            layered = [jax.numpy.flip(leaf, 0) if index in flipped else leaf
                       for index, leaf in enumerate(leaves)]
            whole = [leaf.reshape(shape.shape) if held else leaf for leaf, (_, shape), held in
                     zip(layered, given, stacked, strict=True)]
            out = jax.tree.leaves(optimizer.update(*jax.tree.unflatten(structure, whole)))
            out = [leaf.reshape((-1, *leaf.shape[held:])) if held else leaf
                   for leaf, held in zip(out, made, strict=True)]
            return [jax.numpy.flip(leaf, 0) if index in flipping else leaf for index, leaf in enumerate(out)]

        refusals = []
        for length in sorted({leaf.shape[0] for leaf, held in zip(merged, stacked, strict=True) if held}):
            inputs = {index: 0 for index, (leaf, held) in enumerate(zip(merged, stacked, strict=True))
                      if held and leaf.shape[0] == length}
            outputs = {index: 0 for index, ((_, leaf), held) in enumerate(zip(returned, made, strict=True))
                       if held and math.prod(leaf.shape[:held]) == length}
            for flips in ((frozenset(), frozenset()), (frozenset(inputs), frozenset(outputs))):
                traced = functools.partial(update, flipped=flips[0], flipping=flips[1])
                closed = jax.make_jaxpr(traced)(*merged)
                refused = positional_refusal(closed, inputs, length, outputs)
                if refused is not None:
                    way = " reversed" if flips[0] else ""
                    refusals.append(f"over {length} stored layers{way}, {refused}")
    return refusals
