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
  LAMB's trust ratio and QK-Clip's per-layer maxima each give what they give unstacked. An optax
  transformation given as it is runs over the stacked leaf, so one whose state names a transform Dew
  does not know to act a number at a time is refused by name (`StageLocal.refused`); a stateless one
  (`optax.EmptyState`) that reduces over a whole leaf cannot be told apart, and the docs say so.

Rejected: the stage axis as one more axis the state splits over, as fsdp splits it, which keeps each
leaf one layer but moves the run's parameters to their stage and its gradients back every step, the
traffic a pipeline exists to avoid; and a checkpoint in the stacked form, which every reader of
`layers_N` would have to learn, for every stage count.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.inference.banks import in_namespace
from dew.mesh import STAGE_AXIS, LayoutRefused
from dew.nn.backbones.decoder_stack import DecoderBank, RowCut, RowStack, stacked_rows
from dew.nn.kernels.options import program
from dew.nn.sharding import held_by_stage
from dew.optim import STACKED_STATES

if TYPE_CHECKING:
    from dew.checkpoints import Checkpoints
    from dew.data.dataset import DataPartition
    from dew.distributed.placement import Layout
    from dew.nn.sharding import Schedule
    from dew.state import TrainState


def _walked(tree, sites: Sequence[DecoderBank], convert: Callable[[DecoderBank, str, Mapping], Mapping],
            path: tuple[str, ...] = ()):
    """`tree` with every mapping that holds a site's run converted by `convert(site, where, node)`: a
    variables tree's collection's subtree at the site's namespace, and a mirror's of the parameters
    (an optimizer's moment, an accumulated gradient) whose root is the first mapping its containers
    reach. `path` is the node's dict keys from that first mapping."""
    if not isinstance(tree, Mapping):
        return jax.tree.map(lambda node: _walked(node, sites, convert) if isinstance(node, Mapping) else node,
                            tree, is_leaf=lambda node: isinstance(node, Mapping))
    for site in sites:
        if site.namespace in (path, path[1:]) and site.view.holds(tree):
            return convert(site, "/".join(path), tree)
    return {key: _walked(value, sites, convert, (*path, key)) for key, value in tree.items()}


def stage_mesh(mesh: Mesh, stage: int) -> Mesh:
    """Stage `stage`'s devices of `mesh`, as a mesh of the same axes with a stage axis of one: a row of
    a stacked leaf sits there with the spec the layout gives its layer."""
    return Mesh(np.take(mesh.devices, [stage], axis=mesh.axis_names.index(STAGE_AXIS)), mesh.axis_names,
                axis_types=mesh.axis_types)


def _array_rows(path, leaf: jax.Array, shape: tuple[int, ...]) -> list[jax.Array]:
    return list(leaf.reshape((-1, *leaf.shape[len(shape):])))


def _sharded(sharding: NamedSharding, mesh: Mesh, spec: Sequence) -> NamedSharding:
    return NamedSharding(mesh, P(*spec), memory_kind=sharding.memory_kind)


@dataclasses.dataclass(frozen=True)
class StageLocal:
    """How a trainer over `mesh` keeps the runs `sites` declare stacked by stage (module docstring);
    with no site, every conversion leaves a tree as it is."""

    sites: tuple[DecoderBank, ...]
    mesh: Mesh
    built: bool = True
    """Whether the run's optimizer is Dew's (`OptimConfig`), whose rules read a stacked leaf as its
    layers, rather than an optax transformation given as it is (`refused`)."""

    @classmethod
    def of(cls, objective, variables, mesh: Mesh, *, built: bool) -> StageLocal:
        """How a trainer over `mesh` keeps `objective`'s pipelined runs, from `variables` (their shapes
        or the arrays), its optimizer Dew's where `built`: no run where the mesh has no stage axis or
        the objective declares none (`Objective.stage_sites`)."""
        stages = mesh.shape.get(STAGE_AXIS, 1)
        shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype), variables)
        declared = objective.stage_sites(shapes, stages) if stages > 1 and variables else ()
        # A run a collection holds in part (a stack frozen by layer index) stays on its stored layers in
        # every collection, which the pipeline restacks as it would any stored layers.
        return cls(tuple(site for site in declared if not any(
            site.view.splits(tree) for tree in in_namespace(shapes, site.namespace).values())), mesh, built)

    @property
    def stages(self) -> int:
        return self.mesh.shape[STAGE_AXIS]

    def staged[Leaf](self, tree, stack: RowStack[Leaf] = stacked_rows):
        """`tree`, any tree that holds or mirrors variables, with each run stacked by stage, its rows
        made one leaf by `stack(rows, shape)` (`StackView.stage_form`): arrays, shapes or shardings. A
        run already stacked stays as it is."""
        return _walked(tree, self.sites, lambda site, where, node: site.view.stage_form(where, node, stack))

    def layered[Leaf](self, tree, cut: RowCut[Leaf] = _array_rows):
        """`staged`'s inverse, `cut(path, leaf, shape)` giving a stacked leaf's rows
        (`StackView.layer_form`)."""
        return _walked(tree, self.sites, lambda site, where, node: site.view.layer_form(node, cut))

    def holds(self, variables) -> bool:
        """Whether `variables` hold their runs stacked by stage."""
        staged: list[bool] = []
        _walked(variables, self.sites, lambda site, where, node: staged.append('stages' in node) or node)
        return any(staged)

    @staticmethod
    def state(state, convert: Callable):
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

    def held(self):
        """The context a step traces in, its optimizer's rules reading each stacked run's leaves as the
        stored layers they stack (`held_by_stage`)."""
        return held_by_stage(self.sites)

    def check(self, layout: Layout, variables, placement, mesh: Mesh) -> None:
        """`layout.check` of `variables` placed as `placement`, read as their stored layers: what it
        bounds is what the layout replicates of a layer."""
        layout.check(self.row_shapes(variables), self.rows_placement(placement), mesh)

    def placed(self, state, placement_of: Callable):
        """The placement of `state` (`Trainer.shardings`): `placement_of`'s of a state as it is, or of
        its stored layers' shapes, its stacked runs' placed on the stage axis (`placement`)."""
        if not self.holds(state.variables):
            return placement_of(state)
        return self.state(placement_of(self.state(state, self.row_shapes)), self.placement)

    def supplied(self, state: TrainState, validated: Callable) -> tuple[TrainState, object]:
        """A state given whole, its runs stacked by stage onto the placement `validated` gives the
        stacked state's shapes, and that placement."""
        self.refused(state.opt_state)
        stack = functools.partial(self.state, convert=self.staged)
        placement = validated(jax.eval_shape(stack, state))
        return (jax.jit(stack, out_shardings=placement)(state) if self.sites
                else jax.device_put(state, placement)), placement

    def restore(self, checkpoints: Checkpoints, template, step: int | str | None,
                share: DataPartition | None = None):
        """`checkpoints.restore` of `template`, a state or some of its fields, each stacked run's
        layers read onto their own stage's devices (`checkpointed`) and stacked there (`restored`)."""
        restored, position = checkpoints.restore(self.state(template, self.checkpointed), step, share=share)
        return self.state(restored, self.restored), position

    def outside(self, state: TrainState) -> TrainState:
        """`state` with its variables and EMA as a model run outside the mesh reads them (`whole`)."""
        return dataclasses.replace(state, variables=self.whole(state.variables),
                                   ema=None if state.ema is None else self.whole(state.ema))

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
                assert isinstance(devices, NamedSharding)
                # A stage's rows are the restore's own arrays, no other leaf's: its piece takes them over.
                piece = _sharded(row, devices.mesh, (None, *[None] * (len(shape) - 1), *row.spec))
                pieces.append(jax.jit(_stacked_piece, static_argnames="within", donate_argnums=0,
                                      out_shardings=piece)(held, within=shape[1:]))
            return jax.make_array_from_single_device_arrays(
                (*shape, *rows[0].shape), self._stacked_sharding(row, len(shape)),
                [shard.data for piece in pieces for shard in piece.addressable_shards])
        return self.staged(tree, stack)

    def whole(self, variables):
        """`variables` with each stacked run back on its stored layers, each layer on every stage, as a
        model run outside the mesh reads them: a preview's decode, a rollout's. It costs what a state
        that held every layer on every device held throughout; variables that hold no stacked run
        are returned as they are."""
        if not self.holds(variables):
            return variables
        placement = self.rows_placement(jax.tree.map(lambda leaf: leaf.sharding, variables))
        return program(_layered_arrays, static_argnums=0, out_shardings=placement)(self, variables)

    def refused(self, opt_state):
        """`opt_state`, once its transforms are known to step a stacked run as its layers: a Dew-built
        optimizer's (`built`) always are; an optax transformation given as it is is refused where its
        state names a transform outside `dew.optim.STACKED_STATES`."""
        if self.sites and not self.built:
            kinds = {type(node) for node in _states(opt_state)}
            unknown = sorted(kind.__name__ for kind in kinds - STACKED_STATES)
            if unknown:
                raise LayoutRefused(
                    f"the optimizer's state holds {unknown}, transforms that may read a parameter whole, "
                    f"and a stage axis holds each pipelined run's layers stacked in one; build the "
                    f"optimizer from an OptimConfig, whose rules read a stacked parameter as its layers, "
                    f"or from transforms that act a number at a time, or train without a stage axis")
        return opt_state


_MIRRORS = ("variables", "opt_state", "ema", "compute")
"""The train-state fields that hold or mirror the variables, which a stage-local state stacks alike."""


def _cut_piece(piece: jax.Array, *, stacked: int) -> tuple[jax.Array, ...]:
    """A stage's piece of a leaf stacked over `stacked` leading dimensions, as its rows."""
    return tuple(piece.reshape((-1, *piece.shape[stacked:])))


def _stacked_piece(rows: list[jax.Array], *, within: tuple[int, ...]) -> jax.Array:
    """A stage's rows stacked into its piece of the leaf, `[1, *within, ...]`."""
    return stacked_rows(rows, (1, *within))


def _layered_arrays(local: StageLocal, tree):
    return local.layered(tree)


def _states(node) -> list:
    """Every optimizer-state node `node` nests, its own included: each container of a type of its own
    (a transform's state, a NamedTuple), past the tuples, lists and dicts that hold them."""
    if node is None:
        return []
    children, structure = jax.tree_util.tree_flatten(node, is_leaf=lambda child: child is not node)
    if jax.tree_util.treedef_is_leaf(structure):
        return []
    own = [] if type(node) in (tuple, list, dict) else [node]
    return own + [state for child in children for state in _states(child)]
