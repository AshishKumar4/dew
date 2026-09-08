"""Parameter banks for generation, built and placed one bank at a time.

A scanned run of like layers reads its parameters as one array with the layer
axis in front, and `dew.nn.backbones.causal_transformer.StackView` is the
adapter between that and the `layers_N` subtrees a checkpoint stores. This
builds the banked store directly, so a stack whose weights do not fit in
device memory never exists per layer and banked at once: each bank is read,
stacked and placed on its own, and the rows it came from are released before
the next bank is read.

`Layout.host_parameters` decides which banks land in pinned host memory; the
stack fetches those one layer at a time as it reaches them (`run_stack`).
Everything else is placed exactly as the trainer places a parameter, on the
same mesh under the same rules.

A source is a `LayerBanks`: the shapes it can produce, the variables outside
the layer stack, and one bank of consecutive layers. A run directory
(`CheckpointBanks`) and a tree already in memory (`HeldBanks`) are the two
that ship, and `host_banked` loads either the same way. `CheckpointBanks`
reads Dew's own run checkpoints; a source that streams a published
checkpoint too large to hold is not implemented here.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.backbones.causal_transformer import CausalTransformer, StackView
from dew.objectives.base import Variables, merge

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec, Placement


class LayerBanks(Protocol):
    """Where a banked store's values come from, read one bank at a time.

    The paths are the stored ones, `params/layers_N/...`: a source knows
    nothing about how the runs are grouped, which is the model's business,
    and answers for the layers it is asked for. `bank` stacks them on a new
    leading axis in the order given and places the result, so how much
    memory holding one bank costs is the source's own business, and is what
    bounds a load.
    """

    def shapes(self) -> Variables:
        """The whole store as `jax.ShapeDtypeStruct` leaves, one per layer."""
        ...

    def entry(self, placement: "Placement") -> Variables:
        """The variables outside the layer stack, on those shardings."""
        ...

    def bank(self, layers: Sequence[int], placement: "Placement") -> Variables:
        """One run's bank, on those shardings: those layers stacked on a new
        leading axis in the order given, or, for a run of one layer, that
        layer's own subtree, which is what a run of one is stored as."""
        ...


def layer_index(name: str) -> int | None:
    """`layers_3` as 3; anything that is not one layer of the stack as None."""
    if not name.startswith("layers_"):
        return None
    rest = name[len("layers_"):]
    return int(rest) if rest.isdigit() else None


def stack_depth(shapes: Variables) -> int:
    """How many layers the stack in `shapes` has."""
    return sum(1 for name in shapes.get("params", {}) if layer_index(name) is not None)


def entry_names(shapes: Variables) -> tuple[str, ...]:
    """Everything in the store that is not one layer of the stack, in order."""
    names: list[str] = []
    for tree in shapes.values():
        names.extend(name for name in tree if layer_index(name) is None)
    return tuple(dict.fromkeys(names))


def named(variables: Variables, names: Sequence[str]) -> Variables:
    """The collections of `variables` narrowed to `names`, empty ones dropped."""
    narrowed = {}
    for collection, tree in variables.items():
        held = {name: tree[name] for name in names if name in tree}
        if held:
            narrowed[collection] = held
    return narrowed


def one_layer(variables: Variables, index: int) -> Variables:
    """One layer's subtree per collection, under the collection alone."""
    name = f"layers_{index}"
    return {collection: tree[name] for collection, tree in variables.items() if name in tree}


def at_layer(subtrees: Variables, index: int) -> Variables:
    """The inverse of `one_layer`: those subtrees under that layer's name."""
    return {collection: {f"layers_{index}": tree} for collection, tree in subtrees.items()}


@dataclasses.dataclass(frozen=True)
class HeldBanks:
    """A store already in memory, banked as it is read.

    The tree is the one `model.init` or a loader produced, one subtree per
    layer. It is borrowed: every array stays live and unchanged, nothing is
    donated or deleted, and the caller's tree is as usable afterwards as
    before. So a load through this holds the whole source *and* the growing
    banked destination, and by the last bank it holds two copies of the
    model. It is for weights that fit twice, which is what a test and a
    fixture have; it is not evidence that a store larger than memory can be
    built, and `CheckpointBanks` is what reads one that does not fit.
    """

    variables: Variables

    def shapes(self) -> Variables:
        return jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.result_type(leaf)),
            self.variables)

    def entry(self, placement: "Placement") -> Variables:
        held = named(self.variables, entry_names(self.variables))
        return jax.device_put(held, narrowed(placement, held))

    def bank(self, layers: Sequence[int], placement: "Placement") -> Variables:
        rows = [one_layer(self.variables, index) for index in layers]
        bank = rows[0] if len(rows) == 1 else jax.tree.map(
            lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]), *rows)
        return jax.device_put(bank, placement)


@dataclasses.dataclass(frozen=True)
class CheckpointBanks:
    """A run's published weights, read one bank at a time.

    `ema` merges the averaged copy over the live weights, as
    `dew.sampling.pipelines.restore_variables` does, so a bank holds what the
    run publishes. `step` selects a checkpoint and is resolved to the latest
    one when it is built, so every bank of one load comes from one
    checkpoint.

    The rows are restored onto the placement the bank asks for with its layer
    axis dropped and its memory kind set to device, because stacking them is
    computation and computation reads device memory; the bank the computation
    writes goes where the store wants it. A load stages one bank's rows and
    one bank, and never the model.
    """

    directory: str
    step: int | None = None
    ema: bool = False

    def __post_init__(self):
        from dew.checkpoints import Checkpoints
        if self.step is None:
            latest = Checkpoints(self.directory).latest
            if latest is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
            object.__setattr__(self, "step", latest)

    def shapes(self) -> Variables:
        stored = _stored(self.directory, self.step)
        if self.ema and stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; read the live weights with ema=False")
        return stored["params"]

    def entry(self, placement: "Placement") -> Variables:
        return self._restored(
            narrowed(placement, named(self.shapes(), entry_names(self.shapes()))))

    def bank(self, layers: Sequence[int], placement: "Placement") -> Variables:
        if len(layers) == 1:
            return one_layer(self._restored(at_layer(placement, layers[0])), layers[0])
        rows = [one_layer(self._restored(at_layer(_row_placement(placement), index)), index)
                for index in layers]
        stacked = jax.jit(lambda *held: jax.tree.map(lambda *leaves: jnp.stack(leaves), *held),
                          out_shardings=placement)
        return stacked(*rows)

    def _restored(self, placement: "Placement") -> Variables:
        """The variables `placement` names, restored onto its shardings."""
        from dew.checkpoints import Checkpoints

        names = [name for tree in placement.values() for name in tree]
        shapes = named(self.shapes(), names)
        template = {"params": _typed(shapes, narrowed(placement, shapes))}
        if self.ema:
            averaged = named(_stored(self.directory, self.step)["ema"], names)
            template["ema"] = _typed(averaged, narrowed(placement, averaged))
        values, _ = Checkpoints(self.directory).restore(template, step=self.step)
        return merge(values["params"], values["ema"]) if self.ema else values["params"]


@functools.lru_cache(maxsize=None)
def _stored(directory: str, step: int) -> Variables:
    from dew.checkpoints import Checkpoints
    return Checkpoints(directory).stored(step)


def narrowed(placement: "Placement", shapes: Variables) -> "Placement":
    """`placement` cut down to the paths `shapes` holds."""
    return {collection: {name: placement[collection][name] for name in tree}
            for collection, tree in shapes.items()}


def _typed(shapes: Variables, placement: "Placement") -> Variables:
    return jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        shapes, placement)


def _row_placement(placement: "Placement") -> "Placement":
    """One layer's shardings out of a bank's: the layer axis dropped, in device
    memory, which is where the rows a bank is stacked out of are read."""
    return jax.tree.map(
        lambda sharding: NamedSharding(sharding.mesh, P(*sharding.spec[1:]),
                                       memory_kind="device"), placement)


def _bank_placement(placement: "Placement", stacked: bool) -> "Placement":
    """A run's shardings: the layer axis whole in front of each leaf's own
    spec, in the memory space the layout chose for that leaf."""
    if not stacked:
        return placement
    return jax.tree.map(
        lambda sharding: NamedSharding(sharding.mesh, P(None, *sharding.spec),
                                       memory_kind=sharding.memory_kind), placement)


def host_banked(model: CausalTransformer, source: LayerBanks, *,
                mesh: "MeshSpec | None" = None, layout: "Layout | None" = None) -> Variables:
    """`source`'s weights as the banked store `model`'s runs read.

    Each run's bank is read, stacked and placed on its own, and the copies of
    one bank are waited for before the next bank is read, so the transfers a
    load has in flight are one bank's and not the store's.
    `model.bank_layers` caps how long a run is, which is what makes that
    bound a choice. What the *source* holds while it answers is the source's
    contract, not this one: `HeldBanks` holds the whole tree it borrowed,
    `CheckpointBanks` stages one bank's rows, and a load of either costs the
    store plus whatever its source holds. Nothing here donates or deletes a
    source's arrays.

    `layout.host_parameters` names the parameters kept in pinned host memory.
    Only the layers of the stack can be: the stack is what fetches a layer's
    parameters as it reaches it, and an embedding table or a head brought
    over in one piece would cost the device memory it was meant to save, so
    naming one is refused here, before anything is read. A run whose layers
    the patterns place differently, leaf for leaf, is refused for the same
    reason: a bank is one array with one sharding.
    """
    from dew.training.distributed import (
        Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh)

    if not model.scan_layers:
        raise ValueError(
            "a banked store holds one array per scanned run, which only scan_layers "
            "groups; build the model with scan_layers=True")
    groups = model.bind({}).groups
    shapes = source.shapes()
    depth = sum(count for _, count in groups)
    if stack_depth(shapes) != depth:
        raise ValueError(
            f"the source holds {stack_depth(shapes)} layers and this model's runs "
            f"hold {depth}; a store is banked for the model that reads it")
    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen = DefaultLayout() if layout is None else layout
    placement = chosen.offloaded(device_mesh, shapes)
    chosen.check(shapes["params"], placement["params"], device_mesh)
    _check_consumers(placement, groups)

    store: dict[str, dict] = {collection: {} for collection in shapes}
    entry = jax.block_until_ready(source.entry(placement))
    for collection, tree in entry.items():
        store[collection].update(tree)
    for (first, count), name in zip(groups, StackView(groups).bank_names()):
        bank = jax.block_until_ready(source.bank(
            range(first, first + count),
            _bank_placement(one_layer(placement, first), stacked=count > 1)))
        for collection, tree in bank.items():
            store[collection][name] = tree
    return {collection: tree for collection, tree in store.items() if tree}


def _places(subtree) -> dict[str, tuple[str, str]]:
    """Where each leaf of one layer goes, by its path inside the layer: the
    memory kind and the spec, which is what a bank has to hold in common."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(subtree)
    return {jax.tree_util.keystr(path): (str(sharding.memory_kind), str(sharding.spec))
            for path, sharding in leaves}


def _check_consumers(placement: "Placement", groups: Sequence[tuple[int, int]]) -> None:
    """Refuse host-resident variables nothing fetches, and a run whose layers
    do not agree leaf for leaf about where they go.

    The comparison is per path inside a layer, not over the set of memory
    kinds a layer uses: two layers can use the same two spaces for different
    leaves, and a bank stacks corresponding leaves, so it is the
    correspondence that has to hold. The spec is compared beside the memory
    kind, because a bank is one array and one sharding for the run.
    """
    for collection, tree in placement.items():
        outside = sorted(
            name for name, subtree in tree.items()
            if layer_index(name) is None
            and any(sharding.memory_kind == "pinned_host"
                    for sharding in jax.tree.leaves(subtree)))
        if outside:
            raise ValueError(
                f"host_parameters selected {collection}/{outside}, which the layer "
                f"stack does not fetch: only a run of layers is read one layer at a "
                f"time, and an embedding table, a head or a prediction depth would "
                f"come over whole and cost the device memory the offload saves. "
                f"Select layers_* paths and keep a tied head resident")
    for collection, tree in placement.items():
        for first, count in groups:
            held = [index for index in range(first, first + count)
                    if f"layers_{index}" in tree]
            if len(held) < 2:
                continue
            reference = _places(tree[f"layers_{held[0]}"])
            for index in held[1:]:
                places = _places(tree[f"layers_{index}"])
                differing = sorted(
                    path for path in set(reference) | set(places)
                    if reference.get(path) != places.get(path))
                if differing:
                    first_path = differing[0]
                    raise ValueError(
                        f"layers {first} to {first + count - 1} are one run, so one "
                        f"bank per leaf and one memory space and spec per bank, and "
                        f"{collection} layer {held[0]} and layer {index} disagree "
                        f"about {len(differing)} of them. {first_path} is "
                        f"{reference.get(first_path)} in layer {held[0]} and "
                        f"{places.get(first_path)} in layer {index}. Select whole "
                        f"runs, or set bank_layers so the runs follow the selection")
