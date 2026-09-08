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
(`CheckpointBanks`), a tree already in memory (`HeldBanks`) and generated
weights (`SyntheticBanks`) are the three that ship, and `host_banked` loads
all three the same way.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import zlib
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
    layer. Stacking a bank copies those layers, so this holds the whole model
    and one bank of it: for weights that fit, which is what a test and a
    fixture have.
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


@dataclasses.dataclass(frozen=True)
class SyntheticBanks:
    """Generated weights of a given shape, one bank at a time.

    Every element of every leaf of every layer gets its own value: the index
    of the element inside its leaf, and a key from the leaf's path and the
    layer's depth, go through murmur3's finalizer, which is a bijection on
    32-bit words, so no two elements of a leaf and no two layers of a bank
    hold the same numbers, and nothing is broadcast or aliased. The scale is
    a fan-in init's and a norm's scale sits around one, so a stack this deep
    still returns finite logits. The key comes from a CRC of the path, not
    from `hash`, so two processes generate the same weights.

    A bank is filled row by row into the array that is then placed, so what a
    load holds past the store is one bank and one leaf of one layer.
    """

    held: Variables
    seed: int = 0

    def shapes(self) -> Variables:
        return self.held

    def entry(self, placement: "Placement") -> Variables:
        held = named(self.held, entry_names(self.held))
        return jax.device_put(
            jax.tree_util.tree_map_with_path(
                lambda path, leaf: self._values(path, None, leaf.shape, leaf.dtype), held),
            narrowed(placement, held))

    def bank(self, layers: Sequence[int], placement: "Placement") -> Variables:
        def rows(path, leaf):
            if len(layers) == 1:
                return self._values(path, layers[0], leaf.shape, leaf.dtype)
            bank = np.empty((len(layers),) + leaf.shape, np.dtype(leaf.dtype))
            for offset, index in enumerate(layers):
                bank[offset] = self._values(path, index, leaf.shape, leaf.dtype)
            return bank

        return jax.device_put(
            jax.tree_util.tree_map_with_path(rows, one_layer(self.held, layers[0])),
            placement)


    def _values(self, path, layer: int | None, shape: tuple[int, ...], dtype) -> np.ndarray:
        name = jax.tree_util.keystr(path)
        key = np.uint32(zlib.crc32(f"{name}/{layer}/{self.seed}".encode()) | 1)
        count = math.prod(shape) if shape else 1
        word = np.arange(count, dtype=np.uint32) + key
        word ^= word >> np.uint32(16)
        word *= np.uint32(0x85EBCA6B)
        word ^= word >> np.uint32(13)
        word *= np.uint32(0xC2B2AE35)
        word ^= word >> np.uint32(16)
        unit = (word >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -23) - 1.0
        if name.endswith("['scale']"):
            values = 1.0 + 0.02 * unit
        else:
            fan_in = shape[-2] if len(shape) >= 2 else max(count, 1)
            values = unit * np.float32(1.0 / math.sqrt(fan_in))
        return values.reshape(shape).astype(np.dtype(dtype))


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

    Each run's bank is read, stacked and placed on its own and the rows it
    came from are released before the next run is read, so what a load holds
    past the store is one bank's worth. `model.bank_layers` caps how long a
    run is, which is what makes that bound a choice.

    `layout.host_parameters` names the parameters kept in pinned host memory.
    Only the layers of the stack can be: the stack is what fetches a layer's
    parameters as it reaches it, and an embedding table or a head brought
    over in one piece would cost the device memory it was meant to save, so
    naming one is refused here, before anything is allocated. A run whose
    layers the patterns disagree about is refused for the same reason: a bank
    is one array and sits in one memory space.
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
    for collection, tree in source.entry(placement).items():
        store[collection].update(tree)
    for (first, count), name in zip(groups, StackView(groups).bank_names()):
        bank = source.bank(
            range(first, first + count),
            _bank_placement(one_layer(placement, first), stacked=count > 1))
        for collection, tree in bank.items():
            store[collection][name] = tree
        del bank
    return {collection: tree for collection, tree in store.items() if tree}


def _check_consumers(placement: "Placement", groups: Sequence[tuple[int, int]]) -> None:
    """Refuse host-resident variables nothing fetches, and a run whose layers
    disagree about where they sit."""
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
            kinds = {
                f"{collection}/layers_{index}": tuple(sorted(
                    {sharding.memory_kind
                     for sharding in jax.tree.leaves(tree[f"layers_{index}"])}))
                for index in range(first, first + count) if f"layers_{index}" in tree}
            if len(set(kinds.values())) > 1:
                raise ValueError(
                    f"layers {first} to {first + count - 1} are one run, so one bank "
                    f"in one memory space, and host_parameters puts them in {kinds}. "
                    f"Select whole runs, or set bank_layers so the runs follow the "
                    f"selection")
