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
import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.backbones.causal_transformer import DecoderBank
from dew.objectives.base import Variables, merge

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec, Placement


class LayerBanks(Protocol):
    """Where a banked store's values come from, read one bank at a time.

    Paths are canonical below each collection. A namespace selects a decoder
    inside a wrapper; the source knows nothing about how runs are grouped
    and answers for the layers it is asked for. `bank` stacks them on a new
    leading axis in the order given and places the result, so how much
    memory holding one bank costs is the source's own business, and is what
    bounds a load.
    """

    def shapes(self) -> Variables:
        """The whole store as `jax.ShapeDtypeStruct` leaves, one per layer."""
        ...

    def entry(self, placement: Placement) -> Variables:
        """Read exactly the canonical leaves selected by placement."""
        ...

    def bank(self, layers: Sequence[int], placement: Placement, *,
             namespace: tuple[str, ...] = ()) -> Variables:
        """One run's bank, on those shardings: those layers stacked on a new
        leading axis in the order given, or, for a run of one layer, that
        layer's own subtree. Returned collections are local to namespace."""
        ...


def layer_index(name: str) -> int | None:
    """`layers_3` as 3; anything that is not one layer of the stack as None."""
    if not name.startswith("layers_"):
        return None
    rest = name[len("layers_"):]
    return int(rest) if rest.isdigit() else None






def one_layer(variables: Variables, index: int, *, namespace: tuple[str, ...] = ()) -> Variables:
    """One layer's subtree per collection, local to its decoder namespace."""
    name = f"layers_{index}"
    local = in_namespace(variables, namespace) if namespace else variables
    return {collection: tree[name] for collection, tree in local.items() if name in tree}


def at_layer(subtrees: Variables, index: int, *, namespace: tuple[str, ...] = ()) -> Variables:
    """The inverse of one_layer, preserving the canonical module namespace."""
    return at_namespace({collection: {f"layers_{index}": tree}
                         for collection, tree in subtrees.items()}, namespace)


def in_namespace(variables: Variables, namespace: tuple[str, ...]) -> Variables:
    """Read the same module namespace below each variables collection."""
    selected = {}
    for collection, tree in variables.items():
        for name in namespace:
            if not isinstance(tree, Mapping):
                raise ValueError(f"{collection}/{namespace} crosses a variable leaf")
            if name not in tree:
                break
            tree = tree[name]
        else:
            if not isinstance(tree, Mapping):
                raise ValueError(f"{collection}/{namespace} is not a module subtree")
            if tree:
                selected[collection] = tree
    return selected


def at_namespace(subtrees: Variables, namespace: tuple[str, ...]) -> Variables:
    """Place collection-local subtrees back under their canonical namespace."""
    result = {}
    for collection, tree in subtrees.items():
        for name in reversed(namespace):
            tree = {name: tree}
        result[collection] = tree
    return result



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

    def entry(self, placement: Placement) -> Variables:
        return jax.device_put(narrowed(self.variables, placement), placement)

    def bank(self, layers: Sequence[int], placement: Placement, *,
             namespace: tuple[str, ...] = ()) -> Variables:
        rows = [one_layer(self.variables, index, namespace=namespace) for index in layers]
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
        assert self.step is not None
        stored = _stored(self.directory, self.step)
        if self.ema and stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; read the live weights with ema=False")
        return stored["params"]

    def entry(self, placement: Placement) -> Variables:
        return self._restored(placement)

    def bank(self, layers: Sequence[int], placement: Placement, *,
             namespace: tuple[str, ...] = ()) -> Variables:
        if len(layers) == 1:
            return one_layer(self._restored(at_layer(placement, layers[0], namespace=namespace)),
                             layers[0], namespace=namespace)
        rows = [one_layer(self._restored(at_layer(_row_placement(placement), index, namespace=namespace)),
                          index, namespace=namespace) for index in layers]
        stacked = jax.jit(lambda *held: jax.tree.map(lambda *leaves: jnp.stack(leaves), *held),
                          out_shardings=placement)
        return stacked(*rows)

    def _restored(self, placement: Placement) -> Variables:
        """The variables `placement` names, restored onto its shardings."""
        from dew.checkpoints import Checkpoints

        shapes = narrowed(self.shapes(), placement)
        template = {"params": _typed(shapes, narrowed(placement, shapes))}
        if self.ema:
            assert self.step is not None
            averaged = narrowed(_stored(self.directory, self.step)["ema"], placement)
            if averaged:
                template["ema"] = _typed(averaged, narrowed(placement, averaged))
        values, _ = Checkpoints(self.directory).restore(template, step=self.step)
        return merge(values["params"], values["ema"]) if "ema" in template else values["params"]


@functools.cache
def _stored(directory: str, step: int) -> Variables:
    from dew.checkpoints import Checkpoints
    return Checkpoints(directory).stored(step)


def narrowed(tree: Mapping, selection: Mapping) -> dict:
    """The leaf-level intersection of tree and selection, retaining canonical paths."""
    result = {}
    for name, selected in selection.items():
        if name not in tree:
            continue
        value = tree[name]
        if isinstance(selected, Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"selection descends through leaf {name!r}")
            value = narrowed(value, selected)
            if not value:
                continue
        elif isinstance(value, Mapping):
            raise ValueError(f"selection treats subtree {name!r} as a leaf")
        result[name] = value
    return result


def _typed(shapes: Variables, placement: Placement) -> Variables:
    return jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        shapes, placement)


def _row_placement(placement: Placement) -> Placement:
    """One layer's shardings out of a bank's: the layer axis dropped, in device
    memory, which is where the rows a bank is stacked out of are read."""
    return jax.tree.map(
        lambda sharding: NamedSharding(sharding.mesh, P(*sharding.spec[1:]),
                                       memory_kind="device"), placement)


def _bank_placement(placement: Placement, stacked: bool) -> Placement:
    """A run's shardings: the layer axis whole in front of each leaf's own
    spec, in the memory space the layout chose for that leaf."""
    if not stacked:
        return placement
    return jax.tree.map(
        lambda sharding: NamedSharding(sharding.mesh, P(None, *sharding.spec),
                                       memory_kind=sharding.memory_kind), placement)



class BankedModel(Protocol):
    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]: ...


def bank_sites(model: BankedModel) -> tuple[DecoderBank, ...]:
    """The model's declared stacks, deduplicated by canonical namespace.

    Two readers of one physical scope declare the same site, so a shared
    decoder is packed once. Conflicting views of a namespace, overlapping
    layer ownership and staged or already banked views are refused here,
    before any value is read.
    """
    owners: dict[tuple[str, ...], DecoderBank] = {}
    for site in model.bank_sites:
        if any(not isinstance(name, str) or not name or "/" in name for name in site.namespace):
            raise ValueError("decoder namespaces must contain nonempty module names")
        previous = owners.get(site.namespace)
        if previous is not None and previous.view != site.view:
            raise ValueError(f"conflicting decoder views at namespace {site.namespace}")
        owners[site.namespace] = site
    if not owners:
        raise ValueError("a banked store requires a declared decoder bank owner")
    paths = []
    for site in owners.values():
        if site.view.stages != 1 or site.view.banked:
            raise ValueError("bank sources require a logical, unstaged StackView")
        for first, count in site.view.groups:
            if first < 0 or count < 1:
                raise ValueError("decoder bank groups require nonnegative starts and positive lengths")
            paths.extend((*site.namespace, f"layers_{index}") for index in range(first, first + count))
    ordered = sorted(paths)
    if any(right[:len(left)] == left for left, right in itertools.pairwise(ordered)):
        raise ValueError("declared decoder banks overlap in their canonical layer ownership")
    return tuple(owners.values())


def entry_tree(variables: Variables, sites: Sequence[DecoderBank]) -> Variables:
    """The leaf complement of declared decoder layers, including nested media."""
    layers = {(*site.namespace, f"layers_{index}") for site in sites
              for first, count in site.view.groups for index in range(first, first + count)}

    def outside(tree: Mapping, path: tuple[str, ...]) -> dict:
        result = {}
        for name, value in tree.items():
            current = (*path, name)
            if current in layers:
                continue
            if isinstance(value, Mapping):
                value = outside(value, current)
                if not value:
                    continue
            result[name] = value
        return result

    return {collection: held for collection, tree in variables.items()
            if (held := outside(tree, ()))}


def _check_shapes(shapes: Variables, site: DecoderBank) -> None:
    local = in_namespace(shapes, site.namespace)
    expected = {index for first, count in site.view.groups for index in range(first, first + count)}
    observed = {index for tree in local.values() for name in tree
                if (index := layer_index(name)) is not None}
    if observed != expected:
        raise ValueError(f"namespace {site.namespace} holds layers {sorted(observed)}; "
                         f"the decoder declares {sorted(expected)}")

    def signature(row):
        return {jax.tree_util.keystr(path): (leaf.shape, str(leaf.dtype))
                for path, leaf in jax.tree_util.tree_flatten_with_path(row)[0]}

    for first, count in site.view.groups:
        reference = signature(one_layer(local, first))
        for index in range(first + 1, first + count):
            if signature(one_layer(local, index)) != reference:
                raise ValueError(f"namespace {site.namespace} layers {first} and {index} "
                                 "must have identical leaf paths, shapes and dtypes within one bank")

def host_banked(model: BankedModel, source: LayerBanks, *,
                mesh: MeshSpec | None = None, layout: Layout | None = None) -> Variables:
    """`source`'s weights as the banked store `model`'s runs read.

    Each run's bank is read, stacked and placed on its own, and the copies of
    one bank are waited for before the next bank is read, so the transfers a
    load has in flight are one bank's and not the store's.
    Each declared StackView bounds the run length. What the source holds
    while it answers is the source's
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
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh

    sites = bank_sites(model)
    shapes = source.shapes()
    for site in sites:
        _check_shapes(shapes, site)
    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen = DefaultLayout() if layout is None else layout
    placement = chosen.offloaded(device_mesh, shapes)
    chosen.check(shapes["params"], placement["params"], device_mesh)
    entries = entry_tree(placement, sites)
    outside = [jax.tree_util.keystr(path) for path, sharding in
               jax.tree_util.tree_flatten_with_path(entries)[0] if sharding.memory_kind == "pinned_host"]
    if outside:
        raise ValueError(f"host_parameters selected {outside}, which the layer stack does not fetch; "
                         "select declared decoder layers and keep embeddings, heads and media resident")
    for site in sites:
        _check_consumers(in_namespace(placement, site.namespace), site.view.groups)

    entry_values = jax.block_until_ready(source.entry(entries)) if entries else {}
    bank_store: dict[str, dict] = {}
    for site in sites:
        for (first, count), name in zip(site.view.groups, site.view.bank_names(), strict=True):
            bank = jax.block_until_ready(source.bank(
                range(first, first + count),
                _bank_placement(one_layer(placement, first, namespace=site.namespace), stacked=count > 1),
                namespace=site.namespace))
            for collection, tree in bank.items():
                branch = bank_store.setdefault(collection, {})
                for component in site.namespace:
                    branch = branch.setdefault(component, {})
                branch[name] = tree
    return merge(entry_values, bank_store)


def _places(subtree) -> dict[str, tuple[str, str]]:
    """Where each leaf of one layer goes, by its path inside the layer: the
    memory kind and the spec, which is what a bank has to hold in common."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(subtree)
    return {jax.tree_util.keystr(path): (str(sharding.memory_kind), str(sharding.spec))
            for path, sharding in leaves}


def _check_consumers(placement: Placement, groups: Sequence[tuple[int, int]]) -> None:
    """Require corresponding layers of one bank to agree on placement.

    The comparison is per path inside a layer, not over the set of memory
    kinds a layer uses: two layers can use the same two spaces for different
    leaves, and a bank stacks corresponding leaves, so it is the
    correspondence that has to hold. The spec is compared beside the memory
    kind, because a bank is one array and one sharding for the run.
    """

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
