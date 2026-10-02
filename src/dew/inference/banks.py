"""Build and place parameter banks for generation, one bank at a time.

A scanned run of like layers reads its parameters as one array with the layer
axis in front (`dew.nn.backbones.causal_transformer.StackView` adapts it to
the `layers_N` subtrees a checkpoint stores). This builds that banked store
directly: each bank is read, stacked and placed on its own, and its rows are
released before the next is read, so a stack that does not fit in device
memory never exists per layer and banked at once.

`Layout.host_parameters` puts banks in pinned host memory, which the stack
fetches a layer at a time (`run_stack`); everything else is placed as the
trainer places a parameter. A `LayerBanks` source is a run directory
(`CheckpointBanks`), a tree in memory (`HeldBanks`) or local HF safetensors
(`SafetensorsBanks`). A source's `place` puts whole banks on the mesh;
`SafetensorsBanks.stream` leaves decoder weights on disk for the prefetch
loop to read at execution.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.layout import Format, Layout as DeviceLayout
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding
from jax.typing import DTypeLike

from dew import records
from dew.nn.backbones.causal_transformer import DecoderBank
from dew.nn.hyper_connections import Carried
from dew.objectives.base import Variables, merge

if TYPE_CHECKING:
    from dew.training.distributed import Layout, MeshSpec, Placement


class LayerBanks(Protocol):
    """Serves a banked store's values, read one bank at a time.

    Paths are canonical below each collection, and a namespace selects a
    decoder inside a wrapper. What holding one bank costs is the source's
    own business, and is what bounds a load. A source subclasses this for
    `place`, which builds a model's banked store from it.
    """

    def shapes(self) -> Variables:
        """Return the whole store as `jax.ShapeDtypeStruct` leaves, one per layer."""
        ...

    def entry(self, placement: Placement) -> Variables:
        """Read exactly the canonical leaves selected by placement."""
        ...

    def bank(self, layers: Sequence[int], placement: Placement, *,
             namespace: tuple[str, ...] = ()) -> Variables:
        """Return one run's bank on the given shardings or Formats: layers stacked
        on a new leading axis in order, or a single layer's own subtree.
        Formats carry a physical layout which the source must preserve.
        Returned collections are local to namespace."""
        ...

    def place(self, model: BankedModel, *, mesh: MeshSpec | Mesh | None = None,
              layout: Layout | None = None) -> Variables:
        """Build the banked store `model`'s runs read from these weights.

        Each bank's copies are waited for before the next bank is read, so a
        load has one bank's transfers in flight, and costs the store plus what
        this source holds. Nothing here donates or deletes a source's arrays.

        `layout.host_parameters` may name only layers of the stack, which fetches
        a layer as it reaches it; an embedding or head brought over whole would
        cost the device memory it was meant to save. A run whose layers the
        patterns place differently is refused too, since a bank is one array
        with one sharding. Both are refused before anything is read.
        """
        sites, shapes, _device_mesh, placement, entries = _bank_plan(model, self, mesh, layout)

        entry_values = jax.block_until_ready(self.entry(entries)) if entries else {}
        bank_store: dict[str, dict] = {}
        for site in sites:
            for (first, count), name in zip(site.view.groups, site.view.bank_names(), strict=True):
                bank = jax.block_until_ready(self.bank(
                    range(first, first + count),
                    _bank_placement(one_layer(placement, first, namespace=site.namespace),
                                    one_layer(shapes, first, namespace=site.namespace), stacked=count > 1),
                    namespace=site.namespace))
                for collection, tree in bank.items():
                    branch = bank_store.setdefault(collection, {})
                    for component in site.namespace:
                        branch = branch.setdefault(component, {})
                    branch[name] = tree
        return merge(entry_values, bank_store)


def layer_index(name: str) -> int | None:
    """Return the layer index a `layers_N` module name holds, or None."""
    if not name.startswith("layers_"):
        return None
    rest = name[len("layers_"):]
    return int(rest) if rest.isdigit() else None


def one_layer(variables: Variables, index: int, *, namespace: tuple[str, ...] = ()) -> Variables:
    """Return one layer's subtree per collection, local to its decoder namespace."""
    name = f"layers_{index}"
    local = in_namespace(variables, namespace) if namespace else variables
    return {collection: tree[name] for collection, tree in local.items() if name in tree}


def at_layer(subtrees: Variables, index: int, *, namespace: tuple[str, ...] = ()) -> Variables:
    """Place one layer's subtrees back under the canonical module namespace.

    The inverse of `one_layer`.
    """
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
    collected = {}
    for collection, tree in subtrees.items():
        for name in reversed(namespace):
            tree = {name: tree}
        collected[collection] = tree
    return collected


@dataclasses.dataclass(frozen=True)
class HeldBanks(LayerBanks):
    """Serve banks from a variables tree already held in memory.

    The tree is the one `model.init` or a loader produced, one subtree per
    layer. Arrays are borrowed: nothing is donated or deleted, so the caller's
    tree stays usable. A load therefore holds the source and the growing banked
    store at once, which costs two copies of the model. Use `CheckpointBanks`
    for weights that do not fit twice.
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
class CheckpointBanks(LayerBanks):
    """Serve banks by restoring a run's published weights one bank at a time.

    `ema` merges the averaged copy over the live weights, as
    `dew.sampling.pipelines.restore_variables` does, so a bank holds what the
    run publishes. `step` selects a checkpoint and is resolved to the latest
    one when it is built, so every bank of one load comes from one
    checkpoint.

    Rows are restored to device memory, where stacking them computes, and
    the stacked bank goes where the store wants it, so a load stages one
    bank's rows and one bank, never the model.
    """

    directory: str
    step: int | None = None
    ema: bool = False
    stored: Variables = dataclasses.field(init=False, repr=False, compare=False)
    """What the checkpoint at `step` holds, read once, when the source is built."""

    def __post_init__(self):
        from dew.checkpoints import Checkpoints
        checkpoints = Checkpoints(self.directory)
        step = checkpoints.latest if self.step is None else self.step
        if step is None:
            raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "stored", checkpoints.stored(step))

    def shapes(self) -> Variables:
        if self.ema and self.stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; read the live weights with ema=False")
        return self.stored["params"]

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
        """Restore the variables `placement` names onto its shardings."""
        from dew.checkpoints import Checkpoints

        shapes = narrowed(self.shapes(), placement)
        template = {"params": _typed(shapes, narrowed(placement, shapes))}
        if self.ema:
            averaged = narrowed(self.stored["ema"], placement)
            if averaged:
                template["ema"] = _typed(averaged, narrowed(placement, averaged))
        values, _ = Checkpoints(self.directory).restore(template, step=self.step)
        return merge(values["params"], values["ema"]) if "ema" in template else values["params"]


@dataclasses.dataclass(frozen=True)
class DiskBankStats:
    """Cumulative host reads; read time includes layout conversion and can overlap compute."""

    cache_bytes: int
    hits: int
    misses: int
    bytes_read: int
    read_seconds: float
    wait_seconds: float


class SafetensorsBanks(LayerBanks):
    """Read a local HF decoder checkpoint without materializing its layer stack.

    Translation reuses the ordinary decoder's `SourceLeaf` recipes over
    read-only memory maps, including expert stacking, transposition and dtype
    binding. Quantized sources are refused: their codec may dequantize the
    entire checkpoint before translation, which would violate this bound.

    The cache admits complete layers in read order until `cache_bytes` is
    full and retains them until `close`: a sequential decoder visits every
    layer each token, so LRU would evict all of them once the model exceeds
    the cache, while a retained prefix keeps a partial cache useful. One
    read-ahead slot holds the next layer, even with a zero cache. Host weight
    storage is bounded by cache bytes, two rows and one leaf's conversion
    scratch; embeddings, heads and the KV cache stay resident outside that
    bound, and the kernel's page cache is not counted.

    Keep the source open, and its files unmodified, until all executions
    finish; a compiled callback cannot read a closed source.
    """

    def __init__(self, directory: str | Path, *, cache_bytes: int = 0,
                 param_dtype: DTypeLike | Literal["auto"] = "auto", read_ahead: bool = True):
        from dew.interop.hf_decoders import (
            DecoderFamily,
            _check_tree,
            families,
            translate_config,
            translate_weights,
        )
        from dew.interop.safetensors_io import read_weights
        from dew.registry import dtype_name, models, with_precision

        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        folder = Path(directory)
        config = records.record(json.loads((folder / "config.json").read_text()), "config.json")
        self.config: Mapping[str, object] = MappingProxyType(config)
        if self.config.get("quantization_config") or self.config.get("expert_dtype"):
            raise ValueError("disk banks require unquantized safetensors; a whole-model codec is not bounded")
        record = translate_config(self.config)
        family = records.text(self.config.get("model_type"), "model_type")
        if families()[family].prepare_weights is not DecoderFamily.prepare_weights:
            raise ValueError("disk banks require a family with lazy tensor translation; "
                             "this family's preparation can materialize checkpoint weights")
        tensors = read_weights(folder)
        if param_dtype == "auto":
            from dew.interop.pretrained import _checkpoint_dtype
            storage = _checkpoint_dtype(self.config, tensors)
        else:
            storage = dtype_name(param_dtype)
        self._variables: Variables = translate_weights(
            tensors, record, family, param_dtype=storage, lazy=True)
        self._shapes = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), self._variables
        )
        _check_tree(self._shapes, models.build("causal_transformer", with_precision(
            "causal_transformer", record, dtype="float32", attention_impl="reference")))
        self.cache_limit = cache_bytes
        self.read_ahead = read_ahead
        self._cache: dict[tuple[tuple[str, ...], int], Variables] = {}
        self._cache_bytes = self._hits = self._misses = self._bytes_read = 0
        self._read_seconds = self._wait_seconds = 0.0
        self._lock = threading.Lock()
        self._requests = threading.Lock()
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dew-disk-bank")
        self._pending: tuple[tuple[tuple[str, ...], int], Future[Variables]] | None = None
        self._closed = False

    def __enter__(self) -> SafetensorsBanks:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        with self._requests:
            if self._closed:
                return
            self._closed = True
            self._reader.shutdown(wait=True)
            self._pending = None
            with self._lock:
                self._cache.clear()
                self._cache_bytes = 0
            self._variables = {}

    def stats(self) -> DiskBankStats:
        with self._lock:
            return DiskBankStats(self._cache_bytes, self._hits, self._misses, self._bytes_read,
                                 self._read_seconds, self._wait_seconds)

    def shapes(self) -> Variables:
        return self._shapes

    def layer_bytes(self, index: int, *, namespace: tuple[str, ...] = ()) -> int:
        return sum(math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
                   for leaf in jax.tree.leaves(one_layer(self._shapes, index, namespace=namespace)))

    def entry(self, placement: Placement) -> Variables:
        with self._requests:
            self._check_open()
            return self._placed(narrowed(self._variables, placement), placement)

    @staticmethod
    def _placed(variables: Variables, placement: Placement) -> Variables:
        from dew.training.host import place_leaf

        def place(leaf, target):
            if not isinstance(target, Format):
                return place_leaf(leaf, target)
            try:
                return jax.block_until_ready(jax.device_put(leaf.read(), target))
            finally:
                leaf.release()

        return jax.tree.map(place, variables, placement)

    def bank(self, layers: Sequence[int], placement: Placement, *,
             namespace: tuple[str, ...] = ()) -> Variables:
        if len(layers) == 1:
            with self._requests:
                self._check_open()
                row = narrowed(one_layer(self._variables, layers[0], namespace=namespace), placement)
                return self._placed(row, placement)
        rows = [self.read(index, namespace=namespace) for index in layers]
        bank = jax.tree.map(lambda *leaves: np.stack(leaves), *rows)
        return jax.device_put(bank, placement)

    def stream(self, model: BankedModel, *, mesh: MeshSpec | Mesh | None = None,
               layout: Layout | None = None) -> Variables:
        """Bind disk rows to the ordinary banked inference loop on one device.

        Only non-decoder leaves are loaded now. Each declared run receives a
        static handle in the runtime-only `streaming` collection, which the
        stack's two-row prefetch carry calls at execution.

        Single-device inference only: a mesh above one, host placement or an
        unscanned stack is refused before loading weights. The host callback
        needs the CPU backend beside the accelerator (`JAX_PLATFORMS=cuda,cpu`
        when platforms are explicit). Keep this source open until executions complete.
        """
        sites, _shapes, device_mesh, placement, entries = _bank_plan(model, self, mesh, layout)
        if device_mesh.size != 1:
            raise ValueError("disk streaming requires a single-device mesh")
        if any(not site.scanned for site in sites):
            raise ValueError("disk streaming requires scan_layers=True for bounded device prefetch")
        if any(sharding.memory_kind != "device" for sharding in jax.tree.leaves(placement)):
            raise ValueError(
                "disk streaming uses its own bounded host cache; leave layout host placement empty")
        try:
            jax.devices("cpu")
        except RuntimeError as error:
            raise ValueError(
                "disk streaming requires the CPU callback backend; configure "
                "JAX_PLATFORMS=cuda,cpu (or your accelerator,cpu) before JAX initialization"
            ) from error
        device = SingleDeviceSharding(device_mesh.devices.flat[0])
        handles: Variables = {}
        for site in sites:
            groups = site.view.groups
            local = {
                name: {
                    "bank": StreamedBank(
                        self,
                        first,
                        count,
                        site.namespace,
                        device,
                        groups[index + 1][0] if index + 1 < len(groups) else groups[0][0],
                    )
                }
                for index, ((first, count), name)
                in enumerate(zip(groups, site.view.bank_names(), strict=True))
            }
            handles = merge(handles, at_namespace({"streaming": local}, site.namespace))
        entries = jax.block_until_ready(self.entry(entries)) if entries else {}
        return merge(entries, handles)

    def read(self, index: int, *, namespace: tuple[str, ...] = (),
             following: int | None = None) -> Variables:
        """Read one row and start at most one following row before returning.

        Reads serialize across executions sharing this source. Read-ahead
        is joined before a different request, so concurrent calls cannot
        accumulate an unbounded queue of rows.
        """
        key = (namespace, index)
        with self._requests:
            self._check_open()
            started = time.perf_counter()
            pending, self._pending = self._pending, None
            if pending is not None:
                row = pending[1].result()
                if pending[0] != key:
                    del row
                    row = self._read(key)
            else:
                row = self._read(key)
            with self._lock:
                self._wait_seconds += time.perf_counter() - started
            if self.read_ahead and following is not None:
                next_key = (namespace, following)
                self._pending = (next_key, self._reader.submit(self._read, next_key))
            return row

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the safetensors bank source is closed")

    def _read(self, key: tuple[tuple[str, ...], int]) -> Variables:
        from dew.interop.streaming import SourceLeaf
        from dew.training.host import evict

        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
        started = time.perf_counter()
        namespace, index = key
        stored = one_layer(self._variables, index, namespace=namespace)
        if not stored:
            raise ValueError(f"no layer {index} at namespace {namespace}")

        def read_leaf(leaf):
            try:
                values = leaf.read() if isinstance(leaf, SourceLeaf) else np.asarray(leaf)
                # A transposed, cast or stacked SourceLeaf already owns its
                # C-ordered allocation. Only a borrowed mapped view needs a
                # private copy before its file pages are released.
                if not values.flags.owndata or not values.flags.c_contiguous:
                    values = np.array(values, copy=True, order="C")
                values.setflags(write=False)
                return values
            finally:
                leaf.release() if isinstance(leaf, SourceLeaf) else evict(leaf)

        with jax.profiler.TraceAnnotation("SSD bank read", layer=index):
            row = jax.tree.map(read_leaf, stored)
        size = sum(leaf.nbytes for leaf in jax.tree.leaves(row))
        with self._lock:
            self._misses += 1
            self._bytes_read += size
            self._read_seconds += time.perf_counter() - started
            if self._cache_bytes + size <= self.cache_limit:
                self._cache[key] = row
                self._cache_bytes += size
        return row


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, eq=False)
class StreamedBank:
    """A runtime-only bank handle: no parameter buffer or serialized model field.

    Its identity is static in the variables pytree; changing the source
    retraces, changing tokens does not. The callback is effectful, so it is
    neither folded at trace time nor elided when an executable is reused.
    """

    source: SafetensorsBanks
    first: int
    count: int
    namespace: tuple[str, ...]
    device: SingleDeviceSharding
    following: int | None

    def shapes(self) -> Variables:
        return one_layer(self.source.shapes(), self.first, namespace=self.namespace)

    def fetch(self, index, dependency) -> Variables:
        from jax.experimental import io_callback

        def read(offset, _dependency):
            position = self.first + int(offset)
            following = position + 1 if int(offset) + 1 < self.count else self.following
            return self.source.read(position, namespace=self.namespace, following=following)

        # Only a scalar is copied to the host. Its data dependence orders a
        # fetch after the preceding layer, while allowing this layer's
        # compute to overlap the next host read and device transfer.
        hidden = dependency.streams if isinstance(dependency, Carried) else dependency
        with jax.named_scope("ssd_bank_fetch"):
            return io_callback(read, self.shapes(), index, hidden.reshape(-1)[0],
                               sharding=self.device, ordered=True)


def narrowed(tree: Mapping, selection: Mapping) -> dict:
    """Return the leaf-level intersection of `tree` and `selection`, paths kept."""
    collected = {}
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
        collected[name] = value
    return collected


def _typed(shapes: Variables, placement: Placement) -> Variables:
    return jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        shapes, placement)


def _row_placement(placement: Placement) -> Placement:
    """Return one layer's shardings out of a bank's: the layer axis dropped, in device
    memory, which is where the rows a bank is stacked out of are read."""
    def row(target):
        sharding = target.sharding if isinstance(target, Format) else target
        assert isinstance(sharding, NamedSharding)
        return NamedSharding(sharding.mesh, P(*sharding.spec[1:]), memory_kind="device")

    return jax.tree.map(row, placement)


def _bank_placement(placement: Placement, shapes: Variables, stacked: bool) -> Placement:
    """A bank's logical placement and, on a TPU host, layer-major storage."""
    if not stacked:
        return placement

    def place(sharding, shape):
        bank = NamedSharding(sharding.mesh, P(None, *sharding.spec), memory_kind=sharding.memory_kind)
        if bank.memory_kind == "pinned_host" and next(iter(bank.device_set)).platform == "tpu":
            # A minor layer axis can change TPU tile shape when sliced,
            # making the host-to-device copy halt even before its squeeze.
            return Format(DeviceLayout(major_to_minor=tuple(range(len(shape.shape) + 1))), bank)
        return bank

    return jax.tree.map(place, placement, shapes)


class BankedModel(Protocol):
    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]: ...


def bank_sites(model: BankedModel) -> tuple[DecoderBank, ...]:
    """Return the model's declared stacks, deduplicated by canonical namespace.

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
    """Return the leaf complement of declared decoder layers and their runs' banks,
    including nested media."""
    layers = {(*site.namespace, f"layers_{index}") for site in sites
              for first, count in site.view.groups for index in range(first, first + count)}
    layers.update((*site.namespace, name) for site in sites for name in site.view.bank_names())

    def outside(tree: Mapping, path: tuple[str, ...]) -> dict:
        collected = {}
        for name, value in tree.items():
            current = (*path, name)
            if current in layers:
                continue
            if isinstance(value, Mapping):
                value = outside(value, current)
                if not value:
                    continue
            collected[name] = value
        return collected

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

def _bank_plan(model: BankedModel, source: LayerBanks, mesh: MeshSpec | Mesh | None,
               layout: Layout | None):
    """Validate canonical ownership and placement before either loader reads a value."""
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh

    sites = bank_sites(model)
    shapes = source.shapes()
    for site in sites:
        _check_shapes(shapes, site)
    device_mesh = mesh if isinstance(mesh, Mesh) else (DefaultMesh() if mesh is None else mesh).build()
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

    return sites, shapes, device_mesh, placement, entries


def _places(subtree) -> dict[str, tuple[str, str]]:
    """Return where each leaf of one layer goes, by its path inside the layer: the
    memory kind and the spec, which is what a bank has to hold in common."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(subtree)
    return {jax.tree_util.keystr(path): (str(sharding.memory_kind), str(sharding.spec))
            for path, sharding in leaves}


def _check_consumers(placement: Placement, groups: Sequence[tuple[int, int]]) -> None:
    """Require corresponding leaves of one bank's layers to agree on memory
    kind and spec, since a bank stacks them into one array with one sharding."""

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
