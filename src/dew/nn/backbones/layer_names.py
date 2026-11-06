"""The module names of a decoder stack's layers: a single layer's `layers_N` and a scanned run's
`layers_3_7`, which every reader of a stack's variables (a LoRA, a checkpoint's reader, a host's banks)
keys by, apart from the stack's plan (`layer_plan`), whose modules they need not import; and a
stage-local or banked store's runs cut back onto those layers (`stored_layers`), as such a reader reads
them."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np

from dew._coordination import read_home
from dew.batch import Variables, merge, thaw
from dew.nn.protocols import BankedStacks, StagedStacks


def group_name(first: int, count: int) -> str:
    """The module name of a scanned run: `layers_3_7` runs layers 3 through 7."""
    return f'layers_{first}_{first + count - 1}'


def group_layers(name: str) -> range | None:
    """The layers a stack module name runs: `layers_3_7` as range(3, 8),
    `layers_3` as range(3, 4), anything else as None. The inverse of
    `group_name`, for a reader keyed by single layers that meets the run."""
    if not name.startswith("layers_"):
        return None
    parts = name[len("layers_"):].split("_")
    if not all(part.isdigit() for part in parts) or len(parts) > 2:
        return None
    first, last = int(parts[0]), int(parts[-1])
    return range(first, last + 1)


def layer_index(name: str) -> int | None:
    """Return the layer index a single layer's `layers_N` module name holds (`group_layers`), or
    None, a run's `layers_3_7` (or one-layer run's `layers_3_3`) among the rest."""
    layers = group_layers(name)
    return layers.start if layers is not None and name == f"layers_{layers.start}" else None


__all__ = ["group_layers", "group_name", "layer_index"]


def array_rows(path, leaf, shape: tuple[int, ...]) -> list:
    """A stacked leaf's rows, `StackView.layer_form`'s cut for arrays: its leading `shape` flattened by
    the leaf's own reshape, so a host array stays on the host and a device array where it is."""
    del path
    return list(leaf.reshape((-1, *leaf.shape[len(shape):])))


def stage_count(tree) -> int:
    """How many stages the first run `tree` holds stacked by stage (`stored_by_stage`) stacks over, one
    for a tree that holds none."""
    if not isinstance(tree, Mapping):
        return 1
    if 'stages' in tree and (leaves := jax.tree.leaves(tree['stages'])):
        return np.shape(leaves[0])[0]
    return max((stage_count(child) for child in tree.values()), default=1)


class StoredRow:
    """A layer's row of a run's bank (`layers_3_7`, `dew.inference.banks`) as a reader by layer name reads
    it (`stored_layers`): its shape and dtype known before, sliced where the bank lies and read home
    (`dew._coordination.read_home`) only as it is read, a host array (`np.asarray`, an export's
    `dew.interop.streaming.HostArray`) or a JAX one (`jnp.asarray`), and an index of it (an expert's
    tensor) a row of it in turn. So a pool's export of a banked store holds one row of it at a time, not
    a copy of every bank."""

    def __init__(self, bank, index: tuple[int, ...]):
        self.bank, self.index = bank, index

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.bank.shape[len(self.index):])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.bank.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: int) -> StoredRow:
        return StoredRow(self.bank, (*self.index, index))

    def split(self, widths: Sequence[int]) -> list[Deferred]:
        """The row's last axis in runs of `widths` (a packed projection's members,
        `dew.inference.projections.unpacked`), each read from the row as it is read."""
        starts = np.cumsum([0, *widths[:-1]])
        return [Deferred((*self.shape[:-1], width), self.dtype,
                         functools.partial(lambda start, width: np.asarray(self)[..., start:start + width],
                                           int(start), int(width)))
                for start, width in zip(starts, widths, strict=True)]

    def __jax_array__(self) -> jax.Array:
        return jax.numpy.asarray(self.bank[self.index])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        del copy
        held = read_home(self.bank[self.index])
        return held if dtype is None else held.astype(dtype)


class Deferred:
    """A leaf an export builds on the host as it writes it (`np.asarray`), its shape and dtype known
    before: what a transformation makes of a bank's rows (`StoredRow`), a packed row's member
    (`StoredRow.split`) or an adapter's merge of a row's kernel (`dew.lora.Adapter.merge`), so an export
    of a banked store holds one tensor's work at a time."""

    def __init__(self, shape: tuple[int, ...], dtype, build: Callable[[], np.ndarray | jax.Array]):
        self.shape, self.dtype, self.build = tuple(shape), np.dtype(dtype), build

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        del copy
        held = np.asarray(self.build())
        return held if dtype is None else held.astype(dtype)


def deferred(leaf) -> bool:
    """Whether `leaf` is read as an export writes it (`StoredRow`, `Deferred`)."""
    return isinstance(leaf, (StoredRow, Deferred))


def stored_rows(path, leaf, shape: tuple[int, ...]) -> list[StoredRow]:
    """A banked leaf's rows, `StackView.layer_form`'s cut for a reader by layer name (`stored_layers`):
    each a `StoredRow`, read only as it is read."""
    del path
    return [StoredRow(leaf, tuple(int(at) for at in np.unravel_index(row, shape)))
            for row in range(int(np.prod(shape)))]


def stored_layers(model, variables: Variables, *, banks: bool = False) -> Variables:
    """`variables` with every run a stage-local store stacks by stage on its stored layers (`layers_N`),
    each whole, as a reader by layer name reads them: an export, an adapter's merge or save; with `banks`,
    an export's read, also every run a banked store holds as one array under the run's name
    (`layers_3_7`, `dew.inference.banks`), which a model call reads as it is. The runs are those `model`
    declares (`StagedStacks.stage_sites`, `BankedStacks.bank_sites`), each cut back by the view it was
    stacked under (`StackView.layer_form`): a stage-local one's into arrays, a bank's into rows read as
    they are read (`StoredRow`), beside what the store holds of each layer apart (a DEX layer's own
    parameters); variables that hold no such run are returned as they are. A model run reads the store
    as it is (`off_stages`)."""
    stages = stage_count(variables)
    if stages > 1 and isinstance(model, StagedStacks):
        sites, cut = model.stage_sites(variables, stages), array_rows

        def held(node) -> bool:
            return isinstance(node, Mapping) and 'stages' in node
    elif banks and stages == 1 and isinstance(model, BankedStacks):
        # The model plans its runs from the store whole, a trainer's split of its frozen leaves joined.
        sites, cut = model.bank_sites(thaw(variables)), stored_rows

        def held(node) -> bool:
            return isinstance(node, Mapping) and any(len(group_layers(name) or ()) > 1 for name in node)
    else:
        return variables
    stored = dict(variables)
    for site in sites:
        def layered(node, view=site.view):
            return _with_apart(node, view.layer_form(node, cut)) if held(node) else node
        stored = {name: _at(tree, site.namespace, layered) for name, tree in stored.items()}
    return stored


def _with_apart(node: Mapping, layers: dict) -> dict:
    """`layers`, `node` with its runs cut onto their layers (`StackView.layer_form`), each cut layer joined
    by what `node` held of it apart from its run (a DEX layer's own parameters beside a bank)."""
    return {name: merge(node[name], tree) if name in node and tree is not node[name] else tree
            for name, tree in layers.items()}


def _at(tree, namespace: tuple[str, ...], convert):
    """`tree` with its node at `namespace` converted, where it holds one."""
    if not namespace:
        return convert(tree)
    if not isinstance(tree, Mapping) or namespace[0] not in tree:
        return tree
    return {**tree, namespace[0]: _at(tree[namespace[0]], namespace[1:], convert)}
