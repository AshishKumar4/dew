"""The module names of a decoder stack's layers: a single layer's `layers_N` and a scanned run's
`layers_3_7`, which every reader of a stack's variables (a LoRA, a checkpoint's reader, a host's banks)
keys by, apart from the stack's plan (`layer_plan`), whose modules they need not import; and a
stage-local store's runs cut back onto those layers (`stored_layers`), as such a reader reads them."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpy as np

from dew.batch import Variables
from dew.nn.protocols import StagedStacks


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


def stored_layers(model, variables: Variables) -> Variables:
    """`variables` with every run a stage-local store stacks by stage on its stored layers (`layers_N`),
    each whole, as a reader by layer name reads them: an export, an adapter's merge or save. The runs are
    those `model` declares (`StagedStacks.stage_sites`), each cut back by the view it was stacked under
    (`StackView.layer_form`); variables that hold no such run are returned as they are. A model run
    reads the store as it is (`off_stages`). A store on one stage holds its runs so too where its decoder
    scans them (`StagedStacks.stage_sites` at one stage)."""
    if not isinstance(model, StagedStacks) or not any(
            isinstance(tree, Mapping) and _staged(tree) for tree in variables.values()):
        return variables
    stages = stage_count(variables)
    stored = dict(variables)
    for site in model.stage_sites(variables, stages):
        def layered(node, view=site.view):
            held = isinstance(node, Mapping) and 'stages' in node
            return view.layer_form(node, array_rows) if held else node
        stored = {name: _at(tree, site.namespace, layered) for name, tree in stored.items()}
    return stored


def _staged(tree: Mapping) -> bool:
    """Whether `tree` holds a run stacked by stage (`StackView.stage_form`) anywhere in it."""
    return 'stages' in tree or any(isinstance(child, Mapping) and _staged(child) for child in tree.values())


def _at(tree, namespace: tuple[str, ...], convert):
    """`tree` with its node at `namespace` converted, where it holds one."""
    if not namespace:
        return convert(tree)
    if not isinstance(tree, Mapping) or namespace[0] not in tree:
        return tree
    return {**tree, namespace[0]: _at(tree[namespace[0]], namespace[1:], convert)}
