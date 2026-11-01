"""The module names of a decoder stack's layers: a single layer's `layers_N` and a scanned run's
`layers_3_7`, which every reader of a stack's variables (a LoRA, a checkpoint's reader, a host's banks)
keys by, apart from the stack's plan (`layer_plan`), whose modules they need not import."""

from __future__ import annotations


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
