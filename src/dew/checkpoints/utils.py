"""The train state a checkpoint stores, and finding the newest checkpoint on disk."""

import dataclasses
import os
from typing import Any

import flax.linen as nn
from flax import struct
import jax
import numpy as np


@struct.dataclass
class RestoredState:
    """The arrays a restored train state holds, under the TrainState's names.

    A checkpoint stores arrays, not the model or the optimizer that wrote
    them, so `apply_fn` and `tx` are absent here. `restore_into` grafts the
    values onto a live train state, so a trainer given one as `train_state`
    starts from the restored params, EMA copy, optimizer state and step. It
    is a warm start at that step, not the mid-epoch resume that
    `load_from_checkpoint=<dir>` on the trainer gives: the data iterator's
    position, the run's own rng stream and the best loss so far live outside
    the train state and are not carried by this object.
    """
    step: jax.Array
    params: dict
    opt_state: Any = None
    ema_params: dict | None = None
    metrics: Any = None
    dynamic_scale: Any = None
    rngs: jax.Array | None = None

    @classmethod
    def from_checkpoint(cls, state: dict) -> "RestoredState":
        """Keep the fields a dew checkpoint holds; anything else in the
        directory is reporting, not state, and does not belong here."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in state.items() if k in known})

    def restore_into(self, train_state):
        """Graft these values onto a live train state, keeping its types.

        Orbax restores an optimizer's states as plain lists and dicts, so a
        restored state cannot run a compiled step as it stands. The types
        (adam's moments, the metrics collection) come from `train_state`,
        the values from here, matched leaf by leaf; the count and shape
        check turns a checkpoint from another run's optimizer into a plain
        error instead of a silently wrong state. A field this checkpoint
        predates keeps the live state's value.
        """
        updates = {}
        for field in dataclasses.fields(train_state):
            restored = getattr(self, field.name, None)
            live = getattr(train_state, field.name)
            if restored is None:
                continue
            leaves, treedef = jax.tree_util.tree_flatten(live)
            values = jax.tree_util.tree_leaves(restored)
            if len(leaves) != len(values) or any(
                    np.shape(leaf) != np.shape(value)
                    for leaf, value in zip(leaves, values)):
                raise ValueError(
                    f"The checkpoint's {field.name} does not fit this train "
                    f"state: {len(values)} restored leaves against "
                    f"{len(leaves)}. A checkpoint carries the optimizer "
                    f"state, so a resume needs the model, the optimizer and "
                    f"the gradient accumulation it was written with.")
            updates[field.name] = jax.tree_util.tree_unflatten(treedef, values)
        return train_state.replace(**updates)


def serialize_model(model: nn.Module):
    """
    Serializes the model to a dictionary format.
    """
    model_dict = model.__dict__
    model_dict = {k: v for k, v in model_dict.items() if not k.startswith('_')}
    # Convert all callable attributes to their string representation
    def map(model_dict):
        for k, v in model_dict.items():
            if isinstance(v, dict):
                # Recursively serialize nested dictionaries
                model_dict[k] = map(v)
            elif isinstance(v, list):
                # Recursively serialize lists
                [map(item) if isinstance(item, dict) else item for item in v]
            elif callable(v):
                # If the attribute has __name__, use that as the key
                if hasattr(v, '__name__'):
                    model_dict[k] = v.__name__
                else:
                    model_dict[k] = str(v).split('.')[-1]
    map(model_dict)
    return model_dict


def get_latest_checkpoint(checkpoint_path):
    """Path of the highest-numbered step directory under a checkpoint root.

    Orbax leaves entries that are not steps next to them (lock files, the
    manager's own metadata, `<step>.orbax-checkpoint-tmp` directories from a
    write that was interrupted), and a step directory can exist while still
    being empty. int()-ing every name raises on the first of those, so this
    only considers directories whose name is a number and which hold something.
    """
    steps = []
    for entry in os.listdir(checkpoint_path):
        if not entry.isdecimal():
            continue
        step_path = os.path.join(checkpoint_path, entry)
        if not os.path.isdir(step_path) or not os.listdir(step_path):
            continue
        steps.append(int(entry))
    if not steps:
        raise FileNotFoundError(f"No checkpoint step directory in {checkpoint_path}")
    return os.path.join(checkpoint_path, str(max(steps)))
