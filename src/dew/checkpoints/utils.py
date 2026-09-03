"""Locating the newest checkpoint on disk, and serializing a model's configuration."""

import os

import flax.linen as nn


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
