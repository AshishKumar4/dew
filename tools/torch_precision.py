"""The one precision context the reference generators build and run their float64 oracles under, with
torch alone, so a generator in a torch-only environment imports it as freely as one beside Diffusers
or transformers (tools/diffusers_wan_reference.py widens on top of it)."""

import contextlib

import torch


def wide(dtype):
    """`dtype`, float32 read as float64: what a widened reference casts to where it pins float32."""
    return torch.float64 if dtype == torch.float32 else dtype


@contextlib.contextmanager
def default_dtype(dtype):
    """torch's default floating dtype `dtype` within the block, the previous one restored after it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
