"""Grain's batching of records, into its shared memory where asked."""

from collections.abc import Sequence

def make_batch(values: Sequence[object], *, output_to_shared_memory: bool = ...) -> object: ...
