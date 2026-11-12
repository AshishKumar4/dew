"""Grain's mixing of datasets by weight."""

from collections.abc import Sequence

def _float_to_int_proportions(values: Sequence[float], resolution_deminator: int = ...) -> list[int]: ...
