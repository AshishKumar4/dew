"""What `Dataset.show` prints: the chain as written, each verb and step a preset or function expanded
to in order, then its records, each array as its shape and dtype."""

from __future__ import annotations

import sys
import types
from collections.abc import Hashable, Iterable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from dew.batch import Batch
from dew.data.pipeline import (
    Batched,
    BatchStage,
    Configured,
    DeviceStep,
    Filter,
    FlatMap,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    Native,
    Phases,
    Pipeline,
    RandomMap,
    Stage,
    runs,
)


def shown(chain: Pipeline, rows: Iterable[Batch | bytes]) -> None:
    """Print `chain`, a line a stage, then each of `rows`."""
    from rich.pretty import pprint

    sys.stdout.write("\n".join(_chain_lines(chain)) + "\n")
    for row in rows:
        pprint({name: _shown(value) for name, value in row.items()} if isinstance(row, Mapping) else row)


def _chain_lines(chain: Pipeline, depth: int = 0) -> list[str]:
    """The lines `show` prints of `chain`: its store, or the corpora a mix or phases read, each
    with its own chain indented below, then each stage in order, one a line."""
    indent = "  " * depth
    source = chain.source
    if isinstance(source, Mixture):
        lines = [f"{indent}mix" if source.tokens is None else f"{indent}mix(by='tokens')"]
        for weight, child in zip(source.weights, source.pipelines, strict=True):
            lines += [f"{indent}  weight {weight:g}:", *_chain_lines(child._chain(), depth + 2)]
    elif isinstance(source, Phases):
        lines = [f"{indent}phases"]
        for until, child in zip(source.until_steps, source.pipelines, strict=True):
            lines += [f"{indent}  until step {until}:" if until is not None else f"{indent}  last:",
                      *_chain_lines(child._chain(), depth + 2)]
    else:
        lines = [f"{indent}{type(source).__name__} {source.label()!r}"]
    return lines + [f"{indent}.{_stage_line(stage)}" for stage in chain.stages]


def _stage_line(stage: Stage | Batched | Native) -> str:
    """One stage as `show` prints it: the verb that wrote it and the step it runs, or the
    stage's own settings."""
    ran = runs(stage)
    if ran is None:
        return f"batch({stage.size})" if isinstance(stage, BatchStage) else _brief(repr(stage))
    named = _step_name(ran.step)
    if isinstance(stage, MapBatch):
        where = "device" if isinstance(stage.step, DeviceStep) else "host"
        return f"map({named}) on the {where}, per batch"
    if isinstance(stage, Batched):
        verb = "map" if stage.same_rows else "flat_map"
        return f"{verb}({named}, batched=True, batch_size={stage.size})"
    if isinstance(stage, (Map, RandomMap)):
        drawn = "random_" if isinstance(stage, RandomMap) else ""
        return f"{drawn}{'filter_map' if stage.rejects else 'map'}({named})"
    verbs = {Configured: "map", Filter: "filter", Native: "grain", MapWithIndex: "map_with_index",
             FlatMap: "flat_map"}
    return f"{verbs[type(stage)]}({named})"


def _step_name(step: Hashable) -> str:
    """A step as `show` prints it: a function by its name, any other step by its settings."""
    if isinstance(step, (types.FunctionType, types.MethodType)):
        return step.__qualname__
    return _brief(repr(step))


def _brief(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


@runtime_checkable
class _Shaped(Protocol):
    """An array a step leaves on a device (a JAX array), as `show` reads it."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def size(self) -> int: ...


def _shown[Field](value: np.ndarray | _Shaped | bytes | Field) -> np.ndarray | _Shaped | bytes | Field | str:
    """A field as `show` prints it: an array, on the host or on a device after the chain's device
    steps, as its shape and dtype; bytes as their length."""
    if isinstance(value, (np.ndarray, _Shaped)) and value.size > 16:
        return f"{value.dtype}{list(value.shape)}"
    if isinstance(value, bytes) and len(value) > 64:
        return f"<{len(value)} bytes>"
    return value
