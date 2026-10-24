"""Each Pallas launch's blocks, warps and stages, measured per GPU model and
shape, and the timing every benchmark in Dew measures with.

A launch is a `KernelOp`, tokamax's `Op` (tokamax/_src/ops/op.py) for one
`pallas_call` on primal arrays. It declares its config dataclass, the space it
measures (`_get_autotuning_configs`), the rule it takes without a measurement
(`_get_heuristics_config`) and the arguments whose leading axis counts tokens
(`ROWS`), bucketed to a power of two in its key, as fla keys `triton.autotune`
on its heads and widths. The kernel's own `custom_vjp`, shard placement and
remat names stay around it; the op is never differentiated.

A launch's space, heuristic and measurement are read at its key's shapes
(`KernelOp.keyed`), so one pick serves every shape the key covers. At trace
time a launch takes, in order, a config pinned for its kernel (`pinned`),
the pick tokamax's cache holds for its key on its target GPU model (its
overlays, then Dew's picks, `KernelOp.get_autotuning_cache`), the pick
measured at the nearest stored token count of the same launch, and its own
shapes' heuristic, the picks only where the space holds them; Dew resolves
the config itself, so no tokamax policy measures or refuses inside a trace.

Before a program compiles, `tune` reads its launches from a lowering that
describes them (`launches`), and the pool's process 0 decides every distinct
launch's config, which the others take exactly, so every process compiles the
same program: its pick, or for a launch without one the nearest stored pick
process 0 holds or else the heuristic, which stays unmeasured, so a later pass
still measures it. Only an explicit pass measures (`tune(..., measure=True)`, which
`Trainer(tune_kernels=True)` and tools/kernel_picks.py call), on a GPU the
process has to itself (`dew.cache.measures_picks`): every config of a launch
without a pick compiles on a thread pool and runs alone, its kernel's device
time read by CUPTI after `FLUSH_BYTES` are read, and the fastest is kept.
Any other compile measures nothing and says once how many launches lack a
measurement. Picks persist per GPU model and jaxlib beside XLA's own
(`dew.cache.kernel_picks_path`), in Dew's own JSON (`encoded`, `decoded`), which tokamax's public API
leaves to its callers. Wall-clock timing is
`dew.telemetry.benchmark.samples`, which every benchmark tool in Dew measures with.

Tile choice changes a kernel's reduction order, and its error is not
monotone in tiles, warps or programs, so the space is a declaration of
proven configs: each kernel's GPU tests hold its float64 gate at every config
of its space (`pinned`, tests/tuned.py's `each_config`), and a config that
fails leaves the space. The tuner only times what is proven.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import enum
import importlib
import json
import logging
import os
import threading
import time
import types
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import ClassVar

import filelock
import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from jax.extend.backend import get_default_device

from dew import _coordination, _records, cache
from dew._records import JSON
from dew.files import write_atomically

from . import generation

_log = logging.getLogger(__name__)

FLUSH_BYTES = 256 << 20
"""The bytes read before each timed launch: twice the largest L2 among Dew's
GPUs (an RTX PRO 6000's 128 MB), so no launch finds its operands in cache,
as none does in a training step."""

REPEATS = 5
"""The timed launches of each config."""


class Timings(dict):
    """One launch's timings by config: `tokamax.BenchmarkData` where a config ran, the exception where
    it did not; tokamax's cache lookup returns it as a launch's pick (`KernelOp.get_autotuning_cache`)."""

    @property
    def fastest_config(self) -> Hashable:
        """The config of least median time; a mapping with none that ran raises naming its failures."""
        ran = [(config, timing) for config, timing in self.items()
               if isinstance(timing, tokamax.BenchmarkData)]
        if not ran:
            raise ValueError(f"no config ran: {list(self.values())}")
        return min(ran, key=lambda pair: pair[1].median_evaluation_time_ms)[0]

    def prune_errors(self) -> Timings:
        """The configs that ran."""
        return Timings({config: timing for config, timing in self.items()
                        if isinstance(timing, tokamax.BenchmarkData)})

    def prune(self) -> Timings:
        """The fastest config alone."""
        config = self.fastest_config
        return Timings({config: self[config]})


type Rebuilt = Hashable | list | dict
"""What `decoded` rebuilds: an op's field, a launch's argument or a config."""


class Unreadable(ValueError):
    """A stored launch or timing this checkout cannot rebuild: another checkout's kernel, its fields or
    its config's, or JSON this module did not write. `_read` keeps such an entry as it is."""


def encoded(value: object) -> JSON:
    """`value`, an op's field, a launch's argument or a config, as JSON `decoded` rebuilds equal: a
    scalar, None or string as itself; a shape and dtype, a dtype, a tuple, a list, a mapping, an enum
    member and a dataclass by their parts, a class and a module-level function (a template's hook) by
    its module and name. Anything else, a lambda, a closure or a local function among them, has no JSON
    form here, and raises: a measured pick for a launch with a hook needs a module-level hook."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, jax.ShapeDtypeStruct):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.dtype):
        return {"dtype": str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": _named(type(value)), "name": value.name}
    if isinstance(value, type):
        return {"class": _named(value)}
    if isinstance(value, types.FunctionType):
        return {"function": _function_name(value)}
    if dataclasses.is_dataclass(value):
        return {"dataclass": _named(type(value)),
                "fields": {field.name: encoded(getattr(value, field.name))
                           for field in dataclasses.fields(value) if field.init}}
    if isinstance(value, tuple):
        return {"tuple": [encoded(element) for element in value]}
    if isinstance(value, list):
        return [encoded(element) for element in value]
    if isinstance(value, Mapping):
        return {"mapping": [[encoded(key), encoded(entry)] for key, entry in value.items()]}
    raise TypeError(f"{type(value).__name__} has no JSON form in a kernel's picks: {value!r}")


def decoded(value: _records.JSON) -> Rebuilt:
    """What `encoded` wrote, rebuilt. A class it names that this checkout lacks, a dataclass whose
    stored fields are not this checkout's, or JSON it did not write raises `Unreadable`; whatever the
    rebuilt classes raise propagates."""
    if isinstance(value, list):
        return [decoded(element) for element in value]
    if not isinstance(value, dict):
        return value
    if "shape" in value:
        return jax.ShapeDtypeStruct(_field(_records.integers, value.get("shape"), "shape"),
                                    _dtype(_field(_records.text, value.get("dtype"), "dtype")))
    if "dtype" in value:
        return _dtype(_field(_records.text, value.get("dtype"), "dtype"))
    if "enum" in value:
        members = _located(_field(_records.text, value.get("enum"), "enum")).__dict__
        member = members.get(_field(_records.text, value.get("name"), "name"))
        if not isinstance(member, enum.Enum):
            raise Unreadable(f"no such member in this checkout: {value!r}")
        return member
    if "class" in value:
        return _located(_field(_records.text, value.get("class"), "class"))
    if "function" in value:
        return _located_function(_field(_records.text, value.get("function"), "function"))
    if "dataclass" in value:
        cls = _located(_field(_records.text, value.get("dataclass"), "dataclass"))
        fields = value.get("fields")
        if not dataclasses.is_dataclass(cls) or not isinstance(fields, dict) or set(fields) != {
                field.name for field in dataclasses.fields(cls) if field.init}:
            raise Unreadable(f"{cls.__qualname__}'s stored fields are not this checkout's: {fields!r}")
        return cls(**{name: decoded(field) for name, field in fields.items()})
    if "tuple" in value:
        elements = value.get("tuple")
        if not isinstance(elements, list):
            raise Unreadable(f"a stored tuple is no list: {elements!r}")
        return tuple(decoded(element) for element in elements)
    pairs = value.get("mapping")
    if not isinstance(pairs, list):
        raise Unreadable(f"stored JSON this module did not write: {value!r}")
    rebuilt = {}
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise Unreadable(f"a stored mapping's entry is no pair: {pair!r}")
        rebuilt[_key(decoded(pair[0]))] = decoded(pair[1])
    return rebuilt


def _field[T](reader: Callable[[_records.JSON, str], T], value: _records.JSON, key: str) -> T:
    """`value` read by `dew._records`' `reader`; `Unreadable` where it refuses it."""
    try:
        return reader(value, key)
    except ValueError as error:
        raise Unreadable(str(error)) from error


def _dtype(name: str) -> np.dtype:
    """The dtype `name` names; `Unreadable` where this checkout's NumPy and JAX know none."""
    try:
        return jnp.dtype(name)
    except TypeError as error:
        raise Unreadable(f"no dtype {name!r} in this checkout") from error


def _key(value: Rebuilt) -> Hashable:
    """`value` as a mapping's key or a config, which hash: a list or a mapping does not."""
    if isinstance(value, (list, dict)):
        raise Unreadable(f"expected a hashable value in a kernel's picks: {value!r}")
    return value


def _named(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _function_name(function: types.FunctionType) -> str:
    """A module-level function's module and name, which `_located_function` finds again; a lambda, a
    closure, a local function or one a class holds has none to store."""
    name = f"{function.__module__}:{function.__qualname__}"
    if "." in function.__qualname__ or function.__name__ == "<lambda>" or function.__closure__:
        raise TypeError(f"the hook {name} is no module-level function, which a kernel's stored picks name")
    return name


def _located_function(name: str) -> types.FunctionType:
    """The module-level function `_function_name` names: `Unreadable` where this checkout's module holds
    no such function under that name, or another object (a renamed or shadowed hook)."""
    module, _, qualified = name.partition(":")
    found = vars(_module(module)).get(qualified) if "." not in qualified else None
    if not isinstance(found, types.FunctionType) or f"{found.__module__}:{found.__qualname__}" != name:
        raise Unreadable(f"{name} names no module-level function in this checkout")
    return found


def _module(module: str) -> types.ModuleType:
    """The module `module` names; `Unreadable` where this checkout has no such module."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        # The named module or a parent of it is absent; a module that fails to import otherwise, or
        # imports something else that is absent, is this checkout's error and propagates.
        if error.name is None or not (module == error.name or module.startswith(error.name + ".")):
            raise
        raise Unreadable(f"this checkout has no module {module}") from error


def _located(name: str) -> type:
    """The class `_named` names; `Unreadable` where this checkout has no such class."""
    module, _, qualified = name.partition(":")
    found: object = _module(module)
    for part in qualified.split("."):
        # A module and each class on the way hold the next part among their own attributes.
        found = vars(found).get(part) if isinstance(found, (type, types.ModuleType)) else None
        if found is None:
            raise Unreadable(f"this checkout has no class {name}")
    if not isinstance(found, type):
        raise Unreadable(f"{name} names no class in this checkout")
    return found


def _launch_json(ba: tokamax.BoundArguments) -> dict[str, JSON]:
    """A launch as JSON: its op as its picks name it (`_identity`) and its arguments."""
    return {"op": encoded(_identity(ba.op)), "arguments": {name: encoded(value)
                                                           for name, value in ba.arguments.items()}}


def _launch(entry: object) -> tokamax.BoundArguments:
    """The launch `_launch_json` wrote; `Unreadable` where this checkout cannot rebuild it: its op's
    class or fields, or arguments that are not exactly its kernel's parameters (another checkout's
    signature of it)."""
    if not isinstance(entry, dict) or not isinstance(entry.get("arguments"), dict):
        raise Unreadable(f"a stored launch is no mapping of an op and its arguments: {entry!r}")
    arguments = entry["arguments"]
    op = decoded(entry.get("op"))
    if not isinstance(op, tokamax.Op):
        raise Unreadable(f"a stored launch's op is no tokamax Op: {op!r}")
    ba = tokamax.BoundArguments(op, {name: decoded(value) for name, value in arguments.items()})
    if set(arguments) != set(ba.signature.parameters) - {"config"}:
        raise Unreadable(f"{type(op).__name__}'s stored arguments are not this checkout's: "
                         f"{sorted(arguments)}")
    return ba


TIMING_FIELDS = ("compile_time_ms", "lower_time_ms", "evaluation_times_ms", "metadata", "peak_memory_mb")
"""Every field of `tokamax.BenchmarkData`, which a launch's stored timings keep."""


def _timings_json(timings: Timings) -> list[JSON]:
    """The configs that ran and every field of their times (`TIMING_FIELDS`), as JSON (`_timings`)."""
    return [[encoded(config), {"compile_time_ms": timing.compile_time_ms,
                               "lower_time_ms": timing.lower_time_ms,
                               "evaluation_times_ms": list(timing.evaluation_times_ms),
                               "metadata": encoded(dict(timing.metadata)),
                               "peak_memory_mb": timing.peak_memory_mb}]
            for config, timing in timings.prune_errors().items()]


def _timings(entries: object) -> Timings:
    """The timings `_timings_json` wrote; `Unreadable` where they are not."""
    if not isinstance(entries, list):
        raise Unreadable(f"stored timings are no list: {entries!r}")
    measured: dict[Hashable, tokamax.BenchmarkData | Exception] = {}
    for pair in entries:
        if not (isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], dict)
                and set(pair[1]) == set(TIMING_FIELDS)):
            raise Unreadable(f"a stored timing is not this checkout's: {pair!r}")
        config, fields = pair
        metadata = decoded(fields["metadata"])
        if not isinstance(metadata, dict):
            raise Unreadable(f"a stored timing's metadata is no mapping: {metadata!r}")
        peak = fields["peak_memory_mb"]
        measured[_key(decoded(config))] = tokamax.BenchmarkData(
            compile_time_ms=_field(_records.number, fields["compile_time_ms"], "compile_time_ms"),
            lower_time_ms=_field(_records.number, fields["lower_time_ms"], "lower_time_ms"),
            evaluation_times_ms=_field(_records.numbers, fields["evaluation_times_ms"],
                                       "evaluation_times_ms"),
            metadata=metadata,
            peak_memory_mb=None if peak is None else _field(_records.number, peak, "peak_memory_mb"))
    return Timings(measured)


def dumps_launches(launches: list[tokamax.BoundArguments]) -> str:
    """`launches` as JSON, which `loads_launches` reads back."""
    return json.dumps([_launch_json(ba) for ba in launches])


def loads_launches(text: str) -> list[tokamax.BoundArguments]:
    """The launches `dumps_launches` wrote."""
    return [_launch(entry) for entry in json.loads(text)]


def dumps_picks(device_kind: str, picks) -> str:
    """`picks`, `(launch, Timings)` pairs measured on GPUs of `device_kind`, as the JSON Dew stores
    them (`_read`): each launch with the configs that ran."""
    return json.dumps({"device_kind": device_kind, "data": [
        {**_launch_json(ba), "timings": _timings_json(timings)} for ba, timings in picks]})


class _Picks(dict):
    """One op's picks on one GPU model, by key; a key without one reads None,
    tokamax's miss."""

    def __missing__(self, key):
        return None


# tokamax's `get_bound_args` for a lowered program: its annotation's ParamSpec,
# for a callable's arguments, is one pyright cannot solve without any.
_bound_args: Callable[[jax.stages.Lowered], tuple[tokamax.BoundArguments, ...]] = (
    tokamax.autotuning.get_bound_args)

_PICKS: dict[tuple[tokamax.Op, str], _Picks] = {}
_NEAREST: dict[tuple[tokamax.Op, str], dict[Hashable, Timings | None]] = {}
"""Per op and GPU model, the nearest pick process 0 resolved for each key without a pick of its own,
None for its heuristic (`_settle`): what every process launches there, whatever other picks it holds."""
_LOADED: set[str] = set()
_LOADING = threading.Lock()
_told: set[str] = set()


@dataclasses.dataclass(frozen=True)
class KernelOp(tokamax.Op):
    """One Pallas launch whose config is measured per GPU model and shape.

    A subclass declares `config_cls` (a frozen dataclass), `_fwd(..., config,
    return_residuals)` returning `(outputs, None)`, `_get_autotuning_configs`
    and `_get_heuristics_config`, and names its token-counted arguments in
    `ROWS`. `revision` is part of its stored picks' identity: a kernel whose
    speed changes raises it, and picks measured for the old revision stop
    matching. A kernel whose inputs must hold more than random values to run
    (group sizes, indices) overrides `operands`.
    """

    ROWS: ClassVar[tuple[str, ...]] = ()
    supports_batched_args_capture: ClassVar[bool] = False
    supports_symbolic_shapes: ClassVar[bool] = False
    revision: int = 1

    def __call__(self, *args, **kwargs):
        config = self.config
        if config is None:
            ba = self.bind(*args, **kwargs)
            keyed = self.keyed(ba)
            if (collected := cache.KERNEL_SEEN.value) is not None:
                collected.found.append((self, keyed))
            config = self._chosen(ba, keyed)
        if cache.KERNEL_LAUNCHES.value:
            # Dew chose the launch for this GPU (`generation`); tokamax's device check would ask
            # each traced array for its devices (`keyed`).
            return tokamax.Op.__call__(self.replace(config=config, bypass_device_check=True), *args, **kwargs)
        return self._fwd(*args, config=config, **kwargs)[0]

    def chosen(self, ba: tokamax.BoundArguments) -> Hashable:
        """The config a launch of `ba` takes where it traces now: the op's own `config` where it carries
        one, else a config pinned for its kernel that its key's space holds (`pinned`), else its pick
        (`_pick`, else `_nearest`), else its own shapes' heuristic."""
        return self.config if self.config is not None else self._chosen(ba, self.keyed(ba))

    def _chosen(self, ba: tokamax.BoundArguments, keyed: tokamax.BoundArguments) -> Hashable:
        pin = dict(cache.KERNEL_PINS.value).get(type(self))
        pick = _pick(keyed) or self._nearest(keyed)
        return (pin if pin in keyed.autotuning_configs
                else ba.heuristics_config if pick is None else pick.fastest_config)

    def keyed(self, ba: tokamax.BoundArguments) -> tokamax.BoundArguments:
        """`ba` as its key reads it, where its space, heuristic and measurement
        are read: its arrays as their shapes and dtypes, as tokamax keys them,
        the token-counted ones at their key's shapes, so a pick holds for every
        shape a key covers, and the op as its picks name it (`_identity`).
        A launch's GPU model is then read from the context, never from a
        traced array: asking one for its devices raises an error that walks
        the whole trace to say where the array came from, which was 111 s of
        the packed Qwen3.5-0.8B step's 330 s compile on an A100 (ledger 554)."""
        def shaped(name: str, value):
            if name in self.ROWS and value is not None:
                return _bucketed(jax.ShapeDtypeStruct(value.shape, value.dtype))
            # NumPy operands (an unpacked delta rule's chunk and span descriptors) key as JAX's do.
            return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
                                if isinstance(leaf, (jax.Array, np.ndarray)) else leaf, value)
        return tokamax.BoundArguments(_identity(self), {name: shaped(name, value)
                                                        for name, value in ba.arguments.items()})

    def _nearest(self, ba: tokamax.BoundArguments) -> Timings | None:
        """The pick a launch without its own takes: the one process 0 settled for `ba`'s key
        (`_settle`), else the nearest this process stores (`_stored_nearest`)."""
        settled = _NEAREST.get((_identity(self), _kind(ba)), {})
        key = ba.autotuning_cache_key
        return settled[key] if key in settled else self._stored_nearest(ba)

    def _stored_nearest(self, ba: tokamax.BoundArguments) -> Timings | None:
        """The pick measured for the stored key nearest `ba`'s: every argument
        the same but the token counts (`ROWS`), the fewest powers of two away,
        the smaller on a tie, where `ba`'s space holds its config."""
        key = ba.autotuning_cache_key
        rows = [name for name in self.ROWS if key.get(name) is not None]

        def distance(stored) -> int | None:
            if stored.keys() != key.keys():
                return None
            if any(stored[name] != key[name] for name in key if name not in rows):
                return None
            # A launch without one of the token-counted operands (a norm with no branch) is another launch.
            if any(stored[name] is None or (stored[name].shape[1:], stored[name].dtype)
                   != (key[name].shape[1:], key[name].dtype) for name in rows):
                return None
            return sum(abs(stored[name].shape[0].bit_length() - key[name].shape[0].bit_length())
                       for name in rows)

        picks = self.get_autotuning_cache(_kind(ba)).items() if rows else ()
        near = sorted((((steps, stored[rows[0]].shape[0]), timings) for stored, timings in picks
                       if (steps := distance(stored)) is not None), key=lambda entry: entry[0])
        return next((timings for _, timings in near if timings.fastest_config in ba.autotuning_configs), None)

    def get_autotuning_cache(self, device_kind: str | None = None) -> _Picks:
        """Dew's picks for this op on GPUs of `device_kind` (the default
        device's for None), the cache tokamax reads a launch's pick from; the
        store is read in at the first lookup of each GPU model."""
        kind = device_kind or get_default_device().device_kind
        with _LOADING:
            if kind not in _LOADED:
                for key, picks in _stored_picks(kind).items():
                    _PICKS.setdefault(key, _Picks()).update(picks)
                _LOADED.add(kind)
            return _PICKS.setdefault((_identity(self), kind), _Picks())

    def _get_autotuning_cache_key(self, ba: tokamax.BoundArguments) -> Mapping[str, object]:
        key = super()._get_autotuning_cache_key(ba)
        return type(key)({name: _bucketed(spec) if name in self.ROWS and spec is not None else spec
                          for name, spec in key.items()})

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Arrays at `specs` to time the launch on: normal floats, zero integers."""
        keys = jax.random.split(jax.random.key(seed), len(specs))
        return [jax.random.normal(key, spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.floating)
                else jnp.zeros(spec.shape, spec.dtype) for key, spec in zip(keys, specs, strict=True)]


def _identity(op: tokamax.Op) -> tokamax.Op:
    """`op` as its picks name it: without the config, the vjp and the device
    check a call sets."""
    return op.replace(config=None, vjp=None, bypass_device_check=False)


def _kind(ba: tokamax.BoundArguments) -> str:
    """The GPU model the launch `ba` is chosen for, as tokamax reads it for its cache lookups: its
    concrete operands' devices', else the context mesh's (`Trainer.lower_for`'s abstract GPU), else
    the default device's. A traced or keyed launch (`keyed`) names no device of its own."""
    from jax.core import Tracer

    kinds = {device.device_kind for leaf in jax.tree.leaves(dict(ba.arguments))
             if isinstance(leaf, jax.Array) and not isinstance(leaf, Tracer)
             for device in leaf.devices()}
    if len(kinds) > 1:
        raise ValueError(f"a launch's operands lie on GPUs of several models: {sorted(kinds)}")
    if kinds:
        return kinds.pop()
    device = jax.sharding.get_abstract_mesh().abstract_device
    return (device or get_default_device()).device_kind


def _pick(ba: tokamax.BoundArguments) -> Timings | None:
    """The pick tokamax's cache holds for `ba` (its key's shapes), where
    `ba`'s space holds its fastest config; any other is no pick."""
    stored = ba.cached_autotuning_data
    # Dew's own picks alone (`get_autotuning_cache`); tokamax's overlays hold none of Dew's launches.
    return stored if isinstance(stored, Timings) and stored.fastest_config in ba.autotuning_configs else None


def _bucketed(spec: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    """`spec` with its leading (token) axis at the next power of two."""
    return jax.ShapeDtypeStruct((1 << max(spec.shape[0] - 1, 0).bit_length(), *spec.shape[1:]), spec.dtype)


def _kernel_times(cupti, executable, *args) -> list[tuple[str, float]]:
    """Each kernel's name and device milliseconds over one run of
    `executable`, read by `cupti` (a mosaic `profiler.Cupti`); none where it
    launched none."""
    _, kernels = cupti.measure(executable, aggregate=False)(*args)
    return kernels if isinstance(kernels, list) else []


def _another_subscriber(error: RuntimeError) -> bool:
    """Whether CUPTI refused the tuner's subscription for another subscriber's, a profile capture or
    Nsight's: the launch keeps its heuristic until a later compile. Mosaic's profiler raises it as a
    RuntimeError naming the other subscriber; every other failure is the tuner's to report."""
    return "another subscriber" in str(error)


def _detach(profiler, evicting: jax.stages.Compiled, flush: jax.Array) -> None:
    """Detach CUPTI from the process with a finalizing read of `evicting`, so a profiler can subscribe
    after; where one already holds it (`_another_subscriber`), there is nothing of the tuner's to
    detach."""
    try:
        _kernel_times(profiler.Cupti(), evicting, flush)
    except RuntimeError as error:
        if not _another_subscriber(error):
            raise


def _resource_exhausted(error: Exception) -> bool:
    """Whether a config failed for a block past the GPU's own resources, the failures a config of a
    launch's space may meet on a GPU it was not declared for: its shared memory as XLA's compile
    reports it ("RESOURCE_EXHAUSTED: Shared memory size limit exceeded"), or its registers or threads as
    the driver refuses the launch (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, which XLA passes on as an
    INTERNAL error naming it). Any other failure, device memory's included, is the kernel's or the
    tuner's, and propagates."""
    message = str(error)
    return (message.startswith("RESOURCE_EXHAUSTED: Shared memory size limit exceeded")
            or "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" in message)


def materialized(op: KernelOp, ba: tokamax.BoundArguments):
    """The arrays the tuner times `ba` on, `op.operands` of its `ShapeDtypeStruct` arguments, and
    `fill(arrays) -> (args, kwargs)`, its arguments with those arrays (or traced ones) in their places."""
    leaves, tree = jax.tree.flatten((ba.args, ba.kwargs),
                                    is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    places = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]

    def fill(arrays):
        filled = list(leaves)
        for index, array in zip(places, arrays, strict=True):
            filled[index] = array
        return jax.tree.unflatten(tree, filled)

    return op.operands([leaves[index] for index in places], 0), fill


def _measured(op: KernelOp, ba: tokamax.BoundArguments, profiler=None) -> Timings:
    """Every config of `ba`'s launch (at its key's shapes), each compiled on a
    thread and timed alone: the device time CUPTI reads for the launch's own
    kernels, each run after `FLUSH_BYTES` are read, so every config starts
    with its operands out of cache. `profiler` is Mosaic GPU's, whose `Cupti`
    reads the times, or a stand-in with its `Cupti`."""
    if profiler is None:
        from jax.experimental.mosaic.gpu import profiler

    arrays, fill = materialized(op, ba)
    flush = jnp.zeros(FLUSH_BYTES // 4, jnp.float32)
    evicting = jax.jit(jnp.sum).lower(flush).compile()

    def compiled(config):
        configured = op.replace(config=config)

        def run(flush, arrays):
            # The launch waits for the flush, which reads the whole buffer through the cache.
            total, arrays = jax.lax.optimization_barrier((jnp.sum(flush), arrays))
            args, kwargs = fill(arrays)
            return total, configured(*args, **kwargs)

        start = time.perf_counter()
        lowered = jax.jit(run).lower(flush, arrays)
        middle = time.perf_counter()
        executable = lowered.compile()
        return executable, (middle - start) * 1e3, (time.perf_counter() - middle) * 1e3

    timings: dict[Hashable, tokamax.BenchmarkData | Exception] = {}
    read = {}
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = {config: pool.submit(compiled, config) for config in ba.autotuning_configs}
    # CUPTI reads every run, the flush's own first, subscribing for each and
    # unsubscribing after. It takes one subscriber, so where a profile capture
    # holds it, from the first run or any later one, the launch keeps its
    # heuristic until a later compile.
    cupti = profiler.Cupti(finalize=False)
    flushes: set[str] = set()
    try:
        flushes = {name for name, _ in _kernel_times(cupti, evicting, flush)}
        for config, future in futures.items():
            try:
                executable, lowering, compiling = future.result()
                runs = [_kernel_times(cupti, executable, flush, arrays) for _ in range(REPEATS)]
            except jax.errors.JaxRuntimeError as error:
                if not _resource_exhausted(error):
                    raise
                timings[config] = error
                continue
            read[config] = runs, lowering, compiling
    except RuntimeError as error:
        if not _another_subscriber(error):
            raise
        return Timings(dict.fromkeys(ba.autotuning_configs, error))
    finally:
        _detach(profiler, evicting, flush)
        # Every config measured before an exit, a failed one's or another subscriber's too, is logged.
        observed(op, ba, {**timings, **_benchmarks(read, flushes)}, get_default_device())
    return Timings({**timings, **_benchmarks(read, flushes)})


def _benchmarks(read: Mapping, flushes: set[str]) -> dict[Hashable, tokamax.BenchmarkData]:
    """The configs `_measured` timed, each run's time the sum of its own kernels', the flush's left out."""
    benchmarks = {}
    for config, (runs, lowering, compiling) in read.items():
        times = tuple(sum(milliseconds for name, milliseconds in run if name not in flushes) for run in runs)
        benchmarks[config] = tokamax.BenchmarkData(compile_time_ms=compiling, lower_time_ms=lowering,
                                                   metadata={}, evaluation_times_ms=times)
    return benchmarks


def observed(op: KernelOp, ba: tokamax.BoundArguments, measured: Mapping[Hashable, object],
             device) -> None:
    """Log each config `_measured` timed or saw exhaust the device's resources, with the launch's features
    and estimate where it states them, where `DEW_TIMINGS` asks (`kernels.timings`): after the timing
    loop, on the measuring thread, so logging never runs beside a measurement and changes no pick. A
    failure to log raises, failing the tune: logging was asked for."""
    from . import timings as log
    if log.directory() is None:
        return
    from .templates import cost, features

    target = generation.gpu_target()
    call = json.dumps(_launch_json(ba), sort_keys=True)
    rows = []
    for config, measurement in measured.items():
        found = features.features(op, ba, config, target)
        timed = measurement if isinstance(measurement, tokamax.BenchmarkData) else None
        rows.append(log.observation(
            type(op).__name__, call, json.dumps(encoded(config), sort_keys=True), device=device,
            method=log.CUPTI_EVICTED, status='resource_exhausted' if timed is None else 'timed',
            samples_ms=() if timed is None else tuple(timed.evaluation_times_ms),
            lower_ms=None if timed is None else timed.lower_time_ms,
            compile_ms=None if timed is None else timed.compile_time_ms,
            estimate_s=None if found is None else cost.seconds(found.legacy, target),
            features=None if found is None else found.row(),
            error=measurement if isinstance(measurement, BaseException) else None))
    log.record(rows)


def _config(fastest: Timings | None) -> Hashable:
    return None if fastest is None else fastest.fastest_config


Keyed = dict[tuple[tokamax.Op, Hashable], tuple[tokamax.BoundArguments, Timings]]
"""Launches by their op and cache key, each with its timings (`_keyed`)."""

Launch = tuple[KernelOp, tokamax.BoundArguments]
"""One launch a lowering traced, its op and keyed arguments (`seen`)."""


def _keyed(timed) -> Keyed:
    """Each of the `(launch, timings)` pairs `timed` by its op and cache key, which binds the
    launch's arguments to this checkout's signature of its kernel (ValueError where they do not)."""
    return {(ba.op, ba.autotuning_cache_key): (ba, timings) for ba, timings in timed}


def _kernel_launches(launches: Keyed):
    """`launches`' launches of Dew's kernels, each `(op, key, fastest config's timing)`."""
    for (op, key), (_, timings) in launches.items():
        if isinstance(op, KernelOp) and timings.prune_errors():
            yield op, key, timings.prune()


def _read(path: str) -> tuple[Keyed, list]:
    """The launches stored at `path` that this checkout's kernels read
    (`_keyed`), and the stored entries they do not, as JSON: another
    checkout's launches (a kernel another branch adds, another branch's
    config fields, or another signature of a kernel, with an operand more or
    one fewer), which a write keeps as they are. An entry is read only where
    its argument names are exactly its kernel's parameters. Each launch is
    read alone, so one launch
    this checkout cannot read leaves the others readable. A missing file, or
    one that is not JSON, holds nothing."""
    if not os.path.exists(path):
        return {}, []
    try:
        with open(path) as file:
            stored = json.load(file)
        entries = list(stored["data"])
    except (ValueError, KeyError, TypeError) as error:
        _log.warning("the kernel picks at %s are not a store, so they are measured anew: %s", path, error)
        return {}, []
    read: Keyed = {}
    foreign = []
    for entry in entries:
        try:
            launch = _launch(entry), _timings(entry.get("timings") if isinstance(entry, dict) else None)
        except Unreadable:
            foreign.append(entry)
            continue
        read.update(_keyed([launch]))
    return read, foreign


def _stored_picks(kind: str) -> dict[tuple[tokamax.Op, str], _Picks]:
    """The picks stored for GPUs of `kind` that this checkout reads (`_read`), by op,
    none where the process keeps no picks (`dew.cache.keeps_picks`)."""
    if not cache.keeps_picks():
        return {}
    stored, _ = _read(cache.kernel_picks_path(kind))
    picks: dict[tuple[tokamax.Op, str], _Picks] = {}
    for op, key, fastest in _kernel_launches(stored):
        picks.setdefault((_identity(op), kind), _Picks())[key] = fastest
    return picks


def _store(device_kind: str, picks) -> None:
    """Merge `picks`, `(launch, Timings)` pairs measured on GPUs of `device_kind`, into the stored
    picks for that model, one writer at a time, a launch's newer timings replacing its older, the
    entries this checkout does not read kept as they are (`_read`). The file is replaced whole
    (`write_atomically`), so no reader on the shared filesystem sees a part of it."""
    path = cache.kernel_picks_path(device_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with filelock.FileLock(path + '.lock'):
        stored, foreign = _read(path)
        merged = {**stored, **_keyed(picks)}
        written = json.loads(dumps_picks(device_kind, merged.values()))
        write_atomically(path, json.dumps({**written, "data": [*written["data"], *foreign]}))


@contextlib.contextmanager
def launches() -> Iterator[None]:
    """Each kernel launch traced within carries tokamax's description of
    itself, which `tokamax.autotuning.get_bound_args` reads out of the
    lowered program (`dew.cache.KERNEL_LAUNCHES`)."""
    with cache.KERNEL_LAUNCHES(new_value=True):
        yield


class _Seen:
    """The launches a lowering traced under `seen`, each its op and keyed
    arguments; hashed by identity, as `dew.cache.KERNEL_SEEN` keys on it."""

    def __init__(self) -> None:
        self.found: list[Launch] = []


@contextlib.contextmanager
def seen() -> Iterator[list[Launch]]:
    """Each kernel launch traced within whose config is read from picks, its
    op and keyed arguments (`KernelOp.keyed`), recorded as it traces: a fresh
    collection each time, so a trace cached before traces again."""
    collected = _Seen()
    with cache.KERNEL_SEEN(collected):
        yield collected.found


def tune(lower: Callable[[], jax.stages.Lowered], *,
         measure: bool = False) -> tuple[jax.stages.Lowered, str]:
    """The program `lower()` lowers, with every launch's config settled across
    the pool, and its distinct launches as JSON, which `settled` settles again
    without a lowering. The lowering records its launches as it traces them
    (`seen`); process 0 takes its stored pick for each, with `measure`
    measuring and storing one for a launch without where it has its GPU to
    itself, and every process takes exactly process 0's configs (`_settle`).
    Where that changes a config the lowering traced, it lowers again: a
    launch's description in the program, which tokamax reads its launches out
    of (`launches`), keeps it from the persistent cache, so reading them so
    took a lowering of its own, about 62 s of each cold plan candidate's
    compile of the packed Qwen3.5-0.8B step on an A100 (ledger 561). Without
    `measure` a launch without a pick takes process 0's nearest stored shape's
    or its heuristic, and the process says once how many lack one and how to
    measure them."""
    if not generation.triton_runs():
        return lower(), "[]"
    tuned = cache.KERNEL_TUNES.get_global()
    with seen() as found:
        lowered = lower()
    distinct = _distinct(found)
    _settle(distinct, measure=measure)
    if cache.KERNEL_TUNES.get_global() != tuned:
        lowered = lower()
    return lowered, dumps_launches([ba for _, ba in distinct])


def _distinct(found: list[Launch]) -> list[Launch]:
    """Each keyed launch of `found` once, as first traced: a decoder's layers of one shape launch the
    same kernels at the same keys, which one decision, and one measurement, settles."""
    distinct: dict[tuple[tokamax.Op, Hashable], Launch] = {}
    for op, ba in found:
        distinct.setdefault((_identity(ba.op), ba.autotuning_cache_key), (op, ba))
    return list(distinct.values())


def settled(launches: str) -> bool:
    """Whether every launch of `launches` (`tune`'s) has a stored pick, with
    every process taking process 0's: what a compile whose launches are
    recorded settles instead of lowering for `tune`."""
    if not generation.triton_runs():
        return True
    found = [(op, ba) for ba in loads_launches(launches)
             if isinstance(op := ba.op, KernelOp)]
    return _settle(found, measure=False) == 0


def missing(lowered: jax.stages.Lowered) -> int:
    """How many launches of `lowered`, a program lowered under `launches`,
    have no stored pick on this process."""
    return sum(isinstance(op := ba.op, KernelOp) and _pick(op.keyed(ba)) is None
               for ba in _bound_args(lowered))


def _settle(found: list[Launch], *, measure: bool) -> int:
    """Take process 0's configs for the keyed launches `found` on every process,
    process 0 measuring a launch without a pick where `measure` and it has its
    GPU to itself: its picks, and for a launch it has none for, the nearest
    stored pick it reads, or none for the heuristic, which every process
    launches there (`_NEAREST`) and none stores, so the launch stays
    unmeasured. Each distinct launch is decided, and measured, once. Returns
    how many still have no pick, which the process logs once at INFO, with how
    to measure them."""
    found = _distinct(found)
    if not found:
        return 0
    kind = _kind(found[0][1])

    def decided() -> str | None:
        if jax.process_index() != 0:
            return None
        decisions, measured, nearest = [], [], []
        for op, ba in found:
            if (fastest := _pick(ba)) is not None:
                decisions.append((ba, fastest))
                continue
            if measure and cache.measures_picks():
                timings = _measured(op, ba)
                if timings.prune_errors():
                    measured.append((ba, timings))
                    decisions.append((ba, timings.prune()))
                    continue
                _log.warning("no config of %s compiled or ran; it keeps its heuristic: %s",
                             type(op).__name__, next(iter(timings.values()), None))
            if (near := op._stored_nearest(ba)) is not None:
                nearest.append((ba, near))
        if measured:
            _store(kind, measured)
        return json.dumps([json.loads(dumps_picks(kind, chosen)) for chosen in (decisions, nearest)])

    text = _coordination.broadcast_from_process_zero(_coordination.agreed("settling kernel configs", decided))
    picked, near = (_keyed([(_launch(entry), _timings(entry["timings"])) for entry in chosen["data"]])
                    for chosen in json.loads(text))
    settled: dict[tuple[tokamax.Op, Hashable], Timings] = {
        (op, key): fastest for op, key, fastest in _kernel_launches(picked)}
    def launched() -> list[Hashable]:
        # The configs the launches take, as the trace read them; tokamax's ignore-cache setting keeps
        # the picks None. One launch's pick moves another's nearest, so all are read before any moves.
        return [_config(_pick(ba) or op._nearest(ba)) for op, ba in found]

    traced = launched()
    for op, ba in found:
        picks, key = op.get_autotuning_cache(kind), ba.autotuning_cache_key
        picks.pop(key, None)
        if (fastest := settled.get((ba.op, key))) is not None:
            picks[key] = fastest
        _NEAREST.setdefault((_identity(op), kind), {})[key] = (
            near[(ba.op, key)][1] if (ba.op, key) in near else None)
    changed = launched() != traced
    if any(_coordination.from_every_process(changed)):
        # A program lowered again traces again, with the picks it now reads.
        cache.KERNEL_TUNES.set_global(cache.KERNEL_TUNES.get_global() + 1)
    unmeasured = sum(_pick(ba) is None for _, ba in found)
    if unmeasured and kind not in _told:
        _told.add(kind)
        # A heuristic config is a working default, not a fault, so the note is information, not a
        # warning: a run asks for measured picks itself (`Trainer(tune_kernels=True)`).
        _log.info("%d kernel launches have no config measured on %s, so they run the nearest measured "
                     "shape's or their heuristic: measure them on a GPU to yourself with "
                     "Trainer(tune_kernels=True) or `python tools/kernel_picks.py`", unmeasured, kind)
    return unmeasured


@contextlib.contextmanager
def pinned(configs: Mapping[type[KernelOp], Hashable]) -> Iterator[None]:
    """Each launch of a kernel `configs` names, traced within, takes that
    config where its space at the launch's key holds it, any stored pick
    or heuristic set aside: for the float64 gate at every config of a space,
    and for a lab that times one."""
    with cache.KERNEL_PINS(tuple(configs.items())):
        yield
