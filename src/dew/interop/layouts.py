"""The one layout a family or component reads and writes a checkpoint by.

A checkpoint names its tensors one way and Dew's variables place them another. A `Route` declares one
stored name pattern and the leaf pattern it fills, so one entry is both directions: a load reads a stored
name into its leaf (`Routes.read`), an export writes a leaf back under the name (`Routes.write`), and a
model holding a leaf no route writes is one its family cannot write. Tables compose: in order (`Table`),
respelled (`Spelled`: a family's own names onto the ones a shared table reads), under a prefix
(`Prefixed`: a wrapper's language model), and rebased (`Rebased`: a prediction depth's block, named as
the trunk's layer 0's). A tensor a release stores twice, a prediction depth's copy of the trunk's
embedding, is a `Copy`: read as a check of the leaf it copies, and written from it.

No direction is written beside another: a family declares its names here and both its reader and its
writer derive from them (`tests/test_layouts.py` holds every registered family to it).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path as FilePath
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from dew.interop.config_records import NativeFields
from dew.registry import Configured, configured

if TYPE_CHECKING:
    import numpy as np
    from flax import linen as nn

    from dew.interop.decoder_parts import SourceCode
    from dew.nn.backbones.causal_transformer import CausalTransformer

type Config = Mapping[str, object]
"""The native record a route reads its conditions from (`DecoderFields`): the translated config on a load,
the model's own on an export, so both directions read one record."""

type Path = tuple[str, ...]
"""A leaf's path in a model's variables, its collection first (`params`, `moe`, `constants`)."""

type Renames = tuple[tuple[str, str], ...]
"""A family's own names onto the ones a shared table reads, as (source, shared) pairs of dotted fragments:
a load respells left to right, an export right to left, so one table holds both directions (`renamed`)."""

type Condition = Callable[[Config, Mapping[str, str]], bool]
"""Whether a route holds for a record and its slots' leaf-side values (a layer's index, an expert's)."""


def renamed(name: str, renames: Renames, *, export: bool = False) -> str:
    """Respell `name` through `renames` in one pass over its dotted parts.

    At each part the first pair whose fragment starts there replaces it and the pass moves past it, so a
    respelled part is never read again and the reverse pass undoes the forward one.
    """
    pairs = [(old.split('.'), new) for old, new in
             ((shared, source) if export else (source, shared) for source, shared in renames)]
    parts, spelled, index = name.split('.'), [], 0
    while index < len(parts):
        for old, new in pairs:
            if parts[index:index + len(old)] == old:
                spelled.append(new)
                index += len(old)
                break
        else:
            spelled.append(parts[index])
            index += 1
    return '.'.join(spelled)


@dataclass(frozen=True)
class Slot:
    """What a route's placeholder holds: `stored`, a pattern of the stored name's text, and `leaf`, of the
    leaf path's (the stored one where None), the two the same text, or one an `offset` apart (a prediction
    depth stored past the trunk's layers, `after`), or a dotted run its path splits (`nested`)."""

    stored: str = r'\d+'
    leaf: str | None = None
    offset: Callable[[Config], int] | None = None
    """The number the stored value is past the leaf's, by the record (DeepSeek-V3's depth `d` stored as
    layer `num_layers + d`)."""
    stride: Callable[[Config], int] | None = None
    """The leaf's number's stride in the stored one, by the record (Nemotron-H's prediction step `d` of
    `b` blocks stored as blocks `d * b` to `d * b + b - 1`, each its own `offset`)."""
    nested: bool = False
    """Whether the value is a run of modules, dotted in the name and one path element each in the leaf."""
    names: Mapping[str, str] | None = None
    """Each stored spelling and the leaf's (`mapped`): a release's own name for a module."""

    @property
    def leaf_pattern(self) -> str:
        return self.stored if self.leaf is None else self.leaf

    def to_leaf(self, value: str, config: Config) -> str | None:
        """The leaf-side value of the stored `value`, None where it is no leaf's (a layer of the trunk where
        a depth is asked)."""
        if self.nested:
            return value.replace('.', '/')
        if self.names is not None:
            return self.names[value]
        if self.offset is None and self.stride is None:
            return value
        index = int(value) - (0 if self.offset is None else self.offset(config))
        stride = 1 if self.stride is None else self.stride(config)
        return str(index // stride) if index >= 0 and index % stride == 0 else None

    def to_stored(self, value: str, config: Config) -> str:
        """The stored-side value of the leaf-side `value`."""
        if self.nested:
            return value.replace('/', '.')
        if self.names is not None:
            return next(stored for stored, leaf in self.names.items() if leaf == value)
        if self.offset is None and self.stride is None:
            return value
        stride = 1 if self.stride is None else self.stride(config)
        return str(int(value) * stride + (0 if self.offset is None else self.offset(config)))


INDEX = Slot()
"""A number, the same in the name and the path: a layer's index, an expert's."""


def one_of(*values: str) -> Slot:
    """One of `values`, the same text in the name and the path: a projection of a set."""
    return Slot('|'.join(re.escape(value) for value in sorted(values, key=len, reverse=True)))


def _choice(values: list[str]) -> str:
    return '|'.join(re.escape(value) for value in sorted(values, key=len, reverse=True))


def mapped(names: Mapping[str, str]) -> Slot:
    """One of `names`' stored spellings in the name, its leaf's in the path: a release's own name for a
    module (DeepSeek V4's `wq_a` for the query's down projection, `q_a_proj`), each leaf's spelled once."""
    if len(set(names.values())) != len(names):
        raise ValueError(f"{dict(names)} spells a leaf twice")
    return Slot(_choice(list(names)), _choice(list(names.values())), names=MappingProxyType(dict(names)))


def after(count: Callable[[Config], int]) -> Slot:
    """A number stored `count` past its leaf's: a prediction depth stored as the layer after the trunk's."""
    return Slot(offset=count)


def nested(*modules: str) -> Slot:
    """A run of `modules`, each followed by its separator, possibly none: DeepSeek V4's compressor, its
    indexer and the indexer's scorer, each nesting the next's tensors under its name."""
    choice = '|'.join(re.escape(module) for module in modules)
    return Slot(rf'(?:(?:{choice})\.)*', rf'(?:(?:{choice})/)*', nested=True)


_PLACEHOLDER = re.compile(r'\{(\w+)\}')


def _compiled(template: str, slots: Mapping[str, Slot], *, leaf: bool) -> re.Pattern[str]:
    """`template`'s pattern, each `{slot}` its slot's pattern on that side (`INDEX` where undeclared)."""
    pieces, end = [], 0
    for match in _PLACEHOLDER.finditer(template):
        slot = slots.get(match[1], INDEX)
        pieces += [re.escape(template[end:match.start()]),
                   f'(?P<{match[1]}>{slot.leaf_pattern if leaf else slot.stored})']
        end = match.end()
    return re.compile(''.join(pieces) + re.escape(template[end:]))


@dataclass(frozen=True)
class Leaf:
    """Where a stored tensor goes: its leaf's `path`, None for a tensor that fills none of its own (a tied
    head, a copy checked against the leaf it copies)."""

    path: Path | None


class Routes(Protocol):
    """A layout's tensor names, both directions."""

    def read(self, name: str, config: Config) -> Leaf | None:
        """Where the stored tensor `name` goes; None where no route names it."""
        ...

    def write(self, path: Path, config: Config) -> str | None:
        """The name the leaf at `path` is stored under; None where no route writes it."""
        ...

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        """The tensors a release stores beside the leaves they copy, each with that leaf (`Copy`)."""
        ...


@dataclass(frozen=True, eq=False)
class Route:
    """One stored name pattern (`stored`, dotted) and the leaf pattern it fills (`leaf`, its collection
    first, `/`-separated), None where the tensor fills no leaf of its own (a tied head). `{name}` is a slot
    (`slots`, `INDEX` where undeclared), each held once by both sides; `when` says where the route holds,
    by the record and the slots' leaf-side values."""

    stored: str
    leaf: str | None
    slots: Mapping[str, Slot] = field(default_factory=lambda: MappingProxyType({}))
    when: Condition | None = None

    def __post_init__(self) -> None:
        names = _PLACEHOLDER.findall(self.stored)
        if len(set(names)) != len(names):
            raise ValueError(f"{self.stored} holds a slot twice")
        if self.leaf is not None and not set(_PLACEHOLDER.findall(self.leaf)) <= set(names):
            raise ValueError(f"{self.leaf} holds a slot {self.stored} does not")

    @functools.cached_property
    def _names_whole(self) -> bool:
        """Whether the leaf holds every slot the name does, so a leaf names its tensor: a route whose
        name holds more (a copy's index, `Copy`) reads, and writes only through what owns it."""
        return self.leaf is not None and set(_PLACEHOLDER.findall(self.leaf)) == set(
            _PLACEHOLDER.findall(self.stored))

    @functools.cached_property
    def _stored(self) -> re.Pattern[str]:
        return _compiled(self.stored, self.slots, leaf=False)

    @functools.cached_property
    def _leaf(self) -> re.Pattern[str] | None:
        return None if self.leaf is None else _compiled(self.leaf, self.slots, leaf=True)

    def _slot(self, name: str) -> Slot:
        return self.slots.get(name, INDEX)

    def read(self, name: str, config: Config) -> Leaf | None:
        match = self._stored.fullmatch(name)
        if match is None:
            return None
        values = {slot: self._slot(slot).to_leaf(text, config) for slot, text in match.groupdict().items()}
        if any(value is None for value in values.values()):
            return None
        held = {slot: value for slot, value in values.items() if value is not None}
        if self.when is not None and not self.when(config, held):
            return None
        return Leaf(None if self.leaf is None else tuple(self.leaf.format(**held).split('/')))

    def write(self, path: Path, config: Config) -> str | None:
        if self._leaf is None or not self._names_whole:
            return None
        match = self._leaf.fullmatch('/'.join(path))
        if match is None:
            return None
        values = match.groupdict()
        if self.when is not None and not self.when(config, values):
            return None
        return self.stored.format(**{slot: self._slot(slot).to_stored(value, config)
                                     for slot, value in values.items()})

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        return iter(())

    def moved(self, collection: str) -> Route:
        """This route with its leaf in `collection` (a router's state, which a training step moves)."""
        assert self.leaf is not None
        return Route(self.stored, collection + self.leaf[self.leaf.index('/'):], self.slots, self.when)


@dataclass(frozen=True, eq=False)
class Copy:
    """A tensor a release stores again where its leaf is stored already: `route`, read as a check of the
    leaf it copies (a prediction depth's copy of the trunk's embedding and head, a tied head a text
    model's reader takes from its own tensor), and written from that leaf, at each index of its `slot`
    where it holds one (`count` of them by the record)."""

    route: Route
    slot: str | None = None
    count: Callable[[Config], int] | None = None

    def read(self, name: str, config: Config) -> Leaf | None:
        return self.route.read(name, config)

    def write(self, path: Path, config: Config) -> str | None:
        return None

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        assert self.route.leaf is not None
        leaf = tuple(self.route.leaf.split('/'))
        if self.slot is None:
            if self.route.when is None or self.route.when(config, {}):
                yield self.route.stored, leaf
            return
        assert self.count is not None
        slot = self.route.slots.get(self.slot, INDEX)
        for index in range(self.count(config)):
            values = {self.slot: str(index)}
            if self.route.when is None or self.route.when(config, values):
                yield self.route.stored.format(**{self.slot: slot.to_stored(str(index), config)}), leaf


@dataclass(frozen=True, eq=False)
class Table:
    """`entries` in order: a name or a leaf goes by the first that routes it."""

    entries: tuple[Routes, ...]

    def __init__(self, *entries: Routes) -> None:
        object.__setattr__(self, 'entries', entries)

    def read(self, name: str, config: Config) -> Leaf | None:
        return _first(entry.read(name, config) for entry in self.entries)

    def write(self, path: Path, config: Config) -> str | None:
        return _first(entry.write(path, config) for entry in self.entries)

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        for entry in self.entries:
            yield from entry.copies(config)


def _first[Found](found: Iterator[Found | None]) -> Found | None:
    """The first of `found` that is not None, None where none is: the first entry of a table that routes."""
    return next((one for one in found if one is not None), None)


@dataclass(frozen=True, eq=False)
class Unwritten:
    """Names read by `path` alone (a name and the record into its leaf, None for one that fills none),
    which no route writes: a layout read but not yet declared both ways (a type verified from parts,
    MiMo-V2.6's audio tower), so a model of it is `Unwritable` where it would be written. A registered
    family's routes hold none (`tests/test_layouts.py`)."""

    path: Callable[[str, Config], tuple[str, ...] | None]

    def read(self, name: str, config: Config) -> Leaf | None:
        return Leaf(self.path(name, config))

    def write(self, path: Path, config: Config) -> str | None:
        return None

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        return iter(())


@dataclass(frozen=True, eq=False)
class When:
    """`routes` where `holds` holds of the record: a family's names that differ with its config (LFM3's
    norms with sandwich norms or without)."""

    holds: Callable[[Config], bool]
    routes: Routes

    def read(self, name: str, config: Config) -> Leaf | None:
        return self.routes.read(name, config) if self.holds(config) else None

    def write(self, path: Path, config: Config) -> str | None:
        return self.routes.write(path, config) if self.holds(config) else None

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        return self.routes.copies(config) if self.holds(config) else iter(())


@dataclass(frozen=True, eq=False)
class Spelled:
    """`routes` under a family's own names (`renames` onto the names `routes` read, `renamed`)."""

    renames: Renames
    routes: Routes

    def read(self, name: str, config: Config) -> Leaf | None:
        return self.routes.read(renamed(name, self.renames), config)

    def write(self, path: Path, config: Config) -> str | None:
        name = self.routes.write(path, config)
        return None if name is None else renamed(name, self.renames, export=True)

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        for name, path in self.routes.copies(config):
            yield renamed(name, self.renames, export=True), path


@dataclass(frozen=True, eq=False)
class Prefixed:
    """`routes` stored under `prefix` (a wrapper's `model.language_model.`, a bare model's absent one)."""

    prefix: str
    routes: Routes

    def read(self, name: str, config: Config) -> Leaf | None:
        if not name.startswith(self.prefix):
            return None
        return self.routes.read(name.removeprefix(self.prefix), config)

    def write(self, path: Path, config: Config) -> str | None:
        name = self.routes.write(path, config)
        return None if name is None else self.prefix + name

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        for name, path in self.routes.copies(config):
            yield self.prefix + name, path


@dataclass(frozen=True, eq=False)
class Rebased:
    """A block stored under `stored` (a prediction depth's `mtp.layers.{depth}.`), each of its tensors named
    past it as `routes` name the trunk's layer under `base` (`model.layers.0.`), its leaves under `leaf` in
    place of that layer's (`layers_0`): a depth's block reads and writes by the trunk's own names, in
    whichever collection the trunk's leaf sits."""

    stored: str
    leaf: str
    routes: Routes
    slots: Mapping[str, Slot] = field(default_factory=lambda: MappingProxyType({}))
    when: Condition | None = None
    base: str = 'model.layers.0.'
    layer: str = 'layers_0'

    @functools.cached_property
    def _stored(self) -> re.Pattern[str]:
        return _compiled(self.stored, self.slots, leaf=False)

    @functools.cached_property
    def _leaf(self) -> re.Pattern[str]:
        return _compiled(self.leaf, self.slots, leaf=True)

    def read(self, name: str, config: Config) -> Leaf | None:
        match = self._stored.match(name)
        if match is None:
            return None
        values = {slot: self.slots.get(slot, INDEX).to_leaf(text, config)
                  for slot, text in match.groupdict().items()}
        if any(value is None for value in values.values()):
            return None
        held = {slot: value for slot, value in values.items() if value is not None}
        if self.when is not None and not self.when(config, held):
            return None
        found = self.routes.read(self.base + name[match.end():], config)
        if found is None or found.path is None or found.path[1:2] != (self.layer,):
            return None
        return Leaf((found.path[0], *self.leaf.format(**held).split('/'), *found.path[2:]))

    def write(self, path: Path, config: Config) -> str | None:
        depth = self.leaf.count('/') + 1
        match = self._leaf.fullmatch('/'.join(path[1:1 + depth]))
        if match is None or (self.when is not None and not self.when(config, match.groupdict())):
            return None
        name = self.routes.write((path[0], self.layer, *path[1 + depth:]), config)
        if name is None or not name.startswith(self.base):
            return None
        return self.stored.format(**{slot: self.slots.get(slot, INDEX).to_stored(value, config)
                                     for slot, value in match.groupdict().items()}) + name[len(self.base):]

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        return iter(())


type ConfigRead[Model] = Callable[[Mapping[str, object], set[str], NativeFields[Model]], NativeFields[Model]]
"""A config part's read: the source config, the keys read so far (`used`, which it adds its own to), and the
model's record its earlier parts filled (`NativeFields`), which it returns filled further (a reader of a
whole record its own, as it builds it)."""

type ConfigWrite[Model] = Callable[[Model], Mapping[str, object]]
"""A config part's write: a model's keys in the family's config, refusing a model it cannot spell
(`decoder_parts.ExportRefused`)."""


class ConfigEntry[Model](Protocol):
    """One entry of a family's config layout (`ConfigLayout`), both ways: its keys read into the model's
    record (`read`), the record's fields written back as keys (`write`), and the keys it states by name
    (`keys`), which a layout that reads none of them marks read (`Nested.present`, a kind no layer holds)."""

    def read(self, hf_config: Mapping[str, object], used: set[str], record: NativeFields[Model], /
             ) -> NativeFields[Model]: ...

    def write(self, model: Model, /) -> Mapping[str, object]: ...

    def names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, eq=False)
class ConfigPart[Model]:
    """Config keys a family reads and writes together (`read`, `write`), declared side by side so neither
    changes alone: the shared decoder's geometry, a family's routed experts and their schedule, its rotary
    records, its layer kinds. A family composes its parts (`ConfigLayout`), and its reader and its writer
    are theirs (`DecoderFamily.translate_config`, `export_fields`)."""

    read: ConfigRead[Model]
    write: ConfigWrite[Model]
    keys: tuple[str, ...] = ()
    """The keys its read reads (`names`), which it marks read itself (`used`) where it reads them."""

    def names(self) -> tuple[str, ...]:
        return self.keys


@dataclass(frozen=True)
class Required:
    """The default of a key a config must state, whose class has none (`REQUIRED`)."""


REQUIRED = Required()
"""A key a config must state: its class has no default for it."""


@dataclass(frozen=True, eq=False)
class ConfigField[Model]:
    """One config key (`key`) and the record field it states (`field`), both ways: read as `parse` reads
    the value stated, or `default` where the config omits it (the config class's own), and written as
    `value` gives it of the model (the field itself where None), spelled by `spell` (as stated where
    None)."""

    key: str
    field: str
    default: Configured | Required = REQUIRED
    parse: Callable[[Configured, str], Configured] | None = None
    value: Callable[[Model], object] | None = None
    spell: Callable[[object], object] | None = None
    aliases: tuple[str, ...] = ()
    """Other keys a config may state it under, read where `key` is absent, in order; written as `key`."""

    def read(self, hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        keys = (self.key, *self.aliases)
        used.update(keys)
        key = next((key for key in keys if key in hf_config), None)
        if key is None and self.default is REQUIRED:
            raise ValueError(f"{self.key} is required: the config states none, and its class has no default")
        stated = configured(self.default if key is None else hf_config[key])
        record[self.field] = stated if self.parse is None else self.parse(stated, key or self.key)
        return record

    def write(self, model: Model) -> Mapping[str, object]:
        held = getattr(model, self.field) if self.value is None else self.value(model)
        return {self.key: held if self.spell is None else self.spell(held)}

    def names(self) -> tuple[str, ...]:
        return (self.key, *self.aliases)


def constant[Model](field: str, *, value: Configured) -> ConfigPart[Model]:
    """A field a family's reference fixes, which its config states nowhere: read as `value`, written as
    nothing (the family matches only a model that holds it, `DecoderFamily.matches`)."""
    def read(hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        record[field] = value
        return record
    return ConfigPart(read, lambda model: {})


def unread[Model](*keys: str) -> ConfigPart[Model]:
    """Keys a family's config states that its reference reads to no effect (a version stamp, an
    initializer's range), marked read and written as nothing."""
    def read(hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        used.update(keys)
        return record
    return ConfigPart(read, lambda model: {}, keys=keys)


@dataclass(frozen=True, eq=False)
class Nested[Model, Part]:
    """A record a model holds under one field (`field`: its routed experts, its mixer), its own config
    keys `layout`'s: read into a record of `kind` beside the model's (`build` makes the field's value of
    it, a registry record where the field holds one of several classes), written from the value the
    model holds (`held`; nothing where it holds none)."""

    field: str
    kind: type[Part]
    layout: ConfigLayout[Part]
    held: Callable[[Model], Part | None]
    build: Callable[[NativeFields[Part]], Configured] | None = None
    present: Callable[[Mapping[str, object]], bool] | None = None
    """Whether the source config builds one (a Qwen MoE whose schedule routes no layer builds none), its
    keys then read as nothing; always where None."""

    def read(self, hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        if self.present is not None and not self.present(hf_config):
            used.update(self.layout.names())
            record[self.field] = None
            return record
        part = self.layout.read(hf_config, used, NativeFields(self.kind, {}))
        record[self.field] = part if self.build is None else self.build(part)
        return record

    def write(self, model: Model) -> Mapping[str, object]:
        part = self.held(model)
        return {} if part is None else self.layout.write(part)

    def names(self) -> tuple[str, ...]:
        return self.layout.names()


@dataclass(frozen=True, eq=False)
class Restated[Model]:
    """`layout` under a family's own keys: each shared key read from and written to the family's
    (`own`, shared to own), and the keys a family's class fixes (`fixed`, shared to value) read at their
    value whatever a config states of them (a family's own part refuses a statement its reference would
    not compute) and written as nothing (DeepSeek's one key/value head of its latent attention)."""

    layout: ConfigLayout[Model]
    own: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    fixed: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    defaults: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    """Shared keys at the class's own default where the config states none."""
    undeclared: frozenset[str] = frozenset()
    """Shared keys the family's class declares none of: read where a config states them, written as
    nothing."""

    def read(self, hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        view = {**self.defaults, **hf_config}
        for shared, own in self.own.items():
            if own in view:
                view[shared] = view[own]
        view.update(self.fixed)
        read: set[str] = set()
        record = self.layout.read(view, read, record)
        used.update(self.own.get(key, key) for key in read if key not in self.fixed)
        # The class computes a fixed key's value whatever a config states of it.
        used.update(key for key in self.fixed if key in hf_config)
        return record

    def write(self, model: Model) -> Mapping[str, object]:
        return {self.own.get(key, key): value for key, value in self.layout.write(model).items()
                if key not in self.fixed and key not in self.undeclared}

    def names(self) -> tuple[str, ...]:
        return tuple(self.own.get(key, key) for key in self.layout.names() if key not in self.fixed)


@dataclass(frozen=True, eq=False)
class Choice[Model]:
    """One of two layouts a family's config takes, both ways: `routed` where the source config states it
    (`reads`) and where the model holds it (`writes`), `plain` otherwise (LLaDA's dense release beside its
    MoE one under one model_type)."""

    reads: Callable[[Mapping[str, object]], bool]
    writes: Callable[[Model], bool]
    routed: ConfigLayout[Model]
    plain: ConfigLayout[Model]

    def read(self, hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        return (self.routed if self.reads(hf_config) else self.plain).read(hf_config, used, record)

    def write(self, model: Model) -> Mapping[str, object]:
        return (self.routed if self.writes(model) else self.plain).write(model)

    def names(self) -> tuple[str, ...]:
        return (*self.routed.names(), *self.plain.names())


@dataclass(frozen=True, eq=False)
class ConfigLayout[Model]:
    """A family's config, its entries in order (`ConfigEntry`: a `ConfigPart`, a `ConfigField`, another
    layout): read, each filling the record (`read`), and written, each stating its keys (`write`, a later
    entry's keys over an earlier's)."""

    parts: tuple[ConfigEntry[Model], ...]

    def __init__(self, *parts: ConfigEntry[Model]) -> None:
        object.__setattr__(self, 'parts', parts)

    def read(self, hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        for part in self.parts:
            record = part.read(hf_config, used, record)
        return record

    def names(self) -> tuple[str, ...]:
        """The config keys its entries state by name, each under every spelling it reads."""
        return tuple(key for part in self.parts for key in part.names())

    def write(self, model: Model) -> Mapping[str, object]:
        written: dict[str, object] = {}
        for part in self.parts:
            written.update(part.write(model))
        return written


@dataclass(frozen=True)
class Unwritable:
    """What a model's layout cannot write (`reason`): falsy, so a caller asks once and writes natively
    where its source layout is this."""

    reason: str

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class SourceLayout:
    """A model's release layout (`source_layout`): the config.json it is written with (`config`), each
    component folder a bundle keeps beside it, by the folder, with its config (`folders`; none for a
    decoder), its variables in the release's names (`tensors`), and the modules its family's own code
    is (`assets`), which a release transformers has no class for writes beside its weights."""

    config: Mapping[str, object]
    folders: Mapping[str, Mapping[str, object]]
    writer: Callable[[Mapping[str, object]], Mapping[str, np.ndarray]]
    code: SourceCode | None = None
    """The family's `SourceCode`, whose modules `assets` writes; None for a family transformers implements."""

    def tensors(self, variables: Mapping[str, object]) -> Mapping[str, np.ndarray]:
        """`variables` (the model's, every collection) as the release's tensors, by their names, each built
        when it is read (`safetensors_io.LazyTensors`)."""
        return self.writer(variables)

    def assets(self, directory: str | PathLike) -> None:
        """Write the modules the family's own code is into `directory`, beside its weights."""
        if self.code is None:
            return
        from dew.interop import sources

        for name, module in sorted(sources.code_modules(self.code).items()):
            written = FilePath(directory) / name
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_bytes(module.read_bytes())


def source_layout(model: nn.Module, *,
                  config: Mapping[str, object] | None = None) -> SourceLayout | Unwritable:
    """The layout `model` is written in as its family's release, or what it cannot write (`Unwritable`):
    the one answer every writer asks (a run's export, a bundle's save, a decision's release). A decoder's
    family is the one its fields match (`hf_decoders.export_config`; a differential extension's or a low-rank
    adapter's, the decoder's they extend, folded or merged as it is written), its config the one it writes
    there, read back as the same model (`refuse_lossy_export`), and every parameter and router state a
    route of the family's writes (`decoder_parts.unrouted`). `config` is a loaded source's own config.json,
    which a save writes as published, its representability still the model's."""
    from dew.interop import hf_decoders
    from dew.lora import unadapted
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.protocols import MediaWrapper
    from dew.registry import from_record

    if isinstance(model, MediaWrapper):
        from dew.interop.wrappers import wrapper_layout

        return wrapper_layout(from_record(MultimodalTransformer, model), config=config)
    try:
        decoder = from_record(CausalTransformer, unadapted(model))
    except ValueError as error:
        return Unwritable(f"no family writes a {type(model).__name__}: {error}")
    if decoder.cross_attention is not None:
        return Unwritable("cross-attention has no layout in any family")
    # A differential extension is folded into each layer's output projection and an adapter's factors
    # merged into their kernels as it is written (`Pretrained.export`): the layout is the decoder's they
    # extend.
    base = decoder.with_dex(None)
    held = _decoder_layout(base)
    if isinstance(held, Unwritable):
        return held

    def tensors(variables: Mapping[str, object]) -> Mapping[str, np.ndarray]:
        # The bundle's own export: its storage read as layers, the extension folded, the factors merged,
        # the family's writer.
        from dew.interop.pretrained import PretrainedDecoder

        return PretrainedDecoder.from_model(decoder, variables).bound(model, variables).export()

    return SourceLayout(held if config is None else config, {}, tensors, hf_decoders.release_code(base))


def _decoder_layout(decoder: CausalTransformer) -> Mapping[str, object] | Unwritable:
    """The config a plain decoder (`source_layout`'s) is written with, or what its family's layout cannot
    hold of it: its config refused, or a parameter or router state no route writes, its leaves' paths
    abstract (`jax.eval_shape`: no weight is drawn)."""
    import jax
    import jax.numpy as jnp

    from dew.interop import decoder_parts, hf_decoders

    try:
        written = hf_decoders.export_config(decoder)
    except decoder_parts.ExportRefused as refusal:
        return Unwritable(str(refusal))
    family = hf_decoders.families()[str(written['model_type'])]
    # A sequence longer than its prediction depths, which each read a token past the trunk's.
    tokens = jnp.ones((1, decoder.num_nextn_predict_layers + 2), jnp.int32)
    shapes = jax.eval_shape(lambda: decoder.init(jax.random.key(0), tokens))
    missing = decoder_parts.unrouted(family, decoder, shapes)
    if missing:
        return Unwritable(f"{', '.join('/'.join(path) for path in missing)} has no tensor in the "
                          f"{family.export_model_type} layout")
    return written


__all__ = [
    "INDEX",
    "REQUIRED",
    "Choice",
    "Condition",
    "Config",
    "ConfigEntry",
    "ConfigField",
    "ConfigLayout",
    "ConfigPart",
    "ConfigRead",
    "ConfigWrite",
    "Copy",
    "Leaf",
    "Nested",
    "Path",
    "Prefixed",
    "Rebased",
    "Renames",
    "Required",
    "Restated",
    "Route",
    "Routes",
    "Slot",
    "SourceLayout",
    "Spelled",
    "Table",
    "Unwritable",
    "Unwritten",
    "When",
    "after",
    "constant",
    "mapped",
    "nested",
    "one_of",
    "renamed",
    "source_layout",
    "unread",
]
