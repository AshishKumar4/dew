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
from typing import Protocol

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
        if self.leaf is not None and sorted(_PLACEHOLDER.findall(self.leaf)) != sorted(names):
            raise ValueError(f"{self.stored} and {self.leaf} hold different slots")

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
        if self._leaf is None:
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
        return next((found for entry in self.entries if (found := entry.read(name, config)) is not None),
                    None)

    def write(self, path: Path, config: Config) -> str | None:
        return next((name for entry in self.entries if (name := entry.write(path, config)) is not None),
                    None)

    def copies(self, config: Config) -> Iterator[tuple[str, Path]]:
        for entry in self.entries:
            yield from entry.copies(config)


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
    writer: Callable[[Mapping[str, object]], Mapping[str, object]]
    code: object | None = None
    """The family's `SourceCode`, whose modules `assets` writes; None for a family transformers implements."""

    def tensors(self, variables: Mapping[str, object]) -> Mapping[str, object]:
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


def source_layout(model: object, *, config: Mapping[str, object] | None = None) -> SourceLayout | Unwritable:
    """The layout `model` is written in as its family's release, or what it cannot write (`Unwritable`):
    the one answer every writer asks (a run's export, a bundle's save, a decision's release). A decoder's
    family is the one its fields match (`hf_decoders.export_config`; a differential extension's, the
    decoder's it extends, folded as it is written), its config the one it writes there,
    read back as the same model (`refuse_lossy_export`), and every parameter and router state a route of
    the family's writes (`decoder_parts.unrouted`). `config` is a loaded source's own config.json, which a
    save writes as published, its representability still the model's."""
    import jax
    import jax.numpy as jnp

    from dew.interop import decoder_parts, hf_decoders
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import from_record

    if not isinstance(model, CausalTransformer):
        return Unwritable(f"no family writes a {type(model).__name__}")
    decoder = from_record(CausalTransformer, model)
    if decoder.cross_attention is not None:
        return Unwritable("cross-attention has no layout in any family")
    # A differential extension is folded into each layer's output projection as it is written
    # (`Pretrained.export`): the layout is the decoder's it extends.
    base = decoder.with_dex(None)
    try:
        written = hf_decoders.export_config(base)
    except decoder_parts.ExportRefused as refusal:
        return Unwritable(str(refusal))
    family = hf_decoders.families()[str(written['model_type'])]
    shapes = jax.eval_shape(lambda: base.init(jax.random.key(0), jnp.ones((1, 1), jnp.int32)))
    missing = decoder_parts.unrouted(family, base, shapes)
    if missing:
        return Unwritable(f"{', '.join('/'.join(path) for path in missing)} has no tensor in the "
                          f"{family.export_model_type} layout")

    def tensors(variables: Mapping[str, object]) -> Mapping[str, object]:
        # The bundle's own export: its storage read as layers, the extension folded, the family's writer.
        from dew.interop.pretrained import PretrainedDecoder

        return PretrainedDecoder.from_model(decoder, variables).export()

    return SourceLayout(written if config is None else config, {}, tensors, hf_decoders.release_code(base))


__all__ = ["INDEX", "Condition", "Config", "Copy", "Leaf", "Path", "Prefixed", "Rebased", "Renames",
           "Route", "Routes", "Slot", "SourceLayout", "Spelled", "Table", "Unwritable", "When", "after",
           "mapped", "nested", "one_of", "renamed", "source_layout"]
