"""Every release Dew writes, planned once and written once (EX02).

A release is a model part, a decision head's own files where it holds one, its task's assets and one record,
the release contract's (`dew.config.Release`):

- the model part is the model in its family's source layout, as a bundle writes it (`Pretrained.parts`,
  `Placed`), where its family's layout states the model (`dew.interop.layouts.source_layout`), else the
  run's native release, its step's inference declaration beside its variables (`Native`): which of the two
  is the layout's answer, never a class's name or an attempt's failure;
- the head's files are its decision format's (`dew.decision.formats`: Laya's, the joint head's, the slot
  readout's);
- the assets are the task's tokenizer and generation config;
- the record states what none of these files state (`dew.interop.release_record`).

Every saver writes a plan, and every plan is written by `ReleasePlan.write`: `dew export`
(`dew.interop.saved_sources.export_run`), `Pretrained.save`, `Decide.save_pretrained`,
`SavedModel.save_pretrained`; `ReleasePlan.tensors` is what a release's weights are by name, which
`Pretrained.export` returns and a weight push sends.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import numpy as np
from etils import epath

from dew._coordination import agreed_reading, writes_weights
from dew._files import Entries, Replacements, is_uri, same_entries, swapped, write_atomically
from dew.interop.release_record import RELEASE_FILE, Release
from dew.interop.safetensors_io import (
    CONFIG_FILE,
    MAX_SHARD_SIZE,
    LazyTensors,
    flatten,
    joined_tensors,
    save_sharded,
    tensor_specs,
    weight_stem,
    write_file,
)

if TYPE_CHECKING:
    from dew.batch import Variables
    from dew.interop.hf_decoders import ExportTokenizer
    from dew.interop.layouts import Unwritable
    from dew.interop.pretrained import Pretrained

type Part = tuple[Mapping[str, object], Mapping[str, np.ndarray]]
"""One folder of a model's release: its config and its tensors by their names there."""


def published_release(directory: str | Path, write: Callable[[Path], None], *, fresh: bool = False) -> None:
    """Publish what `write` writes into the directory it is given as the release at `directory`, the whole
    release one generation: built in a sibling directory and exchanged with the one `directory` held in one
    call (`dew._files.swapped`), so a save that fails at any file, a config or a shard, leaves the earlier
    release whole (row 13730), and a `directory` holding entries no release writes or owns
    (`foreign_entries`) is refused, those named. A pool's processes agree where each writes, then their
    writes, then the publisher's replacement, in turn: a process that fails at any of them is heard by the
    others at the next round any takes, after which none takes another (`_coordination._unheard`). A pool's
    process that publishes none (`writes_weights`) writes in place, to a destination of its own; a location
    with a scheme, which has no rename, is written file by file; and a `fresh` directory, one its caller
    made empty and private for this write alone and publishes itself, is written in place."""
    root = Path(directory)
    in_place = fresh or is_uri(str(directory)) or not writes_weights()
    with contextlib.ExitStack() as staged:
        target = agreed_reading(lambda: root if in_place else staged.enter_context(
            swapped(root, foreign_entries, replaces=replaced_entries)))
        agreed_reading(lambda: write(target))
        agreed_reading(staged.close)


@contextlib.contextmanager
def pooled_destination(destination: str | Path, *, publishes: bool, shared: bool = True,
                       stage: Callable[[Path], contextlib.AbstractContextManager[Path]] | None = None
                       ) -> Iterator[str | Path]:
    """The directory this process writes a pool's release into, inside its reading scope
    (`dew._coordination.reading_home`), every process taking the same rounds: made in one the pool agrees
    (`agreed_reading`), the publisher's `destination` (or the staging `stage` makes of it), another
    process's a scratch directory discarded where the pool shares `destination`, so none touches the live
    one, and its own `destination` where it does not (`shared`); and committed in another as the
    block ends, the staging published and the scratch removed, its removal failing nothing. A process that
    fails making its directory, writing, or committing is heard by the others at that boundary, none left
    waiting in a round alone (train 1406's review). One lifecycle for every pooled release: a saved run's
    (`saved_sources.export_run`) and a rollout's weights (`inference.rollouts.SafetensorsReload`). A
    location with a scheme is the publisher's as named, no stage made of it."""
    with contextlib.ExitStack() as held:
        def made() -> str | Path:
            if not publishes and shared:
                return Path(held.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True)))
            if not publishes:
                return destination
            return destination if stage is None else held.enter_context(stage(Path(destination)))

        yield agreed_reading(made)
        agreed_reading(held.close)


def replaced_entries(written: Entries, held: Entries) -> Replacements:
    """A release's entries grouped with the earlier entries they replace (`swapped`'s `replaces`): the
    shards of its weights of one stem in one folder with every shard of that stem the earlier tree held
    there, however the two shard them, so a resharded release keeps the access its earlier weights allowed
    (train 1407's review: `model.safetensors` replaced by `model-00001-of-00002.safetensors`); any other
    entry with the same entry, as `same_entries` groups it."""
    def stem_of(entries: Entries) -> dict[tuple[Path, str], list[Path]]:
        grouped: dict[tuple[Path, str], list[Path]] = {}
        for entry, kind in entries.items():
            if kind == "file" and (stem := weight_stem(entry.name)) is not None:
                grouped.setdefault((entry.parent, stem), []).append(entry)
        return grouped

    shards, earlier = stem_of(written), stem_of(held)
    sharded = {entry for group in shards.values() for entry in group}
    return [*[(entries, held_entries) for entries, held_entries in same_entries(written, held)
              if entries[0] not in sharded],
            *[(group, earlier.get(key, [])) for key, group in shards.items()]]


def foreign_entries(written: Entries, held: Entries, directory: Path) -> set[Path]:
    """What an earlier tree at `directory` holds that a release neither writes, as the same kind of entry,
    nor owns: an earlier export's weights are files its own only in a folder it writes its own weights of
    that stem in (`weight_stem`); the release record (`RELEASE_FILE`) its own whether it writes one or drops
    it (`Release.write`); and the sidecars the earlier release's config declares (`hf_decoders.sidecars`:
    Inkling 2's draft, which a trunk-only export drops)."""
    from dew.interop.hf_decoders import sidecars

    stems = {(entry.parent, stem) for entry, kind in written.items()
             if kind == "file" and (stem := weight_stem(entry.name)) is not None}
    unmatched = {entry: kind for entry, kind in held.items() if written.get(entry) != kind}
    loose = {entry for entry, kind in unmatched.items() if kind == "file" and entry != Path(RELEASE_FILE)
             and (entry.parent, weight_stem(entry.name)) not in stems}
    # The earlier config read only where a regular file is left to own, and only a regular file: a FIFO or
    # another special entry is foreign, never opened.
    if loose and held.get(Path("config.json")) == "file":
        loose -= set(map(Path, sidecars(json.loads((directory / "config.json").read_text()))))
    return loose | {entry for entry, kind in unmatched.items() if kind != "file"}


@dataclasses.dataclass(frozen=True)
class Placed:
    """A model in its family's source layout: each of its parts (`Pretrained.parts`), the root's under
    "", and its source's own files (`files`, a family's remote code, each at its path in the release).

    A decision format places the root part as its files hold it (`dew.decision.formats`): its config in
    `folder` with what the format `stated` beside the family's own, and its tensors in the release's own
    weights, each named with `prefix` for `strip` (Laya's ModernBERT encoder, `encoder/config.json` and its
    `model.` tensors as `encoder.` beside the head's in one `model.safetensors`)."""

    parts: Mapping[str, Part]
    files: Mapping[str, Path] = dataclasses.field(default_factory=dict)
    folder: str = ""
    prefix: str = ""
    strip: str = ""
    stated: Mapping[str, object] = dataclasses.field(default_factory=dict)
    sidecars: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """The files beside the root's weights that hold more of its tensors (`DecoderFamily.sidecars`, Inkling
    2's `mtp.safetensors`), each by the prefix its tensors carry in the root's table: written there under
    their own names, the prefix off, as a load reads them back (`sources.with_sidecars`)."""

    def root(self) -> Part:
        """The root part as the release holds it: its config with what the format states, its tensors
        under the format's names."""
        config, tensors = self.parts[""]
        return {**config, **self.stated}, _renamed(tensors, self.strip, self.prefix)

    def tensors(self) -> Mapping[str, np.ndarray]:
        """Every tensor of this part by its name in the release: the root's as `root` names them, another
        folder's under `<folder>/` (`dew.interop.decoder_parts.folder_name`)."""
        tables = {"the model": self.root()[1]}
        tables.update({f"its {folder}": _renamed(tensors, "", f"{folder}/")
                       for folder, (_, tensors) in self.parts.items() if folder})
        return joined_tensors(tables)


@dataclasses.dataclass(frozen=True)
class Native:
    """A model no family's layout states: its `variables`, named `<collection>/<path>`, and the files its
    task keeps by the step (`records`, each copied to its place), beside its run's inference declaration
    in the release's record (`native_plan`), which `dew.inference.models.ReleasedRun` reads back as the
    run."""

    variables: Variables
    records: Sequence[tuple[epath.Path, PurePath]] = ()

    def tensors(self) -> Mapping[str, np.ndarray]:
        return flatten(self.variables)


@dataclasses.dataclass(frozen=True)
class HeadFiles:
    """A decision head's own files in its release format (`dew.decision.formats`): its `tensors` by name, in
    a safetensors file of their own (`weights`, with `metadata`), in the release's own weights beside the
    model's (`merged`, Laya's) or in a file of the format's (`files`, a readout's torch file); and its other
    files, each written by its writer at its name."""

    tensors: Mapping[str, np.ndarray]
    weights: str | None = None
    merged: bool = False
    files: Mapping[str, Callable[[Path], None]] = dataclasses.field(default_factory=dict)
    metadata: Mapping[str, str] = dataclasses.field(default_factory=lambda: {"format": "pt"})


@dataclasses.dataclass(frozen=True)
class Assets:
    """A release's task assets, which `dew.interop.hf_decoders.save_export_assets` writes into `folder`: its
    tokenizer (a name, a tokenizer that writes its files, or None) and its generation config, where its
    format holds one; `named` where that config has a field to name a byte vocabulary by."""

    tokenizer: str | ExportTokenizer | None = None
    folder: str = ""
    generation_config: Mapping[str, object] | None = None
    named: bool = True


@dataclasses.dataclass(frozen=True)
class ReleasePlan:
    """One release: its model part, a head's files where it holds one, its assets and its record."""

    model: Placed | Native
    head: HeadFiles | None = None
    assets: Assets | None = None
    envelope: Release = dataclasses.field(default_factory=Release)

    def tensors(self) -> Mapping[str, np.ndarray]:
        """Every tensor the release holds, by its name in the release's files, each built as it is read: the
        model part's and the head's, a name both hold refused before any is read (`joined_tensors`)."""
        if self.head is None:
            return self.model.tensors()
        where = self.head.weights or ("beside the model" if self.head.merged else "its own file")
        return joined_tensors({"the release's model": self.model.tensors(),
                               f"its head ({where})": self.head.tensors})

    def write(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE,
              fresh: bool = False) -> None:
        """Write the release as `directory`: the assets, the model part, the head's files and the record,
        the weights in shards of at most `max_shard_size`, the whole release one generation. It is built in
        a sibling directory and exchanged with the one `directory` held in one call (`dew._files.swapped`),
        so a save that fails at any file leaves the earlier release whole, never a draft beside another
        generation's trunk (row 13730); a reader that reads across a save synchronizes with it. A
        `directory` holding files or folders no export writes or owns (`foreign_entries`), at any depth, is
        refused, those entries named. A pool's process that reads its share and publishes none
        (`writes_weights`) writes the release's small files in place, to a destination of its own
        (`saved_sources.export_run` discards one the pool shares), every process agreeing its setup, its
        writes and the publisher's replacement in turn; a location with a scheme, which has no rename, is
        written file by file; and a `fresh` directory, one its caller made empty and private for this write
        alone, which it publishes itself (a cache release's staging, `saved_sources.export_run`), is written
        in place."""
        published_release(directory, lambda target: self._write(target, max_shard_size), fresh=fresh)

    def _write(self, root: Path, max_shard_size: int | str) -> None:
        from dew.interop.hf_decoders import save_export_assets

        root.mkdir(parents=True, exist_ok=True)
        if self.assets is not None:
            save_export_assets(root / self.assets.folder, tokenizer=self.assets.tokenizer,
                               generation_config=self.assets.generation_config, named=self.assets.named)
        if isinstance(self.model, Native):
            save_sharded(self.model.tensors(), root, max_shard_size)
            for source, place in self.model.records:
                (root / place).parent.mkdir(parents=True, exist_ok=True)
                (root / place).write_bytes(source.read_bytes())
        else:
            self._placed(self.model, root, max_shard_size)
        if self.head is not None:
            if self.head.weights is not None:
                write_file(self.head.tensors, root / self.head.weights, self.head.metadata)
            for name, written in self.head.files.items():
                written(root / name)
        self.envelope.write(root)

    def _placed(self, model: Placed, root: Path, max_shard_size: int | str) -> None:
        for name, source in sorted(model.files.items()):
            (root / model.folder / name).parent.mkdir(parents=True, exist_ok=True)
            (root / model.folder / name).write_bytes(source.read_bytes())
        for folder, (config, tensors) in model.parts.items():
            if folder:
                _part(root / folder, config, tensors, max_shard_size)
        config, tensors = model.root()
        tensors = _sidecars_written(tensors, model.sidecars, root)
        if self.head is not None and self.head.merged:
            tensors = joined_tensors({"the model": tensors, "its head": self.head.tensors})
        # The root's weights at the release's root; its config there or in the format's folder.
        save_sharded(tensors, root, max_shard_size)
        (root / model.folder).mkdir(parents=True, exist_ok=True)
        write_atomically(str(root / model.folder / CONFIG_FILE), json.dumps(config, indent=2))


def _sidecars_written(tensors: Mapping[str, np.ndarray], sidecars: Mapping[str, str],
                      root: Path) -> Mapping[str, np.ndarray]:
    """Each sidecar's tensors of `tensors` (`Placed.sidecars`) written to its own file at `root`, the
    prefix off their names, and the rest returned for the root's shards, each read as it is written; the
    release they are written into published whole (`ReleasePlan.write`)."""
    specs = tensor_specs(tensors)
    taken: set[str] = set()
    for filename, prefix in sidecars.items():
        names = {name.removeprefix(prefix): name for name in specs if name.startswith(prefix)}
        if names:
            write_file(LazyTensors({name: specs[source] for name, source in names.items()},
                                   lambda name, names=names: tensors[names[name]]),
                       root / filename, {"format": "pt"})
            taken.update(names.values())
    if not taken:
        return tensors
    return LazyTensors({name: spec for name, spec in specs.items() if name not in taken},
                       lambda name: tensors[name])


def _part(directory: Path, config: Mapping[str, object], tensors: Mapping[str, np.ndarray],
          max_shard_size: int | str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_sharded(tensors, directory, max_shard_size)
    write_atomically(str(directory / CONFIG_FILE), json.dumps(config, indent=2))


def _renamed(tensors: Mapping[str, np.ndarray], strip: str, prefix: str) -> Mapping[str, np.ndarray]:
    """`tensors` named with `prefix` for `strip`, each read from its own as it is read."""
    if not strip and not prefix:
        return tensors
    specs = tensor_specs(tensors)
    names = {f"{prefix}{name.removeprefix(strip)}": name for name in specs}
    if len(names) < len(specs):
        raise ValueError(f"two tensors are one name with {strip!r} taken off")
    return LazyTensors({name: specs[source] for name, source in names.items()},
                       lambda name: tensors[names[name]])


def writable(plan: ReleasePlan | Unwritable, subject: str) -> ReleasePlan:
    """`plan`, refused where it is why no release writes `subject` (`Unwritable`, naming why)."""
    if not isinstance(plan, ReleasePlan):
        raise ValueError(f"no release writes this {subject}: {plan.reason}")
    return plan


def bundle_plan(bundle: Pretrained, variables: Mapping[str, object] | None = None) -> ReleasePlan:
    """A bundle's release over its variables or `variables`, a trainer's split of them: its model in its
    source layout as the bundle writes it (`Pretrained.parts`, with the source's own modules a load ran),
    its processor's or tokenizer's files and generation config, and the record of the run behind it where
    it has one (`dew.interop.release.release_record`)."""
    from dew.inference.tasks import TextSettings
    from dew.interop import hf_decoders as decoders, sources
    from dew.interop.generation_config import unconfigured
    from dew.interop.release import release_record

    values = bundle.variables if variables is None else variables
    # A run's policy transformers' generation config holds no field of, which its reader would run without:
    # refused before a file is written (`unconfigured`), the bundle itself serving it natively.
    dropped = unconfigured(bundle.settings.sampling) if isinstance(bundle.settings, TextSettings) else []
    if dropped:
        raise ValueError(f"this run's policy holds {' and '.join(dropped)}, which transformers' generation "
                         "config holds no field of: release a policy without it, or the run natively")
    # Either may refuse the bundle, each before a file is written.
    parts = bundle.parts(values)
    files = {} if bundle.code is None else sources.code_modules(bundle.code)
    assets = Assets(tokenizer=bundle.processor if bundle.tokenizer is None else bundle.tokenizer,
                    generation_config=dict(bundle.generation_config), named=bundle.names_tokenizer)
    envelope = (Release() if bundle.settings is None
                else release_record(bundle.settings, bundle.numerics, bundle.model, values, bundle.kernels))
    return ReleasePlan(Placed(parts, files, sidecars=decoders.sidecars(bundle.config)), assets=assets,
                       envelope=envelope)


def native_plan(variables: Variables, record: Mapping[str, object], step: int, *, why: str = "",
                averaged: bool = False, records: Sequence[tuple[epath.Path, PurePath]] = ()) -> ReleasePlan:
    """The native release of a run's step (0 for a model released outside a run): its variables, the files
    its task keeps by the step, and in its record the run's declaration `record`, whether `variables` are
    its average and why no family's layout writes its model where one was asked (`why`, the layout's
    reason): `ReleasedRun.of` reads `run`, `step`, `averaged` and `unstated`."""
    from dew._records import json_value

    task: dict[str, object] = {"run": json_value(record, "run"), "step": step, "averaged": averaged}
    if why:
        task["unstated"] = why
    return ReleasePlan(Native(variables, records), envelope=Release(task=task))


__all__ = ["Assets", "HeadFiles", "Native", "Placed", "ReleasePlan", "bundle_plan", "native_plan", "writable"]
