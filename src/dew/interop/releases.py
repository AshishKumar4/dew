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
- the record states what none of these files state (`dew.config.release`).

Every saver writes a plan, and every plan is written by `ReleasePlan.write`: `dew export`
(`dew.interop.saved_sources.export_run`), `Pretrained.save`, `Decide.save_pretrained`,
`SavedModel.save_pretrained`; `ReleasePlan.tensors` is what a release's weights are by name, which
`Pretrained.export` returns and a weight push sends.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import numpy as np
from etils import epath

from dew._files import write_atomically
from dew.config.release import Release
from dew.interop.safetensors_io import (
    CONFIG_FILE,
    MAX_SHARD_SIZE,
    LazyTensors,
    flatten,
    joined_tensors,
    save_sharded,
    tensor_specs,
    write_file,
)

if TYPE_CHECKING:
    from dew.batch import Variables
    from dew.interop.hf_decoders import ExportTokenizer
    from dew.interop.layouts import Unwritable
    from dew.interop.pretrained import Pretrained

type Part = tuple[Mapping[str, object], Mapping[str, np.ndarray]]
"""One folder of a model's release: its config and its tensors by their names there."""


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
    """A model no family's layout states (`why`, the layout's reason): its run's inference declaration
    (`record`) at `step` beside its `variables`, named `<collection>/<path>`, and the files its task keeps
    by the step (`records`, each copied to its place), which `dew.inference.models.ReleasedRun` reads back
    as the run."""

    variables: Variables
    records: Sequence[tuple[epath.Path, PurePath]] = ()
    why: str = ""

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

    def write(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write the release into `directory`: the assets, the model part, the head's files and the record,
        the weights in shards of at most `max_shard_size`, each file replaced whole."""
        from dew.interop.hf_decoders import save_export_assets

        root = Path(directory)
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
        if self.head is not None and self.head.merged:
            tensors = joined_tensors({"the model": tensors, "its head": self.head.tensors})
        # The root's weights at the release's root; its config there or in the format's folder.
        save_sharded(tensors, root, max_shard_size)
        (root / model.folder).mkdir(parents=True, exist_ok=True)
        write_atomically(str(root / model.folder / CONFIG_FILE), json.dumps(config, indent=2))


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
    from dew.interop import sources
    from dew.interop.release import release_record

    values = bundle.variables if variables is None else variables
    # Either may refuse the bundle, each before a file is written.
    parts = bundle.parts(values)
    files = {} if bundle.code is None else sources.code_modules(bundle.code)
    assets = Assets(tokenizer=bundle.processor if bundle.tokenizer is None else bundle.tokenizer,
                    generation_config=dict(bundle.generation_config), named=bundle.names_tokenizer)
    envelope = (Release() if bundle.settings is None
                else release_record(bundle.settings, bundle.numerics, bundle.model, values))
    return ReleasePlan(Placed(parts, files), assets=assets, envelope=envelope)


def native_plan(variables: Variables, record: Mapping[str, object], step: int, *, why: str = "",
                averaged: bool = False, records: Sequence[tuple[epath.Path, PurePath]] = ()) -> ReleasePlan:
    """The native release of a run's step (0 for a model released outside a run): its variables, the files
    its task keeps by the step, and in its record the run's declaration `record`, whether `variables` are
    its average (`ReleasedRun.of` reads `run`, `step` and `averaged`)."""
    from dew._records import json_value

    envelope = Release(task={"run": json_value(record, "run"), "step": step, "averaged": averaged})
    return ReleasePlan(Native(variables, records, why), envelope=envelope)


__all__ = ["Assets", "HeadFiles", "Native", "Placed", "ReleasePlan", "bundle_plan", "native_plan", "writable"]
