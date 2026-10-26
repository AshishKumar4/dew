"""Where a source's files come from: a local directory, or a Hub commit.

`snapshot` resolves a repo at one commit and downloads only what a load
reads: the metadata (`METADATA_PATTERNS`), then the exact weight files the
listing and its indexes name (`safetensors_io.weight_files`), or the
components a pipeline asks for. `load_shards` reads a directory's weights in
their stored dtype. A commit with only transformers' PyTorch pickles loads
SFconvertbot's safetensors conversion of it where one is open, or converts
the pickles once (`dew.interop.pickles`); a source with neither is refused,
naming what it ships instead (`missing_weights`).
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from dew import records
from dew.cache import hub_answer, recorded_hub_answer
from dew.interop import pickles
from dew.interop.safetensors_io import (
    LazyTensors,
    header_specs,
    own,
    read_file,
    read_weights,
    stored_spec,
    weight_files,
)


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    """Read a checkpoint directory's weights, mapped in their stored dtype:
    the shards its index names, or its one model.safetensors. A directory
    with transformers' PyTorch pickles instead reads their safetensors
    conversion (`pickles.converted`)."""
    files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}

    def read(name: str) -> records.JSON:
        return json.loads((directory / name).read_text())

    if weight_files(files, "", read):
        return read_weights(directory)
    pickled = weight_files(files, "", read, stems=pickles.STEMS, suffix=pickles.SUFFIX)
    if not pickled:
        raise FileNotFoundError(missing_weights(str(directory), files))
    return read_weights(pickles.converted(directory, pickled))


_PICKLES = (".bin", ".pt", ".pth")

_log = logging.getLogger(__name__)


def missing_weights(source: str, files: Collection[str]) -> str:
    """Say what a source without safetensors weights or transformers'
    PyTorch pickles ships instead, and what loads."""
    gguf = sorted(name for name in files if name.endswith(".gguf"))
    pickled = sorted(name for name in files if "/" not in name and name.endswith(_PICKLES))
    if pickled:
        return (f"{source} ships PyTorch pickles ({', '.join(pickled[:3])}) but no pytorch_model.bin or "
                "pytorch_model.bin.index.json, the names Dew converts; save the state dict under one of "
                f"those, or convert the repo to safetensors at {pickles.CONVERT_SPACE}")
    if gguf:
        return (f"{source} ships GGUF files ({', '.join(gguf)}); load one with "
                f"Pretrained.load(..., gguf_file={gguf[0]!r})")
    return f"{source} has no model.safetensors or model.safetensors.index.json"


_CONVERSION_TITLE = "Adding `safetensors` variant of this model"


def _conversion_revision(name: str, commit: str) -> str | None:
    """Return SFconvertbot's open safetensors pull request on `commit`, or None.

    transformers' rule (safetensors_conversion.py, `previous_pr` and
    `get_conversion_pr_reference`): an open pull request by SFconvertbot
    under this title whose parent is the commit being loaded. Only looked
    up; nothing is converted or opened.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for discussion in api.get_repo_discussions(name, author="SFconvertbot",
                                               discussion_type="pull_request",
                                               discussion_status="open"):
        if discussion.title != _CONVERSION_TITLE or discussion.git_reference is None:
            continue
        commits = api.list_repo_commits(name, revision=discussion.git_reference)
        if len(commits) > 1 and commits[1].commit_id == commit:
            return discussion.git_reference
    return None


METADATA_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
"""Configs, indexes, tokenizer and chat-template files: everything a load reads but weights."""


def remote_code(name_or_dir: str | Path, directory: Path) -> Path:
    """The directory holding a source's own Python modules, its remote code: a local directory as it
    is, or the snapshot with the repo's `*.py` files fetched at its commit."""
    if not os.path.isdir(name_or_dir):
        from huggingface_hub import snapshot_download

        snapshot_download(str(name_or_dir), revision=directory.name, allow_patterns=["*.py"])
    return directory


def repo_file(name_or_dir: str | Path, directory: Path, filename: str) -> Path:
    """The local path of `filename` in a directory, or `filename` downloaded
    from the repo at the snapshot's commit.

    `directory` is the snapshot the metadata fetch resolved, named by its
    commit, so the file comes from that commit even if the branch moves. A
    missing file is refused, naming the files of its kind that are there.
    """
    suffix = Path(filename).suffix
    if os.path.isdir(name_or_dir):
        path = directory / filename
        if not path.is_file():
            present = sorted(
                entry.relative_to(directory).as_posix() for entry in directory.rglob(f"*{suffix}")
            )
            raise FileNotFoundError(f"{path} does not exist; the {suffix} files in {directory} are {present}")
        return path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(str(name_or_dir), filename, revision=directory.name))
    except EntryNotFoundError as error:
        present = sorted(name for name in repo_files(str(name_or_dir), directory) if name.endswith(suffix))
        raise FileNotFoundError(f"{name_or_dir} at {directory.name} has no {filename!r}; "
                                f"its {suffix} files are {present}") from error


def repo_files(name: str, directory: Path) -> set[str]:
    """Every file of the snapshot's commit, by repo-relative name.

    A dry run lists the Hub tree the metadata fetch has just cached. Offline
    it cannot: without a cached tree it raises DryRunError, and with one it
    raises LocalEntryNotFoundError for the first listed file that was never
    downloaded. The cache is then all a load can read anyway, so the
    snapshot directory's own files are the listing.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import DryRunError, LocalEntryNotFoundError

    try:
        return {entry.filename for entry in snapshot_download(name, revision=directory.name, dry_run=True)}
    except (DryRunError, LocalEntryNotFoundError):
        return {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}


def has_file(repo_id: str, filename: str, revision: str | None) -> bool:
    """Whether the Hub repo `repo_id` holds `filename` at `revision`.

    It asks the Hub; offline (`HF_HUB_OFFLINE`) it answers from the local
    cache, where a file never fetched counts as absent.
    """
    from huggingface_hub import file_exists, try_to_load_from_cache
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        return file_exists(repo_id, filename, revision=revision)
    except OfflineModeIsEnabled:
        return isinstance(try_to_load_from_cache(repo_id, filename, revision=revision), str)


def snapshot(name_or_dir: str, revision: str | None, *,
              weights: bool | tuple[str, ...] = True) -> Path:
    """Resolve a snapshot with the root's (True), no (False) or the named
    components' weights.

    A local directory is returned as it is. From the Hub the metadata comes
    first, then `weight_files` picks, from the commit's listing and the
    indexes just fetched, the exact files that hold the weights, and only
    those download, at the commit the first fetch resolved. Other formats of
    the same weights beside them (Mistral's consolidated.safetensors,
    diffusers' fp16 variants and root single-file checkpoints) stay on the
    Hub. A commit with only transformers' PyTorch pickles resolves to
    SFconvertbot's safetensors pull request on it where one is open, and
    otherwise downloads the pickles, which `load_shards` converts.
    """
    if os.path.isdir(name_or_dir):
        return Path(name_or_dir)
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(name_or_dir, revision=revision,
                                       allow_patterns=METADATA_PATTERNS))
    if weights is False:
        return directory
    files = repo_files(name_or_dir, directory)

    def read(name: str) -> records.JSON:
        return json.loads((directory / name).read_text())

    selected = [name for folder in (("",) if weights is True else weights)
                for name in weight_files(files, folder, read)]
    if weights is True and not selected:
        from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

        selected = list(weight_files(files, "", read, stems=pickles.STEMS, suffix=pickles.SUFFIX))
        if not selected:
            raise FileNotFoundError(missing_weights(f"{name_or_dir} at {directory.name}", files))
        # The conversion a lookup found is recorded (`hub_answer`), so the same
        # load offline reads the conversion it cached rather than pickles it
        # never fetched; a lookup the Hub refuses, or offline one never made,
        # finds none.
        try:
            conversion = hub_answer(("conversion", name_or_dir, directory.name),
                                    lambda: _conversion_revision(name_or_dir, directory.name)).value
        except (HfHubHTTPError, OfflineModeIsEnabled):
            conversion = None
        if conversion is not None:
            _log.warning("%s at %s ships PyTorch pickles; loading SFconvertbot's safetensors conversion of "
                         "that commit at revision %s", name_or_dir, directory.name, conversion)
            return snapshot(name_or_dir, conversion)
    if selected:
        snapshot_download(name_or_dir, revision=directory.name, allow_patterns=selected)
    return directory


def _hub_files(name: str, revision: str) -> list[str]:
    """Every file of a Hub repo's commit, by repo-relative name, listed over HTTP."""
    from huggingface_hub import HfApi

    return sorted(HfApi().list_repo_files(name, revision=revision))


def _hub_header(name: str, revision: str, file: str) -> dict[str, tuple[list[int], str]]:
    """Each tensor of a Hub file's safetensors header, its shape and dtype tag, read over HTTP."""
    from huggingface_hub import HfApi

    header = HfApi().parse_safetensors_file_metadata(name, file, revision=revision)
    return {tensor_name: (list(tensor.shape), tensor.dtype) for tensor_name, tensor in header.tensors.items()}


@dataclass(frozen=True)
class StoredWeights:
    """A source's weight files, read by their headers and fetched by name: what a
    pipeline rebound from saved variables writes its export by, downloading no
    weight it does not write. `directory` is the snapshot the metadata fetch
    resolved (`snapshot(weights=False)`), named by its commit."""

    name_or_dir: str
    directory: Path

    def files(self, folder: str) -> tuple[str, ...]:
        """The repo-relative files that hold `folder`'s weights (`weight_files`): a directory's
        own, or the commit's, as the Hub lists them (`_hub_files`), recorded so an offline run
        reads the same listing (`hub_answer`); offline where it was never listed online, those
        the cached snapshot holds (`repo_files`)."""
        from huggingface_hub.errors import OfflineModeIsEnabled

        if os.path.isdir(self.name_or_dir):
            listing = {path.relative_to(self.directory).as_posix() for path in self.directory.rglob("*")
                       if path.is_file()}
        else:
            try:
                listing = set(hub_answer(("repo files", self.name_or_dir, self.directory.name),
                                         functools.partial(_hub_files, self.name_or_dir,
                                                           self.directory.name)).value)
            except OfflineModeIsEnabled:
                listing = repo_files(self.name_or_dir, self.directory)
        return weight_files(listing, folder, lambda name: json.loads((self.directory / name).read_text()))

    def specs(self, folder: str) -> dict[str, tuple[str, jax.ShapeDtypeStruct]] | None:
        """Each of `folder`'s tensors, by its stored name: the file that holds it and
        the array it reads as, from the file's header. A file the snapshot holds is
        read there; a Hub source's other headers are read over HTTP
        (`HfApi.parse_safetensors_file_metadata`), its weights left on the Hub, and
        recorded so an offline run reads the same answer (`hub_answer`). None where a
        header cannot be read: a directory's file that is not there, or offline a Hub
        header never read online. Binding variables needs no header (a run rebuilds its
        objective over a pipeline's metadata alone), so only the export, which writes by
        them, refuses (`dew.interop.diffusion.save_source`)."""
        from huggingface_hub import constants

        held: dict[str, tuple[str, jax.ShapeDtypeStruct]] = {}
        owner: dict[str, str] = {}
        for file in self.files(folder):
            question = ("safetensors header", self.name_or_dir, self.directory.name, file)
            if (self.directory / file).is_file():
                specs = header_specs(self.directory / file)
            elif os.path.isdir(self.name_or_dir) or (constants.HF_HUB_OFFLINE
                                                     and recorded_hub_answer(question) is None):
                return None
            else:
                header = hub_answer(question, functools.partial(_hub_header, self.name_or_dir,
                                                                self.directory.name, file)).value
                specs = {name: stored_spec(name, shape, tag, file) for name, (shape, tag) in header.items()}
            own(owner, specs, file, self.directory / folder)
            held.update({name: (file, spec) for name, spec in specs.items()})
        return held

    def tensors(self, held: dict[str, tuple[str, jax.ShapeDtypeStruct]]) -> LazyTensors:
        """The tensors `held` names (`specs`), each read from its file when it is read,
        a Hub source's file downloaded then."""

        @functools.cache
        def read(file: str) -> dict[str, np.ndarray]:
            return read_file(repo_file(self.name_or_dir, self.directory, file))[0]

        return LazyTensors({name: spec for name, (_, spec) in held.items()},
                           lambda name: read(held[name][0])[name])
