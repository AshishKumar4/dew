"""Prepared HF Arrow tables and physical Parquet shards from HfFileSystem."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np

from dew.cache import HubAnswer, hub_answer, recorded_hub_answer
from dew.data.dataset import Records
from dew.data.pipeline import Dataset

from .base import Read, ShardManifest, Source, preparing
from .files import Files, Format
from .hf import arrow_rows, class_names


def hold_progress_lock() -> None:
    """Give tqdm's own progress-bar class a lock, before any call into `datasets`. `datasets` maps a split's
    files through tqdm's thread_map (resolving a listing's files to a split, reading or preparing its
    shards), which lends the bar class it is given a lock for the map and deletes it after where neither
    the class nor a class it derives from held one. Two maps at once (compile's readers: a holdout's train
    and held; several streamed sources discovered together) then delete it under each other's bars:
    "type object 'tqdm' has no attribute '_lock'". Held by tqdm's class, which every bar class `datasets`
    lends a map derives from, each map finds it and restores it instead."""
    from tqdm import tqdm as bars

    bars.get_lock()


def configuration(card: str | None, name: str | None) -> tuple[str, str | list | dict | None, str | None]:
    """The configuration `datasets` reads by `name` from a dataset card (its README's text, None for a
    repository without one): the card's `configs` metadata (`MetadataConfigs`), `name` or, where None,
    the card's default, else the one configuration its `dataset_info` describes, with its `data_files`
    and its `data_dir`. A card of no `configs` reads the repository by `datasets`' default patterns as
    its one configuration, `default`, and a configuration named there lies in a directory of its name,
    as the Hub's Parquet conversion lays each one out. A name the card has no configuration of, or
    None where it has several and no default, is refused, as `load_dataset_builder` refuses it."""
    from datasets.info import DatasetInfosDict
    from datasets.utils.metadata import MetadataConfigs
    from huggingface_hub import DatasetCard, DatasetCardData

    metadata = DatasetCard(card).data if card else DatasetCardData()
    assert isinstance(metadata, DatasetCardData), type(metadata)
    configs = MetadataConfigs.from_dataset_card_data(metadata)
    if not configs:
        return ("default", None, None) if name in (None, "default") else (name, None, name)
    chosen = name if name is not None else configs.get_default_config_name()
    described = DatasetInfosDict.from_dataset_card_data(metadata)
    if chosen is None and len(described) == 1:
        chosen = next(iter(described))
    if chosen is None or chosen not in configs:
        raise ValueError(f"the dataset's configurations are {sorted(configs)}; name one of them, not "
                         f"{name!r}")
    stated = configs[chosen]
    return chosen, stated.get("data_files"), stated.get("data_dir")


def split_files(paths: Sequence[str], split: str, *, data_files: str | list | dict | None = None,
                data_dir: str | None = None) -> list[str]:
    """The `paths` of a repository listing that `datasets` gives `split` in a configuration of
    `data_files` patterns, else its default patterns (`datasets.data_files.get_data_patterns`), under
    `data_dir` where one is given (`configuration`), resolved as it resolves a split's patterns
    (`DataFilesList.from_patterns`), over an empty in-memory copy of the listing: the listing itself,
    which the Hub answered once and offline replays, is all the patterns read."""
    from datasets.data_files import DataFilesList, get_data_patterns, sanitize_patterns
    from fsspec.implementations.memory import MemoryFileSystem

    if not paths:
        # No file, so none of any split's, and no copy of the listing to make or remove.
        return []
    # `from_patterns` resolves the files through tqdm's thread_map.
    hold_progress_lock()
    fs, root = MemoryFileSystem(), f"/dew-hub-listing/{uuid.uuid4().hex}"
    for path in paths:
        fs.pipe_file(f"{root}/{path}", b"")
    # A configuration's patterns resolve under its directory, as a builder resolves them.
    base = f"memory://{root}" + (f"/{data_dir.strip('/')}" if data_dir else "")
    try:
        # A listing with no data files at all is datasets' FileNotFoundError (EmptyDatasetError).
        stated = sanitize_patterns(data_files) if data_files is not None else get_data_patterns(base)
        patterns = {str(name): [_listed_pattern(pattern) for pattern in found]
                    for name, found in stated.items()}
        resolved = set(DataFilesList.from_patterns(patterns.get(split, []), base_path=base))
    except FileNotFoundError:
        resolved = set()
    finally:
        fs.rm(root, recursive=True)
    return sorted(url.split(root + "/", 1)[1] for url in resolved)


def _listed_pattern(pattern: str) -> str:
    """A configuration's data-files pattern as the repository's listing names its files: a URL into the
    repository at a revision (`hf://datasets/owner/name@revision/...`) by its path in it; a URL into any
    other store is refused, since a streamed split reads its own repository's files."""
    if "://" not in pattern:
        return pattern
    found = re.fullmatch(r"hf://datasets/[^/@]+/[^/@]+(?:@[^/]+)?/(.*)", pattern)
    if found is None:
        raise ValueError(f"a configuration's data files {pattern!r} lie outside its repository, which a "
                         "streamed split does not read; prepare the split, or read the files with Files")
    return found[1]


@dataclass(frozen=True)
class Hub(Dataset):
    """A pinned split of a Hub dataset: `name` is its configuration, as
    `datasets.load_dataset(repo, name=)` takes it, and `columns` the fields
    each record keeps (all where None). Read whole or by index from the
    Arrow cache it prepares, or streamed from the repo's own shards."""

    repo: str
    name: str | None = None
    split: str = "train"
    columns: tuple[str, ...] | None = None
    revision: str | None = None
    cache_dir: str | None = None

    def with_split(self, name: str) -> Hub:
        return replace(self, split=name)

    @cached_property
    def commit(self) -> str:
        """The commit `revision` names, the default branch's where None, as the Hub last answered
        (`dew.cache.hub_answer`): online its answer, recorded; offline (`HF_HUB_OFFLINE`) the
        answer recorded online."""
        return self._pinned.value

    @cached_property
    def _pinned(self) -> HubAnswer[str]:
        from huggingface_hub import HfApi

        def ask() -> str:
            commit = HfApi().dataset_info(self.repo, revision=self.revision).sha
            if commit is None:
                raise ValueError(f"HF {self.repo!r} did not resolve to a commit")
            return commit

        # A git revision is never named HEAD but the default branch is.
        return hub_answer(("dataset commit", self.repo, self.revision or "HEAD"), ask)

    @property
    def builder(self):
        """`datasets`' builder of this split at `commit`. With the Hub offline, where `commit` is the
        answer recorded online, `datasets` cannot rebuild a Hub dataset's builder, so it is
        `datasets`' own builder of the cache it prepared the split in at that commit (`Cache`), at
        the configuration and version the online builder recorded, and a split never prepared
        raises."""
        if self._builder is None:
            raise FileNotFoundError(
                f"{self.describe()} has no Arrow prepared in {self.cache_dir or 'the datasets cache'}, and "
                "with the Hub offline (HF_HUB_OFFLINE) none can be: provision it online")
        return self._builder

    @cached_property
    def _builder(self):
        try:
            from datasets import load_dataset_builder
        except ImportError as error:
            raise ImportError("HF Arrow reads need pip install 'dewml[streaming]'") from error
        from datasets.naming import camelcase_to_snakecase
        from datasets.packaged_modules.cache.cache import Cache

        hold_progress_lock()
        question = ("dataset builder", self.repo, self.commit, self.name or "", self.split)
        if self._pinned.recorded:
            # Never built online, so never prepared.
            prepared = recorded_hub_answer(question)
            if prepared is None:
                return None
            # Named as `datasets` names a Hub dataset's builder, which names the Arrow files.
            built = Cache(cache_dir=self.cache_dir, repo_id=self.repo,
                          dataset_name=camelcase_to_snakecase(self.repo.split("/")[-1]),
                          config_name=prepared.value["configuration"], version=prepared.value["version"],
                          hash=self.commit)
            return built if Path(built.cache_dir, "dataset_info.json").is_file() else None
        pinned = {"name": self.name, "revision": self.commit, "cache_dir": self.cache_dir}
        whole = load_dataset_builder(self.repo, **pinned)
        # Only this split's own files, as the dataset's own configuration resolves
        # them: preparing the whole dataset would download every split to read one.
        splits = whole.config.data_files or {}
        built = whole if len(splits) < 2 or self.split not in splits else load_dataset_builder(
            self.repo, **pinned, data_files={self.split: list(splits[self.split])})
        # The configuration (its id keeps a split's own files' hash) and version it prepares under,
        # which offline only this record names (`builder`).
        configuration, version = built.config_id, str(built.config.version)
        hub_answer(question, lambda: {"configuration": configuration, "version": version})
        return built

    @property
    def streamed(self) -> str | None:
        """The pinned commit with readable shards of this split, or None.
        Native shards come from the resolved commit. An unpinned default
        revision without native shards can read its Parquet conversion."""
        return None if self._selection is None else self._selection[0]

    @cached_property
    def _selection(self) -> tuple[str, tuple[str, ...], Format] | None:
        from huggingface_hub import HfApi

        revision = self.commit
        paths, format = self._selected(revision)
        if not paths and self.revision is None:

            def converted() -> str | None:
                converts = HfApi().list_repo_refs(self.repo, repo_type="dataset").converts
                return next((ref.target_commit for ref in converts if ref.ref == "refs/convert/parquet"),
                            None)

            # The conversion of the default branch's commit, which the commit's answer pins.
            conversion = hub_answer(("dataset conversion", self.repo, revision), converted).value
            if conversion is not None and conversion != revision:
                revision = conversion
                paths, format = self._selected(revision)
        return (revision, tuple(sorted(paths)), format) if paths else None

    @cached_property
    def files(self) -> Files:
        selection = self._selection
        if selection is None:
            if self.revision is not None:
                raise FileNotFoundError(
                    f"HF {self.repo!r}@{self.commit} split {self.split!r} "
                    "has no native Parquet or JSONL files; "
                    "pin the converted branch's own commit explicitly, "
                    "or provision=True for Arrow at this revision"
                )
            raise FileNotFoundError(
                f"HF {self.repo!r} split {self.split!r} has no physical readable shards; "
                "name an hf:// Parquet URL or provision=True and read it indexed"
            )
        revision, paths, format = selection
        urls = tuple(f"hf://datasets/{self.repo}@{revision}/{path}" for path in paths)
        facts = self._facts(revision, paths, format)
        return Files(urls, format=format, columns=self.columns, file_info=facts)

    def _facts(self, revision: str, paths: tuple[str, ...],
               format: Format) -> Mapping[str, Mapping[str, object]]:
        """Each shard's size, and the sha256 of its content where the Hub states one (an LFS
        file's), else its git blob id, at `revision`, as the Hub last answered
        (`dew.cache.hub_answer`), and so of each JSON lines shard's offset index the commit holds.
        `Files` takes them as the files' own facts, so they keep their identity offline."""
        from huggingface_hub import HfFileSystem

        hub = HfFileSystem()

        def stated(url: str) -> Mapping[str, object]:
            facts = hub.info(url.removeprefix("hf://"))
            stored = facts.get("lfs")
            sha256 = stored.get("sha256") if isinstance(stored, Mapping) else None
            return {"bytes": facts["size"], **({"sha256": sha256} if sha256 else {"ETag": facts["blob_id"]})}

        root = f"hf://datasets/{self.repo}@{revision}/"
        urls = [root + path for path in paths]
        listed = set(self._listed(revision))
        indexes = [index for index in (Files._index_url(url) for url in urls)
                   if format == "jsonl" and index.removeprefix(root) in listed]
        # A commit's files never change, so each answer stands for its file.
        return {url: hub_answer(("dataset file", url), lambda url=url: stated(url)).value
                for url in (*urls, *indexes)}

    def _card(self, revision: str) -> str | None:
        """The dataset card's text at `revision` (its README.md), None where the repo has none, as the
        Hub last answered (`dew.cache.hub_answer`): a commit's card never changes."""
        from huggingface_hub import HfFileSystem

        if "README.md" not in self._listed(revision):
            return None
        return hub_answer(("dataset card", self.repo, revision), lambda: HfFileSystem().read_text(
            f"datasets/{self.repo}@{revision}/README.md")).value

    def _listed(self, revision: str) -> list[str]:
        """Every file of the repo at `revision`, as the Hub last answered (`dew.cache.hub_answer`),
        asked once per source: a commit's files never change, so its answer stands for it."""
        from huggingface_hub import HfApi

        if revision not in self._listings:
            self._listings[revision] = hub_answer(
                ("dataset files", self.repo, revision),
                lambda: HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision),
            ).value
        return self._listings[revision]

    @cached_property
    def _listings(self) -> dict[str, list[str]]:
        return {}

    def _selected(self, revision: str) -> tuple[list[str], Format]:
        """This split's native shards at `revision`, in the first format of Parquet and JSONL that has
        any: the files `datasets` resolves to the split from the revision's listing in the
        configuration it reads (`configuration`, from the card at the source's commit): its
        `data_files` and `data_dir`, else `datasets.data_files.get_data_patterns`' sharded names and
        split keywords in file and directory names, `dev` for `validation` among them, a lone file as
        `train`. The Hub's Parquet conversion holds each configuration in a directory of its name."""
        name, data_files, data_dir = configuration(self._card(self.commit), self.name)
        converted = revision != self.commit
        paths = split_files(self._listed(revision), self.split, data_files=None if converted else data_files,
                            data_dir=name if converted else data_dir)
        # Not Arrow: the `.arrow` files a dataset repo holds are what `save_to_disk` and
        # `push_to_hub` write, Arrow's stream format, which no reader can cut into shards or count
        # without reading it whole; such a split streams its Parquet conversion, or is prepared.
        formats: tuple[Format, ...] = ("parquet", "jsonl")
        for format in formats:
            matches = [path for path in paths if path.endswith("." + format)]
            if matches:
                return matches, format
        return [], "parquet"

    def label(self) -> str:
        return self.repo.rsplit("/", 1)[-1]

    def describe(self) -> str:
        selected = f":{list(self.columns)}" if self.columns else ""
        return f"hf:{self.repo}@{self.commit}:{self.name}:{self.split}{selected}"

    def consumed(self, read: Read) -> Source:
        return self.files if read == "streamed" else self

    def fingerprint(self) -> Mapping[str, object]:
        """The pinned split Arrow is prepared from. Its identity is the same
        on any host, in any cache directory, whether or not Arrow was prepared
        there. A streamed read consumes `files`, whose snapshot names its
        physical shards at their own pinned commit."""
        return {
            "store": "hf",
            "repo": self.repo,
            "revision": self.commit,
            "config": self.name,
            # Only a selection names its columns, so every other split keeps its key.
            **({} if self.columns is None else {"columns": list(self.columns)}),
            "split": self.split,
        }

    def supports(self, read: Read) -> bool:
        if read == "streamed":
            return self.streamed is not None
        return self._builder is not None and Path(self._builder.cache_dir, "dataset_info.json").is_file()

    def can_provision(self, read: Read) -> bool:
        return read != "streamed"

    def provision(self) -> Hub:
        preparing(self, self.builder.info.download_size, self.builder.cache_dir)
        self.builder.download_and_prepare()
        return self

    def records(self, read: Read) -> Records:
        """Each row as the split stores it: an image column's encoded bytes,
        which the image steps decode, so a memory read holds the split, not
        its decoded pixels."""
        if not self.supports(read):
            raise FileNotFoundError(
                f"{self.describe()} has no prepared Arrow split at {self.builder.cache_dir}; "
                "run dew data provision or set provision=True"
            )
        dataset = self.builder.as_dataset(split=self.split, in_memory=read == "memory")
        from datasets import Dataset

        if not isinstance(dataset, Dataset):
            raise TypeError(f"{self.describe()} did not resolve to one split")
        if self.columns is not None:
            dataset = dataset.select_columns(list(self.columns))
        return arrow_rows(dataset)

    def manifest(self) -> ShardManifest:
        return self.files.manifest()

    def stated(self) -> Mapping[str, object]:
        """Its builder's info, which its metadata is, no row read; a split never prepared, with the Hub
        offline, states nothing."""
        return {} if self._builder is None else self.metadata()

    def metadata(self) -> Mapping[str, object]:
        splits = self.builder.info.splits
        rows = (
            {"rows": splits[self.split].num_examples} if splits is not None and self.split in splits else {}
        )
        return {"class_names": class_names(self.builder.info.features), **rows}

    def lengths(self) -> np.ndarray | None:
        return None
