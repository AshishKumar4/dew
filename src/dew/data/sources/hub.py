"""Prepared HF Arrow tables and physical Parquet shards from HfFileSystem."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np

from dew.cache import HubAnswer, hub_answer, hub_commit, recorded_hub_answer
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


def split_files(paths: Sequence[str], split: str, config: Mapping[str, object] | None = None,
                roots: Sequence[str] = ()) -> list[str]:
    """The `paths` of a repository listing that `datasets` gives `split`: of a configuration its card
    states (`config`), by its `data_files` patterns, else by `datasets`' default patterns
    (`datasets.data_files.get_data_patterns`), under its `data_dir` where it names one, as `datasets`
    reads a card's configuration (`datasets.load.create_builder_configs_from_metadata_configs`); resolved
    as it resolves a split's patterns (`DataFilesList.from_patterns`), over an empty in-memory copy of
    the listing: the listing itself, which the Hub answered once and offline replays, is all the
    patterns read. A pattern that names the listing by a URL of its root (`roots`) reads it there; one
    that names a file elsewhere leaves the split none of the listing's."""
    from datasets.data_files import DataFilesList, get_data_patterns, sanitize_patterns
    from datasets.utils.file_utils import is_relative_path
    from fsspec.implementations.memory import MemoryFileSystem

    if not paths:
        # No file, so none of any split's, and no copy of the listing to make or remove.
        return []
    # `from_patterns` resolves the files through tqdm's thread_map.
    hold_progress_lock()
    fs, root = MemoryFileSystem(), f"/dew-hub-listing/{uuid.uuid4().hex}"
    for path in paths:
        fs.pipe_file(f"{root}/{path}", b"")
    stated = {} if config is None else config
    directory = stated.get("data_dir")
    base = f"memory://{root}" + (f"/{directory}" if isinstance(directory, str) and directory else "")
    try:
        # A listing with no data files at all is datasets' FileNotFoundError (EmptyDatasetError).
        files = stated.get("data_files")
        found = sanitize_patterns(files) if isinstance(files, (str, list, dict)) else get_data_patterns(base)
        patterns = next((list(each) for name, each in found.items() if str(name) == split), [])
        listing = f"memory://{root}/"
        located = [next((listing + pattern.removeprefix(url) for url in roots if pattern.startswith(url)),
                        pattern) for pattern in patterns]
        # A pattern of files elsewhere (a URL or an absolute path, as `datasets` tells them apart:
        # `is_relative_path`) names none of the listing's, which it would read from where it names.
        if any(not (pattern.startswith(listing) or is_relative_path(pattern)) for pattern in located):
            return []
        resolved = set(DataFilesList.from_patterns(located, base_path=base))
    except FileNotFoundError:
        resolved = set()
    finally:
        fs.rm(root, recursive=True)
    return sorted(url.split(root + "/", 1)[1] for url in resolved)


def conversion_layout(paths: Sequence[str]) -> bool:
    """Whether a commit's listing is laid out as the Hub's Parquet conversion lays its files out
    (refs/convert/parquet): no card, and each file but its `.gitattributes` a Parquet shard in a
    directory of its split in a directory of its configuration (`<config>/<split>/<shard>.parquet`,
    `partial-<split>` for a split converted in part: https://huggingface.co/docs/dataset-viewer/parquet).
    A repository's own files may be laid out so too; only the conversion's ref says which commit is it."""
    files = [path for path in paths if path != ".gitattributes"]
    return bool(files) and all(path.count("/") == 2 and path.endswith(".parquet") for path in files)


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
        """The commit `revision` names, the default branch's where None (`dew.cache.hub_commit`):
        online the Hub's answer, recorded; offline (`HF_HUB_OFFLINE`) the cache's ref, else the
        answer recorded online."""
        return self._pinned.value

    @cached_property
    def _pinned(self) -> HubAnswer[str]:
        return hub_commit(self.repo, self.revision, repo_type="dataset")

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
        revision = self.commit
        # A commit laid out as the conversion is one where the conversion's ref names it.
        paths, format = self._selected(revision, converted=conversion_layout(self._listed(revision))
                                       and self._conversion(revision) == revision)
        if not paths and self.revision is None:
            # The conversion of the default branch's commit, which the commit's answer pins.
            conversion = self._conversion(revision)
            if conversion is not None and conversion != revision:
                revision = conversion
                paths, format = self._selected(revision, converted=True)
        return (revision, tuple(sorted(paths)), format) if paths else None

    def _conversion(self, revision: str) -> str | None:
        """The commit the repo's Parquet conversion stands at (refs/convert/parquet), as the Hub
        answered where the repo stood at `revision`, None where it has none."""
        from huggingface_hub import HfApi

        def converted() -> str | None:
            converts = HfApi().list_repo_refs(self.repo, repo_type="dataset").converts
            return next((ref.target_commit for ref in converts if ref.ref == "refs/convert/parquet"), None)

        return hub_answer(("dataset conversion", self.repo, revision), converted).value

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

    def _configurations(self, revision: str) -> Mapping[str, Mapping[str, object]]:
        """The configurations the repo's card states at `revision` (its `configs`, as `datasets` reads
        them: `datasets.utils.metadata.MetadataConfigs`), each by its name with the fields that choose
        its files and its default (`data_files`, `data_dir`, `default`), as the Hub last answered
        (`dew.cache.hub_answer`); none where the commit holds no card or its card states none."""
        if "README.md" not in self._listed(revision):
            return {}

        def stated() -> Mapping[str, Mapping[str, object]]:
            from datasets.utils.metadata import MetadataConfigs
            from huggingface_hub import DatasetCard, DatasetCardData, HfFileSystem

            # Read through the Hub's file system, which raises OfflineModeIsEnabled offline, as every
            # Hub call does, so an answer recorded online replays (`hub_answer`).
            card = DatasetCard(HfFileSystem().read_text(f"datasets/{self.repo}@{revision}/README.md"))
            configs = MetadataConfigs.from_dataset_card_data(DatasetCardData(**card.data.to_dict()))
            return {name: {field: params[field] for field in ("data_files", "data_dir", "default")
                           if field in params} for name, params in configs.items()}

        return hub_answer(("dataset configurations", self.repo, revision), stated).value

    def _configuration(self, configs: Mapping[str, Mapping[str, object]]) -> str:
        """The configuration among `configs` this split reads: `name`, else their default as `datasets`
        chooses it (`MetadataConfigs.get_default_config_name`: a lone one, one named "default" or the one
        marked so). Raises ValueError where `name` is none of them, or none is named and none is the
        default, as `datasets` refuses both."""
        from datasets.utils.metadata import MetadataConfigs

        chosen = self.name if self.name is not None else MetadataConfigs(
            {name: dict(params) for name, params in configs.items()}).get_default_config_name()
        if chosen is None:
            raise ValueError(f"HF {self.repo!r} has the configurations {sorted(configs)} and marks none the "
                             "default: name one (Hub(..., name=))")
        if chosen not in configs:
            raise ValueError(f"HF {self.repo!r} has no configuration {chosen!r}; it has {sorted(configs)}")
        return chosen

    def _selected(self, revision: str, converted: bool = False) -> tuple[list[str], Format]:
        """This split's native shards at `revision`, in the first format of Parquet and JSONL that has
        any: the files `datasets` resolves to the split of the configuration this split reads
        (`_configuration`) from the revision's listing (`split_files`: a configuration's own patterns
        where the card states them, else its sharded names, its split keywords in file and directory
        names, `dev` for `validation` among them, a lone file as `train`). A commit whose card states no
        configuration holds the default one alone, and none of another name, which its Parquet
        conversion may hold. The conversion (`converted`) holds each configuration's splits in a
        directory of its name (`conversion_layout`), its configurations those of the card of the commit
        it converts, else its directories."""
        listed = self._listed(revision)
        if converted:
            configs = self._configurations(self.commit) or {
                path.split("/", 1)[0]: {} for path in listed if "/" in path}
            directory = self._configuration(configs) + "/"
            paths = [directory + path for path in split_files(
                [path.removeprefix(directory) for path in listed if path.startswith(directory)], self.split)]
        elif configs := self._configurations(revision):
            roots = (f"hf://datasets/{self.repo}@{revision}/", f"hf://datasets/{self.repo}/")
            paths = split_files(listed, self.split, configs[self._configuration(configs)], roots)
        else:
            paths = split_files(listed, self.split) if self.name in (None, "default") else []
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
