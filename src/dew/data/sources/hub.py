"""Prepared HF Arrow tables and physical Parquet shards from HfFileSystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import grain.python as grain
import numpy as np

from dew.data.dataset import Records
from dew.data.pipeline import Dataset

from .base import Read, ShardManifest, Source, preparing
from .files import Files, Format
from .memory import _record


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
        from huggingface_hub import HfApi

        commit = HfApi().dataset_info(self.repo, revision=self.revision).sha
        if commit is None:
            raise ValueError(f"HF {self.repo!r} did not resolve to a commit")
        return commit

    @cached_property
    def builder(self):
        try:
            from datasets import load_dataset_builder
        except ImportError as error:
            raise ImportError("HF Arrow reads need pip install 'dewml[streaming]'") from error

        pinned = {"name": self.name, "revision": self.commit, "cache_dir": self.cache_dir}
        whole = load_dataset_builder(self.repo, **pinned)
        # Only this split's own files, as the dataset's own configuration resolves
        # them: preparing the whole dataset would download every split to read one.
        splits = whole.config.data_files or {}
        if len(splits) < 2 or self.split not in splits:
            return whole
        return load_dataset_builder(self.repo, **pinned, data_files={self.split: list(splits[self.split])})

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
            converts = HfApi().list_repo_refs(self.repo, repo_type="dataset").converts
            conversion = next(
                (ref.target_commit for ref in converts if ref.ref == "refs/convert/parquet"), None
            )
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
                    "has no native Parquet, Arrow or JSONL files; "
                    "pin the converted branch's own commit explicitly, "
                    "or provision=True for Arrow at this revision"
                )
            raise FileNotFoundError(
                f"HF {self.repo!r} split {self.split!r} has no physical readable shards; "
                "name an hf:// Parquet URL or provision=True and read it indexed"
            )
        revision, paths, format = selection
        urls = tuple(f"hf://datasets/{self.repo}@{revision}/{path}" for path in paths)
        return Files(urls, format=format, columns=self.columns)

    def _selected(self, revision: str) -> tuple[list[str], Format]:
        from huggingface_hub import HfApi

        paths = HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision)
        formats: tuple[Format, ...] = ("parquet", "arrow", "jsonl")
        for format in formats:
            matches = [
                path
                for path in paths
                if path.endswith("." + format)
                and (
                    f"/{self.split}/" in "/" + path
                    or Path(path).name.startswith(self.split + "-")
                    or Path(path).stem == self.split
                )
                and (self.name is None or f"/{self.name}/" in "/" + path)
            ]
            if matches:
                return matches, format
        return [], "parquet"

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
        return Path(self.builder.cache_dir, "dataset_info.json").is_file()

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
        from datasets import Dataset, Image

        if not isinstance(dataset, Dataset):
            raise TypeError(f"{self.describe()} did not resolve to one split")
        if self.columns is not None:
            dataset = dataset.select_columns(list(self.columns))
        for name, feature in dataset.features.items():
            if isinstance(feature, Image) and feature.decode:
                dataset = dataset.cast_column(name, Image(decode=False))
        return grain.MapDataset.source(dataset).map(_record)

    def manifest(self) -> ShardManifest:
        return self.files.manifest()

    def metadata(self) -> Mapping[str, object]:
        from datasets import ClassLabel

        features = self.builder.info.features
        class_names = {
            name: tuple(feature.names)
            for name, feature in (features or {}).items()
            if isinstance(feature, ClassLabel)
        }
        splits = self.builder.info.splits
        rows = (
            {"rows": splits[self.split].num_examples} if splits is not None and self.split in splits else {}
        )
        return {"class_names": class_names, **rows}

    def lengths(self) -> np.ndarray | None:
        return None
