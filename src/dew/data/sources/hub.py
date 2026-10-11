"""Prepared HF Arrow tables and physical Parquet shards from HfFileSystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import grain.python as grain
import numpy as np

from dew.data.dataset import Records

from .base import Read, ShardManifest
from .files import Files
from .memory import _record


@dataclass(frozen=True)
class Hub:
    """A pinned Hub split. Preparing Arrow is always an explicit operation."""

    repo: str
    split: str = "train"
    config: str | None = None
    revision: str | None = None
    cache_dir: str | None = None

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

        return load_dataset_builder(
            self.repo, name=self.config, revision=self.commit, cache_dir=self.cache_dir
        )

    @cached_property
    def parquet(self) -> Files:
        from huggingface_hub import HfApi, HfFileSystem

        fs = HfFileSystem()
        revision = self.commit

        def selected(paths: list[str]) -> list[str]:
            return [
                path
                for path in paths
                if path.endswith(".parquet")
                and (f"/{self.split}/" in "/" + path or Path(path).name.startswith(self.split + "-"))
                and (self.config is None or f"/{self.config}/" in "/" + path)
            ]

        paths = selected(HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision))
        if not paths:
            revision = HfApi().dataset_info(self.repo, revision="refs/convert/parquet").sha
            paths = selected(HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision))
        if not paths:
            raise FileNotFoundError(
                f"HF {self.repo!r} split {self.split!r} has no physical Parquet shards; "
                "name an hf:// Parquet URL or explicitly provision Arrow"
            )
        urls = tuple(f"hf://datasets/{self.repo}@{revision}/{path}" for path in sorted(paths))
        for url in urls:
            fs.info(url)
        return Files(urls, format="parquet")

    def describe(self) -> str:
        return f"hf:{self.repo}@{self.commit}:{self.config}:{self.split}"

    def snapshot(self) -> Mapping[str, object]:
        identity = {
            "store": "hf",
            "repo": self.repo,
            "revision": self.commit,
            "config": self.config,
            "split": self.split,
        }
        builder = self.__dict__.get("builder")
        if builder is not None and Path(builder.cache_dir, "dataset_info.json").is_file():
            from dew.data.storage import Store

            dataset = builder.as_dataset(split=self.split)
            identity["shards"] = [
                {
                    "uri": entry["filename"],
                    "version": Store(entry["filename"]).resolve()[0].ukey(entry["filename"]),
                }
                for entry in dataset.cache_files
            ]
        else:
            identity["shards"] = self.parquet.snapshot()
        return identity

    def supports(self, read: Read) -> bool:
        if read == "streamed":
            return True
        return Path(self.builder.cache_dir, "dataset_info.json").is_file()

    def provision(self) -> Hub:
        self.builder.download_and_prepare()
        return self

    def records(self, read: Read) -> Records:
        if not self.supports(read):
            raise FileNotFoundError(
                f"{self.describe()} has no prepared Arrow split at {self.builder.cache_dir}; "
                "run dew data provision or set provision=True"
            )
        dataset = self.builder.as_dataset(split=self.split, in_memory=read == "memory")
        from datasets import Dataset

        if not isinstance(dataset, Dataset):
            raise TypeError(f"{self.describe()} did not resolve to one split")
        return grain.MapDataset.source(dataset).map(_record)

    def manifest(self) -> ShardManifest:
        return self.parquet.manifest()

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
