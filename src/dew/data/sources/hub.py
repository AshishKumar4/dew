"""Prepared HF Arrow tables and physical Parquet shards from HfFileSystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from dew.data.dataset import Records

from .base import Read, ShardManifest
from .files import Files


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

        return HfApi().dataset_info(self.repo, revision=self.revision).sha

    @cached_property
    def builder(self):
        from datasets import load_dataset_builder

        return load_dataset_builder(self.repo, name=self.config, revision=self.commit, cache_dir=self.cache_dir)

    @cached_property
    def parquet(self) -> Files:
        from huggingface_hub import HfApi, HfFileSystem

        fs = HfFileSystem()
        revision = self.commit

        def selected(paths: list[str]) -> list[str]:
            return [path for path in paths if path.endswith(".parquet")
                    and (f"/{self.split}/" in "/" + path or Path(path).name.startswith(self.split + "-"))
                    and (self.config is None or f"/{self.config}/" in "/" + path)]

        paths = selected(HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision))
        if not paths:
            revision = HfApi().dataset_info(self.repo, revision="refs/convert/parquet").sha
            paths = selected(HfApi().list_repo_files(self.repo, repo_type="dataset", revision=revision))
        if not paths:
            raise FileNotFoundError(f"HF {self.repo!r} split {self.split!r} has no physical Parquet shards; "
                                    "name an hf:// Parquet URL or explicitly provision Arrow")
        urls = tuple(f"hf://datasets/{self.repo}@{revision}/{path}" for path in sorted(paths))
        for url in urls:
            fs.info(url)
        return Files(urls, format="parquet")

    def describe(self) -> str:
        return f"hf:{self.repo}@{self.commit}:{self.config}:{self.split}"

    def snapshot(self) -> Mapping[str, object]:
        return {"store": "hf", "repo": self.repo, "revision": self.commit,
                "config": self.config, "split": self.split}

    def supports(self, read: Read) -> bool:
        if read == "streamed":
            return True
        return Path(self.builder.cache_dir, "dataset_info.json").is_file()

    def provision(self) -> Hub:
        self.builder.download_and_prepare()
        return self

    def records(self, read: Read) -> Records:
        if not self.supports(read):
            raise FileNotFoundError(f"{self.describe()} has no prepared Arrow split at {self.builder.cache_dir}; "
                                    "run dew data provision or set provision=True")
        dataset = self.builder.as_dataset(split=self.split, in_memory=read == "memory")
        from datasets import Dataset

        if not isinstance(dataset, Dataset):
            raise TypeError(f"{self.describe()} did not resolve to one split")
        return dataset

    def manifest(self) -> ShardManifest:
        return self.parquet.manifest()

    def metadata(self) -> Mapping[str, object]:
        return {}

    def lengths(self) -> np.ndarray | None:
        return None
