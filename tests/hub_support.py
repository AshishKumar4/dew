"""A Hugging Face dataset repository as Hub reads it, built from local Parquet: the real
`datasets` builder over each split's file, with the Hub API answering for one commit."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace


def native_builder(splits: Mapping[str, object], directory: Path, cache: Path):
    """The real Parquet builder of `splits`, each a `datasets.Dataset` written to its own file
    under `directory`, caching under `cache`. It is not prepared: a test prepares it, or leaves
    it unprepared, as its case needs."""
    import datasets

    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, dataset in splits.items():
        files[split] = str(directory / f"{split}.parquet")
        assert isinstance(dataset, datasets.Dataset)
        dataset.to_parquet(files[split])
    return datasets.load_dataset_builder("parquet", data_files=files, cache_dir=str(cache))


def native_hub(monkeypatch, directory: Path, repo: str, splits: Mapping[str, object],
               commit: str = "frozen"):
    """`repo` on the Hub at `commit` as the builder of `splits` (`native_builder`), which Hub
    prepares as it would a downloaded one."""
    import datasets
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs

    built = native_builder(splits, directory, directory / "cache")

    def builder(path, *, revision, **options):
        assert path == repo and revision == commit
        return built

    monkeypatch.setattr(HfApi, "dataset_info", lambda self, name, revision=None, **options: SimpleNamespace(
        sha=commit))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **options: ["data.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs", lambda *args, **options: GitRefs(
        branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", builder)
    return built


def stated_shards(monkeypatch):
    """The Hub stating each shard a test names as `HfFileSystem.info` states a file outside LFS:
    a byte, under its own git blob id."""
    from huggingface_hub import HfFileSystem

    monkeypatch.setattr(HfFileSystem, "info", lambda self, path, **options: {
        "name": path, "type": "file", "size": 1, "blob_id": f"blob:{path}"})


def offline(monkeypatch):
    """The Hub offline from here on, as HF_HUB_OFFLINE makes it: `datasets` and every Hub call a
    test stood in for raise OfflineModeIsEnabled, as the real ones then do, and `datasets` builds
    with its own `load_dataset_builder` again."""
    import datasets
    from huggingface_hub import HfApi, HfFileSystem, constants
    from huggingface_hub.errors import OfflineModeIsEnabled

    def unreachable(*args, **options):
        raise OfflineModeIsEnabled("offline mode is enabled (HF_HUB_OFFLINE)")

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True)
    for name in ("dataset_info", "list_repo_files", "list_repo_refs"):
        monkeypatch.setattr(HfApi, name, unreachable)
    monkeypatch.setattr(HfFileSystem, "info", unreachable)
    monkeypatch.setattr(HfFileSystem, "_open", unreachable)
    monkeypatch.setattr(datasets, "load_dataset_builder", datasets.load.load_dataset_builder)


def hub_split(monkeypatch, directory: Path, repo: str, splits: Mapping[str, object], commit: str):
    """`repo` on the Hub at `commit` as `datasets` builds a Hub dataset: its packaged Parquet
    builder over each split's file in the repo, keyed by the repo and the commit, so it prepares
    where a download of the repo would (`<owner>___<name>/<configuration>/<version>/<commit>`
    under `directory / "cache"`). The Hub answers for it, serving each file from `directory`, as
    long as `monkeypatch` holds."""
    import datasets
    from datasets.data_files import DataFilesDict, DataFilesList
    from datasets.packaged_modules.parquet.parquet import Parquet
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs

    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        assert isinstance(rows, datasets.Dataset)
        rows.to_parquet(directory / f"{split}.parquet")
        files[split] = DataFilesList([f"hf://datasets/{repo}@{commit}/{split}.parquet"], origin_metadata=[None])
    built = Parquet(cache_dir=str(directory / "cache"), dataset_name=repo.split("/")[-1], repo_id=repo,
                    hash=commit, data_files=DataFilesDict(files))

    def builder(path, *, revision, **options):
        assert path == repo and revision == commit
        return built

    def download(self, repo_id, filename, *, revision, **options):
        assert repo_id == repo and revision == commit
        return str(directory / filename)

    monkeypatch.setattr(HfApi, "dataset_info", lambda self, name, revision=None, **options: SimpleNamespace(
        sha=commit))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **options: ["data.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs", lambda *args, **options: GitRefs(
        branches=[], converts=[], tags=[]))
    monkeypatch.setattr(HfApi, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "load_dataset_builder", builder)
    return built
