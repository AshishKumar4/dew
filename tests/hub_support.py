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
