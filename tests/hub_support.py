"""A Hugging Face dataset repository as Hub reads it, built from local Parquet: the real
`datasets` builder over each split's file, with the Hub API answering for one commit."""

from pathlib import Path
from types import SimpleNamespace


def native_builder(directory: Path, splits: dict[str, dict[str, list]]):
    """The real Parquet builder of `splits`, each a column mapping written to its own file,
    caching under `directory`."""
    import datasets

    files = {}
    for split, columns in splits.items():
        files[split] = str(directory / f"{split}.parquet")
        datasets.Dataset.from_dict(columns).to_parquet(files[split])
    return datasets.load_dataset_builder("parquet", data_files=files, cache_dir=str(directory / "cache"))


def native_hub(monkeypatch, directory: Path, repo: str, splits: dict[str, dict[str, list]],
               commit: str = "frozen"):
    """`repo` on the Hub at `commit` as the builder of `splits` (`native_builder`), which Hub
    prepares as it would a downloaded one."""
    import datasets
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs

    built = native_builder(directory, splits)

    def builder(name, *, revision, **options):
        assert name == repo and revision == commit
        return built

    monkeypatch.setattr(HfApi, "dataset_info", lambda self, name, revision=None, **options: SimpleNamespace(
        sha=commit))
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **options: ["data.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs", lambda *args, **options: GitRefs(
        branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", builder)
    return built
