"""Fetch a published run or export directory from the Hugging Face Hub.

`pull_from_hub` hands back the snapshot path `huggingface_hub` downloads;
retries, progress and caching stay the hub client's behaviour. Publishing is
the hub client's own call: `HfApi().create_repo(repo_id, exist_ok=True)` and
`HfApi().upload_folder(repo_id=repo_id, folder_path=directory)` on the
directory `Pretrained.save` or `export_run` wrote, or on a run directory
itself, which is the form the tasks' `from_pretrained` pull back.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def pull_from_hub(repo_id: str, revision: str | None = None) -> Path:
    """Download a snapshot of `repo_id` and return the directory holding it.

    `revision` is a branch, tag or commit; None takes the default branch. The
    path is inside the hub cache, so a second call with the same revision
    downloads nothing.
    """
    return Path(snapshot_download(repo_id=repo_id, revision=revision))
