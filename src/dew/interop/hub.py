"""Moving an export directory to and from the Hugging Face Hub.

`save_hf_layout` writes the directory; these two carry it. They wrap the two
`huggingface_hub` calls involved, creating the repo on the way up and handing
back the snapshot path on the way down. Retries, progress and caching stay the
hub client's behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download


def push_to_hub(directory, repo_id: str, *, private: bool = False,
                commit_message: str = "Upload dew export") -> None:
    """Upload `directory` to `repo_id`, creating the repo when it is missing.

    The files land at the root of the repo under the names they have on disk,
    so an export written by `save_hf_layout` arrives as the model.safetensors
    and config.json pair a Hugging Face loader looks for.
    """
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=os.fspath(directory),
        commit_message=commit_message,
    )


def pull_from_hub(repo_id: str, revision: Optional[str] = None) -> Path:
    """Download a snapshot of `repo_id` and return the directory holding it.

    `revision` is a branch, tag or commit; None takes the default branch. The
    path is inside the hub cache, so a second call with the same revision
    downloads nothing.
    """
    return Path(snapshot_download(repo_id=repo_id, revision=revision))
