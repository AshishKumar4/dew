"""Moving an export directory to and from the Hugging Face Hub.

`save_hf_layout` writes the directory; these two carry it. They wrap the two
`huggingface_hub` calls involved, creating the repo on the way up and handing
back the snapshot path on the way down. Retries, progress and caching stay the
hub client's behaviour.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def push_to_hub(directory, repo_id: str, *, private: bool = False,
                commit_message: str = "Upload dew export", raw: bool = False) -> None:
    """Upload `directory` to `repo_id`, creating the repo when it is missing.

    The files land at the root of the repo under the names they have on disk,
    so an export written by `save_hf_layout` arrives as the model.safetensors
    and config.json pair a Hugging Face loader looks for.

    A run directory is exported first, because the orbax checkpoint and the
    `run.json` beside it are Dew's own format and nothing on the Hub reads
    them: what goes up is what `export_run` writes. `raw` uploads the run
    directory itself instead, which is the form `TextToImage.from_run` and
    the text tasks' `from_pretrained` pull back.
    """
    from dew.checkpoints import RUN_FILE

    source = Path(directory)
    if not raw and (source / RUN_FILE).is_file():
        from dew.interop.export import export_run

        with tempfile.TemporaryDirectory() as staged:
            export_run(os.fspath(source), staged)
            _upload(staged, repo_id, private=private, commit_message=commit_message)
        return
    _upload(os.fspath(source), repo_id, private=private, commit_message=commit_message)


def _upload(folder: str, repo_id: str, *, private: bool, commit_message: str) -> None:
    """The two hub calls one upload is, over a directory already on disk."""
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        commit_message=commit_message,
    )


def pull_from_hub(repo_id: str, revision: str | None = None) -> Path:
    """Download a snapshot of `repo_id` and return the directory holding it.

    `revision` is a branch, tag or commit; None takes the default branch. The
    path is inside the hub cache, so a second call with the same revision
    downloads nothing.
    """
    return Path(snapshot_download(repo_id=repo_id, revision=revision))
