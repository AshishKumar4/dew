# Copyright The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""huggingface_hub's DDUF reader, vendored, over a seekable binary stream in place of a path.

The body of `read_dduf_file` and its checks (`_validate_dduf_entry_name`, `_validate_dduf_structure`,
`_get_data_offset`, `DDUF_ALLOWED_ENTRIES`, `DDUF_FOLDER_REQUIRED_ENTRIES`) in
`src/huggingface_hub/serialization/_dduf.py` at huggingface_hub v1.33.0 (commit
9eeace5e467939b8f6af6ff280a134bc7c22802f; license above), carried for the one read the public function
cannot make: a Hub archive's entries through a seekable stream (`HfFileSystem.open`), its weights never
fetched (`dew.interop.dduf.metadata`). Every check and its order as upstream's; an entry is its data's
offset and length, which upstream's `DDUFEntry` holds beside its path. A local archive is read by the
public `read_dduf_file`. Delete this module when `read_dduf_file` takes a binary stream (ledger 5762).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from typing import IO

DDUF_ALLOWED_ENTRIES = {".json", ".model", ".safetensors", ".txt"}
DDUF_FOLDER_REQUIRED_ENTRIES = {"config.json", "tokenizer_config.json", "preprocessor_config.json",
                                "scheduler_config.json"}


def read_dduf_stream(stream: IO[bytes]) -> dict[str, tuple[int, int]]:
    """Each entry of the DDUF archive `stream` holds, its data's offset in the archive and its length, the
    archive's structure checked as `read_dduf_file` checks it; only its directory, each entry's local
    header and `model_index.json` are read."""
    from huggingface_hub.errors import DDUFCorruptedFileError, DDUFInvalidEntryNameError

    entries = {}
    with zipfile.ZipFile(stream, "r") as zf:
        for member in zf.infolist():
            if member.compress_type != zipfile.ZIP_STORED:
                raise DDUFCorruptedFileError("Data must not be compressed in DDUF file.")
            try:
                _validate_dduf_entry_name(member.filename)
            except DDUFInvalidEntryNameError as e:
                raise DDUFCorruptedFileError(f"Invalid entry name in DDUF file: {member.filename}") from e
            entries[member.filename] = (_get_data_offset(stream, member), member.file_size)
    if "model_index.json" not in entries:
        raise DDUFCorruptedFileError("Missing required 'model_index.json' entry in DDUF file.")
    offset, length = entries["model_index.json"]
    stream.seek(offset)
    _validate_dduf_structure(json.loads(stream.read(length).decode("utf-8")), entries.keys())
    return entries


def _validate_dduf_entry_name(entry_name: str) -> str:
    from huggingface_hub.errors import DDUFInvalidEntryNameError

    if "." + entry_name.split(".")[-1] not in DDUF_ALLOWED_ENTRIES:
        raise DDUFInvalidEntryNameError(f"File type not allowed: {entry_name}")
    if "\\" in entry_name:
        raise DDUFInvalidEntryNameError(f"Entry names must use UNIX separators ('/'). Got {entry_name}.")
    entry_name = entry_name.strip("/")
    if entry_name.count("/") > 1:
        raise DDUFInvalidEntryNameError(f"DDUF only supports 1 level of directory. Got {entry_name}.")
    return entry_name


def _validate_dduf_structure(index: object, entry_names: Iterable[str]) -> None:
    from huggingface_hub.errors import DDUFCorruptedFileError

    if not isinstance(index, dict):
        raise DDUFCorruptedFileError(
            f"Invalid 'model_index.json' content. Must be a dictionary. Got {type(index)}.")
    dduf_folders = {entry.split("/")[0] for entry in entry_names if "/" in entry}
    for folder in dduf_folders:
        if folder not in index:
            raise DDUFCorruptedFileError(f"Missing required entry '{folder}' in 'model_index.json'.")
        if not any(f"{folder}/{required_entry}" in entry_names
                   for required_entry in DDUF_FOLDER_REQUIRED_ENTRIES):
            raise DDUFCorruptedFileError(
                f"Missing required file in folder '{folder}'. Must contains at least one of "
                f"{DDUF_FOLDER_REQUIRED_ENTRIES}."
            )


def _get_data_offset(stream: IO[bytes], member: zipfile.ZipInfo) -> int:
    """The offset of `member`'s data in the archive: past its local file header (upstream reads it through
    the `ZipFile`'s own handle on the archive, this one through the stream it was opened over)."""
    from huggingface_hub.errors import DDUFCorruptedFileError

    stream.seek(member.header_offset)
    local_file_header = stream.read(30)
    if len(local_file_header) < 30:
        raise DDUFCorruptedFileError("Incomplete local file header.")
    filename_len = int.from_bytes(local_file_header[26:28], "little")
    extra_field_len = int.from_bytes(local_file_header[28:30], "little")
    return member.header_offset + 30 + filename_len + extra_field_len
