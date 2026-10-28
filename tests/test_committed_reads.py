"""Remote files read as their manifest committed them (row 1871, DATA-03): a Dew export records each
file's store version beside its digest, and a read is checked against it as it opens the file; a file
committed by its digest alone is read from its copy in the transport's mirror, checked whole as it
downloads, and refused where the mirror cannot hold it."""

import json

import pytest
from storage_support import moto_bucket

from dew.data.compile import iterate
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.sources import Files, Memory


def _replaced_shard(remote, root: str, ids: range) -> str:
    """The export's one Parquet shard under `root` replaced by another of as many rows, other ids, as a
    writer replacing a file under a committed manifest would leave it; its path."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    [shard] = [path for path in remote.find(root) if path.endswith(".parquet")]
    written = io.BytesIO()
    pq.write_table(pa.table({"id": list(ids)}), written)
    remote.pipe(shard, written.getvalue())
    return shard


def test_a_remote_export_is_read_at_the_version_it_committed_and_a_replaced_shard_is_refused():
    """A Dew export to a store that versions its objects records each file's store version beside its
    digest, and a read of it is checked against that version as it opens the file: its rows read
    back, and a shard replaced since by as many other rows is refused, where it once read the other
    rows under the saved fingerprint (row 1871, DATA-03)."""
    import fsspec

    from dew.data.sources.files import _COMMITTED

    with moto_bucket("committed") as (bucket, options):
        root = f"s3://{bucket}/export"
        export(Memory([{"id": index} for index in range(6)]), root, storage_options=options)
        manifest = json.loads(fsspec.filesystem("s3", **options).cat(f"{bucket}/export/_dew_manifest.json"))
        shards = [entry for entry in manifest["files"] if entry["path"].endswith(".parquet")]
        assert shards and all(entry["version"].startswith("etag:") for entry in shards)
        assert sorted(int(row["id"]) for row in iterate(Pipeline(Files(root, storage_options=options)),
                                                        read="streamed")) == list(range(6))
        _replaced_shard(fsspec.filesystem("s3", **options), f"{bucket}/export", range(100, 106))
        _COMMITTED.clear()
        with pytest.raises(OSError):
            list(iterate(Pipeline(Files(root, storage_options=options)), read="streamed"))


@pytest.mark.parametrize("replaced", [False, True])
def test_a_remote_file_committed_by_its_digest_alone_is_read_from_its_checked_mirror(tmp_path, replaced):
    """A file whose manifest commits it by its sha256 alone, as a manifest written before stores'
    versions were recorded does, is read from its copy in the transport's mirror, checked whole against
    the digest as it downloads: it reads its rows, a shard replaced since is refused, and a mirror with
    no room for it refuses it, naming the remedy, rather than streaming it unchecked."""
    import fsspec

    from dew.data.sources.files import _COMMITTED

    with moto_bucket("digest") as (bucket, options):
        remote = fsspec.filesystem("s3", **options)
        root = f"s3://{bucket}/export"
        export(Memory([{"id": index} for index in range(6)]), root, storage_options=options)
        manifest_path = f"{bucket}/export/_dew_manifest.json"
        manifest = json.loads(remote.cat(manifest_path))
        # A manifest of digests alone, as one written before stores' versions were recorded is.
        for entry in manifest["files"]:
            entry.pop("version", None)
        remote.pipe(manifest_path, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        mirrored = {**options, "dewcache": {"cache_storage": str(tmp_path / "mirror")}}
        if replaced:
            _replaced_shard(remote, f"{bucket}/export", range(100, 106))
        _COMMITTED.clear()
        read = Pipeline(Files(root, storage_options=mirrored))
        if replaced:
            with pytest.raises(OSError):
                list(iterate(read, read="streamed"))
            return
        assert sorted(int(row["id"]) for row in iterate(read, read="streamed")) == list(range(6))
        _COMMITTED.clear()
        cramped = {**options, "dewcache": {"cache_storage": str(tmp_path / "cramped"), "max_bytes": 1}}
        with pytest.raises(OSError):
            list(iterate(Pipeline(Files(root, storage_options=cramped)), read="streamed"))
