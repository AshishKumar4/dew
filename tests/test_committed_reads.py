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

    from dew.data.storage import _CHECKED

    with moto_bucket("committed") as (bucket, options):
        root = f"s3://{bucket}/export"
        export(Memory([{"id": index} for index in range(6)]), root, storage_options=options)
        manifest = json.loads(fsspec.filesystem("s3", **options).cat(f"{bucket}/export/_dew_manifest.json"))
        shards = [entry for entry in manifest["files"] if entry["path"].endswith(".parquet")]
        assert shards and all(entry["version"].startswith("etag:") for entry in shards)
        assert sorted(int(row["id"]) for row in iterate(Pipeline(Files(root, storage_options=options)),
                                                        read="streamed")) == list(range(6))
        _replaced_shard(fsspec.filesystem("s3", **options), f"{bucket}/export", range(100, 106))
        # A new process's read, which asks the store again.
        _CHECKED.clear()
        with pytest.raises(OSError):
            list(iterate(Pipeline(Files(root, storage_options=options)), read="streamed"))


@pytest.mark.parametrize("replaced", [False, True])
def test_a_remote_file_committed_by_its_digest_alone_is_read_from_its_checked_mirror(tmp_path, replaced):
    """A file whose manifest commits it by its sha256 alone, as a manifest written before stores'
    versions were recorded does, is read from its copy in the transport's mirror, checked whole against
    the digest as it downloads: it reads its rows, a shard replaced since is refused, and a mirror with
    no room for it refuses it, naming the remedy, rather than streaming it unchecked."""
    import fsspec

    from dew.data.storage import _CHECKED

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
        # A new process's read, which asks the store again.
        _CHECKED.clear()
        read = Pipeline(Files(root, storage_options=mirrored))
        if replaced:
            with pytest.raises(OSError):
                list(iterate(read, read="streamed"))
            return
        assert sorted(int(row["id"]) for row in iterate(read, read="streamed")) == list(range(6))
        # A new process's read, which asks the store again.
        _CHECKED.clear()
        cramped = {**options, "dewcache": {"cache_storage": str(tmp_path / "cramped"), "max_bytes": 1}}
        with pytest.raises(OSError):
            list(iterate(Pipeline(Files(root, storage_options=cramped)), read="streamed"))


def test_a_worker_checks_the_files_it_opens_after_its_parent_compiled_the_read():
    """A read compiled in one process and read by a loading worker checks each committed file in the
    worker, as the worker opens it: a shard replaced after the parent found its fragments is refused,
    where the worker once read the parent's fragments unchecked."""
    import fsspec

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading

    with moto_bucket("worker") as (bucket, options):
        root = f"s3://{bucket}/export"
        export(Memory([{"id": index} for index in range(6)]), root, storage_options=options)
        loader = compile(Pipeline(Files(root, storage_options=options)).batch(2), read="streamed",
                         loading=Loading(workers=1))
        _replaced_shard(fsspec.filesystem("s3", **options), f"{bucket}/export", range(100, 106))
        stream = loader.train(DataPartition())
        try:
            with pytest.raises(OSError):
                next(stream)
        finally:
            stream.close()


def test_a_suspended_s3_buckets_null_version_names_no_version_and_the_etag_stands_for_it():
    """An S3 bucket whose versioning is suspended states every object's version as "null", which a
    replacement keeps: the version a writer records, and a reader checks, is the object's ETag."""
    from dew.data.storage import store_version

    assert store_version({"VersionId": "null", "ETag": '"5d41402a"'}) == "etag:5d41402a"
    assert store_version({"VersionId": "3HL4kqtJ", "ETag": '"5d41402a"'}) == "versionid:3HL4kqtJ"
    assert store_version({"generation": "1700000000000000"}) == "generation:1700000000000000"
    assert store_version({"size": 5}) is None


def _digests_alone(remote, manifest_path: str) -> None:
    """The manifest at `manifest_path` with its files' store versions taken out, as a manifest written
    before they were recorded holds its files' digests alone."""
    manifest = json.loads(remote.cat(manifest_path))
    for entry in manifest["files"]:
        entry.pop("version", None)
    remote.pipe(manifest_path, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


def test_a_partitioned_export_read_from_its_mirror_keeps_its_partitions(tmp_path):
    """A partitioned export whose manifest commits its files by their digests alone is read from its
    mirrored copies by the paths it was written under, so its partition column, which only the paths
    hold, reads back with each row."""
    import fsspec

    with moto_bucket("partitioned") as (bucket, options):
        remote = fsspec.filesystem("s3", **options)
        root = f"s3://{bucket}/export"
        rows = [{"id": index, "language": "en" if index % 2 else "fr"} for index in range(6)]
        export(Memory(rows), root, storage_options=options, partitioning=("language",))
        _digests_alone(remote, f"{bucket}/export/_dew_manifest.json")
        mirrored = {**options, "dewcache": {"cache_storage": str(tmp_path / "mirror")}}
        read = Files(root, storage_options=mirrored, partitioning="hive")
        assert sorted((int(row["id"]), str(row["language"])) for row in iterate(Pipeline(read),
                                                                                   read="streamed")) == [
            (int(row["id"]), row["language"]) for row in rows]


def test_a_compressed_line_file_read_from_its_mirror_is_decompressed_by_its_own_name(tmp_path):
    """A gzipped JSON lines file a manifest commits by its digest alone is read from its mirrored copy,
    whose own name names no compression, decompressed as the file's URL names it."""
    import gzip
    import hashlib

    import fsspec

    with moto_bucket("lines") as (bucket, options):
        remote = fsspec.filesystem("s3", **options)
        payload = gzip.compress(b"".join(json.dumps({"id": index}).encode() + b"\n" for index in range(4)))
        remote.pipe(f"{bucket}/lines/part.jsonl.gz", payload)
        manifest = {"schema": 1, "format": "jsonl", "codec": None, "rows": 4, "metadata": {},
                    "files": [{"path": "part.jsonl.gz", "bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()}]}
        remote.pipe(f"{bucket}/lines/_dew_manifest.json", json.dumps(manifest).encode())
        mirrored = {**options, "dewcache": {"cache_storage": str(tmp_path / "mirror")}}
        read = Files(f"s3://{bucket}/lines", storage_options=mirrored)
        assert sorted(int(row["id"]) for row in iterate(Pipeline(read), read="streamed")) == list(range(4))


def test_a_token_exports_shard_replaced_by_one_of_equal_document_lengths_is_refused(tmp_path):
    """A token export's shards are read as its manifest committed them: a shard replaced by another
    export's of the same documents' lengths and other tokens is refused, where its reader once read
    the other tokens under the export's identity."""
    import fsspec
    from token_support import export_byte_tokens

    from dew.data.sources import TokenFiles

    first = export_byte_tokens(tmp_path / "first", ("first", "next"), "--format", "array_record")
    other = export_byte_tokens(tmp_path / "other", ("fixed", "nope"), "--format", "array_record")
    [shard] = sorted(first.glob("shards/*/*.array_record"))
    [replacement] = sorted(other.glob("shards/*/*.array_record"))
    with moto_bucket("tokens") as (bucket, options):
        remote = fsspec.filesystem("s3", **options)
        remote.put(str(first), f"{bucket}/export", recursive=True)
        remote.put(str(replacement), f"{bucket}/export/{shard.relative_to(first).as_posix()}")
        mirrored = {**options, "dewcache": {"cache_storage": str(tmp_path / "mirror")}}
        with pytest.raises(OSError):
            TokenFiles(f"s3://{bucket}/export", storage_options=mirrored).tokens[:]

