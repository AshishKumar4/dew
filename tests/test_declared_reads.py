"""Reads of a file as its producer declared it (`dew.data.declared`): an export's manifest, a source's
`file_info` or a generation's, held to the version a store published the file at, checked against its
digest, or read at the commit its URL pins; never as other bytes."""

import contextlib
import hashlib
import http.server
import importlib
import io
import json
import os
import tarfile
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from storage_support import moto_bucket
from token_support import export_byte_tokens

from dew.data.declared import Replaced
from dew.data.sinks import export
from dew.data.sources import Files, Memory
from dew.data.storage import Store


@contextlib.contextmanager
def export_store(transport: str):
    """An export root on `transport` and the storage options that reach it: this host's disk, fsspec's
    memory store, or a bucket on a local S3 server."""
    if transport == "local":
        with tempfile.TemporaryDirectory(prefix="dew-export-") as directory:
            yield str(Path(directory) / "export"), None
        return
    if transport == "memory":
        yield f"memory://export-{uuid4().hex}", None
        return
    with moto_bucket("export") as (bucket, options):
        yield f"s3://{bucket}/export", options


@pytest.mark.parametrize("transport", ["local", "memory", "s3"])
@pytest.mark.parametrize("format", ["parquet", "array_record"])
@pytest.mark.parametrize("read", ["streamed", "indexed"])
def test_a_shard_replaced_under_its_exports_manifest_is_refused_where_it_is_read(transport, format, read):
    """A committed export's manifest states each shard's bytes and digest, which its source's identity
    holds, as a saved run's plan does, and the version its store published it at (row 1871). A shard
    rewritten under the manifest, its size kept or not, is never read under that identity, which does not
    move, by a source whose identity was taken before the rewrite or a fresh one, as a resume reads it:
    each read is refused, or reads the shard as it was published (from Dew's checked copy of it, where a
    reader takes paths alone)."""
    with export_store(transport) as (root, options):
        export(Memory([{"id": index, "text": str(index)} for index in range(8)]), root, format=format,
               rows_per_shard=4, storage_options=options)
        taken = Files(root, format=format, storage_options=options)
        identity = taken.fingerprint()

        def rows(source):
            if read == "streamed":
                manifest = source.manifest()
                return [row["id"] for part in manifest.shards for row in manifest.reader(part)]
            records = source.records("indexed")
            return [records[index]["id"] for index in range(len(records))]

        assert rows(taken) == list(range(8))
        files = json.loads(Store(root + "/_dew_manifest.json", options).read())["files"]
        shard = Store(root, options).child(files[0]["path"])
        held = shard.read()
        middle = len(held) // 2
        for rewritten in (held[:middle] + bytes([held[middle] ^ 1]) + held[middle + 1:], held + b"\0"):
            filesystem, path = shard.resolve()
            filesystem.pipe_file(path, rewritten)
            filesystem.invalidate_cache()
            for source in (Files(root, format=format, storage_options=options), taken):
                if transport == "local" and source is not taken:
                    # A fresh source versions this host's files as it fingerprints them, and refuses there.
                    with pytest.raises(ValueError):
                        source.fingerprint()
                    continue
                assert source.fingerprint() == identity
                with contextlib.suppress(Replaced):
                    assert rows(source) == list(range(8))


@pytest.mark.parametrize("read", ["streamed", "indexed"])
def test_a_declared_line_file_replaced_under_its_facts_is_refused_where_it_is_read(read):
    """A line file read under facts its publisher declared (`file_info`: its bytes and digest, no store's
    version of it) is checked against them where a read opens it: rewritten, it is refused."""
    root = Store(f"memory://declared-lines-{uuid4().hex}/rows.jsonl")
    root.write(b'{"id":0}\n{"id":1}\n')
    Files(root.url).provision()
    index = root.url.removesuffix(".jsonl") + ".index.npy"
    facts = {url: {"bytes": len(Store(url).read()), "sha256": hashlib.sha256(Store(url).read()).hexdigest()}
             for url in (root.url, index)}

    def rows():
        source = Files((root.url,), format="jsonl", index=index, file_info=facts)
        if read == "streamed":
            manifest = source.manifest()
            return [row["id"] for part in manifest.shards for row in manifest.reader(part)]
        records = source.records("indexed")
        return [records[position]["id"] for position in range(len(records))]

    assert rows() == [0, 1]
    root.write(b'{"id":7}\n{"id":1}\n')
    with pytest.raises(Replaced):
        rows()
    # An offset index provisioned for it reads the declared bytes too.
    with pytest.raises(Replaced):
        Files((root.url,), format="jsonl", file_info=facts).provision()


def test_a_declared_archive_replaced_under_its_facts_is_refused_where_it_is_read():
    """A WebDataset archive read under facts its publisher declared (its bytes and digest) is checked
    where a read opens it: replaced by another valid archive, it is refused, not read under them."""
    def archive(text: str) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as written:
            data = text.encode()
            entry = tarfile.TarInfo("0.txt")
            entry.size = len(data)
            written.addfile(entry, io.BytesIO(data))
        return buffer.getvalue()

    root = Store(f"memory://declared-tar-{uuid4().hex}/data.tar")
    root.write(archive("held"))
    facts = {root.url: {"bytes": len(root.read()), "sha256": hashlib.sha256(root.read()).hexdigest()}}

    def rows():
        files = Files((root.url,), format="webdataset", file_info=facts)
        return list(files.manifest().reader(files.manifest().shards[0]))

    assert rows() == [{"__key__": "0", "txt": b"held"}]
    root.write(archive("swap"))
    with pytest.raises(Replaced):
        rows()


def test_a_token_exports_shard_replaced_under_its_manifest_is_refused_where_it_is_read(tmp_path):
    """A token export's shard replaced by another of as many documents and tokens, whose lengths index
    reads alike, is refused where its tokens are read, not read under the manifest's digest."""
    from dew.data.sources import TokenFiles

    (tmp_path / "held").mkdir()
    (tmp_path / "other").mkdir()
    held = export_byte_tokens(tmp_path / "held", ("first", "next"), "--format", "array_record")
    other = export_byte_tokens(tmp_path / "other", ("fjrst", "nexy"), "--format", "array_record")
    taken = TokenFiles(str(held))
    # Five bytes and an end, four and an end.
    assert len(taken.tokens[:]) == 11
    [entry] = json.loads((held / "_dew_manifest.json").read_text())["files"]
    [swapped] = json.loads((other / "_dew_manifest.json").read_text())["files"]
    (held / entry["path"]).write_bytes((other / swapped["path"]).read_bytes())
    with pytest.raises(Replaced):
        TokenFiles(str(held)).tokens[:]


def test_a_token_exports_lengths_index_replaced_under_its_manifest_is_refused_where_it_is_read(tmp_path):
    """A token export's lengths index, which its manifest declares beside its shard, replaced by another
    export's index of as many documents is refused as its shard would be (`Replaced`), not read under the
    manifest's digest."""
    from dew.data.sources import TokenFiles

    (tmp_path / "held").mkdir()
    (tmp_path / "other").mkdir()
    held = export_byte_tokens(tmp_path / "held", ("first", "next"), "--format", "array_record")
    other = export_byte_tokens(tmp_path / "other", ("fjrst", "nexty"), "--format", "array_record")
    [entry] = json.loads((held / "_dew_manifest.json").read_text())["files"]
    [swapped] = json.loads((other / "_dew_manifest.json").read_text())["files"]
    (held / entry["lengths"]["path"]).write_bytes((other / swapped["lengths"]["path"]).read_bytes())
    with pytest.raises(Replaced):
        TokenFiles(str(held)).tokens[:]


def test_a_declared_offset_index_replaced_under_its_facts_is_refused_where_it_is_read():
    """An offset index read under facts its publisher declared (its bytes and digest), replaced by
    another valid index of as many records, is refused (`Replaced`) where an indexed read opens it."""
    root = Store(f"memory://declared-index-{uuid4().hex}/rows.jsonl")
    root.write(b'{"id":0}\n{"id":1}\n')
    Files(root.url).provision()
    index = Store(root.url.removesuffix(".jsonl") + ".index.npy")
    facts = {store.url: {"bytes": len(store.read()), "sha256": hashlib.sha256(store.read()).hexdigest()}
             for store in (root, index)}
    other = Store(f"memory://declared-index-{uuid4().hex}/rows.jsonl")
    other.write(b'{"id":10}\n{"id":11}\n')
    Files(other.url).provision()
    index.write(Store(other.url.removesuffix(".jsonl") + ".index.npy").read())
    with pytest.raises(Replaced):
        records = Files((root.url,), format="jsonl", index=index.url, file_info=facts).records("indexed")
        records[0]


def test_an_indexed_reader_holds_the_declared_file_it_checked(tmp_path):
    """An indexed reader of a local file its publisher declared checks it once, as it first reads it, and
    reads every later record from the file it checked: a file put in its place since (a rename over it)
    leaves the reader reading the declared records, which a fresh reader refuses (`Replaced`); the file it
    holds rewritten in place is refused by the reader holding it too."""
    declared = b"".join(json.dumps({"id": index}).encode() + b"\n" for index in range(4))
    path = tmp_path / "rows.jsonl"
    path.write_bytes(declared)
    Files(str(path)).provision()
    index = str(tmp_path / "rows.index.npy")
    facts = {url: {"bytes": Path(url).stat().st_size,
                   "sha256": hashlib.sha256(Path(url).read_bytes()).hexdigest()}
             for url in (str(path), index)}

    def reader():
        return Files((str(path),), format="jsonl", index=index, file_info=facts).records("indexed")

    def put(payload: bytes) -> None:
        placed = tmp_path / "placed.jsonl"
        placed.write_bytes(payload)
        os.replace(placed, path)

    held = reader()
    assert held[0] == {"id": 0}
    put(declared.replace(b'"id": 2', b'"id": 7'))
    assert [held[position]["id"] for position in range(4)] == [0, 1, 2, 3]
    with pytest.raises(Replaced):
        reader()[0]
    put(declared)
    holding = reader()
    assert holding[0] == {"id": 0}
    with open(path, "r+b") as file:
        file.write(b'{"id": 9}')
    with pytest.raises(Replaced):
        holding[1]


def test_a_hub_export_read_by_its_branch_reads_every_file_at_the_commit_it_named(monkeypatch, tmp_path):
    """An export on a Hub repository read by its branch is read at the commit the branch named as its
    manifest was read, each file it lists at that commit (a Hub file at a commit is one version by its
    URL): the branch moved on since, to a commit whose files hold other records, the source reads the
    records its manifest listed, from the files it listed."""
    from types import SimpleNamespace

    from huggingface_hub import HfApi, HfFileSystem, parse_hf_uri
    from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath

    commits = {}
    for index, ids in enumerate(([0, 1], [10, 11, 12])):
        directory = tmp_path / f"commit-{index}"
        export(Memory([{"id": value} for value in ids]), str(directory / "export"), format="parquet")
        commits[f"{index}" * 40] = directory
    branch = {"main": "0" * 40}

    class LocalHub(HfFileSystem):
        """A Hub of one dataset repository, each commit's files a local directory, `main` naming one."""

        def resolve_path(self, path, revision=None):
            uri = parse_hf_uri(f"hf://{path}")
            return HfFileSystemResolvedRepositoryPath(f"{uri.type}", uri.id, uri.revision or "main",
                                                      uri.path_in_repo)

        def _local(self, path) -> Path:
            resolved = self.resolve_path(path)
            return commits[branch.get(resolved.revision, resolved.revision)] / resolved.path_in_repo

        def exists(self, path, **options):
            return self._local(path).exists()

        def info(self, path, **options):
            return {"name": path, "type": "file", "size": self._local(path).stat().st_size}

        def cat_file(self, path, start=None, end=None, **options):
            return self._local(path).read_bytes()[start:end]

        def _open(self, path, mode="rb", **options):
            return open(self._local(path), mode)

    monkeypatch.setattr(HfApi, "repo_info", lambda self, repo_id, repo_type=None, revision=None, **options:
                        SimpleNamespace(sha=branch.get(revision or "main", revision)))
    # The real Hub's filesystem again once the test ends.
    monkeypatch.setitem(importlib.import_module("fsspec.registry")._registry, "hf", LocalHub)
    source = Files("hf://datasets/owner/data@main/export", format="parquet")
    # Its manifest read, none of its files yet, as a run's identity is taken before its reads.
    assert source.committed is not None
    branch["main"] = "1" * 40
    manifest = source.manifest()
    assert [row["id"] for part in manifest.shards for row in manifest.reader(part)] == [0, 1]


def test_a_declaration_without_a_digest_or_a_version_is_refused_where_it_is_read():
    """Facts that state a file's size alone hold no read of it to anything: a read of it is refused,
    where it would read whatever the file holds under them."""
    root = Store(f"memory://undeclared-{uuid4().hex}/rows.jsonl")
    root.write(b'{"id":0}\n')
    source = Files((root.url,), format="jsonl", file_info={root.url: {"bytes": len(root.read())}})
    manifest = source.manifest()
    with pytest.raises(ValueError):
        list(manifest.reader(manifest.shards[0]))


@contextlib.contextmanager
def etagged_server(objects: dict[str, bytes], token: str | None = None):
    """An HTTP server of `objects`, by path, each named by its content's digest as its ETag and served
    whole or a range of it (`Range`), a read conditioned on another ETag refused (`If-Match`: 412, RFC
    9110 13.1.1), and with `token` a read without it as its bearer refused (401); its URL."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if token is not None and self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            payload = objects.get(self.path)
            if payload is None:
                self.send_response(404)
                self.end_headers()
                return
            etag = '"' + hashlib.sha256(payload).hexdigest() + '"'
            if self.headers.get("If-Match") not in (None, etag):
                self.send_response(412)
                self.end_headers()
                return
            ranged = self.headers.get("Range")
            if ranged is not None:
                first, last = ranged.removeprefix("bytes=").split("-")
                payload = payload[int(first):int(last) + 1]
            self.send_response(206 if ranged is not None else 200)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_an_http_line_file_read_on_its_published_etag_is_refused_once_replaced():
    """A line file served over HTTP, declared by its size and the ETag it was published at, is read a
    range at a time, each read held to that ETag beside the headers its storage options send:
    replaced, its next read is refused, where an unheld read would take the replacement's bytes under
    the declaration."""
    objects = {"/rows.jsonl": b'{"id":0}\n{"id":1}\n'}
    # A server that also asks for the bearer token the storage options send with every read.
    with etagged_server(objects, token="secret") as url:
        location = url + "/rows.jsonl"
        facts = {location: {"bytes": len(objects["/rows.jsonl"]),
                            "ETag": '"' + hashlib.sha256(objects["/rows.jsonl"]).hexdigest() + '"'}}
        options = {"headers": {"Authorization": "Bearer secret"}}

        def rows():
            source = Files((location,), format="jsonl", file_info=facts, storage_options=options)
            manifest = source.manifest()
            return [row["id"] for part in manifest.shards for row in manifest.reader(part)]

        assert rows() == [0, 1]
        objects["/rows.jsonl"] = b'{"id":7}\n{"id":1}\n'
        with pytest.raises(Replaced):
            rows()


def test_a_version_only_array_record_shard_is_refused_where_its_reader_needs_its_path():
    """An ArrayRecord shard declared by its size and store version alone, which its reader would read
    whole by its path where no version holds that read, is refused: the declaration names no digest to
    check a whole copy against."""
    with moto_bucket("version-only") as (bucket, options):
        root = Store(f"s3://{bucket}/export", options)
        export(Memory([{"id": index} for index in range(4)]), root.url, format="array_record",
               rows_per_shard=4, storage_options=options)
        [entry] = json.loads(Store(root.url + "/_dew_manifest.json", options).read())["files"]
        url = root.child(entry["path"]).url
        source = Files((url,), format="array_record", storage_options=options,
                       file_info={url: {"bytes": entry["bytes"], "stored": entry["stored"]}})
        manifest = source.manifest()
        with pytest.raises(ValueError):
            list(manifest.reader(manifest.shards[0]))


def test_an_indexed_reader_lets_go_of_idle_held_copies_where_the_mirror_needs_room(tmp_path):
    """Indexed reads of files declared by their digest on a store no read can be held to a version of,
    each read from its checked copy in Dew's mirror, need room for one copy at a time beside the indexes:
    the copies a reader holds and no read uses let go where the next needs room."""
    local = tmp_path / "local"
    local.mkdir()
    for index in range(3):
        path = local / f"part-{index}.jsonl"
        path.write_bytes(b"".join(json.dumps({"id": 2 * index + row}).encode() + b"\n" for row in range(2)))
        Files(str(path)).provision()
    objects = {f"/{path.name}": path.read_bytes() for path in local.iterdir()}
    with etagged_server(objects) as url:
        data = sorted(name for name in objects if name.endswith(".jsonl"))
        facts = {url + name: {"bytes": len(objects[name]),
                              "sha256": hashlib.sha256(objects[name]).hexdigest()} for name in data}
        indexes = sum(len(payload) for name, payload in objects.items() if not name.endswith(".jsonl"))
        options = {"dewcache": {"cache_storage": str(tmp_path / "mirror"),
                                "max_bytes": indexes + max(len(objects[name]) for name in data)}}
        records = Files(tuple(url + name for name in data), format="jsonl", file_info=facts,
                        storage_options=options).records("indexed")
        assert [records[position]["id"] for position in range(len(records))] == list(range(6))


def test_declared_files_held_for_ranged_reads_keep_their_mirror_copy_while_held(tmp_path):
    """Files declared by their digest on a store no read can be held to a version of (an HTTP server
    whose ETags the declaration does not state), held for reads of ranges of them as an offset index
    reads them (`Declared.held`), are each read whole into Dew's mirror, checked, and read from that copy,
    which the mirror keeps while it is held: a mirror with room for one of them refuses to hold a second
    beside it, and holds each in turn once the one before lets go."""
    from dew.data.declared import Declared

    objects = {f"/part-{index}.jsonl": json.dumps({"id": index}).encode() + b"\n" for index in range(3)}
    with etagged_server(objects) as url:
        options = {"dewcache": {"cache_storage": str(tmp_path / "mirror"),
                                "max_bytes": max(map(len, objects.values()))}}
        declared = [Declared(Store(url + name, options), {"bytes": len(payload),
                                                          "sha256": hashlib.sha256(payload).hexdigest()})
                    for name, payload in objects.items()]
        first = declared[0].held()
        assert json.loads(first[0:declared[0].size]) == {"id": 0}
        with pytest.raises(ValueError):
            declared[1].held()
        first.close()
        for index, each in enumerate(declared):
            held = each.held()
            assert json.loads(held[0:each.size]) == {"id": index}
            held.close()
