"""Resource release, record validation, memory materialization and position bounds across data sources."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import tarfile
from uuid import uuid4

import grain.python as grain
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition, Loading
from dew.data.declared import Declared, Replaced
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.sources import Files, TokenFiles, from_grain, from_hf, from_iter
from dew.data.sources.records import ArrayRecords
from dew.data.storage import EXPORT_MANIFEST, Store


def _open_paths() -> set[str]:
    return {opened.path for opened in psutil.Process().open_files()}


def test_indexed_line_token_and_arrayrecord_sources_release_files_and_leases_on_close(tmp_path):
    """Closing indexed JSONL (projected or whole) and TokenFiles readers, reading `Files` or
    `TokenFiles` into memory, or failing partway through `ArrayRecords.reader` releases every held
    descriptor and bounded mirror lease without relying on garbage collection."""
    jsonl_path = tmp_path / "declared.jsonl"
    jsonl_path.write_bytes(b"".join(json.dumps({"id": i, "extra": i}).encode() + b"\n" for i in range(8)))
    Files(str(jsonl_path)).provision()
    index_path = tmp_path / "declared.index.npy"
    facts = {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (jsonl_path, index_path)
    }
    real_jsonl = os.path.realpath(jsonl_path)

    gc.disable()
    try:
        for columns in (None, ("id",)):
            source = Files((str(jsonl_path),), format="jsonl", index=str(index_path), columns=columns,
                           file_info=facts)
            assert source.metadata()["rows"] == 8
            assert source.locality()[0] == (8,)
            assert real_jsonl not in _open_paths()
            records = source.records("indexed")
            assert isinstance(records, Closeable)
            assert records[0]["id"] == 0
            assert real_jsonl in _open_paths()
            records.close()
            assert real_jsonl not in _open_paths()

        token_dir = tmp_path / "tokens"
        export(
            [{"text": np.asarray([1, 2, 0], np.int32)}, {"text": np.asarray([3, 4, 0], np.int32)}],
            str(token_dir),
            format="array_record",
            rows_per_shard=1,
            length_field="text",
            metadata={
                "token_store": True, "dtype": "int32", "tokenizer": "byte", "vocab_size": 256, "eos_id": 0,
            },
        )
        shard_reals = {os.path.realpath(path) for path in token_dir.rglob("*.array_record")}
        token_source = TokenFiles(str(token_dir))
        assert len(token_source.records("memory")) == 2
        assert not (shard_reals & _open_paths())
        token_docs = token_source.records("indexed")
        assert isinstance(token_docs, Closeable)
        assert token_docs[0]["text"].tolist() == [1, 2, 0]
        assert shard_reals & _open_paths()
        token_docs.close()
        assert not (shard_reals & _open_paths())

        # Partial failure in ArrayRecords.reader releases mirror leases acquired for earlier shards.
        shard_bytes = next(token_dir.rglob("*.array_record")).read_bytes()
        remote_root = Store(f"memory://partial-open-{uuid4().hex}")
        first = remote_root.child("0.array_record")
        second = remote_root.child("1.array_record")
        first.write(shard_bytes)
        second.write(shard_bytes)
        options = {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": len(shard_bytes)}}
        good_decl = Declared(Store(first.url, options), {
            "bytes": len(shard_bytes),
            "sha256": hashlib.sha256(shard_bytes).hexdigest(),
        })
        bad_decl = Declared(Store(second.url, options), {
            "bytes": len(shard_bytes),
            "sha256": "0" * 64,
        })
        failing = ArrayRecords((first.url, second.url), options, declared=(good_decl, bad_decl))
        with pytest.raises((Replaced, ValueError)):
            _ = failing.reader
        # Because `first`'s lease was closed on failure, the 1-shard mirror has room to open `second`.
        second_decl = Declared(Store(second.url, options), {
            "bytes": len(shard_bytes),
            "sha256": hashlib.sha256(shard_bytes).hexdigest(),
        })
        recovered = ArrayRecords((second.url,), options, declared=(second_decl,))
        try:
            assert len(recovered) == 1
        finally:
            recovered.close()
    finally:
        gc.enable()


def test_json_grain_and_hf_sources_validate_records_materialize_memory_and_shard_streams(tmp_path):
    """JSON/JSONL and Grain sources refuse non-mapping records and out-of-bounds positions, while
    `from_grain(MapDataset)` and `from_hf(Dataset)` materialize `memory` reads once and shard
    multi-thousand-row `streamed` reads across multiple shares."""
    import datasets

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[1, 2]\n")
    with pytest.raises(TypeError):
        list(Files(str(bad_jsonl)).records("memory"))

    json_array = tmp_path / "array.json"
    json_array.write_text(json.dumps([{"id": 0}, {"id": 1}]))
    manifest = Files(str(json_array)).manifest()
    iterator = iter(manifest.reader(manifest.shards[0]))
    try:
        for invalid in (-1, 3, True):
            with pytest.raises(ValueError):
                iterator.set_state({"read": invalid})
    finally:
        iterator.close()

    bad_json = tmp_path / "bad_array.json"
    bad_json.write_text("[42]")
    with pytest.raises(TypeError):
        list(Files(str(bad_json)).records("memory"))

    with pytest.raises(TypeError):
        list(from_grain(grain.MapDataset.source([42]), "bad@1").records("memory"))

    reads = 0

    class CountedSequence:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, int]:
            nonlocal reads
            reads += 1
            return {"id": index}

    grain_source = from_grain(grain.MapDataset.source(CountedSequence()), "counted@1")
    loader = compile(
        Pipeline(grain_source).repeat(3).batch(2),
        read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1),
    )
    reader = loader.train(DataPartition())
    try:
        batches = [int(v) for batch in reader for v in batch["id"]]
    finally:
        reader.close()
    assert batches == [0, 1, 2, 3] * 3
    assert reads == 4

    hf_arrow = from_hf(datasets.Dataset.from_dict({"id": list(range(1500))}), identity="hf-arrow@1")
    assert isinstance(hf_arrow.records("memory"), tuple)
    grain_large = from_grain(
        grain.MapDataset.source([{"id": index} for index in range(1500)]),
        "grain-large@1",
    )
    for source in (grain_large, hf_arrow):
        # Single-reader streams keep one whole-store shard, so checkpoints past 1,024 rows resume.
        single_manifest = source.manifest()
        assert len(single_manifest.shards) == 1 and single_manifest.shards[0].rows == 1500
        single_loader = compile(
            Pipeline(source).batch(1),
            read="streamed",
            epochs=1,
            loading=Loading(workers=0, threads=1, read_buffer=1),
        )
        saving, resuming = single_loader.train(DataPartition()), single_loader.train(DataPartition())
        try:
            for _ in range(1100):
                next(saving)
            saved_pos = saving.get_state()
            resuming.set_state(saved_pos)
            assert int(next(resuming)["id"][0]) == int(next(saving)["id"][0])
        finally:
            saving.close()
            resuming.close()

        compiled = compile(
            Pipeline(source).batch(4),
            read="streamed",
            epochs=1,
            loading=Loading(workers=0, threads=1, read_buffer=1),
        )
        seen: list[int] = []
        for share in range(2):
            share_reader = compiled.train(DataPartition(share, 2))
            try:
                seen.extend(int(v) for batch in share_reader for v in batch["id"])
            finally:
                share_reader.close()
        assert sorted(seen) == list(range(1500))


def test_source_iterators_and_token_exports_reject_boolean_positions_and_eos_ids(tmp_path):
    """`bool` is a subclass of `int` in Python, so checkpointed positions and `eos_id` metadata
    explicitly refuse `True` and `False` rather than reading them as `1` and `0`."""

    parquet_path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": 0}, {"id": 1}]), parquet_path)
    pq_manifest = Files(str(parquet_path)).manifest()
    pq_iter = iter(pq_manifest.reader(pq_manifest.shards[0]))
    try:
        with pytest.raises(ValueError):
            pq_iter.set_state({"group": False, "row": True})
    finally:
        pq_iter.close()

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('{"id": 0}\n{"id": 1}\n')
    line_manifest = Files(str(jsonl_path)).manifest()
    line_iter = iter(line_manifest.reader(line_manifest.shards[0]))
    try:
        with pytest.raises(ValueError):
            line_iter.set_state({"offset": True})
    finally:
        line_iter.close()

    tar_path = tmp_path / "rows.tar"
    with tarfile.open(tar_path, mode="w") as archive:
        info = tarfile.TarInfo("0.txt")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"hi"))
    tar_manifest = Files(str(tar_path)).manifest()
    tar_iter = iter(tar_manifest.reader(tar_manifest.shards[0]))
    try:
        with pytest.raises(ValueError):
            tar_iter.set_state({"position": True})
    finally:
        tar_iter.close()

    iter_source = from_iter(
        lambda: iter([{"id": 0}]),
        identity="bool-pos@1",
        state=lambda _: 0,
        load_state=lambda _, saved: None,
    )
    iter_manifest = iter_source.manifest()
    stream_iter = iter(iter_manifest.reader(iter_manifest.shards[0]))
    try:
        saved = dict(stream_iter.get_state())
        saved["read"] = True
        with pytest.raises(ValueError):
            stream_iter.set_state(saved)
    finally:
        stream_iter.close()

    token_dir = tmp_path / "bad-eos"
    export(
        [{"text": np.asarray([1, 2], np.int32)}],
        str(token_dir),
        format="array_record",
        length_field="text",
        metadata={
            "token_store": True, "dtype": "int32", "tokenizer": "byte", "vocab_size": 256, "eos_id": None,
        },
    )
    manifest_doc = json.loads((token_dir / EXPORT_MANIFEST).read_text())
    manifest_doc["metadata"]["eos_id"] = True
    (token_dir / EXPORT_MANIFEST).write_text(json.dumps(manifest_doc))
    with pytest.raises(ValueError):
        _ = TokenFiles(str(token_dir)).tokens

    from dew.data.cache import Cache
    from dew.data.plan import Manifest

    cache_store = Cache(str(tmp_path / "cache-ord"))
    manifest_key = Manifest(Files(str(jsonl_path)).fingerprint())
    cached_opener = cache_store.streamed(line_manifest.reader, manifest_key)
    cached_iter = iter(cached_opener(line_manifest.shards[0]))
    try:
        parent_state = {"offset": line_manifest.shards[0].start}
        cached_iter.set_state({"parent": parent_state, "ordinal": 0})
        for bad_ord in (True, -1):
            with pytest.raises(ValueError):
                cached_iter.set_state({"parent": parent_state, "ordinal": bad_ord})
    finally:
        cached_iter.close()
