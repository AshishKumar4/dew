"""Artifact wire-schema round trips across cache, export, iterator checkpoints and validation."""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import math
from decimal import Decimal
from zoneinfo import ZoneInfo

import ml_dtypes
import numpy as np
import pytest
from process_support import REPO_ROOT, run_pool, spawn

from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition, Loading
from dew.data.sinks import export, write_array_record
from dew.data.sources import Files, Memory
from dew.data.sources.codec import Unencodable, content_fingerprint, decode, encode, pack_dict_of_byte_arrays
from dew.data.sources.iterator import from_iter
from dew.data.steps.text import ValidTargets
from dew.data.storage import Store, commit_export
from dew.distributed.batches import restore_position


def test_artifact_codec_preserves_arrow_scalars_and_nested_arrays():
    row = {
        "when": datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.UTC),
        "price": Decimal("12.340"),
        "interval": datetime.timedelta(days=2, seconds=3),
        "features": [np.arange(4, dtype=np.uint16), {"nan": float("nan")}],
        "binary": b"\0\xff",
        "messages": np.asarray([[{"role": "user", "content": "hello"}]], dtype=object),
    }
    actual = decode(encode(row))
    assert actual["when"] == row["when"]
    assert actual["price"].as_tuple() == row["price"].as_tuple()
    assert actual["interval"] == row["interval"]
    assert np.array_equal(actual["features"][0], row["features"][0])
    assert np.isnan(actual["features"][1]["nan"])
    assert actual["binary"] == row["binary"]
    assert actual["messages"].dtype == row["messages"].dtype
    assert actual["messages"].tolist() == row["messages"].tolist()


def test_artifact_codec_preserves_extended_numpy_dtypes():
    values = {"array": np.asarray([1.0, 2.5], dtype=ml_dtypes.bfloat16), "scalar": ml_dtypes.bfloat16(3.0)}
    restored = decode(encode(values))
    assert restored["array"].dtype == values["array"].dtype
    assert restored["scalar"].dtype == values["scalar"].dtype
    assert np.array_equal(restored["array"], values["array"])
    assert restored["scalar"] == values["scalar"]


@pytest.mark.parametrize("value", [
    np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
    np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
    np.arange(6, dtype=">i4"),
    np.asarray(3.5, dtype=np.float64),
    np.zeros((0, 5), dtype=np.int16),
    np.asarray([True, False]),
    np.asarray(["ab", "cde"]),
    np.asarray([b"ab", b"c"]),
    np.asarray([1, 2], dtype="datetime64[ns]"),
    np.zeros(2, dtype=[("x", "<f4"), ("y", "<i8")]),
    np.arange(2 ** 17, dtype=np.int64).reshape(256, 512),
    np.ndarray((2,), dtype="V0"),
    np.zeros(3, dtype="S0"),
], ids=["uint8-3d", "fortran", "big-endian", "0d", "empty", "bool", "unicode", "bytes", "datetime",
        "structured", "long-header-free", "void-0", "bytes-0"])
def test_artifact_codec_reads_back_every_array_as_saved(value):
    """Every array a record holds reads back as it was saved, its dtype, shape, memory order and
    values, whether its header is read straight or by `np.load`, and owns its writable memory."""
    restored = decode(encode({"value": value}))["value"]
    assert restored.dtype == value.dtype and restored.shape == value.shape
    assert restored.flags.f_contiguous == value.flags.f_contiguous or value.ndim < 2
    assert restored.flags.writeable
    np.testing.assert_array_equal(restored, value)


def test_artifact_codec_reads_back_a_complex_field_exactly():
    """A record's Python complex field reads back as written, both parts exactly, as a record holds it
    beside its other fields."""
    values = {"z": complex(0.1, -2.5e-300), "zs": [complex(1, 0), complex(float("inf"), float("nan"))]}
    restored = decode(encode(values))
    assert restored["z"] == values["z"] and isinstance(restored["z"], complex)
    assert restored["zs"][0] == values["zs"][0]
    assert restored["zs"][1].real == float("inf") and np.isnan(restored["zs"][1].imag)


def test_content_fingerprint_preserves_array_identity_and_sequence_order():
    ordered = {"a": [b"first", b"last"], "z": np.arange(6, dtype=np.int32).reshape(2, 3)}
    reordered = {"z": ordered["z"], "a": ordered["a"]}
    digest = content_fingerprint(ordered)
    assert content_fingerprint(reordered) == digest == hashlib.sha256(encode(ordered)).digest()
    assert encode(reordered) != encode(ordered)
    assert list(decode(encode(reordered))) == ["z", "a"]
    variants = [
        {**ordered, "z": ordered["z"].astype(np.int64)},
        {**ordered, "z": ordered["z"].reshape(3, 2)},
        {**ordered, "z": ordered["z"] + 1},
        {**ordered, "a": list(reversed(ordered["a"]))},
        {**ordered, "a": tuple(ordered["a"])},
    ]
    assert all(content_fingerprint(row) != digest for row in variants)
    assert content_fingerprint(np.int16(4)) != content_fingerprint(np.int32(4))

    boundary = np.zeros((1,) * 15, dtype=np.int32)
    npy_ref = io.BytesIO()
    np.save(npy_ref, boundary, allow_pickle=False)
    assert pack_dict_of_byte_arrays is not None
    from dew.data.sources.codec import unpack_dict_of_byte_arrays

    assert unpack_dict_of_byte_arrays(encode(boundary))["0"] == npy_ref.getvalue()
    replay_edge = {
        "ts": np.asarray([1, 2], dtype="datetime64[ns]"),
        "td": np.asarray([3, 4], dtype="timedelta64[s]"),
        "empty": np.zeros((0, 5), dtype=np.int16),
        "boundary": boundary,
    }
    assert len(content_fingerprint(replay_edge, opaque=True)) == 32


def _variant_record(index: int) -> dict:
    zone = ZoneInfo("America/New_York")
    est = datetime.timezone(datetime.timedelta(hours=-5), "EST")
    objects = np.empty(2, dtype=object)
    objects[0] = np.asarray([index, index + 1], dtype=np.int32)
    objects[1] = np.asarray([index + 2, index + 3], dtype=np.int32)
    return {
        "id": index,
        "fall_0": datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0),
        "fall_1": datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1),
        "uncached_0": datetime.datetime(
            2026, 11, 1, 1, 30, tzinfo=ZoneInfo.no_cache("America/New_York"), fold=0
        ),
        "uncached_1": datetime.datetime(
            2026, 11, 1, 1, 30, tzinfo=ZoneInfo.no_cache("America/New_York"), fold=1
        ),
        "named_tz": datetime.datetime(2026, 1, 2, 3, 4, tzinfo=est, fold=1),
        "alarm": datetime.time(1, 30, 15, tzinfo=zone, fold=1),
        "day": datetime.date(2026, 11, 1 + index),
        "span": datetime.timedelta(days=200, seconds=index),
        "price": Decimal(f"{index}.250"),
        "nonfinite": [float("inf"), float("-inf"), float("nan")],
        "phasor": complex(0.5 + index, -1.25),
        "raw": bytes([index, 255]),
        "scalar": ml_dtypes.bfloat16(1.5 + index),
        "dense": np.asfortranarray(np.arange(6, dtype=np.int16).reshape(2, 3) + index),
        "objects": objects,
        "pair": ("part", index),
        "nested": {"float": 1.25, "ok": True, "empty": None},
        "targets": ValidTargets(role=1, pad_id=index),
    }


def _observed_record(row: dict) -> dict:
    later = row["fall_1"] + row["span"]
    gap = row["fall_1"].astimezone(datetime.UTC) - row["fall_0"].astimezone(datetime.UTC)
    uncached_diff = row["uncached_1"] - row["uncached_0"]
    summer_alarm = datetime.datetime.combine(datetime.date(2026, 7, 1), row["alarm"])
    return {
        "id": row["id"],
        "later_offset": None if later.utcoffset() is None else int(later.utcoffset().total_seconds()),
        "gap_seconds": int(gap.total_seconds()),
        "uncached_gap": int(uncached_diff.total_seconds()),
        "fall_zone": row["fall_1"].tzinfo.key if isinstance(row["fall_1"].tzinfo, ZoneInfo) else None,
        "fall_fold": row["fall_1"].fold,
        "named_tz": row["named_tz"].tzname(),
        "named_fold": row["named_tz"].fold,
        "alarm_fold": row["alarm"].fold,
        "summer_alarm_offset": (
            None if summer_alarm.utcoffset() is None else int(summer_alarm.utcoffset().total_seconds())
        ),
        "day": row["day"].isoformat(),
        "price": row["price"].as_tuple(),
        "nonfinite": [
            math.isinf(row["nonfinite"][0]) and row["nonfinite"][0] > 0,
            math.isinf(row["nonfinite"][1]) and row["nonfinite"][1] < 0,
            math.isnan(row["nonfinite"][2]),
        ],
        "phasor": (row["phasor"].real, row["phasor"].imag),
        "raw": row["raw"],
        "scalar": (row["scalar"].dtype.name, float(row["scalar"])),
        "fortran": bool(row["dense"].flags.f_contiguous),
        "objects": (
            row["objects"].shape,
            [isinstance(entry, np.ndarray) and entry.tolist() for entry in row["objects"]],
        ),
        "pair": (isinstance(row["pair"], tuple), row["pair"]),
        "nested": row["nested"],
        "targets": row["targets"],
    }

def test_wire_variants_and_temporal_rules_round_trip_through_cache_and_export(tmp_path):
    """Every artifact wire variant, including ambiguous DST folds, ZoneInfo rules, named fixed-offset
    timezones and object arrays of equal-length arrays, round-trips without loss through a cache and an
    ArrayRecord export."""
    uncached = [_observed_record(_variant_record(index)) for index in range(2)]
    assert uncached[0]["later_offset"] == -14400 and uncached[0]["gap_seconds"] == 3600
    assert uncached[0]["uncached_gap"] == 3600 and uncached[0]["summer_alarm_offset"] == -14400
    calls = 0

    def build(row: dict) -> dict:
        nonlocal calls
        calls += 1
        return _variant_record(int(row["id"]))

    cached_chain = (
        Memory([{"id": 0}, {"id": 1}])
        .map(build, identity="build@1")
        .cache(str(tmp_path / "cache"))
    )
    loading = Loading(workers=0, threads=1, read_buffer=1)
    for _ in range(2):
        cached_reader = iter(iterate(cached_chain, read="indexed", loading=loading))
        try:
            cached_rows = [_observed_record(row) for row in cached_reader]
        finally:
            cached_reader.close()
        assert cached_rows == uncached
    assert calls == 2
    export_dir = tmp_path / "export"
    export(Memory([_variant_record(0), _variant_record(1)]), str(export_dir), format="array_record")
    exported = Files(str(export_dir)).records("memory")
    assert [_observed_record(exported[index]) for index in range(len(exported))] == uncached


def test_saved_iterator_position_preserves_temporal_fold_and_zone_rules_across_dst():
    """A resumable `from_iter` source whose checkpointed state carries `ZoneInfo` and `fold=1` temporal
    values resumes with the exact calendar behavior across a DST boundary."""
    zone = ZoneInfo("America/New_York")

    class CalendarStream:
        def __init__(self) -> None:
            self.index = 0
            self.origin = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
            self.alarm = datetime.time(1, 30, tzinfo=zone, fold=1)

        def __iter__(self):
            return self

        def __next__(self) -> dict:
            if self.index >= 3:
                raise StopIteration
            when = self.origin if not self.index else self.origin + datetime.timedelta(days=100 * self.index)
            self.index += 1
            offset = when.utcoffset()
            assert offset is not None
            return {
                "step": np.int32(self.index),
                "offset": np.int32(int(offset.total_seconds())),
                "origin_fold": np.int32(self.origin.fold),
                "alarm_fold": np.int32(self.alarm.fold),
            }

    stream = from_iter(
        CalendarStream,
        identity="calendar@1",
        state=lambda cursor: {"index": cursor.index, "origin": cursor.origin, "alarm": cursor.alarm},
        load_state=lambda cursor, saved: (
            setattr(cursor, "index", saved["index"]),
            setattr(cursor, "origin", saved["origin"]),
            setattr(cursor, "alarm", saved["alarm"]),
        ),
    )
    loading = Loading(workers=0, threads=1, read_buffer=1)
    loader = compile(stream.batch(1), read="streamed", steps=3, loading=loading)
    whole = loader.train(DataPartition())
    resumed = loader.train(DataPartition())
    try:
        first = next(whole)
        saved_position = whole.get_state()
        expected = [
            (int(b["step"][0]), int(b["offset"][0]), int(b["origin_fold"][0]), int(b["alarm_fold"][0]))
            for b in (next(whole), next(whole))
        ]
        restore_position(resumed, saved_position)
        actual = [
            (int(b["step"][0]), int(b["offset"][0]), int(b["origin_fold"][0]), int(b["alarm_fold"][0]))
            for b in (next(resumed), next(resumed))
        ]
    finally:
        whole.close()
        resumed.close()
    assert int(first["offset"][0]) == -18000
    assert actual == expected == [(2, -18000, 1, 1), (3, -14400, 1, 1)]


def _replace_iterator_state(tree: object, encoded_state: str) -> int:
    replaced = 0
    if isinstance(tree, dict):
        if "read" in tree and isinstance(tree.get("state"), str):
            tree["state"] = encoded_state
            return 1
        for child in tree.values():
            replaced += _replace_iterator_state(child, encoded_state)
    elif isinstance(tree, list):
        for child in tree:
            replaced += _replace_iterator_state(child, encoded_state)
    return replaced


def test_earlier_iso_temporal_records_still_read_and_custom_timezones_refuse(tmp_path):
    """Artifacts and saved iterator positions written with ISO-only temporal nodes read back with
    `fromisoformat`'s fixed offset (or naive time) and `fold=0`, while custom `tzinfo` subclasses
    without a wire schema raise `Unencodable` rather than losing rules silently."""
    legacy_schema = json.dumps(
        {"mapping": {"when": {"datetime": "2026-11-01T01:30:00-05:00"}, "at": {"time": "01:30:00"}}}
    ).encode()
    legacy_payload = pack_dict_of_byte_arrays({"schema": legacy_schema})
    legacy_root = tmp_path / "legacy-export"
    shard_path = legacy_root / "shards" / "part-000000.array_record"
    shard_path.parent.mkdir(parents=True)
    write_array_record(shard_path, [legacy_payload])
    raw_bytes = shard_path.read_bytes()
    commit_export(
        Store(str(legacy_root)),
        format="array_record",
        codec=1,
        rows=1,
        files=[{
            "path": "shards/part-000000.array_record",
            "rows": 1,
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }],
    )
    expected_legacy = {
        "when": datetime.datetime(2026, 11, 1, 1, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))),
        "at": datetime.time(1, 30),
    }
    restored = Files(str(legacy_root)).records("memory")[0]
    assert restored == expected_legacy
    assert restored["when"].fold == 0 and restored["at"].fold == 0

    loaded: list[dict] = []
    stream = from_iter(
        lambda: iter([{"id": np.int32(0)}, {"id": np.int32(1)}]),
        identity="legacy-iter@1",
        state=lambda _: {"when": datetime.datetime(2026, 11, 1, 1, 30, tzinfo=datetime.UTC)},
        load_state=lambda _, saved: loaded.append(saved),
    )
    loading = Loading(workers=0, threads=1, read_buffer=1)
    loader = compile(stream.batch(1), read="streamed", epochs=1, loading=loading)
    reader = loader.train(DataPartition())
    try:
        next(reader)
        state_doc = json.loads(reader.get_state())
        assert _replace_iterator_state(state_doc, base64.b64encode(legacy_payload).decode()) == 1
        restore_position(reader, json.dumps(state_doc).encode())
        assert int(next(reader)["id"][0]) == 0
    finally:
        reader.close()
    assert loaded == [expected_legacy]
    assert loaded[0]["when"].fold == 0 and loaded[0]["at"].fold == 0

    class CustomZone(datetime.tzinfo):
        def utcoffset(self, dt):
            return datetime.timedelta(hours=-5)

        def dst(self, dt):
            return datetime.timedelta(0)

        def tzname(self, dt):
            return "CUSTOM"

    custom = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=CustomZone())
    assert len(content_fingerprint({"when": custom}, opaque=True)) == 32
    with pytest.raises(Unencodable):
        export([{"when": custom}], str(tmp_path / "refused"), format="array_record")



def test_replay_and_memory_checkpoints_refuse_temporal_fold_and_zone_edits():
    """Checkpoints of `from_iter(..., replay=True)` streams and `Memory` plans include temporal fold
    and timezone metadata in their digests, refusing resumes when a record's fold or zone changes."""
    naive_0 = datetime.datetime(2026, 11, 1, 1, 30, fold=0)
    naive_1 = datetime.datetime(2026, 11, 1, 1, 30, fold=1)
    loading = Loading(workers=0, threads=0, read_buffer=0)

    def rows(when):
        return [{"id": np.int32(0), "when": when}, {"id": np.int32(1), "when": when}]

    for make in (
        lambda w: compile(
            from_iter(lambda: iter(rows(w)), identity="replay-fold@1", replay=True).batch(1),
            read="streamed",
            epochs=1,
            loading=loading,
        ).train(DataPartition()),
        lambda w: compile(Memory(rows(w)).batch(1), read="indexed", epochs=1, loading=loading).train(
            DataPartition()
        ),
    ):
        saving, altered = make(naive_0), make(naive_1)
        try:
            next(saving)
            saved = saving.get_state()
            with pytest.raises(ValueError):
                restore_position(altered, saved)
                next(altered)
        finally:
            saving.close()
            altered.close()


def test_streamed_validation_with_unencodable_host_column_refuses_together_across_processes(tmp_path):
    """When a streamed validation batch carries an unencodable host column, root encodes inside the
    canonical-read agreement so rank 0 raises `Unencodable` and peer ranks raise `PeerFailure` in the
    same phase instead of advancing into the size broadcast."""
    worker = tmp_path / "streamed_val_worker.py"
    worker.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from process_support import prepare_worker, worker_parser\n"
        "def check(args):\n"
        "    prepare_worker(args)\n"
        "    import jax\n"
        "    from dew import Evaluation\n"
        "    from dew._coordination import agreed, agreement_patience\n"
        "    from dew.data.compile import validation\n"
        "    from dew.data.dataset import DataPartition, Loading\n"
        "    from dew.data.sources.iterator import from_iter\n"
        "    from dew.nn.backbones import CausalTransformer\n"
        "    from dew.objectives.lm import LMObjective\n"
        "    from dew.training import MeshSpec\n"
        "    rows = [{'text': np.ones(5, np.int32), 'note': Path('/opaque')}] * 4\n"
        "    chain = (\n"
        "        from_iter(lambda: iter(rows), identity='opaque-val@1')\n"
        "        .read('streamed').workers(0).batch(2)\n"
        "    )\n"
        "    mesh = MeshSpec().build()\n"
        "    partition = DataPartition.of(mesh)\n"
        "    loading = Loading(workers=0, threads=1, read_buffer=1)\n"
        "    model = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2,\n"
        "                              mlp_features=32, max_seq_len=8, attention_impl='reference')\n"
        "    objective = LMObjective(model, seq_len=4, ema_decay=None)\n"
        "    variables = objective.init(jax.random.key(0))\n"
        "    agreed('worker init ready', lambda: None)\n"
        "    agreement_patience(5.0)\n"
        "    reader = validation(chain, batch=2, read='streamed', loading=loading)(partition)\n"
        "    try:\n"
        "        next(reader)\n"
        "        iter_refused = None\n"
        "    except Exception as error:\n"
        "        iter_refused = type(error).__name__\n"
        "    finally:\n"
        "        reader.close()\n"
        "    try:\n"
        "        Evaluation.run(objective, variables, chain, key=0, loss=True, mesh=mesh)\n"
        "        eval_refused = None\n"
        "    except Exception as error:\n"
        "        eval_refused = type(error).__name__\n"
        "    return {'rank': jax.process_index(), 'iter': iter_refused, 'eval': eval_refused}\n"
        "parser = worker_parser({'validate': check}, None)\n"
        "args = parser.parse_args()\n"
        "args.out.write_text(json.dumps(check(args)))\n"
    )
    env = {"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'tests'}"}
    reports = run_pool(
        "validate",
        tmp_path / "pool",
        2,
        devices=1,
        timeout=120,
        start=lambda *a, **kw: spawn(*a, script=worker, environment=env, **kw),
    )
    assert [(r["rank"], r["iter"], r["eval"]) for r in reports] == [
        (0, "Unencodable", "Unencodable"),
        (1, "PeerFailure", "PeerFailure"),
    ]
