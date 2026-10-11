"""The pipeline's observable order, branch scope and exact positions."""

import dataclasses
import itertools
import json

import grain.python as grain
import numpy as np
import pytest

from dew.data.compile import compile
from dew.data.cache import Cache
from dew.data.dataset import DataPartition, Loading, train_stream
from dew.data.mix import SearchBudget, TokenDeficit, token_weights
from dew.data.pipeline import Artifact, Pipeline, Rejected
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources.base import ShardManifest
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@dataclasses.dataclass(frozen=True)
class Memory:
    values: tuple[dict, ...]
    name: str = "records"

    def describe(self):
        return self.name

    def snapshot(self):
        return {"name": self.name}

    def supports(self, read):
        return read != "streamed"

    def records(self, read):
        return self.values

    def manifest(self) -> ShardManifest:
        raise ValueError("no streamed test source")

    def metadata(self):
        return {}

    def lengths(self):
        return None


@dataclasses.dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": np.int64(rng.integers(0, 2**32))}


@dataclasses.dataclass(frozen=True)
class Odd(grain.MapTransform):
    rejects = True

    def map(self, element):
        return element if int(element["id"]) % 2 else Rejected("even")


@dataclasses.dataclass(frozen=True)
class Shift:
    by: int

    def __call__(self, batch):
        return {**batch, "id": batch["id"] + self.by}


@dataclasses.dataclass(frozen=True)
class Targets:
    def __call__(self, row):
        return int(row["targets"])


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


def take(reader, count):
    return [{name: np.asarray(value).tolist() for name, value in row.items()}
            for row in itertools.islice(reader, count)]


def test_pipeline_global_order_matches_the_existing_grain_order():
    stored = source()
    data = compile(Pipeline(stored).shuffle(7).repeat().random_map(Draw()).batch(4), loading=READ, seed=7)
    old = train_stream(stored.values, [Draw()], batch=4, seed=7, loading=READ)
    reader = data.train(DataPartition())
    oracle = old(DataPartition())
    try:
        assert take(reader, 9) == take(oracle, 9)
    finally:
        reader.close()
        oracle.close()


def test_memory_rejections_are_dense_before_global_order():
    stored = source()
    data = compile(Pipeline(stored).map(Odd()).shuffle(3).repeat().batch(4), read="memory", loading=READ, seed=3)
    rows = tuple(value for value in stored.values if int(value["id"]) % 2)
    oracle = train_stream(rows, [], batch=4, seed=3, loading=READ)
    reader = data.train(DataPartition())
    old = oracle(DataPartition())
    try:
        assert data.records == 12
        assert take(reader, 4) == take(old, 4)
        assert "dew_global_position" in json.loads(reader.get_state())
    finally:
        reader.close()
        old.close()


def test_global_position_resumes_across_share_counts_with_rng():
    data = compile(Pipeline(source()).shuffle(7).repeat().random_map(Draw()).batch(4), loading=READ, seed=7)
    reader = data.train(DataPartition())
    take(reader, 3)
    saved = reader.get_state()
    expected = take(reader, 5)
    reader.close()
    shares = []
    for index in range(2):
        reader = data.train(DataPartition(index, 2))
        reader.set_state(saved)
        shares.append(take(reader, 5))
        reader.close()
    joined = [{key: [value for pair in zip(left[key], right[key], strict=True) for value in pair]
               for key in left} for left, right in zip(*shares, strict=True)]
    assert joined == expected


def test_host_steps_are_branch_scoped_and_common_steps_run_on_both():
    prefix = Pipeline(source(8))
    data = compile(prefix.batch(4).map_batch(Shift(10)), validation=prefix.batch(4),
                   host=(Shift(100),), loading=READ)
    train = data.train(DataPartition())
    held = data.val(DataPartition())
    try:
        assert take(train, 1)[0]["id"] == [110, 111, 112, 113]
        assert take(held, 1)[0]["id"] == [100, 101, 102, 103]
    finally:
        train.close()
        held.close()


def test_rejection_iterator_resumes_exactly_and_refuses_topology_change():
    data = compile(Pipeline(source()).shuffle(2).repeat().map(Odd()).batch(3), loading=READ, seed=2)
    reader = data.train(DataPartition())
    take(reader, 3)
    saved = reader.get_state()
    rest = take(reader, 4)
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(saved)
    assert take(resumed, 4) == rest
    resumed.close()
    changed = data.train(DataPartition(0, 3))
    with pytest.raises(ValueError, match="topology"):
        changed.set_state(saved)
    changed.close()


def test_cache_refuses_random_prefix_before_opening_the_artifact():
    pipeline = Pipeline(source()).random_map(Draw()).then(Artifact(object())).batch(4)
    with pytest.raises(ValueError, match="cache prefix.*Draw"):
        compile(pipeline, loading=READ)


def test_map_batch_before_batch_and_random_validation_are_named_refusals():
    with pytest.raises(ValueError, match="map_batch.*batch"):
        compile(Pipeline(source()).map_batch(Shift(1)).batch(4), loading=READ)
    with pytest.raises(ValueError, match="validation.*Draw.*canonical"):
        compile(Pipeline(source()).batch(4),
                validation=Pipeline(source()).shuffle(0).random_map(Draw()).batch(4), loading=READ)


def test_token_deficit_mix_counts_targets_and_restores_delivered_counts():
    corpora = [grain.MapDataset.source([{"id": tag, "targets": targets}]).repeat().to_iter_dataset()
               for tag, targets in [(0, 2), (1, 7)]]
    mix = TokenDeficit(corpora, [1, 3], tokens=Targets(), names=["short", "long"])
    reader = iter(mix)
    delivered = [0, 0]
    for row in itertools.islice(reader, 100):
        delivered[row["id"]] += row["targets"]
        assert abs(delivered[0] - sum(delivered) / 4) <= 7
    saved = reader.get_state()
    rest = take(reader, 20)
    resumed = iter(mix)
    resumed.set_state(saved)
    assert take(resumed, 20) == rest
    reader.close()
    resumed.close()


def test_zero_target_corpus_exhausts_named_rejection_budget_without_starvation():
    parents = [grain.MapDataset.source([{"targets": count}]).repeat().to_iter_dataset()
               for count in (0, 4)]
    reader = iter(TokenDeficit(parents, [1, 1], tokens=Targets(), names=["all-zero", "useful"],
                              budget=SearchBudget(consecutive=5, seconds=None)))
    with pytest.raises(ValueError, match="all-zero.*search budget exhausted.*zero_targets"):
        next(reader)
    counts = reader.get_state()["parents"][0]["counts"]
    assert counts["attempted"] == 5
    assert counts["rejected"] == {"zero_targets": 5}
    reader.close()


def test_authoritative_zero_target_count_is_refused_before_division():
    with pytest.raises(ValueError, match="empty-targets.*zero valid target"):
        token_weights(["empty-targets", "useful"], [1, 1], [3, 2], [0, 10])


def test_dependency_change_preserves_global_resume_only_when_derived_order_matches():
    manifest = Manifest({"name": "test"}, revision="before")
    plan = Plan("global", manifest, {"seed": 0}, {"accepted_ids": fingerprint([1, 3, 5])})
    changed = dataclasses.replace(plan, manifest=dataclasses.replace(manifest, revision="after"))
    assert changed.check(plan.record()) == ("revision",)
    changed = dataclasses.replace(changed, derived={"accepted_ids": fingerprint([1, 3, 7])})
    with pytest.raises(ValueError, match="accepted_ids"):
        changed.check(plan.record())


def test_lambda_identity_requires_explicit_non_exact_resume():
    identity = StepIdentity.of(lambda row: row)
    plan = Plan("global", Manifest({}, (identity,)), {"seed": 0})
    with pytest.raises(ValueError, match="non-exact"):
        plan.check(plan.record())
    assert plan.check(plan.record(), resume="non-exact") == ()


def test_cold_warm_and_resumed_cache_keep_per_epoch_randomness(tmp_path):
    stored = source(8)
    cache = Cache(str(tmp_path / "cache"), shard_bytes=2048, spool_bytes=65536)
    prefix = Pipeline(stored).cache(cache)

    def build(prefix):
        return compile(prefix.shuffle(5).repeat().random_map(Draw()).batch(4),
                       read="memory", seed=5, loading=READ)

    cold = build(prefix)
    reader = cold.train(DataPartition())
    first_epoch = take(reader, 2)
    saved = reader.get_state()
    second_epoch = take(reader, 2)
    reader.close()
    by_id = lambda epoch: {identifier: draw for batch in epoch for identifier, draw in
                          zip(batch["id"], batch["draw"], strict=True)}
    assert by_id(first_epoch) != by_id(second_epoch)
    manifest = Manifest.of(stored.snapshot(), (), fill={"read": "memory", "row_ids": "indexed"})
    generation = cache.publish(manifest, stored)
    resumed = build(prefix).train(DataPartition())
    resumed.set_state(saved)
    assert take(resumed, 2) == second_epoch
    resumed.close()
    warm = build(Pipeline(stored).cache(generation)).train(DataPartition())
    warm_first, warm_second = take(warm, 2), take(warm, 2)
    assert by_id(warm_first) != by_id(warm_second)
    warm.close()


def test_causal_windows_keep_each_boundary_target_and_restore_carry():
    from dew.data.pipeline import Windows
    from dew.data.steps.windows import CausalWindows

    rows = grain.MapDataset.source([{"text": np.arange(4, dtype=np.int32)},
                                    {"text": np.arange(4, 11, dtype=np.int32)}]).to_iter_dataset()
    windows = CausalWindows(rows, Windows(3))
    reader = iter(windows)
    assert next(reader)["text"].tolist() == [0, 1, 2, 3]
    saved = reader.get_state()
    rest = take(reader, 10)
    assert rest == [{"text": [3, 4, 5, 6]}, {"text": [6, 7, 8, 9]}]
    resumed = iter(windows)
    resumed.set_state(saved)
    assert take(resumed, 10) == rest
    reader.close()
    resumed.close()


def test_validation_takes_accepted_rows_before_the_share_cut():
    prefix = Pipeline(source(11)).map(Odd()).take(3).batch(4)
    data = compile(Pipeline(source()).batch(4), validation=prefix, loading=READ)
    for share in range(2):
        held = data.val(DataPartition(share, 2))
        rows = list(held)
        assert len(rows) == 1
        assert rows[0]["id"][rows[0][VALID_ROWS]].tolist() == ([1, 5] if share == 0 else [3])
        held.close()
