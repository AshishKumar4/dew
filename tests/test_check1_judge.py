"""Check 1 sharded across armada's map tasks and judged by their pool (tests/map_judge.py,
tests/check1_judge.py): a judge judges a site as the site judges itself whole, runs no truth, and judges a
test from its map tasks' records alone."""

import json

import jax
import jax.numpy as jnp
import pytest
from gate_cases import a_product_draw, cancelling, cancelling_draw, found, sharded_site
from reference_error import assert_as_exact

from dew.parity import GateFailure


def test_a_sharded_sites_map_tasks_size_its_cancelling_rows_and_its_judge_runs_no_truth(
        tmp_path, monkeypatch):
    """A site whose truth holds a row of float64 zero (`cancelling`) sized by its terms by default
    (`_cancelling_terms`) is so in each map task (DRAWS), which runs the truth on its first draw to find
    them, and its records carry each row's size; its judge (POOL) reads them, running no truth at all, and
    judges as the site does whole: on par admitted, a leaked term refused alike."""
    truths = []

    def counted(x, w):
        # Each run of the truth, compiled or not, on the host (a trace alone counts nothing).
        jax.debug.callback(lambda: truths.append(1))
        return cancelling(split=False)(x, w)

    def site(dew, label):
        def run():
            truths.clear()
            assert_as_exact(dew, cancelling(split=False), cancelling_draw, label, seeds=16,
                            truth_fn=counted)
            return len(truths)
        return run

    drawn, judged = sharded_site(monkeypatch, tmp_path / "on par", [(0, 40), (40, 80)],
                                 site(cancelling(split=True), "on par"))
    assert all(count >= 1 for count in drawn), "each map task sizes its rows from the truth"
    assert judged == 0, "a judge runs no truth"
    assert_as_exact(cancelling(split=True), cancelling(split=False), cancelling_draw, "whole", seeds=16,
                    truth_fn=counted)
    with pytest.raises(GateFailure) as whole:
        assert_as_exact(cancelling(split=True, leak=True), cancelling(split=False), cancelling_draw,
                        "leaked", seeds=16, truth_fn=counted)
    with pytest.raises(GateFailure) as pooled:
        sharded_site(monkeypatch, tmp_path / "leaked", [(0, 40), (40, 80)],
                     site(cancelling(split=True, leak=True), "leaked"))
    # The judge, the last to run, ran no truth before it refused.
    assert truths == [], "a judge runs no truth"
    assert sorted(found(pooled.value)) == sorted(found(whole.value))


def test_a_site_sharded_across_map_tasks_is_judged_by_their_pooled_draws_as_whole(tmp_path, monkeypatch):
    """armada's map and judge (`DRAWS`, `POOL`): a site whose seeds are sharded across map tasks, each writing
    its draws and judging nothing, is judged over their pooled draws as it is when it draws them itself: a
    product on par passes, and a bfloat16 one fails with the same findings; a pool that stops short of a
    stage's seeds, or holds a gap, is refused, never judged."""
    import map_judge
    from map_judge import DRAWS, POOL, RECORDS, PoolRefused

    def judged(dew, label, shards):
        sharded_site(monkeypatch, tmp_path / label, shards,
                     lambda: assert_as_exact(dew, jnp.matmul, a_product_draw, label, seeds=16))

    def halved(x, w):
        return (x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)).astype(jnp.float32)

    judged(jnp.matmul, "on par", [(0, 30), (30, 80)])
    with pytest.raises(GateFailure) as whole:
        assert_as_exact(halved, jnp.matmul, a_product_draw, "halved", seeds=16)
    with pytest.raises(GateFailure) as pooled:
        judged(halved, "halved", [(0, 40), (40, 80), (40, 80)])
    assert sorted(found(pooled.value)) == sorted(found(whole.value))
    with pytest.raises(PoolRefused) as short:
        judged(halved, "halved, short", [(0, 16)])
    with pytest.raises(PoolRefused) as gapped:
        judged(halved, "halved, with a gap", [(0, 8), (16, 80)])
    assert (short.value.held, gapped.value.held) == (16, 8)
    # A judge that asks for its later seeds (LATER): over its first stage's seeds alone it passes where that
    # stage passes, and where it finds something asks for the second stage's, writing its ask; over every
    # seed it judges as a site that drew them all.
    from map_judge import LATER, LaterUnits
    asked = tmp_path / "later.json"
    monkeypatch.setenv(LATER, str(asked))
    judged(jnp.matmul, "on par, its first stage", [(0, 16)])
    assert not asked.exists()
    with pytest.raises(LaterUnits):
        judged(halved, "halved, its first stage", [(0, 16)])
    assert (json.loads(asked.read_text())["from"], json.loads(asked.read_text())["to"]) == (16, 80)
    with pytest.raises(GateFailure) as both:
        sharded_site(monkeypatch, tmp_path / "both stages", [(0, 16), (16, 56), (56, 80)],
                     lambda: assert_as_exact(halved, jnp.matmul, a_product_draw, "halved", seeds=16))
    assert sorted(found(both.value)) == sorted(found(whole.value))
    monkeypatch.delenv(LATER)
    # Two tests' sites of one label, their records under one pool: each judge pools its own test's alone.
    def as_test(test, dew, *, records=None, pool=None):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/test_{test}.py::test_{test} (call)")
        for name, value in ((DRAWS, records and "0:80"), (RECORDS, records), (POOL, pool)):
            if value:
                monkeypatch.setenv(name, value)
            else:
                monkeypatch.delenv(name, raising=False)
        assert_as_exact(dew, jnp.matmul, a_product_draw, "one label", seeds=16)

    as_test("halved", halved, records=str(tmp_path / "shared" / "halved"))
    as_test("on_par", jnp.matmul, records=str(tmp_path / "shared" / "on par"))
    as_test("on_par", jnp.matmul, pool=str(tmp_path / "shared"))
    with pytest.raises(GateFailure):
        as_test("halved", halved, pool=str(tmp_path / "shared"))
    # One site, its id, label and ranges alike, run on two Pythons into one pool: each Python's judge pools
    # its own Python's draws alone.
    for python, dew in (("3.12", jnp.matmul), ("3.14", halved)):
        monkeypatch.setattr(map_judge, "python", lambda python=python: python)
        as_test("one_site", dew, records=str(tmp_path / "pythons" / python))
    monkeypatch.setattr(map_judge, "python", lambda: "3.12")
    as_test("one_site", jnp.matmul, pool=str(tmp_path / "pythons"))
    monkeypatch.setattr(map_judge, "python", lambda: "3.14")
    with pytest.raises(GateFailure):
        as_test("one_site", halved, pool=str(tmp_path / "pythons"))


def test_a_judge_judges_a_tests_sites_from_its_map_tasks_records_alone(tmp_path, monkeypatch):
    """A sharded test's judge (POOL) judges the sites its map tasks' manifests name from their records,
    none of its body run (`judged_from_records`): a product on par passes and a bfloat16 one fails with the
    findings the body's own judge finds; a test whose body expects its site's failure runs its body; a pool
    missing one map task's draws, short of the first stage's seeds, or drawn at another commit is refused
    naming the record, never judged."""
    import map_judge
    from check1_judge import judged_from_records
    from check1_wire import DRAWS, POOL, RECORDS
    from map_judge import PoolRefused

    test = "tests/test_judged.py::test_judged"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{test} (call)")

    def halved(x, w):
        return (x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)).astype(jnp.float32)

    def body(dew):
        assert_as_exact(dew, jnp.matmul, a_product_draw, "a product", seeds=16)

    def mapped(root, dew, shards, *, expecting: bool = False):
        """The test's body as each map task runs it, then its manifest written, as conftest writes it; a body
        `expecting` its site's failure says it decides the verdict itself (`judged_in_its_body`)."""
        for index, (start, stop) in enumerate(shards):
            monkeypatch.setenv(DRAWS, f"{start}:{stop}")
            monkeypatch.setenv(RECORDS, str(root / str(index)))
            if expecting:
                map_judge.judged_in_its_body("a product")
            body(dew)
            map_judge.manifested(test)
        monkeypatch.delenv(DRAWS)
        monkeypatch.delenv(RECORDS)

    def judged(root, judge):
        monkeypatch.setenv(POOL, str(root))
        try:
            return judge()
        finally:
            monkeypatch.delenv(POOL)

    mapped(tmp_path / "on par", jnp.matmul, [(0, 40), (40, 80)])
    assert judged(tmp_path / "on par", lambda: judged_from_records(test))
    mapped(tmp_path / "halved", halved, [(0, 40), (40, 80)])
    with pytest.raises(GateFailure) as from_records:
        judged(tmp_path / "halved", lambda: judged_from_records(test))
    with pytest.raises(GateFailure) as from_body:
        judged(tmp_path / "halved", lambda: body(halved))
    assert sorted(found(from_records.value)) == sorted(found(from_body.value))
    # A test no map task wrote a manifest of runs its body, and so does one whose body decides its site's
    # verdict itself, a failure it expects.
    assert not judged(tmp_path / "halved", lambda: judged_from_records("tests/test_judged.py::test_other"))
    mapped(tmp_path / "expected", halved, [(0, 40), (40, 80)], expecting=True)
    assert not judged(tmp_path / "expected", lambda: judged_from_records(test))
    # One map task's draws missing, or the first stage's seeds short: refused, naming what the pool holds.
    for path in (tmp_path / "halved" / "1").glob("*.pkl"):
        if ".manifest." not in path.name:
            path.unlink()
    with pytest.raises(PoolRefused) as short:
        judged(tmp_path / "halved", lambda: judged_from_records(test))
    assert short.value.held == 40
    mapped(tmp_path / "few", jnp.matmul, [(0, 8)])
    with pytest.raises(PoolRefused) as few:
        judged(tmp_path / "few", lambda: judged_from_records(test))
    assert few.value.held == 8
    # A whole pool, drawn at another commit than its judge's: refused.
    monkeypatch.setattr(map_judge, "commit", lambda: "another")
    with pytest.raises(PoolRefused):
        judged(tmp_path / "on par", lambda: judged_from_records(test))
