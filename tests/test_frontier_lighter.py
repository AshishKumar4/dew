"""A plan made ahead with its frontier lists, below it, the lighter rungs its remat-first search passed over
that fit (`FitSearch.below`, the plan's `lighter`; row 10817), on compile-only GPUs (`compile_ahead`)."""

import pytest
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune, gpus, need_ahead
from token_support import token_rows

from dew._records import listed, record
from dew.training import MeshSpec


@pytest.mark.shared_setup("sm90_full_tiled_needs")
def test_a_frontier_lists_each_lighter_rung_the_search_passed_over_that_fits(tmp_path, monkeypatch):
    """A step given room for itself recomputing all ('full') and for its head tiled recomputing nothing, but
    not whole recomputing nothing (each measured, `need_ahead`): its search settles a remat rung at its
    whole head in its fewest compiles, as without its frontier, and the frontier lists, below that plan,
    each lighter rung the search passed over that fits, the tiled head at no recomputation among them,
    the run left at its plan; and a plan taken from the
    record the first left lists the same (row 10817: the cheaper rung was compiled by no search and told no
    run)."""
    from dew.training.ahead import compile_ahead
    from dew.training.memory import recompute_record

    room = max(need_ahead("sm90", split=1, remat="full"), need_ahead("sm90", split=1, remat=None, tiled=True))
    assert need_ahead("sm90", split=1, remat=None) > room
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults", "narrow")
    alone = record(compile_ahead(fine_tune(MeshSpec().build(gpus("sm90", limit=room)), split=1),
                                 token_rows(VOCAB, BATCH, SEQ_LEN + 1)).plan, "plan")
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "again"))
    trainer = fine_tune(MeshSpec().build(gpus("sm90", limit=room)), split=1)
    plan = record(compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True).plan, "plan")
    assert {name: plan[name] for name in rung} == {name: alone[name] for name in rung}
    assert plan["remat"] is not None and plan["head_tile"] is None, plan["probes"]
    assert recompute_record(trainer.objective) == plan["remat"] and trainer.objective.head_tile is None
    lighter = listed(plan["lighter"], "lighter", read=record, what="lighter")
    assert all(entry["fits"] is True for entry in lighter), lighter
    assert any(entry["remat"] is None and entry["head_tile"] is not None for entry in lighter), lighter
    # A plan taken from the record the first plan left lists the same lighter rungs, its search asking none.
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    warm = record(compile_ahead(fine_tune(MeshSpec().build(gpus("sm90", limit=room)), split=1),
                                token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True).plan, "plan")
    # Its first compile the recorded rung, which fits.
    first = record(listed(warm["probes"], "probes", read=record, what="probes")[0], "probe")
    assert first["fits"] is True
    assert {name: first[name] for name in rung} == {name: plan[name] for name in rung}
    warm_lighter = listed(warm["lighter"], "lighter", read=record, what="lighter")
    assert ([{name: entry[name] for name in rung} for entry in warm_lighter]
            == [{name: entry[name] for name in rung} for entry in lighter])
