"""A plan made ahead with its frontier lists, below it, the lighter rungs that fit which no search asked
(`FitSearch.below`, the plan's `lighter`; row 10817), on compile-only GPUs (`compile_ahead`)."""

import pytest
from model_support import attending
from test_compile_ahead import SEQ_LEN, TILED_BATCH, TILED_VOCAB, fine_tune, gpus, need_ahead
from token_support import token_rows

from dew._records import listed, record
from dew.training import MeshSpec

RUNG = ("split", "remat", "head_tile", "head_fused", "xla_defaults", "narrow")
"""The fields that name a plan's rung."""


def room() -> int:
    """Room for the step recomputing all ('full') and for its head tiled recomputing nothing, but not whole
    recomputing nothing (each measured, `need_ahead`)."""
    held = max(need_ahead("sm90", split=1, remat="full"), need_ahead("sm90", split=1, remat=None, tiled=True))
    assert need_ahead("sm90", split=1, remat=None) > held
    return held


def planned(limit: int, frontier: bool = False, remat: str | None = None):
    """The fine-tuned step's trainer on a compile-only H100 holding `limit`, and the plan it compiles
    ahead."""
    from dew.training.ahead import compile_ahead

    mesh = MeshSpec().build(gpus("sm90", limit=limit))
    trainer = fine_tune(mesh, split=1, remat=remat, vocab=TILED_VOCAB)
    plan = compile_ahead(trainer, token_rows(TILED_VOCAB, TILED_BATCH, SEQ_LEN + 1), frontier=frontier).plan
    return trainer, record(plan, "plan")


def probed(plan) -> list[dict]:
    """The plan's compiles, in order."""
    return [record(probe, "probe") for probe in listed(plan["probes"], "probes", read=record, what="probes")]


def held_at(trainer, plan) -> bool:
    """Whether the run is left at the plan's rung: its remat and its head's tile."""
    from dew.training.memory import recompute_record

    tile = trainer.objective.head_tile
    return (recompute_record(trainer.objective) == plan["remat"]
            and (None if tile is None else list(tile)) == plan["head_tile"])


@attending("xla")
@pytest.mark.shared_setup("sm90_full_tiled_needs")
def test_a_frontier_keeps_the_plan_its_search_admits_in_two_compiles(tmp_path, monkeypatch):
    """The step started whole recomputing nothing in that room: its second compile fits, as without its
    frontier, the run left at the plan, every rung its frontier lists fitting; and a plan taken from the
    record the first left compiles that rung first, which fits, and lists the same."""
    limit = room()
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "alone"))
    _, alone = planned(limit)
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer, plan = planned(limit, frontier=True)
    probes = probed(plan)
    assert [probe["fits"] for probe in probes] == [False, True], plan["probes"]
    # The second compile the jump's, not a fallback climb's: the jump names its rung and split.
    jump = record(plan["jump"], "jump")
    assert jump["jumped"] is True and jump["split"] == probes[1]["split"], jump
    named = {name: value for name, value in record(jump["rung"], "rung").items() if name in RUNG}
    assert "remat" in named and named == {name: probes[1][name] for name in named}, (jump, probes[1])
    assert {name: plan[name] for name in RUNG} == {name: alone[name] for name in RUNG}
    assert held_at(trainer, plan)
    lighter = listed(plan["lighter"], "lighter", read=record, what="lighter")
    assert all(entry["fits"] is True for entry in lighter), lighter
    _, warm = planned(limit, frontier=True)
    first = probed(warm)[0]
    assert first["fits"] is True
    assert {name: first[name] for name in RUNG} == {name: plan[name] for name in RUNG}
    assert ([{name: entry[name] for name in RUNG} for entry in listed(warm["lighter"], "lighter", read=record,
                                                                         what="lighter")]
            == [{name: entry[name] for name in RUNG} for entry in lighter])


@attending("xla")
@pytest.mark.shared_setup("sm90_full_tiled_needs")
def test_a_frontier_lists_the_lighter_rung_a_recorded_heavier_plan_passed_over(tmp_path, monkeypatch):
    """The step in that room whose record names the heavier rung, all recomputed at its whole head (as a run
    stating 'full' records it, kept as this run's: a record an earlier run left before its tile fitted):
    its plan is that rung, its one compile, which fits, the run left there; and its frontier lists below it
    the lighter rungs that fit which no search asked, the tiled head recomputing nothing among them, which
    `need_ahead` measures fitting (row 10817: the cheaper rung was compiled by no search and told no run)."""
    from dew import _cache
    from dew.training.rungs import keep_rung

    limit = room()
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "full"))
    planned(limit, remat="full")
    (stated,) = (tmp_path / "full").glob("*.json")
    heavier = record(_cache.read_kept(stated), "rung record")["rung"]
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    planned(limit)
    (own,) = (tmp_path / "rungs").glob("*.json")
    keep_rung(own, record(_cache.read_kept(own), "rung record")["key"], heavier)
    trainer, plan = planned(limit, frontier=True)
    probes = probed(plan)
    assert len(probes) == 1 and probes[0]["fits"] is True, probes
    assert held_at(trainer, plan) and plan["remat"] == record(heavier, "rung")["remat"]
    lighter = listed(plan["lighter"], "lighter", read=record, what="lighter")
    assert all(entry["fits"] is True for entry in lighter), lighter
    assert any(entry["remat"] is None and entry["head_tile"] is not None for entry in lighter), lighter
