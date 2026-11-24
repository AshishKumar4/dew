"""A plan's release of a Muon step's joined Newton-Schulz stacks (`ns_group` 1) as its search orders it
against each rung's miss (`dew.training.planning.FitSearch.searched`, `Parting`, row 12463)."""

import json
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_plan import free_until_placed, needs, one_device

from dew import newton_schulz
from dew._records import JSON
from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.distributed.placement import MeshSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.objectives.lm import LMObjective
from dew.optim import Muon, OptimConfig, ParamGroup
from dew.training import Trainer
from dew.training.memory import MemoryMiss, recompute_more, recompute_record
from dew.training.planning import FitSearch, Probe

MUON_ROW = 512
"""The tokens a row of `muon_long`'s holds."""


def muon_long(remat: str | None = None, **opts) -> Trainer:
    """A four-layer LM trained by Muon over one microbatch a step of eight rows of MUON_ROW tokens: narrow
    enough that its joined Newton-Schulz stacks hold far less than the activations its forward keeps for
    its backward (its attention's scores among them), which recomputing frees; recomputing as `remat` names
    where given."""
    recomputing = {} if remat is None else {"remat": REMAT_POLICIES[remat]}
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=4, num_heads=2, mlp_features=64,
                              max_seq_len=MUON_ROW, **recomputing)
    config = OptimConfig(optimizer=Muon(**opts), lr=1e-2)
    return one_device(Trainer(LMObjective(model, seq_len=MUON_ROW), config, key=jax.random.key(0), split=1))


def fitted_long(trainer: Trainer) -> None:
    """`trainer` fit one step of eight rows of MUON_ROW tokens, its plan made as a fit makes it."""
    text = (np.arange(8 * (MUON_ROW + 1)).reshape(8, MUON_ROW + 1) * 7 % 64).astype(np.int32)
    trainer.fit(Pipeline(Memory({"text": text})).repeat().batch(8), steps=1)


def test_a_muon_step_short_of_more_than_parting_frees_recomputes_without_parting_first(monkeypatch):
    """A Muon step whose start falls short of the devices' room by far more than parting its joined
    Newton-Schulz stacks frees, and which fits at the first remat rung of its climb that recomputes its
    layers' insides ('minimal', the rungs below it its attention's same program), its split fixed: its plan
    settles there with its stacks joined, asking
    no step of them parted, where it parted them first, a compile no parting could fit (owner, 2026-11-23,
    row 12463: a 4B cold start's climb parted its stacks before the remat rung its miss needed, three
    compiles where two were its contract)."""
    with pytest.MonkeyPatch.context() as measuring:
        whole = needs(measuring, trainer := muon_long(), lambda: fitted_long(trainer))
        arguments = trainer.plan["probes"][0]["argument"]
        parted = needs(measuring, trainer := muon_long(ns_group=1), lambda: fitted_long(trainer))
        recomputing = needs(measuring, trainer := muon_long("minimal"), lambda: fitted_long(trainer))
        settled = trainer.plan["remat"]
    assert whole - parted < (parted - recomputing) // 2, (whole, parted, recomputing)
    free_until_placed(monkeypatch, (parted + recomputing) // 2 + arguments)
    planned = muon_long()
    fitted_long(planned)
    groups = [probe["ns_group"] for probe in planned.plan["probes"]]
    plan = planned.plan
    assert plan["ns_group"] is None and plan["remat"] == settled, (plan["ns_group"], plan["remat"], settled)
    assert 1 not in groups, groups


def test_a_runs_param_groups_count_every_muon_transforms_joined_stacks():
    """The joined stacks a step's update holds by count are every Muon transform's together: a run whose
    param groups split four matrices of one shape between two Muon transforms counts what the same four
    joined under one transform count (`newton_schulz.joined_held`), not one group's (review of row 12463:
    a step of several param groups read the largest group's alone)."""
    params = {name: {"w1": jnp.zeros((64, 64)), "w2": jnp.zeros((64, 64))} for name in ("a", "b")}
    grouped = OptimConfig(optimizer=Muon(), lr=1e-2, param_groups=(
        ParamGroup("a", patterns=("a/*",)), ParamGroup("b", patterns=("b/*",))))

    def held(config: OptimConfig) -> int:
        solver = config.build(4)
        state = solver.init(params)
        return newton_schulz.joined_held(lambda: jax.eval_shape(solver.update, params, state, params))[1]

    assert held(grouped) == held(OptimConfig(optimizer=Muon(), lr=1e-2)) > 0


class Scripted(FitSearch):
    """A search over a small LM's ladder whose compiles are a table (`need`: the bytes a rung needs, by its
    place among the remat rungs a climb visits and whether it parts its stacks), judged against `room`, a
    miss short by the difference; its stacks holding `count` bytes by count. Every rung it compiled, in
    order (`compiled`), as `(place, ns_group)`."""

    def __init__(self, need: Callable[[int, int | None], int], room: int, count: int):
        model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                                  max_seq_len=16)
        climber = LMObjective(model, seq_len=16)
        self.places = [json.dumps(recompute_record(climber), sort_keys=True)]
        while recompute_more(climber, None):
            self.places.append(json.dumps(recompute_record(climber), sort_keys=True))
        super().__init__(lower=None, options=lambda candidate: None, objective=LMObjective(model, seq_len=16),
                         mesh=MeshSpec().build(jax.devices()[:1]), xla_defaults=False, narrow=False,
                         ns_group=None, base=None, resumed=None)
        self.need, self.room, self.count = need, room, count
        self.compiled: list[tuple[int, int | None]] = []

    def stacks_held(self) -> int:
        return self.count

    def place(self) -> tuple[int, int | None]:
        rung: JSON = json.loads(self.rung())
        assert isinstance(rung, dict)
        return self.places.index(json.dumps(rung["remat"], sort_keys=True)), self.ns_group

    def fits_here(self, split: int) -> bool:
        place = self.place()
        self.compiled.append(place)
        short = self.need(*place) - self.room
        self.probes.append(Probe(split, 1, self.current(), short <= 0, -short, 0, 0))
        if short > 0:
            self.misses[(self.rung(), split)] = MemoryMiss(split, 0, short * self.factor, self.factor)
        return short <= 0


def test_each_rung_is_asked_joined_before_its_stacks_part():
    """Parting is each rung's own: a start whose parted step missed leaves the next remat rung asked with
    its stacks joined, where it fits, the parted program's padding and lost batching not taken; and a
    start whose parting freed nothing leaves the next rung, whose miss is within the stacks' count, parted
    at once (review of row 12463: the parted group, once taken, held for every rung after, and once
    dropped, for none)."""
    joined_fits = Scripted(lambda place, group: {(0, None): 100, (0, 1): 95}.get((place, group), 90),
                           room=90, count=20)
    assert joined_fits.searched([1], 0) == 1
    asked = joined_fits.compiled
    assert asked[:2] == [(0, None), (0, 1)] and len(asked) == 3 and asked[2][0] > 0 and asked[2][1] is None, (
        asked)
    parted_fits = Scripted(lambda place, group: 100 if place == 0 else 95 if group is None else 90, room=90,
                           count=20)
    assert parted_fits.searched([1], 0) == 1
    asked = parted_fits.compiled
    assert [group for _, group in asked] == [None, 1, None, 1] and asked[2][0] == asked[3][0] > 0, asked


def test_a_parted_step_past_the_stacks_count_is_asked_before_the_search_gives_up():
    """Every rung asked joined missing by more than the stacks hold by count, the search asks each of
    their parted steps in the climb's order before it gives up, and settles at the first that fits: the
    count orders the asks and rules none out, as XLA's buffers can hold more than it counts (review of
    row 12463: a spectral preconditioning's SVD wrote eighty times the count)."""
    search = Scripted(lambda place, group: 40 if group == 1 and place > 0 else 100, room=50, count=5)
    assert search.searched([1], 0) == 1
    asked = search.compiled
    joined = [at for at, (_, group) in enumerate(asked) if group is None]
    parted = [at for at, (_, group) in enumerate(asked) if group == 1]
    assert joined and parted and max(joined) < min(parted), asked
    assert asked[parted[0]] == (0, 1) and asked[-1][1] == 1 and search.place() == asked[-1], asked


def test_a_frontier_below_a_plan_settled_joined_reads_the_parted_steps_beneath_it():
    """A search that settles joined on a higher rung, its start's parted step deferred (its miss past the
    stacks' count), leaves that parted step, which fits, to the frontier a plan made offline reads beneath
    it (`FitSearch.below`, `dew.training.ahead.compile_ahead(frontier=True)`'s lighter plans): the run's
    speed is chosen among them (review of row 12463: the frontier read the joined rungs alone)."""
    search = Scripted(lambda place, group: {(0, None): 100, (0, 1): 90}.get((place, group), 90), room=90,
                      count=5)
    origin = search.rung_now()
    assert search.searched([1], 0) == 1
    settled, place = search.rung_now(), search.place()
    assert search.compiled == [(0, None), place] and place[0] > 0, search.compiled
    lighter = search.below([1], origin, settled)
    assert [(probe.rung["ns_group"], probe.fits) for probe in lighter] == [(1, True)], search.compiled


def test_a_process_whose_trace_noted_no_stacks_reads_the_frontier_its_peers_read():
    """A process whose step's trace JAX's cache answered notes no joined stacks (`joined_held`'s 0), where a
    peer's trace noted them: the frontier it reads beneath a plan settled joined is its peer's, the parted
    steps among it, so every process compiles the same programs in the same order and none waits alone in
    a compile's collectives (review of row 12463: the frontier branched on a process's own count)."""
    search = Scripted(lambda place, group: {(0, None): 100, (0, 1): 90}.get((place, group), 90), room=90,
                      count=0)
    # Its peer answers each gathered verdict as it does, but for the stacks' count: 5 where it read 0.
    search.gather = lambda value: [value, 5 if type(value) is int and value == 0 else value]
    origin = search.rung_now()
    assert search.searched([1], 0) == 1
    settled = search.rung_now()
    lighter = search.below([1], origin, settled)
    assert [(probe.rung["ns_group"], probe.fits) for probe in lighter] == [(1, True)], search.compiled
