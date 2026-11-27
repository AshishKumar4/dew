"""A fit search reads its first miss's live set (`dew.training.live_buffers.peak_live`) to choose what it asks
next (row 12463): the remat rung whose dropped values cover what the step was short of asked first. A 4B
decision step on 2 x H100 climbed one rung at a time. Compiled for compile-only H100s on the CPU, never
run; each search read by the probes its plan or refusal records."""

import jax
import numpy as np
import pytest

from dew.lora import LoRA
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.kernels.generation import chosen, compile_only_gpus
from dew.objectives.lm import LMObjective
from dew.optim import Adam, OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.training.ahead import AheadOfTime, compile_ahead
from dew.training.memory import least_limit
from dew.training.planning import PlanRefused


def _ahead(model, optimizer, seq: int, rows: int, limit: int, variables=None
           ) -> tuple[AheadOfTime | None, list[dict]]:
    """`model`'s step compiled ahead for one compile-only H100 of `limit` bytes, all its pool's, over `rows`
    rows of `seq` tokens, its `variables` where given, None where it fit none, and each probe its search
    compiled, in order."""
    mesh = MeshSpec().build(compile_only_gpus("sm90", 1, limit, share=1.0))
    built = OptimConfig(optimizer=optimizer, lr=1e-3).build(100)
    objective = LMObjective(model, seq) if variables is None else LMObjective(model, seq, variables=variables)
    trainer = Trainer(objective, built, key=jax.random.key(0), mesh=mesh,
                      advanced=TrainerAdvanced(layout=Layout(min_shard=256)))
    try:
        ahead = compile_ahead(trainer, {"text": np.zeros((rows, seq + 1), np.int32)})
    except PlanRefused as refused:
        return None, [probe.record() for probe in refused.probes]
    return ahead, list(ahead.plan["probes"])


def _needed(ahead: AheadOfTime) -> int:
    """The least memory an H100 whose pool holds all of it (a share of 1, as `_ahead`'s) states for
    `ahead`'s step to be admitted, by the terms its search admitted it by (its fitting probe's, `admitted`:
    its resident arguments, less the copies its first call builds, which it places beside them, and its
    temporaries; `least_limit`): its pool's limit, there its memory, as `compile_only_gpus` takes it."""
    gpu = compile_only_gpus("sm90", 1, 2 ** 34, share=1.0)[0]
    probe = next(probe for probe in reversed(ahead.plan["probes"]) if probe["fits"])
    admitted = probe["admitted"]
    assert isinstance(admitted, list) and all(isinstance(term, int) for term in admitted), probe
    resident, beside, temp = admitted
    return least_limit(gpu, [(resident, beside, temp)])


@pytest.mark.parametrize("stack", ["plain", "scanned", "rows"])
def test_a_search_asks_first_the_rung_its_miss_reads_as_dropping_what_it_was_short_of(stack):
    """A step of one row, which no split shrinks, whose MLPs' forward values hold most of its temporaries,
    that misses its devices at no remat on an H100 between what 'save_dot_except_mlp' needs and what
    'minimal' needs, its stack a layer at a time, asks 'save_dot_except_mlp' next, the rung its miss reads
    as dropping what it was short of (`RecomputeStatement`), passing over 'minimal', which drops too little,
    and is admitted there: two compiles, where the climb a rung at a time compiled 'minimal' between them.
    Its stack scanned, or scanned reading a frozen base's rows where they lie under a LoRA adapter
    (`decoder_stack._rows_recomputed`), the residuals at its peak are the scan's stacked outputs, which no
    block's scope names (`dew.nn.protocols.RUNG_SCOPE`): the miss reads no rung, and the search climbs the
    ladder in its order to the rung that fits, asking each once."""
    model = models.build("causal_transformer", vocab_size=256, emb_features=256, num_layers=4, num_heads=4,
                         num_kv_heads=2, head_dim=64, mlp_features=8192, max_seq_len=1024, dtype="bfloat16")

    def ahead_at(remat: str | None, limit: int) -> tuple[AheadOfTime | None, list[dict]]:
        at = model if remat is None else model.clone(remat=REMAT_POLICIES[remat])
        if stack != "rows":
            return _ahead(at, Adam(), 1024, 1, limit)
        base = at.init(jax.random.key(0), np.zeros((1, 1024), np.int32))
        adapter = LoRA(rank=2, modules=("q_proj",)).apply(at, base, key=1)
        return _ahead(adapter.model, Adam(), 1024, 1, limit, adapter.variables)

    # Its stack scanned or a layer at a time, its attention XLA's own, whatever the selector would take.
    with chosen(layer_scan="plain" if stack == "plain" else "scanned", attention="xla"):
        rung = {name: ahead_at(name, 2 ** 34)[0] for name in ("minimal", "save_dot_except_mlp")}
        assert all(ahead is not None for ahead in rung.values()), rung
        lighter, dropping = (_needed(ahead) for ahead in rung.values() if ahead is not None)
        # Its devices between the two rungs' needs: 'minimal' misses there, 'save_dot_except_mlp' fits.
        assert dropping < lighter, (dropping, lighter)
        ahead, probes = ahead_at(None, (dropping + lighter) // 2)
    assert ahead is not None and ahead.plan["remat"] == "save_dot_except_mlp", probes
    asked = [probe["remat"] for probe in probes]
    assert asked == ([None, "save_dot_except_mlp"] if stack == "plain"
                     else [None, "minimal", "save_dot_except_mlp"]), probes
    assert [probe["fits"] for probe in probes] == [False] * (len(asked) - 1) + [True], probes


def test_a_place_past_the_copies_is_asked_after_the_rest_where_what_the_search_measured_predicts_it_short():
    """Past the narrow copies' climb, a place is predicted short by its remat's shortfall with the copies
    less what each change it makes freed where the search measured it (`predicted_short`): the 4B GH200
    start's figures (row 12463, UB-4B-COLDPLAN, GiB), whose search asked 41 places, predict every place it
    missed at short and its fitting one, save_dot_except_mlp with its head tiled, as fitting; a place whose
    change no rung measured yet is asked as the ladder orders it."""
    import flax.linen as nn

    from dew.training.planning import Rung, predicted_short

    remats = {name: (nn.Dense(width),) for width, name in enumerate(
        ("none", "minimal_with_scan_and_conv", "minimal", "save_dot_except_mlp", "full"), start=1)}
    tile = (4096, 8192)

    def at(remat: str, *, narrow: bool = True, tiled: bool = False) -> Rung:
        return Rung(remats[remat], tile if tiled else None, fused=False, xla_defaults=False, narrow=narrow)

    start = at("none")

    measured = [(at("none"), 162), (at("minimal_with_scan_and_conv"), 62), (at("minimal"), 44),
                (at("save_dot_except_mlp"), 8), (at("full"), 2),
                (at("none", narrow=False), 158), (at("none", narrow=False, tiled=True), 132)]
    assert predicted_short(at("minimal", narrow=False, tiled=True), measured, start) > 0
    assert predicted_short(at("minimal_with_scan_and_conv", narrow=False), measured, start) > 0
    assert predicted_short(at("save_dot_except_mlp", narrow=False), measured, start) > 0
    assert predicted_short(at("save_dot_except_mlp", narrow=False, tiled=True), measured, start) <= 0
    # A change no rung was measured both ways is no prediction: the place is asked in its turn.
    assert predicted_short(at("minimal", narrow=False, tiled=True), measured[:6], start) is None


def test_every_process_defers_a_place_any_predicts_short_and_asks_it_after_the_rest():
    """Past the narrow copies every process defers a place where any predicts it short (`FitSearch.gather`),
    so each asks the same place at once, whatever its own room predicts; a deferred place is asked once the
    rest miss (`FitSearch.past_copies`): here this process predicts nothing, another process predicts the
    first place short, so both ask the second first and the first after it, which fits."""
    import flax.linen as nn

    from dew.training.planning import FitSearch, Rung

    model = models.build("causal_transformer", vocab_size=32, emb_features=16, num_layers=1, num_heads=2,
                         mlp_features=32, max_seq_len=8)
    first = Rung((nn.Dense(1),), None, fused=False, xla_defaults=False, narrow=False)
    second = Rung((nn.Dense(2),), None, fused=False, xla_defaults=False, narrow=False)
    others = iter([True, False, False])
    search = FitSearch(lower=None, options=lambda candidate: None, objective=LMObjective(model, seq_len=8),
                       mesh=MeshSpec().build(jax.devices()[:1]), xla_defaults=False, narrow=False,
                       ns_group=None, base=None, resumed=None, gather=lambda mine: [mine, next(others)])
    named: list[str] = []

    def ask(rung: Rung, said: str) -> int | None:
        search.asked.append(rung)
        named.append(str(len(named)))
        return 2 if rung == first else None

    assert search.past_copies([first, second], second, named, ask, lambda rung: None) == 2
    assert search.asked == [second, first]


def scripted_search(gather):
    """A fit search over a tiny decoder on one device, its rounds `gather`'s, its rung the plain one."""
    from dew.training.planning import FitSearch, Rung

    model = models.build("causal_transformer", vocab_size=32, emb_features=16, num_layers=1, num_heads=2,
                         mlp_features=32, max_seq_len=8)
    search = FitSearch(lower=None, options=lambda candidate: None, objective=LMObjective(model, seq_len=8),
                       mesh=MeshSpec().build(jax.devices()[:1]), xla_defaults=False, narrow=False,
                       ns_group=None, base=None, resumed=None, gather=gather)
    return search, Rung((), None, fused=False, xla_defaults=False, narrow=False)


def test_a_jumped_rung_fitting_at_a_split_its_walk_passed_over_is_admitted_there():
    """A search whose start misses at every split, whose jump lands on the ladder's first place at split
    2, and whose place misses at 2 and, its forecast reading no split fitting, at 8: the place fits at 4,
    a split its walk passed over, and the search admits it there rather than refusing (train 1387's P0)."""
    from dew.training.memory import MemoryMiss
    from dew.training.planning import FitSearch

    model = models.build("causal_transformer", vocab_size=32, emb_features=16, num_layers=1, num_heads=2,
                         mlp_features=32, max_seq_len=8)

    class Scripted(FitSearch):
        place: str | None = None

        def fits_here(self, split: int) -> bool:
            rung = self.rung()
            self.lowered[(split, rung)] = None
            if rung == self.place and split == 4:
                return True
            self.misses[(rung, split)] = MemoryMiss(split, scaled=0, shortfall=10, factor=1)
            return False

        def jumped(self, start, narrow, ns_group, candidates, begun):
            places = [rung for rungs in self.ladder(start, narrow, ns_group, remat_set=self.remat_set,
                                                     defaults=self.own_defaults()) for rung in rungs]
            self.at(places[0])
            self.place = self.rung()
            self.at(start)
            return places[0], list(candidates).index(2)

    search = Scripted(lower=None, options=lambda candidate: None, objective=LMObjective(model, seq_len=8),
                      mesh=MeshSpec().build(jax.devices()[:1]), xla_defaults=False, narrow=False,
                      ns_group=None, base=None, resumed=None, gather=lambda mine: [mine])
    assert search.searched([1, 2, 4, 8], 0) == 4
    assert search.rung() == search.place


def test_a_miss_reads_its_next_split_and_whether_it_fits_as_every_process_does():
    """After a miss, the split asked next is the latest any process computes, and it is read as fitting
    only where every process computes it so (`forecast`): another process computing none fitting, the
    next split is its last-resort one and no process reads it as fitting (train 1387's P1)."""
    from dew.training.memory import MemoryMiss

    search, rung = scripted_search(lambda mine: [mine, [3, False]])
    search.at(rung)
    search.misses[(search.rung(), 1)] = MemoryMiss(1, scaled=100, shortfall=40, factor=1)
    assert search.forecast([1, 2, 4, 8], 0) == (3, False)

