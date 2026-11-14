"""Check 1's verdict on a site's draws, whichever process draws them (tests/reference_error.py's
`assert_as_exact`): each reference's staged judgement (`_verdicts`) by what the site's verdict reads beside
its draws (`Verdict`), logged a stage a judgement (STAGES_LOG), and a sharded test's judge's verdicts over its
map tasks' pooled records (`_judged_pooled`), from their manifests alone where the judge runs none of the
test's body (`judged_from_records`, tests/map_judge.py)."""

import dataclasses
import functools
import json
import os
from collections.abc import Callable

import numpy as np
from check1_wire import DRAWS, STAGES
from map_judge import PoolRefused, ask_later, manifests, pooled_units

from dew.parity import NARROW_FACTORS, GateFailure, Inconclusive, _Draws, _staged, _verdict

SWITCHED: list[dict] = []
"""The sites decided onto YNNPACK's fusions (`_ynn_decided`) since the last test's report took them: each its
label and XLA:CPU's refusal, which tests/conftest.py reports as the test's JUnit property `ynn_switched`."""


STAGES_LOG = STAGES
"""The environment variable (`check1_wire.STAGES`) naming a file each check-1 judgement appends its stage, its
verdict, its leaves of narrow rows (NARROW_FACTORS) and whether its reference equalled the truth on every
first-stage seed (`exact`: a site that measures no rounding, or a truth that is its reference run again) to,
one JSON line (`_logged`): what a train reads every site's stage-two rate from, every attempt counted, and
which sites' local errors rest on check 2."""


def _logged(label: str, stage: int, verdict: str, narrow_leaves: list[str], ynn: str | None = None,
            exact: bool | None = None) -> None:
    path = os.environ.get(STAGES_LOG)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
                                  "label": label, "stage": stage, "verdict": verdict,
                                  "narrow_leaves": narrow_leaves, "ynn": ynn, "exact": exact}) + "\n")


def _with_mode(findings: list, refused: str | None) -> list:
    """`findings`, each naming its site's arms' YNNPACK mode where they were decided onto the fusions
    (`_ynn_decided`, `_mode_of`: XLA:CPU's refusal), so a failure says which code its arms ran."""
    if refused is None:
        return findings
    return [dataclasses.replace(finding, message=f"{finding.message} (both arms with YNNPACK's fusions: "
                                                 f"XLA:CPU refused one without, {refused!r})")
            for finding in findings]


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What a check-1 site's verdict reads beside its draws (`_verdicts`): its declared `seeds`, its
    `labels` (the site's own, then one against each further reference, `also`, in its draws' order),
    whether a `library` arm stands in for the reference on some leaves, and the site's weighed check
    (`_deferred`) it `defers_to` where it stays undecided. A map task records it with its draws (`_recorded`),
    and the judge reads that record (`judged_from_records`): one object, whichever process judges."""

    seeds: int
    labels: tuple[str, ...]
    library: bool
    defers_to: str | None = None


def _narrow(draws: _Draws) -> list[str]:
    """The leaves of rows of one or two values (NARROW_FACTORS)."""
    return [name for name, width in zip(draws.names, draws.widths, strict=True) if width in NARROW_FACTORS]


def _verdicts(drawn_for, verdict: Verdict, narrow_leaves: list[str], mode: Callable[[], str | None]) -> None:
    """Each reference's check 1 over its draws (`_staged`, `drawn_for(index)` the draws against reference
    `index`, judged under `verdict.labels[index]` over its declared seeds), each logged (STAGES_LOG) with
    whether its reference equalled the truth on every first-stage seed, unknown (None) for the first where a
    `library` arm's error stands in for its reference's on some leaves, and the arms' YNNPACK mode (`mode`,
    read once they have run); a GateFailure where any leaf is over, else an Inconclusive where any is left
    unresolved."""
    overs, unresolveds, seeds = [], [], verdict.seeds
    for index, named in enumerate(verdict.labels):
        over, unresolved, stage = _staged(drawn_for(index), seeds, named)
        exact = None if verdict.library and not index else not np.any(drawn_for(index)(0, seeds).theirs)
        _logged(named, stage, _verdict(over, unresolved), narrow_leaves, mode(), exact)
        overs += over
        unresolveds += unresolved
    if overs:
        raise GateFailure(_with_mode(overs + unresolveds, mode()))
    if unresolveds:
        raise Inconclusive(_with_mode(unresolveds, mode()))


def _pooled_draws(label: str, references: int):
    """A judge's draws (`pooled_units`) and the YNNPACK mode its map tasks ran the arms in, which must be
    one: `drawn_for(index)` over its map tasks' records of this site, `drawn(start, stop)` the pooled draws
    of those seeds, refused past the seeds the records hold from seed 0 on."""
    pieces = pooled_units(label)
    held = pieces[-1][1]
    modes = {payload["refused"] for _, _, payload in pieces}
    if len(modes) > 1:
        raise PoolRefused(f"{label}: the map tasks ran the arms in different modes: "
                          f"{sorted(map(str, modes))}", held)
    refused = modes.pop()
    if len({payload.get("heavy") for _, _, payload in pieces}) > 1:
        raise PoolRefused(f"{label}: the map tasks weighed by different classes (`Conditioning.digest`)",
                          held)
    if refused is not None:
        SWITCHED.append({"label": label, "refused": refused})
    pooled = [functools.reduce(_Draws.joined, (payload["draws"][index] for _, _, payload in pieces))
              for index in range(references)]

    def drawn_for(index: int):
        def drawn(start, stop):
            if stop > held and start == held:
                # The second stage's seeds, drawn by later map tasks a judge asks for (`ask_later`).
                ask_later(label, held, stop)
            if stop > held:
                raise PoolRefused(f"{label}: its stage draws seeds {start} to {stop}; the map tasks drew 0 "
                                  f"to {held} ({DRAWS} covers 0 to (1 + REDRAW) times the declared seeds)",
                                  held)
            return pooled[index].between(start, stop)
        return drawn
    return drawn_for, refused


def _judged_pooled(label: str, verdict: Verdict) -> None:
    """A judge's verdict on site `label` (POOL): its map tasks' pooled draws (`_pooled_draws`) judged by
    `verdict`, the leaves of narrow rows read off its first draw."""
    drawn_for, refused = _pooled_draws(label, len(verdict.labels))
    _verdicts(drawn_for, verdict, _narrow(drawn_for(0)(0, 1)), lambda: refused)


def judged_from_records(test: str) -> bool:
    """A judge's verdicts on `test` from its map tasks' records, none of its body run: each site their
    manifests name (`map_judge.manifests`), in the order the body reached them, judged by the verdict a map
    task recorded (`_judged_pooled`), a site that stays undecided judging the weighed check it defers to
    (`_deferred`), the first failure the test's. A missing record, a stage's seeds short, or another commit's
    is refused, naming it (`map_judge.pooled_units`). False, the body to run, where no map task wrote a
    manifest of `test`, a site it reached judges its own units (`map_judge.sharded`), or its body decides a
    site's verdict itself (`map_judge.judged_in_its_body`: a failure it expects)."""
    sites = manifests(test)
    if sites is None or any(verdict is None for _, verdict in sites):
        return False
    verdicts = dict(sites)
    deferred = {verdict.defers_to for verdict in verdicts.values()}
    for label, verdict in sites:
        if label in deferred:
            continue
        try:
            _judged_pooled(label, verdict)
        except Inconclusive:
            if verdict.defers_to is None:
                raise
            _judged_pooled(verdict.defers_to, verdicts[verdict.defers_to])
    return True
