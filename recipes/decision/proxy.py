"""Decision Index proxy sampling without a dependency on the kit or a model.

The spec supplies area and benchmark weights. Requests stay in their kit
groups, including all tracks and variants of an item. A request budget is
a ceiling: when whole groups cannot fill it exactly, report the actual
count. Every benchmark and track is represented, and BANKING77 and CLINC150
keep at least one group for every observed gold class, including OOS.
"""

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping


def area_weights(spec: dict) -> dict[str, float]:
    """The spec's fixed shares, with the rest proportional to the square root of benchmark count."""
    sizing = spec["area_sizing"]
    if sizing["rule"] != "sqrt":
        raise ValueError(f"unknown area sizing rule {sizing['rule']!r}")
    fixed = sizing["fixed"]
    sizes = {area["id"]: math.sqrt(len(area["benchmarks"])) for area in spec["areas"]
             if area["id"] not in fixed}
    remaining = 1.0 - sum(fixed.values())
    return {area["id"]: (fixed[area["id"]] if area["id"] in fixed else
                         remaining * sizes[area["id"]] / sum(sizes.values())) for area in spec["areas"]}


def _quotas[K](total: int, weights: Mapping[K, float], floors: Mapping[K, int],
               capacities: Mapping[K, int]) -> dict[K, int]:
    """Integer weighted shares bounded by mandatory requests and available requests."""
    if total < sum(floors.values()):
        raise ValueError(f"{total} requests cannot cover the {sum(floors.values())} mandatory requests")
    total = min(total, sum(capacities.values()))
    low, high = 0.0, max(total / weight for weight in weights.values())
    for _ in range(80):
        scale = (low + high) / 2
        shares = {key: min(capacities[key], max(floors[key], scale * weight))
                  for key, weight in weights.items()}
        if sum(shares.values()) < total:
            low = scale
        else:
            high = scale
    quotas = {key: int(value) for key, value in shares.items()}
    order = sorted(weights, key=lambda key: (-(shares[key] - quotas[key]), str(key)))
    for key in order:
        if sum(quotas.values()) == total:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
    return quotas


def groups(rows: Iterable[dict]) -> dict[int, list[list[dict]]]:
    """Rows grouped by the kit's `(catalog_id, group_id)`, in a stable order independent of file order."""
    found = defaultdict(list)
    run_ids = set()
    for row in rows:
        evaluation = row["_evaluation"]
        run_id = evaluation["run_id"]
        if run_id in run_ids:
            raise ValueError(f"duplicate request id {run_id!r}")
        run_ids.add(run_id)
        found[(evaluation["catalog_id"], evaluation["group_id"])].append(row)
    by_benchmark = defaultdict(list)
    for (number, _), members in sorted(found.items()):
        by_benchmark[number].append(sorted(members, key=lambda row: row["_evaluation"]["run_id"]))
    return dict(by_benchmark)


def _mandatory(ranked: list[list[dict]], *, classes: bool) -> set[int]:
    """One whole group per track, and for the intent benchmarks per observed gold class."""
    selected, tracks, golds = set(), set(), set()
    for index, members in enumerate(ranked):
        member_tracks = {row["_evaluation"]["track"] for row in members}
        member_golds = {(key, str(gold)) for row in members for key, gold in row["expected"].items()}
        if member_tracks - tracks or (classes and member_golds - golds):
            selected.add(index)
            tracks.update(member_tracks)
            golds.update(member_golds)
    return selected


def _draw(ranked: list[list[dict]], mandatory: set[int], quota: int) -> set[int]:
    """Fill a quota by subset sums of whole groups; seeded priority breaks ties."""
    remaining = quota - sum(len(ranked[index]) for index in mandatory)
    reachable: dict[int, tuple[int, int]] = {0: (0, -1)}
    per_size = defaultdict(int)
    for index, members in enumerate(ranked):
        size = len(members)
        if index in mandatory or size > remaining or per_size[size] >= remaining // size:
            continue
        per_size[size] += 1
        for used in sorted(reachable, reverse=True):
            after = used + size
            if after <= remaining and after not in reachable:
                reachable[after] = (used, index)
        if remaining in reachable:
            break
    selected = set(mandatory)
    used = max(reachable)
    while used:
        used, index = reachable[used]
        selected.add(index)
    return selected


def allocate(spec: dict, rows: Iterable[dict], *, requests: int = 2500, seed: int = 0) -> list[dict]:
    """A deterministic weighted subset of at most `requests`, with whole groups and intent-class floors.

    Apportion across areas first, then their benchmarks, using the spec's
    gold weights. Floors and finite source capacities constrain the shares;
    an impossible floor budget raises rather than dropping a class. Fill
    integer group-size gaps where possible, without ever splitting a group.
    """
    if requests < 1:
        raise ValueError("requests must be positive")
    numbers = {number for area in spec["areas"] for number in area["benchmarks"]}
    by_benchmark = groups(row for row in rows if row["_evaluation"]["catalog_id"] in numbers)
    if missing := numbers - by_benchmark.keys():
        raise ValueError(f"suite is missing index benchmarks {sorted(missing)}")
    ranked, mandatory, floors, capacities = {}, {}, {}, {}
    for number in sorted(numbers):
        ranked[number] = sorted(by_benchmark[number], key=lambda members: hashlib.sha256(
            f"{seed}:{number}:{members[0]['_evaluation']['group_id']}".encode()).digest())
        name = spec["chance"][str(number)]["name"]
        mandatory[number] = _mandatory(ranked[number], classes=name in ("BANKING77", "CLINC150"))
        floors[number] = sum(len(ranked[number][index]) for index in mandatory[number])
        capacities[number] = sum(map(len, ranked[number]))
    area_floors = {area["id"]: sum(floors[n] for n in area["benchmarks"]) for area in spec["areas"]}
    area_capacities = {area["id"]: sum(capacities[n] for n in area["benchmarks"]) for area in spec["areas"]}
    areas = _quotas(requests, area_weights(spec), area_floors, area_capacities)
    quotas = {}
    for area in spec["areas"]:
        weights = {n: float(spec.get("gold", {}).get(str(n), 1.0)) for n in area["benchmarks"]}
        quotas.update(_quotas(areas[area["id"]], weights, {n: floors[n] for n in weights},
                              {n: capacities[n] for n in weights}))
    selected = {n: _draw(ranked[n], mandatory[n], quotas[n]) for n in sorted(numbers)}
    counts = {n: sum(len(ranked[n][i]) for i in selected[n]) for n in selected}
    remaining = min(requests, sum(capacities.values())) - sum(counts.values())
    while remaining:
        candidates = {n: next((i for i, members in enumerate(ranked[n])
                               if i not in selected[n] and len(members) <= remaining), None)
                      for n in selected}
        available = [n for n in selected if candidates[n] is not None]
        if not available:
            break
        number = max(available, key=lambda n: (quotas[n] - counts[n], -n))
        index = candidates[number]
        selected[number].add(index)
        size = len(ranked[number][index])
        counts[number] += size
        remaining -= size
    return [row for n in sorted(selected) for i in sorted(selected[n]) for row in ranked[n][i]]


def resample(rows: list[dict], results: dict, rng: random.Random) -> tuple[list[dict], dict]:
    """A group bootstrap within each benchmark, giving repeated draws distinct group and request ids."""
    sampled, responses = [], {}
    for number, members in groups(rows).items():
        for draw in range(len(members)):
            for row in rng.choice(members):
                evaluation = row["_evaluation"]
                run_id = f"bootstrap:{number}:{draw}:{evaluation['run_id']}"
                sampled.append({**row, "_evaluation": {**evaluation, "run_id": run_id,
                                                       "group_id": f"bootstrap:{number}:{draw}"}})
                if evaluation["run_id"] in results:
                    responses[run_id] = {**results[evaluation["run_id"]], "run_id": run_id}
    return sampled, responses


def interval(rows: list[dict], results: dict, score: Callable[[list[dict], dict], float], *,
             samples: int = 1000, seed: int = 0) -> list[float]:
    """The central 80% percentile interval from resampling whole groups within each benchmark."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    values = sorted(score(*resample(rows, results, rng)) for _ in range(samples))

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return [percentile(0.1), percentile(0.9)]
