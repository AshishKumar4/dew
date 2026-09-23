#!/usr/bin/env python3
"""Compare runs of one reference case against the bound the reference sets.

Every run is measured from the truth, the reference framework's fp32 run
(TF32 off) on the same weights, batches and schedule. The bound in each
window of steps is the reference's own bf16 run's distance from that truth:

    rms_w(run - truth) <= FACTOR * rms_w(reference_bf16 - truth)

the root mean square taken over the window's steps. It is the rule
tests/reference_error.py applies to single forwards, read over a training
trajectory: a run that rounds where the reference rounds, or at as many
points of the same size, lands near a ratio of one; FACTOR 2 admits three
roundings of the reference's size per one of its own, and a dropped term,
a wrong scale or a different optimizer moves the ratio by more. Losses are
compared as differences (nats), gradient norms as relative differences.

Step 0 is reported on its own: both runs hold the same weights and read the
same batch, so its loss and gradient-norm differences are the forward and
backward rounding alone, before any trajectory has parted.

A trajectory whose routing is discrete (an MoE router's top-k) decorrelates
under any rounding: a near-tie flips, that token's gradient changes by
order one, and within tens of steps two runs that differ only in rounding
sit as far apart as a bf16 run sits from the fp32 one. `--rerun` names a
second run of the reference at its own precision; each window's unit is
then the larger of the reference's distance from the truth and the two
reference runs' distance from each other, the reference's run-to-run spread.

    python tools/reference_runs/compare.py --truth torch-fp32.json \\
        --reference torch-autocast.json --run dew=dew-bf16.json --window 16
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from common import kernel_summary, trimmed_kernels

FACTOR = 2.0
CONDITIONS = ("batch", "seq", "schedule_steps", "data_sha256", "lr_peak", "lr_init", "lr_end", "warmup",
              "b1", "b2", "eps", "weight_decay", "clip")


def load(path: str) -> dict:
    """A run's record, its profile summary recomputed from the trimmed
    kernel rows beside it (`<record>/kernels.json.gz`) when they are there,
    so every record is categorised by this checkout's `kernel_category`."""
    record = json.loads(Path(path).read_text())
    record["path"] = path
    rows = Path(path).with_suffix("") / "kernels.json.gz"
    profile = record.get("profile") or {}
    if rows.is_file() and "profiled_steps" in profile:
        record["profile"] = {**profile, **kernel_summary(trimmed_kernels(rows), profile["profiled_steps"])}
    return record


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def conditions(records: dict[str, dict]) -> list[str]:
    """The run conditions that differ between records, and the learning-rate
    sequences if they part."""
    problems = []
    names = list(records)
    base = records[names[0]]
    for key in CONDITIONS:
        values = {name: record["config"].get(key) for name, record in records.items()}
        if len({json.dumps(value) for value in values.values()}) > 1:
            problems.append(f"{key}: {values}")
    for name in names[1:]:
        steps = min(len(base["lr"]), len(records[name]["lr"]))
        gap = np.max(np.abs(np.subtract(base["lr"][:steps], records[name]["lr"][:steps])) /
                     np.maximum(np.abs(base["lr"][:steps]), 1e-30))
        if gap > 1e-6:
            problems.append(f"lr of {name} parts from {names[0]} by {gap:.2e} (relative)")
    return problems


def windows(steps: int, width: int) -> list[tuple[int, int]]:
    return [(start, min(start + width, steps)) for start in range(0, steps, width)]


def divergence(truth: dict, reference: dict, runs: dict[str, dict], key: str, relative: bool,
               width: int, factor: float, rerun: dict | None = None) -> dict:
    """Per-window RMS distance from the truth of the reference and each run,
    and each run's ratio to the bound's unit.

    The unit is the reference's distance from the truth, or, with `rerun`
    (a second run of the reference at the same precision), the larger of
    that and the two reference runs' distance from each other: the
    reference's own run-to-run spread, which a trajectory that decorrelates
    under rounding (a router's top-k flipping) reaches long before its
    distance from the truth settles."""
    records = (truth, reference, *runs.values(), *(() if rerun is None else (rerun,)))
    steps = min(len(record[key]) for record in records)
    base = np.asarray(truth[key][:steps], np.float64)

    def apart(record):
        values = np.asarray(record[key][:steps], np.float64)
        return (values - base) / np.abs(base) if relative else values - base

    reference_gap = apart(reference)
    spread_gap = None if rerun is None else apart(rerun) - reference_gap
    result = {"steps": steps, "windows": [], "overall": {}}
    gaps = {name: apart(record) for name, record in runs.items()}

    def judged(section: slice) -> dict:
        reference_rms = rms(reference_gap[section])
        spread = None if spread_gap is None else rms(spread_gap[section])
        unit = reference_rms if spread is None else max(reference_rms, spread)
        row = {"reference": reference_rms, "spread": spread, "bound": factor * unit}
        for name, gap in gaps.items():
            mine = rms(gap[section])
            row[name] = {"rms": mine, "ratio": mine / unit if unit > 0 else math.inf,
                         "within": bool(mine <= factor * unit)}
        return row

    for start, stop in windows(steps, width):
        result["windows"].append({"steps": [start, stop], **judged(slice(start, stop))})
    result["overall"] = judged(slice(0, steps))
    result["step0"] = {"truth": float(base[0]), "reference": float(reference_gap[0]),
                       **{name: float(gap[0]) for name, gap in gaps.items()}}
    result["last_window_mean"] = {
        "truth": float(np.mean(base[-width:])),
        "reference": float(np.mean(reference_gap[-width:])),
        **{name: float(np.mean(gap[-width:])) for name, gap in gaps.items()}}
    return result


def performance(records: dict[str, dict]) -> dict:
    rows = {}
    for name, record in records.items():
        timed = record.get("throughput") or {}
        profile = record.get("profile") or {}
        rows[name] = {
            "framework": record["framework"], "precision": record["precision"],
            "parallel": record["parallel"], "world": record["world"],
            "rate": timed.get("tokens_per_s"),
            "step_ms": timed.get("step_ms_mean"),
            "mfu": record.get("mfu"),
            "peak_gib": max(record["memory"]["peak_allocated_bytes"]) / 2 ** 30,
            "busy_ms": profile.get("device_busy_ms_per_step"),
            "window_ms": profile.get("device_window_ms_per_step"),
            "busy_percent": profile.get("device_busy_percent"),
            "kernels_per_step": profile.get("kernels_per_step"),
            "categories": profile.get("kernel_ms_by_category", {}),
            "top_kernels": profile.get("top_kernels", [])[:10],
        }
        row = rows[name]
        # The MFU the kernels alone reach: the same FLOPs over the device's
        # busy time rather than the wall clock, so the gap between the two
        # is time the device sat idle between kernels.
        row["busy_mfu"] = (None if None in (row["mfu"], row["step_ms"], row["busy_ms"])
                           else row["mfu"] * row["step_ms"] / row["busy_ms"])
    return rows


def router_probes(truth: dict, reference: dict, runs: dict[str, dict], factor: float,
                  rerun: dict | None = None) -> list[dict]:
    """At every probed step, each run's RMS distance from the truth over the
    routers' per-expert load shares, and over their mean entropies, against
    the unit `divergence` takes: the reference's distance, or the larger of
    it and the reference's run-to-run spread."""
    def by_step(record):
        return {probe["step"]: probe for probe in record["router_probe"]}

    truths, references = by_step(truth), by_step(reference)
    reruns = None if rerun is None else by_step(rerun)
    others = {name: by_step(record) for name, record in runs.items()}
    rows = []
    print(f"\nrouter probe on a fixed batch, rms from the truth; bound = {factor:g} x unit")
    print(f"  {'step':>5} {'quantity':<8} {'unit':>10}" + "".join(f" {name:>18}" for name in runs))
    steps = set(truths) & set(references)
    for probes in (*others.values(), *(() if reruns is None else (reruns,))):
        steps &= set(probes)
    for step in sorted(steps):
        row = {"step": step}
        for quantity in ("load", "entropy"):
            base = np.asarray(truths[step][quantity], np.float64)
            theirs = np.asarray(references[step][quantity], np.float64)
            unit = rms(theirs - base)
            if reruns is not None:
                unit = max(unit, rms(np.asarray(reruns[step][quantity], np.float64) - theirs))
            row[quantity] = {"unit": unit, **{
                name: rms(np.asarray(probes[step][quantity], np.float64) - base)
                for name, probes in others.items()}}
            print(f"  {step:>5} {quantity:<8} {unit:>10.3e}" + "".join(
                f" {row[quantity][name]:>10.3e} {row[quantity][name] / unit if unit else math.inf:>5.2f}"
                f"{'  ' if row[quantity][name] <= factor * unit else ' !'}" for name in runs))
        rows.append(row)
    return rows


def category_gaps(rows: dict[str, dict], left: str, right: str) -> list[tuple[str, float, float]]:
    """Kernel categories by how much more time `left` spends in them than
    `right`, per device per step."""
    names = set(rows[left]["categories"]) | set(rows[right]["categories"])
    gaps = [(name, rows[left]["categories"].get(name, 0.0), rows[right]["categories"].get(name, 0.0))
            for name in names]
    return sorted(gaps, key=lambda item: -abs(item[1] - item[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="the reference framework's fp32 run")
    parser.add_argument("--reference", required=True, help="the reference framework's bf16 run")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                        help="a run held to the bound; repeat for several")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--factor", type=float, default=FACTOR)
    parser.add_argument("--pair", action="append", default=[], metavar="LEFT=RIGHT",
                        help="kernel-category gaps of LEFT against RIGHT; unset is every run "
                             "against the reference")
    parser.add_argument("--rerun", default=None,
                        help="a second run of the reference at its precision, whose distance from "
                             "the first is the reference's run-to-run spread")
    parser.add_argument("--out", default=None, help="write the comparison as JSON")
    args = parser.parse_args()

    truth, reference = load(args.truth), load(args.reference)
    rerun = None if args.rerun is None else load(args.rerun)
    runs = {name: load(path) for name, path in (item.split("=", 1) for item in args.run)}
    everything = {"truth": truth, "reference": reference, **runs}

    problems = conditions(everything)
    print("conditions:", "identical" if not problems else "")
    for problem in problems:
        print("  DIFFERS", problem)

    report = {"conditions": problems, "factor": args.factor, "window": args.window}
    series = [("loss", False, "nats"), ("grad_norm", True, "relative")]
    if all("aux_loss" in record for record in everything.values()):
        series.append(("aux_loss", False, "absolute"))
    for key, relative, unit in series:
        result = divergence(truth, reference, runs, key, relative, args.window, args.factor, rerun)
        report[key] = result
        unit_name = "reference" if rerun is None else "max(reference, spread)"
        print(f"\n{key} ({unit}) measured from the truth, rms per window; bound = "
              f"{args.factor:g} x {unit_name}")
        print(f"  step 0: truth {result['step0']['truth']:.6f}, reference {result['step0']['reference']:+.3e}, "
              + ", ".join(f"{name} {result['step0'][name]:+.3e}" for name in runs))
        spread_head = "" if rerun is None else f" {'spread':>10}"
        header = f"  {'steps':>9} {'reference':>10}{spread_head} {'bound':>10}" + "".join(
            f" {name:>18}" for name in runs)
        print(header)
        for row in result["windows"]:
            cells = "".join(
                f" {row[name]['rms']:>10.3e} {row[name]['ratio']:>5.2f}{'  ' if row[name]['within'] else ' !'}"
                for name in runs)
            spread = "" if row["spread"] is None else f" {row['spread']:>10.3e}"
            print(f"  {row['steps'][0]:>4}-{row['steps'][1]:<4} {row['reference']:>10.3e}{spread} "
                  f"{row['bound']:>10.3e}{cells}")
        overall = result["overall"]
        spread = "" if overall["spread"] is None else f" {overall['spread']:>10.3e}"
        print(f"  {'all':>9} {overall['reference']:>10.3e}{spread} {overall['bound']:>10.3e}" + "".join(
            f" {overall[name]['rms']:>10.3e} {overall[name]['ratio']:>5.2f}"
            f"{'  ' if overall[name]['within'] else ' !'}" for name in runs))
        last = result["last_window_mean"]
        print(f"  last-window mean: truth {last['truth']:.5f}, reference {last['reference']:+.2e}, "
              + ", ".join(f"{name} {last[name]:+.2e}" for name in runs))

    if all("router_probe" in record for record in everything.values()):
        report["router_probe"] = router_probes(truth, reference, runs, args.factor, rerun)

    rows = performance(everything)
    report["performance"] = rows
    print("\nthroughput, from each side's own timed window and profile")
    print(f"  {'run':<12} {'side':<8} {'precision':<10} {'parallel':<9} {'tok/s':>8} {'ms/step':>8} "
          f"{'MFU%':>6} {'busyMFU%':>8} {'peak GiB':>9} {'busy ms':>8} {'busy%':>6} {'kern/step':>9}")
    for name, row in rows.items():
        def cell(value, fmt):
            return format(value, fmt) if value is not None else "-"
        print(f"  {name:<12} {row['framework']:<8} {row['precision']:<10} {row['parallel']:<9} "
              f"{cell(row['rate'], '8.0f')} {cell(row['step_ms'], '8.1f')} "
              f"{cell(None if row['mfu'] is None else 100 * row['mfu'], '6.1f')} "
              f"{cell(None if row['busy_mfu'] is None else 100 * row['busy_mfu'], '8.1f')} "
              f"{row['peak_gib']:>9.2f} {cell(row['busy_ms'], '8.1f')} {cell(row['busy_percent'], '6.1f')} "
              f"{cell(row['kernels_per_step'], '9.0f')}")
    pairs = [tuple(item.split("=", 1)) for item in args.pair] or [(name, "reference") for name in runs]
    for left, right in pairs:
        if not rows[left]["categories"] or not rows[right]["categories"]:
            continue
        print(f"\n  kernel ms per device-step, {left} against {right}, largest gaps first")
        for category, mine, theirs in category_gaps(rows, left, right)[:8]:
            print(f"    {category:<12} {mine:>8.2f} {theirs:>8.2f}  ({mine - theirs:+.2f})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
