"""How tools/armada/ci.py folds main's CI verdict into the durations record's entries a test at a time
(`entries`), beside its fold of each file whole (ci.py's `folded`), and what a red or cut-off row raises
(`raised`). The standard library alone, loaded by ci.py by its path: ci.py hands each function the session
module it reads entries by (`session`) and what it read of the verdict and the checkout."""

from collections.abc import Callable, Mapping


def raised(session, record: Mapping[str, float], held: list[dict], marked: dict) -> dict[str, float]:
    """What a file's red, cut-off or incomplete rows `held` time of its tests, as ci.py's `folded` takes it:
    each test a row timed past its record raises it, and a test a row its task's deadline cut off finished
    (all but its last, which the deadline stopped, timed only as far as it ran) is taken where the record
    lacks it; the run counts as no green one, so nothing it timed lowers an entry or drops one. A test that
    grew past its record and makes its group outrun its deadline is timed so, or its group is cut by the old
    time every run and dies at it again (Main, 2026-11-17: base/kernels' test_attention_mosaic_compile
    groups, recorded at about 100 s, ran 12 to 25 min). A site's entries stay as they are (ci.py's
    `site_entries`). A time a bad red run inflated (a hang) comes down where a green run of its test takes
    well under it (`entries`)."""
    taken: dict[str, float] = {}
    for row_ in held:
        timed = list(row_.get("tests", {}).items())
        stopped = row_.get("kind") == "deadline"
        finished = {node for node, _ in (timed[:-1] if stopped else timed)}
        for node, seconds in timed:
            if session.site_name(node, marked) in marked:
                continue
            grew = node in record and seconds > record[node]
            if grew or (node not in record and stopped and node in finished):
                taken[node] = round(seconds, 3)
    return taken


def evidence(row_: dict) -> bool:
    """Whether `row_` is planning evidence: a run this verdict made (no `cached` row, whose times an earlier
    fold took) of its tests packed as the plan packs them (no row armada ran alone after it failed beside
    other tasks, `aloneAfter`, on a container of its own)."""
    return not row_.get("cached") and row_.get("aloneAfter") is None


def lowered(session, warm: Mapping[str, float], rows: Mapping[str, list[dict]], marks: Mapping[str, dict],
            target: float) -> dict[str, float]:
    """The record's `warm` times, each lowered to the most its green rows took (`evidence`; `rows` by file
    as ci.py's `verdict_rows`, each file's sites `marks`) where it holds the test past that by `session.KEPT`
    and by over an eighth of the `target`: a warm run's time is its own and a cold run's bounds it from above,
    so a time a red, cut-off or contended run once raised comes down test by test, a test that compiles cold
    every run included, where the whole-file fold never finds its file stale (row 10776:
    test_compile_ahead's frontier test held 274.7 s against the 42 s each green run took); a site's entries
    stay as they are (`raised`). ci.py folds each file whole from what this returns, so one verdict folded
    again moves nothing: a lowered time is past no green run's by both bounds."""
    took: dict[str, float] = {}
    for name, held in rows.items():
        for row_ in held:
            if row_["exitCode"] == 0 and evidence(row_):
                for node, seconds in row_.get("tests", {}).items():
                    if node in warm and session.site_name(node, marks[name]) not in marks[name]:
                        took[node] = max(took.get(node, 0.0), seconds)
    return {node: took[node] if node in took and seconds > session.KEPT * took[node]
            and seconds - took[node] > target / 8 else seconds for node, seconds in warm.items()}


def entries(session, record: Mapping, kept: Mapping[str, float], rows: Mapping[str, list[dict]],
            owning: Callable[[str], str]) -> dict:
    """The durations `record`'s entries with main's CI verdict folded in (`session.entry`), from the warm
    times ci.py's folds `kept` of it (`lowered`, then each file whole: a test the warm record drops loses its
    cold time and peak with it, one record, so no test is timed cold or sized that the record no longer
    holds; Main, 2026-11-18, rows 7939 and 6003) and the verdict's `rows` by file (ci.py's `verdict_rows`):

    - its cold time, a run in which it compiled a program the shared cache lacked (a row's `cold`): the most
      its green rows so took (`evidence`), the newest run's, so one verdict folded again moves nothing (it was
      the least ever, which held test_loop's threshold case at 33.8 s against 156 s cold); a red or cut-off
      row's past it raises it, whatever the green rows took;
    - its peak the newest of its rows' (`session.moved_peak`)."""
    cold = {node: found for node, held in record.items()
            if (found := session.warm_cold(held)[1]) is not None}
    peaks = {node: found for node, held in record.items() if (found := session.peak(held)) is not None}
    green_cold: dict[str, float] = {}
    red_cold: dict[str, float] = {}
    for held in rows.values():
        for row_ in held:
            timed = row_.get("tests", {})
            for node in row_.get("cold", ()):
                if node in kept and (seconds := timed.get(node)) is not None and not row_.get("cached"):
                    if row_["exitCode"] != 0:
                        red_cold[node] = max(red_cold.get(node, 0.0), seconds)
                    elif evidence(row_):
                        green_cold[node] = max(green_cold.get(node, 0.0), seconds)
            # Each test's peak, the newest task's that ran it, red or green: a bound that follows its test as
            # it grows, never the least of runs it has outgrown (`task_peak`).
            if (measured := row_.get("peak")) is not None:
                for node in timed:
                    if node in kept:
                        peaks[node] = session.moved_peak(peaks.get(node), int(measured))
                # A task killed before it reported a case (its pytest OOM-killed) still bounds its file's
                # tests: their peaks raised to it, never lowered by a run that timed none of them (review).
                if not timed:
                    file = row_["name"].split(":", 1)[1].split("#", 1)[0]
                    for node in kept:
                        if owning(node) == file:
                            peaks[node] = max(peaks.get(node, 0), int(measured))
    cold.update(green_cold)
    cold.update({node: max(cold.get(node, 0.0), seconds) for node, seconds in red_cold.items()})
    return {node: session.entry(seconds, cold.get(node), peaks.get(node)) for node, seconds in kept.items()}
