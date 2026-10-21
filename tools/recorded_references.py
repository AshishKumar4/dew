"""Recording a reference over seeds, for check 1's `indexed_recording` (tests/reference_error.py).

A reference that cannot run in the test (a CUDA-only library, a fetched source) is recorded once, over
the seeds check 1 draws: `record_case` plans the count on a proxy that can run (Dew's own XLA path),
records (1 + REDRAW) times it, confirms the count on the recorded reference itself with `seeds_for`,
records the confirmed count's whole redraw and returns the `<site>.input.*`, `<site>.reference.*`,
`<site>.truth.*` and `<site>.seeds` arrays to save beside its `provenance`. The proxy only plans: the
declared count is the recorded reference's own. (tools/recorded_draws.py is another thing: it records
the random draws a reference generator's library makes, so a second precision reads the same samples.)
"""

import datetime
import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_error import REDRAW, seeds_for


def provenance(reference: str, revision: str, device: str, command: str, **details) -> np.ndarray:
    """A recording's reference, its revision, the date, the device and the public command that
    reproduces it, as a JSON string array to save beside the recorded arrays."""
    return np.asarray(json.dumps({"reference": reference, "revision": revision, "device": device,
                                  "date": datetime.date.today().isoformat(), "command": command, **details}))


def record_case(label: str, draw, evaluate, proxy, *, truth_proxy=None) -> dict[str, np.ndarray]:
    """`label`'s recording: `draw(seed)` gives a seed's inputs (a dict of arrays), `evaluate(inputs)` the
    reference's outputs and their float64 truth (dicts), and `proxy(inputs)` (with `truth_proxy`) a
    function that can run here, which plans the count. The recorded reference then confirms it,
    recording further seeds as `seeds_for` asks, and the confirmed count's whole redraw is recorded."""
    planned = seeds_for(jax.jit(proxy), lambda seed: (draw(seed),), label,
                        truth_fn=None if truth_proxy is None else jax.jit(truth_proxy))
    print(f"{label}: planned {planned} seeds on the proxy", flush=True)
    records = []

    def ensure(count):
        for seed in range(len(records), count):
            inputs = draw(seed)
            reference, truth = evaluate(inputs)
            records.append((inputs, jax.tree.map(np.asarray, reference), jax.tree.map(np.asarray, truth)))

    def recorded_draw(seed):
        ensure(seed + 1)
        return np.int32(seed), records[seed][0]

    ensure((1 + REDRAW) * planned)
    confirmed = seeds_for(lambda index, inputs: records[int(index)][1], recorded_draw, label,
                          truth_fn=lambda index, inputs: records[int(index)][2])
    ensure((1 + REDRAW) * confirmed)
    result = {}
    for index, kind in enumerate(("input", "reference", "truth")):
        for name in records[0][index]:
            result[f"{label}.{kind}.{name}"] = np.stack([record[index][name] for record in records])
    result.update({f"{label}.seeds": np.int32(confirmed), f"{label}.redraw": np.int32(REDRAW)})
    print(f"{label}: declared {confirmed} seeds; recorded {len(records)} draws", flush=True)
    return result
