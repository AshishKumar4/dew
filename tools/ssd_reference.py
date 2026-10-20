"""Record native SSD draws and confirm each site's gate seeds against its recorded reference."""

import datetime
import json

import jax
import numpy as np
from reference_error import REDRAW, seeds_for


def provenance(reference, revision, device, command, **details):
    """The reference, revision, recording date, device and reproducible public command."""
    return np.asarray(json.dumps({"reference": reference, "revision": revision, "device": device,
                                  "date": datetime.date.today().isoformat(), "command": command, **details}))


def record_case(label, draw, evaluate, proxy, *, truth_proxy=None):
    """Plan K on XLA, record the gate's redraw, and confirm K on the actual reference.

    The proxy only plans capacity. Reference sizing draws more recorded seeds on demand, then the
    recorder fills that confirmed K's entire redraw. Only the confirmed count is declared.
    """
    planned = seeds_for(jax.jit(proxy), lambda seed: (draw(seed),), label,
                        truth_fn=None if truth_proxy is None else jax.jit(truth_proxy))
    print(f"{label}: XLA capacity plan {planned} seeds", flush=True)
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
    print(f"{label}: declared {confirmed} seeds; recorded {len(records)} draws (REDRAW={REDRAW})", flush=True)
    return result
