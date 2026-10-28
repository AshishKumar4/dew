"""Probe (2113): the run's own plan on compile-only GPUs with a stated capacity, derived beside XLA's."""
import json

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.training import MeshSpec

LINES = []


def test_probe(monkeypatch, tmp_path):
    from dew.training.ahead import compile_ahead, compile_only_gpus

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    for generation, devices, width in (("sm90", 1, None), ("sm80", 1, 256), ("sm90", 2, 256)):
        for capacity in (2**30, 2**22, 2**20):
            trainer = fine_tune(MeshSpec().build(compile_only_gpus(generation, devices)), width=width)
            try:
                ahead = compile_ahead(trainer, batch, capacity)
            except Exception as error:
                LINES.append(f"{generation}x{devices} w{width} cap {capacity}: {type(error).__name__}: {str(error)[:300]}")
                continue
            plan = ahead.plan
            LINES.append(f"{generation}x{devices} w{width} cap {capacity}: {ahead.planned_seconds:.1f} s, chosen split "
                         f"{plan['split']} remat {plan['remat']}")
            for probe in plan["probes"]:
                d = probe["derived"]
                LINES.append(f"   split {probe['split']} remat {probe['remat']} fits {probe['fits']} headroom "
                             f"{probe['headroom']} temp {probe['temp']} arg {probe['argument']} derived {json.dumps(d)}")
    raise SystemExit("PROBE\n" + "\n".join(LINES))
