"""Probe (2113): derived residuals and gradients beside XLA's temporaries, and the trace's cost on a 4B."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from test_plan import BATCH, LIMIT, free, lm, trained

from dew.nn.backbones.decoder_block import REMAT_POLICIES

LINES = []


def test_probe(monkeypatch, tmp_path):
    free(monkeypatch, LIMIT)
    for remat in (None, "minimal", "save_dot_except_mlp", "full"):
        for split in (1, 2):
            monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / f"{remat}-{split}"))
            trainer = lm(split=split, remat=None if remat is None else REMAT_POLICIES[remat])
            trained(trainer, steps=0)
            for probe in trainer.plan["probes"]:
                derived = probe["derived"]
                LINES.append(f"remat {remat} split {split}: temp {probe['temp']} argument {probe['argument']} "
                             f"derived {json.dumps(derived)} sum {derived['residuals'] + derived['gradients']}")
    # The trace's cost on a 4B: the pullback of a Dory-1.0-4B loss at 2 x 4096, abstract.
    from dew.interop.hf_decoders import translate_config
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training.footprint import residual_bytes

    config = json.load(open("tests/fixtures/hf/dory-1.0-4b/config.json"))
    config = config.get("text_config", config)
    for remat in (None, "full"):
        model = models.build("causal_transformer", {**translate_config(config), "max_seq_len": 4096,
                                                    "remat": None if remat is None else REMAT_POLICIES[remat]})
        objective = LMObjective(model, seq_len=4096)
        variables = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
        batch = {"text": jax.ShapeDtypeStruct((2, 4097), jnp.int32)}
        from dew.objectives.base import Step

        def loss(params, variables, batch):
            step = Step(jnp.zeros((), jnp.int32), jax.random.key(1), None)
            return objective._loss({**variables, "params": params}, batch, step)

        started = time.perf_counter()
        total = residual_bytes(loss, variables["params"], (variables, batch), 2, 1)
        LINES.append(f"dory-4b remat {remat}: residuals {total / 2**30:.2f} GiB traced in "
                     f"{time.perf_counter() - started:.1f} s")
    raise SystemExit("PROBE\n" + "\n".join(LINES))
