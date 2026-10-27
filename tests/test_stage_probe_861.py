"""Diagnostic probe for row 861 (not for landing): a pipelined decoder step's per-device bytes,
temporaries, FLOPs and stage-axis collectives on a forced CPU mesh, through public APIs alone."""

import json

import jax
import optax
import pytest
from model_support import TINY_DECODER
from sharded import collective_axes
from token_support import token_rows

from dew.distributed.batches import shard_batch
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.telemetry.instrumentation import collective, hlo_instructions, step_flops
from dew.training import Layout, MeshSpec, Trainer

pytestmark = pytest.mark.mesh

SEQ, ROWS, LAYERS = 16, 16, 8


def held(tree, device):
    return sum(shard.data.nbytes for leaf in jax.tree.leaves(tree) for shard in leaf.addressable_shards
               if shard.device == device)


def probe(stages, microbatches, remat):
    fields = {**TINY_DECODER, "emb_features": 64, "mlp_features": 256, "num_layers": LAYERS,
              "num_kv_heads": 2, "remat": remat}
    model = models.build("causal_transformer", vocab_size=64, **fields)
    mesh = MeshSpec(stage=stages)
    trainer = Trainer(LMObjective(model, SEQ, ema_decay=0.9), optax.adam(1e-3), key=jax.random.key(0),
                      mesh=mesh, layout=Layout(min_shard=256), checkpoints=None, tracker=None,
                      microbatches=microbatches if stages > 1 else None)
    trainer.device_mesh = mesh.build(jax.devices()[:stages])
    raw = token_rows(64, ROWS, SEQ + 1)
    report = trainer.validate(raw)
    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, raw)
    trainer.compile(state, batch)
    compiled = trainer.executable
    stats = compiled.memory_analysis()
    text = compiled.as_text()
    moved = {}
    trips = 0
    for instruction in hlo_instructions(text).values():
        kind = collective(instruction.op)
        if kind is None or instruction.op.endswith("-start"):
            continue
        static, weighted = moved.get(kind, (0, 0.0))
        size = sum(instruction.buffers)
        moved[kind] = (static + size, weighted + size * instruction.runs)
        if kind == "collective-permute":
            trips = max(trips, instruction.runs)
    axes = sorted(f"{kind}:{'+'.join(sorted(names))}"
                  for kind, names in collective_axes(text, trainer.device_mesh))
    flops = step_flops(compiled)
    device = trainer.device_mesh.devices.flat[0]
    out = {"S": stages, "M": microbatches, "remat": remat, "args": stats.argument_size_in_bytes,
           "temp": stats.temp_size_in_bytes, "state": held(state, device), "batch": held(batch, device),
           "validate": max(report.device_bytes.values()),
           "implied": {name: list(ops) for name, ops in report.collectives.items()},
           "flops": None if flops is None else flops.hardware,
           "model_flops": None if flops is None else flops.model,
           "moved": moved, "trips": trips, "axes": axes}
    print("PROBE861", json.dumps(out))


CASES = [(1, None, None), (1, None, "full"),
         (2, 2, None), (2, 4, None), (2, 8, None), (2, 16, None),
         (2, 2, "full"), (2, 4, "full"), (2, 8, "full"), (2, 16, "full"),
         (4, 4, None), (4, 8, None), (4, 16, None),
         (4, 4, "full"), (4, 8, "full"), (4, 16, "full")]


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("case", CASES, ids=[f"S{s}-M{m}-{r}" for s, m, r in CASES])
def test_probe(case):
    probe(*case)
