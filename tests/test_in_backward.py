"""A scanned decoder's step that updates each layer's weights in its backward (`dew.nn.in_backward`, row
4796): the same step as the one that takes the whole gradient first, without holding it."""

import jax
import pytest
from model_support import TINY_DECODER
from reference_error import assert_as_exact
from token_support import token_rows

from dew.distributed.batches import shard_batch
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead, compile_only_gpus

VOCAB, SEQ, BATCH = 128, 8, 8
"""The decoder's vocabulary, each row's length and the step's rows."""

TINY_SHARD = 1
"""The smallest leaf the layout shards, so the tiny decoder's are."""

STEP_SEEDS = 16
"""Check 1's declared seeds for a scanned step against a layer-at-a-time one, sized by `seeds_for` on
armada."""


def decoder(*, scan: bool, dtype: str = "float32", num_layers: int = 3):
    fields = {**TINY_DECODER, "vocab_size": VOCAB, "max_seq_len": SEQ, "num_layers": num_layers}
    return models.build("causal_transformer", **fields, dtype=dtype, scan_layers=scan)


def trainer(model, optimizer: OptimConfig, *, key: int = 0) -> Trainer:
    run = Trainer(LMObjective(model, SEQ), optimizer, key=jax.random.key(key), mesh=MeshSpec(fsdp=2),
                  layout=Layout(min_shard=TINY_SHARD), checkpoints=None, tracker=None)
    run.device_mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    run.validate(steps=1)
    return run


ADAMW = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1)


def stepped(run: Trainer):
    """`run`'s first step over a batch drawn at a seed: its weights and moments on their stored layers."""
    state, _, _ = run.place()

    def step(seed):
        batch = shard_batch(run.device_mesh, token_rows(VOCAB, BATCH, SEQ + 1, seed=seed))
        moved = run.stored(run.compile(state, batch)(state, batch)[0])
        return moved.variables["params"], moved.opt_state
    return step


@pytest.mark.mesh(devices=8)
def test_a_scanned_step_updates_each_layer_in_its_backward_as_a_layer_at_a_time_would():
    """A one-stage step over a decoder that scans its layers, under an update that reads each gradient
    alone (adamw), updates its runs in their backward (`Transaction.planned`) and moves its weights and
    moments as the same step a layer at a time does, on par with it (check 1): two programs, the scanned
    one never holding the run's gradient whole. Three layers, so a layer's update below reads the layer
    above as its forward did, before the backward wrote it."""
    def draw(seed):
        return (seed,)

    assert_as_exact(stepped(trainer(decoder(scan=True), ADAMW)), stepped(trainer(decoder(scan=False), ADAMW)),
                    draw, "a scanned step updating in its backward", seeds=STEP_SEEDS)


@pytest.mark.parametrize("generation, dtype", [("sm90", "float32"), ("sm80", "bfloat16")])
def test_a_scanned_step_holds_no_more_than_a_layer_at_a_time(generation, dtype):
    """The scanned step that updates in its backward holds no temporary its layer-at-a-time step does not
    (row 4098), compiled for a GPU (`compile_ahead`): the run's whole gradient, which a scan's backward made
    before an update read any, is never made; under bf16 copies of float32 weights, the masters' update is
    the backward's too."""
    held = []
    for scan in (False, True):
        run = Trainer(LMObjective(decoder(scan=scan, dtype=dtype, num_layers=8), SEQ), ADAMW,
                      key=jax.random.key(0), mesh=MeshSpec().build(compile_only_gpus(generation, 1)),
                      layout=Layout(min_shard=TINY_SHARD), split=1, accumulation=1, checkpoints=None,
                      tracker=None)
        run.validate(steps=8)
        held.append(compile_ahead(run, token_rows(VOCAB, 1, SEQ + 1)).temp_bytes)
    plain, scanned = held
    assert scanned <= plain, (scanned, plain)
