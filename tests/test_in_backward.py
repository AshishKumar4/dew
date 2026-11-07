"""A scanned decoder's step that updates each layer's weights in its backward (`dew.nn.in_backward`, row
4796): the same step as the one that takes the whole gradient first, without holding it."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_as_exact, widened
from token_support import token_rows

from dew.distributed.batches import shard_batch
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig, ParamGroup
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead, compile_only_gpus

VOCAB, SEQ, BATCH = 128, 8, 8
"""The decoder's vocabulary, each row's length and the step's rows."""

TINY_SHARD = 1
"""The smallest leaf the layout shards, so the tiny decoder's are."""

def decoder(*, scan: bool, dtype: str = "float32", num_layers: int = 3, **wider):
    fields = {**TINY_DECODER, "vocab_size": VOCAB, "max_seq_len": SEQ, "num_layers": num_layers, **wider}
    return models.build("causal_transformer", **fields, dtype=dtype, scan_layers=scan)


def trainer(model, optimizer: OptimConfig, *, key: int = 0) -> Trainer:
    run = Trainer(LMObjective(model, SEQ), optimizer, key=jax.random.key(key), mesh=MeshSpec(fsdp=2),
                  layout=Layout(min_shard=TINY_SHARD), checkpoints=None, tracker=None)
    run.device_mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    run.validate(steps=1)
    return run


ADAMW = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1)

GROUPED = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1,
                      param_groups=(ParamGroup("attention", ("*self_attn*",), learning_rate_multiplier=2.0),
                                    ParamGroup("rest", ("*",))))
"""Adamw over two groups (`optax.multi_transform`), whose moments each group holds masked where the other
steps."""


RUN_SEEDS = 64
"""Check 1's declared seeds for a run updating in its backward against the whole step's update, sized by
`seeds_for` paired on armada: the two updates share the layers' weights and moments, and the reference
compared with itself does not resolve within 512 seeds."""


def test_a_run_updating_in_its_backward_moves_as_the_whole_steps_update_would():
    """A scanned run of three layers that updates in its backward (`dew.nn.in_backward.scanned_run`) under
    adamw, each layer's update the optimizer's on the layer alone, moves each layer's weights as adamw's
    update of the whole tree a layer at a time moves them, on par with it (check 1): two programs, the scanned
    one never holding the run's gradient whole, a layer below reading the layer above as its forward did."""
    from dew.nn import in_backward

    scanned = decoder(scan=True)
    plain, wide = scanned.clone(scan_layers=False), scanned.clone(scan_layers=False, dtype=jnp.float64)
    ids = jnp.zeros((2, SEQ), jnp.int32)
    # Stored a layer at a time, as both stacks read them: the scan stacks its run as it reads it.
    layered = nn.unbox(scanned.init(jax.random.key(0), ids))["params"]
    layers = [f"layers_{index}" for index in range(3)]
    solver = optax.adamw(1e-2, weight_decay=0.1)

    def update(path, index, gradient, weights, moments):
        state = (optax.ScaleByAdamState(count=jnp.zeros((), jnp.int32), mu=moments["mu"], nu=moments["nu"]),
                 optax.EmptyState(), optax.EmptyState())
        moved, after = solver.update(gradient, state, weights)
        return optax.apply_updates(weights, moved), {"mu": after[0].mu, "nu": after[0].nu}

    def loss(model, params, extra, tokens):
        return jnp.mean(model.apply({"params": params, **extra}, tokens, train=True).astype(jnp.float32) ** 2)

    def in_backward_step(tokens):
        zeros = jax.tree.map(jnp.zeros_like, layered)
        norms = {name: {"sumsq": jnp.zeros(()), "nonfinite": jnp.zeros(())} for name in layers}
        extra = {"mu": zeros, "nu": zeros, in_backward.NORMS: norms}
        with in_backward.planned(in_backward.InBackward(update=update, moments=("mu", "nu"))):
            moved = jax.grad(lambda params: loss(scanned, params, extra, tokens))(layered)
        return [moved[name] for name in layers]

    def whole_step(model, params):
        def step(tokens):
            gradient = jax.grad(lambda params: loss(model, params, {}, tokens))(params)
            moved = optax.apply_updates(params, solver.update(gradient, solver.init(params), params)[0])
            return [moved[name] for name in layers]
        return step

    def draw(seed):
        return (jax.random.randint(jax.random.key(seed), (2, SEQ), 0, VOCAB),)

    assert_as_exact(jax.jit(in_backward_step), jax.jit(whole_step(plain, layered)), draw,
                    "a run updating in its backward", seeds=RUN_SEEDS,
                    truth_fn=jax.jit(whole_step(wide, widened(layered))))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("optimizer", [ADAMW, GROUPED], ids=["adamw", "groups"])
def test_a_scanned_steps_runs_update_in_their_backward_and_train(optimizer):
    """A one-stage step over a decoder that scans its layers, under an update that reads each gradient
    alone (adamw, whole or by groups), updates its runs in their backward (`Transaction.planned`): it moves
    their weights and moments, advances the optimizer's count once, and its loss falls."""
    run = trainer(decoder(scan=True), optimizer)
    state, _, _ = run.place()
    before = jax.device_get(state.variables["params"]["stages"])
    batch = shard_batch(run.device_mesh, token_rows(VOCAB, BATCH, SEQ + 1))
    step = run.compile(state, batch)
    losses, moved = [], state
    for _ in range(3):
        moved, loss, *_ = step(moved, batch)
        losses.append(float(loss))
    assert losses[-1] < losses[0], losses
    after = jax.device_get(moved.variables["params"]["stages"])
    pairs = zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    assert all(not np.array_equal(old, new) for old, new in pairs)
    assert int(jax.device_get(moved.updates)) == 3


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


@pytest.mark.mesh(devices=8)
def test_a_scanned_step_keeping_8_bit_moments_trains():
    """A scanned step whose optimizer keeps its layers' moments in 8 bits (`OptimConfig.state_dtype`), codes
    and their blocks' scales no backward writes as a float cotangent, takes the whole gradient first, and
    trains: its loss falls and its runs' weights move."""
    run = trainer(decoder(scan=True, emb_features=64, mlp_features=256),
                  OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, state_dtype="int8"))
    state, _, _ = run.place()
    before = jax.device_get(state.variables["params"]["stages"])
    batch = shard_batch(run.device_mesh, token_rows(VOCAB, BATCH, SEQ + 1))
    step = run.compile(state, batch)
    losses, moved = [], state
    for _ in range(3):
        moved, loss, *_ = step(moved, batch)
        losses.append(float(loss))
    assert losses[-1] < losses[0], losses
    after = jax.device_get(moved.variables["params"]["stages"])
    assert all(not np.array_equal(old, new)
               for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True))
