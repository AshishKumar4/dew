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
                      param_groups=(ParamGroup("embedding", ("*embed_tokens*",), learning_rate_multiplier=.5),
                                    ParamGroup("attention", ("*self_attn*",), learning_rate_multiplier=2.0),
                                    ParamGroup("rest", ("*",))))
"""Adamw over three groups (`optax.multi_transform`), whose moments each group holds masked where another
steps: the embedding's holds none in the scanned run, the attention's some of it."""


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
        # In float32 at least, a float64 truth's own float64.
        logits = model.apply({"params": params, **extra}, tokens, train=True)
        return jnp.mean(logits.astype(jnp.promote_types(logits.dtype, jnp.float32)) ** 2)

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


SHARED_SEEDS = 64
"""Check 1's declared seeds for a shared input's gradient through a run updating in its backward."""


def test_what_every_layer_of_a_run_updating_in_its_backward_reads_takes_its_layers_gradient():
    """A floating array every layer of a run updating in its backward reads beside its residual (a
    cross-attention's context, which a trainable encoder computes upstream) takes the sum of the layers'
    cotangents, the gradient an ordinary scan of the same layers gives it, on par (check 1, truth float64):
    the run's rule gave it none, so whatever computed it never trained (audit3 nn-core P1)."""
    from dew.nn import in_backward

    def kept(path, index, gradient, weights, moments):
        # The update is no part of what this holds: the weights stay as they are.
        return weights, {}

    def layer(weights, x, key, context):
        return jnp.tanh(x @ weights["kernel"] + context)

    def through_run(kernel, x, context):
        norms = {"sumsq": jnp.zeros((3,), kernel.dtype), "nonfinite": jnp.zeros((3,), kernel.dtype)}
        with in_backward.planned(in_backward.InBackward(update=kept, moments=())):
            y = in_backward.scanned_run(("layers_0_2",), layer, {"kernel": kernel}, None, {}, norms, x, None,
                                        (context,))
        return jnp.sum(y ** 2)

    def through_scan(kernel, x, context):
        y, _ = jax.lax.scan(lambda x, w: (layer({"kernel": w}, x, None, context), None), x, kernel)
        return jnp.sum(y ** 2)

    def draw(seed):
        kernel, x, context = (jax.random.normal(key, shape) * scale for key, shape, scale in zip(
            jax.random.split(jax.random.key(seed), 3), ((3, 4, 4), (2, 4), (2, 4)), (0.5, 1.0, 0.3),
            strict=True))
        return kernel, x, context

    def gradient(loss):
        return jax.jit(lambda kernel, x, context: jax.grad(loss, argnums=2)(kernel, x, context))

    assert_as_exact(gradient(through_run), gradient(through_scan), draw, "a shared input's gradient",
                    seeds=SHARED_SEEDS)


EIGHT_BIT = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, state_dtype="int8")
"""Adamw keeping its moments in 8 bits (`OptimConfig.state_dtype`), codes and their blocks' scales no
backward writes as a float cotangent: its step takes the whole gradient first."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("optimizer, wider", [(ADAMW, {}), (GROUPED, {}),
                                              (EIGHT_BIT, {"emb_features": 64, "mlp_features": 256})],
                         ids=["adamw", "groups", "8-bit"])
def test_a_scanned_steps_runs_train_under_each_update(optimizer, wider):
    """A one-stage step over a decoder that scans its layers trains under each update: one that reads each
    gradient alone (adamw, whole or by groups) updates its runs in their backward (`Transaction.planned`),
    and one keeping its layers' moments in 8 bits takes the whole gradient first. Each moves the runs'
    weights, advances the optimizer's count once a step, and its loss falls."""
    run = trainer(decoder(scan=True, **wider), optimizer)
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
        gpu = compile_only_gpus(generation, 1, 80 * 2**30)
        run = Trainer(LMObjective(decoder(scan=scan, dtype=dtype, num_layers=8), SEQ), ADAMW,
                      key=jax.random.key(0), mesh=MeshSpec().build(gpu),
                      layout=Layout(min_shard=TINY_SHARD), split=1, accumulation=1, checkpoints=None,
                      tracker=None)
        run.validate(steps=8)
        held.append(compile_ahead(run, token_rows(VOCAB, 1, SEQ + 1)).temp_bytes)
    plain, scanned = held
    assert scanned <= plain, (scanned, plain)


PLANNED_SEEDS = 256
"""Check 1's declared seeds for the scanned steps updating in their backward against a layer at a time's,
sized by `seeds_for` paired on armada (256 for each update): the two arms share every draw's state and
batch."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("optimizer", "fields"),
                         [(ADAMW, {}), (GROUPED, {}), (ADAMW, {"layer_scalar": "frozen"})],
                         ids=["adamw", "groups", "frozen scalar"])
def test_a_scanned_step_updating_in_its_backward_trains_as_a_layer_at_a_time(optimizer, fields):
    """Two steps of a decoder whose one stage scans its layers, each updating them in its backward
    (`Transaction.planned`: the step runs `in_backward.scanned_run`), leave its variables, optimizer state
    as a checkpoint holds them on par with the same steps a layer at a time, whose update reads the
    whole gradient (check 1, against one device's float64): the plan's update of a layer's slice of the
    state is the whole update's (row 4857), a layer reading a read-only collection beside its weights (a
    frozen layer scalar's) as the ordinary scan does. That the plan engages is the step's memory
    (`test_a_scanned_step_holds_no_more_than_a_layer_at_a_time`): a scanned step taking its whole gradient
    first holds more than a layer at a time does."""
    from test_stage_local import decoder as stage_decoder, draw, initials, stepped

    scanned, plain = stage_decoder(scan_layers=True, **fields), stage_decoder(**fields)
    # No average beside them: an EMA two steps old is its draw's, whose rounding both arms share.
    initial = initials(plain, optimizer, ema=None)
    one_stage = MeshSpec(fsdp=2)
    updating = stepped(scanned, optimizer, one_stage, 2, initial, ema=None)
    whole = stepped(plain, optimizer, one_stage, 2, initial, ema=None)
    truth = stepped(plain, optimizer, MeshSpec(), 1, initial, exact=True, ema=None)
    assert_as_exact(updating, whole, draw, "a scanned step updating in its backward", seeds=PLANNED_SEEDS,
                    truth_fn=truth)
