"""A scanned decoder's step that updates each layer's weights in its backward (`dew.nn.in_backward`, row
4796): the same step as the one that takes the whole gradient first, without holding it."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, scanning, stacked_as, under_kernels, unrolled
from reference_error import assert_as_exact, dew_path, widened
from token_support import token_rows

from dew.distributed.batches import shard_batch
from dew.nn.backbones.layer_names import group_layers
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.options import program
from dew.nn.precision import at_least_fp32
from dew.objectives.base import Ratio
from dew.objectives.lm import LMObjective
from dew.optim import AdamW, OptimConfig, ParamGroup
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced
from dew.training.ahead import compile_ahead
from dew.training.narrow import forward_variables

VOCAB, SEQ, BATCH = 128, 8, 8
"""The decoder's vocabulary, each row's length and the step's rows."""

TINY_SHARD = 1
"""The smallest leaf the layout shards, so the tiny decoder's are."""

def decoder(*, dtype: str = "float32", num_layers: int = 3, **wider):
    """The tiny decoder `wider` widens, which scans its run of like layers (a plain arm runs it under
    `unrolled`)."""
    fields = {**TINY_DECODER, "vocab_size": VOCAB, "max_seq_len": SEQ, "num_layers": num_layers, **wider}
    return models.build("causal_transformer", **fields, dtype=dtype)


def trainer(model, optimizer: OptimConfig, *, key: int = 0) -> Trainer:
    run = Trainer(LMObjective(model, SEQ), optimizer, key=jax.random.key(key), mesh=MeshSpec(fsdp=2),
                  advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)), checkpoints=None,
                  tracker=None)
    run.device_mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    run.validate(steps=1)
    return run


ADAMW = OptimConfig(optimizer=AdamW(weight_decay=0.1), lr=1e-2)

GROUPED = OptimConfig(optimizer=AdamW(weight_decay=0.1), lr=1e-2,
                      param_groups=(ParamGroup("embedding", ("*embed_tokens*",), learning_rate_multiplier=.5),
                                    ParamGroup("attention", ("*self_attn*",), learning_rate_multiplier=2.0),
                                    ParamGroup("rest", ("*",))))
"""Adamw over three groups (`optax.multi_transform`), whose moments each group holds masked where another
steps: the embedding's holds none in the scanned run, the attention's some of it."""


RUN_SEEDS = 64
"""Check 1's declared seeds for a run updating in its backward against the whole step's update, sized by
`seeds_for` paired on armada: the two updates share the layers' weights and moments, and the reference
compared with itself does not resolve within 512 seeds."""
RUN_CONDITIONING = 8
"""The weighed check's declared seeds for the same site (`seeds_for(..., conditioning=True)` on armada):
adamw's first step moves a weight by its rate times g / (|g| + eps), so at a gradient a few eps across the
update turns on the gradient's last bits; where the unweighted check stays undecided, or one element rules
a worst row, each element is weighed by the whole step's own rounding there (row 8752: a norm scale's
element at seed 170, whose two float32 gradients were 1.98e-8 and 1.30e-8 from float64, ruled its row's
worst on CUDA, the reference's own arithmetic against itself deciding the same rule red in 3 of 8 runs,
row 10145)."""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("barriered", [False, True], ids=["marked", "barriered"])
def test_a_run_updating_in_its_backward_moves_as_the_whole_steps_update_would(barriered):
    """A scanned run of three layers that updates in its backward (`dew.nn.in_backward.scanned_run`) under
    adamw, each layer's update the optimizer's on the layer alone, moves each layer's weights as adamw's
    update of the whole tree a layer at a time moves them, on par with it (check 1): two programs, the scanned
    one never holding the run's gradient whole, a layer below reading the layer above as its forward did. On
    both paths of precision's error-free transforms: their arithmetic marked to round alone, and behind a
    barrier (`dew._cache.BARRIERED`, row 8378)."""
    from dew.nn import in_backward

    scanned = decoder()
    plain, wide = scanned, scanned.clone(dtype=jnp.float64)
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

    from dew import _cache

    with _cache.BARRIERED(new_value=barriered):
        # The whole step's arms a layer at a time (`unrolled`).
        assert_as_exact(jax.jit(in_backward_step), dew_path(unrolled(jax.jit(whole_step(plain, layered)))),
                        draw, "a run updating in its backward", seeds=RUN_SEEDS,
                        truth_fn=unrolled(jax.jit(whole_step(wide, widened(layered)))),
                        conditioning=RUN_CONDITIONING)


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


EIGHT_BIT = OptimConfig(optimizer=AdamW(weight_decay=0.1, state_dtype="int8"), lr=1e-2)
"""Adamw keeping its moments in 8 bits (`AdamW(state_dtype='int8')`), codes and their blocks' scales no
backward writes as a float cotangent: its step takes the whole gradient first."""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("optimizer, wider", [(ADAMW, {}), (GROUPED, {}),
                                              (EIGHT_BIT, {"emb_features": 64, "mlp_features": 256})],
                         ids=["adamw", "groups", "8-bit"])
def test_a_scanned_steps_runs_train_under_each_update(optimizer, wider):
    """A one-stage step over a decoder that scans its layers trains under each update: one that reads each
    gradient alone (adamw, whole or by groups) updates its runs in their backward (`Transaction.planned`),
    and one keeping its layers' moments in 8 bits takes the whole gradient first. Each moves the runs'
    weights, advances the optimizer's count once a step, and its loss falls."""
    run = trainer(decoder(**wider), optimizer)
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


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("generation, dtype, width", [("sm90", "float32", None), ("sm80", "bfloat16", None),
                                                     ("sm80", "bfloat16", 512), ("sm90", "bfloat16", 512)])
def test_a_scanned_step_holds_no_more_than_a_layer_at_a_time(generation, dtype, width):
    """The scanned step that updates in its backward (row 4098), compiled for a GPU (`compile_ahead`), under
    bf16 copies of float32 weights too, where the masters' update is the backward's, holds no more
    temporaries than its reference: the same scanned step under global-norm clipping, whose compile holds
    the layers' gradient whole, less that gradient (each of the layers' weights at the dtype the step
    differentiates it in, its compute copy's where the step reads one), plus one layer's rows of the state
    the walk moves (the layers' weights as the state holds them and the optimizer's moments of them, over
    the run's layers): the rows a layer's update reads, which XLA keeps beside the stacks or fuses into
    their writes as it fuses (row 7953: one barrier in the loss flipped which). A regression bound, read
    off the clipped step's own input state: at width 512 on an A100 it catches a walk that kept slices of
    the stacks past a layer's writes (row 8389)."""
    wider = {} if width is None else {"emb_features": width, "mlp_features": 4 * width,
                                      "num_heads": max(2, width // 64)}

    def built(optimizer: OptimConfig):
        gpu = compile_only_gpus(generation, 1, 80 * 2**30)
        model = decoder(dtype=dtype, num_layers=LAYERS, **wider)
        run = Trainer(LMObjective(model, SEQ), optimizer, key=jax.random.key(0), mesh=MeshSpec().build(gpu),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)), split=1,
                      checkpoints=None,
                      tracker=None)
        run.validate(steps=8)
        return run

    def layer_leaves(tree) -> list:
        # The layers' leaves: under a layer's own tree (`layers_<i>`) or a run's stacked one
        # (`layers_<a>_<b>`, a staged start's `stages/<run>`), `group_layers`'s names.
        leaves = [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
                  if any(isinstance(key, jax.tree_util.DictKey) and isinstance(key.key, str)
                         and group_layers(key.key) is not None for key in path)]
        assert leaves, jax.tree_util.tree_structure(tree)
        return leaves

    def held(leaves) -> int:
        return sum(int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize for leaf in leaves)

    updating = compile_ahead(built(ADAMW), token_rows(VOCAB, 1, SEQ + 1)).temp_bytes
    run = built(dataclasses.replace(ADAMW, clip=1.0))
    clipped = compile_ahead(run, token_rows(VOCAB, 1, SEQ + 1)).temp_bytes
    executable = run.executable
    assert executable is not None
    (state, *_), _ = executable.args_info
    gradient = held(layer_leaves(forward_variables(state.variables, state.compute)["params"]))
    rows = (held(layer_leaves(state.variables["params"])) + held(layer_leaves(state.opt_state))) // LAYERS
    assert updating <= clipped - gradient + rows, (updating, clipped, gradient, rows)


LAYERS = 8
"""The scanned run's layers, a row of each of its stacked leaves each."""


PLANNED_SEEDS = 256
"""Check 1's declared seeds for the scanned steps updating in their backward against a layer at a time's,
sized by `seeds_for` paired on armada (256 for each update): the two arms share every draw's state and
batch. Each update a site of its own, drawn in armada's map tasks of 64 (`check1_shards`, its count this
one): each cell took 466 to 595 s whole (CI 11-14), and one site over the three 173 s a task of 32, each
task compiling every cell's arms."""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("optimizer", "fields"), [
    pytest.param(ADAMW, {}, id="adamw", marks=pytest.mark.check1_shards(seeds=256, per_task=64)),
    pytest.param(GROUPED, {}, id="groups", marks=pytest.mark.check1_shards(seeds=256, per_task=64)),
    pytest.param(ADAMW, {"layer_scalar": "frozen"}, id="frozen-scalar",
                 marks=pytest.mark.check1_shards(seeds=256, per_task=64)),
])
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

    scanned, plain = stage_decoder(**fields), stage_decoder(**fields)
    # No average beside them: an EMA two steps old is its draw's, whose rounding both arms share.
    initial = initials(plain, optimizer, ema=None)
    one_stage = MeshSpec(fsdp=2)
    updating = stepped(scanned, optimizer, one_stage, 2, initial, ema=None)
    # The steps a layer at a time (`unrolled`), the plain loop's.
    whole = unrolled(stepped(plain, optimizer, one_stage, 2, initial, ema=None))
    truth = unrolled(stepped(plain, optimizer, MeshSpec(), 1, initial, exact=True, ema=None))
    assert_as_exact(updating, whole, draw, "a scanned step updating in its backward", seeds=PLANNED_SEEDS,
                    truth_fn=truth)


def _walked(e, *, finite: bool = True, top: float = 1.0, runs: int = 1, through_context: bool = False):
    """Three layers, each `y = a x + sqrt(0 x + s - s' + e) + c k` with `s'` the stop-gradient of `s` and
    `k` a constant: a finite forward, whose gradient is not finite at a layer whose `e` is zero, in its own
    weights and in the input it hands the layer below (0 times infinity), and whose `c` takes the gradient
    `k`, 1 at every layer but the walk's first (the top of the upper run), whose is `top`. Updated in their
    backward (`dew.nn.in_backward`) by an Adam-like rule that counts its steps per layer, under a step whose
    loss is `finite`, as a run that skips steps not finite walks it: the weights, moments and per-layer
    non-finite counts the walk leaves. Given `runs` of
    two, the same three layers run twice, the upper reading the lower's output: each run's own; through
    `through_context`, the upper reading the lower's output as a context every layer reads (`shared`, an
    encoder's to a decoder's cross-attention), its own input a constant."""
    from dew.nn import in_backward

    count = 3

    def weights_of(run):
        first = jnp.arange(count) == count - 1 if run == runs - 1 else jnp.zeros((count,), bool)
        return {"a": jnp.full((count,), 0.5), "s": jnp.ones((count,)), "e": jnp.asarray(e, jnp.float32),
                "c": jnp.zeros((count,)), "k": jnp.where(first, top, 1.0).astype(jnp.float32)}

    weights = [weights_of(run) for run in range(runs)]
    moments = [{"mu": jax.tree.map(jnp.zeros_like, held), "nu": jax.tree.map(jnp.zeros_like, held),
                "count": {name: jnp.zeros((count,)) for name in held}} for held in weights]
    norms = [{"sumsq": jnp.zeros((count,)), "nonfinite": jnp.zeros((count,))} for _ in range(runs)]

    def layer(w, x, per_layer, *context):
        read = x if not context else x + context[0]
        return (w["a"] * read + jnp.sqrt(0 * x + w["s"] - jax.lax.stop_gradient(w["s"]) + w["e"])
                + w["c"] * jax.lax.stop_gradient(w["k"]))

    def update(path, index, gradient, held, moments):
        mu = jax.tree.map(lambda m, g: 0.9 * m + g, moments["mu"], gradient)
        nu = jax.tree.map(lambda v, g: 0.99 * v + 0.01 * jnp.square(g), moments["nu"], gradient)
        moved = jax.tree.map(lambda w, m, v: w - 0.1 * m / (jnp.sqrt(v) + 1.0), held, mu, nu)
        return moved, {"mu": mu, "nu": nu, "count": jax.tree.map(lambda c: c + 1, moments["count"])}

    def loss(weights, moments, norms):
        x = jnp.ones(())
        for run in range(runs):
            if through_context and run:
                x = in_backward.scanned_run((f"run{run}",), layer, weights[run], None, moments[run],
                                            norms[run], jnp.ones(()), None, (x,))
            else:
                x = in_backward.scanned_run((f"run{run}",), layer, weights[run], None, moments[run],
                                            norms[run], x, None, ())
        return x

    with in_backward.planned(in_backward.InBackward(update=update, moments=("mu", "nu", "count"))), \
            in_backward.gated(jnp.asarray(a=True), jnp.asarray(finite)):
        # Compiled, as a step is: each run's backward traced on its own (`marked`), so the latch crosses
        # runs as data the walk carries, not as a value one trace leaves another.
        walked = program(jax.grad(loss, argnums=(0, 1, 2)))(weights, moments, norms)
    return walked if runs > 1 else tuple(each[0] for each in walked)


def test_a_layer_walked_past_a_gradient_that_is_not_finite_keeps_its_weights_and_steps_its_count():
    """A step updating in its backward whose gradient first turns non-finite at the middle of three layers,
    a finite loss above it: the top layer, walked first, moves and stays finite; the middle layer and the
    one below, whose gradients are not finite, keep their weights and take the zero gradient's update, so
    every layer's count steps alike; the walk counts both non-finite (training.md, a step that is not
    finite; train 912's review, F2), as a run that skips such steps walks it; a step whose loss is not
    finite moves no layer."""
    weights, moments, norms = _walked([1.0, 0.0, 1.0])
    assert bool(jnp.all(jnp.isfinite(weights["a"]))) and float(weights["a"][2]) != 0.5
    np.testing.assert_array_equal(np.asarray(weights["a"][:2]), [0.5, 0.5])
    np.testing.assert_array_equal(np.asarray(moments["count"]["a"]), [1.0, 1.0, 1.0])
    assert [bool(count > 0) for count in norms["nonfinite"]] == [True, True, False]
    unmoved, _, _ = _walked([1.0, 1.0, 1.0], finite=False)
    np.testing.assert_array_equal(np.asarray(unmoved["a"]), [0.5, 0.5, 0.5])


def test_a_walk_names_the_first_layer_it_met_not_finite_across_its_stage_axis():
    """The first layer a backward walk met a gradient not finite at, of runs whose counts lie on a stage axis
    and a slot axis (one stage): the top layer of the highest slot holding one, a periodic run's slot its
    period's layers, numbered across the runs in order. The stage axis was read as the layers, naming
    layer 0 (the round-2 review's F2), and a period's slot as one layer (the head review's F6)."""
    from dew.training.transaction import _unfinite_layer

    assert int(_unfinite_layer([(jnp.asarray([[1.0, 1.0, 0.0]]), 1)])) == 1
    assert int(_unfinite_layer([(jnp.asarray([[0.0, 0.0]]), 1), (jnp.asarray([[1.0, 0.0, 0.0]]), 1)])) == 2
    assert int(_unfinite_layer([(jnp.asarray([[0.0, 0.0, 0.0]]), 1)])) == -1
    # A periodic run of two repeats of a six-layer period, its upper repeat holding the count: layers 6 to 11,
    # topped by 11; a run after it numbered past its twelve layers.
    assert int(_unfinite_layer([(jnp.asarray([[0.0, 1.0]]), 6)])) == 11
    assert int(_unfinite_layer([(jnp.asarray([[0.0, 0.0]]), 6), (jnp.asarray([[1.0]]), 1)])) == 12


@pytest.mark.parametrize("through_context", [False, True], ids=["its input", "a context"])
def test_a_layer_held_in_an_upper_run_holds_every_layer_of_the_runs_walked_after_it(through_context):
    """A compiled step of two in-backward runs, one after the other, whose upper run's top layer, the walk's
    first, takes a gradient that is not finite, under a run that skips such steps: every layer of both runs
    keeps its
    weights, each count stepping alike, as one run's layers after a held one do (training.md, a step that
    is not finite), whether the lower run reaches the upper as its input or as a context every upper layer
    reads (an encoder's). The latch started anew in each run, so the lower run's layers moved (train 921's
    review, F1), carried as a value between runs it escaped the compiled backward's own trace (the head
    review's F1), and carried on the input's cotangent alone an encoder beneath moved (its round 2, F3)."""
    weights, moments, _ = _walked([1.0, 1.0, 1.0], top=np.inf, runs=2, through_context=through_context)
    for run in range(2):
        np.testing.assert_array_equal(np.asarray(weights[run]["a"]), [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(np.asarray(moments[run]["count"]["a"]), [1.0, 1.0, 1.0])


class CollectionWriting(LMObjective):
    """The language model's loss beside a collection its forward writes: the exponential of its first
    token id's mean times `scale`, past float32 where `scale` is large, its loss and gradient finite."""

    scale = 1.0

    def fresh_variables(self, key, held):
        return {**super().fresh_variables(key, held), "seen": {"value": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        stats, aux = super().loss(variables, batch, step)
        seen = jnp.exp(jnp.mean(batch["text"][..., 0].astype(jnp.float32)) * self.scale)
        return stats, dataclasses.replace(aux, variables={**(aux.variables or {}), "seen": {"value": seen}})


@pytest.mark.mesh(devices=8)
def test_a_scanned_step_whose_collection_is_not_finite_moves_no_weight():
    """A step of a run that skips steps not finite, updating its runs in their backward, whose loss and
    gradient are finite and whose forward writes a collection that is not finite (an exponential past
    float32) moves no weight of its walked
    runs nor any other, and is reported not finite: the walk reads the forward's statistics and collections
    as the commit does. It read the loss alone, and its layers moved where the commit rejected the step (the
    round-3 head review's F1)."""
    objective = type("Overflowing", (CollectionWriting,), {"scale": 1e4})(decoder(), SEQ)
    run = Trainer(objective, ADAMW, key=jax.random.key(0), mesh=MeshSpec(fsdp=2),
                  advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD), skip_nonfinite=True),
                  checkpoints=None, tracker=None)
    run.device_mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    run.validate(steps=1)
    state, _, _ = run.place()
    before = jax.device_get(state.variables["params"])
    batch = shard_batch(run.device_mesh, token_rows(VOCAB, BATCH, SEQ + 1))
    moved, loss, _, finite, *_ = run.compile(state, batch)(state, batch)
    assert bool(jnp.isfinite(loss)) and not bool(finite) and int(jax.device_get(moved.updates)) == 0
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(jax.device_get(moved.variables["params"])),
                        strict=True):
        np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize("grouped, input_lm, context_backbone, seeds", [
    pytest.param(False, 0.0, False, 32, id="decision",
                 marks=pytest.mark.check1_shards(seeds=32, per_task=32)),
    pytest.param(True, 0.0, False, 32, id="decision-grouped",
                 marks=pytest.mark.check1_shards(seeds=32, per_task=32)),
    pytest.param(False, 0.25, False, 160, id="decision-input-lm",
                 marks=pytest.mark.check1_shards(seeds=160, per_task=32)),
    # Its judgement resolved past 32 draws alone, within 160 (the map judge's units).
    pytest.param(False, 0.0, True, 160, id="decision-context-backbone",
                 marks=pytest.mark.check1_shards(seeds=160, per_task=32)),
])
def test_a_decision_step_over_a_scanned_backbone_trains_as_its_ordinary_backward(grouped, input_lm,
                                                                                   context_backbone, seeds):
    """A decision objective's scanned backbone, which its step updates in its backward
    (`dew.nn.in_backward`), runs inside `component('backbone')`, so its plan keys its run under
    `backbone/stages/...` (row 12076: the run's path lacked `backbone`, raising `KeyError: 'stages'`), and two
    steps leave every parameter on par with the same steps through the ordinary backward of an unscanned
    backbone (check 1, against one device's float64), its parameters grouped or not. With `input_lm` or a
    context the backbone encodes in its own pass, the loss differentiates the backbone twice a step, so the
    step updates none in its backward (`Objective._updates_in_backward`) and trains as the ordinary scan
    does, on par alike (review of row 12076's fix: finite values alone were checked); a loss that
    differentiated a run twice and said nothing is refused
    (`test_a_loss_differentiating_a_scanned_run_twice_is_refused_its_update_in_the_backward`)."""
    from arm_forms import as_given
    from decision_support import decision_rows
    from test_stage_local import wide

    from dew.data.dataset import DataPartition
    from dew.data.text import ByteTokenizer
    from dew.decision import Choice, DecisionObjective, Example, Specials
    from dew.decision.layout import SegmentLayout
    from dew.distributed.batches import shard_batch
    from dew.nn.cross_attention import CrossAttention

    team = Choice("Which team?", ["billing", "technical"])
    examples = [
        Example(text, {"team": team}, {"team": label}, context={"plan": "pro"} if context_backbone else None)
        for text, label in (("charged twice", 0), ("site down", 1))
    ] * 2

    def backbone(*, dtype: str = "float32"):
        model = decoder(dtype=dtype, vocab_size=256, max_seq_len=32, num_layers=3)
        return model.with_cross_attention(CrossAttention(layer_ids=(0,), context_length=16)) if (
            context_backbone) else model
    # eps=1e-4 keeps AdamW from magnifying 1e-8 rounding noise on DecisionHead's shift-invariant k_proj.bias.
    adamw = AdamW(weight_decay=0.1, eps=1e-4)
    optimizer = (
        OptimConfig(optimizer=adamw, lr=1e-2,
                    param_groups=(ParamGroup("backbone", ("*backbone*",), learning_rate_multiplier=2.0),
                                  ParamGroup("rest", ("*",))))
        if grouped else OptimConfig(optimizer=adamw, lr=1e-2)
    )
    layout = SegmentLayout.state_first_row(max_len=32, head_max_len=24, option_tokens=6)
    opts = {"tokenizer": ByteTokenizer(), "specials": Specials(None, 10, 0, "\x00", 255),
            "layout": layout, "input_lm": input_lm,
            "context_encoder": "backbone" if context_backbone else "table"}
    seed = DecisionObjective(backbone(), **opts)
    batch = next(decision_rows(examples, 2, objective=seed.data()).train(DataPartition()))

    def built(*, exact: bool = False) -> Trainer:
        model = backbone(dtype="float64" if exact else "float32")
        run = Trainer(DecisionObjective(model, **opts), optimizer, key=jax.random.key(0), mesh=MeshSpec(),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=2**30)), split=1, checkpoints=None,
                      tracker=None)
        run.device_mesh = MeshSpec().build(jax.devices()[:1])
        run.validate(steps=2)
        return run

    with jax.enable_x64(new_val=False):
        start = unrolled(built)().initial_state()
    init, made = jax.jit(seed.init), {}

    def initial(drawn: int):
        # Each seed's own variables, the optimizer state every draw starts from.
        if drawn not in made:
            with jax.enable_x64(new_val=False):
                variables = nn.unbox(init(jax.random.key(drawn)))
                variables = jax.tree.map(lambda held, new: jnp.asarray(new, held.dtype), start.variables,
                                         variables)
                made[drawn] = dataclasses.replace(start, variables=variables)
        return made[drawn]

    def stepped(*, scan: bool, exact: bool = False):
        # Its stack scanned or a layer at a time, compiled and stepped under that choice (`stacked_as`).
        compiled = []

        @as_given
        @stacked_as("scanned" if scan else "plain")
        def run(drawn):
            if not compiled:
                compiled.append(built(exact=exact))
            steps = compiled[0]
            first = initial(int(drawn))
            state = steps.staged(wide(first) if exact else jax.tree.map(jnp.copy, first))
            placed = shard_batch(steps.device_mesh, batch)
            if len(compiled) == 1:
                compiled.append(steps.compile(state, placed))
            for _ in range(2):
                state, *_ = compiled[1](state, placed)
            # The backbone's moments beside the weights: an update in its backward writes them apart. The
            # head's, which both arms update alike, include a shift-invariant bias's second moment, the
            # square of rounding noise its gradient is, which no draw resolves.
            stored = steps.stored(state)
            moments = jax.tree_util.tree_map_with_path(
                lambda path, leaf: leaf if "backbone" in jax.tree_util.keystr(path) else None,
                stored.opt_state)
            return jax.device_get({"params": stored.variables["params"], "opt_state": moments})
        return run

    assert_as_exact(stepped(scan=True), stepped(scan=False), lambda drawn: (np.int64(drawn),),
                    "a decision step over a scanned backbone", seeds=seeds,
                    truth_fn=stepped(scan=False, exact=True))


def test_a_loss_differentiating_a_scanned_run_twice_is_refused_its_update_in_the_backward():
    """A loss that runs a scanned stack twice and differentiates both (a language model's loss counted
    twice) would take the sum of two updates as its weights were its step to update the stack in its
    backward (`dew.nn.in_backward`): its plan, handling the run twice, is refused by name
    (`Objective._updates_in_backward`), never trained silently (row 12076's review)."""
    from dew.data.dataset import Loader

    class Twice(LMObjective):
        def loss(self, variables, batch, step):
            stats, aux = super().loss(variables, batch, step)
            again, _ = super().loss(variables, batch, step)
            return jax.tree.map(lambda first, second: first + second, stats, again), aux

    objective = Twice(decoder(), SEQ)
    tokens = token_rows(VOCAB, 2, SEQ + 1)
    data = Loader(train=lambda partition: iter([tokens, tokens]), val=None, records=2, batch=2)
    run = Trainer(objective, OptimConfig(optimizer=AdamW(), lr=1e-2), key=jax.random.key(0),
                  mesh=MeshSpec(), advanced=TrainerAdvanced(layout=Layout(min_shard=2**30)), split=1,
                  checkpoints=None,
                  tracker=None)
    run.device_mesh = MeshSpec().build(jax.devices()[:1])
    with pytest.raises(ValueError):
        scanning(run.fit)(data, steps=1, log_every=1)


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize(("generation", "dtype"), [("sm90", "float32"), ("sm80", "bfloat16")])
def test_an_accumulating_scanned_step_holds_no_extra_gradient_buffer(generation, dtype):
    """A scanned step that accumulates its update over microbatches, its batches or its batch's split,
    pools each layer's gradient into the window's accumulator as its backward walks the layer
    (`Transaction._planned_accumulation`): its temporaries hold no stacked `[L, ...]` gradient beside the
    accumulator, nor the scan's forward residuals unrematerialized, past what a one-microbatch step holds
    (row 12080: an sm90 f32 step held 5.6 MB more, an sm80 bf16 one 2.4 MB)."""

    wide = {**TINY_DECODER, "emb_features": 64, "mlp_features": 2048}
    model = models.build("causal_transformer", vocab_size=96, **wide, num_kv_heads=2, dtype=dtype)

    def ahead(accumulation: int, split: int, rows: int):
        gpu = compile_only_gpus(generation, 1, 64 * 2**30)
        run = Trainer(LMObjective(model, 16), OptimConfig(optimizer=AdamW(), lr=1e-3),
                      key=jax.random.key(0), mesh=MeshSpec().build(gpu),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=256)), accumulation=accumulation,
                      split=split)
        run.validate(steps=8)
        return compile_ahead(run, token_rows(96, rows, 17))

    one = ahead(1, 1, 1)
    accumulated = ahead(4, 1, 1)
    split = ahead(1, 2, 2)
    shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, 16), jnp.int32))["params"]
    narrow = 2 * sum(leaf.size for leaf in jax.tree.leaves(shapes))

    def beside(ahead) -> int:
        """What a step holds beside its arguments: its outputs that alias none and its temporaries, so a
        second stacked sum held as an unaliased output counts as one held among its temporaries does."""
        return ahead.output_bytes - ahead.alias_bytes + ahead.temp_bytes

    assert beside(accumulated) - beside(one) < narrow, (beside(accumulated), beside(one), narrow)
    assert beside(split) - beside(one) < narrow, (beside(split), beside(one), narrow)


ACCUMULATING_SEEDS = 64
"""Check 1's seeds for a scanned step pooling in its backward against the unrolled step's window: the
layer-at-a-time sites' (`RUN_SEEDS`), whose judgement the map judge widens where it needs more units."""


class _Penalized(LMObjective):
    """An LM whose loss adds a term on its stacked layers' parameters themselves, read where they lie beside
    the layers' scan: a run whose parameters another term reads pools nothing in its backward
    (`Transaction._pools_once`)."""

    def loss(self, variables, batch, step):
        stats, aux = super().loss(variables, batch, step)
        assert isinstance(stats, Ratio), type(stats)
        stacked = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(variables["params"])
                   if "layers" in jax.tree_util.keystr(path)]
        penalty = sum(jnp.sum(jnp.square(leaf.astype(at_least_fp32(leaf.dtype)))) for leaf in stacked)
        return Ratio(stats.total + 1e-3 * penalty * stats.mass, stats.mass), aux


class _Branched(LMObjective):
    """An LM whose loss runs its decoder in a branch its batch chooses (`jax.lax.cond`): a batch whose first
    token is even is scored, an odd one's counted, its tokens' mass kept, the decoder never run, its loss
    none. A run read in a branch pools nothing in its backward (`Transaction._pools_once`)."""

    def loss(self, variables, batch, step):
        def scored(held):
            return LMObjective.loss(self, held, batch, step)

        stats, aux = jax.eval_shape(scored, variables)

        def counted(held):
            del held
            mass = jnp.sum(batch["text"][:, 1:] != 0).astype(stats.mass.dtype)
            return (Ratio(jnp.zeros(stats.total.shape, stats.total.dtype), mass),
                    jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), aux))
        return jax.lax.cond(batch["text"][0, 0] % 2 == 0, scored, counted, variables)


class _Repeated(LMObjective):
    """An LM whose loss reads its batch twice in a loop (`jax.lax.scan`), its two passes' statistics summed:
    a run read in a loop pools nothing in its backward (`Transaction._pools_once`)."""

    def loss(self, variables, batch, step):
        def scored(carry, _):
            return carry, LMObjective.loss(self, variables, batch, step)
        _, (stats, aux) = jax.lax.scan(scored, None, None, length=2)
        assert isinstance(stats, Ratio), type(stats)
        return (Ratio(jnp.sum(stats.total), jnp.sum(stats.mass)), jax.tree.map(lambda leaf: leaf[0], aux))


OBJECTIVES = {"lm": LMObjective, "penalized": _Penalized, "branched": _Branched, "repeated": _Repeated}
"""Each accumulating witness's objective: the LM, whose runs pool in their backward, and those whose runs
the step accumulates ordinarily, read beside a term on the weights, in a branch or in a loop."""


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("dtype", "clip", "periodic", "objective"), [
    pytest.param("float32", 0.0, False, "lm", id="fp32",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param("bfloat16", 1.0, False, "lm", id="bf16-clipped",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param("float32", 0.0, True, "lm", id="periodic",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param("float32", 0.0, False, "penalized", id="direct-reader",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param("float32", 0.0, False, "branched", id="branched",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param("float32", 0.0, False, "repeated", id="repeated",
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
])
def test_an_accumulating_scanned_step_matches_unrolled(dtype, clip, periodic, objective):
    """A scanned step accumulating a window of two batches each split in two, its parts of unequal token
    counts (padding scored as none), leaves its parameters and optimizer state after two updates on par
    with the unrolled step's (check 1, against one device's float64 unrolled step), on two devices
    splitting its parameters: pooling its gradient in its backward (`Transaction._planned_accumulation`),
    clipped with narrow copies or over a periodic stack; or accumulating it as an unscanned step does where
    its loss reads a run beside a term on its weights, in a branch or in a loop (`_pools_once`)."""
    from arm_forms import as_given
    from test_stage_local import wide


    extra = ({"num_layers": 4, "layer_types": ("sliding_attention", "full_attention") * 2,
              "kinds": {"sliding_attention": {"window": 4}}}
             if periodic else {"num_layers": 2})
    optimizer = OptimConfig(optimizer=AdamW(weight_decay=0.1), lr=1e-2, clip=clip)

    def built(*, exact: bool = False) -> Trainer:
        model = decoder(dtype="float64" if exact else dtype, **extra)
        mesh = MeshSpec() if exact else MeshSpec(fsdp=2)
        scored = OBJECTIVES[objective](model, SEQ, pad_id=0)
        run = Trainer(scored, optimizer, key=jax.random.key(0), mesh=mesh,
                      advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)), accumulation=2,
                      split=2, checkpoints=None, tracker=None)
        run.device_mesh = mesh.build(jax.devices()[:1 if exact else 2])
        run.validate(steps=4)
        return run

    with jax.enable_x64(new_val=False):
        start = unrolled(built)().initial_state()
    init, made = jax.jit(decoder(dtype=dtype, **extra).init), {}

    def initial(drawn: int):
        # Each seed's own variables, the optimizer state every draw starts from.
        if drawn not in made:
            with jax.enable_x64(new_val=False):
                variables = nn.unbox(init(jax.random.key(drawn), jnp.zeros((1, SEQ), jnp.int32)))
                variables = jax.tree.map(lambda held, new: jnp.asarray(new, held.dtype), start.variables,
                                         variables)
                made[drawn] = dataclasses.replace(start, variables=variables)
        return made[drawn]

    def stepped(*, scan: bool, exact: bool = False):
        # Its stack scanned or a layer at a time, a bf16 step reading narrow copies, compiled and stepped
        # under those choices (`under_kernels`).
        compiled = []

        @as_given
        @under_kernels(layer_scan="scanned" if scan else "plain",
                       narrow_copies="on" if dtype == "bfloat16" and not exact else "off")
        def run(tokens, drawn):
            if not compiled:
                compiled.append(built(exact=exact))
            steps = compiled[0]
            first = initial(int(drawn))
            state = steps.staged(wide(first) if exact else jax.tree.map(jnp.copy, first))
            placed = [shard_batch(steps.device_mesh, {"text": rows}) for rows in tokens]
            if len(compiled) == 1:
                compiled.append(steps.compile(state, placed[0]))
            for batch in placed:
                state, *_ = compiled[1](state, batch)
            stored = steps.stored(state)
            assert int(stored.updates) == 2, int(stored.updates)
            return jax.device_get({"params": stored.variables["params"], "opt_state": stored.opt_state})
        return run

    def draw(seed):
        # Four batches of four rows, two updates of two batches each, no token the padding one. Each batch
        # splits into its even rows and its odd ones, the odd part's second row ending in padding of the
        # batch's own length, so no two parts count as many tokens; each batch's first token even where it
        # is the first of its update and odd where it is the second, so a branching loss takes both of its
        # ways in one update.
        batches = []
        for at in range(4):
            rows = token_rows(VOCAB, 4, SEQ + 1, seed=4 * seed + at, low=1)["text"]
            rows[3, SEQ + 1 - (at + 2):] = 0
            rows[0, 0] = 2 if at % 2 == 0 else 1
            batches.append(rows)
        return np.stack(batches), np.int64(seed)

    assert_as_exact(stepped(scan=True), stepped(scan=False), draw, "an accumulating scanned step",
                    seeds=ACCUMULATING_SEEDS, truth_fn=stepped(scan=False, exact=True))


def test_an_accumulating_scanned_step_that_watches_its_layers_logs_what_they_made():
    """A scanned step accumulating two batches that watches its MLPs' outputs (`Inspection.modules`, no
    cotangents) runs them as the ordinary scan does, which opens the collections its watch reads, and logs
    their summaries (review of row 12080: its layers, pooled in their backward, ran with no collection
    open and the watch raised)."""
    from test_inspection import Kept
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.training.inspection import Inspection

    tracker = Kept()
    run = Trainer(LMObjective(decoder(num_layers=2), SEQ), ADAMW, key=jax.random.key(0), mesh=MeshSpec(),
                  advanced=TrainerAdvanced(layout=Layout(min_shard=2**30),
                                           inspection=Inspection(modules=("*MLP",), cotangents=False,
                                                                 every=1)),
                  accumulation=2, tracker=tracker, checkpoints=None)
    run.device_mesh = MeshSpec().build(jax.devices()[:1])
    loader = Loader(train=lambda partition: repeated_token_batch(VOCAB, 4, SEQ + 1), val=None, records=None,
                    batch=4)
    scanning(run.fit)(loader, steps=2, log_every=1, eval_every=None)
    logged = {name for row in tracker.rows.values() for name in row}
    assert any(name.startswith("inspect/activations/") for name in logged), sorted(logged)
