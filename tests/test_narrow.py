"""The narrow copies of the parameters the forward reads (`dew.training.narrow`)."""
import contextlib
import gc
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from arm_arithmetic import summed_as_run
from map_judge import drawing
from model_support import attending
from placement_parity import layouts_agree, x64

from dew.checkpoints import _filled
from dew.nn.kernels.generation import chosen
from dew.optim import Optax, OptimConfig
from dew.training import TrainerAdvanced
from dew.training.narrow import narrowed, narrowed_paths, program_reads


def test_only_a_weight_read_once_through_its_cast_is_copied():
    """A weight whose one use is a cast to bf16 is copied, inside a jit
    and a rematerialized region too. One cast twice, as a tied or shared
    weight is, would sum its two cotangents in bf16 where they are summed
    in fp32 now; a weight read in fp32, or by a custom VJP whose backward
    may widen what it returns, is read as it is; so none of those is copied."""
    @jax.custom_vjp
    def head(x, table):
        return x @ table.astype(jnp.bfloat16).T

    head.defvjp(lambda x, table: (head(x, table), (x, table)),
                lambda saved, g: (g @ saved[1].astype(g.dtype), g.T @ saved[0]))

    def loss(params, x):
        once = jax.jit(lambda x, w: x @ w.astype(jnp.bfloat16))(x, params["once"])
        rematted = jax.checkpoint(lambda h, w: h @ w.astype(jnp.bfloat16))(once, params["rematted"])
        shared = rematted @ params["shared"].astype(jnp.bfloat16) @ params["shared"].astype(jnp.bfloat16)
        scaled = shared.astype(jnp.float32) * params["scale"]
        return jnp.sum(head(scaled.astype(jnp.bfloat16), params["table"]).astype(jnp.float32))

    params = {name: jnp.ones(shape, jnp.float32) for name, shape in
              {"once": (4, 4), "rematted": (4, 4), "shared": (4, 4), "scale": (4,), "table": (8, 4)}.items()}
    paths = narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16))
    assert paths == {("once",): jnp.dtype(jnp.bfloat16), ("rematted",): jnp.dtype(jnp.bfloat16)}


def test_a_weight_whose_other_read_the_program_discards_is_copied():
    """A weight cast once where the loss reads it, and cast again where nothing reads the cast (a selector's
    plan that declined it, the side-by-side product's of q, k and v: 0099a4701's), is read once by the
    program as it runs: it is copied, as its one live cast reads it."""
    def loss(params, x):
        params["w"].astype(jnp.bfloat16)  # a cast nothing reads
        return jnp.sum((x @ params["w"].astype(jnp.bfloat16)).astype(jnp.float32))

    paths = narrowed_paths(loss, {"w": jnp.ones((4, 4), jnp.float32)}, jnp.ones((2, 4), jnp.bfloat16))
    assert paths == {("w",): jnp.dtype(jnp.bfloat16)}


def test_a_users_primitive_named_as_a_cast_is_read_as_its_own():
    """A primitive of a user's that takes the name of JAX's cast (`convert_element_type`) and none of its
    parameters is read as its own, not the cast its name spells (an operation is known by its object): a
    weight it reads is not copied narrower, where reading its parameters as a cast's raised KeyError (row
    4094)."""
    from jax.extend import core as jcore
    from jax.interpreters import mlir

    halved = jcore.Primitive("convert_element_type")
    halved.def_impl(lambda value: value * 0.5)
    halved.def_abstract_eval(lambda value: value)
    mlir.register_lowering(halved, mlir.lower_fun(lambda value: value * 0.5, multiple_results=False))

    def loss(params, x):
        return (halved.bind(params["w"]) * x).sum()

    params = {"w": jnp.ones((4,), jnp.float32)}
    assert narrowed_paths(loss, params, jnp.ones((4,), jnp.float32)) == {}


def test_a_scan_copies_its_stacked_layers_not_its_constants():
    """A scan over stacked layers hands each iteration its own slice, whose
    cotangent it stacks, so the stack can be read through a copy; a weight
    every iteration reads is summed over the iterations in its own dtype,
    so it keeps its fp32 read."""
    def loss(params, x):
        def layer(h, w):
            return h @ w.astype(jnp.bfloat16) @ params["every"].astype(jnp.bfloat16), None
        return jnp.sum(jax.lax.scan(layer, x, params["stacked"])[0].astype(jnp.float32))

    params = {"stacked": jnp.ones((3, 4, 4), jnp.float32), "every": jnp.ones((4, 4), jnp.float32)}
    paths = narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16))
    assert paths == {("stacked",): jnp.dtype(jnp.bfloat16)}


def test_a_scan_over_a_stack_of_layers_copies_each_layer():
    """A scanned run stacks its layers' weights, each stored apart, and the scan slices and casts the
    stack: each layer's weight is read through a copy, whose stack is the stack's cast. A weight that
    is broadcast into many places sums its cotangents in its own dtype, so it keeps its fp32 read."""
    def loss(params, x):
        stacked = jnp.stack([params["first"], params["second"]])
        repeated = jnp.broadcast_to(params["repeated"], (2, 4, 4))

        def layer(h, weights):
            w, r = weights
            return h @ w.astype(jnp.bfloat16) @ r.astype(jnp.bfloat16), None
        return jnp.sum(jax.lax.scan(layer, x, (stacked, repeated))[0].astype(jnp.float32))

    params = {name: jnp.ones((4, 4), jnp.float32) for name in ("first", "second", "repeated")}
    paths = narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16))
    assert paths == {("first",): jnp.dtype(jnp.bfloat16), ("second",): jnp.dtype(jnp.bfloat16)}


def test_a_stack_of_weights_one_of_which_is_read_elsewhere_copies_none():
    """A stack takes one dtype: where one of its weights is read apart from it, in its own
    precision, none of the stack's weights is copied, and the loss traces with copies as without."""
    def loss(params, x):
        stacked = jax.lax.stack([params["first"], params["second"]])

        def layer(h, w):
            return h @ w.astype(jnp.bfloat16), None
        scanned = jax.lax.scan(layer, x, stacked)[0].astype(jnp.float32)
        return jnp.sum(scanned) + jnp.sum(params["second"])

    params = {name: jnp.ones((4, 4), jnp.float32) for name in ("first", "second")}
    assert narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16)) == {}


def test_a_stack_inside_a_call_whose_weight_is_read_outside_it_copies_none():
    """Where a stack sits inside a call and one of its weights is also read outside the call, the
    weight is read more than once, so no weight of the stack is copied."""
    joined = jax.jit(lambda a, b: jax.lax.stack([a, b]).astype(jnp.bfloat16))

    def loss(params, x):
        return jnp.sum(joined(params["a"], params["b"]).astype(jnp.float32)) + jnp.sum(params["b"])

    params = {name: jnp.ones((4, 4), jnp.float32) for name in ("a", "b")}
    assert narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16)) == {}


def test_a_call_given_a_literal_beside_a_weight_still_copies_the_weight():
    """A call may take a constant beside a weight; the weight it casts once is copied."""
    scaled = jax.jit(lambda w, s: jnp.sum(w.astype(jnp.bfloat16)) * s)

    def loss(params, x):
        return scaled(params["w"], 2.0).astype(jnp.float32)

    params = {"w": jnp.ones((4, 4), jnp.float32)}
    assert narrowed_paths(loss, params, jnp.ones((2, 4), jnp.bfloat16)) == {("w",): jnp.dtype(jnp.bfloat16)}


def _keeping_gradients(optimizer):
    """`optimizer` after a transformation whose state is the last gradient
    the update read, so a run's state holds its gradient as committed."""
    kept = optax.GradientTransformation(lambda params: jax.tree.map(jnp.zeros_like, params),
                                        lambda gradient, state, params=None: (gradient, gradient))
    return optax.chain(kept, optimizer)


def keeping_sgd(learning_rate) -> optax.GradientTransformation:
    """SGD at `learning_rate` keeping the last gradient as its state (`_keeping_gradients`): the optimizer
    `KEEPING_SGD` states by its import path."""
    return _keeping_gradients(optax.sgd(learning_rate))


ADAM_2, ADAM_3 = (OptimConfig(optimizer=Optax('optax:adam'), lr=rate) for rate in (1e-2, 1e-3))
"""Optax's Adam at 1e-2 and 1e-3, as configs, which key the step."""

SGD = OptimConfig(optimizer=Optax('optax:sgd'), lr=0.5)
"""Optax's SGD at 0.5, as a config, which keys the step."""

KEEPING_SGD = OptimConfig(optimizer=Optax('test_narrow:keeping_sgd'), lr=0.5)
"""SGD at 0.5 keeping its last gradient, stated as a config, which keys the step (`Trainer._step_key`): every
draw's fits of one shape read the step the first traced and planned, where each built its own (row 5774:
61 plans and compiles, 542 s of a profiled 605)."""


def _decoder(*, scanned=False, remat=None, mixture=False, rows=8, dex=False, seed=0, dtype=jnp.bfloat16):
    """`_trained`'s tied decoder over fp32 parameters, computing in `dtype` (bf16), as its LM objective,
    and the batch of `rows` rows `seed` draws for it to train on; with `dex`, its first layer's two heads a
    differential pair whose lambda anneals over its first ten updates, the whole decoder training beside
    it."""
    from dew.batch import TEXT_KEY
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.objectives.routing import ExpertBalance
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=4 if scanned else 2, num_heads=2,
                              num_kv_heads=1, mlp="swiglu", mlp_features=64, max_seq_len=16, qk_norm=True,
                              tie_embeddings=True, dtype=dtype, remat=remat,
                              mixture={"experts": 4, "top_k": 2} if mixture else None)
    if dex:
        from dew.nn.dex import DifferentialExtension

        model = model.with_dex(DifferentialExtension(((0, (0, 1)),), anneal_steps=10, trains="all"))
    objective = LMObjective(model, seq_len=8, pad_id=0,
                            balance=ExpertBalance(aux_loss_alpha=.01) if mixture else None)
    tokens = np.random.default_rng(seed).integers(1, 64, (rows, 9))
    return objective, {TEXT_KEY: jnp.asarray(tokens, jnp.int32)}


def _trained(optimizer, *, copies, steps=3, mesh=None, padded=0, accumulation=1, split="auto",
             mixture=False, rows=8, ragged=False, scanned=False, remat=None, longer=None, dex=False, seed=0,
             observe=None):
    """A tied bf16 decoder over fp32 parameters, trained `steps` steps on
    one batch and then `padded` steps on a batch of padding alone, with
    `copies` on or off on this device's generation, on `mesh` with every
    parameter split where it divides, `accumulation` steps an update of `split`
    microbatches each, over `rows` rows. A `mixture` routes its MLPs over
    four experts under a balance loss. A `ragged` run's even steps end
    every other row in padding, so no two neighbouring microbatches count
    as many tokens, whether a window reads steps or a step's parts. A
    `scanned` decoder of four layers runs them as one scan over their
    stacked weights, each layer's stored apart, and each block recomputes
    as `remat` names. With `longer`, every other step reads a batch of that
    many rows, whose shape the trainer plans apart. A `dex` decoder's first
    layer is a differential pair (`_decoder`). `seed` draws the batch and the
    start; a float64 process trains a float64 decoder (a check-1 truth,
    `_on_par_with_the_plain_update`). `observe(objective, state, batch)` sees
    each step's state and batch before the step. Returns the state, each
    step's loss and the trainer, its plan the last shape's."""
    from dew.batch import TEXT_KEY
    from dew.training import Layout, MeshSpec, Trainer
    wide = x64()
    objective, batch = _decoder(scanned=scanned, remat=remat, mixture=mixture, rows=rows, dex=dex, seed=seed,
                                dtype=jnp.float64 if wide else jnp.bfloat16)
    long = longer and {TEXT_KEY: jnp.asarray(np.random.default_rng(1).integers(1, 64, (longer, 9)),
                                             jnp.int32)}
    # A `scanned` decoder's stack scans, any other's runs a layer at a time (`generation.Choice.LAYER_SCAN`).
    with chosen(narrow_copies='on' if copies else 'off', layer_scan='scanned' if scanned else 'plain'):
        trainer = Trainer(objective, optimizer, key=jax.random.key(seed), mesh=mesh or MeshSpec(),
                          advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0),
                                                   param_dtype="float64" if wide else None),
                          accumulation=accumulation,
                          split=split)
        # A configured optimizer built over the run's updates, as a fit builds it.
        trainer.scheduled((steps + padded) // accumulation)
        state = trainer.initial_state()
        steps_by = {False: (trainer.compile(state, batch), None)}
        if long:
            steps_by[True] = (trainer.compile(state, long), long)
    shorter = {TEXT_KEY: batch[TEXT_KEY].at[::2, -3:].set(0)}
    trained = [shorter if ragged and index % 2 == 0 else batch for index in range(steps)]
    losses = []
    for index, current in enumerate(trained + [{TEXT_KEY: jnp.zeros_like(batch[TEXT_KEY])}] * padded):
        step, read = steps_by[bool(long) and index % 2 == 1]
        if observe is not None:
            observe(objective, state, current if read is None else read)
        state, loss, *_ = step(state, current if read is None else read)
        losses.append(np.asarray(loss))
    return state, losses, trainer


def test_a_phase_planned_afresh_builds_its_window_over_the_outgoing_one():
    """At an update's end, a phase of other passes planned afresh holds its state as a fresh one does
    (`freshly_held`): the narrow copies the outgoing step made are released, their buffers deleted, so the
    devices hold none of them while the new step is planned. Its empty window it keeps, and the step of
    another split builds its own over it (`laid_out`, `Trainer._window_of`): each of its leaves of a new
    leaf's shape donated, the new window's buffer the old one's, so none of them is allocated beside it. A
    Qwen3.5-4B's third phase on a GH200, its window built afresh beside the old one's 16.7 GiB, was refused
    8.26 GiB short of one block with 53.6 GiB free in all (M-MC-3q)."""
    from dew.state import taken
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.compiler import freshly_held

    device = MeshSpec().build(jax.devices()[:1])
    state, _, trainer = _trained(optax.sgd(0.5), copies=True, split=2, rows=16, mesh=device)
    copies = jax.tree.leaves(state.compute)
    assert copies and state.window is not None
    fresh = freshly_held(state)
    assert fresh.compute is None and all(leaf.is_deleted() for leaf in copies)
    assert fresh.window is state.window
    donor = jax.tree.leaves(fresh.window)
    gradient = {(leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(fresh.window.gradient)}
    shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), fresh.window)
    assert gradient
    finer = Trainer(trainer.objective, optax.sgd(0.5), key=jax.random.key(0), mesh=device,
                    advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)), split=4)
    batch = _decoder(rows=16)[1]
    stepped, *_ = finer.compile(fresh, batch)(fresh, batch)
    assert int(stepped.window_size) == 4
    # Every leaf of the outgoing window went to the new one's build, donated, or was let go; the leaves the
    # new window allocated afresh are its split-sized parts alone, none of the gradient's shapes.
    assert all(leaf.is_deleted() for leaf in donor)
    built, _ = taken(shapes, stepped.window)
    assert not any((leaf.shape, leaf.dtype) in gradient for leaf in built)


def test_a_phase_planned_afresh_is_planned_at_its_own_split_past_the_window_it_holds():
    """The empty window a phase planned afresh holds binds no split (`StepCompiler.afresh`,
    `resumed_stepping`): a run that leaves its split to the planner plans the new phase's step at its own,
    one microbatch where the batch fits whole, and lets the outgoing split-2 window go, where the window
    bound it to two (a Qwen3.5-4B's third phase on a GH200 searched 2 x 2 alone and was refused, M-MC-3u)."""
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.compiler import freshly_held

    device = MeshSpec().build(jax.devices()[:1])
    state, _, trainer = _trained(optax.sgd(0.5), copies=True, split=2, rows=16, mesh=device)
    assert int(state.window_size) == 2
    fresh = freshly_held(state)
    donor = jax.tree.leaves(fresh.window)
    planner = Trainer(trainer.objective, optax.sgd(0.5), key=jax.random.key(0), mesh=device,
                      advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)))
    planner._compiler.afresh = True  # as the boundary sets it (`Trainer._compiled_for`)
    batch = _decoder(rows=16)[1]
    stepped, *_ = planner.compile(fresh, batch)(fresh, batch)
    assert planner.plan["split"] == 1 and stepped.window is None
    assert all(leaf.is_deleted() for leaf in donor)


def _on_par_with_the_plain_update(optimizer, label: str, **options):
    """`_trained`'s run through copies on par with the same run without them, its parameters, optimizer
    state (the last gradient, `_keeping_gradients`) and losses, over the starts and batches the seeds draw
    (check 1), the truth the run without copies in float64: the two compile two programs. A mixture's draws
    clear its routing margin (`selection_margins.clearing`, testing.md's selection rule)."""
    def run(copies, seed):
        state, losses, _ = _trained(optimizer(), copies=copies, seed=int(seed), **options)
        return {"state": (state.variables, state.opt_state), "losses": losses}

    def drawn(seed):
        return (np.int32(seed),)

    def clears(inputs):
        """Whether a mixture's run routes with a margin (`selection_margins.MARGIN`) at every step: the
        parameters each step reads and the batch it reads, through the training forward and the inference
        one, over the plain run. A top-2 flip between the arms and the float64 truth decides by rounding,
        which no precision bounds."""
        import selection_margins
        from arm_arithmetic import summed_as_run

        from dew.algebra import passes
        from dew.batch import TEXT_KEY
        from dew.objectives.base import model_rngs

        margins = []

        def observe(objective, state, current):
            scored = jax.jit(lambda tree, tokens, train: objective.token_scores(
                tree, tokens, routing=True, train=train,
                rngs=model_rngs(jax.random.key(0), training=train)).routing, static_argnums=2)
            variables = jax.device_get(state.variables)
            for train in (True, False):
                sown = jax.device_get(scored(variables, current[TEXT_KEY], train))
                margins.append(selection_margins.routing_margin(sown, variables.get("moe", {}), 2))

        # Under the reference arm's arithmetic (`summed_as_run`, the algebra pass off), so the steps it
        # checks are the ones the reference runs.
        with summed_as_run(), passes(on=False):
            _trained(optimizer(), copies=False, seed=int(inputs[0]), observe=observe, **options)
        return min(margins) >= selection_margins.MARGIN

    if options.get("mixture"):
        import selection_margins

        drawn = selection_margins.clearing(drawn, clears)
    layouts_agree(run, drawn, label, layout=True, reference=False)


def _copies_of(narrow):
    """`narrow`'s copies, each split as its parameter is, so a gather moves bf16, and the cast of the
    parameters the last update wrote, bit for bit: one exactly rounded cast of the same values. Returns
    the copies' paths under `params`."""
    masters = {tuple(key.key for key in path): leaf.sharding
               for path, leaf in jax.tree_util.tree_flatten_with_path(narrow.variables)[0]}
    paths = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(narrow.compute)[0]:
        keys = tuple(key.key for key in path)
        assert leaf.sharding.is_equivalent_to(masters[keys], leaf.ndim), keys
        paths[keys[1:]] = leaf.dtype
    for a, b in zip(jax.tree.leaves(narrow.compute), jax.tree.leaves(narrowed(narrow.variables, paths)),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    return paths


# Each mesh a site of its own, its map tasks compiling its arms alone (one site of three took 141 s a task).
@pytest.mark.parametrize("fsdp", [
    pytest.param(1, id="1", marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(2, id="2", marks=[pytest.mark.mesh(devices=2),
                                   pytest.mark.check1_shards(seeds=8, per_task=4)]),
    pytest.param(8, id="8", marks=[pytest.mark.mesh(devices=8),
                                   pytest.mark.check1_shards(seeds=8, per_task=4)]),
])
def test_the_tied_decoder_reads_its_kernels_through_copies_with_the_same_gradients(fsdp):
    """Through the trainer, a tied decoder's projection kernels are read
    through bf16 copies and its tied table and norm scales are not. Under
    plain SGD, whose update reads the gradient once, every parameter and the
    last gradient after three steps are on par with what the in-forward cast
    gives (check 1): the copies change no gradient, and the tied table's two
    uses still sum in fp32. So too where the batch is split over data-parallel
    devices, whose gradients are summed across them, and where every
    parameter is split over fsdp devices too (the lane's eight: data 8,
    data 4 x fsdp 2, fsdp 8)."""
    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=fsdp)
    _on_par_with_the_plain_update(lambda: KEEPING_SGD, "a tied decoder's copies",
                                  mesh=mesh)
    if drawing():
        return
    narrow, _, _ = _trained(KEEPING_SGD, copies=True, mesh=mesh)
    names = {"/".join(path) for path in _copies_of(narrow)}
    assert names and all(name.endswith("kernel") and "proj" in name for name in names), names


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_scanned_decoder_reads_every_layers_kernels_through_copies_with_the_same_gradients():
    """A decoder that scans its layers over their stacked weights reads each layer's projection
    kernels through bf16 copies, stacked as the scan reads them, as an unscanned decoder does: under
    plain SGD the parameters, the last gradient and the losses are on par with the in-forward cast's
    (check 1)."""
    _on_par_with_the_plain_update(lambda: KEEPING_SGD, "a scanned decoder's copies",
                                  scanned=True)
    narrow, _, _ = _trained(KEEPING_SGD, copies=True, scanned=True)
    names = {"/".join(path) for path in _copies_of(narrow)}
    layers = {name.split("/")[0] for name in names}
    assert {f"layers_{index}" for index in range(4)} <= layers, names


# Each cell a site of its own, its map tasks compiling its arms alone: one site over the six, each task
# compiling every cell's arms for its draws, took 470 to 560 s a task in CI, setting each train's wall
# (rows 6237, 6332, 4765). On armada a pooled cell's four draws take 50 to 80 s a task, a replayed (mixture)
# cell's two draws about 100, each clearing its routing margin first.
@pytest.mark.parametrize(("accumulation", "split", "mixture", "fsdp"), [
    pytest.param(2, 1, False, 1, id="accumulated-pooled",
                 marks=pytest.mark.check1_shards(seeds=8, per_task=4)),
    pytest.param(1, 2, True, 1, id="split-replayed", marks=pytest.mark.check1_shards(seeds=8, per_task=2)),
    pytest.param(2, 1, True, 2, id="accumulated-replayed-fsdp2",
                 marks=[pytest.mark.mesh(devices=2), pytest.mark.check1_shards(seeds=8, per_task=2)]),
    pytest.param(1, 2, False, 2, id="split-pooled-fsdp2",
                 marks=[pytest.mark.mesh(devices=2), pytest.mark.check1_shards(seeds=8, per_task=4)]),
    pytest.param(2, 1, False, 8, id="accumulated-pooled-fsdp8",
                 marks=[pytest.mark.mesh(devices=8), pytest.mark.check1_shards(seeds=8, per_task=4)]),
    pytest.param(1, 2, True, 8, id="split-replayed-fsdp8",
                 marks=[pytest.mark.mesh(devices=8), pytest.mark.check1_shards(seeds=8, per_task=2)]),
])
def test_a_window_of_microbatches_reads_its_copies_and_commits_the_plain_update(
        accumulation, split, mixture, fsdp):
    """A step that pools several microbatches, by accumulation or a batch
    read in parts, reads the copies through its window and rewrites them as
    it commits. Each microbatch's gradient is widened as it is pooled, so
    weighing microbatches of unequal token counts into one mean rounds in
    fp32 as without copies, and a loss whose balance terms pool apart (a
    mixture's) replays the window's microbatches through the copies too:
    under SGD every parameter and the last pooled gradient after two
    windows are on par with the plain run's (check 1), with the parameters and their
    copies split over fsdp devices too (pairwise over the window's kind,
    its pooling and the mesh)."""
    from dew.training import MeshSpec

    # Each microbatch keeps a row for every row shard of the lane's devices.
    run = {"steps": 2 * accumulation, "accumulation": accumulation, "split": split, "mixture": mixture,
           "rows": split * max(8, jax.device_count()), "ragged": True, "mesh": MeshSpec(fsdp=fsdp)}
    _on_par_with_the_plain_update(lambda: KEEPING_SGD, "a window's copies", **run)
    if drawing():
        # A map task records its draws; the judge holds the run's copies.
        return
    narrow, _, _ = _trained(KEEPING_SGD, copies=True, **run)
    assert int(narrow.updates) == 2
    assert _copies_of(narrow)


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_dex_decoder_through_its_copies_trains_as_it_does_without_them():
    """A decoder whose differential lambda anneals over the run's updates reads, in training, a float32
    lambda the anneal computes on before its cast to bfloat16, which outside training (no updates) it
    casts at once: the copies, chosen off the loss the step runs, its committed updates among its
    inputs, leave lambda out, so through them every parameter, the last gradient and every loss over
    three updates are on par with a plain run's (check 1)."""
    _on_par_with_the_plain_update(lambda: KEEPING_SGD, "a dex decoder's copies",
                                  steps=3, dex=True)
    narrow, _, _ = _trained(KEEPING_SGD, copies=True, steps=3, dex=True)
    assert int(narrow.updates) == 3
    assert _copies_of(narrow)


def test_a_skipped_update_keeps_the_copies_of_the_parameters_it_kept():
    """A batch of padding alone has nothing to predict, so its step commits
    no update: the parameters stay as the last update wrote them, and so do
    their copies. Adam's moments would have moved every parameter had the
    update been taken, so copies cast from the update the step discarded
    would differ from the parameters it kept."""
    trained, *_ = _trained(ADAM_2, copies=True, steps=2)
    skipped, *_ = _trained(ADAM_2, copies=True, steps=2, padded=1)
    assert int(skipped.updates) == int(trained.updates) == 2
    for a, b in zip(jax.tree.leaves(trained.variables), jax.tree.leaves(skipped.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    paths = {tuple(key.key for key in path)[1:]: leaf.dtype
             for path, leaf in jax.tree_util.tree_flatten_with_path(skipped.compute)[0]}
    assert paths
    for a, b in zip(jax.tree.leaves(skipped.compute), jax.tree.leaves(narrowed(skipped.variables, paths)),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_fit_check_counts_the_copies_the_first_step_builds(monkeypatch):
    """The fit check reads the state's arguments as already on the devices;
    the copies are not yet when the first step compiles, since `run` builds
    them before its first call, so their bytes count beside the step with the
    prefetched batches: the headroom the check leaves is the step's own,
    `step_headroom` of the devices' free bytes, less the bytes held beside it."""
    from dew.distributed.placement import placed_bytes
    from dew.training.memory import step_headroom

    limit = 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: {"bytes_limit": limit, "bytes_in_use": 0})
    held = []

    def trained(*, copies):
        state, _, trainer = _trained(ADAM_3, copies=copies, steps=1)
        devices = [device for device in trainer.device_mesh.devices.flat
                   if device.process_index == jax.process_index()]
        held.append(step_headroom(trainer.executable, devices) - trainer.plan["probes"][0]["headroom"])
        return state

    narrow = trained(copies=True)
    trained(copies=False)
    copies = placed_bytes(jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), narrow.compute))
    assert copies > 0 and held[0] - held[1] == copies


def test_a_step_that_fits_no_recomputation_with_its_copies_drops_them_where_that_frees_bytes(monkeypatch):
    """A tight start asks each remat rung with its copies first (`FitSearch.recomputing`, row 6740), then,
    where none fits, drops the copies at its own rung and keeps them dropped where the step without them falls
    less short; where it falls no less short, the copies come back, their 1-2% of a step kept, and the ladder
    climbs with them. The plan says which (`narrow`). In the geometry of row 1661's run, a window of two
    steps, the devices hold the step fully recomputed without its copies and not with them. On a CPU the drop
    frees bytes, the step's casts fused into their uses; on a GPU the step casts its weights whole either way
    (a GH200: 68.352 MB without its copies, 68.344 MB with them, rows 5041, train 857 F2)."""
    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})

    def trained(*, copies, remat):
        return _trained(ADAM_3, copies=copies, steps=1, accumulation=2, split=1, remat=remat)

    _, _, recomputed = trained(copies=False, remat="full")
    free -= recomputed.plan["probes"][0]["headroom"]
    state, _, trainer = trained(copies=True, remat=None)
    probes = trainer.plan["probes"]
    dropped = next(index for index, probe in enumerate(probes) if not probe["narrow"])
    first, second = probes[0], probes[dropped]
    # Every remat rung asked with the copies, and missed, before they are dropped at the start's rung.
    assert all(probe["narrow"] and not probe["fits"] for probe in probes[:dropped]), probes
    assert len({probe["remat"] for probe in probes[:dropped]}) > 1, probes
    assert second["remat"] == first["remat"] and probes[-1]["fits"] is True
    # On a CPU the step without them falls less short, and they stay dropped; on a GPU it falls no less
    # short, and they come back.
    frees = jax.default_backend() == "cpu"
    assert (second["headroom"] > first["headroom"]) is frees
    assert trainer.plan["narrow"] is not frees
    assert (state.compute is None) is frees


def test_a_fit_whose_state_fits_but_not_with_its_copies_trains_without_them(monkeypatch):
    """Row 1661's run (1765): devices that hold the state a run keeps between steps, its parameters,
    optimizer state and the window's accumulator, and that `validate` reports the same with the copies on
    as off: the copies are the plan's first rung, dropped where the step does not fit with them. Where the
    step's measured need without its copies is below its need with them (a CPU's), devices whose free
    bytes lie between the two train it without them, at its own rung; where dropping them frees nothing
    (a GPU's step, whose temporaries hold its fusions' casts either way: GH200 68.352 MB against 68.344 MB,
    row 6310), devices short of the need with them keep the copies and climb the ladder (row 8562)."""
    from model_support import allocating
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.distributed.placement import placed_bytes
    from dew.telemetry.compiles import Compiles
    from dew.training import Layout, MeshSpec, Trainer

    # Every device an allocator counting what the run places on it, its copies as its search builds them
    # (`Building.fits`), `free` bytes free before the run: a need measured so holds all of the run's.
    free, limit = 2 ** 39, 2 ** 40

    def built() -> tuple[Trainer, dict]:
        objective, batch = _decoder(remat="full")
        return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(),
                       advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)), accumulation=2,
                       split=1), batch

    def streamed() -> Loader:
        return Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)

    def need(*, copies: bool) -> int:
        """The bytes the run's fit needs placed with or without its copies (its state and the batches it
        prefetches among them), as its plan measured them: each run apart, nothing of the one before it
        live."""
        gc.collect()
        allocating(monkeypatch, limit, limit, limit - free)
        run, _ = built()
        with chosen(narrow_copies='on' if copies else 'off'):
            run.fit(streamed(), steps=1)
        return free - run.plan["probes"][0]["headroom"]

    with_copies, without = need(copies=True), need(copies=False)
    narrow, _, _ = _trained(ADAM_3, copies=True, steps=1, accumulation=2, split=1, remat="full")
    copies = placed_bytes(jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), narrow.compute))
    del narrow

    def held(*, copies: bool) -> int:
        run, batch = built()
        with chosen(narrow_copies='on' if copies else 'off'):
            return max(run.validate(batch).device_bytes.values())

    state = held(copies=False)
    assert copies > 0 and held(copies=True) == state
    run, _ = built()
    # Devices free, before the run places anything, between the step's needs with and without its copies,
    # as each plan measured them (the state among them): the state fits (`validate` checks it alone against
    # the limit), the step fits only without its copies. Both stated from the needs the cell measured, not
    # the state's bytes, which a GPU step's temporaries dwarf (M-MC-22: a 349,744 B state, a 68 MB step).
    # The plan made before the state is placed (`Trainer._planned_unplaced`), which reads those bytes as
    # holding the state too, holds none, and the step is planned once the state is placed.
    frees = without < with_copies
    free = (with_copies + without) // 2 if frees else with_copies - 1
    assert state < free < with_copies, (without, free, with_copies, state)
    gc.collect()
    allocating(monkeypatch, limit, limit, limit - free)
    stream = streamed()
    with chosen(narrow_copies='on'):
        run.fit(stream, steps=2)
    probes = run.plan["probes"]
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults")
    own = {key: probes[0][key] for key in rung} == {key: run.plan[key] for key in rung}
    assert probes[-1]["fits"] is True and not probes[0]["fits"]
    # Where dropping them freed what it was short, it trains without them at its own rung, nothing
    # recomputed for it; where it frees nothing, they come back and the ladder climbs past its own rung.
    assert (run.plan["narrow"], own) == ((False, True) if frees else (True, False)), run.plan
    # A pool grown to the state's size, half of it one free block, whose limit holds the state and the step
    # with its copies in all, the step in what the pool can still take: a fresh GH200 4B run's (M-MC-3m).
    # The step is planned before the state is placed on the pool's bytes in all, not on its largest block,
    # which placing the state reshapes, and judged on the allocator once the state is on it: compiled once,
    # beside the start, not again once the state is placed. Its growth holds the step with its copies
    # twice: a BFC pool whose largest block cannot take twice the bytes placed beside the temporaries
    # gives half its growth as one block (`memory.placeable_beside`), which a GPU's step, its temporaries
    # dwarfing its state, meets (row 5040: a GH200's step of 0.03 GiB of temporaries was refused on a
    # growth that held them once).
    grown = {"bytes_limit": state + 2 * with_copies + state // 2, "bytes_in_use": 0, "pool_bytes": state,
             "largest_free_block_bytes": state // 2}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: grown)
    run, _ = built()
    with chosen(narrow_copies='on'), Compiles("step") as steps:
        run.fit(stream, steps=2)
    assert run.plan["narrow"] is True and steps.count == 1
    # Its own probe records the placed judgement whole: the headroom the placed devices hold and the bytes
    # that judgement placed beside the step, not the unplaced search's charge of the state it planned
    # before placing, which the placed step holds among its arguments (row 8962).
    # Here what it places beside the step, its outputs and its batches, is under the state's bytes.
    placed, own = run.plan["placed"], run.plan["probes"][0]
    assert own["headroom"] == placed["headroom"] and own["beside"] < state, (own, placed, state)


def test_a_fresh_plan_refused_with_its_copies_built_deletes_them_before_it_is_searched_again(monkeypatch):
    """A fresh run's first step planned with its narrow copies before its state is placed, on devices whose
    free bytes in all hold the step and its copies, and whose pool, the state and the copies on it, has no
    block for its temporaries, nor growth to give one: the placed check builds the copies where they land
    (`Building.fits`), is refused, and deletes them before the plan is searched again on the placed devices,
    whose allocator then reads the copies' bytes free (row 8766; the cells above cover the bound and the
    copies taken)."""
    from model_support import allocating, device_buffers
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.planning import PlanRefused

    device = MeshSpec().build(jax.devices()[:1])

    def built() -> Trainer:
        objective, _ = _decoder(remat="full")
        return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=device,
                       advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)))

    def streamed() -> Loader:
        return Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)

    narrow, _, _ = _trained(ADAM_3, copies=True, steps=1, remat="full", mesh=device)
    copies = sorted(leaf.nbytes for leaf in jax.tree.leaves(narrow.compute))
    del narrow
    # The fresh plan's need with its copies, the state and its batches among it, and its temporaries, as a
    # roomy allocator's run measured them.
    free, limit = 2 ** 39, 2 ** 40
    gc.collect()
    allocating(monkeypatch, limit, limit, limit - free)
    roomy = built()
    with chosen(narrow_copies="on"):
        roomy.fit(streamed(), steps=1)
    probe = roomy.plan["probes"][0]
    assert probe["narrow"] and probe["fits"] and copies, (probe, copies)
    need, temporaries = free - probe["headroom"], probe["temp"]
    del roomy
    # Free in all, the need and half the temporaries: the plan made before the state is placed holds. The
    # pool's growth one byte short of the temporaries, and its blocks, the state and copies on it, the
    # rest: half the temporaries and what is placed beside them, no block for them.
    gc.collect()
    allocating(monkeypatch, limit, limit - temporaries + 1, limit - need - temporaries // 2)
    stats, buffers = jax.Device.memory_stats, []

    def reading(device):
        # The buffers on the device at each of its allocator's reads, as the read stands.
        buffers.append(device_buffers(device))
        return stats(device)
    monkeypatch.setattr(jax.Device, "memory_stats", reading)
    run = built()
    with chosen(narrow_copies="on"), contextlib.suppress(PlanRefused):
        run.fit(streamed(), steps=1)
    # Between the judgement with the copies built and the search after it, the copies' buffers alone are
    # gone, one a leaf, the state's left: the buffers each read's next no longer holds.
    freed = [sorted(size for address, size in before.items() if address not in after)
             for before, after in itertools.pairwise(buffers)]
    assert copies in freed, (copies, [each for each in freed if each])


def test_a_fresh_plan_its_placed_devices_refuse_records_no_rung(monkeypatch, tmp_path):
    """A fresh run's first step planned before its state is placed records its rung (`keep_rung`) only once
    the devices hold it with the state placed: a pool whose limit holds the step in all and that can give
    it no block once the state is placed refuses the run, and leaves no record a later fresh run would start
    its plan from, untraced."""
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.planning import PlanRefused

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    held = {"bytes_limit": 2 ** 40, "bytes_in_use": 0, "pool_bytes": 2 ** 40, "largest_free_block_bytes": 0}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: held)
    objective, _ = _decoder(remat="full")
    run = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec().build(jax.devices()[:1]),
                  advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)))
    stream = Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)
    with chosen(narrow_copies='off'), pytest.raises(PlanRefused):
        run.fit(stream, steps=2)
    assert list(tmp_path.iterdir()) == []


def test_a_fresh_fit_whose_placed_pool_holds_its_step_in_two_microbatches_splits_it_in_two(monkeypatch):
    """A fresh run's first step is planned before its state is placed, on the devices' free bytes in all,
    and judged on their allocators once the state is on them (`StepCompiler.compile`). A pool whose limit
    holds the state and the step whole, and whose largest block holds the step in two microbatches and not
    in one, takes the unplaced plan's one microbatch, refuses it once the state is placed, and plans afresh
    at the planner's own split, not the held plan's: two microbatches, where a run on the GH200 that
    re-planned at the held plan's split was refused (M-MC-3n)."""
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.training import Layout, MeshSpec, Trainer

    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})
    # One device, whose batch of 8 rows splits in two.
    device = MeshSpec().build(jax.devices()[:1])
    one, two = (free - _trained(ADAM_3, copies=False, steps=1, split=split, remat="full",
                                mesh=device)[2].plan["probes"][0]["headroom"] for split in (1, 2))
    assert two < one, (one, two)
    objective, batch = _decoder(remat="full")
    run = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=device,
                  advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)))
    with chosen(narrow_copies='off'):
        state = max(run.validate(batch).device_bytes.values())
    # The pool's largest free block, and what it can still grow by, between the step's needs at two
    # microbatches and at one: twice what a request placed beside the step takes, so each splits it
    # (`memory.placeable_beside`).
    room = (one + two) // 2
    total = 2 * (state + one)
    grown = {"bytes_limit": total, "bytes_in_use": 0, "pool_bytes": total - room,
             "largest_free_block_bytes": room}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: grown)
    stream = Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)
    with chosen(narrow_copies='off'):
        run.fit(stream, steps=2)
    assert run.plan["split"] == 2, run.plan["probes"]


def test_a_plan_a_rung_above_that_fits_with_its_copies_keeps_them(monkeypatch):
    """A window of one step that recomputes nothing holds about as much without its copies as with them:
    its backward keeps the whole gradient, in float32 where the copies' is bfloat16. Where it fits its own
    rung neither way, the plan asks the remat rungs with its copies first (`FitSearch.recomputing`, row
    6740) and settles the first that fits, its copies kept, none dropped: once a plan fits, a run compiles
    no other."""
    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})

    def trained(*, copies):
        return _trained(ADAM_3, copies=copies, steps=1, split=1, rows=64)

    base = trained(copies=True)[2].plan["probes"][0]
    free -= max(base["headroom"], trained(copies=False)[2].plan["probes"][0]["headroom"]) + 1
    state, _, trainer = trained(copies=True)
    probes, plan = trainer.plan["probes"], trainer.plan
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults")
    # Its own rung short with its copies, then each remat rung asked with them until one fits.
    assert not probes[0]["fits"] and all(probe["narrow"] for probe in probes), probes
    assert probes[-1]["fits"] is True and plan["remat"] != base["remat"], probes
    assert {key: plan[key] for key in rung} != {key: base[key] for key in rung}
    assert plan["narrow"] is True and state.compute is not None


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_run_whose_shapes_plan_their_copies_apart_steps_each_as_its_plan_reads(monkeypatch):
    """Each shape a run meets plans its copies itself (`StepCompiler.at_base_rung`): where a longer batch
    fits only without them and a shorter one with them, the steps take turns, each reading the state as its
    own plan compiled it, the copies dropped before the longer batch's step and built again before the
    shorter's as the cast of the parameters the last update wrote (`_copies_of`), and under plain SGD the run
    lands on par with the parameters and losses of one without copies (check 1)."""
    free = 2 ** 39

    def stats(device):
        # A float64 truth's wider arrays planned in the whole memory, read from the planning thread's own
        # precision: the gate runs the truth beside the arms.
        return {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - (2 ** 39 if x64() else free)}
    monkeypatch.setattr(jax.Device, "memory_stats", stats)

    def trained(*, copies, rows=8, longer=None, steps=0):
        # Sized under the arithmetic check 1 runs its arms in (`summed_as_run`), whose float32 dots compile
        # to other footprints than the default precision's.
        with summed_as_run():
            return _trained(SGD, copies=copies, steps=steps, accumulation=2, split=1,
                            remat="full", rows=rows, longer=longer)

    def used(rows, *, copies):
        return free - trained(copies=copies, rows=rows)[2].plan["probes"][0]["headroom"]

    free = max(used(8, copies=True), used(64, copies=False))
    if not drawing():
        # The longer batch with its copies past that memory, and the run's own plan, which a map task's
        # draws need not see: its judge holds them, a map task sizing two plans where it sized three
        # (139 to 188 s a task of two draws on armada, row 6351).
        assert free < used(64, copies=True)
        narrow, _, trainer = trained(copies=True, longer=64, steps=3)
        assert not trainer.plan["narrow"] and narrow.compute is not None
        assert _copies_of(narrow)
    _on_par_with_the_plain_update(lambda: SGD, "a run whose shapes plan their copies apart",
                                  steps=3, accumulation=2, split=1, remat="full", rows=8, longer=64)


def test_a_restored_state_drops_its_templates_copies():
    """The copies are of the template's own parameters; a restored state
    starts without and the step rebuilds them from what was restored."""
    from dew.state import TrainState

    zero, one = jnp.zeros((), jnp.int32), jnp.ones((), jnp.int32)
    template = TrainState(step=zero, microstep=zero, updates=zero, variables={"params": {"w": jnp.ones(2)}},
                          opt_state=(), ema=None, key=jax.random.key(0), scale=None, window_size=one,
                          compute={"params": {"w": jnp.ones(2, jnp.bfloat16)}})
    saved = {"step": zero, "window_size": one, "variables": {"params": {"w": jnp.zeros(2)}}}
    restored = _filled(template, saved, 0)
    assert isinstance(restored, TrainState)
    assert restored.compute is None



@pytest.mark.parametrize("resident,beside,temporaries", [(0, 0, 1), (3 << 30, 1 << 30, 5 << 30),
                                                         (12345, 678, 91011)])
def test_a_gpu_of_the_memory_a_step_needs_admits_it_and_one_byte_less_does_not(resident, beside, temporaries):
    """`needed` is the inverse of `admitted`: a GPU stated to hold the bytes it returns admits the step
    (headroom zero or more, read by `admitted` through that GPU's stated memory), and one stated a byte
    fewer does not; the bytes a pool of GPUs is sized by, through the one admission the planner reads."""
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import admitted, needed

    device = jax.devices("cpu")[0]
    least = needed(device, resident, beside, temporaries)

    def headroom(memory: int) -> int:
        with stand_in_gpus([device], memory) as stood:
            spare = admitted(stood, resident, beside, temporaries)
        assert spare is not None
        return spare

    assert headroom(least) >= 0 > headroom(least - 1)


@pytest.mark.parametrize("built", ["copies", "window"])
def test_a_steps_buffers_built_before_it_runs_are_judged_where_they_land(monkeypatch, built):
    """A step whose narrow copies, or window of two microbatches, are built before its program runs, on
    devices whose allocator holds one free block that its temporaries fit before those buffers are placed
    and not after, nor in the pool's growth: its fit counts those bytes placed beside its temporaries where
    they leave the least room (`memory.placeable_beside`: in the largest free block, the one the pool can
    give its temporaries), so its first rung misses, where a reading of the block as it stood admitted it
    (M-MC-24c: an H100's 7.27 GB of copies took the one 21.64 GB block its 20.76 GB of temporaries were
    admitted to, and its first step was refused)."""
    from model_support import allocating

    from dew.distributed.batches import batch_shardings
    from dew.distributed.placement import placed_bytes
    from dew.training import MeshSpec, Trainer
    from dew.training.planning import PlanRefused, Trials

    copies, split = built == "copies", 2 if built == "window" else 1
    objective, batch = _decoder(remat="full")

    def compiled(**allocator):
        trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec().build(
            jax.devices()[:1]), split=split)
        state, _, _ = trainer.place()
        placed = jax.device_put(batch, batch_shardings(trainer.device_mesh, batch))
        allocating(monkeypatch, **allocator)
        trials = Trials(state, placed)
        with chosen(narrow_copies="on" if copies else "off"):
            step = trainer.compile(state, placed, trials)
        return trainer, step, trials, placed

    roomy, step, trials, placed = compiled(limit=2 ** 40, pool=2 ** 39)
    probe = roomy.plan["probes"][0]
    # Its buffers as its step holds them once run.
    stepped, *_ = step(trials.state, placed)
    held = stepped.compute if copies else stepped.accumulation
    buffers = placed_bytes(jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
        leaf.shape, leaf.dtype, sharding=leaf.sharding), held))
    del roomy, step, trials, placed, stepped, held
    assert probe["fits"] and buffers > 0, probe
    temporaries = probe["temp"]
    # One block holding the temporaries and half the buffers, the pool's growth less than the temporaries:
    # the buffers built, no block holds the temporaries.
    pool = temporaries + buffers // 2
    try:
        tight = compiled(limit=pool + temporaries - 1, pool=pool)[0]
        first = tight.plan["probes"][0]
    except PlanRefused as refused:
        first = refused.probes[0].record()
    assert first["fits"] is False, first


def test_a_steps_reads_of_its_parameters_pass_over_another_collections_attribute_leaves():
    """A state whose variables hold, beside `params`, collections of structured leaves sorted before and
    after it (a packed checkpoint's `quantization`, an NvFp4 weight's attributes) has its step's reads of
    `params` read, each leaf bound to its own input: its one cast planned, its copy built from the plan and
    held as planned (`narrowed`, `held_copies`); the other collections' leaves are named nowhere, and a
    parameter reached through an attribute is planned no copy, which no copy's overlay reaches (row 11730:
    the attribute paths raised before a default GPU fit's first step)."""
    from typing import NamedTuple

    from dew.training.narrow import held_copies

    class Packed(NamedTuple):
        """Leaves reached through attributes, as a packed weight's are."""
        values: jax.Array
        scale: jax.Array

    def packed():
        return Packed(jnp.ones((4, 2), jnp.uint8), jnp.ones((4,), jnp.float32))

    variables = {"packing": {"head": packed()},
                 "params": {"w": jnp.ones((4, 4), jnp.float32), "b": jnp.ones((4,), jnp.float32),
                            "attribute": packed()},
                 "quantization": {"head": packed()}}

    def loss(variables, x):
        params = variables["params"]
        others = variables["packing"]["head"].scale + variables["quantization"]["head"].scale
        return jnp.sum(x @ params["w"].astype(jnp.bfloat16)) + jnp.sum(params["attribute"].scale + others)

    reads = program_reads(jax.make_jaxpr(loss)(variables, jnp.ones((2, 4), jnp.bfloat16)), variables,
                          ("params",))
    assert reads.casts == {("w",): jnp.dtype(jnp.bfloat16)} and reads.read == {("w",)}, reads
    copies = narrowed(variables, reads.casts)
    assert held_copies(copies) == reads.casts
    np.testing.assert_array_equal(np.asarray(copies["params"]["w"]),
                                  np.asarray(variables["params"]["w"].astype(jnp.bfloat16)))


def test_a_fit_over_a_state_holding_a_packed_collection_of_attribute_leaves_steps_with_its_copies():
    """A fit whose state holds, beside its parameters, a collection of structured leaves its step reads no
    copy of (a packed checkpoint's `quantization`, an NvFp4 weight's attributes), its narrow copies on, plans
    and steps, and the step it runs builds copies of its parameters alone, the collection's leaves held
    through it as they were (row 11730: its first step raised on those attribute paths before it ran; row
    11779: its proof read the analysis alone)."""
    from typing import NamedTuple

    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, MeshSpec, Trainer

    class Packed(NamedTuple):
        """Leaves reached through attributes, as a packed weight's are."""
        values: jax.Array
        scale: jax.Array

    plain, _ = _decoder()
    variables = dict(plain.init(jax.random.key(0)))
    variables["quantization"] = {"head": Packed(jnp.ones((4, 2), jnp.uint8), jnp.ones((4,), jnp.float32))}
    objective = LMObjective(plain.model, seq_len=8, pad_id=0, variables=variables)
    with chosen(narrow_copies="on"):
        trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(),
                          advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)))
        state = trainer.fit(Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None,
                                   records=None, batch=8), steps=1)
    assert int(state.step) == 1
    # The step a fit runs, called as compiled, returns the state its copies are built in (a fit's own state
    # holds none past its end): copies of its parameters alone, the packed collection's two leaves held
    # through the step as they were.
    _, batch = _decoder()
    with chosen(narrow_copies="on"):
        stepped, *_ = trainer.compile(state, batch)(state, batch)
    copied = {path[0].key for path, _ in jax.tree_util.tree_flatten_with_path(stepped.compute or {})[0]}
    assert copied == {"params"}, copied
    held = stepped.variables["quantization"]["head"]
    np.testing.assert_array_equal(np.asarray(held.values), np.ones((4, 2), np.uint8))
    np.testing.assert_array_equal(np.asarray(held.scale), np.ones(4))


@attending("template")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_a_forwards_parameter_reads_are_its_drawn_forwards(dtype):
    """A decoder's forward whose every launch taking a parameter runs a template where a GPU runs it (the
    GEMM template's projections, the fused RMS norm's scale, SwiGLU's fused product, the attention template,
    a mixture's grouped matmul; its fp32 parameters computed in 16 bits, an fp32 product none of them takes),
    and its language model's loss over it (`Objective.scalar_loss`, its head's launches among the loss's),
    each traced as an H100 builds it, read their parameters as they read them drawn (`generation.drawing`:
    every selector its generic program, no launch derived, traced or certified), each parameter's one
    narrowing cast and every read alike (`program_reads`), and return the same shapes: so the loss's shape
    trace a plan's narrow copies read (`Trainer._loss_shape`) is drawn, and the step's own trace derives,
    traces and certifies its launches once (row 11803: the trace made them twice, a third of an every-template
    ahead compile's main thread)."""
    from dew.nn import programs
    from dew.nn.kernels import generation
    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective
    from dew.registry import models

    model = models.build("causal_transformer", vocab_size=64, emb_features=128, num_layers=2, num_heads=2,
                         num_kv_heads=1, head_dim=64, mlp_features=256, max_seq_len=128, dtype=dtype,
                         mixture={"experts": 4, "top_k": 2, "layers": (1,)})
    tokens = jax.ShapeDtypeStruct((2, 128), jnp.int32)
    variables = jax.eval_shape(model.init, jax.random.key(0), tokens)
    objective, step = LMObjective(model, 127), Step(jnp.asarray(0), jax.random.key(1), None)

    def traced(function, operand, *, drawn: bool):
        templates = chosen(grouped_matmul="mosaic", rms_norm="pallas", swiglu="pallas", attention="template")
        with (generation.assumed("sm90"), templates,
              generation.drawing() if drawn else contextlib.nullcontext()):
            closed, shapes = jax.make_jaxpr(function, return_shape=True)(variables, operand)
        launches = [equation for equation in programs.equations(closed.jaxpr)
                    if equation.primitive in programs.NATIVE_KERNELS]
        return program_reads(closed, variables, ("params",)), shapes, launches

    def loss(variables, batch):
        return objective.scalar_loss(variables, batch, step)

    for function, operand in ((model.apply, tokens), (loss, {"text": tokens})):
        reads, shapes, launches = traced(function, operand, drawn=False)
        drawn_reads, drawn_shapes, drawn_launches = traced(function, operand, drawn=True)
        assert launches and not drawn_launches, (function, len(launches), len(drawn_launches))
        assert drawn_reads == reads and drawn_shapes == shapes, (function, reads, drawn_reads)
