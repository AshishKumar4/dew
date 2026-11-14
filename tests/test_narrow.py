"""The narrow copies of the parameters the forward reads (`dew.training.narrow`)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from arm_arithmetic import summed_as_run
from placement_parity import layouts_agree, x64

from dew.checkpoints import _filled
from dew.nn.kernels.generation import chosen
from dew.optim import OptimConfig
from dew.training.narrow import narrowed, narrowed_paths


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


ADAM_2, ADAM_3 = (OptimConfig(optimizer="optax:adam", learning_rate=rate) for rate in (1e-2, 1e-3))
"""Optax's Adam at 1e-2 and 1e-3, as configs, which key the step."""

SGD = OptimConfig(optimizer="optax:sgd", learning_rate=0.5)
"""Optax's SGD at 0.5, as a config, which keys the step."""

KEEPING_SGD = OptimConfig(optimizer="test_narrow:keeping_sgd", learning_rate=0.5)
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
                              tie_embeddings=True, dtype=dtype, scan_layers=scanned, remat=remat,
                              mixture={"experts": 4, "top_k": 2} if mixture else None)
    if dex:
        from dew.nn.dex import DifferentialExtension

        model = model.with_dex(DifferentialExtension(((0, (0, 1)),), anneal_steps=10, trains="all"))
    objective = LMObjective(model, seq_len=8, pad_id=0,
                            balance=ExpertBalance(aux_loss_alpha=.01) if mixture else None)
    tokens = np.random.default_rng(seed).integers(1, 64, (rows, 9))
    return objective, {TEXT_KEY: jnp.asarray(tokens, jnp.int32)}


def _trained(optimizer, *, copies, steps=3, mesh=None, padded=0, accumulation=1, split=None,
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
    with chosen(narrow_copies='on' if copies else 'off'):
        trainer = Trainer(objective, optimizer, key=jax.random.key(seed), mesh=mesh or MeshSpec(),
                          layout=Layout(min_shard=1, tolerance=1.0), accumulation=accumulation, split=split,
                          param_dtype="float64" if wide else None)
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
                    layout=Layout(min_shard=1, tolerance=1.0), split=4)
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
                      layout=Layout(min_shard=1, tolerance=1.0))
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


@pytest.mark.check1_shards(seeds=8, per_task=2)
@pytest.mark.parametrize("fsdp", [1, pytest.param(2, marks=pytest.mark.mesh(devices=2)),
                                  pytest.param(8, marks=pytest.mark.mesh(devices=8))])
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


@pytest.mark.check1_shards(seeds=8, per_task=2)
@pytest.mark.parametrize(("accumulation", "split", "mixture", "fsdp"), [
    (2, 1, False, 1), (1, 2, True, 1),
    pytest.param(2, 1, True, 2, marks=pytest.mark.mesh(devices=2)),
    pytest.param(1, 2, False, 2, marks=pytest.mark.mesh(devices=2)),
    pytest.param(2, 1, False, 8, marks=pytest.mark.mesh(devices=8)),
    pytest.param(1, 2, True, 8, marks=pytest.mark.mesh(devices=8)),
], ids=["accumulated-pooled", "split-replayed", "accumulated-replayed-fsdp2", "split-pooled-fsdp2",
        "accumulated-pooled-fsdp8", "split-replayed-fsdp8"])
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


def test_a_step_that_fits_only_without_its_copies_drops_them_before_it_recomputes(monkeypatch):
    """The narrow copies are the fit ladder's first rung. In the geometry of row 1661's run, a window of
    two steps under full recomputation, the backward adds each gradient into the window's sum as it is
    made, so the copies, two bytes a parameter held all run, are bytes the step gives back: one that
    does not fit with them drops them before it recomputes more, and its plan says so
    (`narrow`), on devices whose free bytes lie between its footprints with and without them."""
    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})

    def trained(*, copies):
        return _trained(ADAM_3, copies=copies, steps=1, accumulation=2, split=1, remat="full")

    def used(*, copies):
        return free - trained(copies=copies)[2].plan["probes"][0]["headroom"]

    with_copies, without = used(copies=True), used(copies=False)
    assert without < with_copies
    free = (with_copies + without) // 2
    state, _, trainer = trained(copies=True)
    probes = trainer.plan["probes"]
    assert [(probe["narrow"], probe["fits"]) for probe in probes] == [(True, False), (False, True)]
    assert (trainer.plan["narrow"], trainer.plan["remat"], trainer.plan["split"]) == (False, "full", 1)
    assert state.compute is None


def test_a_fit_whose_state_fits_but_not_with_its_copies_trains_without_them(monkeypatch):
    """Row 1661's run (1765): devices that hold the state a run keeps between steps, its parameters,
    optimizer state and the window's accumulator, but not that state and its narrow bf16 copies. The
    check `fit` makes before it places anything holds that state alone, which `validate` reports the
    same with the copies on as off: the copies are the plan's first rung, dropped where the step does
    not fit with them. On devices of that capacity, whose free bytes lie between the step's footprints
    with and without its copies, fit trains, its plan without them."""
    from token_support import repeated_token_batch

    from dew.data.dataset import Loader
    from dew.distributed.placement import placed_bytes
    from dew.telemetry.compiles import Compiles
    from dew.training import Layout, MeshSpec, Trainer

    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})
    geometry = {"steps": 1, "accumulation": 2, "split": 1, "remat": "full"}
    narrow, _, trainer = _trained(ADAM_3, copies=True, **geometry)
    with_copies = free - trainer.plan["probes"][0]["headroom"]
    without = free - _trained(ADAM_3, copies=False, **geometry)[2].plan["probes"][0]["headroom"]
    copies = placed_bytes(jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), narrow.compute))

    def built() -> tuple[Trainer, dict]:
        objective, batch = _decoder(remat="full")
        return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(),
                       layout=Layout(min_shard=1, tolerance=1.0), accumulation=2, split=1), batch

    def held(*, copies: bool) -> int:
        run, batch = built()
        with chosen(narrow_copies='on' if copies else 'off'):
            return max(run.validate(batch).device_bytes.values())

    state = held(copies=False)
    assert copies > 0 and held(copies=True) == state
    run, _ = built()
    # Devices whose allocator holds the state and half its copies, its step's free bytes between its needs
    # beside the state: the plan made before the state is placed (`Trainer._planned_unplaced`), which reads
    # those bytes as holding the state too, holds none, and the step is planned once the state is placed.
    free, limit = (with_copies + without) // 2, state + copies // 2
    assert limit > free, (limit, free)
    stream = Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)
    with chosen(narrow_copies='on'):
        run.fit(stream, steps=2)
    probes = run.plan["probes"]
    assert [(probe["narrow"], probe["fits"]) for probe in probes] == [(True, False), (False, True)]
    # A pool grown to the state's size, half of it one free block, whose limit holds the state and the step
    # with its copies in all, the step in what the pool can still take: a fresh GH200 4B run's (M-MC-3m).
    # The step is planned before the state is placed on the pool's bytes in all, not on its largest block,
    # which placing the state reshapes, and judged on the allocator once the state is on it: compiled once,
    # beside the start, not again once the state is placed.
    grown = {"bytes_limit": state + with_copies + state // 2, "bytes_in_use": 0, "pool_bytes": state,
             "largest_free_block_bytes": state // 2}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: grown)
    run, _ = built()
    with chosen(narrow_copies='on'), Compiles("step") as steps:
        run.fit(stream, steps=2)
    assert [(probe["narrow"], probe["fits"]) for probe in run.plan["probes"]] == [(True, True)]
    assert steps.count == 1


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
                  mesh=MeshSpec().build(jax.devices()[:1]), layout=Layout(min_shard=1, tolerance=1.0))
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
                  layout=Layout(min_shard=1, tolerance=1.0))
    with chosen(narrow_copies='off'):
        state = max(run.validate(batch).device_bytes.values())
    room = (one + two) // 2
    total = 2 * (state + one)
    grown = {"bytes_limit": total, "bytes_in_use": 0, "pool_bytes": total - room,
             "largest_free_block_bytes": room // 2}
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: grown)
    stream = Loader(train=lambda partition: repeated_token_batch(64, 8, 9), val=None, records=None, batch=8)
    with chosen(narrow_copies='off'):
        run.fit(stream, steps=2)
    assert run.plan["probes"][-1]["split"] == 2


def test_a_plan_a_rung_above_its_dropped_copies_keeps_them_dropped(monkeypatch):
    """A window of one step that recomputes nothing holds about as much without its copies as with them:
    its backward keeps the whole gradient, in float32 where the copies' is bfloat16. Where it fits its
    own rung neither way, the plan climbs past the rung that dropped them, and keeps them dropped at the
    rung and split it reaches: once a plan fits, a run compiles no other (whether the copies would be
    faster there is the offline plan's to weigh, `compile_ahead`)."""
    free, limit = 2 ** 39, 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": limit, "bytes_in_use": limit - free})

    def trained(*, copies):
        return _trained(ADAM_3, copies=copies, steps=1, split=1, rows=64)

    base = trained(copies=True)[2].plan["probes"][0]
    free -= max(base["headroom"], trained(copies=False)[2].plan["probes"][0]["headroom"]) + 1
    state, _, trainer = trained(copies=True)
    probes, plan = trainer.plan["probes"], trainer.plan
    assert [(probe["narrow"], probe["fits"]) for probe in probes[:2]] == [(True, False), (False, False)]
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults")
    planned = {key: plan[key] for key in rung}
    last = probes[-1]
    assert planned != {key: base[key] for key in rung}
    assert {key: last[key] for key in rung} == planned and (last["narrow"], last["fits"]) == (False, True)
    assert not plan["narrow"] and state.compute is None


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

    shorter, longer, longer_without = used(8, copies=True), used(64, copies=True), used(64, copies=False)
    free = max(shorter, longer_without)
    assert free < longer
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

