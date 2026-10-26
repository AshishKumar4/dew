"""The narrow copies of the parameters the forward reads (`dew.training.narrow`)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import _filled
from dew.nn.kernels.generation import chosen
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


def _trained(optimizer, *, copies, steps=3, mesh=None, padded=0, accumulation=1, split=None,
             mixture=False, rows=8, ragged=False, scanned=False, remat=None, longer=None):
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
    many rows, whose shape the trainer plans apart. Returns the state, each
    step's loss and the trainer, its plan the last shape's."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import TEXT_KEY, LMObjective
    from dew.training import Layout, MeshSpec, Trainer
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=4 if scanned else 2, num_heads=2,
                              num_kv_heads=1, mlp="swiglu", mlp_features=64, max_seq_len=16, qk_norm=True,
                              tie_embeddings=True, dtype=jnp.bfloat16, scan_layers=scanned, remat=remat,
                              mixture={"experts": 4, "top_k": 2} if mixture else None)
    objective = LMObjective(model, seq_len=8, pad_id=0, aux_loss_alpha=.01 if mixture else None)
    batch = {TEXT_KEY: jnp.asarray(np.random.default_rng(0).integers(1, 64, (rows, 9)), jnp.int32)}
    long = longer and {TEXT_KEY: jnp.asarray(np.random.default_rng(1).integers(1, 64, (longer, 9)),
                                             jnp.int32)}
    with chosen(narrow_copies='on' if copies else 'off'):
        trainer = Trainer(objective, optimizer, key=jax.random.key(0), mesh=mesh or MeshSpec(),
                          layout=Layout(min_shard=1, tolerance=1.0), accumulation=accumulation, split=split)
        state = trainer.initial_state()
        steps_by = {False: (trainer.compile(state, batch), None)}
        if long:
            steps_by[True] = (trainer.compile(state, long), long)
    shorter = {TEXT_KEY: batch[TEXT_KEY].at[::2, -3:].set(0)}
    trained = [shorter if ragged and index % 2 == 0 else batch for index in range(steps)]
    losses = []
    for index, current in enumerate(trained + [{TEXT_KEY: jnp.zeros_like(batch[TEXT_KEY])}] * padded):
        step, read = steps_by[bool(long) and index % 2 == 1]
        state, loss, *_ = step(state, current if read is None else read)
        losses.append(np.asarray(loss))
    return state, losses, trainer


def _assert_plain_update(narrow, losses, plain, plain_losses):
    """`narrow`, trained through copies, holds bitwise the parameters, last
    gradient (`_keeping_gradients`) and losses of `plain`, trained without;
    each copy is split as its parameter is, so a gather moves bf16, and is
    the cast of the parameters the last update wrote. Returns the copies'
    paths under `params`."""
    assert plain.compute is None
    for a, b in zip(jax.tree.leaves((narrow.variables, narrow.opt_state)),
                    jax.tree.leaves((plain.variables, plain.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(losses, plain_losses, strict=True):
        np.testing.assert_array_equal(a, b)
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


@pytest.mark.parametrize("fsdp", [1, pytest.param(2, marks=pytest.mark.mesh(devices=2)),
                                  pytest.param(8, marks=pytest.mark.mesh(devices=8))])
def test_the_tied_decoder_reads_its_kernels_through_copies_with_the_same_gradients(fsdp):
    """Through the trainer, a tied decoder's projection kernels are read
    through bf16 copies and its tied table and norm scales are not. Under
    plain SGD, whose update reads the gradient once, every parameter and the
    last gradient after three steps are bitwise what the in-forward cast
    gives: the copies change no gradient, and the tied table's two uses
    still sum in fp32. So too where the batch is split over data-parallel
    devices, whose gradients are summed across them, and where every
    parameter is split over fsdp devices too (the lane's eight: data 8,
    data 4 x fsdp 2, fsdp 8)."""
    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=fsdp)
    narrow, losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=True, mesh=mesh)
    plain, plain_losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=False,
                                      mesh=mesh)
    names = {"/".join(path) for path in _assert_plain_update(narrow, losses, plain, plain_losses)}
    assert names and all(name.endswith("kernel") and "proj" in name for name in names), names


def test_a_scanned_decoder_reads_every_layers_kernels_through_copies_with_the_same_gradients():
    """A decoder that scans its layers over their stacked weights reads each layer's projection
    kernels through bf16 copies, stacked as the scan reads them, as an unscanned decoder does: under
    plain SGD the parameters, the last gradient and the losses are bitwise the in-forward cast's."""
    narrow, losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=True, scanned=True)
    plain, plain_losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=False, scanned=True)
    names = {"/".join(path) for path in _assert_plain_update(narrow, losses, plain, plain_losses)}
    layers = {name.split("/")[0] for name in names}
    assert {f"layers_{index}" for index in range(4)} <= layers, names


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
    windows are bitwise the plain run's, with the parameters and their
    copies split over fsdp devices too (pairwise over the window's kind,
    its pooling and the mesh)."""
    from dew.training import MeshSpec

    # Each microbatch keeps a row for every row shard of the lane's devices.
    run = {"steps": 2 * accumulation, "accumulation": accumulation, "split": split, "mixture": mixture,
           "rows": split * max(8, jax.device_count()), "ragged": True, "mesh": MeshSpec(fsdp=fsdp)}
    narrow, losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=True, **run)
    plain, plain_losses, _ = _trained(_keeping_gradients(optax.sgd(0.5)), copies=False, **run)
    assert int(narrow.updates) == int(plain.updates) == 2
    assert _assert_plain_update(narrow, losses, plain, plain_losses)


def test_a_skipped_update_keeps_the_copies_of_the_parameters_it_kept():
    """A batch of padding alone has nothing to predict, so its step commits
    no update: the parameters stay as the last update wrote them, and so do
    their copies. Adam's moments would have moved every parameter had the
    update been taken, so copies cast from the update the step discarded
    would differ from the parameters it kept."""
    trained, *_ = _trained(optax.adam(1e-2), copies=True, steps=2)
    skipped, *_ = _trained(optax.adam(1e-2), copies=True, steps=2, padded=1)
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
    from dew.training.distributed import placed_bytes
    from dew.training.memory import step_headroom

    limit = 2 ** 40
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: {"bytes_limit": limit, "bytes_in_use": 0})
    held = []

    def trained(*, copies):
        state, _, trainer = _trained(optax.adam(1e-3), copies=copies, steps=1)
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
    free = 2 ** 39
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - free})

    def trained(*, copies):
        return _trained(optax.adam(1e-3), copies=copies, steps=1, accumulation=2, split=1, remat="full")

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


def test_a_plan_a_rung_above_its_dropped_copies_takes_them_back_where_they_fit(monkeypatch):
    """A window of one step that recomputes nothing holds about as much without its copies as with them:
    its backward keeps the whole gradient, in float32 where the copies' is bfloat16. Where it fits its
    own rung neither way, the plan climbs past the rung that dropped them, and at the rung and split it
    reaches it compiles the step with them once more and keeps them, as they fit there too."""
    free = 2 ** 39
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - free})

    def trained(*, copies):
        return _trained(optax.adam(1e-3), copies=copies, steps=1, split=1, rows=64)

    base = trained(copies=True)[2].plan["probes"][0]
    free -= max(base["headroom"], trained(copies=False)[2].plan["probes"][0]["headroom"]) + 1
    state, _, trainer = trained(copies=True)
    probes, plan = trainer.plan["probes"], trainer.plan
    assert [(probe["narrow"], probe["fits"]) for probe in probes[:2]] == [(True, False), (False, False)]
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults")
    planned = {key: plan[key] for key in rung}
    fitted, last = probes[-2], probes[-1]
    assert planned != {key: base[key] for key in rung}
    assert [{key: probe[key] for key in rung} for probe in (fitted, last)] == [planned, planned]
    assert [(probe["narrow"], probe["fits"]) for probe in (fitted, last)] == [(False, True), (True, True)]
    assert plan["narrow"] and state.compute is not None


def test_a_run_whose_shapes_plan_their_copies_apart_steps_each_as_its_plan_reads(monkeypatch):
    """Each shape a run meets plans its copies itself (`_at_base_rung`): where a longer batch fits only
    without them and a shorter one with them, the steps take turns, each reading the state as its own plan
    compiled it, the copies dropped before the longer batch's step and built again before the shorter's,
    and under plain SGD the run lands bitwise on the parameters and losses of one without copies."""
    free = 2 ** 39
    monkeypatch.setattr(jax.Device, "memory_stats",
                        lambda device: {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - free})

    def trained(*, copies, rows=8, longer=None, steps=0):
        return _trained(optax.sgd(0.5), copies=copies, steps=steps, accumulation=2, split=1, remat="full",
                        rows=rows, longer=longer)

    def used(rows, *, copies):
        return free - trained(copies=copies, rows=rows)[2].plan["probes"][0]["headroom"]

    shorter, longer, longer_without = used(8, copies=True), used(64, copies=True), used(64, copies=False)
    free = max(shorter, longer_without)
    assert free < longer
    narrow, losses, trainer = trained(copies=True, longer=64, steps=3)
    assert not trainer.plan["narrow"] and narrow.compute is not None
    plain, plain_losses, _ = trained(copies=False, longer=64, steps=3)
    _assert_plain_update(narrow, losses, plain, plain_losses)


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

