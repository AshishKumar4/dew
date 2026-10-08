"""The narrow copies of the parameters the forward reads (`dew.training.narrow`)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import _filled
from dew.nn.kernels.generation import device_generation
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


def _trained(monkeypatch, generations, optimizer, steps=3, mesh=None):
    """A tied bf16 decoder over fp32 parameters, trained `steps` steps on
    one batch, with copies on the generations named, on `mesh` with every
    parameter split where it divides."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels import KERNELS
    from dew.objectives.lm import TEXT_KEY, LMObjective
    from dew.training import Layout, MeshSpec, Trainer
    monkeypatch.setitem(KERNELS, "narrow_copies", dict.fromkeys(generations, "on"))
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1,
                              mlp="swiglu", mlp_features=64, max_seq_len=16, qk_norm=True,
                              tie_embeddings=True, dtype=jnp.bfloat16)
    trainer = Trainer(LMObjective(model, seq_len=8), optimizer, key=jax.random.key(0),
                      mesh=mesh or MeshSpec(), layout=Layout(min_shard=1, tolerance=1.0))
    state = trainer.initial_state()
    batch = {TEXT_KEY: jnp.asarray(np.random.default_rng(0).integers(0, 64, (8, 9)), jnp.int32)}
    step = trainer.compile(state, batch)
    losses = []
    for _ in range(steps):
        state, loss, *_ = step(state, batch)
        losses.append(np.asarray(loss))
    return state, losses


@pytest.mark.parametrize("fsdp", [1, pytest.param(2, marks=pytest.mark.mesh(devices=2)),
                                  pytest.param(8, marks=pytest.mark.mesh(devices=8))])
def test_the_tied_decoder_reads_its_kernels_through_copies_with_the_same_gradients(monkeypatch, fsdp):
    """Through the trainer, a tied decoder's projection kernels are read
    through bf16 copies and its tied table and norm scales are not. Under
    plain SGD, whose update reads the gradient once, every parameter after
    three steps is bitwise what the in-forward cast gives: the copies
    change no gradient, and the tied table's two uses still sum in fp32.
    So too where the batch is split over data-parallel devices, whose
    gradients are summed across them, and where every parameter is split
    over fsdp devices too (the lane's eight: data 8, data 4 x fsdp 2, fsdp 8)."""
    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=fsdp)
    narrow, losses = _trained(monkeypatch, {device_generation()}, optax.sgd(0.5), mesh=mesh)
    plain, plain_losses = _trained(monkeypatch, set(), optax.sgd(0.5), mesh=mesh)
    paths = {tuple(key.key for key in path)[1:]: leaf.dtype
             for path, leaf in jax.tree_util.tree_flatten_with_path(narrow.compute)[0]}
    names = {"/".join(path) for path in paths}
    assert names and all(name.endswith("kernel") and "proj" in name for name in names), names
    assert plain.compute is None
    for a, b in zip(jax.tree.leaves(narrow.variables), jax.tree.leaves(plain.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(losses, plain_losses, strict=True):
        np.testing.assert_array_equal(a, b)
    # The copies are the cast of the parameters the last update wrote.
    for a, b in zip(jax.tree.leaves(narrow.compute), jax.tree.leaves(narrowed(narrow.variables, paths)),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_no_copy_off_the_measured_generations(monkeypatch):
    """Off `KERNELS['narrow_copies']` the state holds no copies."""
    state, _ = _trained(monkeypatch, set(), optax.adam(1e-3), steps=1)
    assert state.compute is None


def test_a_restored_state_drops_its_templates_copies():
    """The copies are of the template's own parameters; a restored state
    starts without and the step rebuilds them from what was restored."""
    from dew.training.state import TrainState

    zero, one = jnp.zeros((), jnp.int32), jnp.ones((), jnp.int32)
    template = TrainState(step=zero, microstep=zero, updates=zero, variables={"params": {"w": jnp.ones(2)}},
                          opt_state=(), ema=None, key=jax.random.key(0), scale=None, window_size=one,
                          compute={"params": {"w": jnp.ones(2, jnp.bfloat16)}})
    saved = {"step": zero, "window_size": one, "variables": {"params": {"w": jnp.zeros(2)}}}
    restored = _filled(template, saved, 0)
    assert isinstance(restored, TrainState)
    assert restored.compute is None

