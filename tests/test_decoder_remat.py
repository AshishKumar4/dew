"""Block recomputation preserves training, stateful collections and checkpoints.

A recomputed block is the plain block's arithmetic in another order, so each claim is check 1 between
the two over drawn decoders, both from the plain block's float64 run: a committed step's variables,
optimizer state, loss, gradients, metrics and per-head QK maxima, on one device and across pipeline
stages, and each named policy's loss and gradients against `full`. Cached decoding is bitwise
identical, and a checkpoint restores across remat policies value for value. The offloaded policies run
on the CPU backend, whose lowering of a host transfer is the identity, so their residuals are typed in
host memory space and compile to the same program as their saved counterparts; the transfer itself
needs a GPU or TPU.
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import selection_margins
from model_support import largest_difference
from reference_error import assert_as_exact, without_ynn

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step, model_rngs
from dew.objectives.lm import LMObjective
from dew.training.distributed import Layout, MeshSpec, shard_batch
from dew.training.state import TrainState
from dew.training.transaction import write_back

SHAPES = {
    "dense": {"num_nextn_predict_layers": 1},
    "sparse": {"mixture": {"experts": 4, "top_k": 2, "bias": True},
               "num_nextn_predict_layers": 1},
    "shared": {"kv_shared_layers": (2, 3), "per_layer_input_dim": 8,
               "use_double_wide_mlp": True, "sandwich_norms": True},
    "altup": {"altup": {"num_inputs": 2}, "kv_shared_layers": (2, 3),
              "per_layer_input_dim": 8, "laurel_rank": 4},
    "mla": {"mixer": {"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
                      "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4}}},
}


def model_for(shape, **overrides):
    return CausalTransformer(
        vocab_size=32, emb_features=16, num_layers=4, num_heads=2,
        num_kv_heads=1, mlp_features=32, max_seq_len=16,
        **{"dropout_rate": 0.15, **SHAPES[shape], **overrides})


def batch():
    return {"text": jnp.asarray(np.random.default_rng(5).integers(1, 32, (8, 9)),
                                jnp.int32)}


def objective_for(model):
    sparse = model.mixture is not None
    return LMObjective(model, 8, qk_stats=True,
                       balance_rate=0.01 if sparse else None,
                       aux_loss_alpha=0.001 if sparse else None,
                       mtp_weight=0.2 if model.num_nextn_predict_layers else None)


def training_step(model):
    """One committed step as the trainer takes it: the optimizer update, then
    sequential collection writes and the objective's deferred effects."""
    objective = objective_for(model)
    optimizer = optax.sgd(1e-3, momentum=0.9)

    def step(variables, state, tokens, key):
        info = Step(jnp.zeros((), jnp.int32), key, None)
        (loss, aux), grads = jax.value_and_grad(
            lambda params: objective.scalar_loss({**variables, "params": params},
                                       tokens, info),
            has_aux=True)(variables["params"])
        updates, state = optimizer.update(grads, state, variables["params"])
        moved = write_back({**variables, "params": optax.apply_updates(variables["params"], updates)},
                           aux.variables)
        if aux.effects is not None:
            moved = write_back(moved, objective.apply_effects(moved, aux.effects))
        return moved, state, loss, grads, aux

    return jax.jit(step), optimizer


def train_state(step, variables, opt_state, key):
    """A single-microbatch state after `step` committed updates."""
    count = jnp.asarray(step, jnp.int32)
    return TrainState(step=count, microstep=count, updates=count, variables=variables,
                      opt_state=opt_state, ema=None, key=key, scale=None,
                      window_size=jnp.asarray(1, jnp.int32))


STEP_SEEDS = 8
"""Check 1's declared seeds for a recomputed block's committed step against the plain block's (`seeds_for`,
paired: both arms are one decoder's arithmetic, saved or recomputed), and for its forward."""

MLA_STEP_SEEDS = 16
"""The latent-attention block's step, unscanned, sized at 16 (`seeds_for`, paired)."""


def wide_step(model):
    """`training_step` on `model` in float64: the truth of a remat check 1."""
    return training_step(model.clone(dtype=jnp.float64))[0]


def drawn_steps(plain, stepped, *, routed: bool, within=contextlib.nullcontext, place=None):
    """Each seed's draw of a committed step's inputs: the decoder initialized from the seed, its
    optimizer state, the batch and the step's key. An even seed's decoder is initialized, its momentum
    zero; an odd seed's has taken one plain step under its own key, its momentum a step's. A routed
    decoder takes the seed's first draw whose routing clears `selection_margins.MARGIN` in the step's
    own training forward (its dropout, its prediction depth, `within` the step's context on the
    inputs as `place` lays them out), so its float32 runs and its float64 run route alike; routing
    near a tie is the router's own tests'. `stepped` is `training_step(plain)`, compiled once for the
    draws and the reference arm."""
    run, solver = stepped
    ids = batch()["text"][:, :-1]
    initialized = jax.jit(lambda key: plain.init(key, ids))
    objective = objective_for(plain)
    scored = jax.jit(lambda variables, tokens, key: objective.token_scores(
        variables, tokens, train=True, rngs=model_rngs(key, training=True), routing=True,
        depths=objective.mtp_weight is not None, qk_stats=objective.qk_stats).sown)

    def phase(primed: bool):
        def keyed(seed):
            variables = initialized(jax.random.key(seed))
            key = jax.random.key(seed + (1 << 20))
            state = solver.init(variables["params"])
            if primed:
                variables, state, *_ = run(variables, state, batch(), key)
            # The key travels as its bits, which a float64 truth's widening leaves as they are.
            return variables, state, batch(), jax.random.key_data(jax.random.fold_in(key, 1))
        return selection_margins.clearing(keyed, clears) if routed else keyed

    def clears(inputs):
        variables, _, tokens, key = inputs
        laid = (variables, tokens) if place is None else place(variables, tokens)
        with within():
            sown = scored(laid[0], laid[1]["text"], jax.random.wrap_key_data(key))
        return selection_margins.routing_margin(jax.device_get(sown), variables.get("moe", {}), 2) \
            >= selection_margins.MARGIN

    initial, primed = phase(primed=False), phase(primed=True)
    return lambda seed: (primed if seed % 2 else initial)(seed // 2)


def committed(step):
    """A step's committed values as check 1 compares them: the variables and optimizer state it leaves,
    its loss and gradients, and the metrics and per-head QK maxima it reports."""
    def run(variables, state, tokens, key):
        moved, next_state, loss, grads, aux = step(variables, state, tokens, jax.random.wrap_key_data(key))
        return moved, next_state, loss, grads, aux.metrics, aux.qk_stats
    return run


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_blocks_train_the_same_model(shape, scan):
    """Dropout, MTP, MoE balancing and shared K/V retain their derivatives.

    Both models start from the same initialized variables and the same
    plain forward, and over drawn decoders, one plain step on, the remat
    model's committed step (its variables, optimizer state, loss, gradients,
    metrics and QK maxima, the deferred router-bias effects included) is on
    par with the plain model's, both from the plain step's float64 run
    (check 1). The router's slot counts are integers, routed alike. Chains
    are compared a step at a time: recomputation changes rounding in the
    backward pass, and a token within rounding of a routing tie moves to
    another expert at the next step."""
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat='full')
    ids = batch()["text"][:, :-1]
    variables = plain.init(jax.random.key(0), ids)
    recomputed = remat.init(jax.random.key(0), ids)
    assert jax.tree.structure(variables) == jax.tree.structure(recomputed)
    assert largest_difference(variables, recomputed) == 0
    stepped = training_step(plain)
    run, _ = stepped
    rerun, _ = training_step(remat)
    draw = drawn_steps(plain, stepped, routed=shape == "sparse")
    wide = plain.clone(dtype=jnp.float64)
    assert_as_exact(without_ynn(lambda variables, *_: remat.apply(variables, ids)),
                    without_ynn(lambda variables, *_: plain.apply(variables, ids)), draw,
                    f"a recomputed {shape} block's forward, scan {scan}", seeds=STEP_SEEDS,
                    truth_fn=jax.jit(lambda variables, *_: wide.apply(variables, ids)))
    assert_as_exact(without_ynn(committed(rerun)), without_ynn(committed(run)), draw,
                    f"a recomputed {shape} block's step, scan {scan}",
                    seeds=MLA_STEP_SEEDS if shape == "mla" and not scan else STEP_SEEDS,
                    truth_fn=committed(wide_step(plain)))
    moved, state, tokens, key = draw(0)
    key = jax.random.wrap_key_data(key)
    after, _, _, _, aux = run(moved, state, tokens, key)
    _, _, _, _, other_aux = rerun(moved, state, tokens, key)
    assert not np.array_equal(after["params"]["embed_tokens"]["embedding"],
                              moved["params"]["embed_tokens"]["embedding"])
    if shape == "sparse":
        np.testing.assert_array_equal(jax.tree.leaves(aux.effects), jax.tree.leaves(other_aux.effects))
        assert not all(np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(after["moe"]), jax.tree.leaves(moved["moe"]), strict=True))


@pytest.mark.parametrize("shape", ["dense", "shared", "altup"])
@pytest.mark.parametrize("scan", [False, True])
def test_remat_prefills_and_appends_the_same_cache(shape, scan):
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat='full')
    ids = batch()["text"][:2]
    plain_vars = plain.init(jax.random.key(0), ids[:, :2], decode=True)
    remat_vars = remat.init(jax.random.key(0), ids[:, :2], decode=True)
    for start, end in [(0, 5), (5, 6), (6, 8)]:
        expected, cache = plain.apply(plain_vars, ids[:, start:end],
                                       decode=True, mutable=["cache"])
        actual, recomputed_cache = remat.apply(remat_vars, ids[:, start:end],
                                               decode=True, mutable=["cache"])
        assert largest_difference(actual, expected) == 0

        plain_vars = {**plain_vars, **cache}
        remat_vars = {**remat_vars, **recomputed_cache}


@pytest.mark.mesh
@pytest.mark.parametrize("shape", ["dense", "sparse"])
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_pipeline_preserves_updates_and_sown_values(shape, scan):
    """Over drawn decoders, the remat model's committed step across two
    pipeline stages of two microbatches each, on par with the plain model's
    (check 1), both from the plain step's float64 run across the same
    stages."""
    plain = model_for(shape, scan_layers=scan, num_nextn_predict_layers=0)
    remat = plain.clone(remat='full')
    # Eight rows split four ways over data x fsdp hold two a device, which
    # two microbatches divide.
    mesh = MeshSpec(fsdp=2, stage=2).build()
    stepped = training_step(plain)
    run, _ = stepped
    rerun, _ = training_step(remat)

    def place(variables, tokens):
        laid = jax.device_put(variables, Layout(min_shard=16).shardings(mesh, variables))
        return laid, shard_batch(mesh, tokens)

    @contextlib.contextmanager
    def staged():
        with jax.set_mesh(mesh), pipeline_microbatches(2):
            yield

    def placed(step):
        def on_mesh(variables, state, tokens, key):
            variables, tokens = place(variables, tokens)
            with staged():
                return jax.device_get(committed(step)(variables, state, tokens, key))
        return on_mesh

    draw = drawn_steps(plain, stepped, routed=shape == "sparse", within=staged, place=place)
    assert_as_exact(placed(rerun), placed(run), draw, f"a recomputed {shape} pipeline's step, scan {scan}",
                    seeds=STEP_SEEDS, truth_fn=placed(wide_step(plain)))


@pytest.mark.parametrize("saved_remat", [None, 'full'])
def test_checkpoint_resumes_with_remat_switched(tmp_path, saved_remat):
    """A checkpoint saved under one remat policy restores, value for value, into a template of the other,
    and the other model steps from it as from the state saved: the restore is exact, and so is the same
    program on the same values (that the two policies train alike is
    `test_recomputed_blocks_train_the_same_model`'s)."""
    model = model_for("shared", scan_layers=True, remat=saved_remat)
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    run, optimizer = training_step(model)
    moved, opt_state, _, _, _ = run(
        variables, optimizer.init(variables["params"]), batch(), jax.random.key(3))
    state = train_state(1, moved, opt_state, jax.random.key(4))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    resumed_model = model.clone(remat='full' if saved_remat is None else None)
    template_vars = resumed_model.init(jax.random.key(8), batch()["text"][:, :-1])
    template = train_state(0, template_vars, optimizer.init(template_vars["params"]),
                           jax.random.key(0))
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=value.sharding), template)
    restored, position = checkpoints.restore(template)
    assert position is None
    assert int(restored.step) == 1 and int(restored.updates) == 1
    for saved, loaded in zip(jax.tree.leaves((state.variables, state.opt_state)),
                             jax.tree.leaves((restored.variables, restored.opt_state)), strict=True):
        np.testing.assert_array_equal(loaded, saved)
    resume, _ = training_step(resumed_model)
    actual = resume(restored.variables, restored.opt_state, batch(), restored.key)
    expected = resume(state.variables, state.opt_state, batch(), state.key)
    assert np.isfinite(actual[2])
    for got, want in zip(jax.tree.leaves(actual[:4]), jax.tree.leaves(expected[:4]), strict=True):
        np.testing.assert_array_equal(got, want)


NAMED = [name for name in REMAT_POLICIES if name != 'full']

POLICY_SEEDS = 8
"""Check 1's declared seeds for a named policy's loss and gradients against `full`'s (`seeds_for`,
paired)."""


def gradient_step(model):
    """The objective's loss, its reports and its gradient in the parameters, from drawn variables and a
    step key."""
    objective = objective_for(model)

    def step(variables, key):
        def loss(params):
            info = Step(jnp.zeros((), jnp.int32), jax.random.wrap_key_data(key), None)
            return objective.scalar_loss({**variables, "params": params}, batch(), info)

        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(variables["params"])
        # Perplexity is the cross entropy's exponential, which magnifies the same rounding; its
        # logarithm is the cross entropy itself, so the metrics compare in cross-entropy units.
        metrics = {name: jnp.log(metric) if name == "perplexity" else metric
                   for name, metric in aux.metrics.items()}
        return value, grads, metrics
    return step


@functools.cache
def against_full(scan):
    """`full`'s decoder, its draws, and its step compiled once for every policy: the reference arm and,
    in float64, the truth."""
    full = model_for("dense", scan_layers=scan, remat='full')
    initialized = jax.jit(lambda key: full.init(key, batch()["text"][:, :-1]))

    def draw(seed):
        return initialized(jax.random.key(seed)), jax.random.key_data(jax.random.key(seed + (1 << 20)))

    return full, draw, without_ynn(gradient_step(full)), jax.jit(gradient_step(full.clone(dtype=jnp.float64)))


@pytest.mark.parametrize("policy", NAMED)
@pytest.mark.parametrize("scan", [False, True])
def test_named_policies_train_the_same_model_as_full(policy, scan):
    """Saved and offloaded residuals feed the backward pass the values the
    recomputation would have produced, dropout and MTP included: over drawn
    decoders and keys, a named policy's loss, gradients and metrics on par
    with `full`'s, both from `full`'s float64 run (check 1)."""
    full, draw, reference, truth = against_full(scan)
    assert_as_exact(without_ynn(gradient_step(full.clone(remat=policy))), reference, draw,
                    f"{policy} against full, scan {scan}", seeds=POLICY_SEEDS, truth_fn=truth)
