"""Effective-loss and checkpoint transactions over tiny deterministic objectives.

Replay and effect commits are Dew transaction policies; Optax and Flax oracle their arithmetic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import raw_leaf
from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib
from reference_error import assert_as_exact, dew_programs

from dew.checkpoints import Checkpoints
from dew.data.dataset import DataPartition
from dew.objectives import Aux, EMASpec, Objective, Ratio
from dew.objectives.base import under
from dew.training import Trainer


@struct.dataclass
class Terms:
    prediction: Ratio
    rows: Ratio
    scores: jax.Array
    counts: jax.Array
    positions: jax.Array


class Tiny(Objective[Ratio | Terms, None]):
    ema = EMASpec(optax.constant_schedule(.5), select=under("params"))

    def __init__(self, composite=False):
        self.composite = composite

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.array(.2)}, "stats": {"seen": jnp.array(0.)}}

    def loss(self, variables, batch, step):
        w = variables["params"]["w"]
        noise = jax.random.uniform(step.key, batch["y"].shape) * .03
        prediction = (w + noise + variables["stats"]["seen"] * .01
                      + step.ema["stats"]["seen"] * .1 + step.step * .02)
        errors = (prediction - batch["y"]) ** 2
        bad = jax.lax.cond(batch["bad"],
                           lambda x: jnp.sqrt(x - jax.lax.stop_gradient(x)),
                           lambda x: x * 0, w)
        main = Ratio(jnp.sum(errors * batch["mask"]) + bad, jnp.sum(batch["mask"]))
        aux = Aux({}, variables={"stats": {"seen": variables["stats"]["seen"] + 1}})
        if not self.composite:
            return main, aux
        logits = jnp.stack((prediction, -prediction, prediction * .3), axis=-1)
        scores = jax.nn.softmax(logits, axis=-1)
        counts = jnp.bincount(jnp.argmax(scores, axis=-1), length=3)
        row = Ratio(jnp.sum((prediction + batch["y"]) ** 2) * batch["active"],
                   jnp.asarray(batch["y"].size) * batch["active"])
        return Terms(main, row, jnp.sum(scores, axis=0) * batch["active"],
                     counts * batch["active"], jnp.asarray(scores.shape[0]) * batch["active"]), aux

    def reduce_loss(self, stats):
        if isinstance(stats, Ratio):
            return stats.mean()
        main, a = stats.prediction.mean()
        row, b = stats.rows.mean()
        positions = jnp.where(stats.positions > 0, stats.positions, 1)
        auxiliary = .2 * 3 * jnp.vdot(jax.lax.stop_gradient(stats.counts), stats.scores) / positions ** 2
        return main + .1 * row + auxiliary, a | b | (stats.positions > 0)

def batches():
    return [{"y": jnp.tile(jnp.array([1., 2.]), jax.device_count()),
                 "mask": jnp.tile(jnp.array(mask), jax.device_count()),
                 "bad": jnp.array(bad), "active": jnp.array(1)}
            for mask, bad in [([1., 0.], False), ([1., 1.], True), ([.5, 1.], False), ([1., 1.], False)]]

SHORT_SCALE = dynamic_scale_lib.DynamicScale(growth_interval=1)
"""A dynamic loss scale that grows after every finite step, so a few steps reach its growth."""


def trainer(composite=False, k=2, checkpoints=None):
    return Trainer(Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
                   accumulation=k, dynamic_scale=SHORT_SCALE, checkpoints=checkpoints)


def test_a_commit_counts_the_tokens_of_its_windows_microbatches_alone():
    """A window of two whose second microbatch is rejected (a non-finite loss) is dropped, its first
    with it (`dropped_window`): the next window, the third and fourth microbatches, commits, and its
    update trained on their tokens alone, which its commit reports (`Committed.tokens`), the dropped
    window's never, and nothing before the commit."""
    from dew.training.model_flops import TOKENS, _TokenCount

    train = trainer()
    rows = 2 * jax.device_count()
    data = [{**batch, TOKENS: _TokenCount.words(np.full(rows, tokens))}
            for batch, tokens in zip(batches(), (3, 5, 7, 11), strict=True)]
    state = train.initial_state()
    run = train.compile(state, data[0])
    reported = []
    for batch in data:
        state, _, _, _, accepted, committed = run(state, batch)
        reported.append((bool(accepted), _TokenCount.host(jax.device_get(committed.tokens))))
    assert reported == [(True, 0), (False, 0), (True, 0), (True, (7 + 11) * rows)]


@pytest.mark.parametrize("composite", [False, True])
def test_a_boundary_rejection_drops_its_window_and_keeps_its_mutable_writes(composite):
    """A window of two whose closing microbatch is rejected is dropped whole (`dropped_window`): its
    parameters stay, its scale halves, its sums are cleared and the next microbatch takes its first
    slot. The mutable collection its accepted microbatch wrote keeps the write, as torch's AMP keeps a
    skipped update's running statistics, so the next window's two microbatches read it, and their
    update is the one the pooled loss over them takes, each under the key of its attempt."""
    train = trainer(composite)
    initial = train.initial_state()
    data = batches()
    run = train.compile(initial, data[0])
    # the step consumes the state
    start, opt_state, key = (
        jax.tree.map(np.asarray, initial.variables),
        jax.tree.map(np.asarray, initial.opt_state),
        np.asarray(jax.random.key_data(initial.key)),
    )
    prefix, *_ = run(initial, data[0])
    # the step consumes the state
    prefix_scale = float(prefix.scale.scale)
    prefix_params = jax.tree.map(np.asarray, prefix.variables)
    rejected, _, _, finite, accepted, _ = run(prefix, data[1])
    assert bool(finite) and not bool(accepted)
    assert (int(rejected.step), int(rejected.microstep)) == (2, 0)
    assert float(rejected.scale.scale) == prefix_scale / 2
    for before, after in zip(jax.tree.leaves(prefix_params), jax.tree.leaves(rejected.variables),
                             strict=True):
        np.testing.assert_array_equal(before, after)
    # Every sum of the window is cleared, its QK maxima at their identity; the gradient sum alone stays,
    # unread until the next window's first microbatch replaces it.
    window = rejected.accumulation
    for name in ("statistics", "effects", "mass", "tokens", "batches", "attempts", "variables"):
        assert all(not np.any(np.asarray(leaf)) for leaf in jax.tree.leaves(getattr(window, name))), name
    assert all(np.all(np.asarray(leaf) == -np.inf) for leaf in jax.tree.leaves(window.qk_stats)
               if jnp.issubdtype(leaf.dtype, jnp.inexact))
    pending, _, _, _, accepted, _ = run(rejected, data[2])
    assert bool(accepted) and int(pending.updates) == 0
    final, _, _, _, accepted, _ = run(pending, data[3])
    assert bool(accepted) and int(final.updates) == 1
    assert float(final.variables["stats"]["seen"]) == 3
    assert float(start["stats"]["seen"]) == 0

    def combined(w):
        stats = []
        for attempt, seen, microstep in [(2, 1., 0), (3, 2., 1)]:
            variables = {"params": {"w": w}, "stats": {"seen": jnp.array(seen)}}
            from dew.objectives import Step
            info = Step(jnp.asarray(microstep), jax.random.fold_in(key, attempt), variables)
            value, _ = train.objective.loss(variables, data[attempt], info)
            stats.append(value)
        pooled = jax.tree.map(lambda a, b: a + b, *stats)
        return train.objective.reduce_loss(pooled)[0]
    gradient = jax.grad(combined)(start["params"]["w"])
    update, expected_opt = train.optimizer.update({"w": gradient}, opt_state, start["params"])
    expected = optax.apply_updates(start["params"], update)
    np.testing.assert_allclose(final.variables["params"]["w"], expected["w"], rtol=1e-6, atol=1e-7)
    for want, got in zip(jax.tree.leaves(expected_opt), jax.tree.leaves(final.opt_state), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("auxiliary", [None, "rows", "global"])
def test_real_lm_mtp_router_and_qk_update_matches_combined_batch(auxiliary):
    from dew.data.chat import Role
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective
    from dew.optim import scale_by_qk_clip, stepped

    # QK-Clip bounds logits only where no QK-norm divides the kernels' scale
    # back out, and refuses a normed layer.
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=16,
                              num_nextn_predict_layers=2, qk_norm=False,
                              mixture={"experts": 4, "top_k": 2, "bias": True,
                                       "score_function": "softmax"})
    objective = LMObjective(model, 10, head_chunks=1, mtp_weight=.3,
                            loss_role=Role.ASSISTANT, balance_rate=.02,
                            aux_loss_alpha=None if auxiliary is None else .2,
                            seq_aux=auxiliary != "global", qk_stats=True)
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.sgd(.03), scale_by_qk_clip(.05))
    train = Trainer(objective, optimizer, key=jax.random.PRNGKey(7), accumulation=2,
                    dynamic_scale=True)
    initial = train.initial_state()
    rows = jax.device_count()
    rng = np.random.default_rng(4)
    data = []
    for targets in (1, 9):
        roles = np.full((rows, 11), int(Role.USER), np.int32)
        roles[:, -targets:] = int(Role.ASSISTANT)
        data.append({"text": jnp.asarray(rng.integers(0, 16, size=(rows, 11)), jnp.int32),
                     "text_roles": jnp.asarray(roles)})
    run = train.compile(initial, data[0])
    # the step consumes the state
    start, opt_state, key = (
        jax.tree.map(np.asarray, initial.variables),
        jax.tree.map(np.asarray, initial.opt_state),
        np.asarray(jax.random.key_data(initial.key)),
    )
    partial, *_ = run(initial, data[0])
    for before, after in zip(jax.tree.leaves(start), jax.tree.leaves(partial.variables), strict=True):
        np.testing.assert_array_equal(before, after)
    actual, actual_loss, _, _, accepted, _ = run(partial, data[1])
    assert bool(accepted) and int(actual.updates) == 1
    combined = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), *data)
    info = Step(jnp.array(0), jax.random.fold_in(key, 0), start)
    (expected_loss, aux), gradient = jax.value_and_grad(
        lambda p: objective.scalar_loss({**start, "params": p}, combined, info),
        has_aux=True)(start["params"])
    updates, clipped = optimizer.update(gradient, opt_state, start["params"], qk_stats=aux.qk_stats)
    # The clip rescales the stepped weights, as the step stores them (`stepped`).
    expected = stepped(start["params"], updates, opt_state, clipped)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6, atol=2e-6)
    for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual.variables["params"]), strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    expected_moe = objective.apply_effects(start, aux.effects)["moe"]
    for want, got in zip(jax.tree.leaves(expected_moe),
                         jax.tree.leaves(actual.variables["moe"]), strict=True):
        np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))

    # An independent role-mask equation catches a shared bug in both batching paths.
    if auxiliary is None:
        scores = objective.token_scores(start, combined["text"], depths=True)
        roles = combined["text_roles"] == int(Role.ASSISTANT)
        numerator = jnp.sum(scores.losses * roles[:, 1:])
        for depth, (losses, _, _) in enumerate(scores.depths, 1):
            numerator += .3 / 2 * jnp.sum(losses * roles[:, depth + 1:])
        np.testing.assert_allclose(expected_loss, numerator / jnp.sum(roles[:, 1:]), rtol=1e-6)


def test_replay_preserves_half_precision_cotangents_and_integer_support():
    class Half(Tiny):
        def loss(self, variables, batch, step):
            w = variables["params"]["w"]
            count = jnp.asarray(batch["y"].size)
            prediction = Ratio(((w - 1) ** 2 * count).astype(jnp.float16), count)
            row = Ratio(((w + 3) ** 2).astype(jnp.float16), jnp.asarray(1))
            return Terms(prediction, row, jnp.zeros(3, jnp.float16),
                         jnp.zeros(3, jnp.int32), jnp.asarray(0)), Aux({})
    train = Trainer(Half(), optax.sgd(.1), key=jax.random.PRNGKey(1), accumulation=2)
    initial = train.initial_state()
    data = batches()[0]
    step = train.compile(initial, data)
    # the step consumes the state
    w = np.asarray(initial.variables["params"]["w"])
    state, *_ = step(initial, data)
    state, *_ = step(state, data)
    expected = w - .1 * (2 * (w - 1) + .2 * (w + 3))
    # Original fp16 statistic pullbacks round their coefficients to fp16.
    np.testing.assert_allclose(state.variables["params"]["w"], expected, rtol=0, atol=5e-5)


def test_local_partial_snapshot_survives_continued_training_and_weight_restore(tmp_path):
    checkpoints = Checkpoints(str(tmp_path / "persistent"),
                              local_directory=str(tmp_path / "local"), local_every=1)
    train = trainer(composite=True, checkpoints=checkpoints)
    initial = train.initial_state()
    data = batches()
    step = train.compile(initial, data[0])
    prefix, *_ = step(initial, data[0])
    checkpoints.save_local(1, prefix, b"1", share=DataPartition())
    # the step consumes the state
    prefix_params = jax.tree.map(np.asarray, prefix.variables)
    final, *_ = step(prefix, data[1])
    final, *_ = step(final, data[2])
    checkpoints.wait()
    resumed_trainer = trainer(composite=True, checkpoints=checkpoints)
    restored, _, position = resumed_trainer.place()
    assert position == b"1"
    step = resumed_trainer.compile(restored, data[1])
    resumed, *_ = step(restored, data[1])
    resumed, *_ = step(resumed, data[2])
    for want, got in zip(jax.tree.leaves(final), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))
    template = {"variables": jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), restored.variables)}
    selected, _ = checkpoints.restore(template, 1)
    for want, got in zip(jax.tree.leaves(prefix_params), jax.tree.leaves(selected["variables"]), strict=True):
        np.testing.assert_array_equal(raw_leaf(got), raw_leaf(want))


def test_every_microbatch_of_an_accumulated_update_reads_the_updates_committed_before_it():
    """An update of two microbatches (`accumulation=2`) hands each of them the updates the run
    committed before its own (`Step.updates`), the clock an alternating objective reads its phase
    from, beside the accepted microbatches its schedules read (`Step.step`)."""
    from affine_run import Regression, batches

    from dew.data.dataset import Loader

    seen = []

    class Clocked(Regression):
        def loss(self, variables, batch, step):
            # Unordered, as a sharded program's callback is, and once a device: the set says each.
            jax.debug.callback(lambda updates, microbatch: seen.append((int(updates), int(microbatch))),
                               step.updates, step.step)
            return super().loss(variables, batch, step)

    state = Trainer(Clocked(), optax.sgd(0.1), key=jax.random.PRNGKey(0), accumulation=2).fit(
        Loader(lambda partition: batches(), None, None, 8), steps=6, log_every=100)
    assert int(state.updates) == 3
    # A debug callback's host work runs apart from the arrays it reads; the barrier waits for it.
    jax.effects_barrier()
    assert sorted(set(seen), key=lambda clocks: clocks[1]) == [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4),
                                                                (2, 5)]


@pytest.mark.gpu("XLA:GPU's simplifier speculates the branch; XLA:CPU's pipeline kept it a branch")
def test_an_accumulating_step_updates_the_optimizer_in_its_commit_alone():
    """A window's microbatches before its last only accumulate: nothing in the
    step outside the commit's branch reads an Adam moment. XLA's conditional
    simplifier runs a branch of elementwise ops and reductions, as a clipped
    AdamW's update is (`OptimConfig`'s), on every call and selects its
    result; on Qwen3.5-0.8B that was the whole update every microbatch."""
    from dew.config import OptimConfig
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.telemetry.instrumentation import hlo_instructions

    model = CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=8)
    train = Trainer(LMObjective(model, 8), OptimConfig(learning_rate=1e-3).build(10),
                    key=jax.random.PRNGKey(0), accumulation=2)
    state = train.initial_state()
    rows = jax.device_count()
    train.compile(state, {"text": np.arange(rows * 9, dtype=np.int32).reshape(rows, 9) % 64})
    moving = {"conditional", "tuple", "get-tuple-element", "bitcast", "copy", "parameter"}
    speculated = sorted(name for name, instruction in hlo_instructions(train.executable.as_text()).items()
                        if instruction.op not in moving
                        and any(".mu" in read or ".nu" in read for read in instruction.reads))
    assert not speculated, speculated


def test_a_given_loss_scale_starts_each_state_of_its_own():
    """A trainer given its dynamic loss scale (`Trainer(dynamic_scale=...)`) starts each state from
    a copy of it: the step donates its state, so a scale of arrays, as Flax's updates leave one, is
    still whole after a step for the next state and for another trainer given it."""
    scale = jax.tree.map(jnp.asarray, dynamic_scale_lib.DynamicScale(growth_interval=1))
    first = Trainer(Tiny(), optax.sgd(.03), key=jax.random.PRNGKey(9), dynamic_scale=scale)
    state = first.initial_state()
    batch = batches()[0]
    first.compile(state, batch)(state, batch)
    again = Trainer(Tiny(), optax.sgd(.03), key=jax.random.PRNGKey(9), dynamic_scale=scale)
    for trainer in (first, again):
        held = trainer.initial_state().scale
        np.testing.assert_array_equal(np.asarray(held.scale), np.asarray(scale.scale))


CLIPPED_STEP_SEEDS = 64
"""Check 1's declared seeds for a QK-Clip run's stepped kernels against the paper's, sized paired by
`seeds_for` on armada."""


def test_a_clipped_run_stores_its_stepped_kernels_times_the_clip():
    """A QK-Clip run's step stores each query and key kernel as the paper's `gamma * (W + update)`,
    its rescale of the stepped kernels by the root of each head's `tau / s_max` (Megatron Core's
    `clip_qk`): on par with the paper's equations (tools/muonclip_reference.py) over drawn kernels and
    tokens (check 1), the truth the paper's on the float64 model. The update is a constant, so every
    arm steps by the same `W + update` and the clip alone sets the kernels apart; the threshold sits
    below every head's logits, so each head clips. Folding the rescale into an update,
    `(gamma - 1) W + gamma u`, rounded at `W`'s scale twice (ledger 1659)."""
    from dew.data.chat import Role
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective
    from dew.optim import scale_by_qk_clip
    from tools.muonclip_reference import clip_qk_kernel, clip_scale

    tau, step, rows = 1e-3, 1e-3, jax.device_count()
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                              max_seq_len=16, qk_norm=False)

    def objective_of(of):
        return LMObjective(of, 10, head_chunks=1, loss_role=Role.ASSISTANT, qk_stats=True)

    constant = optax.stateless(lambda updates, _: jax.tree.map(lambda g: jnp.full_like(g, step), updates))
    train = Trainer(objective_of(model), optax.chain(constant, scale_by_qk_clip(tau)),
                    key=jax.random.PRNGKey(7))
    initial = train.initial_state()
    # The key as its bits, wrapped again each run: a key array reads back as the device array itself.
    impl = jax.random.key_impl(initial.key)
    host = jax.device_get(initial.replace(key=jax.random.key_data(initial.key)))
    roles = np.full((rows, 11), int(Role.ASSISTANT), np.int32)
    compiled = {}

    def draw(seed):
        rng = np.random.default_rng(seed)
        params = jax.tree.map(lambda leaf: (rng.standard_normal(np.shape(leaf)) * 0.5).astype(leaf.dtype),
                              host.variables["params"])
        return params, rng.integers(0, 16, size=roles.shape).astype(np.int32)

    def batch_of(tokens):
        return {"text": jnp.asarray(tokens), "text_roles": jnp.asarray(roles)}

    def attention(params):
        return params["layers_0"]["self_attn"]

    @dew_programs
    def trained(params, tokens):
        # The step donates its state, so each run places its own copy.
        key = jax.random.wrap_key_data(host.key, impl=impl)
        state = jax.device_put(host.replace(variables={**host.variables, "params": params}, key=key))
        if "step" not in compiled:
            compiled["step"] = train.compile(state, batch_of(tokens))
        after, *_ = compiled["step"](state, batch_of(tokens))
        held = attention(jax.device_get(after.variables["params"]))
        return {proj: held[proj]["kernel"] for proj in ("q_proj", "k_proj")}

    def paper(of):
        def clipped(params, tokens):
            # The logits the step's own loss sows, on the weights before the step.
            info = Step(jnp.array(0), jax.random.PRNGKey(0), None)
            _, aux = objective_of(of).loss({**host.variables, "params": params}, batch_of(tokens), info)
            s_max = jnp.max(aux.qk_stats["layers_0"]["self_attn"]["max_logits"][0], axis=0)
            gamma = clip_scale(s_max, tau)
            return {proj: clip_qk_kernel(attention(params)[proj]["kernel"] + step, gamma, 2)
                    for proj in ("q_proj", "k_proj")}
        return clipped

    assert_as_exact(trained, paper(model), draw, "a clipped run's stepped kernels", seeds=CLIPPED_STEP_SEEDS,
                    truth_fn=paper(model.clone(dtype=jnp.float64)))
