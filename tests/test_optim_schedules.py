"""Schedule records and parameter groups against torch's schedulers and Adam.

The references are tests/fixtures/torch_optim/reference.npz, which
tools/torch_optim_reference.py writes with torch 2.14.1 on CPU. torch's
schedulers compute in float64, so a schedule is torch's where Dew's, run in
float64, is the oracle's to its own float64 rounding (the gate's oracle rule,
`assert_computes_the_oracle`): any other definition parts from it by orders of
magnitude more.
"""

import dataclasses
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.optim import (
    Adam,
    Cosine,
    Exponential,
    Linear,
    OneCycle,
    OptimConfig,
    ParamGroup,
    Power,
    post_step_order,
    stepped,
)
from dew.training import TrainerAdvanced

TORCH = np.load(Path(__file__).resolve().parent / "fixtures" / "torch_optim" / "reference.npz")
EPOCHS, PER_EPOCH = 6, 2
MOMENTUM = OneCycle(peak_value=0.85, init_value=0.95, end_value=0.95)
"""torch's Adam beta1 cycle under `OneCycleLR(cycle_momentum=True)`."""

WEIGHT_DECAY = 1e-2
"""The reference run's kernel decay (tools/torch_optim_reference.py)."""


def assert_torchs(schedule, steps: int, torch_values, label: str) -> None:
    """`schedule` over `steps` is torch's (`torch_values`, float64): Dew's run in float64 within the
    float64 rounding of its own chain of operations (`chain_roundings`) of torch's."""
    with jax.enable_x64(new_val=True):
        built = schedule.schedule(steps)
        computed = np.asarray([built(count) for count in range(steps)], np.float64)
        roundings = chain_roundings(jax.make_jaxpr(built)(0))
    assert_computes_the_oracle(computed, torch_values, label, roundings=roundings)


def test_one_cycle_is_torchs_one_cycle_lr():
    """Dew's one cycle at a peak of 5e-3, and one with its own start, end and
    warmup share, is torch's OneCycleLR (`assert_torchs`)."""
    assert_torchs(OneCycle(peak_value=5e-3), 40, TORCH["one_cycle/lr"], "one cycle")
    custom = OneCycle(peak_value=0.1, init_value=0.01, end_value=1e-4, warmup_fraction=0.25)
    assert_torchs(custom, 23, TORCH["one_cycle_custom/lr"], "a custom one cycle")


def test_one_cycle_cycles_adams_b1_as_torch_does():
    assert_torchs(MOMENTUM, 40, TORCH["one_cycle/b1"], "one cycle's b1")


def test_exponential_is_torchs_exponential_lr_stepped_to_decay_steps_then_held():
    """The geometric decay from 12 to 0.23 over 10 steps, then 0.23 for the
    rest, plus the offset torch has no field for."""
    assert_torchs(Exponential(init_value=12.0, end_value=0.23, decay_steps=10), 16, TORCH["exponential/lr"],
                  "exponential")
    decayed = Exponential(init_value=12.0, end_value=0.23, decay_steps=10, offset=0.27)
    assert_torchs(decayed, 16, TORCH["exponential/lr"] + 0.27, "an offset exponential")


def test_cosine_without_warmup_is_torchs_cosine_annealing():
    assert_torchs(Cosine(peak_value=0.1, warmup_steps=0), 15, TORCH["cosine/lr"], "cosine")


def test_every_holds_each_value_for_an_epoch_as_a_torch_scheduler_stepped_per_epoch():
    """A run of 40 epochs of 3 updates reads epoch e's value at each of its updates."""
    assert_torchs(OneCycle(peak_value=5e-3, every=3), 40 * 3, np.repeat(TORCH["one_cycle/lr"], 3),
                  "one cycle held three updates")


def test_every_counts_the_schedules_fields_in_its_own_steps():
    """`decay_steps` counts epochs: the decay ends after 10 epochs of 4 updates."""
    assert_torchs(Exponential(init_value=12.0, end_value=0.23, decay_steps=10, every=4), 16 * 4,
                  np.repeat(TORCH["exponential/lr"], 4), "exponential held four updates")


FLOAT64_MOMENTS = {"state_dtype": "params"}
"""Adam's moments stored in its float64 parameters' dtype, where a run stores them in float32
(`dew.optim.Adam.state_dtype`): the same algorithm read in float64 for the oracle rule."""


def snn_delays_config(**opts) -> OptimConfig:
    """SNN-delays' optimizer at the reference's size, as records only; `opts` its optimizer's."""
    rate = OneCycle(peak_value=5e-3, every=PER_EPOCH)
    b1 = OneCycle(peak_value=0.85, init_value=0.95, end_value=0.95, every=PER_EPOCH)
    return OptimConfig(optimizer=Adam(**opts), param_groups=(
        ParamGroup("delays", ("delay",), schedule=Cosine(peak_value=0.1, warmup_steps=0, every=PER_EPOCH),
                   bounds=(0.0, 3.0)),
        ParamGroup("weights", ("kernel",), schedule=rate, b1=b1, weight_decay=WEIGHT_DECAY),
        ParamGroup("norms", ("*",), schedule=rate, b1=b1),
    ))


def run(solver: optax.GradientTransformation, initial, grads) -> dict[str, list[jax.Array]]:
    """The parameters after each update, each reading its stored gradients."""
    params = dict(initial)
    state = solver.init(params)
    update = jax.jit(solver.update)
    trace: dict[str, list[jax.Array]] = {name: [] for name in params}
    for step in range(len(grads["kernel"])):
        updates, after = update({name: grads[name][step] for name in params}, state, params)
        # A bounded group's clamp acts on the stepped weights (`stepped`).
        params, state = stepped(params, updates, state, after, post_step_order(solver, params)), after
        for name, value in params.items():
            trace[name].append(value)
    return trace


NAMES = ("kernel", "scale", "delay")


def test_grouped_adam_is_torchs_adam_with_param_groups_a_momentum_cycle_and_a_clamp():
    """Twelve updates over six epochs: Adam with coupled weight decay and
    the beta1 cycle on the kernel, the cycle alone on the scale, and the
    clamped positions on a cosine from 0.1, are torch's two Adams and their
    schedulers: Dew's run in float64 is torch's (`params/`, float64) to the
    float64 rounding of its own chain of operations (the oracle rule)."""
    with jax.enable_x64(new_val=True):
        initial = {name: jnp.asarray(TORCH[f"initial/{name}"], jnp.float64) for name in NAMES}
        grads = {name: jnp.asarray(TORCH[f"grads/{name}"], jnp.float64) for name in NAMES}
        solver = snn_delays_config(**FLOAT64_MOMENTS).build(EPOCHS * PER_EPOCH)
        trace = run(solver, initial, grads)
        roundings = chain_roundings(jax.make_jaxpr(
            lambda initial, grads: run(solver, initial, grads))(initial, grads))
    for name in NAMES:
        assert_computes_the_oracle(np.stack(trace[name]), TORCH[f"params/{name}"], name, roundings=roundings)


def test_post_step_effects_act_in_the_order_a_named_chain_declares():
    """A named chain's post-step effects act on the stepped weights in the order it declares them
    (docs/design/training.md: in the chain's order), its state a dict by name, which JAX flattens by sorted
    names: QK-Clip's rescale by 0.5 named `z_clip`, then a clamp into [0, 2] named `a_bound`, take a
    stepped 3 to 1.5, where the clamp first gave 1.0. Each effect is judged by its own counter where the
    step's argument holds the chain sorted by name, as a step's argument comes in: the clamp into [0, 1]
    the update deferred leaves the rescaled 1.5. The order is the chain's own (`post_step_order`), read from
    its init however a wrapper's state comes back sorted: `optax.MultiSteps`' `lax.cond` sorts it too."""
    from dew.optim import BoundsState, QKClipState, _keep_within, scale_by_qk_clip

    params, path = {"q_proj": {"kernel": jnp.asarray([[3.0]])}}, "['q_proj']['kernel']"
    unmoved = jax.tree.map(jnp.zeros_like, params)

    def state(clipped: int, bounded: int, rescale: float, upper: float, order=("z_clip", "a_bound")) -> dict:
        held = {"z_clip": QKClipState(jnp.asarray(clipped), {path: jnp.asarray([rescale])},
                                      {path: jnp.asarray([0.0])}),
                "a_bound": BoundsState(jnp.asarray(bounded), {path: jnp.asarray([0.0, upper])})}
        return {name: held[name] for name in order}

    chain = optax.named_chain(("z_clip", scale_by_qk_clip(1.0)), ("a_bound", _keep_within(0.0, 2.0)))
    declared = post_step_order(chain, params)
    assert [path[-1] for path in post_step_order(optax.MultiSteps(chain, 2), params)] == ["z_clip", "a_bound"]
    moved = stepped(params, unmoved, state(0, 0, 1.0, 2.0, order=("a_bound", "z_clip")),
                    state(1, 1, 0.5, 2.0, order=("a_bound", "z_clip")), declared)
    np.testing.assert_array_equal(moved["q_proj"]["kernel"], [[1.5]])
    deferred = stepped(params, unmoved, state(5, 2, 1.0, 1.0, order=("a_bound", "z_clip")),
                       state(6, 2, 0.5, 1.0), declared)
    np.testing.assert_array_equal(deferred["q_proj"]["kernel"], [[1.5]])


class UnpooledClip(NamedTuple):
    """A QK-Clip state as Dew held it before its pooled maxima: the updates it took part in and its
    rescales, the fields a checkpoint of then holds."""

    steps: jax.Array
    rescales: dict


def unpooled_clip(tau: float) -> optax.GradientTransformationExtraArgs:
    """A transform whose state is a QK-Clip state of the form before its pooled maxima (`UnpooledClip`),
    stepping aside: what a run saved before them held."""
    from dew.optim import _rescales

    return optax.GradientTransformationExtraArgs(
        lambda params: UnpooledClip(jnp.zeros([], jnp.int32), _rescales(params, None, tau, None)),
        lambda updates, state, params=None, **extra: (updates, state))


@pytest.mark.parametrize("groups", [None, 2], ids=["one copy", "diloco"])
def test_a_run_saved_before_the_clip_pooled_its_maxima_resumes_and_steps_on(tmp_path, groups):
    """A checkpoint whose QK-Clip state holds no pooled maxima, as one saved before the state held them,
    restores into a run of this Dew through the translation at the restore boundary
    (`dew.checkpoints._clip_pooled_unsaved`: no microstep held back, zero) and steps on, its clip taking
    part in the update it resumes with: a run's own, and a DiLoCo run's groups' inner one under its outer
    SGD, whose resume was refused, its inner pooled maxima missing (the morning audit of 2026-11-17,
    PRODUCT-AM-02)."""
    from dew.checkpoints import Checkpoints
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.optim import scale_by_qk_clip
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=12, dtype="float32", qk_norm=False)
    rows = 2 * jax.device_count()
    text = (np.arange(rows * 8, dtype=np.int32) % 13).reshape(rows, 8)
    data = Pipeline(Memory({"text": text})).repeat().batch(rows)

    def fit(clip, steps):
        from dew.training import DiLoCo, MeshSpec

        checkpoints = Checkpoints(str(tmp_path))
        chain = optax.named_chain(("sgd", optax.sgd(1e-2)), ("z_clip", clip))
        grouped = {} if groups is None else {"mesh": MeshSpec(diloco=groups).build(jax.devices()[:groups]),
                                             "advanced": TrainerAdvanced(diloco=DiLoCo(every=2))}
        state = Trainer(LMObjective(model, seq_len=7), chain, key=jax.random.key(0), checkpoints=checkpoints,
                        **grouped).fit(data, steps=steps, checkpoint_every=1)
        checkpoints.wait()
        return state

    fit(unpooled_clip(100.0), 1)
    resumed = fit(scale_by_qk_clip(100.0), 2)
    clip = (resumed.opt_state if groups is None else resumed.inner.opt_state)["z_clip"]
    assert int(resumed.step) == 2 and np.all(np.asarray(clip.steps) == 1)
    assert all(np.all(np.asarray(pooled) == 0) for pooled in clip.pooled.values())


@pytest.mark.parametrize("wrapped", ["bare", "plain", "multisteps"])
def test_a_fit_moves_its_weights_by_its_named_chains_post_step_effects_in_its_order(wrapped):
    """Through `Trainer.fit`, bare and wrapped as a plain transformation (which declares no extra
    arguments), a named chain's QK-Clip (named `z_clip`, after a zero step) rescales the attention kernels
    before its clamp into [0.1, 1] (named `a_bound`) acts, the order the chain declares where its state's
    names sort the other way (row 5964): every stored weight is within the bounds, where the rescale after
    the clamp would leave a clipped kernel's under them, and the clip fired, its kernels shrunk under the
    bound and lifted to it, the plain wrapper's clip handed the logits' maxima as the bare chain's is (row
    6552). Inside `optax.MultiSteps` of two steps, whose inner update runs at its window's last step alone,
    the run trains two steps and its window's clip fires as the bare chain's does: each step's commit
    pools its maxima in the clip's state (`dew.optim.pooled_update`), where the run was refused (row 6685)."""
    from flax.traverse_util import flatten_dict

    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.optim import _keep_within, scale_by_qk_clip
    from dew.training import Trainer

    lower, upper = 0.1, 1.0
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=12, dtype="float32", qk_norm=False)
    chain = optax.named_chain(("z_clip", optax.chain(optax.sgd(0.0), scale_by_qk_clip(1e-6))),
                              ("a_bound", _keep_within(lower, upper)))
    optimizer = {"bare": chain, "plain": optax.GradientTransformation(chain.init, chain.update),
                 "multisteps": optax.MultiSteps(chain, 2).gradient_transformation()}[wrapped]
    trainer = Trainer(LMObjective(model, seq_len=7), optimizer, key=jax.random.key(0))
    rows = 2 * jax.device_count()
    text = (np.arange(rows * 8, dtype=np.int32) % 13).reshape(rows, 8)
    data = Pipeline(Memory({"text": text})).repeat().batch(rows)
    final = trainer.fit(data, steps=2 if wrapped == "multisteps" else 1)
    params = flatten_dict(jax.device_get(final.variables["params"]))
    for path, leaf in params.items():
        assert np.all((leaf >= lower) & (leaf <= upper)), (path, np.min(leaf), np.max(leaf))
    attention = [leaf for path, leaf in params.items() if {"q_proj", "k_proj"} & set(path)]
    assert attention and all(np.all(leaf == np.float32(lower)) for leaf in attention), attention


def test_a_bounded_group_clamps_its_parameters_and_no_others():
    params = {"delay": jnp.asarray([0.05, 2.95]), "kernel": jnp.asarray([0.05, 2.95])}
    config = OptimConfig(optimizer=Adam(), lr=0.5, param_groups=(
        ParamGroup("delays", ("delay",), bounds=(0.0, 3.0)), ParamGroup("rest", ("*",))))
    solver = config.build(1)
    grads = {"delay": jnp.asarray([1.0, -1.0]), "kernel": jnp.asarray([1.0, -1.0])}
    initial = solver.init(params)
    updates, state = solver.update(grads, initial, params)
    moved = stepped(params, updates, initial, state, post_step_order(solver, params))
    # The clamp is exact: the bounds themselves, where the step overshot them; no other group's
    # parameters are clamped, the step applied as it is.
    np.testing.assert_array_equal(moved["delay"], [0.0, 3.0])
    np.testing.assert_array_equal(moved["kernel"], optax.apply_updates(params, updates)["kernel"])
    # The kernel takes Adam's first step unclamped, 0.5 * g / (|g| + 1e-8) off: Dew's step in float64 is
    # that formula's to its own float64 rounding (the oracle rule).
    with jax.enable_x64(new_val=True):
        wide = {name: jnp.asarray(value, jnp.float64) for name, value in params.items()}
        widened = {name: jnp.asarray(value, jnp.float64) for name, value in grads.items()}
        read = dataclasses.replace(config, optimizer=Adam(**FLOAT64_MOMENTS)).build(1)

        def step(params, grads):
            first = read.init(params)
            updates, state = read.update(grads, first, params)
            return stepped(params, updates, first, state, post_step_order(read, params))
        kernel = np.asarray(step(wide, widened)["kernel"])
        roundings = chain_roundings(jax.make_jaxpr(step)(wide, widened))
    # From the float32 parameters and gradients the step read, widened.
    start, gradient = np.asarray(wide["kernel"]), np.asarray(widened["kernel"])
    adam = start - 0.5 * gradient / (np.abs(gradient) + 1e-8)
    assert_computes_the_oracle(kernel, adam, "the unbounded kernel", roundings=roundings)


@pytest.mark.parametrize("schedule", [
    lambda warmup: Cosine(peak_value=1.0, warmup_steps=warmup, decay_steps=100),
    lambda warmup: Linear(peak_value=1.0, warmup_steps=warmup, decay_steps=100),
    lambda warmup: Power(peak_value=1.0, warmup_steps=warmup, a=0.5),
], ids=["cosine", "linear", "power"])
def test_a_warmup_may_be_a_share_of_the_run(schedule):
    """A warmup written as a share of the run is the whole updates nearest that share of its updates,
    and the schedule it makes is the one written with those updates: 5% of 200 is 10, and of a
    cosine's own 100-step horizon 5 where the run states none."""
    shared, written = schedule("5%"), schedule(10)
    for step in (0, 3, 9, 10, 11, 50, 99):
        assert float(shared.schedule(200)(step)) == float(written.schedule(200)(step))
    assert shared.warmup(200) == 10
    if isinstance(shared, Cosine):
        assert shared.warmup(None) == 5


@pytest.mark.parametrize("warmup", ["0%", "101%", "5", "five%"])
def test_a_warmup_that_is_neither_steps_nor_a_share_is_refused(warmup):
    with pytest.raises(ValueError):
        Cosine(peak_value=1.0, warmup_steps=warmup, decay_steps=100).schedule(200)



def test_a_one_cycle_rise_may_be_a_share_of_the_run():
    """OneCycle's `warmup_fraction` may be written as a share: `30%` is the schedule 0.3 makes, step for
    step, and its warmup is the steps to its peak, torch's 0.3 * 100 - 1."""
    shared, written = OneCycle(peak_value=1.0, warmup_fraction="30%"), OneCycle(peak_value=1.0,
                                                                                warmup_fraction=0.3)
    for step in (0, 10, 28, 29, 30, 60, 99):
        assert float(shared.schedule(100)(step)) == float(written.schedule(100)(step))
    assert shared.warmup(100) == written.warmup(100) == 29
    assert OptimConfig(schedule=shared).warmups(100) == {"schedule": 29}
    with pytest.raises(ValueError):
        OneCycle(peak_value=1.0, warmup_fraction="30").schedule(100)


def test_a_config_records_the_warmups_of_the_schedules_its_optimizer_runs():
    """A config records the warmups of the schedules its solvers run (`ParamGroup.rate`): where every
    group names its own, the config's rate runs nowhere and is not resolved, so a share of a run with
    no planned length is no error; a group naming none runs the config's rate, recorded once."""
    unused = Power(peak_value=1.0, warmup_steps="5%", a=0.5)
    own = OptimConfig(schedule=unused, param_groups=(
        ParamGroup("weights", ("kernel",), schedule=Cosine(peak_value=1.0, warmup_steps=4, decay_steps=10)),
        ParamGroup("rest", ("*",), schedule=Cosine(peak_value=1.0, warmup_steps=2, decay_steps=10))))
    assert own.warmups(None) == {"param_groups.weights.schedule": 4, "param_groups.rest.schedule": 2}
    inherited = OptimConfig(schedule=unused, param_groups=(
        ParamGroup("weights", ("kernel",)), ParamGroup("rest", ("*",))))
    assert inherited.warmups(200) == {"schedule": 10}
