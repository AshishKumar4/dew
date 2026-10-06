"""Schedule records and parameter groups against torch's schedulers and Adam.

The references are tests/fixtures/torch_optim/reference.npz, which
tools/torch_optim_reference.py writes with torch 2.14.1 on CPU. Dew computes
in float32 and torch's schedulers in float64, so a schedule agrees to float32
rounding of its value.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.config import OptimConfig
from dew.training.optim import Cosine, Exponential, Linear, OneCycle, ParamGroup, _keep_within

TORCH = np.load(Path(__file__).resolve().parent / "fixtures" / "torch_optim" / "reference.npz")
EPOCHS, PER_EPOCH = 6, 2
MOMENTUM = OneCycle(peak=0.85, init=0.95, end=0.95)
"""torch's Adam beta1 cycle under `OneCycleLR(cycle_momentum=True)`."""

# float32 rounding of a value, two ulps of it. Near a cosine's end, where
# `cos + 1` cancels, the error is instead an ulp or two of the peak, which
# `atol` states per schedule as `ULPS * peak`.
SCHEDULE_RTOL = 2e-6
ULPS = 2.4e-7
WEIGHT_DECAY = 1e-2
"""The reference run's kernel decay (tools/torch_optim_reference.py)."""


def values(schedule, steps: int) -> np.ndarray:
    built = schedule.schedule(steps)
    return np.asarray([built(count) for count in range(steps)], np.float64)


def test_one_cycle_is_torchs_one_cycle_lr():
    """Largest difference observed 4.8e-10 at a peak of 5e-3; the largest
    relative one, 3.6e-6, is at the second-to-last step, where the cosine's
    `cos + 1` cancels in float32."""
    np.testing.assert_allclose(values(OneCycle(peak=5e-3), 40), TORCH["one_cycle/lr"],
                               rtol=SCHEDULE_RTOL, atol=ULPS * 5e-3)
    custom = OneCycle(peak=0.1, init=0.01, end=1e-4, warmup_fraction=0.25)
    np.testing.assert_allclose(values(custom, 23), TORCH["one_cycle_custom/lr"],
                               rtol=SCHEDULE_RTOL, atol=ULPS * 0.1)


def test_one_cycle_cycles_adams_b1_as_torch_does():
    np.testing.assert_allclose(values(MOMENTUM, 40), TORCH["one_cycle/b1"], rtol=SCHEDULE_RTOL)


def test_one_cycle_holds_its_end_past_the_run():
    """torch raises past its total steps; the record holds the value of its last step."""
    built = OneCycle(peak=5e-3).schedule(40)
    np.testing.assert_allclose([built(40), built(100)], TORCH["one_cycle/lr"][-1], rtol=SCHEDULE_RTOL)


def test_a_one_cycle_with_no_fall_is_refused():
    with pytest.raises(ValueError, match="no rise or no fall"):
        OneCycle(peak=1.0).schedule(3)


def test_exponential_is_torchs_exponential_lr_stepped_to_decay_steps_then_held():
    """The geometric decay from 12 to 0.23 over 10 steps, then 0.23 for the
    rest, plus the offset torch has no field for."""
    np.testing.assert_allclose(values(Exponential(init=12.0, end=0.23, decay_steps=10), 16),
                               TORCH["exponential/lr"], rtol=SCHEDULE_RTOL)
    np.testing.assert_allclose(values(Exponential(init=12.0, end=0.23, decay_steps=10, offset=0.27), 16),
                               TORCH["exponential/lr"] + 0.27, rtol=SCHEDULE_RTOL)


def test_cosine_without_warmup_is_torchs_cosine_annealing():
    np.testing.assert_allclose(values(Cosine(peak=0.1, warmup_steps=0), 15), TORCH["cosine/lr"],
                               rtol=SCHEDULE_RTOL, atol=ULPS * 0.1)


def test_a_linear_schedule_from_its_peak_to_the_same_end_is_constant():
    """Dew has no constant record; this is the one a group's constant rate uses."""
    np.testing.assert_array_equal(values(Linear(peak=0.3, end=0.3), 7), np.float32(0.3))


def test_every_holds_each_value_for_an_epoch_as_a_torch_scheduler_stepped_per_epoch():
    """A run of 40 epochs of 3 updates reads epoch e's value at each of its updates."""
    stepped = values(OneCycle(peak=5e-3, every=3), 40 * 3)
    np.testing.assert_allclose(stepped, np.repeat(TORCH["one_cycle/lr"], 3), rtol=SCHEDULE_RTOL,
                               atol=ULPS * 5e-3)


def test_every_counts_the_schedules_fields_in_its_own_steps():
    """`decay_steps` counts epochs: the decay ends after 10 epochs of 4 updates."""
    stepped = values(Exponential(init=12.0, end=0.23, decay_steps=10, every=4), 16 * 4)
    np.testing.assert_allclose(stepped, np.repeat(TORCH["exponential/lr"], 4), rtol=SCHEDULE_RTOL)


def test_a_run_too_short_for_one_step_of_the_schedule_is_refused():
    with pytest.raises(ValueError, match="never advances"):
        Cosine(peak=0.1, every=10).schedule(9)
    with pytest.raises(ValueError, match="once every 1 or more"):
        Cosine(peak=0.1, every=0)


def snn_delays_config() -> OptimConfig:
    """SNN-delays' optimizer at the reference's size, as records only."""
    rate = OneCycle(peak=5e-3, every=PER_EPOCH)
    b1 = OneCycle(peak=0.85, init=0.95, end=0.95, every=PER_EPOCH)
    return OptimConfig(optimizer="adam", param_groups=(
        ParamGroup("delays", ("delay",), schedule=Cosine(peak=0.1, warmup_steps=0, every=PER_EPOCH),
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
        updates, state = update({name: grads[name][step] for name in params}, state, params)
        params = optax.apply_updates(params, updates)
        for name, value in params.items():
            trace[name].append(value)
    return trace


def torch_run():
    names = ("kernel", "scale", "delay")
    initial = {name: jnp.asarray(TORCH[f"initial/{name}"]) for name in names}
    grads = {name: jnp.asarray(TORCH[f"grads/{name}"]) for name in names}
    return initial, grads, {name: TORCH[f"params/{name}"] for name in names}


# optax computes Adam's bias corrections `1 - b ** t` in float32 and torch in
# float64. At b2 = 0.999 that is a relative error of 1.3e-5 in the second
# moment's correction and 6.4e-6 in each step, so a group's bound is that
# error over twelve steps of its largest rate, with room for rounding. Run in
# float64, Dew's positions are within 4.6e-7 of torch's float32 ones.
PARAMETER_ATOL = {"kernel": 1e-6, "scale": 1e-6, "delay": 1e-5}


def test_grouped_adam_is_torchs_adam_with_param_groups_a_momentum_cycle_and_a_clamp():
    """Twelve updates over six epochs: Adam with coupled weight decay and
    the beta1 cycle on the kernel, the cycle alone on the scale, and the
    clamped positions on a cosine from 0.1, against torch's two Adams and
    their schedulers. Largest differences observed: 3.6e-7 on the kernel,
    1.2e-7 on the scale and 5.0e-6 on the positions, whose rate is twenty
    times higher."""
    initial, grads, expected = torch_run()
    trace = run(snn_delays_config().build(EPOCHS * PER_EPOCH), initial, grads)
    for name, reference in expected.items():
        np.testing.assert_allclose(np.stack(trace[name]), reference, rtol=0, atol=PARAMETER_ATOL[name],
                                   err_msg=name)


def test_the_reference_run_moves_with_the_coupled_decay_and_the_momentum_cycle():
    """Dropping either term from the kernel's group leaves the torch run, so
    the parity above covers both."""
    initial, grads, expected = torch_run()
    config = snn_delays_config()
    delays, weights, norms = config.param_groups
    for changed in (ParamGroup("weights", ("kernel",), schedule=weights.schedule, b1=weights.b1),
                    ParamGroup("weights", ("kernel",), schedule=weights.schedule, weight_decay=WEIGHT_DECAY)):
        mutated = OptimConfig(optimizer="adam", param_groups=(delays, changed, norms))
        kernel = np.stack(run(mutated.build(EPOCHS * PER_EPOCH), initial, grads)["kernel"])
        assert np.max(np.abs(kernel - expected["kernel"])) > 10 * PARAMETER_ATOL["kernel"]


def test_adams_weight_decay_is_added_to_the_gradient():
    """On a zero gradient the first coupled-decay step is Adam's normalized
    step of the decay itself, `-lr * sign(param)`; adamw's would be
    `-lr * decay * param`."""
    params = {"w": jnp.asarray([2.0, -0.5, 0.25])}
    solver = OptimConfig(optimizer="adam", learning_rate=0.1, weight_decay=0.01).build(1)
    updates, _ = solver.update({"w": jnp.zeros(3)}, solver.init(params), params)
    np.testing.assert_allclose(updates["w"], -0.1 * np.sign(params["w"]), rtol=1e-4)


def test_a_bounded_group_clamps_its_parameters_and_no_others():
    params = {"delay": jnp.asarray([0.05, 2.95]), "kernel": jnp.asarray([0.05, 2.95])}
    config = OptimConfig(optimizer="adam", learning_rate=0.5, param_groups=(
        ParamGroup("delays", ("delay",), bounds=(0.0, 3.0)), ParamGroup("rest", ("*",))))
    solver = config.build(1)
    grads = {"delay": jnp.asarray([1.0, -1.0]), "kernel": jnp.asarray([1.0, -1.0])}
    updates, _ = solver.update(grads, solver.init(params), params)
    stepped = optax.apply_updates(params, updates)
    np.testing.assert_allclose(stepped["delay"], [0.0, 3.0], atol=1e-7)
    np.testing.assert_allclose(stepped["kernel"], [-0.45, 3.45], rtol=1e-5)


def test_keep_within_lands_a_parameter_on_its_bound():
    params = jnp.asarray([0.3, 0.7, 2.0])
    updates, _ = _keep_within(0.0, 1.0).update(jnp.asarray([-1.0, 0.1, -0.5]), optax.EmptyState(), params)
    np.testing.assert_allclose(params + updates, [0.0, 0.8, 1.0], atol=1e-7)


def test_bounds_from_high_to_low_are_refused():
    with pytest.raises(ValueError, match="lower to a higher"):
        ParamGroup("delays", ("delay",), bounds=(3.0, 0.0))


def test_a_b1_schedule_on_an_optimizer_without_b1_is_refused():
    config = OptimConfig(optimizer="muon", param_groups=(ParamGroup("all", ("*",), b1=MOMENTUM),))
    with pytest.raises(ValueError, match="schedules b1"):
        config.build(10)
