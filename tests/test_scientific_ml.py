"""High-precision scientific ML: Fourier feature pyramids, Gauss-Newton steps and float64 runs.

A pyramid level reads its grid by the trigonometric interpolant of beignet's Eq. 5 and its derivative is
that interpolant's spectral derivative (Eq. 8), each on par with the same sums in NumPy from their float64
evaluation (check 1). A `Residuals` objective with a damping states the damped Gauss-Newton step as its
gradient, in the primal form where the residuals outnumber the parameters and the dual form where they do
not, each on par with NumPy's solve of the same normal equations, and the trainer takes that step whole.
A float64 run keeps float64 through its parameters, optimizer state, loss and checkpoints.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import as_given, assert_as_exact, seeds_for

from dew.data import Memory
from dew.gauss_newton import gauss_newton
from dew.nn.fourier import FourierPyramid, interpolated
from dew.objectives.base import Step
from dew.objectives.residuals import Residuals
from dew.training import Checkpoints, Layout, Trainer

LAYOUT = Layout(min_shard=1, tolerance=1.0)


def spectral(grid: np.ndarray, x: np.ndarray, derivative: int) -> np.ndarray:
    """Eq. 5 (derivative 0) or Eq. 8 (derivative 1) of beignet in NumPy at `grid`'s own dtype: Re of the
    sum over the signed FFT modes k of the grid's DFT over its size, times (2 pi i k)^derivative and
    exp(2 pi i k x), [points, channels]."""
    size = grid.shape[0]
    spectrum = np.fft.fft(grid, axis=0) / size
    complex_type = np.complex64 if grid.dtype == np.float32 else np.complex128
    modes = (np.fft.fftfreq(size, 1 / size)).astype(grid.dtype)
    factor = (2j * np.pi * modes) ** derivative
    phases = np.exp(2j * np.pi * modes[:, None] * x[None, :]).astype(complex_type)
    return np.einsum("kc,kn->nc", (spectrum * factor[:, None]).astype(complex_type), phases).real


@pytest.mark.parametrize("derivative", [0, 1], ids=["values", "slopes"])
def test_a_pyramid_level_reads_its_grid_by_its_interpolant_and_its_slope_by_its_spectrum(derivative):
    """A 64-point grid of 4 channels read at drawn coordinates in [0, 1): the interpolant (`interpolated`)
    and its derivative in x by automatic differentiation are beignet's Eq. 5 and Eq. 8, on par with NumPy's
    float32 sums of the same modes from their float64 evaluation (check 1)."""
    def draw(seed):
        generator = np.random.default_rng([0x5F, derivative, seed])
        return (generator.normal(size=(64, 4)).astype(np.float32),
                generator.uniform(0, 1, 256).astype(np.float32))

    def dews(grid, x):
        if derivative == 0:
            return interpolated(grid, x[:, None])
        return jax.vmap(jax.jacfwd(lambda point: interpolated(grid, point[None, None])[0]))(x)

    numpy = as_given(lambda grid, x: spectral(grid, x, derivative))
    exact = as_given(lambda grid, x: spectral(grid.astype(np.float64), x.astype(np.float64), derivative))
    label = f"pyramid level {('values', 'slopes')[derivative]}"
    assert_as_exact(jax.jit(dews), numpy, draw, label, seeds=seeds_for(numpy, draw, label, truth_fn=exact),
                    truth_fn=exact)


def test_a_pyramid_reads_each_grid_at_its_own_points_and_scales_its_levels():
    """At a level's grid points the interpolant is the grid itself, and each level's features are scaled by
    `level_scale` to its index and all by `global_scale`: a pyramid of grids of two and four points read at
    0 and 1/2 gives the coarse grid's points times 1000 and the fine grid's even points times 1000 * 0.3,
    on par with NumPy's float32 interpolants scaled alike, from those grid values (check 1)."""
    pyramid = FourierPyramid((2, 4), features=3, global_scale=1000.0, level_scale=0.3)

    def draw(seed):
        generator = np.random.default_rng([0x70, seed])
        return (generator.normal(size=(2, 3)).astype(np.float32),
                generator.normal(size=(4, 3)).astype(np.float32))

    def dews(coarse, fine):
        return pyramid.apply({"params": {"level_0": coarse, "level_1": fine}}, jnp.asarray([[0.0], [0.5]]))

    def scaled(coarse, fine, dtype):
        points = np.asarray([0.0, 0.5], dtype)
        return np.concatenate([spectral(coarse.astype(dtype), points, 0) * dtype(1000),
                               spectral(fine.astype(dtype), points, 0) * dtype(1000 * 0.3)], axis=-1)

    numpy = as_given(lambda coarse, fine: scaled(coarse, fine, np.float32))
    # At its grid points a level's interpolant is the grid itself.
    exact = as_given(lambda coarse, fine: np.concatenate(
        [coarse.astype(np.float64) * 1000, fine[::2].astype(np.float64) * 1000 * 0.3], axis=-1))
    label = "pyramid at its grid points"
    assert_as_exact(jax.jit(dews), numpy, draw, label, seeds=seeds_for(numpy, draw, label, truth_fn=exact),
                    truth_fn=exact)


class Line(nn.Module):
    """A linear map of `features` inputs to one output, with a bias where `bias`, its weights stored in
    `param_dtype`."""

    features: int
    param_dtype: jnp.dtype = jnp.float32
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, use_bias=self.bias, kernel_init=nn.initializers.zeros, dtype=self.param_dtype,
                        param_dtype=self.param_dtype)(x)[..., 0]


class LeastSquares(Residuals):
    """The residuals of a line through the batch's rows, `x @ w - y` (`+ b` where it has a `bias`), one a
    row."""

    def __init__(self, features: int, param_dtype=jnp.float32, bias: bool = False):
        self.model = self.bind_model(Line(features, param_dtype, bias))
        self.features = features

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, self.features)))

    def residuals(self, variables, batch, step: Step):
        return (self.model.apply(variables, batch["x"]) - batch["y"])[:, None]


def least_squares_draw(rows: int, features: int, stream: int):
    """A draw of `rows` rows of `features` unit normals and their targets, from `stream`'s seeds."""
    def draw(seed):
        generator = np.random.default_rng([stream, rows, seed])
        return (generator.normal(size=(rows, features)).astype(np.float32),
                generator.normal(size=rows).astype(np.float32))
    return draw


def solved(x: np.ndarray, y: np.ndarray, damping: float) -> np.ndarray:
    """The damped Gauss-Newton step of the residuals `x @ 0 - y` in NumPy at x's own dtype: the normal
    equations' solve where the rows outnumber the features, the Gram form's otherwise."""
    residual = -y
    if x.shape[0] >= x.shape[1]:
        normal = x.T @ x + x.dtype.type(damping) * np.eye(x.shape[1], dtype=x.dtype)
        return np.linalg.solve(normal, x.T @ residual)
    gram = x @ x.T + x.dtype.type(damping) * np.eye(x.shape[0], dtype=x.dtype)
    return x.T @ np.linalg.solve(gram, residual)


@pytest.mark.parametrize("rows,features", [(64, 8), (8, 64)], ids=["primal", "dual"])
def test_a_damped_gauss_newton_step_is_the_normal_equations_solve(rows, features):
    """A least-squares objective states its residuals' system where its step asks (`Step.equations`), and
    Gauss-Newton's update of damping 1e-3 steps by its solve at w = 0: on par with NumPy's float32 solve
    of the same normal equations (or their Gram form, where the rows are fewer) from its float64 solve
    (check 1). Its loss is the mean square still, its gradient its own."""
    objective = LeastSquares(features)
    params = objective.init(jax.random.key(0))["params"]
    step = Step(jnp.asarray(0), jax.random.key(0), None, equations=True)
    optimizer = gauss_newton(damping=1e-3)
    draw = least_squares_draw(rows, features, 0x6A)

    def dews(x, y):
        statistics = objective.loss({"params": params}, {"x": x, "y": y}, step)
        update, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params,
                                     statistics=statistics)
        # The weights start at zero, so the update is the solve's step, negated.
        return -update["Dense_0"]["kernel"][:, 0]

    # No derivative carries the step: the loss's gradient is the mean square's own, stated system or not.
    x, y = draw(0)
    gradients = [jax.grad(lambda p, asked=asked: objective.scalar_loss({"params": p}, {"x": x, "y": y},
                                                                      asked)[0])(params)
                 for asked in (step, Step(jnp.asarray(0), jax.random.key(0), None))]
    for stated, plain in zip(jax.tree.leaves(gradients[0]), jax.tree.leaves(gradients[1]), strict=True):
        np.testing.assert_array_equal(np.asarray(stated), np.asarray(plain))
    numpy = as_given(lambda x, y: solved(x, y, 1e-3))
    exact = as_given(lambda x, y: solved(x.astype(np.float64), y.astype(np.float64), 1e-3))
    label = f"gauss-newton {rows}x{features}"
    assert_as_exact(jax.jit(dews), numpy, draw, label, seeds=seeds_for(numpy, draw, label, truth_fn=exact),
                    truth_fn=exact)


@pytest.mark.parametrize("damping", [1e-3, 0.0], ids=["damped", "undamped"])
@pytest.mark.parametrize("rows,features", [(64, 8), (8, 64)], ids=["primal", "dual"])
def test_a_gauss_newton_step_solves_only_its_real_rows(rows, features, damping):
    """A batch whose last quarter of rows its mask marks unreal (`VALID_ROWS`, a share's exhausted rows
    on the examples clock, or an evaluation's fill) states the system of its real rows alone, as its
    loss counts them: its step, damped or at Gauss-Newton's default of none, is NumPy's solve of the real
    rows' normal equations (or their Gram form) at w = 0 (check 1), where the unreal rows' residuals
    entered the system and the step solved another objective, and undamped, their empty rows of the Gram
    form left it singular (row 3929)."""
    from dew.batch import VALID_ROWS

    objective = LeastSquares(features)
    params = objective.init(jax.random.key(0))["params"]
    step = Step(jnp.asarray(0), jax.random.key(0), None, equations=True)
    optimizer = gauss_newton(damping=damping)
    draw = least_squares_draw(rows, features, 0x6B)
    real = np.arange(rows) < rows * 3 // 4

    def dews(x, y):
        statistics = objective.loss({"params": params}, {"x": x, "y": y, VALID_ROWS: jnp.asarray(real)}, step)
        update, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params,
                                     statistics=statistics)
        return -update["Dense_0"]["kernel"][:, 0]

    numpy = as_given(lambda x, y: solved(x[real], y[real], damping))
    exact = as_given(lambda x, y: solved(x[real].astype(np.float64), y[real].astype(np.float64), damping))
    label = f"masked gauss-newton {rows}x{features} damping {damping}"
    assert_as_exact(jax.jit(dews), numpy, draw, label, seeds=seeds_for(numpy, draw, label, truth_fn=exact),
                    truth_fn=exact)


def gauss_newton_arm(rows: int, features: int, accumulation: int, optimizer=None, bias: bool = False):
    """One update of a least-squares objective of damping 1e-3 (by `optimizer` where given), from zero
    weights, of a draw's `rows` rows read in `accumulation` microbatches of an accumulation window: its
    weights after it, its bias last where it has one."""
    trainer = Trainer(LeastSquares(features, bias=bias), optimizer or gauss_newton(damping=1e-3), key=0,
                      layout=LAYOUT, accumulation=accumulation)
    template = jax.tree.map(jnp.copy, trainer.initial_state())
    compiled = []

    @as_given
    def stepped(x, y):
        # The step donates its state: each draw steps a copy of the untouched template.
        state = jax.tree.map(jnp.copy, template)
        parts = [{"x": part_x, "y": part_y} for part_x, part_y in
                 zip(np.split(x, accumulation), np.split(y, accumulation), strict=True)]
        if not compiled:
            compiled.append(trainer.compile(state, parts[0]))
        for part in parts:
            state, *_ = compiled[0](state, part)
        assert int(state.updates) == 1
        weights = state.variables["params"]["Dense_0"]
        return np.concatenate([np.asarray(weights["kernel"])[:, 0], np.asarray(weights.get("bias", []))])

    return stepped


UPDATES = {"primal": (64, 8, 2), "short-microbatches": (64, 16, 8), "dual": (32, 64, 4)}
"""Each accumulated update the Gauss-Newton parity reads: its rows, its weights and its microbatches.
Two microbatches of 32 rows over 8 weights; eight of 8 rows over 16 weights, each microbatch shorter
than the weights though the update is not; four of 8 rows over 64 weights, the update itself shorter."""


@pytest.mark.parametrize("named", ["gauss_newton", "dew.gauss_newton:gauss_newton"])
def test_a_grouped_gauss_newton_run_scales_each_groups_leaves_of_one_solve(named):
    """Gauss-Newton under two param groups, a line's kernel and its bias, by its name or its import path:
    one damped solve over both (the rows' features beside a column of ones), each group's leaves of it
    scaled by the group's rate, the kernel's half and the bias's a quarter: on par with NumPy's float32
    solve so scaled from its float64 solve (check 1)."""
    from dew.optim import OptimConfig, ParamGroup

    config = OptimConfig(optimizer=named, learning_rate=0.5, optimizer_opts={"damping": 1e-3},
                         param_groups=(ParamGroup("kernels", ("*kernel",)),
                                       ParamGroup("rest", ("*",), learning_rate_multiplier=0.5)))
    rows, features = 64, 8
    draw = least_squares_draw(rows, features, 0x6D)
    rates = np.asarray([0.5] * features + [0.25])

    def scaled(x, y):
        lifted = np.concatenate([x, np.ones((len(x), 1), x.dtype)], axis=1)
        return -(rates * solved(lifted, y, 1e-3)).astype(x.dtype)

    numpy = as_given(scaled)
    exact = as_given(lambda x, y: scaled(x.astype(np.float64), y.astype(np.float64)))
    grouped = gauss_newton_arm(rows, features, 1, config.build(1), bias=True)
    label = f"grouped gauss-newton {named}"
    assert_as_exact(grouped, numpy, draw, label, truth_fn=exact,
                    seeds=seeds_for(numpy, draw, label, truth_fn=exact, dew_fn=grouped))


@pytest.mark.parametrize("clip", [0.0, 0.05])
def test_a_run_that_names_gauss_newton_and_sets_no_rate_takes_its_whole_damped_step(clip, tmp_path):
    """A run whose command line names Gauss-Newton by its import path and its damping 1e-3, setting no
    learning rate: its rate is Gauss-Newton's own, the whole step (1.0, recorded as the config's), so its
    optimizer steps the damped solve's step from zero, on par with NumPy's float32 solve from its float64
    solve (check 1), with `clip_grads` set that step capped to the clip's norm; and the run's own path
    (`RunConfig.train`) trains the least-squares objective one step with it (rows 3359, 3360)."""
    from dew.config import RunConfig, TrainerConfig
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.gauss_newton import solves_equations

    flags = ["--optim.optimizer", "dew.gauss_newton:gauss_newton", "--optim.optimizer-opts",
             json.dumps({"damping": 1e-3}), "--optim.clip-grads", str(clip)]
    trainer = TrainerConfig(steps=1, checkpoint_final=False, checkpoint_dir=str(tmp_path), eval_every=None,
                            checkpoint_every=None)
    config = dataclasses.replace(RunConfig.cli(flags), trainer=trainer)
    assert config.optim.learning_rate == 1.0
    rows, features = 64, 8
    draw = least_squares_draw(rows, features, 0x6E)

    def capped(step):
        norm = np.linalg.norm(step)
        return step if not clip or norm <= clip else step * (clip / norm)

    numpy = as_given(lambda x, y: capped(-solved(x, y, 1e-3)).astype(x.dtype))
    exact = as_given(lambda x, y: capped(-solved(x.astype(np.float64), y.astype(np.float64), 1e-3)))
    configured = gauss_newton_arm(rows, features, 1, config.optim.build(1))
    label = f"configured whole gauss-newton step clip {clip}"
    assert_as_exact(configured, numpy, draw, label, truth_fn=exact,
                    seeds=seeds_for(numpy, draw, label, truth_fn=exact, dew_fn=configured))
    x, y = draw(0)
    state = config.train(LeastSquares(features), Pipeline(Memory({"x": x, "y": y})).batch(rows), name="run")
    assert int(state.updates) == 1 and solves_equations(state.opt_state)


@pytest.mark.parametrize("groups", [False, True])
def test_a_gauss_newton_cap_its_options_state_holds_in_param_groups_as_without(groups):
    """Gauss-Newton capped by its own option (`optimizer_opts={"clip": 0.05}`, `clip_grads` unset), under one
    param group that claims every parameter or under none: the unit system's whole step of 1 capped to
    0.05, the same either way (row 3360's review)."""
    from dew.gauss_newton import NormalEquations
    from dew.optim import OptimConfig, ParamGroup

    grouped = (ParamGroup("all", ("*",)),) if groups else ()
    config = OptimConfig(optimizer="gauss_newton", learning_rate=1.0,
                         optimizer_opts={"clip": 0.05}, param_groups=grouped)
    solver = config.build(1)
    params = {"w": jnp.zeros((1,), jnp.float32)}
    system = NormalEquations(total=jnp.zeros(()), mass=jnp.ones(()), rows=jnp.asarray(1),
                             pulled=jnp.asarray([1.0], jnp.float32),
                             normal=jnp.asarray([[1.0]], jnp.float32), update=1)
    update, _ = solver.update(params, solver.init(params), params, statistics=system)
    np.testing.assert_array_equal(np.asarray(update["w"]), np.asarray([-0.05], np.float32))


def test_a_capped_gauss_newton_direction_keeps_a_cap_far_below_its_norm():
    """A float32 solved direction of norm 1e18 capped at 1e-30 (`solved_directions(clip=)`) is the
    direction scaled to that cap, as Optax's global-norm clip scales a gradient, not a zero underflowed
    from the cap over the norm (row 3360's review)."""
    from dew.gauss_newton import NormalEquations, solved_directions

    capped = solved_directions(0.0, clip=1e-30)
    params = {"w": jnp.zeros((1,), jnp.float32)}
    system = NormalEquations(total=jnp.zeros(()), mass=jnp.ones(()), rows=jnp.asarray(1),
                             pulled=jnp.asarray([1e18], jnp.float32),
                             normal=jnp.asarray([[1.0]], jnp.float32), update=1)
    direction, _ = capped.update(params, capped.init(params), params, statistics=system)
    np.testing.assert_array_equal(np.asarray(direction["w"]), np.asarray([1e-30], np.float32))


@pytest.mark.parametrize("clip", [0.0, 0.05])
@pytest.mark.parametrize("groups", [False, True])
def test_a_configured_gauss_newton_states_its_optimizer_as_the_runs_saved_before_it(clip, groups):
    """A configured Gauss-Newton optimizer's state, clipped or not, in param groups or not, holds the tree
    the optimizer a run saved before its clip took effect held (the rate injected around the whole update,
    a clip chained before it), so such a run's checkpoint restores onto it (row 3360's review)."""
    from dew.gauss_newton import solved_directions
    from dew.optim import OptimConfig, ParamGroup, _rate_only, param_labels, scheduled_solver

    params = LeastSquares(8, bias=True).init(jax.random.key(0))["params"]
    grouped = (ParamGroup("kernels", ("*kernel",)), ParamGroup("rest", ("*",))) if groups else ()
    config = OptimConfig(optimizer="gauss_newton", optimizer_opts={"damping": 1e-3}, clip_grads=clip,
                         param_groups=grouped)
    if groups:
        solvers = {group.name: group.solver(_rate_only, 1.0, 1, {}) for group in grouped}
        earlier = optax.chain(solved_directions(1e-3), optax.multi_transform(solvers, param_labels(grouped)))
    else:
        earlier = scheduled_solver(gauss_newton, 1.0, 1, {"damping": 1e-3})
    if clip:
        earlier = optax.chain(optax.clip_by_global_norm(clip), earlier)
    assert jax.tree.structure(config.build(1).init(params)) == jax.tree.structure(earlier.init(params))


def test_a_run_configured_for_gauss_newton_steps_its_rate_times_the_damped_solve():
    """A run whose optimizer the command line names (`--optim.optimizer gauss_newton`, its rate 0.5 and
    its damping 1e-3 among `optimizer_opts`), built as `OptimConfig` builds every optimizer
    (`scheduled_solver`), takes half the damped Gauss-Newton step of its batch: on par with half the float64
    solve's step (check 1)."""
    from dew.config import RunConfig

    config = RunConfig.cli(["--optim.optimizer", "gauss_newton", "--optim.learning-rate", "0.5",
                            "--optim.optimizer-opts", json.dumps({"damping": 1e-3})])
    rows, features = 64, 8
    draw = least_squares_draw(rows, features, 0x6C)
    exact = as_given(lambda x, y: -0.5 * solved(x.astype(np.float64), y.astype(np.float64), 1e-3))
    configured = gauss_newton_arm(rows, features, 1, config.optim.build(1))
    halved = gauss_newton_arm(rows, features, 1, gauss_newton(0.5, damping=1e-3))
    label = "configured gauss-newton"
    assert_as_exact(configured, halved, draw, label, truth_fn=exact,
                    seeds=seeds_for(halved, draw, label, truth_fn=exact, dew_fn=configured))


@pytest.mark.parametrize("update", sorted(UPDATES))
def test_an_accumulated_gauss_newton_update_solves_its_whole_batch_once(update):
    """An update of a least-squares objective read in microbatches of an accumulation window takes one
    damped Gauss-Newton step of its whole batch, its microbatches' shape aside: each states its part of
    the update's system, its normal equations where the update's residuals are at least its weights,
    whatever its own rows, and its rows of the dual's where they are fewer, the trainer pools them, and
    the one solve's step is on par with the whole batch's own update from the float64 solve (check 1). A
    solve per microbatch, averaged, would be another step: for the rows (1, 1) and (2, 0) at damping 1,
    -1/4 where the batch's is -1/6."""
    rows, features, parts = UPDATES[update]
    draw = least_squares_draw(rows, features, 0x6B)

    # The weights start at zero, so the step's weights are the solve's step, negated.
    exact = as_given(lambda x, y: -solved(x.astype(np.float64), y.astype(np.float64), 1e-3))
    pooled, whole = gauss_newton_arm(rows, features, parts), gauss_newton_arm(rows, features, 1)
    label = f"accumulated gauss-newton {update}"
    assert_as_exact(pooled, whole, draw, label, truth_fn=exact,
                    seeds=seeds_for(whole, draw, label, truth_fn=exact, dew_fn=pooled))


def test_a_float64_gauss_newton_run_keeps_float64_through_its_state_and_checkpoints(tmp_path):
    """Under x64, a least-squares run of float64 weights steps by Gauss-Newton (`gauss_newton`), and
    resumed from its checkpoint, steps on: its parameters and optimizer state stay float64, and its last
    checkpoint restores them bit for bit, float64 still."""
    # A float64 run sets x64 for its process, its data's worker threads among it, as
    # examples/pinn_burgers.py does.
    jax.config.update("jax_enable_x64", val=True)
    try:
        generator = np.random.default_rng(3)
        x = generator.normal(size=(8, 4))
        y = x @ np.asarray([1.5, -2.0, 0.25, 3.0])
        rows = Memory([{"x": row, "y": target} for row, target in zip(x, y, strict=True)]).repeat().batch(8)
        objective = LeastSquares(4, param_dtype=jnp.float64)
        checkpoints = Checkpoints(str(tmp_path / "run"))
        solver = gauss_newton(damping=1e-12)
        Trainer(objective, solver, key=0, layout=LAYOUT, checkpoints=checkpoints).fit(rows, steps=1)
        state = Trainer(objective, solver, key=0, layout=LAYOUT, checkpoints=checkpoints).fit(rows, steps=3)
        floats = [leaf for leaf in jax.tree.leaves((state.variables, state.opt_state))
                  if jnp.issubdtype(leaf.dtype, jnp.floating)]
        assert floats and all(leaf.dtype == jnp.float64 for leaf in floats)
        checkpoints.wait()
        restored, _ = checkpoints.restore()
        for left, right in zip(jax.tree.leaves(restored["variables"]), jax.tree.leaves(state.variables),
                               strict=True):
            assert left.dtype == np.float64
            np.testing.assert_array_equal(left, np.asarray(right))
    finally:
        jax.config.update("jax_enable_x64", val=False)


def test_a_gauss_newton_update_held_by_a_wrapper_that_drops_its_statistics_is_refused():
    """A Gauss-Newton update inside `optax.MultiSteps`, which hands its inner update no extra arguments and
    pools no system across the steps it holds, is refused as its step is traced, naming the accumulation
    that pools the update's system (`Trainer(accumulation=k)`), never stepped by a system it lacks."""
    held = optax.MultiSteps(gauss_newton(damping=1e-3), every_k_schedule=2).gradient_transformation()
    trainer = Trainer(LeastSquares(4), held, key=0, layout=LAYOUT)
    batch = {"x": np.ones((8, 4), np.float32), "y": np.ones(8, np.float32)}
    with pytest.raises(ValueError):
        trainer.compile(trainer.initial_state(), batch)
    # Unwrapped, the same update compiles: the refusal is the wrapper's dropped statistics.
    trainer = Trainer(LeastSquares(4), gauss_newton(damping=1e-3), key=0, layout=LAYOUT)
    trainer.compile(trainer.initial_state(), batch)
