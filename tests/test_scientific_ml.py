"""High-precision scientific ML: Fourier feature pyramids, Gauss-Newton steps and float64 runs.

A pyramid level reads its grid by the trigonometric interpolant of beignet's Eq. 5 and its derivative is
that interpolant's spectral derivative (Eq. 8), each on par with the same sums in NumPy from their float64
evaluation (check 1). A `Residuals` objective with a damping states the damped Gauss-Newton step as its
gradient, in the primal form where the residuals outnumber the parameters and the dual form where they do
not, each on par with NumPy's solve of the same normal equations, and the trainer takes that step whole.
A float64 run keeps float64 through its parameters, optimizer state, loss and checkpoints.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import as_given, assert_as_exact, seeds_for

from dew.data import Memory
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
    """A linear map of `features` inputs to one output, no bias, its weights stored in `param_dtype`."""

    features: int
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, use_bias=False, kernel_init=nn.initializers.zeros, dtype=self.param_dtype,
                        param_dtype=self.param_dtype)(x)[..., 0]


class LeastSquares(Residuals):
    """The residuals of a line through the batch's rows, `x @ w - y`, one a row."""

    def __init__(self, features: int, damping: float | None, param_dtype=jnp.float32):
        self.model = self.bind_model(Line(features, param_dtype))
        self.features, self.damping = features, damping

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, self.features)))

    def residuals(self, variables, batch, step: Step):
        return (self.model.apply(variables, batch["x"]) - batch["y"])[:, None]


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
    """A least-squares objective with damping 1e-3 states, as its gradient at w = 0, the damped
    Gauss-Newton step: on par with NumPy's float32 solve of the same normal equations (or their Gram
    form, where the rows are fewer) from its float64 solve (check 1)."""
    objective = LeastSquares(features, damping=1e-3)
    params = objective.init(jax.random.key(0))["params"]
    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def draw(seed):
        generator = np.random.default_rng([0x6A, rows, seed])
        return (generator.normal(size=(rows, features)).astype(np.float32),
                generator.normal(size=rows).astype(np.float32))

    def dews(x, y):
        gradient = jax.grad(lambda p: objective.scalar_loss({"params": p}, {"x": x, "y": y}, step)[0])(params)
        return gradient["Dense_0"]["kernel"][:, 0]

    numpy = as_given(lambda x, y: solved(x, y, 1e-3))
    exact = as_given(lambda x, y: solved(x.astype(np.float64), y.astype(np.float64), 1e-3))
    label = f"gauss-newton {rows}x{features}"
    assert_as_exact(jax.jit(dews), numpy, draw, label, seeds=seeds_for(numpy, draw, label, truth_fn=exact),
                    truth_fn=exact)


def test_a_float64_gauss_newton_run_keeps_float64_through_its_state_and_checkpoints(tmp_path):
    """Under x64, a least-squares run of float64 weights steps by Gauss-Newton with `optax.sgd(1.0)`, and
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
        objective = LeastSquares(4, damping=1e-12, param_dtype=jnp.float64)
        checkpoints = Checkpoints(str(tmp_path / "run"))
        Trainer(objective, optax.sgd(1.0), key=0, layout=LAYOUT, checkpoints=checkpoints).fit(rows, steps=1)
        state = Trainer(objective, optax.sgd(1.0), key=0, layout=LAYOUT, checkpoints=checkpoints).fit(
            rows, steps=3)
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
