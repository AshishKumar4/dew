"""The reverse process as one scan over a time grid."""

from collections.abc import Callable, Sequence
from typing import overload

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax.core import Tracer
from jax.typing import ArrayLike

from dew.diffusion.discrete import DiscreteDenoiser
from dew.diffusion.process import Denoiser
from dew.nn.autoencoders import AutoEncoder
from dew.sampling.guidance import Guidance, Walk, guidance_modules
from dew.sampling.solvers import RowKeys, RowSolver, Solver


@overload
def sample[StateT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: int | None = None,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array | RowKeys,
    times: ArrayLike | Sequence[float] | None = None,
    initial: StateT | None = None,
    final_denoise: bool = True,
    record: None = None,
) -> jax.Array: ...


@overload
def sample[StateT, RecordT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: int | None = None,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array | RowKeys,
    times: ArrayLike | Sequence[float] | None = None,
    initial: StateT | None = None,
    final_denoise: bool = True,
    record: Callable[[jax.Array, StateT, jax.Array], RecordT],
) -> tuple[jax.Array, RecordT]: ...


def sample[StateT, RecordT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: int | None = None,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array | RowKeys,
    times: ArrayLike | Sequence[float] | None = None,
    initial: StateT | None = None,
    final_denoise: bool = True,
    record: Callable[[jax.Array, StateT, jax.Array], RecordT] | None = None,
) -> jax.Array | tuple[jax.Array, RecordT]:
    """Run the reverse process from `x_T` with `solver` and return the final sample.

    The time grid runs from T to 0. The solver takes one step across each
    interval, and the result is the model's clean prediction at the last
    point. Pass exactly one of `steps` and `times`. With `steps`, the
    process supplies its grid (`times(steps)`): that many points for a
    Gaussian process, and MDLM's grid of that many reveal steps for a masked
    one (`DiscreteProcess.times`). An explicit `times` grid is
    used as the trajectory, descending and concrete, for a source whose
    solver pairs its own sigma and model-time tables. Its length sets the
    number of steps, so a grid of `steps + 1` points ending on the terminal
    is allowed, and a single point takes no solver step.
    `final_denoise=False` returns the last point's state without the closing
    clean prediction, which is how those solvers end.

    A caller that compiles one walk for many grids of a length passes `times`
    as data, traced, with `initial`, the solver's state before the first step
    that it built on the host from the concrete grid (`solver.init`), whose
    checks and tables need the grid's values.

    `record` reads each step's new state, the solver's state after it and
    the clean prediction the step took, and the sample comes back with those
    readings stacked over the steps, `(sample, records)`; Flow-GRPO records
    each transition and its density this way (`FlowSDE.trajectory`), and a
    figure of the walk each step's clean prediction. A record needs at least
    one step.

    `denoise` is `process.denoiser(...)`, which holds the process the solver
    reads, and `guidance` wraps it. Every step's noise comes from `key`
    folded with the step index, so a trajectory is reproducible from one
    key.

    `key` may be `RowKeys`, one key a row, for a solver that draws row by row
    (`RowSolver`, `Unmask`): each row's draws are its own key's, folded with the
    step index, while the model still runs once over the batch. Another
    solver refuses them by name.

    The trajectory is one `lax.scan`, traced at each call. Under a caller's
    `jax.jit` it compiles once, which is how the pipelines and objectives
    call it. Called eagerly, it traces and compiles the scan again every
    time.
    """
    if (steps is None) == (times is None):
        raise ValueError("pass exactly one of steps and times")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    key = _sampling_key(key, solver, x_T.shape[0])
    process = denoise.process

    # A process whose model predicts over an interval reads, at each step,
    # the interval to the next grid point.
    spanned = denoise if isinstance(denoise, Denoiser) and denoise.process.interval else None
    with jax.ensure_compile_time_eval():
        if times is None:
            assert steps is not None
            times = process.times(steps)
        else:
            times = jnp.asarray(times)
            if times.ndim != 1 or times.shape[0] < 1:
                raise ValueError(f"times must be a descending grid of at least one point, got {times.shape}")
        if isinstance(times, Tracer):
            if initial is None:
                raise ValueError("a traced grid needs the solver state its caller built from the "
                                 "concrete grid (initial=)")
        elif bool(jnp.any(~jnp.isfinite(times))) or bool(jnp.any(jnp.diff(times) > 0)):
            raise ValueError("times must be a finite descending grid")
    count = times.shape[0] - 1
    walk = Walk.over(denoise, guidance, count)
    batch = x_T.shape[0]
    guided = walk.init(x_T)
    if times.shape[0] == 1:
        if record is not None:
            raise ValueError("a record reads the steps, and a grid of one point takes none")
        return walk.at(guided, count)(x_T, jnp.full((batch,), times[0]))[0] if final_denoise else x_T

    if initial is None:
        with jax.ensure_compile_time_eval():
            initial = solver.init(x_T, times, process, key=key)

    def scanned[ReadT](read: Callable[[jax.Array, StateT, jax.Array], ReadT]) -> tuple[jax.Array, ReadT]:
        """The sample, and `read` of every step stacked over the steps."""
        def body(carry, inputs):
            x, state, guided = carry
            t, t_next, index = inputs
            t = jnp.full((batch,), t)
            t_next = jnp.full((batch,), t_next)
            stepping = walk if spanned is None else Walk.over(spanned.spanning(t, t_next), guidance, count)
            (denoised, eps), guided = stepping.step(x, t, index, guided)
            x, state = _stepped(solver, key, index)(x, t, t_next, denoised, eps, state,
                                                    process=process, denoise=stepping.at(guided, index))
            return (x, state, guided), read(x, state, denoised)

        (x, _, walked), readings = lax.scan(
            body, (x_T, initial, guided),
            (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
        if final_denoise:
            x = walk.at(walked, count)(x, jnp.full((batch,), times[-1]))[0]
        return x, readings

    if record is None:
        return scanned(lambda x, state, denoised: None)[0]
    return scanned(record)


def _sampling_key(key: int | jax.Array | RowKeys, solver: Solver, rows: int) -> jax.Array | RowKeys:
    """`key` as `sample` steps from it: one typed key, or `RowKeys` of typed keys for a solver that
    draws row by row (`RowSolver`), which another solver refuses by name."""
    from dew.nn.inputs import request_key, request_rows

    if not isinstance(key, RowKeys):
        return request_key(key)
    if not isinstance(solver, RowSolver):
        raise ValueError(f"row keys draw each row by its own key, which a solver does in step_rows "
                         f"(RowSolver), and {type(solver).__name__} draws the batch from one key")
    keys = request_rows(jnp.asarray(key.keys) if isinstance(key.keys, (list, tuple)) else key.keys)
    if keys.shape != (rows,):
        raise ValueError(f"row keys are one key per row: {rows} rows, keys {keys.shape}")
    return RowKeys(keys)


def _stepped(solver: Solver, key: jax.Array | RowKeys, index: jax.Array) -> Callable[..., tuple]:
    """`solver`'s step at the step `index`, from `key` folded with it: `step_rows` over each row's
    key folded alike for `RowKeys`, `step` over the one key otherwise."""
    if isinstance(key, RowKeys):
        assert isinstance(solver, RowSolver)
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(key.keys, index)
        return lambda x, t, t_next, denoised, eps, state, *, process, denoise: solver.step_rows(
            x, t, t_next, denoised, eps, state, keys, process, denoise)
    folded = jax.random.fold_in(key, index)
    return lambda x, t, t_next, denoised, eps, state, *, process, denoise: solver.step(
        x, t, t_next, denoised, eps, state, folded, process, denoise)


def sampling_modules(model: nn.Module, guidance: Guidance | None,
                     autoencoder: AutoEncoder | None) -> tuple[nn.Module, ...]:
    """Every network a walk of `model` under `guidance` that `autoencoder` decodes runs: the
    denoiser, an autoguide (`guidance_modules`) and the autoencoder's model, which the program
    that walks it compiles for (`compiler_options`), a task's and an objective's alike."""
    return (model, *guidance_modules(guidance), *(() if autoencoder is None else (autoencoder.model,)))


def walk_and_decode(denoise: Denoiser, autoencoder: AutoEncoder | None, params, x_T: jax.Array, *,
                    solver: Solver, guidance: Guidance | None, key: jax.Array, times, final_denoise: bool,
                    initial=None, decode: bool = True) -> tuple[jax.Array | None, jax.Array]:
    """`denoise`'s reverse walk from `x_T`: the images, the latents it reaches decoded by
    `autoencoder` (its weights `params["autoencoder"]`) and clipped to [-1, 1] (None without
    `decode`), and the latents. The one sampling program a task (`dew.inference.TextToImage`) and an
    objective's evaluation and preview (`DiffusionObjective.evaluate`) run."""
    latents = sample(denoise, x_T, solver=solver, guidance=guidance, key=key, times=times,
                     initial=initial, final_denoise=final_denoise)
    if not decode:
        return None, latents
    images = autoencoder.decode(params["autoencoder"], latents) if autoencoder is not None else latents
    return jnp.clip(images, -1.0, 1.0), latents
