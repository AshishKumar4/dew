"""Mamba's selective scan held to recorded mamba_ssm (tools/mamba_selective_reference.py): a recorded case's
seeded inputs drawn again without torch, and Dew's portable scan over them with every gradient, whose
float64 run is the recorder's cross-check of its oracle."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba import selective_scan

_recorder = load("mamba_selective_reference")
SEEDED_CASES = _recorder.SEEDED_CASES


def selective_draw(seed: int, case: str) -> dict[str, np.ndarray]:
    """`case`'s inputs at `seed` as the recorder drew them, its narrow operands' values stored in fp32."""
    *shape, dtype, regime = SEEDED_CASES[case]
    return _recorder.operands(np.random.default_rng(seed), *shape,
                              lambda value: value.astype(jnp.dtype(dtype)).astype(np.float32), regime)


def stepwise(x, dt, A, B, C, D, state, starts=None):
    """The selective scan a token at a time, `mamba_selective_state_update` (modeling_mamba.py:129-172) from
    `state`, the state zeroed entering each document's first token where `starts` marks them, its readout
    summed pairwise over the states as `recurrence_algebra.selective_read` defines it: the selective scans'
    token oracle (test_mamba's, test_selective_template's), whose readout order sets a cancelling one's
    rounding near float32's largest."""
    # One token a step of a `lax.scan`, each step the loop's arithmetic, compiled once a step where a Python
    # loop unrolled every order of its tokens (row 9511); the state in the steps' type as the first step's
    # update promotes it.
    held = jnp.result_type(dt, A, state, x, B)

    def token(state, inputs):
        x_t, dt_t, b_t, c_t, start = inputs
        if starts is not None:
            state = jnp.where(start[:, None, None], 0.0, state)
        state = jnp.exp(dt_t[..., None] * A) * state + (dt_t * x_t)[..., None] * b_t[:, None, :]
        terms = c_t[:, None, :] * state
        while terms.shape[-1] > 1:
            # Pairs of the states' terms, an odd one last as it stands.
            paired = terms[..., 0:terms.shape[-1] // 2 * 2:2] + terms[..., 1:terms.shape[-1] // 2 * 2:2]
            terms = jnp.concatenate([paired, terms[..., terms.shape[-1] // 2 * 2:]], axis=-1)
        return state, terms[..., 0] + x_t * D

    marks = jnp.zeros(x.shape[:2], bool) if starts is None else starts
    state, outputs = jax.lax.scan(token, jnp.asarray(state, held),
                                  tuple(jnp.moveaxis(t, 1, 0) for t in (x, dt, B, C, marks)))
    return jnp.moveaxis(outputs, 0, 1), state


NEAR_THE_LARGEST = {"an entering state": None, "a faint write after it": 63,
                    "a cotangent near the largest": None}
"""The selective scans' cases near float32's largest (`near_the_largest`), by the token a document starts at
(None: one sequence)."""

DIFFERENTIATED = ("a cotangent near the largest",)
"""The cases whose gradients a float32 recurrence holds where float64's are float32s: the others' states
near the largest times cotangents of one put their gradients' products past it in any order."""


def near_the_largest(case: str, seed: int) -> tuple:
    """`seed`'s fp32 inputs `(x, dt, A, B, C, D, state)` of a selective scan near float32's largest and the
    cotangents of its output and final state: one row of 128 tokens (64 where differentiated, its
    recurrence's gradient unrolled a token at a time) and 128 channels over four states, unit
    steps, decays of one, no skip, every token but those named writing 2^118 times uniform on [-1, 1] on
    one state (B and C its unit vector), the cotangents standard normal but where named. An entering state:
    the state entering at -1.9 2^127 and two adjacent tokens at a random place writing 2^127 times uniform
    on [1, 1.9] each, every state finite where the two writes summed from a zero state pass float32's
    largest (row 8465, base review f8c72a0d6 F1). A faint write after it: those two at tokens 61 and 62 and
    a document starting at 63 (`NEAR_THE_LARGEST`), whose token writes 2^-126 times uniform on [1, 2], read
    there alone with C 2^126: a write halved there was a subnormal, flushed to zero (head review of
    80a29f74d, P2). A cotangent near the largest: no state entering, inputs 2^-20 times uniform on [-1, 1]
    (so every gradient's float64 lies within float32), the output's cotangent 2^116 times uniform on [-1, 1]
    but at two adjacent tokens, 2^127 times uniform on [1, 1.9] each, and the final state's -1.9 2^127 on the
    state read, every state's cotangent finite where the two summed from a zero cotangent pass the largest
    (row 8486)."""
    rng = np.random.default_rng([7878, seed])
    rows, channels, states = 1, 128, 4
    length = 64 if case in DIFFERENTIATED else 128
    unit = np.broadcast_to(np.eye(states)[rng.integers(states)], (rows, length, states))
    c = np.array(unit)
    x = 2.0 ** 118 * rng.uniform(-1.0, 1.0, (rows, length, channels))
    entering = -1.9 * 2.0 ** 127 * np.ones((rows, channels, 1)) * unit[:, :1]
    dy, dfinal = rng.standard_normal((rows, length, channels)), rng.standard_normal((rows, channels, states))
    at = rng.integers(length - 1) if NEAR_THE_LARGEST[case] is None else NEAR_THE_LARGEST[case] - 2
    if case == "a cotangent near the largest":
        x = 2.0 ** -20 * rng.uniform(-1.0, 1.0, (rows, length, channels))
        dy = 2.0 ** 116 * rng.uniform(-1.0, 1.0, (rows, length, channels))
        dy[:, at:at + 2] = 2.0 ** 127 * rng.uniform(1.0, 1.9, (rows, 2, channels))
        dfinal, entering = entering * np.ones((1, 1, states)) * unit[:, :1], np.zeros_like(entering)
    else:
        x[:, at:at + 2] = 2.0 ** 127 * rng.uniform(1.0, 1.9, (rows, 2, channels))
    if case == "a faint write after it":
        x[:, at + 2] = 2.0 ** -126 * rng.uniform(1.0, 2.0, (rows, channels))
        c[:, at + 2] *= 2.0 ** 126
    steps, decays = np.ones((rows, length, channels)), np.zeros((channels, states))
    inputs = (x, steps, decays, unit, c, np.zeros(channels), entering)
    narrow = tuple(jnp.asarray(t, jnp.float32) for t in inputs)
    return (*narrow, tuple(jnp.asarray(t, jnp.float32) for t in (dy, dfinal)))


def starts_near_the_largest(case: str) -> jax.Array | None:
    """`near_the_largest`'s row's document starts of `case`: its one start, or None."""
    at = NEAR_THE_LARGEST[case]
    return None if at is None else jnp.zeros((1, 128), bool).at[0, at].set(True)


RATES_PAST_THE_RANGE = ("vanishing decays", "a decay of e^-3")
"""`rate_past_the_range`'s cases."""


def rate_past_the_range(case: str, seed: int) -> tuple:
    """`seed`'s fp32 inputs `(x, dt, A, B, C, D, state)` of a selective scan whose rate passes float32's range
    in base 2, and its output's and final state's cotangents: one row of 50 tokens (a ragged chunk's, padded
    with zero steps), 128 channels and 4 states, A = -3 2^126 (its `A log2(e)` past float32's largest),
    inputs, B and C standard normal, no skip, the cotangents standard normal (row 8491). Vanishing decays:
    steps softplus of standard normals, each decay `exp(dt A)` zero, no state entering. A decay of e^-3:
    every step 2^-126, each decay e^-3, the state entering standard normal, which a clamp of the rate taken
    to base 2 first would not give (row 8491's head review)."""
    # The first case's draws as row 8491 drew them, each later case's a stream of its own.
    index = RATES_PAST_THE_RANGE.index(case)
    rng = np.random.default_rng([8491, seed, index] if index else [8491, seed])
    rows, length, channels, states = 1, 50, 128, 4
    x = rng.standard_normal((rows, length, channels))
    dt = np.logaddexp(0.0, rng.standard_normal((rows, length, channels)))
    entering = np.zeros((rows, channels, states))
    if case == "a decay of e^-3":
        dt, entering = np.full_like(dt, 2.0 ** -126), rng.standard_normal(entering.shape)
    b, c = (rng.standard_normal((rows, length, states)) for _ in range(2))
    inputs = (x, dt, np.full((channels, states), -3.0 * 2.0 ** 126), b, c, np.zeros(channels), entering)
    shapes = ((rows, length, channels), (rows, channels, states))
    cotangents = tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32) for shape in shapes)
    return (*(jnp.asarray(t, jnp.float32) for t in inputs), cotangents)


def starts_of(arrays) -> jax.Array | None:
    """Each token's whether it starts a document, from the recorded `seq`, or None where none is packed."""
    seq = arrays.get("seq")
    if seq is None:
        return None
    return jnp.concatenate([jnp.ones_like(seq[:, :1], bool), seq[:, 1:] > seq[:, :-1]], axis=1)


def selective_xla(arrays, dtype=None, scan=selective_scan) -> dict[str, jax.Array]:
    """`scan` (Dew's portable selective scan unless given, its operands `(x, dt, A, B, C, D, state, starts)`)
    over a case's inputs, `x`, the step, `B`, `C` and the cotangent in `dtype` where given: its output and
    each operand's gradient under the recorded cotangent, named as the recording names them."""
    names = ("x", "dt", "A", "B", "C", "D")
    narrow = ("x", "dt", "B", "C")
    operands = tuple(jnp.asarray(arrays[name], dtype if name in narrow else None) for name in names)
    starts = starts_of(arrays)

    def run(*operands):
        return scan(*operands, None, starts)[0]

    out = jax.eval_shape(run, *operands)
    found = with_gradients(run)(*operands, jnp.asarray(arrays["dy"], out.dtype))
    return dict(zip(("y", *names), found, strict=True))


MAMBA_SELECTIVE_SIDE = '''
import sys, time, numpy as np, torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
begun = time.perf_counter()
given = np.load(sys.argv[1])
kinds = dict(zip(str(given["leaves"]).split(","), str(given["dtypes"]).split(",")))
held = {name: torch.from_numpy(given[name]).to("cuda", getattr(torch, kinds[name])).requires_grad_()
        for name in kinds}
draws, rows, length, channels = held["x"].shape
states = held["a"].shape[-1]
# Each draw its own channels (`[B, N D, L]`) and its own group of B and C (`[B, N, states, L]`), its decays
# and skip its channels' (`[N D, states]`, `[N D]`): one call of the draws, each a group of the upstream's.
u, delta = (held[name].permute(1, 0, 3, 2).reshape(rows, draws * channels, length) for name in ("x", "dt"))
b, c = (held[name].permute(1, 0, 3, 2) for name in ("b", "c"))
a, skip = held["a"].reshape(draws * channels, states), held["skip"].reshape(draws * channels)
bounds = [int(x) for x in given["bounds"]]
pieces = [selective_scan_fn(u[..., lo:hi].contiguous(), delta[..., lo:hi].contiguous(), a,
                            b[..., lo:hi].contiguous(), c[..., lo:hi].contiguous(), skip.float())
          for lo, hi in zip(bounds[:-1], bounds[1:])]
y = torch.cat(pieces, dim=-1).reshape(rows, draws, channels, length).permute(1, 0, 3, 2)
torch.cuda.synchronize()
ran = time.perf_counter()
dy = torch.from_numpy(given["dy"]).to("cuda", torch.float32)
torch.autograd.backward(y.float(), dy)
found = {"y": y, **{f"d_{name}": held[name].grad for name in kinds}}
np.savez(sys.argv[2], **{name: value.detach().float().cpu().numpy() for name, value in found.items()})
print(f"K mamba_ssm selective {draws} draws: kernels {ran - begun:.2f} s, "
      f"back {time.perf_counter() - ran:.2f} s", flush=True)
'''
"""mamba_ssm's own `selective_scan_fn` (its CUDA kernel, `exp2f` under `--use_fast_math`) in the interpreter
`DEW_MAMBA_PYTHON` names, over a block of draws run as one call (each draw its own channels and its own group
of B and C), a row's documents each a call of its own from a zero state (the kernel takes no documents) at
the request's `bounds`, its parameters' gradients the fp32 sum of the documents' pulls."""


def mamba_selective_side(drawn: list, starts) -> list:
    """mamba_ssm's selective scan (`MAMBA_SELECTIVE_SIDE`) on a block of draws, each `(x, dt, a, b, c, skip,
    dy)` as the selective template's sites draw them (`[B, L, D]` and `[B, L, N]` in their dtype, the decays
    `[D, N]` and skip `[D]` fp32): each draw's output and its six operands' gradients, `(y, dx, ddt, da, db,
    dc, dskip)` in their operands' dtypes. `starts` `[B, L]` the rows' documents, alike in every row (None:
    one sequence a row)."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import pytest

    python = os.environ.get("DEW_MAMBA_PYTHON")
    if not python:
        pytest.fail("DEW_MAMBA_PYTHON names no interpreter with mamba_ssm 2.3.2 and torch: the GPU tier's "
                    "reference, which a run selecting the mamba reference gives it")
    names = ("x", "dt", "a", "b", "c", "skip")
    length = drawn[0][0].shape[1]
    marks = np.zeros(length, bool) if starts is None else np.asarray(starts, bool)[0]
    if starts is not None:
        assert np.all(np.asarray(starts, bool) == marks), "a block's rows share their documents"
    given = {name: np.stack([np.asarray(d[i], np.float32) for d in drawn]) for i, name in enumerate(names)}
    # B and C in the values' dtype where theirs is narrower (a bf16 C beside fp32 x): the same values, as the
    # upstream's kernel takes B and C in its inputs' type.
    widest = jnp.dtype(drawn[0][0].dtype).name
    kinds = [jnp.dtype(drawn[0][i].dtype).name for i in range(6)]
    kinds[3:5] = [widest, widest]
    given |= {"leaves": np.asarray(",".join(names)), "dtypes": np.asarray(",".join(kinds)),
              "bounds": np.asarray(sorted({0, *np.flatnonzero(marks).tolist(), length}), np.int64),
              "dy": np.stack([np.asarray(d[6], np.float32) for d in drawn])}
    with tempfile.TemporaryDirectory() as root:
        np.savez(Path(root) / "in.npz", **given)
        subprocess.run([python, "-c", MAMBA_SELECTIVE_SIDE, Path(root) / "in.npz", Path(root) / "out.npz"],
                       check=True)
        with np.load(Path(root) / "out.npz") as found:
            outputs = dict(found)
    return [(jnp.asarray(outputs["y"][i], drawn[i][0].dtype),
             *(jnp.asarray(outputs[f"d_{name}"][i], drawn[i][k].dtype) for k, name in enumerate(names)))
            for i in range(len(drawn))]


def recurred(decay, write, entering, axis: int):
    """`ssm.affine_states`' recurrence a token at a time, `h_t = a_t h_(t-1) + b_t` along `axis` from
    `entering`, one token a step of a `lax.scan` (each step the loop's multiply and add, the entering state
    in the steps' type as the first step's product promotes it): the affine scan's token oracle, compiled
    once a step where a Python loop of 192 tokens unrolled every order of it, 198 s of a cold run's cell
    (row 9511)."""
    dtype = jnp.result_type(decay, write, entering)

    def step(state, token):
        a, b = token
        state = a * state + b
        return state, state

    _, states = jax.lax.scan(step, jnp.asarray(entering, dtype),
                             tuple(jnp.moveaxis(t, axis, 0) for t in (decay, write)))
    return jnp.moveaxis(states, 0, axis % decay.ndim)


FAINT_BESIDE_THE_LARGEST = ("a faint write", "a faint cotangent")
"""`faint_beside_the_largest`'s cases: the first's states alone (unit cotangents times its states near the
largest put its decay's gradient past it), the second's with every gradient."""


def faint_beside_the_largest(case: str, seed: int) -> tuple:
    """`seed`'s fp32 `(decay, write, entering)` of an affine scan `[1, 16, 64]` along its tokens and the
    cotangent of its states: faint values beside sums past float32's largest in each recurrence, a reset
    (decay zero) between them. A faint write: the state entering at -1.9 2^127, decays of one, tokens 0 to 5
    writing 2^118 [-1, 1] and tokens 6 and 7 2^127 [1, 1.9] each (their sum from a zero state past the
    largest, row 8465), a reset at token 8 writing 2^-125 [1, 2) and none after, under decays of one half,
    its state halved at token 9 a normal float. A faint cotangent, mirrored: writes of 2^-20 [-1, 1], none
    entering, cotangents of 2^116 [-1, 1] at tokens 9 to 12, of 2^127 [1, 1.9] at tokens 13 and 14 and token
    15's -1.9 2^127, a reset at token 8 and a cotangent of 2^-125 [1, 2) at token 7 alone, halved into token
    6's by its decay of one half (row 8486: a recurrence halved where it reached the largest flushed its
    faint half)."""
    rng = np.random.default_rng([8486, seed])
    shape = (1, 16, 64)
    largest = 2.0 ** 127 * rng.uniform(1.0, 1.9, (1, 2, 64))
    faint = 2.0 ** -125 * rng.uniform(1.0, 2.0, (1, 64))
    decay = np.ones(shape)
    decay[:, 8] = 0.0
    if case == "a faint write":
        decay[:, 9:] = 0.5
        write, cotangent = np.zeros(shape), np.ones(shape)
        write[:, :6] = 2.0 ** 118 * rng.uniform(-1.0, 1.0, (1, 6, 64))
        write[:, 6:8], write[:, 8] = largest, faint
        entering = np.full((1, 64), -1.9 * 2.0 ** 127)
    else:
        decay[:, 7] = 0.5
        write, cotangent = 2.0 ** -20 * rng.uniform(-1.0, 1.0, shape), np.zeros(shape)
        cotangent[:, 9:13] = 2.0 ** 116 * rng.uniform(-1.0, 1.0, (1, 4, 64))
        cotangent[:, 13:15], cotangent[:, 15], cotangent[:, 7] = largest, -1.9 * 2.0 ** 127, faint
        entering = np.zeros((1, 64))
    return tuple(jnp.asarray(t, jnp.float32) for t in (decay, write, entering, cotangent))


def second_beside_the_largest(seed: int) -> tuple:
    """`seed`'s fp32 `(decay, write, entering)` of an affine scan `[1, 3, 64]`, its states' cotangent and a
    direction of its writes' cotangent: no writes or state entering, a decay in [1/4, 3/4] at token 0 and
    resets at tokens 1 and 2, cotangents of 2^-2 [-1, 1] at token 1 and 2^127 [1, 1.9] at token 2, the
    direction 2^127 [1, 1.9] at token 0 alone: the decays' gradient of the writes' cotangent along it is a
    float32, the second token's 2^125 [-1, 1] (the head review of 29231f794: autodiff of a halving's scaling
    doubled the cotangent of 2^127 past the largest, NaN)."""
    rng = np.random.default_rng([8486, 2, seed])
    shape = (1, 3, 64)
    decay = rng.uniform(0.25, 0.75, shape)
    decay[:, 1:] = 0.0
    cotangent, direction = np.zeros(shape), np.zeros(shape)
    cotangent[:, 1] = 2.0 ** -2 * rng.uniform(-1.0, 1.0, (1, 64))
    cotangent[:, 2] = direction[:, 0] = 2.0 ** 127 * rng.uniform(1.0, 1.9, (1, 64))
    return tuple(jnp.asarray(t, jnp.float32) for t in (decay, np.zeros(shape), np.zeros((1, 64)), cotangent,
                                                        direction))
