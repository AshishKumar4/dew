"""The S5 layer against a float64 NumPy transcription of its recurrence.

`S5Layer` discretizes diagonal complex poles by zero-order hold and runs the
recurrence in chunks of pole-power products (`diagonal_recurrence`), in real
arithmetic over stacked real and imaginary parts, or on a TPU as a complex
`associative_scan`; both forms are held to the same bounds. The oracle here is written from
the equations (Smith, Warrington and Linderman, "Simplified State Space
Layers for Sequence Modeling", 2023, eqs. 2-6), step by step, in complex128:

    A = -exp(log_A_real) + i A_imag,  dt = exp(log_dt)
    A_bar = exp(A dt),  B_bar = (A_bar - 1) / A * B
    x_k = A_bar x_{k-1} + B_bar u_k,  x_0 = 0
    y_k = Re(C x_k) + D u_k

with its gradients from the hand-written adjoint of the same recurrence,
itself checked against central differences in float64.

Tolerances are running-error bounds, not fits. Every fp32 operation rounds
with relative error at most u = 2^-24, so a quantity reached through a chain
of at most K roundings carries at most K u times the sum of the absolute
values of its terms (Higham, "Accuracy and Stability of Numerical
Algorithms", 2nd ed., section 3.1). The oracle evaluates that sum of
absolute terms by running the same recurrence and the same adjoint on the
magnitudes of every factor. K counts the longest chain:

- forward: the pole A dt and its exponential, complex, 8 roundings; A_bar's
  error compounds once per step through the running products that form its
  powers, at most S of them, and each state is a sum of at most 2 S real
  terms; 9 per step over S steps bounds both; the discretized input
  (A_bar - 1) / A * B and its F-term product with u, F + 8; the output
  product, 2N real terms, and the skip, 2N + 4. K = 9 S + F + 2N + 20.
- gradients: the forward chain, the adjoint recurrence run back over the
  same S steps with the same 9 roundings a step, the chain rule back
  through the discretization, 16, and the sum over the B S positions a
  parameter gradient collects. K = 2 (9 S) + B S + F + 2N + 36.

At B 2, F 4, N 8 and S 16 that is K = 184 forward (1.1e-5 relative to the
absolute terms) and K = 376 on the gradients (2.2e-5); at S 40, two whole
chunks and a padded third, K = 400 and 856.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn import ssm
from dew.nn.ssm import BidirectionalS5Layer, S5Layer, diagonal_recurrence

U = 2.0 ** -24
BATCH, FEATURES, STATE = 2, 4, 8


def forward_chain(steps: int, features: int = FEATURES, state: int = STATE) -> int:
    return 9 * steps + features + 2 * state + 20


def gradient_chain(steps: int, batch: int = BATCH, features: int = FEATURES, state: int = STATE) -> int:
    return 2 * 9 * steps + batch * steps + features + 2 * state + 36
LEAVES = ("log_A_real", "A_imag", "B_re", "B_im", "C_re", "C_im", "D", "log_dt")


class Arithmetic:
    """The oracle's operations, exact (complex128) or on magnitudes. On
    magnitudes every product multiplies absolute values, every difference
    adds them and the real part of a complex value is its modulus, which
    turns the same code into the sum of absolute terms of each result."""

    def __init__(self, absolute: bool):
        self.absolute = absolute

    def value(self, z):
        return np.abs(z) if self.absolute else z

    def sub(self, a, b):
        return np.abs(a) + np.abs(b) if self.absolute else a - b

    def real(self, z):
        return np.abs(z) if self.absolute else z.real

    def imag(self, z):
        return np.abs(z) if self.absolute else z.imag

    def conj(self, z):
        return z if self.absolute else np.conj(z)


def float64_params(params):
    return {name: np.asarray(params[name], np.float64) for name in LEAVES}


def discretized(p, op: Arithmetic):
    """The poles, their step and the zero-order hold of the input map."""
    dt = np.exp(p["log_dt"])
    a = op.value(-np.exp(p["log_A_real"]) + 1j * p["A_imag"])
    a_bar = op.value(np.exp((-np.exp(p["log_A_real"]) + 1j * p["A_imag"]) * dt))
    hold = op.sub(a_bar, 1.0) / a
    b = op.value(p["B_re"] + 1j * p["B_im"])
    return a, dt, a_bar, hold, b


def forward(p, u, op: Arithmetic):
    """`y [B, S, F]` and the states `x [B, S, N]` the recurrence visits."""
    _, _, a_bar, hold, b = discretized(p, op)
    b_bar = hold[:, None] * b
    c = op.value(p["C_re"] + 1j * p["C_im"])
    u = op.value(u)
    x = np.zeros((u.shape[0], u.shape[1], a_bar.shape[0]), np.complex128)
    state = np.zeros((u.shape[0], a_bar.shape[0]), np.complex128)
    for k in range(u.shape[1]):
        state = a_bar * state + u[:, k] @ b_bar.T
        x[:, k] = state
    y = op.real(np.einsum("fn,bsn->bsf", c, x)) + op.value(p["D"]) * u
    return y, x


def adjoint(p, u, w, op: Arithmetic):
    """The gradient of `sum(w * y)` with respect to every parameter and to
    `u`. A real function of complex z has gradient g_z = dL/dRe z + i
    dL/dIm z; through z = a b it passes g_a = g_z conj(b), and a real
    parameter t of z = f(t) receives Re(g_z conj(f'(t)))."""
    a, dt, a_bar, hold, b = discretized(p, op)
    c = op.value(p["C_re"] + 1j * p["C_im"])
    u, w = op.value(u), op.value(w)
    _, x = forward(p, u, op)
    batch, steps, _ = u.shape
    g_x = np.zeros_like(x)
    carried = np.zeros((batch, a_bar.shape[0]), np.complex128)
    for k in reversed(range(steps)):
        carried = w[:, k] @ op.conj(c) + op.conj(a_bar) * carried
        g_x[:, k] = carried
    previous = np.concatenate([np.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    g_a_bar = np.einsum("bsn,bsn->n", g_x, op.conj(previous))
    g_b_bar = np.einsum("bsn,bsf->nf", g_x, u)
    g_u = op.real(np.einsum("bsn,nf->bsf", g_x, op.conj(hold[:, None] * b))) + op.value(p["D"]) * w
    g_c = np.einsum("bsf,bsn->fn", w, op.conj(x))
    g_b = g_b_bar * op.conj(hold)[:, None]
    g_hold = np.einsum("nf,nf->n", g_b_bar, op.conj(b))
    g_a_bar = g_a_bar + g_hold * op.conj(1.0 / a)
    g_a = g_hold * op.conj(op.sub(a_bar, 1.0) / (a * a)) * (1 if op.absolute else -1)
    g_a = g_a + g_a_bar * op.conj(dt * a_bar)
    g_dt = op.real(g_a_bar * op.conj(a * a_bar))
    magnitude = np.exp(p["log_A_real"])
    return {
        "log_A_real": magnitude * op.real(g_a) * (1 if op.absolute else -1),
        "A_imag": op.imag(g_a),
        "B_re": op.real(g_b), "B_im": op.imag(g_b),
        "C_re": op.real(g_c), "C_im": op.imag(g_c),
        "D": np.einsum("bsf,bsf->f", w, u),
        "log_dt": g_dt * dt,
        "u": g_u,
    }


EXACT, MAGNITUDES = Arithmetic(absolute=False), Arithmetic(absolute=True)


# 16 positions are one chunk of the layer's recurrence; 40 are two whole
# chunks and a padded third, so the carry between chunks is exercised.
@pytest.fixture(scope="module", params=[16, 40])
def case(request):
    steps = request.param
    layer = S5Layer(features=FEATURES, state_dim=STATE)
    keys = jax.random.split(jax.random.key(0), 3)
    u = jax.random.normal(keys[0], (BATCH, steps, FEATURES), jnp.float32)
    params = layer.init(keys[1], u)["params"]
    # Steps up to 1 instead of the init's 0.1, so every pole turns and
    # decays visibly inside 16 steps and the hold differs from an Euler step.
    params = {**params, "log_dt": jnp.log(jnp.linspace(0.05, 1.0, STATE))}
    w = jax.random.normal(keys[2], (BATCH, steps, FEATURES), jnp.float32)
    return layer, params, u, w


def test_the_oracle_adjoint_is_the_derivative_of_the_oracle(case):
    """Central differences in float64 on every parameter and input entry:
    step 1e-6 leaves truncation near 1e-12 and cancellation near 1e-10
    relative, so 1e-7 of the largest gradient separates a right adjoint from
    a wrong one by orders of magnitude."""
    _, params, u, w = case
    p, u, w = float64_params(params), np.asarray(u, np.float64), np.asarray(w, np.float64)
    gradients = adjoint(p, u, w, EXACT)

    def loss(p, u):
        return float(np.sum(w * forward(p, u, EXACT)[0]))

    step = 1e-6
    for name in (*LEAVES, "u"):
        values = u if name == "u" else p[name]
        numeric = np.zeros_like(values)
        for index in np.ndindex(values.shape):
            moved = []
            for sign in (1, -1):
                shifted = values.copy()
                shifted[index] += sign * step
                moved.append(loss(p, shifted) if name == "u" else loss({**p, name: shifted}, u))
            numeric[index] = (moved[0] - moved[1]) / (2 * step)
        scale = np.max(np.abs(numeric))
        assert np.max(np.abs(gradients[name] - numeric)) <= 1e-7 * scale, name


@pytest.fixture(params=["chunked", "scanned"])
def form(request, monkeypatch):
    """The layer's directions run as a GPU runs them (`_chunked_directions`)
    or as a TPU does (`_scanned_directions`), both here on the CPU."""
    chosen = {"chunked": ssm._chunked_directions, "scanned": ssm._scanned_directions}[request.param]
    monkeypatch.setattr(ssm, "_directions", chosen)
    return request.param


def test_the_layer_computes_the_zero_order_hold_recurrence(case, form):
    """Every output within K u of its absolute terms, K = 184 at 16 positions."""
    layer, params, u, _ = case
    expected, _ = forward(float64_params(params), np.asarray(u, np.float64), EXACT)
    bound = forward_chain(u.shape[1]) * U * forward(
        float64_params(params), np.asarray(u, np.float64), MAGNITUDES)[0]

    actual = np.asarray(layer.apply({"params": params}, u), np.float64)

    assert np.all(np.abs(actual - expected) <= bound)


def test_the_layer_gradients_are_the_recurrence_gradients(case, form):
    """Every parameter's and the input's gradient within K u of its absolute
    terms, K = 376 at 16 positions."""
    layer, params, u, w = case
    p, u64, w64 = float64_params(params), np.asarray(u, np.float64), np.asarray(w, np.float64)
    expected = adjoint(p, u64, w64, EXACT)
    terms = adjoint(p, u64, w64, MAGNITUDES)

    def loss(variables, u):
        return jnp.sum(w * layer.apply({"params": variables}, u))

    parameters, inputs = jax.grad(loss, argnums=(0, 1))(params, u)
    actual = {**{name: parameters[name] for name in LEAVES}, "u": inputs}

    for name, value in actual.items():
        error = np.abs(np.asarray(value, np.float64) - expected[name])
        assert np.all(error <= gradient_chain(u.shape[1]) * U * terms[name]), name


def test_a_long_sequence_stays_within_the_bound():
    """4096 positions are 256 chunks, whose carry runs as a scan of its
    own. Steps from 0.001 keep the slowest pole's memory across the whole
    sequence, and outputs and gradients stay within K u of their absolute
    terms."""
    batch, steps, features, state = 1, 4096, 2, 4
    layer = S5Layer(features=features, state_dim=state)
    keys = jax.random.split(jax.random.key(1), 3)
    u = jax.random.normal(keys[0], (batch, steps, features), jnp.float32)
    params = layer.init(keys[1], u)["params"]
    params = {**params, "log_dt": jnp.log(jnp.linspace(0.001, 0.05, state))}
    w = jax.random.normal(keys[2], u.shape, jnp.float32)
    p, u64, w64 = float64_params(params), np.asarray(u, np.float64), np.asarray(w, np.float64)

    expected, _ = forward(p, u64, EXACT)
    bound = forward_chain(steps, features, state) * U * forward(p, u64, MAGNITUDES)[0]
    assert np.all(np.abs(np.asarray(layer.apply({"params": params}, u), np.float64) - expected) <= bound)

    # Under jit: run eagerly, the scan's gradient dispatches thousands of small ops.
    parameters, inputs = jax.jit(jax.grad(lambda params, u: jnp.sum(w * layer.apply({"params": params}, u)),
                                          argnums=(0, 1)))(params, u)
    exact, terms = adjoint(p, u64, w64, EXACT), adjoint(p, u64, w64, MAGNITUDES)
    for name, value in {**{name: parameters[name] for name in LEAVES}, "u": inputs}.items():
        error = np.abs(np.asarray(value, np.float64) - exact[name])
        assert np.all(error <= gradient_chain(steps, batch, features, state) * U * terms[name]), name


def test_the_recurrence_holds_for_poles_all_around_the_circle():
    """The chunks build the pole powers by doubling, so a pole past the
    imaginary axis (negative real part) or at the branch cut has to give the
    powers a running product gives. Poles at every angle of
    the circle, the cut included, at radii up to 0.999 and over 100 positions
    (seven chunks): the states within K u of their absolute terms, against a
    complex128 recurrence. S4D-Lin's imaginary parts, up to pi (N - 1), put a
    trained layer's poles anywhere on the circle once dt reaches 1 / N."""
    states, steps = 33, 100
    angles = np.linspace(-np.pi, np.pi, states)
    radii = np.linspace(0.9, 0.999, states)
    pole = (radii * np.exp(1j * angles)).astype(np.complex64)
    rng = np.random.default_rng(3)
    v = (rng.normal(size=(2, steps, states)) + 1j * rng.normal(size=(2, steps, states))).astype(np.complex64)

    def run(pole, v):
        x, out = np.zeros(v.shape[::2], v.dtype), []
        for k in range(steps):
            x = pole * x + v[:, k]
            out.append(x)
        return np.stack(out, axis=1)

    expected = run(pole.astype(np.complex128), v.astype(np.complex128))
    magnitude = run(np.abs(pole).astype(np.float64), np.abs(v).astype(np.float64))
    inputs = np.concatenate([v.real, v.imag], axis=-1)
    actual = np.asarray(jax.jit(diagonal_recurrence)(jnp.asarray(pole), jnp.asarray(inputs)), np.float64)
    actual = actual[..., :states] + 1j * actual[..., states:]
    assert np.all(np.abs(actual - expected) <= forward_chain(steps, 1, states) * U * magnitude)


@pytest.mark.parametrize("platform, scanned", [("tpu", True), ("cuda", False), ("cpu", False)])
def test_a_tpu_runs_the_layer_as_it_first_did(platform, scanned):
    """The TPU lowering takes `_scanned_directions`, the layer's first form,
    and the others the chunked pole-power products, whose Toeplitz product
    names its einsum."""
    layer = BidirectionalS5Layer(features=4, state_dim=4)
    u = jnp.zeros((1, 40, 4), jnp.float32)
    variables = layer.init(jax.random.key(0), u)
    text = jax.jit(layer.apply).trace(variables, u).lower(lowering_platforms=(platform,)).as_text(
        debug_info=True)
    assert ("kjn,bcjn->bckn" not in text) == scanned


def test_the_forward_bound_rejects_an_euler_step(case):
    """The mutation the bound has to catch: B_bar = dt B, the first-order
    discretization in place of the hold, lands outside K u of the hold's
    outputs, so a layer that made it would fail the test above."""
    _, params, u, _ = case
    p, u64 = float64_params(params), np.asarray(u, np.float64)
    _, dt, _, hold, b = discretized(p, EXACT)
    euler_b = b * (dt / hold)[:, None]
    euler = forward({**p, "B_re": euler_b.real, "B_im": euler_b.imag}, u64, EXACT)[0]
    expected = forward(p, u64, EXACT)[0]
    bound = forward_chain(u.shape[1]) * U * forward(p, u64, MAGNITUDES)[0]
    assert np.any(np.abs(euler - expected) > bound)


def test_the_poles_start_at_s4d_lin():
    """S4D-Lin (Gu, Gupta, Goel and Re, "On the Parameterization and
    Initialization of Diagonal State Space Models", 2022, section 4): the
    poles -1/2 + i pi n, n = 0 .. N - 1, a real part shared by every state."""
    layer = S5Layer(features=FEATURES, state_dim=STATE)
    params = layer.init(jax.random.key(0), jnp.zeros((1, 2, FEATURES)))["params"]
    poles = (-np.exp(np.asarray(params["log_A_real"], np.float64))
             + 1j * np.asarray(params["A_imag"], np.float64))
    expected = -0.5 + 1j * np.pi * np.arange(STATE)
    np.testing.assert_allclose(poles, expected, rtol=4 * U, atol=0)
