"""The Pallas recurrence of the chunked delta rule against the XLA scan it replaces.

`dew.nn.kernels.delta_chunks` computes what `dew.nn.linear.xla_chunk_states`
computes. On a host without a GPU an explicit 'pallas' runs the kernels
through `pallas_call(interpret=True)`, so the CPU suite tests their
formulation: the block specs and column blocks, the recurrence in both
directions and the hand-written backward. What it leaves untested is
Triton's own arithmetic and anything about speed; a GPU host runs the same
cases on the compiled kernels and the device-sized one at Qwen3.5-9B's
widths.

Tolerances and the differences observed, fp32 on CPU:

- the recurrence, both shapes : entered states, corrected values and final
  state at most @@ apart, relative to their largest entry; tolerance 1e-5
- its gradients               : at most @@ relative; tolerance 1e-5
- through the rule            : output and final state @@, the six
  gradients @@, relative; tolerance 1e-5
- against float64             : the kernels' RMS distance at most @@ times
  the scan's over every output and gradient; bound 2
  (tests/reference_error.py)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn import kernels, linear
from dew.nn.kernels import delta_chunks
from dew.nn.linear import chunk_gated_delta_rule, chunk_states_kernel, l2norm, xla_chunk_states
from dew.training import MeshSpec

BOUND = 1e-5
INTERPRETED = jax.default_backend() != "gpu"


def relative(got, want) -> float:
    want = np.asarray(want, np.float64)
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1.0))


def recurrence_operands(shape, seed: int = 0):
    """The recurrence's operands at `(batch, heads, chunks, chunk, key width,
    value width)`: unit keys, per-token log decays around -0.05 cumulated
    within each chunk, `w` the keys scaled by beta and their decay as the
    rule's `k_cumdecay` is before its in-chunk correction, and a held state."""
    batch, heads, chunks, chunk, width, columns = shape
    keys = jax.random.split(jax.random.key(seed), 5)
    lead = (batch, heads, chunks, chunk)
    key = l2norm(jax.random.normal(keys[0], (*lead, width)))
    gc = jnp.cumsum(-jnp.exp(jax.random.normal(keys[1], lead) - 3.0), axis=-1)
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], lead))
    w = key * (beta * jnp.exp(gc))[..., None]
    u = jax.random.normal(keys[3], (*lead, columns)) * beta[..., None]
    state = 0.1 * jax.random.normal(keys[4], (batch, heads, width, columns))
    return key, w, u, gc, state


def cotangents(outputs, seed: int = 1):
    """One random cotangent per output, so no term drops out of a gradient."""
    keys = jax.random.split(jax.random.key(seed), len(outputs))
    return tuple(jax.random.normal(k, out.shape) for k, out in zip(keys, outputs, strict=True))


def kernel_states(*operands):
    return delta_chunks.chunk_states(*operands, INTERPRETED)


# Three chunks of the narrowest product Triton takes, two blocks of value
# columns; and one row-head at Qwen3.5's widths, four blocks.
SHAPES = [pytest.param((2, 2, 3, 16, 32, 64), id="narrow"),
          pytest.param((1, 1, 3, 64, 128, 128), id="qwen-widths")]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_kernels_compute_the_xla_recurrence(shape):
    """The state every chunk entered, the corrected values and the state
    leaving the last chunk."""
    operands = recurrence_operands(shape)
    expected = xla_chunk_states(*operands)

    got = kernel_states(*operands)

    for name, want, have in zip(("entered", "corrected", "final"), expected, got, strict=True):
        assert have.shape == want.shape, name
        assert relative(have, want) < BOUND, name


@pytest.mark.parametrize("shape", SHAPES)
def test_the_kernels_compute_the_xla_gradients(shape):
    """The reverse recurrence and the products after it against autodiff of
    the scan, for the keys, `w`, the values, the decays and the held state."""
    operands = recurrence_operands(shape)
    seeded = cotangents(xla_chunk_states(*operands))
    expected = jax.vjp(xla_chunk_states, *operands)[1](seeded)

    gradients = jax.vjp(kernel_states, *operands)[1](seeded)

    for name, want, got in zip(("key", "w", "u", "gc", "state"), expected, gradients, strict=True):
        assert relative(got, want) < BOUND, name


def rule_operands(seed: int = 0):
    """`[B, S, H, D]` operands as GatedDeltaNet hands them to the rule: 150
    tokens, two whole chunks and a ragged one, two heads of 32 keys and 64
    values, and a held state."""
    keys = jax.random.split(jax.random.key(seed), 6)
    B, S, H = 2, 150, 2
    query, key = (l2norm(jax.random.normal(drawn, (B, S, H, 32))) for drawn in keys[:2])
    return (query, key, jax.random.normal(keys[2], (B, S, H, 64)),
            -jnp.exp(jax.random.normal(keys[3], (B, S, H)) - 2.0),
            jax.nn.sigmoid(jax.random.normal(keys[4], (B, S, H))),
            0.1 * jax.random.normal(keys[5], (B, H, 32, 64)))


def test_the_rule_computes_the_same_values_and_gradients_on_either_recurrence():
    """Through the public entry, so the prep, the products after the
    recurrence and the custom VJP's seam with autodiff are on both sides:
    the output, the final state and the gradients of all six operands."""
    operands = rule_operands()
    outputs = chunk_gated_delta_rule(*operands, implementation='xla')
    seeded = cotangents(outputs)

    def loss(implementation):
        def scored(*args):
            out, final = chunk_gated_delta_rule(*args, implementation=implementation)
            return jnp.sum(out * seeded[0]) + jnp.sum(final * seeded[1])
        return jax.grad(scored, argnums=tuple(range(6)))

    got = chunk_gated_delta_rule(*operands, implementation='pallas')
    for name, want, have in zip(("output", "final"), outputs, got, strict=True):
        assert relative(have, want) < BOUND, name
    for name, want, have in zip(("query", "key", "value", "g", "beta", "state"),
                                loss('xla')(*operands), loss('pallas')(*operands), strict=True):
        assert relative(have, want) < BOUND, name


def test_the_kernels_are_as_exact_as_the_xla_scan():
    """What separates the two at fp32 is rounding, so the kernels sit within
    fp32 rounding of the same recurrence in float64 as the scan does: never
    more than twice as far from it, on any output or gradient."""
    operands = recurrence_operands((2, 2, 3, 64, 32, 64))
    seeded = cotangents(xla_chunk_states(*operands))
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in operands]
        truth = xla_chunk_states(*exact)
        truth_gradients = jax.vjp(xla_chunk_states, *exact)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seeded))

    scanned = xla_chunk_states(*operands)
    states = kernel_states(*operands)
    scan_gradients = jax.vjp(xla_chunk_states, *operands)[1](seeded)
    kernel_gradients = jax.vjp(kernel_states, *operands)[1](seeded)

    names = ("entered", "corrected", "final", "key", "w", "u", "gc", "state")
    for name, mine, theirs, want in zip(names, (*states, *kernel_gradients), (*scanned, *scan_gradients),
                                        (*truth, *truth_gradients), strict=True):
        assert_as_exact_as_the_reference(mine, theirs, want, name)


@pytest.mark.parametrize("generation, gpu, chosen", [
    ("sm80", True, "pallas"), ("sm80", False, "xla"), ("sm89", True, "xla"), ("cpu", False, "xla")])
def test_auto_takes_the_kernels_where_they_were_measured(monkeypatch, generation, gpu, chosen):
    """'auto' runs the kernels on a generation `KERNELS` names them for and
    they compile on, and the scan everywhere else."""
    monkeypatch.setitem(kernels.KERNELS, "gated_delta_rule", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: generation)
    monkeypatch.setattr(linear, "triton_runs", lambda: gpu)
    key, _, u, _, _ = recurrence_operands((1, 1, 2, 64, 128, 128))
    assert chunk_states_kernel("auto", key, u) == chosen


REFUSED = {
    "a float64 rule": ((1, 1, 2, 64, 128, 128), jnp.float64, "the rule runs in float64"),
    "keys 96 wide": ((1, 1, 2, 64, 96, 128), jnp.float32, "chunks of 64 and keys 96 wide"),
    "a chunk of 8": ((1, 1, 2, 8, 128, 128), jnp.float32, "chunks of 8 and keys 128 wide"),
    "values 48 wide": ((1, 1, 2, 64, 128, 48), jnp.float32, "values 48 wide"),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_call_the_kernels_cannot_take_runs_the_scan_and_says_why(monkeypatch, caplog, case):
    """Where the kernels were measured fastest, a call they cannot take runs
    the scan, and the first such call logs the reason by name."""
    shape, dtype, reason = REFUSED[case]
    monkeypatch.setitem(kernels.KERNELS, "gated_delta_rule", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm80")
    monkeypatch.setattr(kernels.generation, "_logged", set())
    monkeypatch.setattr(linear, "triton_runs", lambda: True)
    with jax.enable_x64(new_val=dtype == jnp.float64):
        key, _, u, _, _ = (jnp.asarray(t, dtype) for t in recurrence_operands(shape))
        assert [chunk_states_kernel("auto", key, u) for _ in range(2)] == ["xla"] * 2
    logged = [record.getMessage() for record in caplog.records if record.name == kernels.generation.__name__]
    assert len(logged) == 1 and reason in logged[0], logged


def test_forward_mode_runs_the_scan():
    """The kernels define only a reverse-mode derivative. Under
    `forward_mode_attention`, which a consistency model's `jax.jvp` runs in,
    the rule takes the scan and its JVP is the scan's."""
    from dew.nn.attention import forward_mode_attention

    operands = rule_operands()
    tangents = tuple(jnp.ones_like(t) for t in operands)
    with forward_mode_attention():
        assert chunk_states_kernel("pallas", operands[1], operands[2]) == "xla"
        _, got = jax.jvp(lambda *a: chunk_gated_delta_rule(*a, implementation='pallas'), operands, tangents)
    _, want = jax.jvp(lambda *a: chunk_gated_delta_rule(*a, implementation='xla'), operands, tangents)
    for have, wanted in zip(got, want, strict=True):
        np.testing.assert_array_equal(have, wanted)


@pytest.mark.mesh(devices=2)
def test_a_split_batch_runs_the_scan():
    """The kernels see whole arrays; under a mesh that splits the rows they
    would run every row on every device, so the call takes the scan."""
    key, _, u, _, _ = recurrence_operands((2, 1, 2, 64, 128, 128))
    with jax.set_mesh(MeshSpec(fsdp=2).build(jax.devices()[:2])):
        assert chunk_states_kernel("pallas", key, u) == "xla"
    assert chunk_states_kernel("pallas", key, u) == "pallas"


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="the compiled kernels run on CUDA")
def test_the_compiled_kernels_are_as_exact_as_xla_at_qwen35_widths():
    """One row of Qwen3.5-9B's gated delta net at the default matmul
    precision, which both paths take (TF32 on the GPU): 1280 tokens, 32
    heads of 128. Against the rule in float64 on the host, the kernels' output
    and gradients are never more than twice as far as XLA's scan, and two
    calls agree bit for bit, since fla pins its forward kernel to two warps
    on Blackwell for a Triton race (fla-org/flash-linear-attention#945)."""
    keys = jax.random.split(jax.random.key(3), 6)
    B, S, H, D = 1, 1280, 32, 128
    query, key = (l2norm(jax.random.normal(drawn, (B, S, H, D))) for drawn in keys[:2])
    operands = (query, key, jax.random.normal(keys[2], (B, S, H, D)),
                -jnp.exp(jax.random.normal(keys[3], (B, S, H)) - 3.0),
                jax.nn.sigmoid(jax.random.normal(keys[4], (B, S, H))))
    seeded = jax.random.normal(keys[5], (B, S, H, D))

    def gradients(implementation):
        return jax.jit(jax.value_and_grad(
            lambda *a: jnp.sum(chunk_gated_delta_rule(*a, implementation=implementation)[0] * seeded),
            argnums=tuple(range(5))))

    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        exact = [jax.device_put(np.asarray(t, np.float64), cpu) for t in (*operands, seeded)]
        truth = jax.grad(lambda *a: jnp.sum(chunk_gated_delta_rule(*a, implementation='xla')[0] * exact[5]),
                         argnums=tuple(range(5)))(*exact[:5])
        truth_out = chunk_gated_delta_rule(*exact[:5], implementation='xla')[0]

    xla_out = jax.jit(lambda *a: chunk_gated_delta_rule(*a, implementation='xla')[0])(*operands)
    kernel_out = jax.jit(lambda *a: chunk_gated_delta_rule(*a, implementation='pallas')[0])(*operands)
    _, xla_grads = gradients('xla')(*operands)
    _, first = gradients('pallas')(*operands)
    _, second = gradients('pallas')(*operands)

    for name, mine, theirs, want in zip(("output", "query", "key", "value", "g", "beta"),
                                        (kernel_out, *first), (xla_out, *xla_grads), (truth_out, *truth),
                                        strict=True):
        assert_as_exact_as_the_reference(mine, theirs, want, name)
    for name, one, two in zip(("query", "key", "value", "g", "beta"), first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(two), err_msg=name)
