"""The Pallas SSD kernel against the XLA scan in the reference's order.

`dew.nn.kernels.ssd` computes what `xla_chunk_scan`, transformers' `mamba2_chunk_scan` in its order,
computes. The kernel is written for TPU (a GPU runs mamba_ssm's own kernels,
`dew.nn.kernels.mamba_ssd`); a host without one runs it through
`pallas_call(interpret=True)`, which checks shapes, dtypes, resets and state gradients. Check 1 holds
every geometry's numbers on a TPU, compiled, and the host geometries' on the CPU lane too, interpreted.

Interpreted, the kernel and XLA's scan are two Dew paths on the CPU, each compiled without
XLA:CPU's YNNPACK fusions (`without_ynn`), whose microkernels follow the host's instruction set, so
that the host geometries' verdicts do not.

One driver (`assert_kernel_parity`) holds every case to the same rule: the
kernel's forward, final state and all five gradients finite and, by
tests/reference_error.py's pooled gate, on par with XLA's distance from the stepwise recurrence in
float64 (`stepwise_scan`, which shares no line with either chunked path), the XLA path's arithmetic in float64
first held to the recurrence. The cases (`CASES`): the reference fixture's scan
at a chunk of 32 (three chunks, a ragged tail) and of 128 (one chunk, mostly
padding), with and without document resets (`RESET_DECAY` in `A dt`, which the
kernel's segment sums multiply by the 0/1 triangle where a -inf would make
NaN); tails one short of and one over a chunk, whole chunks, ragged ones and
the default chunk at a production width; and, compiled on a TPU, the geometry
`tools/benchmark_ssd.py --parity` times.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, assert_computes_the_oracle, with_gradients, without_ynn
from ssd_support import ssd_orbit, stepwise_scan

from dew.algebra.scan import RESET_DECAY
from dew.nn.kernels.recurrence_chunk import ssd_operands
from dew.nn.kernels.ssd import _chunk_scan

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


def blocks(x, dt, A, B, C, state, chunk_size: int, dtype=jnp.float32):
    """The operands as the layer hands them to the chunk scan (`ssd_operands`), with the state, in
    `dtype`."""
    return tuple(jnp.asarray(t, dtype) for t in (*ssd_operands(x, dt, A, B, C, chunk_size), state))


def cotangents(y, final, seed: int = 0):
    """A loss that reads every output, so no term of the scan drops out of the
    gradients under test."""
    keys = jax.random.split(jax.random.key(seed))
    return (jax.random.normal(keys[0], y.shape, jnp.float32),
            jax.random.normal(keys[1], final.shape, jnp.float32))


def mixer_operands(shape, seed: int = 0):
    """Operands in the layout `chunk_ssd` takes them, at a geometry the kernel
    is selected for: `(batch, steps, heads, head width, state, groups)`."""
    batch, steps, heads, head_dim, state_size, groups = shape
    keys = jax.random.split(jax.random.key(seed), 7)
    step = jax.nn.softplus(jax.random.normal(keys[1], (batch, steps, heads), jnp.float32))
    return (jax.random.normal(keys[0], (batch, steps, heads, head_dim), jnp.float32), step,
            -jnp.exp(jax.random.normal(keys[2], (heads,), jnp.float32)),
            jax.random.normal(keys[3], (batch, steps, groups, state_size), jnp.float32),
            jax.random.normal(keys[4], (batch, steps, groups, state_size), jnp.float32),
            jax.random.normal(keys[5], (heads,), jnp.float32),
            jax.random.normal(keys[6], (batch, heads, head_dim, state_size), jnp.float32))


def segment_sum(x):
    """`segment_sum` (modeling_mamba2.py:73-90): over the last axis of `x`,
    `out[..., i, j] = sum_{j < k <= i} x[..., k]` on and below the diagonal,
    `-inf` above, as the reference's masked cumulative sum rather than a
    difference of cumulative sums, so exp of it is the same numbers."""
    size = x.shape[-1]
    strict = jnp.tril(jnp.ones((size, size), jnp.bool_), -1)
    expanded = jnp.where(strict, jnp.broadcast_to(x[..., :, None], (*x.shape, size)), 0.0)
    summed = jnp.cumsum(expanded, axis=-2)
    inclusive = jnp.tril(jnp.ones((size, size), jnp.bool_))
    return jnp.where(inclusive, summed, -jnp.inf)


def _chunk_writes(x_c, B_c, a_c):
    """Each chunk's own write into the state, decayed to the chunk's end:
    `sum_j exp(sum_{j<k<C} a_k) x_j B_j^T`, `[NC, B, H, P, N]`.

    The decay from step j to the chunk's end is the suffix sum after j, a
    reverse cumulative sum shifted by one, where the reference takes the
    difference `acs_C-1 - acs_j` of two prefix sums. The two are the same
    number in exact arithmetic; the suffix sum never subtracts, which is
    what keeps a document reset exact (`RESET_DECAY`): a reset before j sits
    in both prefix sums, and their difference would lose every digit below
    its magnitude."""
    suffix = jax.lax.cumsum(a_c, axis=a_c.ndim - 1, reverse=True)
    to_end = jnp.exp(jnp.pad(suffix[..., 1:], [(0, 0)] * (a_c.ndim - 1) + [(0, 1)]))
    return exact_einsum('nbjhs,nbhj,nbjhp->nbhps', B_c, to_end, x_c)


def _carry(previous, step):
    """One chunk (or one shard) across the state: decay what entered by the
    span's total log decay and add the span's own write."""
    writes, log_decay = step
    return jnp.exp(log_decay)[..., None, None] * previous + writes, previous


def exact_einsum(spec: str, *operands):
    """`jnp.einsum` at HIGHEST, which the reference states: fp32 products in fp32, where a TPU's default
    takes one bf16 pass, which hid the kernel's own default-precision matmuls from check 1 (template
    numerics review 2026-10-26 follow-up, P1-1)."""
    return jnp.einsum(spec, *operands, precision=jax.lax.Precision.HIGHEST)


def xla_chunk_scan(x, B, C, a, carried, chunk_size: int):
    """The scan over chunks in plain XLA, the oracle and the fallback for
    `dew.nn.kernels.ssd`, on the operands as `chunk_ssd` hands either: `x`
    `[B, T, H, P]` already scaled by `dt`, `B` and `C` `[B, T, H, N]` with the
    groups expanded, `a` `[B, T, H]` the per-step `A dt`, the length whole
    chunks of `chunk_size`, and `carried` `[B, H, P, N]` the state entering the
    sequence; returns the output `[B, T, H, P]` and the state leaving it. The
    chunks lead its own einsums."""
    batch, total, heads, head_dim = x.shape

    def chunks(t):  # [B, T, ...] -> [NC, B, C, ...]
        blocked = t.reshape(batch, total // chunk_size, chunk_size, *t.shape[2:])
        return jnp.moveaxis(blocked, 1, 0)

    x_c, B_c, C_c = chunks(x), chunks(B), chunks(C)         # [NC, B, C, H, ...]
    a_c = jnp.moveaxis(chunks(a), 3, 2)                     # [NC, B, H, C]
    a_cumsum = jnp.cumsum(a_c, axis=-1)
    # 1. The intra-chunk output: L[i, j] = exp(segsum) masks C_i . B_j.
    L = jnp.exp(segment_sum(a_c))                           # [NC, B, H, C, C]
    G = exact_einsum('nbihs,nbjhs->nbhij', C_c, B_c)
    y_diag = exact_einsum('nbhij,nbjhp->nbihp', G * L, x_c)
    # 2. Each chunk's own write into the state, decayed to the chunk's end.
    states = _chunk_writes(x_c, B_c, a_c)
    # 3. The state crossing the chunks: the reference's `decay_chunk @ states`
    # over the padded [NC+1, NC+1] segment sums is this product carried
    # chunk by chunk.
    final, previous = jax.lax.scan(_carry, carried, (states, a_cumsum[..., -1]))
    # 4. The state each chunk reads, decayed to every position of it.
    y_off = exact_einsum('nbihs,nbhps,nbhi->nbihp', C_c, previous, jnp.exp(a_cumsum))
    return jnp.moveaxis(y_diag + y_off, 0, 1).reshape(batch, total, heads, head_dim), final


def evaluated(scan, operands, seeded):
    """A scan's output, final state and every operand gradient under both cotangents, pulled back through
    the gate (`reference_error.with_gradients`)."""
    output, *gradients = with_gradients(scan)(*operands, seeded)
    return dict(zip(("output", "final", "x", "B", "C", "A dt", "state"),
                    (*output, *gradients), strict=True))


def parity_functions(chunk_size: int, roundings: int, platform: str = "tpu"):
    """The kernel's output, final state and all five gradients for `platform` (interpreted off it),
    XLA's, and XLA's independently checked float64 truth. The kernel and the scan, two Dew paths,
    compile without YNNPACK fusions on the CPU lane (`without_ynn`)."""
    mine = without_ynn(lambda operands, seeded: evaluated(
        lambda *o: _chunk_scan(*o, None, chunk_size, platform), operands, seeded))
    xla = without_ynn(lambda operands, seeded: evaluated(
        lambda *o: xla_chunk_scan(*o, chunk_size), operands, seeded))
    recurrence = jax.jit(lambda operands, seeded: evaluated(stepwise_scan, operands, seeded))

    @as_given
    def truth(operands, seeded):
        with jax.default_device(jax.devices("cpu")[0]):
            exact, twin = recurrence(operands, seeded), xla(operands, seeded)
        for name in exact:
            assert_computes_the_oracle(np.asarray(twin[name]), np.asarray(exact[name]), name,
                                      roundings=roundings)
        return exact

    return mine, xla, truth


def assert_kernel_parity(draw, chunk_size: int, *, compiled: bool, roundings: int, seeds: int, label: str):
    """Every forward and gradient leaf on par over fresh inputs and exact coordinate orbits.

    The float64 XLA scan must still compute the independent recurrence on every draw.
    """
    platform = jax.default_backend() if compiled else "tpu"
    if compiled:
        # pallas interprets only for a platform the process holds no device of
        assert any(device.platform == platform for device in jax.devices())

    mine, xla, truth = parity_functions(chunk_size, roundings, platform)
    assert_as_exact(mine, xla, draw, label, seeds=seeds, truth_fn=truth)


def shape_case(shape, chunk_size: int, seed: int = 0):
    """`mixer_operands` at `shape`: a nonzero entering state, steps that decay a chunk's state by
    about e (Mamba-2's small steps; at `mixer_operands`' own a chunk's carry vanishes and its gradient
    with it), and document resets at a chunk's first step and inside one, in chunks apart from ones
    that carry."""
    x, dt, A, B, C, _, state = mixer_operands(shape, seed)
    x, b, c, a, state = blocks(x, dt / chunk_size, A, B, C, state, chunk_size)
    # A reset zeroes its chunk's carry, so the chunks it misses keep one.
    if a.shape[0] > 1:
        a = a.at[-1, 5].set(RESET_DECAY)
    if a.shape[1] > chunk_size:
        a = a.at[0, chunk_size].set(RESET_DECAY)
    return (x, b, c, a, state), chunk_size, shape[1] + shape[3] + shape[4]


def case_draw(reference, kind, case):
    """Fresh inputs at each original geometry; exact joint head, P and N permutations.

    The `fixture` cases use recorded input geometry, but regenerate XLA's outputs on every draw.
    Their reset positions and padding, and the shape cases' carry-preserving steps, stay unchanged.
    """
    def draw(seed):
        if kind == "fixture":
            chunk, resets = case
            shape = (*reference["scan.x"].shape, reference["scan.B"].shape[-1],
                     reference["scan.B"].shape[2])
            x, dt, A, B, C, _, state = mixer_operands(shape, seed)
            operands = blocks(x, dt, A, B, C, state, chunk)
            x, b, c, a, state = operands
            if resets:
                a = a.at[0, 7].set(RESET_DECAY).at[1, 20].set(RESET_DECAY)
                if a.shape[1] > chunk:
                    a = a.at[0, chunk].set(RESET_DECAY)
            operands = x, b, c, a, state
        else:
            operands, _, _ = shape_case(*case, seed)
        seeded = cotangents(operands[0], operands[-1], seed)
        arrays = ssd_orbit(dict(zip(("x", "B", "C", "a", "h0", "dy", "dfinal"),
                                   (*operands, *seeded), strict=True)), seed)
        return (tuple(arrays[name] for name in ("x", "B", "C", "a", "h0")),
                tuple(arrays[name] for name in ("dy", "dfinal")))

    return draw


SEEDS = {
    ("fixture", (32, False)): 256,
    ("fixture", (32, True)): 256,
    ("fixture", (128, False)): 256,
    ("fixture", (128, True)): 256,
    ("shape", ((2, 63, 4, 8, 8, 2), 64)): 128,
    ("shape", ((2, 64, 4, 8, 8, 2), 64)): 64,
    ("shape", ((2, 65, 4, 16, 8, 2), 64)): 32,
    ("shape", ((2, 193, 4, 16, 8, 2), 64)): 32,
    ("shape", ((2, 257, 2, 64, 128, 1), 256)): 128,
    ("shape", ((1, 1024, 4, 64, 64, 1), 128)): 64,
    ("shape", ((1, 4096, 4, 64, 64, 1), 128)): 256,
}
"""Sized by seeds_for on XLA's output, final state and all five gradients (on armada), resolving at
these counts against the TPU's own XLA on a v5e (11 cases) and, interpreted without YNNPACK's
fusions, the nine host cases on an AMD x86-64-v4 host."""


on_device = pytest.mark.tpu(reason="needs a tpu, the one backend the kernel is chosen on")
host_only = pytest.mark.skipif(
    jax.default_backend() in ("gpu", "tpu"),
    reason="interpreted on the CPU lane; a TPU judges its compiled copies, and no user runs it on a GPU")

# (batch, steps, heads, head width, state, groups), chunk: a tail of one step short, whole chunks, a tail
# of one step over, four chunks with a ragged tail, at widths the kernel takes and fewer groups than heads;
# then the default chunk at a production head width and state; compiled on a TPU, 1,024 and 4,096 steps at
# a chunk of 128 over 4 heads of 64 channels and a state of 64, which `ssd_kernel_runs` takes.
CASES = [
    *(pytest.param("fixture", (chunk, resets), False, id=f"fixture-{label}{'-resets' if resets else ''}",
                   marks=host_only)
      for chunk, label in ((32, "three-chunks-ragged"), (128, "one-chunk")) for resets in (False, True)),
    pytest.param("shape", ((2, 63, 4, 8, 8, 2), 64), False, id="one-short", marks=host_only),
    pytest.param("shape", ((2, 64, 4, 8, 8, 2), 64), False, id="whole-chunk", marks=host_only),
    pytest.param("shape", ((2, 65, 4, 16, 8, 2), 64), False, id="one-over", marks=host_only),
    pytest.param("shape", ((2, 193, 4, 16, 8, 2), 64), False, id="four-chunks-ragged", marks=host_only),
    pytest.param("shape", ((2, 257, 2, 64, 128, 1), 256), False, id="default-chunk", marks=host_only),
    *(pytest.param("shape", ((1, length, 4, 64, 64, 1), 128), True, id=f"compiled-{length}-steps",
                   marks=on_device) for length in (1024, 4096)),
]


# The host geometries compiled on a TPU, marked so that the TPU tier, which selects its tests by mark,
# runs them.
TPU_COPIES = [pytest.param(*param.values, id=f"tpu-{param.id}", marks=on_device)
              for param in CASES if not param.values[2]]


@pytest.mark.parametrize(("kind", "case", "compiled"), CASES + TPU_COPIES)
def test_the_kernel_is_as_exact_as_the_xla_scan(reference, kind, case, compiled):
    """The kernel, interpreted on the host or compiled on a TPU, against float64: every forward and
    gradient leaf over declared draws (check 1)."""
    if kind == "fixture":
        chunk_size = case[0]
        _, length, _, width = reference["scan.x"].shape
        roundings = length + width + reference["scan.B"].shape[-1]
    else:
        shape, chunk_size = case
        roundings = shape[1] + shape[3] + shape[4]
    assert_kernel_parity(case_draw(reference, kind, case), chunk_size, compiled=compiled,
                         roundings=roundings, seeds=SEEDS[kind, case], label=f"{kind} {case}")


HOST_CASES = [pytest.param(*param.values[:2], id=param.id) for param in CASES if not param.values[2]]


@pytest.mark.parametrize(("kind", "case"), HOST_CASES)
def test_the_scan_preserves_resets_and_initial_state_gradients(reference, kind, case):
    """The original host geometries run with every output/gradient shape and dtype intact.

    Resets remove entering-state influence exactly, including its final-only cotangent gradient;
    before a reset a nonzero state still influences the outputs.
    """
    operands, seeded = case_draw(reference, kind, case)(0)
    chunk_size = case[0] if kind == "fixture" else case[1]
    run = jax.jit(lambda inputs, cotangents: evaluated(
        lambda *args: _chunk_scan(*args, None, chunk_size, "tpu"), inputs, cotangents))
    actual = jax.tree.map(np.asarray, run(operands, seeded))
    expected = {"output": operands[0], "final": operands[-1],
                **dict(zip(("x", "B", "C", "A dt", "state"), operands, strict=True))}
    for name, value in actual.items():
        assert value.shape == expected[name].shape, name
        assert value.dtype == expected[name].dtype, name
        assert np.isfinite(value).all(), name

    zero_state = (*operands[:-1], np.zeros_like(operands[-1]))
    without_state = jax.tree.map(np.asarray, run(zero_state, seeded))
    final_only = jax.tree.map(np.asarray, run(operands, (np.zeros_like(seeded[0]), seeded[1])))
    resets = np.all(operands[3] == RESET_DECAY, axis=-1)
    for row in range(resets.shape[0]):
        starts = np.flatnonzero(resets[row])
        prefix = int(starts[0]) if starts.size else resets.shape[1]
        if prefix:
            assert not np.array_equal(actual["output"][row, :prefix],
                                      without_state["output"][row, :prefix])
        if starts.size:
            np.testing.assert_array_equal(actual["output"][row, prefix:],
                                          without_state["output"][row, prefix:])
            np.testing.assert_array_equal(actual["final"][row], without_state["final"][row])
            np.testing.assert_array_equal(final_only["state"][row], np.zeros_like(final_only["state"][row]))


DOCUMENTS = ((0, 70, 100), (0, 40, 150))
"""Each row's documents' first steps, over rows of 190 steps: a document that ends inside a chunk, one
shorter than a chunk, and one longer than two."""

FORKED = (((1, 2, 2), (1, 0, 0)), ((1, 2, 3), (1, 2, 0)))
"""Each row's documents' table slots and stored slots (`recurrence_chunk.Chunk.loads`, `Chunk.stores`), one
given memory a row then two stored: the first row's start continued by its first document, whose final both
later ones enter; the second row a chain from its start, each document entering its predecessor's final.
Every memory is entered, so no gradient's row is float64 zero, whose terms the gate would trace each
draw."""


def documents_draw(seed: int):
    """Two rows of three documents (`DOCUMENTS`) of a forked call (`FORKED`) over one given memory a row,
    each document's entry write (`recurrence_chunk.Chunk.entry_key`: two terms of a state's keys `[N]` and
    a head's values `[P]`, a weight a head), and cotangents of the output and the row's last memory."""
    x, dt, A, B, C, _, _ = mixer_operands((2, 190, 2, 16, 16, 1), seed)
    x, b, c, a = (jnp.asarray(t, jnp.float32) for t in ssd_operands(x, dt / 64, A, B, C, 1))
    keys = jax.random.split(jax.random.key(seed + 11), 6)
    given = 0.3 * jax.random.normal(keys[0], (2, 1, 2, 16, 16), jnp.float32)
    entry = (0.3 * jax.random.normal(keys[3], (2, 3, 2, 2, 16), jnp.float32),
             0.3 * jax.random.normal(keys[4], (2, 3, 2, 2, 16), jnp.float32),
             jax.random.uniform(keys[5], (2, 3, 2), jnp.float32))
    seeded = (jax.random.normal(keys[1], x.shape, jnp.float32),
              jax.random.normal(keys[2], (2, 2, 16, 16), jnp.float32))
    return (x, b, c, a, given, *entry), seeded


def documents_on_the_kernel(x, b, c, a, given, key, value, weight):
    """The kernel over a forked call, each document laid from a chunk's edge and taking its table slot's
    memory up there with its entry write added (`recurrence_chunk._ported_documents`): the output and the
    row's last memory."""
    from dew.nn.kernels.chunked_launch import AlignedDocuments, Forked
    from dew.nn.kernels.recurrence_chunk import _ported_documents

    loads, stores = (jnp.asarray([row[part] for row in FORKED], jnp.int32) for part in (0, 1))
    forked = Forked(jnp.asarray(DOCUMENTS, jnp.int32), loads, stores, (key, value, weight), 2)
    # The table's memories as a chunk's call holds them, `[N, P]`, the SSD's transposed.
    aligned, documents = _ported_documents(jnp.swapaxes(given, -1, -2), forked, x.shape[1], 64, jnp.float32,
                                           (16, 16))
    assert isinstance(aligned, AlignedDocuments)
    laid = aligned.laid(x, b, c, a)
    start = jnp.zeros((x.shape[0], *given.shape[2:]), jnp.float32)
    y, final = _chunk_scan(*laid, start, documents, 64, "tpu", 2)
    return aligned.unlaid(y), final


def documents_one_by_one(x, b, c, a, given, key, value, weight):
    """Each document a sequence of its own, a step at a time (`stepwise_scan`), in the row's order from its
    table slot's memory (a given one, or the final an earlier document stored) with its entry write
    added."""
    rows, lasts = [], []
    ends = np.concatenate([np.asarray(DOCUMENTS), np.full((2, 1), x.shape[1])], axis=1)
    for row in range(x.shape[0]):
        outs, table, final = [], [*given[row], None, None], None
        loads, stores = FORKED[row]
        for index, (first, end) in enumerate(zip(ends[row, :-1], ends[row, 1:], strict=True)):
            span = slice(int(first), int(end))
            entering = table[loads[index] - 1] if loads[index] else jnp.zeros_like(given[row, 0])
            entering = entering + weight[row, index][:, None, None] * jnp.einsum(
                'rhp,rhn->hpn', value[row, index], key[row, index])
            y, final = stepwise_scan(x[row:row + 1, span], b[row:row + 1, span], c[row:row + 1, span],
                                     a[row:row + 1, span], entering[None])
            outs.append(y[0])
            if stores[index]:
                table[given.shape[1] + stores[index] - 1] = final[0]
        rows.append(jnp.concatenate(outs, axis=0))
        lasts.append(final[0])
    return jnp.stack(rows), jnp.stack(lasts)


def test_documents_entering_their_own_states_scan_as_each_alone():
    """The TPU kernel over a forked call (a request's questions over its shared prefix's state, a chain's
    each over its predecessor's, `Chunk.loads`): each document laid from a chunk's edge, the kernel's carry
    taking its table slot's state up there with its entry write added (`Chunk.entry_key`) and its resident
    table keeping the finals its documents store; its output, the row's last state and every operand's
    gradient, the given states' and the entry writes' included, as each
    document scanned alone a step at a time in the row's order, from float64 (check 1)."""
    def under(scan):
        def run(operands, seeded):
            output, *gradients = with_gradients(lambda *o: scan(*o))(*operands, seeded)
            return dict(zip(("output", "final", "x", "B", "C", "A dt", "given", "entry keys", "entry values",
                             "entry weights"), (*output, *gradients[:8]), strict=True))
        return run

    mine = without_ynn(under(documents_on_the_kernel))
    theirs = jax.jit(under(documents_one_by_one))
    assert_as_exact(mine, theirs, documents_draw, "the TPU kernel's forked call", seeds=24)


@pytest.mark.tpu(reason="the recurrence matrix's TPU cells run on the TPU tier's own TPU")
def test_the_recurrence_matrix_holds_on_this_tpu():
    """The recurrence's check-1 matrix (tests/kernel_matrix.py) on a TPU: the rated scan's TPU kernel's
    training step on par with the portable math's, each cell on its own TPU (a v6e or a v5e)."""
    import recurrence_matrix
    from gpu_flow import cells_of, run_cells

    run_cells(cells_of(recurrence_matrix.a_recurrences_training_step_on_a_tpu_is_on_par))
