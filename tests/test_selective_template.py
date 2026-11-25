"""Mamba's selective scan on the recurrence template (`templates.recurrence.selective`): its output and every
input's gradient on par from float64 with its reference's (`held_to`, check 1): the portable scan's
(`dew.nn.mixers.mamba.selective_scan`) interpreted, whose arithmetic the interpreted template's is, and on a
GPU mamba_ssm's own, which the template ports; one row and packed documents, interpreted off a GPU over two
channel blocks (a block's view of a wider row strided in global memory), and at Falcon Mamba 7B's widths on
a GPU (`tests/test_gpu_recurrence.py`);
and a Mamba layer through its selector under the template's choice (`recurrence_chunk.SELECTIVE`) on par
with the layer's portable scan, at the mixer's own dtypes (its scan in fp32), with packed documents and
with a held state."""
import contextlib
import dataclasses
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, here, interprets, lowers
from reference_error import (
    FEWEST_SEEDS,
    ahead_arm,
    as_given,
    assert_as_exact,
    declared_seeds,
    indexed_recording,
    recording,
    run_to_run,
    with_gradients,
)
from selective_support import (
    RATES_PAST_THE_RANGE,
    mamba_selective_side,
    rate_past_the_range,
    selective_draw,
    selective_xla,
    stepwise,
)
from test_recurrence_chunk import ROUTES, on_route
from upstream_support import upstream_arm

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import generation, recurrence_chunk
from dew.nn.kernels.chunked_launch import Forked
from dew.nn.kernels.templates import bundles, mosaic_sm90, report
from dew.nn.kernels.templates.recurrence import selective
from dew.nn.mixers.mamba import Mamba, selective_scan as portable
from dew.parity import MOST_SEEDS
from dew.parity_stages import REDRAW

pytestmark = pytest.mark.kernel

INTERPRETED = interprets()
"""The template interprets off a GPU, by the wgmma bundle, whose rings and barriers the interpreter models;
a GPU tier part's prepare states it as the node's GPU runs it (`generation_support.interprets`)."""

SHAPES = {"interpreted": (1, 70, 256, 16), "planned": (1, 512, 128, 16),
          "falcon-mamba-7b": (2, 2048, 8192, 16)}
"""Each case's rows, tokens, channels and states: a ragged chunk over two channel blocks, a channel block of
512 tokens whose backward blocks the tuner's token counts set apart (`PLANNED_VARIANTS`), and Falcon Mamba
7B's widths."""


ATOMIC = "the selective scan's B and C gradients sum by atomic adds (4448)"
"""Why the template's arms vary run to run on a GPU (`reference_error.run_to_run`): its backward adds B's and
C's gradients atomically by default, each run within check 1 (row 4448's declared contract: a fixed order
2 to 2.2 times the step), and in their programs' order where a run asks its reductions ordered."""


def draw(dtype, seed: int, rows: int, length: int, channels: int, states: int, packed: bool):
    """A draw's inputs, their outputs' cotangents last, host arrays (NumPy's: a GPU tier part's prepare
    computes nothing on its compile-only device, where JAX's draws raised before a site, rows 8242-10588):
    steps after a softplus, decays `-exp(N(0, 1/4))`, input and readout gates normal, `x`, `dt`, `B`, `C` and
    the cotangents in `dtype`, documents starting at a third and two thirds of each row where `packed`."""
    rng = np.random.default_rng([4448, seed])

    def normal(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    x = normal(rows, length, channels).astype(dtype)
    dt = np.logaddexp(0.0, normal(rows, length, channels) - 3.0).astype(dtype)
    a = -np.exp(0.5 * normal(channels, states))
    b, c = (normal(rows, length, states).astype(dtype) for _ in range(2))
    skip = normal(channels)
    starts = None
    if packed:
        starts = np.zeros((rows, length), bool)
        starts[:, (0, length // 3, 2 * length // 3)] = True
    dy = normal(rows, length, channels).astype(dtype)
    return (x, dt, a, b, c, skip), starts, dy


REGIMES = ("decays near zero", "a document at every chunk")
"""The regimes the template's backward fixes were measured in (`selective.PARTIALS`, its pairwise sums):
steps of softplus(N(3, 1)), so each channel's state decays to near zero within a few tokens; and a document
starting at every chunk's first token, so every chunk's carried gradient is cut."""


def regime_draw(regime: str, dtype, seed: int, rows: int, length: int, channels: int, states: int):
    """`draw`'s inputs in `regime` (`REGIMES`), with their documents' starts."""
    (x, dt, a, b, c, skip), _, dy = draw(dtype, seed, rows, length, channels, states, packed=False)
    starts = None
    if regime == "decays near zero":
        stepped = np.random.default_rng([4448, seed, 1]).standard_normal(dt.shape).astype(np.float32)
        dt = np.logaddexp(0.0, stepped + 3.0).astype(dtype)
    else:
        starts = np.zeros((rows, length), bool)
        starts[:, ::selective.CHUNK] = True
    return (x, dt, a, b, c, skip), starts, dy


def scanned(fn, starts):
    """`fn`'s output alone, its documents' starts `starts`."""
    return lambda *inputs: fn(*inputs, starts=starts)[0]


def held_to(portable_arm, draws, starts, label: str, template_arm):
    """A site's reference (check 1): on a GPU mamba_ssm's own selective scan run live (`mamba_selective_side`,
    a packed row's documents each a call from a zero state), the upstream the template ports, whose decay is
    `exp2f` under `--use_fast_math`, as the template's `exp2` lowers to the hardware's `ex2.approx` there
    (M-JS-79, M-JS-82); its distance from the portable scan reported beside the verdict, never judged. Off a
    GPU the portable scan: a lane is held to the tightest reference its own arithmetic can meet, and the
    interpreted template's `exp2` is the portable scan's own."""
    if INTERPRETED:
        return portable_arm
    distances = []
    for seed in range(FEWEST_SEEDS):
        args = draws(seed)
        exact = jax.tree.map(lambda x: np.asarray(x, np.float64), portable_arm(*args))
        for arm in (template_arm, portable_arm):
            found = jax.tree.leaves(arm(*args))
            distances.append([float(np.mean((np.asarray(a, np.float64) - t) ** 2))
                              for a, t in zip(found, jax.tree.leaves(exact), strict=True)])
    print(f"{label}: the template's leaves against the portable scan's (reported, not judged): "
          f"{np.asarray(distances[0::2]).mean(0)}", flush=True)
    return upstream_arm(lambda drawn: mamba_selective_side(drawn, starts), draws, (1 + REDRAW) * FEWEST_SEEDS,
                        most=8 * MOST_SEEDS, keyed=lambda *args: args[:-1])


@pytest.mark.parametrize("case", RATES_PAST_THE_RANGE)
def test_a_rate_past_float32s_range_scans_as_the_recurrence_interpreted(case):
    """The template's output, final state and every input's gradient at a rate whose `A log2(e)` passes
    float32's range over a ragged chunk (`selective_support.rate_past_the_range`, both vanishing decays and
    `e^-3` decays at subnormal `dt = 2^-126`) on par from float64 with the recurrence a token at a time
    (`selective_support.stepwise`, check 1; rows 8491, 8722)."""
    template = taken_scan(jnp.float32)
    exact = jax.jit(with_gradients(stepwise))
    assert_as_exact(jax.jit(with_gradients(template)), exact,
                    functools.partial(rate_past_the_range, case),
                    f"the selective template at a rate past float32's range, {case}", seeds=FEWEST_SEEDS,
                    truth_fn=exact)


@pytest.mark.parametrize("case", ("normal step", "zero step", "skip cancelling",
                                  "subnormal skip cancelling", "subnormal step",
                                  "decaying skip cancelling"))
def test_large_input_gates_with_normal_or_zero_steps_keep_dx_finite_interpreted(case):
    """Large `B` across four states on a selector-supported shape (`length=32`, `states=4`) through the
    public `recurrence_chunk.selective_scan` route (`mosaic_gpu`) with normal (`2^-4`), zero (`0.0`),
    skip-cancelling (`dt = 1`, `D = -2^127`, `A = 0` or `-3`), subnormal skip-cancelling (`dt = 2^-110`,
    `D = -2^-10`, `B = (2^100, 2^71)`), or subnormal (`dt = 2^-130`, `B = (2^120, 0)`) steps: its output,
    final state and every input's gradient on par from float64 with an independent range-safe reference
    (check 1; rows 8722, 9072, 9089, base reviews b9f004f74 F7, da734bf54 F3, c486fe8d7 F1)."""
    cases = ("normal step", "zero step", "skip cancelling", "subnormal skip cancelling", "subnormal step",
             "decaying skip cancelling")

    def draw(seed):
        rng = np.random.default_rng([8722, seed, cases.index(case)])
        length, channels, states = 32, 128, 4
        x = np.zeros((1, length, channels), np.float32)
        a = np.full((channels, states), -3.0 if case == "decaying skip cancelling" else 0.0, np.float32)
        scale = rng.uniform(0.5, 1.0, (1, length, 2)).astype(np.float32)
        b = np.zeros((1, length, states), np.float32)
        b[:, :, :2] = (2.0 ** 127) * scale
        c = np.zeros((1, length, states), np.float32)
        c[:, :, :2] = 1.0
        entering = np.zeros((1, channels, states), np.float32)
        if case == "normal step":
            dt = np.full((1, length, channels), 2.0 ** -4, np.float32)
            skip = np.zeros((channels,), np.float32)
        elif case == "zero step":
            dt = np.zeros((1, length, channels), np.float32)
            skip = np.zeros((channels,), np.float32)
        elif case in ("skip cancelling", "decaying skip cancelling"):
            factor = np.float32(rng.uniform(0.5, 1.0))
            dt = np.ones((1, length, channels), np.float32)
            skip = np.full((channels,), -(2.0 ** 127) * factor, np.float32)
            b[:, :, :2] = (2.0 ** 127) * factor
        elif case == "subnormal step":
            factor = np.float32(rng.uniform(0.5, 1.0))
            dt = np.full((1, length, channels), 2.0 ** -130, np.float32)
            skip = np.zeros((channels,), np.float32)
            b[:, :, 0] = (2.0 ** 120) * factor
            b[:, :, 1] = 0.0
        else:
            factor = np.float32(rng.uniform(0.5, 1.0))
            dt = np.full((1, length, channels), 2.0 ** -110, np.float32)
            skip = np.full((channels,), -(2.0 ** -10) * factor, np.float32)
            b[:, :, 0] = (2.0 ** 100) * factor
            b[:, :, 1] = (2.0 ** 71) * factor
        dy = np.zeros((1, length, channels), np.float32)
        dy[:, -1, :] = 1.0
        dfinal = np.zeros((1, channels, states), np.float32)
        return (jnp.asarray(x), jnp.asarray(dt), jnp.asarray(a), jnp.asarray(b), jnp.asarray(c),
                jnp.asarray(skip), jnp.asarray(entering), (jnp.asarray(dy), jnp.asarray(dfinal)))

    def routed(*operands):
        with generation.chosen(selective_scan="mosaic_gpu"):
            return recurrence_chunk.selective_scan(*operands, None, portable=portable)

    exact = jax.jit(with_gradients(stepwise))

    def range_safe(*operands):
        with jax.enable_x64(new_val=True):
            out = exact(*jax.tree.map(lambda arr: arr.astype(jnp.float64), operands))
        return jax.tree.map(lambda arr: arr.astype(jnp.float32), out)
    reference = range_safe if case in ("subnormal skip cancelling", "subnormal step") else exact
    assert_as_exact(jax.jit(with_gradients(routed)), reference, draw,
                    f"the selective template at large B, {case}", seeds=FEWEST_SEEDS, truth_fn=exact)


def taken_scan(dtype, **options):
    """The template's scan with `options` at the bundle it takes here (`Selective._fwd`'s): the interpreted
    target's off a GPU, interpreted, the device's on one (a GPU tier part's prepare the node's,
    `generation_support.here`), so no launch runs a family the host's multiprocessor does not (an A100
    interpreting wgmma's raised its consumers' registers past what it takes, the promotion tier's row 5413).
    An arm stated by the bundle's generation and dtype, the family built where it runs: a family object has
    no form an identity can state (`dew._stated`), and an arm closing over one ran unprepared on a GPU tier
    (rows 8491, 8722's 'its dew arm has no statement of what it computes')."""
    taken = generation.interpreted_target().generation if INTERPRETED else here()

    def scan(*operands, **given):
        return selective.selective_scan(*operands, family=bundles.of(taken, dtype), interpret=INTERPRETED,
                                        **options, **given)

    return scan


def the_template_is_as_exact_as_its_reference(case: str, packed: bool, dtype=jnp.bfloat16,
                                              tiles: selective.Tiles | None = None):
    """`case`'s output and every input's gradient through the template, at `tiles` where given (else the
    device's plan), on par from float64 with its reference's (`held_to`: mamba_ssm's on a GPU, the portable
    scan's interpreted), over check 1's fewest draws."""
    shape = SHAPES[case]
    planned = {} if tiles is None else {"chunk": tiles.chunk, "plan": (tiles.positions, tiles.resident)}
    template = taken_scan(dtype, **planned)

    def draws(seed):
        inputs, _, dy = draw(dtype, seed, *shape, packed)
        return (*inputs, dy)

    starts = draw(dtype, 0, *shape, packed)[1]
    label = f"the selective template, {case}{', packed' if packed else ''}"
    templated = jax.jit(with_gradients(scanned(template, starts)))
    reference = held_to(jax.jit(with_gradients(scanned(portable, starts))), draws, starts, label, templated)
    assert_as_exact(run_to_run(templated, ATOMIC), reference, draws, label, seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


# The template's scan interpreted, each case a check-1 site of its own, its eight draws in armada map tasks of
# four (`check1_shards`): whole, both took 200 s (site-tasks2 at d87423e30), a task of two seeds of both 91 s
# (site-tasks at c17a54df2); on each route (`ROUTES`).
FORKED = {
    "shared and chained": Forked(jnp.asarray([[0, 25, 46], [0, 10, 48]], jnp.int32),
                                 jnp.asarray([[1, 2, 2], [1, 2, 3]], jnp.int32),
                                 jnp.asarray([[1, 0, 0], [1, 2, 0]], jnp.int32), None, 2),
    "a slot a given state": Forked(jnp.asarray([[0, 30, 50], [0, 20, 44]], jnp.int32),
                                   jnp.asarray([[1, 2, 2], [1, 0, 2]], jnp.int32),
                                   jnp.asarray([[1, 0, 0], [1, 0, 0]], jnp.int32), None, 1)}
"""The interpreted shape's forked calls (`Chunk.loads`), two rows of three documents from one given state
each, laid from chunks' edges: a request's questions entering its shared part's final (the first row's later
two) and a chain, each document its predecessor's final; and one stored slot a row beside its one given state,
a document from a zero state among them, so the backward's given and stored slots are as many."""


def the_template_is_as_exact_over_a_forked_call(layout: str, dtype=jnp.bfloat16, channels: int | None = None):
    """`FORKED[layout]` through the template (`selective_scan`'s `forked`): its output, the row's last state
    and every input's gradient, its given states' included, on par from float64 with the portable scan's
    forked path (`dew.nn.mixers.mamba.selective_scan`'s), over check 1's fewest draws (row 5288), its
    `channels` the interpreted shape's where None."""
    forked = FORKED[layout]
    _, length, shaped, states = SHAPES["interpreted"]
    channels = shaped if channels is None else channels
    rows = forked.firsts.shape[0]
    template = taken_scan(dtype, forked=forked)

    def draws(seed):
        inputs, _, dy = draw(dtype, seed, rows, length, channels, states, packed=False)
        keys = jax.random.split(jax.random.key(seed + 9))
        given = 0.3 * jax.random.normal(keys[0], (rows, 1, channels, states))
        # The row's last state's cotangent beside the output's, so the backward's final reaches its slots.
        dfinal = jax.random.normal(keys[1], (rows, channels, states)).astype(dtype)
        return (*inputs, given, (dy, dfinal))

    reference = jax.jit(with_gradients(functools.partial(portable, chunk_size=selective.CHUNK,
                                                         forked=forked)))
    assert_as_exact(jax.jit(with_gradients(template)), reference, draws,
                    f"the selective template, a forked call, {layout}", seeds=FEWEST_SEEDS,
                    truth_fn=reference)


@pytest.mark.parametrize("layout", list(FORKED))
@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_the_template_is_as_exact_over_a_forked_call_interpreted(layout):
    """`the_template_is_as_exact_over_a_forked_call` off a GPU, over one forward program's block of 128
    channels: its backward interprets a program a row's channel, and at the shape's 256 a draw took 621 s
    of a CI task (row 6294), 99 s on an idle host against 50 at 128. A forked slot's reads and stores take
    the program's columns as its other operands do, which the unforked sites hold over two blocks; the GPU
    flow runs it at 256 (tests/test_gpu_recurrence.py)."""
    the_template_is_as_exact_over_a_forked_call(layout, channels=128)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_is_as_exact_as_its_reference_interpreted(route):
    """`the_template_is_as_exact_as_its_reference` over one row, on the route's bundle (`on_route`)."""
    with on_route(route):
        the_template_is_as_exact_as_its_reference("interpreted", packed=False)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_is_as_exact_as_its_reference_over_packed_documents_interpreted(route):
    """`the_template_is_as_exact_as_its_reference` over packed documents, on the route's bundle
    (`on_route`)."""
    with on_route(route):
        the_template_is_as_exact_as_its_reference("interpreted", packed=True)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_holds_its_hardest_regimes_interpreted(regime):
    """The template's output and every input's gradient on par from float64 with the portable scan's in each
    regime its backward's fixes were measured in (`REGIMES`), over check 1's fewest draws of fp32 operands,
    whose rounding a bf16 draw's hides, one row of a ragged chunk over two channel blocks: a backward adding
    B's and C's gradients through one buffer put B's 1.95 times the portable scan's distance with a document
    at every chunk (`selective.PARTIALS`), which no CPU test held once the regime cases went to the GPU flow
    alone (row 3133)."""
    shape = SHAPES["interpreted"]
    template = taken_scan(jnp.float32)

    def draws(seed):
        inputs, _, dy = regime_draw(regime, jnp.float32, seed, *shape)
        return (*inputs, dy)

    starts = regime_draw(regime, jnp.float32, 0, *shape)[1]
    assert_as_exact(run_to_run(jax.jit(with_gradients(scanned(template, starts))), ATOMIC),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template under {regime}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_with_owned_partials_is_as_exact_as_its_reference_interpreted(
        with_deterministic_ops):
    """`the_template_is_as_exact_as_its_reference` over packed documents where the backward's programs
    own their partial sums and walk their channels in turn (`Scan.owned`, a run asking its reductions
    ordered)."""
    the_template_is_as_exact_as_its_reference("interpreted", packed=True)


@pytest.mark.gpu(reason="the backward's atomic adds land in a GPU's order, which an interpreter fixes")
def test_the_backward_repeats_its_state_gradients_bit_for_bit_where_reductions_are_ordered(
        with_deterministic_ops):
    """The template's backward over 512 channels, 16 of them to each partial buffer (`selective.PARTIALS`),
    under a run asking its reductions ordered (`--xla_gpu_deterministic_ops`): every gradient bit for bit
    the same over repeated runs of one compiled step, B's and C's too, which the default's atomic adds sum
    in whatever order their programs run (row 4448's declared contract)."""
    shape = (1, 256, 512, 16)
    family = bundles.of(here(), jnp.float32)
    template = functools.partial(selective.selective_scan, family=family)
    inputs, starts, dy = draw(jnp.float32, 0, *shape, packed=False)
    # The step as an arm of no site (`ahead_arm`): a GPU tier's prepare compiles it, its run calls it.
    step = ahead_arm(jax.jit(with_gradients(scanned(template, starts))), "the ordered backward, repeated")
    first = step(*inputs, dy)
    for _ in range(4):
        again = step(*inputs, dy)
        pairs = zip(jax.tree.leaves(first), jax.tree.leaves(again), strict=True)
        for leaf, (one, other) in enumerate(pairs):
            assert np.array_equal(np.asarray(one), np.asarray(other)), leaf


SEEDED = Path(__file__).resolve().parent / "fixtures" / "mamba" / "selective_mamba_ssm_seeded.npz"


def the_template_is_as_exact_as_mamba_ssm(case: str):
    """`case` of the recording (tools/mamba_selective_reference.py's) through the scan's selector with the
    template named, under the run's fast forms, its operands in the case's dtype: the output and every
    input's gradient on par with recorded mamba_ssm on the case's declared draws, mamba_ssm's own distance
    from float64 at the same inputs the bound (check 1), the truth Dew's float64 portable scan, which the
    recorder held to the oracle at every seed. A GPU cell (tests/test_gpu_recurrence.py), the template
    compiled."""
    stored = recording(SEEDED)
    seeds = declared_seeds(SEEDED)[case]
    dtype = jnp.dtype(str(stored[f"{case}.dtype"]))
    draw, reference, truth = indexed_recording(
        stored, case, seeds, inputs_of=functools.partial(selective_draw, case=case),
        truth_fn=jax.jit(selective_xla))
    scan = functools.partial(recurrence_chunk.selective_scan, portable=portable)

    def run(index, arrays):
        del index
        with generation.chosen(selective_scan="mosaic_gpu"):
            return selective_xla(arrays, dtype, scan=scan)

    assert_as_exact(run_to_run(jax.jit(run), ATOMIC), as_given(reference), draw,
                    f"the selective template against mamba_ssm, "
                    f"{case}", seeds=seeds, truth_fn=as_given(truth))


def test_the_training_forward_keeps_a_state_a_backward_block():
    """What the template's training forward keeps for its backward over 8192 tokens, eight backward blocks of
    1024 (`selective.block_positions`): beyond its operands, the entering state of each block's first chunk,
    not of every 32-token chunk, which kept 64 times the states 2048-token blocks read, 4.3 GB over Falcon
    Mamba 7B's step (row 2970)."""
    rows, length, channels, states = 1, 8192, 128, 16
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))
    template = taken_scan(jnp.bfloat16)
    kept = jax.eval_shape(lambda *operands: jax.vjp(template, *operands)[1], *operands)
    beyond = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(kept)) - sum(
        operand.size * operand.dtype.itemsize for operand in operands)
    chunks = length // selective.CHUNK
    assert beyond < chunks * rows * states * channels * 4, beyond


LAYER = {"emb_features": 64, "intermediate_size": 128, "state_size": 16, "time_step_rank": 8,
         "chunk_size": 32}
"""A Mamba layer one channel block wide (`selective.block_scan.THREADS`), Mamba's 16 states."""


HELD = {"into the template": (False, True), "out of the template": (True, False),
        "through the template": (True, True)}
"""Where a held state's case takes the template (`the_layer_is_as_exact_under_the_template`), by name: on its
first call, which leaves the state, and on its second, which starts from it."""


def the_layer_is_as_exact_under_the_template(packed: bool, held: str | None, *, length: int = 70,
                                             dtype=jnp.bfloat16):
    """A Mamba layer in `dtype` (its scan widened to fp32, as the mixer runs it) over two rows of `length`
    tokens, through its selector with the template chosen (`generation.chosen`), on par from float64 with
    the layer's portable scan, output and every input's and parameter's gradient, over check 1's fewest
    draws: documents starting a third and two thirds in where `packed` (`MixerCall`'s segments), and the
    scan from a held state where `held` names one (`HELD`: the decode cache a first call of `length // 2`
    tokens leaves, the second call's tokens alone compared, the template on the first call, the second or
    both)."""
    layer = Mamba(**LAYER, dtype=dtype)

    def called() -> MixerCall:
        # Formed in the programs that read it, as the parameters in the draw: formed before the site, a GPU
        # tier's prepare ran them on its compile-only device, which runs no program (row 8098).
        metadata = None
        if packed:
            starts = jnp.zeros((2, length), jnp.int32).at[:, (length // 3, 2 * length // 3)].set(1)
            metadata = AttentionMetadata(coordinates=coordinates(segments=jnp.cumsum(starts, axis=1)))
        return MixerCall(metadata, decode=held is not None)

    def template_chosen():
        return generation.chosen(selective_scan="mosaic_gpu")

    def portable_chosen():
        return generation.chosen(selective_scan="xla")

    def run(module, *, template: bool):
        """The layer's arm: the template where `template` takes it (on the calls `held` names), else the
        portable scan."""
        taken = HELD.get(held, (True, True)) if template else (False, False)
        leaving, starting = (template_chosen if each else portable_chosen for each in taken)

        def mixed(variables, x):
            call = called()
            if held is None:
                with starting():
                    return module.apply(variables, x, call)
            first = length // 2
            with leaving():
                allocated = module.apply(variables, x[:, :1], call, mutable=["cache"])[1]
                _, cache = module.apply({**variables, **allocated}, x[:, :first], call, mutable=["cache"])
            with starting():
                return module.apply({**variables, **cache}, x[:, first:], call, mutable=["cache"])[0]

        return mixed

    def draws(seed):
        keys = jax.random.split(jax.random.key(seed))
        x = jax.random.normal(keys[0], (2, length, LAYER["emb_features"])).astype(dtype)
        out = (2, length if held is None else length - length // 2, LAYER["emb_features"])
        variables = layer.init(jax.random.key(1), jnp.zeros((2, length, LAYER["emb_features"]), dtype))
        return variables, x, jax.random.normal(keys[1], out).astype(dtype)

    truth = layer.clone(dtype=jnp.float64)
    assert_as_exact(run_to_run(jax.jit(with_gradients(run(layer, template=True))), ATOMIC),
                    jax.jit(with_gradients(run(layer, template=False))), draws,
                    f"a Mamba layer under the template{', packed' if packed else ''}"
                    f"{f', a held state {held}' if held else ''}",
                    seeds=FEWEST_SEEDS, truth_fn=jax.jit(with_gradients(run(truth, template=False))))


# The layer's cases interpreted, each a check-1 site of its own, its eight draws in armada map tasks of
# two (`check1_shards`): whole, each took 125 to 157 s (test_durations at 8e7d4ce444, row 3758), a task of
# two seeds of the four 170 s (site-tasks at c17a54df2), about 20 s a case's seed.
@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_mamba_layer_is_as_exact_under_the_template_interpreted():
    """`the_layer_is_as_exact_under_the_template` over one row."""
    the_layer_is_as_exact_under_the_template(packed=False, held=None)


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_packed_mamba_layer_is_as_exact_under_the_template_interpreted():
    """`the_layer_is_as_exact_under_the_template` over packed documents."""
    the_layer_is_as_exact_under_the_template(packed=True, held=None)


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_mamba_layer_from_a_held_state_into_the_template_is_as_exact_interpreted():
    """`the_layer_is_as_exact_under_the_template` from a held state, the template on the second call."""
    the_layer_is_as_exact_under_the_template(packed=False, held="into the template")


@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_a_mamba_layer_from_a_held_state_out_of_the_template_is_as_exact_interpreted():
    """`the_layer_is_as_exact_under_the_template` from a held state, the template on the first call."""
    the_layer_is_as_exact_under_the_template(packed=False, held="out of the template")


def dispatched(forms: str, *, named: bool, c_dtype, target: str | None = None):
    """The selective scan's dispatch (`recurrence_chunk.selective_scan`) under the run's `forms`, the template
    named where `named` (interpreted off a GPU), `C` in `c_dtype` and every other operand as drawn, traced
    for `target` where given (lowered for it on any host, its mesh's abstract devices stating that GPU,
    `generation.gpu_mesh`: an A100 host's own device lowered a kernel for its cp.async copies, rows 6235 and
    6236; never run here): the output alone."""
    def run(x, dt, a, b, c, skip):
        named_template = generation.chosen(selective_scan="mosaic_gpu") if named else contextlib.nullcontext()
        lowered_for = contextlib.ExitStack()
        if target is not None:
            lowered_for.enter_context(generation.assumed(target))
            lowered_for.enter_context(jax.sharding.use_abstract_mesh(
                generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), target)))
        with (lowered_for, named_template,
              generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms))):
            return recurrence_chunk.selective_scan(x, dt, a, b, c.astype(c_dtype), skip, None, None,
                                                   portable=portable)[0]

    return run


ROUTE_CASES = [("fast", jnp.float32), ("accurate", jnp.float32), ("fast", jnp.bfloat16)]
"""The scans whose dispatch takes the template where a call names it, interpreted here as off any GPU: under
the fast forms and the accurate ones, and a bf16 `C` beside fp32 `x` and `B`, whose rows a bulk copy takes
whole at Mamba's 16 states. Which kernel an H100's or an A100's selector takes unnamed is the GPU flow's to
run (tests/test_gpu_recurrence.py), each held there to recorded mamba_ssm."""


@pytest.mark.parametrize(("forms", "c_dtype"), ROUTE_CASES,
                         ids=[f"{forms}-C {jnp.dtype(c).name}" for forms, c in ROUTE_CASES])
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_selective_scan_through_its_dispatch_is_as_exact_as_its_reference(forms, c_dtype):
    """The selective scan through its dispatch with the template named, its placement and operand dtypes as
    a call hands them over: its output and every input's gradient on par from float64 with its reference's
    (`held_to`: mamba_ssm's on a GPU, the portable scan's interpreted) over drawn fp32 operands, `C` drawn
    in `c_dtype` (check 1), one row of a ragged chunk over two channel blocks (KT5: no route held, the
    scan's behaviour)."""
    shape = SHAPES["interpreted"]

    def draws(seed):
        (x, dt, a, b, c, skip), _, dy = draw(jnp.float32, seed, *shape, packed=False)
        return x, dt, a, b, c.astype(c_dtype), skip, dy

    def reference(x, dt, a, b, c, skip):
        return portable(x, dt, a, b, c.astype(jnp.float32), skip, None, None)[0]

    label = (f"the selective scan's dispatch, the template named, under the {forms} forms, C in "
             f"{jnp.dtype(c_dtype).name}")
    templated = jax.jit(with_gradients(dispatched(forms, named=True, c_dtype=c_dtype)))
    assert_as_exact(run_to_run(templated, ATOMIC),
                    held_to(jax.jit(with_gradients(reference)), draws, None, label, templated), draws, label,
                    seeds=FEWEST_SEEDS, truth_fn=jax.jit(with_gradients(reference)))


@pytest.mark.parametrize("states", [4, 12])
def test_a_named_template_refuses_a_state_row_no_bulk_copy_takes(states):
    """fp32 `x` and `B` with a bf16 `C` at `states` states, whose B rows a bulk copy's 16 bytes divide and
    whose C rows (8 or 24 bytes) they do not: the template's streams copy each unpadded in its own dtype, so a
    call naming the template is refused for C's rows, typed (row 2975), at the trust boundary of its operands'
    layout, where the same call with an fp32 C (16 or 48 bytes) is taken (row 4304)."""
    from dew.nn.kernels.selection import KernelRefused

    rows, length, channels = 1, selective.CHUNK, selective.block_scan.THREADS
    inputs = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in (
        (rows, length, channels), (rows, length, channels), (channels, states), (rows, length, states),
        (rows, length, states), (channels,)))
    jax.eval_shape(dispatched("fast", named=True, c_dtype=jnp.float32, target="sm90"), *inputs)
    with pytest.raises(KernelRefused) as refused:
        jax.eval_shape(dispatched("fast", named=True, c_dtype=jnp.bfloat16, target="sm90"), *inputs)
    assert [refusal.code for refusal in refused.value.refusals] == ["state rows"]


@COMPILES_ON_CPU
@lowers('sm90')
@pytest.mark.parametrize('packed', [False, True], ids=['one sequence', 'packed'])
def test_the_backward_holds_its_registers_at_the_residency_it_launches(packed):
    """The selective scan's backward over Mamba's 16 states compiled for an H100 (`kernel_compile`), its
    documents packed or not: its launch spills nothing at the registers its derived residency leaves a thread
    (`selective.backward_plan`, `mosaic.compiler_params`' `resident`), so a backward whose need grows past its
    share fails here rather than spilling on the GPU (rows 3589, 3801)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    rows, length, channels, states = 1, 2048, 256, 16
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))
    starts = jax.ShapeDtypeStruct((rows, length), bool) if packed else None

    def call():
        def trained(x, dt, A, B, C, D, starts=None):
            def scanned(*given):
                return selective.selective_scan(*given, starts=starts, family=mosaic_sm90.WGMMA)[0]

            out, pull = jax.vjp(scanned, x, dt, A, B, C, D)
            return pull(out)

        return (trained, (*operands, starts)) if packed else (trained, operands)

    (record,) = compile_mosaic_kernels([KernelCase('selective backward', call)], 'sm90')
    assert not mosaic_failures([record]), mosaic_failures([record])
    assert [kernel for kernel in record['kernels'] if 'selective_scan_backward' in kernel['kernel']], record


@COMPILES_ON_CPU
@pytest.mark.parametrize('target', [pytest.param(name, marks=lowers(name)) for name in ('sm80', 'sm90')])
def test_a_forked_calls_training_step_compiles(target):
    """The selective scan's training step over a forked call (`FORKED`'s, each row's documents from its
    table) compiled for an A100 and an H100 (`kernel_compile`): both launches lower (`failures`; a spill is
    its residency's, row 12905), where a document's state taken up from its table selected each vector of
    it on a scalar (its slot's load), which no warpgroup lowering broadcasts (a GH200's forked exactness
    test, 2026-11-19, and both compiles failed so, row 12904)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, failures

    forked = FORKED["shared and chained"]
    rows, length, channels, states = 2, 2048, 256, 16
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32),
        # The table's one given state a row (`FORKED`'s).
        jax.ShapeDtypeStruct((rows, 1, channels, states), jnp.float32))
    family = bundles.of(target, jnp.bfloat16)

    def call():
        def trained(*given):
            out, pull = jax.vjp(lambda *a: selective.selective_scan(*a, family=family, forked=forked)[0],
                                *given)
            return pull(out)

        return trained, operands

    (record,) = compile_mosaic_kernels([KernelCase('selective forked training', call)], target)
    assert not failures([record]), failures([record])
    names = [kernel['kernel'] for kernel in record['kernels']]
    assert [name for name in names if 'selective_scan_backward' in name], names
    assert [name for name in names if 'selective_scan' in name and 'backward' not in name], names


@lowers('sm90')
def test_a_launchs_backward_kernel_carries_its_launch():
    """The selective scan's training step lowered for an H100 (`kernel_compile.lowered_for`): its backward
    kernel, which JAX traces on the pullback after the launch's scope ended, carries the launch that traced
    its forward (`report.relaunched`), so a compile that finds it spilling refuses that launch and steps the
    backward's tokens down (row 3851)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    rows, length, channels, states = 1, 2048, 256, 16
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))

    def trained(*given):
        out, pull = jax.vjp(lambda *a: selective.SELECTIVE(*a)[0], *given)
        return pull(out)

    lowered = lowered_for(trained, operands, 'sm90')
    launches = {call.name: call.launched for call in kernel_calls(lowered, unrun=True)
                if call.kind == 'mosaic_gpu'}
    assert set(launches) == {'selective_scan', 'selective_scan_backward'}, launches
    assert launches['selective_scan'] is not None
    assert launches['selective_scan_backward'] == launches['selective_scan'], launches


def _bound(length: int):
    """The selective scan's template launch bound to a bf16 call of `length` tokens (one row, 256 channels,
    Mamba's 16 states), interpreted."""
    operands = (jax.ShapeDtypeStruct((1, length, 256), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((256, 16), jnp.float32),
        *(jax.ShapeDtypeStruct((1, length, 16), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((256,), jnp.float32))
    return selective.SELECTIVE.bind(*operands, interpret=True)


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_a_spill_steps_the_kernel_that_spilled():
    """`Selective.derived` past a compile's spill (`report.refuse`, a refusals scope): a backward kernel that
    spilled gives the call another config, legal and not the one refused, under which its output and every
    input's gradient are on par from float64 with the portable scan's (check 1); a forward kernel's spill,
    whose registers no tile sets, leaves the call no config (rows 3851, 3861)."""
    bound = _bound(SHAPES["interpreted"][1])
    target = generation.gpu_target('sm90')
    with generation.assumed(target):
        derived = selective.SELECTIVE.derived(bound, target)
        identity = report._identity(selective.SELECTIVE, bound, generation.gpu_target())
        with report.refusals():
            report.refuse([(identity, derived, {'selective_scan_backward'})])
            stepped = selective.SELECTIVE.derived(bound, target)
        with report.refusals():
            report.refuse([(identity, derived, {'selective_scan'})])
            assert selective.SELECTIVE.derived(bound, target) is None
        assert stepped is not None and stepped != derived
        assert selective.SELECTIVE.legal(bound, stepped, target)
    the_template_is_as_exact_as_its_reference("interpreted", packed=False, tiles=stepped)


@pytest.mark.parametrize('target', ['sm80', 'sm90'])
def test_the_tuners_variants_lead_with_the_derived_config_each_legal(target):
    """`Selective.variants` on an A100's and an H100's facts for Mamba-1 2.8B's bf16 scan: the derived config
    first, then its neighbours (`Selective.neighbours`), distinct and each legal there, so an opt-in tuning
    run has more than the derivation to time (row 1300: an A100's backward 14.35 ms beside mamba_ssm's 9.9,
    M-JS-237)."""
    operands = (jax.ShapeDtypeStruct((1, 2048, 5120), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((5120, 16), jnp.float32),
        *(jax.ShapeDtypeStruct((1, 2048, 16), jnp.bfloat16),) * 2, jax.ShapeDtypeStruct((5120,), jnp.float32))
    facts = generation.gpu_target(target)
    with generation.assumed(facts):
        bound = selective.SELECTIVE.bind(*operands)
        variants = selective.SELECTIVE.variants(bound)
        assert variants[0] == selective.SELECTIVE.derived(bound, facts)
        assert len(set(variants)) == len(variants) > 1
        assert all(selective.SELECTIVE.legal(bound, tiles, facts) for tiles in variants)


def _planned_variants() -> list[selective.Tiles]:
    """The configs the opt-in tuner may pick (`Selective.variants`) for the planned shape, 512 tokens
    (`SHAPES['planned']`: blocks of 128, 256 and 512 tokens at 1, 2 and 4 tokens a thread, 8 and 16 clamped
    to 512's), one of each the scan runs apart: its backward's block, residency and forward chunk."""
    rows, length, channels, states = SHAPES["planned"]
    operands = (jax.ShapeDtypeStruct((rows, length, channels), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((channels, states), jnp.float32),
        *(jax.ShapeDtypeStruct((rows, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((channels,), jnp.float32))
    found = {}
    for tiles in selective.SELECTIVE.variants(selective.SELECTIVE.bind(*operands, interpret=True)):
        planned = (selective.block_positions(length, tiles.positions), tiles.resident, tiles.chunk)
        found.setdefault(planned, tiles)
    return list(found.values())


PLANNED_VARIANTS = (selective.Tiles(128, 8, 3), selective.Tiles(128, 1, 7), selective.Tiles(128, 2, 6),
                    selective.Tiles(128, 4, 4), selective.Tiles(128, 16, 2), selective.Tiles(64, 8, 3))
"""The every-variant test's check-1 sites (below), one a config the tuner may pick that the scan runs apart,
stated: derived as the module was collected, every shard derived them again (11 s of
`test_kernel_matrix.py`'s 13 s collection) and a TPU host's raised, its device no multiprocessor (rows 12453,
12461, 12467). `test_every_variant_the_tuner_derives_is_a_gated_site` holds every derived one among them."""


@pytest.mark.parametrize("host", [None, "v5e"], ids=["this host", "a TPU host"])
def test_every_variant_the_tuner_derives_is_a_gated_site(host):
    """Every config the opt-in tuner may pick for the planned shape that the scan runs apart
    (`_planned_variants`) is a site the every-variant test holds to check 1 (`PLANNED_VARIANTS`), so no
    variant the tuner may time goes ungated; derived alike on a TPU host, an interpreted call's register model
    read under the interpreted target's facts (`Selective.proposes`), where under the TPU's it raised (rows
    12453, 12467)."""
    with generation.assumed(host):
        assert set(_planned_variants()) <= set(PLANNED_VARIANTS)


@pytest.mark.parametrize('tiles', PLANNED_VARIANTS, ids=lambda tiles: (
    f'{tiles.positions} tokens, {tiles.resident} programs, chunk {tiles.chunk}'))
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_every_tuner_variant_is_as_exact_interpreted(tiles):
    """Each config the opt-in tuner may pick (`Selective.variants`, `kernel-templates.md`: every config a
    tuning run may pick held to the gate) that the scan runs apart, interpreted at 512 tokens: the scan's
    output and every input's gradient on par from float64 with the portable scan's (check 1), its backward's
    block, residency and forward chunk each changing which thread and block computes a token (row 12343; at
    70 tokens every token count clamped to one block). Native arithmetic (FMAs, approximate exp2) is the GPU
    tier's to judge."""
    the_template_is_as_exact_as_its_reference("planned", packed=False, tiles=tiles)


def test_a_step_past_float32s_range_times_a_small_rate_decays_as_the_recurrence_interpreted():
    """A token whose step is `3 2^126` (finite in float32) decaying a live entering state with no write, its
    rates `-2^-126` and 0: each state `exp(-3)` and 1 times what entered, through the template and the
    portable scan alike, on par from float64 with the recurrence stepped by hand (check 1, every draw the
    same): the step taken to base 2 before its rate, `dt log2(e)` overflowed, decaying to 0 and NaN
    (mamba1b's review)."""
    rows, length, channels, states = 1, 32, 128, 16
    rates = np.where(np.arange(channels)[:, None] % 2 == 0, -2.0 ** -126, 0.0) * np.ones((1, states))

    def draws(seed):
        del seed
        dt = np.zeros((rows, length, channels), np.float32)
        dt[:, 0] = 3 * 2.0 ** 126
        zeros = np.zeros((rows, length, channels), np.float32)
        ones = np.ones((rows, length, states), np.float32)
        # Host arrays, as `draw`'s: a GPU tier part's prepare computes nothing on its device.
        return (zeros.astype(jnp.bfloat16), dt.astype(jnp.bfloat16), rates.astype(np.float32),
                ones.astype(jnp.bfloat16), ones.astype(jnp.bfloat16), np.zeros((channels,), np.float32),
                np.ones((rows, channels, states), np.float32))

    def stepped(x, dt, A, B, C, D, state):
        del x, B, D
        h = np.asarray(state, np.float64)
        outputs = []
        for t in range(length):
            h = np.exp(np.asarray(dt, np.float64)[:, t, :, None] * np.asarray(A, np.float64)) * h
            outputs.append(np.einsum('rdn,rn->rd', h, np.asarray(C, np.float64)[:, t]))
        return np.stack(outputs, axis=1), h

    template = taken_scan(jnp.bfloat16)
    assert_as_exact(jax.jit(template), jax.jit(portable), draws, "a step past float32's range, small rates",
                    seeds=FEWEST_SEEDS, truth_fn=as_given(stepped))


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_a_pullback_takes_each_operands_gradient_in_its_own_dtype_interpreted():
    """The template's scan with `x` in fp16 and the step `dt` in fp32, interpreted: its pullback's
    gradients in their operands' dtypes, the backward storing `dx` and `ddt` each in its own (row 1300: a
    custom derivative's cotangent of another dtype is refused), every gradient on par from float64 with the
    portable scan's (check 1)."""
    rows, length, channels, states = SHAPES["interpreted"]
    template = taken_scan(jnp.float16)

    def draws(seed):
        (x, dt, *rest), _, dy = draw(jnp.float16, seed, rows, length, channels, states, packed=False)
        return (x, dt.astype(jnp.float32), *rest, dy)

    templated = jax.jit(with_gradients(scanned(template, None)))
    reference = jax.jit(with_gradients(scanned(portable, None)))
    assert_as_exact(run_to_run(templated, ATOMIC), reference, draws, "the selective template, mixed dtypes",
                    seeds=FEWEST_SEEDS, truth_fn=reference)


@pytest.mark.parametrize('length', [129, 2048], ids=['129 tokens', '2048 tokens'])
def test_a_chunk_walks_no_more_padding_than_the_shortest(length):
    """The forward's derived chunk on an H100's facts walks the call's tokens padded to the shortest chunk's
    (`selective.CHUNK`) and no further, at tiles the template holds legal: 129 tokens walked 256 at a chunk of
    128 before row 3853, where 32 a chunk walks 160. Which chunk the derivation takes is its own."""
    target = generation.gpu_target('sm90')
    bound = _bound(length)
    with generation.assumed(target):
        tiles = selective.SELECTIVE.derived(bound, target)
        assert selective.SELECTIVE.legal(bound, tiles, target)
    shortest = -(-length // selective.CHUNK) * selective.CHUNK
    assert -(-length // tiles.chunk) * tiles.chunk == shortest


@lowers('sm90')
@pytest.mark.parametrize('packed', [False, True], ids=['one sequence', 'packed'])
def test_the_forwards_shared_memory_is_what_mosaic_allocates(packed, tmp_path):
    """The forward's shared memory as the template counts it (`forward_scope`, `resources.scratch_bytes`) the
    scratch Mosaic GPU's resource estimate gives its kernel lowered for an H100, at the derived chunk; and
    counted for an A100's family, whose walks scope their streams (row 3854, 3859)."""
    from kernel_compile import SHARED, _dumped, _estimates, dumps, lowered_for

    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.kernels.templates import resources

    length, states = 2048, 16
    operands = (jax.ShapeDtypeStruct((1, length, 256), jnp.bfloat16),) * 2 + (
        jax.ShapeDtypeStruct((256, states), jnp.float32),
        *(jax.ShapeDtypeStruct((1, length, states), jnp.bfloat16),) * 2,
        jax.ShapeDtypeStruct((256,), jnp.float32), None,
        jax.ShapeDtypeStruct((1, length), bool) if packed else None)
    target = generation.gpu_target('sm90')
    with generation.assumed(target):
        chunk = selective.SELECTIVE.derived(selective.SELECTIVE.bind(*operands), target).chunk
    with _dumped(str(tmp_path)):
        lowered = lowered_for(lambda *given: selective.SELECTIVE(*given)[0], operands, 'sm90')
    calls = [call for call in kernel_calls(lowered, unrun=True) if call.kind == 'mosaic_gpu']
    estimates = _estimates(calls, dumps(str(tmp_path), '.resources.txt'))
    (estimate,) = [found for name, found in estimates.items() if name.startswith('selective_scan__')
                   or name == 'selective_scan']
    dtypes = (jnp.bfloat16,) * 4
    scope = selective.forward_scope(mosaic_sm90.WGMMA, chunk, states, dtypes, packed)
    counted = resources.scratch_bytes(mosaic_sm90.WGMMA, scope)
    assert counted == int(SHARED.search(estimate).group(1)), (counted, estimate)
    ampere = bundles.of('sm80', jnp.bfloat16)
    assert resources.shared_bytes(ampere, selective.forward_scope(ampere, chunk, states, dtypes, packed)) > 0



def test_a_pick_measured_before_the_selective_scans_revision_names_another_op():
    """Row 12421: the selective scan's revision is its op's field, which its stored picks' identity reads
    (`tuning._identity`), so a pick measured at the revision before its backward's exp2 and fused joins names
    another op than the scan's: a class attribute alone, the instance kept KernelOp's revision 1, and those
    picks matched."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates.recurrence.selective import SELECTIVE

    before = dataclasses.replace(SELECTIVE, revision=1)
    assert tuning.encoded(tuning._identity(SELECTIVE)) != tuning.encoded(tuning._identity(before))
