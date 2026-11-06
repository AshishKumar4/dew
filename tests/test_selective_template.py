"""Mamba's selective scan on the recurrence template (`templates.recurrence.selective`): its output and every
input's gradient on par from float64 with the portable scan's (`dew.nn.mixers.mamba.selective_scan`, check
1), one row and packed documents, interpreted off a GPU over two channel blocks (a block's view of a wider
row strided in global memory), and at Falcon Mamba 7B's widths on a GPU (`tests/test_gpu_recurrence.py`);
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
from generation_support import lowers
from reference_error import (
    FEWEST_SEEDS,
    as_given,
    assert_as_exact,
    declared_seeds,
    indexed_recording,
    recording,
    with_gradients,
)
from selective_support import selective_draw, selective_xla

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import generation, recurrence_chunk
from dew.nn.kernels.templates import bundles, mosaic_sm90, report
from dew.nn.kernels.templates.recurrence import selective
from dew.nn.mixers.mamba import Mamba, selective_scan as portable

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"
"""The template interprets off a GPU, by the wgmma bundle, whose rings and barriers the interpreter models."""

SHAPES = {"interpreted": (1, 70, 256, 16), "falcon-mamba-7b": (2, 2048, 8192, 16)}
"""Each case's rows, tokens, channels and states: a ragged chunk over two channel blocks, and Falcon Mamba
7B's widths."""


def draw(dtype, seed: int, rows: int, length: int, channels: int, states: int, packed: bool):
    """A draw's inputs, their outputs' cotangents last: steps after a softplus, decays `-exp(N(0, 1/4))`,
    input and readout gates normal, `x`, `dt`, `B`, `C` and the cotangents in `dtype`, documents starting at a
    third and two thirds of each row where `packed`."""
    keys = jax.random.split(jax.random.key(seed), 7)
    x = jax.random.normal(keys[0], (rows, length, channels)).astype(dtype)
    dt = jax.nn.softplus(jax.random.normal(keys[1], (rows, length, channels)) - 3.0).astype(dtype)
    a = -jnp.exp(0.5 * jax.random.normal(keys[2], (channels, states)))
    b, c = (jax.random.normal(key, (rows, length, states)).astype(dtype) for key in keys[3:5])
    skip = jax.random.normal(keys[5], (channels,))
    starts = None
    if packed:
        starts = jnp.zeros((rows, length), bool).at[:, (0, length // 3, 2 * length // 3)].set(True)
    dy = jax.random.normal(keys[6], (rows, length, channels)).astype(dtype)
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
        dt = jax.nn.softplus(jax.random.normal(jax.random.key(seed + 1), dt.shape) + 3.0).astype(dtype)
    else:
        starts = jnp.zeros((rows, length), bool).at[:, ::selective.CHUNK].set(True)
    return (x, dt, a, b, c, skip), starts, dy


def scanned(fn, starts):
    """`fn`'s output alone, its documents' starts `starts`."""
    return lambda *inputs: fn(*inputs, starts=starts)[0]


def the_template_is_as_exact_as_the_portable_scan(case: str, packed: bool, dtype=jnp.bfloat16):
    """`case`'s output and every input's gradient through the template on par from float64 with the portable
    scan's, over check 1's fewest draws."""
    shape = SHAPES[case]
    family = mosaic_sm90.WGMMA if INTERPRETED else bundles.of(generation.device_generation(), dtype)
    template = functools.partial(selective.selective_scan, family=family, interpret=INTERPRETED)

    def draws(seed):
        inputs, _, dy = draw(dtype, seed, *shape, packed)
        return (*inputs, dy)

    starts = draw(dtype, 0, *shape, packed)[1]
    assert_as_exact(jax.jit(with_gradients(scanned(template, starts))),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template, {case}{', packed' if packed else ''}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


# The template's scan interpreted, each case a check-1 site of its own, its eight draws in armada map tasks of
# four (`check1_shards`): whole, both took 200 s (site-tasks2 at d87423e30), a task of two seeds of both 91 s
# (site-tasks at c17a54df2).
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_is_as_exact_as_the_portable_scan_interpreted():
    """`the_template_is_as_exact_as_the_portable_scan` over one row."""
    the_template_is_as_exact_as_the_portable_scan("interpreted", packed=False)


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_template_is_as_exact_as_the_portable_scan_over_packed_documents_interpreted():
    """`the_template_is_as_exact_as_the_portable_scan` over packed documents."""
    the_template_is_as_exact_as_the_portable_scan("interpreted", packed=True)


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
    family = mosaic_sm90.WGMMA if INTERPRETED else bundles.of(generation.device_generation(), jnp.float32)
    template = functools.partial(selective.selective_scan, family=family, interpret=INTERPRETED)

    def draws(seed):
        inputs, _, dy = regime_draw(regime, jnp.float32, seed, *shape)
        return (*inputs, dy)

    starts = regime_draw(regime, jnp.float32, 0, *shape)[1]
    assert_as_exact(jax.jit(with_gradients(scanned(template, starts))),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template under {regime}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


@pytest.mark.gpu(reason="a GPU's programs run in its own order, which an interpreter fixes")
def test_the_backward_repeats_its_gradients_bit_for_bit():
    """The template's backward over 512 channels, several to each partial buffer (`selective.partials_of`):
    every gradient bit for bit the same over repeated runs of one compiled step, as check 1's pairing of an
    arm's draws takes it. With every channel adding atomically into 32 buffers, B's gradient differed in its
    last bits between two runs on an H100 (rows 4392, 4448)."""
    shape = (1, 256, 512, 16)
    family = bundles.of(generation.device_generation(), jnp.float32)
    template = functools.partial(selective.selective_scan, family=family)
    inputs, starts, dy = draw(jnp.float32, 0, *shape, packed=False)
    step = jax.jit(with_gradients(scanned(template, starts)))
    first = jax.block_until_ready(step(*inputs, dy))
    for _ in range(4):
        again = jax.block_until_ready(step(*inputs, dy))
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

    assert_as_exact(jax.jit(run), as_given(reference), draw, f"the selective template against mamba_ssm, "
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
    template = functools.partial(selective.selective_scan, family=mosaic_sm90.WGMMA, interpret=True)
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
    variables = layer.init(jax.random.key(1), jnp.zeros((2, length, LAYER["emb_features"]), dtype))
    metadata = None
    if packed:
        starts = jnp.zeros((2, length), jnp.int32).at[:, (length // 3, 2 * length // 3)].set(1)
        metadata = AttentionMetadata(coordinates=coordinates(segments=jnp.cumsum(starts, axis=1)))
    call = MixerCall(metadata, decode=held is not None)

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
        return variables, x, jax.random.normal(keys[1], out).astype(dtype)

    truth = layer.clone(dtype=jnp.float64)
    assert_as_exact(jax.jit(with_gradients(run(layer, template=True))),
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
    for `target` where given (lowered for it, never run here): the output alone."""
    def run(x, dt, a, b, c, skip):
        named_template = generation.chosen(selective_scan="mosaic_gpu") if named else contextlib.nullcontext()
        assumed = contextlib.nullcontext() if target is None else generation.assumed(target)
        with (assumed, named_template,
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
def test_the_selective_scan_through_its_dispatch_is_as_exact_as_the_portable_scan(forms, c_dtype):
    """The selective scan through its dispatch with the template named, its placement and operand dtypes as
    a call hands them over: its output and every input's gradient on par from float64 with the portable
    scan's over drawn fp32 operands, `C` drawn in `c_dtype` (check 1), one row of a ragged chunk over two
    channel blocks (KT5: no route held, the scan's behaviour)."""
    shape = SHAPES["interpreted"]

    def draws(seed):
        (x, dt, a, b, c, skip), _, dy = draw(jnp.float32, seed, *shape, packed=False)
        return x, dt, a, b, c.astype(c_dtype), skip, dy

    def reference(x, dt, a, b, c, skip):
        return portable(x, dt, a, b, c.astype(jnp.float32), skip, None, None)[0]

    assert_as_exact(jax.jit(with_gradients(dispatched(forms, named=True, c_dtype=c_dtype))),
                    jax.jit(with_gradients(reference)), draws,
                    f"the selective scan's dispatch, the template named, under the {forms} forms, C in "
                    f"{jnp.dtype(c_dtype).name}",
                    seeds=FEWEST_SEEDS, truth_fn=jax.jit(with_gradients(reference)))


@pytest.mark.parametrize("states", [4, 12])
def test_a_named_template_refuses_a_state_row_no_bulk_copy_takes(states):
    """fp32 `x` and `B` with a bf16 `C` at `states` states, whose B rows a bulk copy's 16 bytes divide and
    whose C rows (8 or 24 bytes) they do not: the template's streams copy each unpadded in its own dtype, so a
    call naming the template is refused for C's rows, typed (row 2975), at the trust boundary of its operands'
    layout, where the same call with an fp32 C (16 or 48 bytes) is taken (row 4304)."""
    from dew.nn.kernels.selection import KernelRefused

    shape = (1, selective.CHUNK, selective.block_scan.THREADS, states)
    inputs, _, _ = draw(jnp.float32, 0, *shape, packed=False)
    jax.eval_shape(dispatched("fast", named=True, c_dtype=jnp.float32, target="sm90"), *inputs)
    with pytest.raises(KernelRefused) as refused:
        jax.eval_shape(dispatched("fast", named=True, c_dtype=jnp.bfloat16, target="sm90"), *inputs)
    assert [refusal.code for refusal in refused.value.refusals] == ["state rows"]


@lowers('sm90')
@pytest.mark.parametrize('packed', [False, True], ids=['one sequence', 'packed'])
def test_the_backward_holds_its_registers_at_the_residency_it_launches(packed):
    """The selective scan's backward over Mamba's 16 states compiled for an H100 (`kernel_compile`), its
    documents packed or not: its launch spills nothing at the registers its derived residency leaves a thread
    (`selective.backward_plan`, `mosaic.compiler_params`' `resident`), and the most programs the least
    spill-free cap ptxas finds (`register_need`) leaves room for is the residency its register model gives
    (`selective.backward_registers`), so a model whose decision parts from ptxas's, or a backward whose need
    grows past its share, fails here rather than spilling or leaving a program out on the GPU (rows 3589,
    3801)."""
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

    (record,) = compile_mosaic_kernels([KernelCase('selective backward', call)], 'sm90', least=True)
    assert not mosaic_failures([record]), mosaic_failures([record])
    (kernel,) = [kernel for kernel in record['kernels'] if 'selective_scan_backward' in kernel['kernel']]
    need = kernel['least']
    assert need is not None, kernel
    target = generation.gpu_target('sm90')
    positions, resident = selective.backward_plan(target, 16)
    held = selective.backward_programs(need, target)
    assert held == selective.backward_programs(selective.backward_registers(positions), target) >= resident, (
        need, held, resident, positions)


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
    launches = {call.name: call.launch for call in kernel_calls(lowered, unrun=True)
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


def test_a_spill_steps_the_kernel_that_spilled():
    """`Selective.derived` past a compile's spill (`report.refuse`, a refusals scope): a backward kernel that
    spilled halves the backward's tokens, its residency the planner's at their registers, the forward's
    chunk kept; a forward kernel's spill, whose registers no tile sets, leaves the call no config (rows 3851,
    3861)."""
    bound = _bound(2048)
    target = generation.gpu_target('sm90')
    with generation.assumed(target):
        derived = selective.SELECTIVE.derived(bound, target)
        identity = report._identity(selective.SELECTIVE, bound)
        with report.refusals():
            report.refuse([(identity, derived, {'selective_scan_backward'})])
            stepped = selective.SELECTIVE.derived(bound, target)
        with report.refusals():
            report.refuse([(identity, derived, {'selective_scan'})])
            assert selective.SELECTIVE.derived(bound, target) is None
    facts = generation.gpu_target(generation.INTERPRETED_GENERATION)
    fewer = derived.positions // 2
    assert stepped == selective.Tiles(derived.chunk, fewer, selective.backward_resident(fewer, 16, facts))


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
def test_the_forwards_shared_memory_is_what_mosaic_allocates(packed):
    """The forward's shared memory as the template counts it (`forward_scope`, `resources.scratch_bytes`) the
    scratch Mosaic GPU's resource estimate gives its kernel lowered for an H100, at the derived chunk; and
    counted for an A100's family, whose walks scope their streams (row 3854, 3859)."""
    from kernel_compile import SHARED, _estimates, _printed, lowered_for

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
    with _printed(MOSAIC_GPU_DUMP_RESOURCES='1') as printed:
        lowered = lowered_for(lambda *given: selective.SELECTIVE(*given)[0], operands, 'sm90')
    calls = [call for call in kernel_calls(lowered, unrun=True) if call.kind == 'mosaic_gpu']
    estimates = _estimates(calls, ''.join(printed))
    (estimate,) = [found for name, found in estimates.items() if name.startswith('selective_scan__')
                   or name == 'selective_scan']
    dtypes = (jnp.bfloat16,) * 4
    scope = selective.forward_scope(mosaic_sm90.WGMMA, chunk, states, dtypes, packed)
    counted = resources.scratch_bytes(mosaic_sm90.WGMMA, scope)
    assert counted == int(SHARED.search(estimate).group(1)), (counted, estimate)
    ampere = bundles.of('sm80', jnp.bfloat16)
    assert resources.shared_bytes(ampere, selective.forward_scope(ampere, chunk, states, dtypes, packed)) > 0

