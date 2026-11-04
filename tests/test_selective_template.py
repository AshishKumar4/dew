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
from dew.nn.kernels.templates import bundles, mosaic_sm90
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


@pytest.mark.parametrize("packed", [False, True], ids=["one row", "packed documents"])
def test_the_template_is_as_exact_as_the_portable_scan_interpreted(packed):
    the_template_is_as_exact_as_the_portable_scan("interpreted", packed)


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


@pytest.mark.parametrize("packed, held", [(False, None), (True, None), (False, "into the template"),
                                          (False, "out of the template")],
                         ids=["one row", "packed documents", "a held state into the template",
                              "a held state out of the template"])
def test_a_mamba_layer_is_as_exact_under_the_template_interpreted(packed, held):
    """The layer's cases interpreted, a held state's into the template and out of it apart: through it on
    both calls it took 189 s on armada (6f7a1978a), each call 11 s a draw, past a test's two minutes."""
    the_layer_is_as_exact_under_the_template(packed, held)


def routed(target: str, states: int, c_dtype=jnp.float32, *, forms: str = "fast", named: bool = False):
    """The kernels and refusals the selective scan's dispatch (`recurrence_chunk.selective_scan`) reports
    for a scan traced for `target` over fp32 `x` and `B` and a `c_dtype` `C` of `states` states, two rows of
    a chunk's tokens and a channel block, under the run's `forms`, the template named where `named`: each
    route's `(kernel, refusal code)`; or the refusal it raises."""
    x = jnp.zeros((2, selective.CHUNK, selective.block_scan.THREADS), jnp.float32)
    a, b = jnp.zeros((x.shape[-1], states)), jnp.zeros((*x.shape[:2], states))
    named_template = generation.chosen(selective_scan="mosaic_gpu") if named else contextlib.nullcontext()
    with (generation.assumed(target), named_template, generation.routes() as found,
          generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms))):
        jax.eval_shape(functools.partial(recurrence_chunk.selective_scan, portable=portable),
                       x, x, a, b, b.astype(c_dtype), x[0, 0], None, None)
    return {(route.kernel, route.refusal and route.refusal.code) for route in found}


def test_the_selector_takes_the_template_where_it_is_qualified():
    """An H100's scan of whole blocks of Mamba's 16 states takes the template under the fast forms, held to
    recorded mamba_ssm there (row 2968), and the portable math under the accurate forms with those its
    refusal, as a call naming the template overrides; an A100's refuses it for its strided copies whatever
    the call names."""
    from dew.nn.kernels.selection import KernelRefused

    assert routed("sm90", 16) == {("mosaic_gpu", None)}
    assert routed("sm90", 16, forms="accurate") == {("xla", "forms")}
    assert routed("sm90", 16, forms="accurate", named=True) == {("mosaic_gpu", None)}
    with pytest.raises(KernelRefused) as refused:
        routed("sm80", 16, named=True)
    assert [refusal.code for refusal in refused.value.refusals] == ["strided copies"]


@pytest.mark.parametrize("states", [4, 12])
def test_the_selector_refuses_a_state_row_of_either_stream_no_bulk_copy_takes(states):
    """fp32 `x` and `B` with a bf16 `C` at `states` states, whose B rows a bulk copy's 16 bytes divide and
    whose C rows (8 or 24 bytes) they do not: the template's streams copy each unpadded in its own dtype, so
    a call naming the template is refused for C's rows (row 2975), and the same scan with an fp32 `C`, every
    row whole, is taken."""
    from dew.nn.kernels.selection import KernelRefused

    with pytest.raises(KernelRefused) as refused:
        routed("sm90", states, jnp.bfloat16, named=True)
    assert [refusal.code for refusal in refused.value.refusals] == ["state rows"]
    assert routed("sm90", states, named=True) == {("mosaic_gpu", None)}


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
    positions, resident = selective.backward_plan(target)
    held = selective.backward_programs(need, target)
    assert held == resident == selective.backward_programs(selective.backward_registers(positions), target), (
        need, held, resident, positions)


@lowers('sm90')
def test_a_launchs_backward_kernel_carries_its_launch():
    """The selective scan's training step lowered for an H100 (`kernel_compile.lowered_for`): its backward
    kernel, which JAX traces on the pullback after the launch's scope ended, carries the launch that traced
    its forward (`report.relaunched`), so a compile that finds it spilling refuses that launch and steps the
    backward's tokens down (row 3851)."""
    from kernel_compile import lowered_for

    from dew.training._lowering import kernel_calls

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

