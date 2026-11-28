"""The algebra's prices on a target (docs/design/algebra.md §1.6, §6.5): a product in the tensor mode its GPU
runs it in, a TPU's matrix units and its VMEM, XLA's schedule of kernels at its part's fixed latency, device
memory at its measured stream, and an XLA program's arithmetic at the rate each of its forms takes."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_algebra import rows
from test_algebra_program import two_products_summed, written_price

from dew import algebra
from dew.algebra import cost, program, terms
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost as kernel_cost


@pytest.mark.parametrize(("generation_name", "speedup"), [("sm90", 2), ("sm80", 1)])
def test_an_fp8_product_is_priced_in_the_tensor_mode_its_gpu_runs_it_in(generation_name, speedup):
    """An FP8 product's seconds on a GPU: a bf16 product's halved where its tensor cores multiply FP8
    (Hopper's warpgroup MMA), a bf16 product's where its codes are unpacked into 16-bit products (an A100's
    mma.sync), and on no GPU the CUDA cores' (row 3612)."""
    target = generation.TARGETS[generation_name]

    def seconds(dtype):
        priced, unpriced = kernel_cost.units(cost.product_work(4096, 4096, 4096, dtype=dtype),
                                             kernel_cost.rates(target, device=True))
        assert not any(unpriced)
        return max(priced)

    # Exact: each mode's price is the same FLOPs over the same facts, its rate a power of two apart.
    assert seconds(jnp.float8_e4m3fn) == seconds(jnp.bfloat16) / speedup


@pytest.mark.parametrize("target_name", ["v5e", "v6e"])
def test_an_fp32_product_on_a_tpu_is_its_matrix_units_bf16_passes(target_name):
    """An fp32 product priced for a TPU is its matrix units' bfloat16 passes (`jax.lax.Precision`'s: one at
    the default precision, three at HIGH, six at HIGHEST), a bf16 product's work that many times, all of it
    work the TPU's rates price (row 7201: counted as CUDA-core operations, which a TPU states no rate of,
    every product of a scan's chunk forms priced at nothing)."""
    tpu = generation.TARGETS[target_name]

    def work(dtype, precision):
        with program.passes(target=target_name):
            made = cost.product_work(1024, 1024, 1024, dtype=dtype, precision=precision)
        assert not any(kernel_cost.units(made, kernel_cost.rates(tpu, device=True))[1])
        return made.flops

    bf16 = work(jnp.bfloat16, None)
    assert [work(jnp.float32, precision) for precision in ("default", "high", "highest")] == [
        bf16, 3 * bf16, 6 * bf16]


def test_xlas_schedule_prices_each_kernel_its_parts_fixed_latency():
    """XLA's schedule (`cost.fused`) prices each kernel of a form, a product's or a reduction's (a root of
    XLA's fusion), at its part's measured fixed latency (`GpuTarget.fixed_seconds`, its launch and first
    byte) beside its work: a traced product and a chunk form's products (`cost.product_work`, one a product)
    alike, row 4081's receipts finding a small fusion's time that latency. On a part no one measured, the
    kernels are work no rate prices (`Schedule.unpriced`)."""
    hopper = generation.gpu_target("NVIDIA H100 80GB HBM3")
    schedule = cost.fused(hopper)

    def priced(fn, *shapes):
        region = terms.region(jax.make_jaxpr(fn)(*(jnp.ones(shape, jnp.bfloat16) for shape in shapes)))
        return written_price(region, schedule).work

    traced = priced(lambda a, b: jnp.sum(jnp.dot(a, b), axis=-1), (64, 64), (64, 64))
    assert traced.kernels == 2
    one, two = cost.product_work(64, 64, 64), cost.product_work(64, 64, 64, count=2)
    throughput = sum(kernel_cost.units(one, kernel_cost.rates(hopper, device=True))[0])
    # Exact: two products' work and kernels are one's doubled, each unit's seconds a power of two apart.
    assert schedule(one) == throughput + hopper.fixed_seconds
    assert schedule(two) == 2 * (throughput + hopper.fixed_seconds)
    assert cost.fused(generation.gpu_target("sm86")).unpriced(two)[-1] == 2


@pytest.mark.parametrize("target_name", ["v5e", "v6e"])
def test_xlas_schedule_on_a_tpu_prices_what_its_kernels_keep_in_vmem(target_name):
    """XLA's schedule on a TPU (`cost.fused`) prices the bytes its kernels move between the vector registers
    and VMEM (`Work.kept_bytes`) at the TPU's VMEM rate (`GpuTarget.vmem_rate`, row 7283): a leaf read once,
    a product's result and a reduction's written once and read once (a reduction's at a lane's four bytes),
    a chunk form's product's too (`cost.product_work`). A GPU's XLA keeps them on chip at no price, and a TPU
    no one measured leaves them work no rate prices."""
    tpu = generation.TARGETS[target_name]

    def kept(fn, *shapes):
        with program.passes(target=target_name):
            region = terms.region(jax.make_jaxpr(fn)(*(jnp.ones(shape, jnp.float32) for shape in shapes)))
            return written_price(region, cost.fused(tpu)).work

    product, summed = kept(jnp.dot, (64, 128), (128, 256)), kept(lambda x: jnp.sum(x, axis=-1), (64, 128))
    assert product.kept_bytes == 4 * (64 * 128 + 128 * 256 + 2 * 64 * 256)
    assert summed.kept_bytes == 4 * (64 * 128 + 2 * 64)
    assert kept(lambda x: jnp.sum(x, axis=-1), (4, 0)).kept_bytes == 2 * 4 * 4
    assert cost.product_work(64, 128, 256, count=3).kept_bytes == 3 * 2 * 64 * 256 * 4
    # On-chip memory's unit (`templates.cost.units`' fourth): exact, one quotient beside a zero.
    on_chip = 3
    assert (kernel_cost.units(product, kernel_cost.rates(tpu, device=True))[0][on_chip]
            == product.kept_bytes / tpu.vmem_rate)
    hopper, unmeasured = (kernel_cost.rates(generation.TARGETS[name], device=True) for name in ("sm90", "v4"))
    assert kernel_cost.units(product, hopper)[0][on_chip] == 0
    assert product.kept_bytes in kernel_cost.units(product, unmeasured)[1]


@pytest.mark.parametrize("target_name", ["v5e", "v6e"])
def test_a_tpu_tile_walks_staged_bytes_are_priced_at_its_vmem_rate(target_name):
    """A Pallas TPU template's staged bytes (`Work.shared_bytes`, its tiles copied into VMEM) are priced on
    on-chip memory's unit at the chip's VMEM rate in a tile walk (`templates.cost.rates`, `device=False`: the
    chip its one multiprocessor), where they were work no rate priced and a TPU template's forms were
    compared without them (row 8729); a TPU no one measured leaves them unpriced."""
    tpu, unmeasured = generation.TARGETS[target_name], generation.TARGETS["v4"]
    staged = kernel_cost.Work(shared_bytes=4.0 * 512 * 1024)
    on_chip = 3
    priced, unpriced = kernel_cost.units(staged, kernel_cost.rates(tpu, device=False))
    assert priced[on_chip] == staged.shared_bytes / tpu.vmem_rate
    assert not any(unpriced)
    assert staged.shared_bytes in kernel_cost.units(staged, kernel_cost.rates(unmeasured, device=False))[1]


def test_a_chunk_forms_vmem_traffic_is_priced_on_a_tpu():
    """A scan's chunk form on a v5e (`linear_scan.derive` under XLA's schedule, as `scan` derives it) keeps
    its products' results and a token step's state on chip (`Work.kept_bytes`), which the TPU's VMEM rate
    prices (row 7283): derived where the v5e states no rate, those bytes are work no rate prices, and the
    cheapest form's seconds are fewer by them."""
    from dew.algebra import linear_scan

    v5e = generation.TARGETS["v5e"]

    def derived(target):
        # Derived as scan derives on a v5e (`program.passes`' target): its products priced in the v5e's modes.
        with program.passes(target="v5e"):
            return linear_scan.derive(linear_scan.Extents(64, 128, 128, 1), corrects=False,
                                      schedule=cost.fused(target), precision=None,
                                      saturation=algebra.SATURATION)

    priced, unmeasured = derived(v5e), derived(dataclasses.replace(v5e, vmem_rate=None))
    assert priced.price.work.kept_bytes > 0
    assert unmeasured.price.work.kept_bytes in unmeasured.price.schedule.unpriced(unmeasured.price.work)
    assert priced.price.value > unmeasured.price.value


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_a_derivatives_sum_is_priced_as_the_sum_it_computes(dtype):
    """Price a cotangent sum's work whether it is a floating add term or an opaque integer equation."""
    from jax._src.ad_util import add_jaxvals_p

    def priced(fn):
        x, y = jnp.ones((64, 128), dtype), jnp.ones((64, 128), dtype)
        traced = jax.make_jaxpr(fn)(x, y)
        if (out := traced.jaxpr.eqns[-1]).primitive is add_jaxvals_p:
            assert terms.spoken_as(out) == ("add" if dtype == jnp.float32 else None)
        region = terms.region(traced)
        return written_price(region, cost.fused(generation.TARGETS["sm90"])).work

    accumulated = priced(lambda x, y: add_jaxvals_p.bind(x * y, x))
    added = priced(lambda x, y: x * y + x)
    assert accumulated.operations == added.operations
    assert accumulated.integer_operations == added.integer_operations


def test_a_product_of_two_16_bit_formats_is_priced_at_the_dtype_jax_gives_it():
    """A `lax.dot_general` of a bfloat16 and a float16 matrix, no preferred type stated, priced at the dtype
    JAX makes it (`terms.Product.made`, its traced result's, which the region records as its stated type),
    its result kept on chip at two bytes an element beside its leaves' (`Work.kept_bytes`); NumPy's
    promotion of the pair, which it refuses, raised there (base review 0b8b7ed98 F1). A bare product heads no
    rewrite window, so `dew.jit` alone never prices it (`test_algebra_fusion`'s witness holds its value)."""
    def product(a, b):
        return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))

    a, b = jnp.ones((2, 3), jnp.bfloat16), jnp.ones((3, 4), jnp.float16)
    region = terms.region(jax.make_jaxpr(product)(a, b))
    work = written_price(region, cost.fused(generation.TARGETS["v5e"])).work
    made = jax.eval_shape(product, a, b).dtype.itemsize
    assert work.kept_bytes == 2 * (2 * 3 + 3 * 4) + 2 * made * (2 * 4)


def test_a_product_of_two_16_bit_formats_replays_as_traced():
    """A product of a bfloat16 and a float16 matrix, no preferred type stated, its region's term replayed as
    the engine binds a term's equation (`terms.value_of`): its product bound with its parameters as traced,
    its preferred type unset, and its value the traced program's bit for bit, one program. The region wrote
    the dtype it makes into that preferred type, which the replay then bound (base review 3232beb7c); it
    keeps it apart (`terms.Product.made`)."""
    def product(a, b):
        return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))

    a, b = jnp.full((2, 3), 0.75, jnp.bfloat16), jnp.full((3, 4), 2.5, jnp.float16)
    traced = jax.make_jaxpr(product)(a, b)
    region = terms.region(traced)

    def replayed(a, b):
        def value(term):
            return terms.value_of(term, value, region, (a, b))
        return value(region.outputs[0])

    def dots(jaxpr) -> list[dict]:
        # Each product's parameters, within the programs an equation calls too (a jit's).
        return [dict(equation.params) for equation in programs.equations(jaxpr)
                if equation.primitive is jax.lax.dot_general_p]

    assert dots(jax.make_jaxpr(replayed)(a, b).jaxpr) == dots(traced.jaxpr)
    assert np.asarray(jax.jit(replayed)(a, b)).tolist() == np.asarray(jax.jit(product)(a, b)).tolist()


def test_a_tpu7x_prices_its_fp8_products_at_its_fp8_peak_where_it_states_no_rate_a_clock():
    """A TPU7x states its catalogued peaks and no MXU rate a clock (row 9026): a whole device prices an FP8
    product at its FP8 peak, and a tile walk, with no clock, prices neither."""
    target = generation.TARGETS["tpu7x"]
    device, tile = kernel_cost.rates(target, device=True), kernel_cost.rates(target, device=False)
    assert (device.tensor, device.eight) == (target.bf16_peak, target.fp8_peak)
    assert (tile.tensor, tile.eight) == (None, None)


def test_device_memory_is_priced_at_the_parts_measured_stream():
    """A GPU's device-memory bytes are priced at its part's measured stream (`GpuTarget.part`,
    tools/device_facts.py's plateau), whatever bandwidth its memory clock and bus state, which no transfer
    reaches; a part no one measured, at its clocks' (row 3612)."""
    hopper = generation.TARGETS["sm90"]

    def memory_rate(device_kind, clocked):
        target = dataclasses.replace(hopper, device_kind=device_kind, memory_bandwidth=clocked)
        return kernel_cost.rates(target, device=True).device

    (measured,) = {memory_rate("NVIDIA GH200 480GB", clocked) for clocked in (4.0e12, 4.9e12)}
    assert measured is not None and measured < 4.0e12
    # A part no device reports (its kind the test's own), which no row's streams can name: its clocks'.
    assert memory_rate("a Hopper part no one measured", 2.0e12) == 2.0e12


def test_a_dies_step_latencies_are_its_parts_at_its_clock_and_a_parts_memory_its_own():
    """A step's latencies are a die's at its clock (`GpuTarget.clocked`), every part of it at that clock
    reading them: an A100-SXM4's 40 GB and 80 GB parts alike (one GA100 at one 1410 MHz boost), its PCIe
    part, of another power limit, none; a part's memory its own (`GpuTarget.measured`): the 40 GB part's
    stream, which its 80 GB part does not read. A Workstation RTX PRO 6000, of another clock and power than
    the Server Edition measured, reads none of its latencies, and a GH200 its own, not an H100 SXM's (row
    14183: each was carried to every part of its generation)."""
    def facts(kind):
        target = generation.gpu_target(kind)
        return (target.step_seconds, target.kernel_step_seconds, target.product_latency,
                target.barrier_seconds), target.fixed_seconds

    sxm40, sxm80, pcie = (facts(kind) for kind in ("NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB",
                                                   "NVIDIA A100 PCIe"))
    assert sxm40[0] == sxm80[0] and sxm40[0][1] is not None and pcie[0] == (None,) * 4
    assert sxm40[1] is not None and sxm80[1] is None
    server, workstation = (facts(f"NVIDIA RTX PRO 6000 Blackwell {edition} Edition")
                           for edition in ("Server", "Workstation"))
    assert None not in server[0] and workstation[0] == (None,) * 4
    # A GH200 its own (M-FS-114), never the H100 SXM's of another power limit.
    gh200, h100 = (generation.gpu_target(kind).kernel_step_seconds
                   for kind in ("NVIDIA GH200 480GB", "NVIDIA H100 80GB HBM3"))
    assert None not in (gh200, h100) and gh200 != h100


@pytest.mark.parametrize("tpu", ["v5e", "v6e"])
def test_a_tpus_program_is_rewritten_by_the_rates_it_states(tpu):
    """A TPU's target states its MXU's rate and not its vector or transcendental units' (`cost.cheaper`): a
    program of a product reduced (`sum(a * b)` and two joinable sums of products) traces and runs rewritten
    for it, on par with the program (check 1), a form taken only where it does no more work at a rate the
    TPU states none of."""
    with algebra.passes(target=tpu):
        assert_as_exact(jax.jit(algebra.rewritten(two_products_summed)), jax.jit(two_products_summed),
                        lambda seed: rows(seed, (8, 64), (8, 64), (8, 64)), f"a {tpu} program", seeds=16)


def test_a_products_mode_prices_it():
    """A product of fp32 operands at the default precision (`Precision.DEFAULT`, stated: the suite configures
    the highest) takes the tensor cores' TF32 mode, one of 16-bit operands their 16-bit mode, and one at
    `Precision.HIGHEST` the CUDA cores (`cost.work`): an H100's TF32 rate is half its 16-bit one and its
    CUDA cores' far below both, so the three price apart."""
    def product(precision, dtype, result=jnp.float32):
        def contracted(a, b):
            return jnp.dot(a, b, precision=precision, preferred_element_type=result)

        a, b = (jnp.ones((64, 64), dtype) for _ in range(2))
        region = terms.region(jax.make_jaxpr(contracted)(a, b))
        return written_price(region, cost.fused(cost.target("sm90"))).work

    half, tf32, highest = (product(None, jnp.bfloat16), product(jax.lax.Precision.DEFAULT, jnp.float32),
                           product(jax.lax.Precision.HIGHEST, jnp.float32))
    flops = 2.0 * 64 ** 3
    # The CUDA cores' FFMAs, a multiply an instruction and its add fused into it (`Work.fused_multiply_adds`).
    assert (half.flops, tf32.tf32_flops, highest.operations, highest.fused_multiply_adds) == (
        flops, flops, flops / 2, flops / 2)
    # A dot algorithm by its fields is its preset's: three 16-bit products.
    spelled = jax.lax.DotAlgorithm(jnp.bfloat16, jnp.bfloat16, jnp.float32, num_primitive_operations=3)
    assert product(spelled, jnp.float32) == product(jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3, jnp.float32)
    assert product(spelled, jnp.float32).flops == 3 * flops
    # A preset is priced by what JAX lowers it to, whatever its operands' storage: bf16 products of fp32
    # storage accumulated in bf16 are the 16-bit mode's, where a preset not named `..._F32` was the CUDA
    # cores'.
    narrow = product(jax.lax.DotAlgorithmPreset.BF16_BF16_BF16, jnp.float32, jnp.bfloat16)
    assert (narrow.flops, narrow.operations) == (flops, 0.0)
    # An FP8 dot algorithm's products are the 8-bit mode's, as FP8 operands' are.
    eight = jax.lax.DotAlgorithm(jnp.float8_e4m3fn, jnp.float8_e4m3fn, jnp.float32)
    assert product(eight, jnp.float8_e4m3fn).fp8_flops == flops


def _priced(fn, *shapes, schedule=None):
    """`fn`'s program over fp32 ones of `shapes`, its price (`cost.model`, under XLA's schedule on an H100
    where none is given) and its traced program."""
    traced = jax.make_jaxpr(fn)(*(jnp.ones(shape, jnp.float32) for shape in shapes))
    region = terms.region(traced)
    return written_price(region, schedule or cost.fused(generation.TARGETS["sm90"])), traced


@pytest.mark.parametrize("name", ["power", "clamp", "exponential", "selection", "conversion", "reduction"])
def test_a_terms_arithmetic_is_what_the_kernels_counter_prices_its_operation_at(name):
    """A term's arithmetic (`cost.work`) is what the kernels' counter prices its equation at in a kernel's
    body (`emission.operation_work`, the one rule): a power an exponential an element, a clamp a maximum and
    a minimum, an exponential, a comparison and a selection, a conversion and a reduction theirs, both on the
    H100 the price names, where the algebra's own table priced a power as an operation and the counter a
    clamp as one (row 5901; on the host's own target, an exponential's forms differed, row 13154)."""
    from dew.nn.kernels.templates import emission

    fn = {"power": jax.lax.pow,
          "clamp": lambda x, y: jax.lax.clamp(np.float32(-0.5), x, np.float32(0.5)) + y,
          "exponential": lambda x, y: jax.lax.exp(x) * y,
          "selection": lambda x, y: jax.lax.select(x < y, x, y),
          "conversion": lambda x, y: jax.lax.convert_element_type(x, jnp.bfloat16) * y.astype(jnp.bfloat16),
          "reduction": lambda x, y: jnp.sum(x * y, axis=-1)}[name]
    price, traced = _priced(fn, (64, 128), (64, 128))
    # Counted on the H100 its price names, whatever GPU the process holds or none: an fp32 exponential's
    # forms are its multiprocessor's (`emission.ACCURATE_FORMS`), a CPU host's one MUFU apart from them.
    with generation.assumed("sm90"):
        counted = emission.emitted(traced.jaxpr, 1, None).fixed
    fields = ("operations", "exponentials", "conversions")
    assert [getattr(price.work, field) for field in fields] == [getattr(counted, field) for field in fields]


def test_an_xla_programs_division_prices_on_its_target_at_its_dtype():
    """An XLA program's fp32 division priced for an H100 issues `div.rn.f32`'s MUFU and refinement
    (`emission.IEEE_DIVISION`, M-FS-77) on whatever host prices it, and for a TPU an operation an element; an
    int32 division and a bfloat16 one an operation an element on either. Traced casts to fp32 make an fp32
    quotient even when its operands are stored in bf16 (rows 12974, 13033)."""
    hopper, tpu = generation.TARGETS["sm90"], generation.TARGETS["v5e"]
    elements = 64 * 128

    def priced(dtype, target):
        rows = jax.ShapeDtypeStruct((64, 128), dtype)
        closed = jax.make_jaxpr(lambda x, y: jax.lax.div(x, y))(rows, rows).jaxpr
        return program.priced(closed, target).work

    assert (priced(jnp.float32, hopper).exponentials, priced(jnp.float32, hopper).operations) == (
        elements, 6 * elements)
    for dtype, target in ((jnp.float32, tpu), (jnp.int32, hopper), (jnp.bfloat16, hopper)):
        assert priced(dtype, target).exponentials == 0, (dtype, target)
    # A masked fp32 division (its mask's bools beside its fp32 values) is fp32's.
    rows = jax.ShapeDtypeStruct((64, 128), jnp.float32)
    masked = jax.make_jaxpr(lambda x, y: jnp.where(y > 0, x / y, 0.0))(rows, rows).jaxpr
    assert program.priced(masked, hopper).work.exponentials == elements
    narrow = jax.ShapeDtypeStruct((64, 128), jnp.bfloat16)
    mixed = jax.make_jaxpr(lambda x, y: x.astype(jnp.float32) / y.astype(jnp.float32))(narrow, narrow).jaxpr
    mixed_work = program.priced(mixed, hopper).work
    assert (mixed_work.conversions, mixed_work.exponentials, mixed_work.operations) == (
        2 * elements, elements, 6 * elements)
    reciprocal = jax.make_jaxpr(lambda x: jax.lax.integer_pow(x.astype(jnp.float32), -1))(narrow).jaxpr
    reciprocal_work = program.priced(reciprocal, hopper).work
    assert (reciprocal_work.conversions, reciprocal_work.exponentials, reciprocal_work.operations) == (
        elements, elements, 6 * elements)
    # A quotient a rewrite made, which no trace typed (`rules._ranged`'s of a bfloat16 program), its dtype
    # unproven: an operation an element, as a bfloat16 one (the review of lane/toolkit f8963150b).
    assert cost._priced(jax.lax.div_p, {}, [(64,), (64,)], (64,), hopper) == cost.Work(operations=64)


def test_an_xla_programs_special_function_and_power_price_their_forms_at_fp32_alone():
    """An XLA program's fp32 log priced for an H100 issues libdevice's polynomial (`emission.XLA_FORMS`,
    M-FS-78) and its `x ** -2` a multiply beside `div.rn.f32`'s MUFU and refinement; a float64 program's
    log and power, whose forms Dew measured none of, an exponential an element and a multiply and an
    operation (the review of lane/toolkit e59a86a32: the algebra priced every unary term at fp32)."""
    from dew.nn.kernels.templates import emission

    hopper = generation.TARGETS["sm90"]
    elements = 64 * 128

    def priced(function, dtype):
        rows = jax.ShapeDtypeStruct((64, 128), dtype)
        return program.priced(jax.make_jaxpr(function)(rows).jaxpr, hopper).work

    log = emission.XLA_FORMS[jax.lax.log_p]
    assert (priced(jnp.log, jnp.float32).operations, priced(jnp.log, jnp.float32).exponentials) == (
        log.operations * elements, 0)
    assert priced(lambda x: x ** -2, jnp.float32).exponentials == elements
    narrow = jax.ShapeDtypeStruct((64, 128), jnp.bfloat16)
    wide = jax.ShapeDtypeStruct((64, 128), jnp.float32)
    index = jax.ShapeDtypeStruct((64, 128), jnp.int32)
    mixed = jax.make_jaxpr(lambda x: jnp.log(x.astype(jnp.float32)))(narrow).jaxpr
    indexed = jax.make_jaxpr(lambda x, i: (jnp.log(x), i + 1))(wide, index).jaxpr
    for traced, conversions, integer_operations in ((mixed, 1, 0), (indexed, 0, 1)):
        work = program.priced(traced, hopper).work
        assert (work.operations, work.conversions, work.integer_operations, work.exponentials) == (
            log.operations * elements, (log.conversions + conversions) * elements,
            (log.integer_operations + integer_operations) * elements, 0)
    with jax.enable_x64(new_val=True):
        assert (priced(jnp.log, jnp.float64).exponentials, priced(jnp.log, jnp.float64).operations) == (
            elements, 0)
        power = priced(lambda x: x ** -2, jnp.float64)
        assert (power.exponentials, power.operations) == (0, 2 * elements)


def test_an_operation_no_rule_prices_makes_no_form_cheaper_for_taking_more_of_it():
    """An operation no rule prices (an atan2: no kernel's body holds one and no probe has measured it) is
    work at no rate (`Work.unpriced_operations`), so a form taking more of it is never cheaper
    (`cost.cheaper`), whatever its priced work's seconds: priced as one operation an element, two atan2s
    compared cheaper than one beside a product."""
    one, _ = _priced(lambda x, y, w: jax.lax.atan2(x, y) + jnp.sum(jnp.dot(w, w)), (64, 128), (64, 128),
                     (512, 512))
    two, _ = _priced(lambda x, y, w: jax.lax.atan2(x, y) + jax.lax.atan2(y, x) + jnp.sum(w), (64, 128),
                     (64, 128), (512, 512))
    assert two.value < one.value and not cost.cheaper(two, one)
    assert (one.work.unpriced_operations, two.work.unpriced_operations) == (64 * 128, 2 * 64 * 128)
