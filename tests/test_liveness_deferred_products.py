"""The corrected liveness pass on independent warpgroup products, beside ptxas.

A loop of `BATCHES` batches of `PRODUCTS` independent zero-started products ([64, 16] fp32, K = 64 as four
chained `m64n16k16` steps, `commit_group` and `wait_group 0` after each, as Mosaic emits them), each summed
into its batch's running sum right after its wait and the sum folded into the batch's own carried tile (the
recurrence's states launch: 12 products in two batches, the sm90 SSD fp32 forward). The program order needs
one product's accumulators beside the running sum and the carried tiles. ptxas issues every product of the
straight-line region before it reads any accumulator, so it allocates them all (`mosaic_gpu` Nemotron-H 4B
SSD, `recurrence_states` at Tiles(16, 3, None, 1): 96 of its 202 live registers at the peak, the pass's 117;
its need 200)."""


import kernel_compile
import kernel_liveness
import pytest
from jax._src.interpreters import mlir as jax_mlir
from jax.extend.mlir import ir as mlir_ir
from jaxlib.mlir.dialects import gpu as gpu_dialect  # noqa: F401  (registers the dialect with the context)

STRUCT = "!llvm.struct<(" + ", ".join(["vector<1xf32>"] * 8) + ")>"
ASM = (
    '"{ .reg .pred p; setp.ne.b32 p, $18, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 '
    '{$0,$1,$2,$3,$4,$5,$6,$7}, $16, $17, p, $19, $20, $21, $22; }\\0A", '
    '"=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,l,l,n,n,n,n,n"'
)
TYPES = "(" + ", ".join(["vector<1xf32>"] * 8 + ["i64", "i64"] + ["i32"] * 5) + ")"
SHAPES = [(2, 6), (3, 4), (1, 12)]


def module_text(batches: int, products: int, fenced: bool = False) -> str:
    """A Mosaic-shaped module: one warpgroup's `gpu.launch` of a loop over 4 steps carrying `batches` [64, 16]
    state tiles; where `fenced`, a fence of shared memory to the async proxy before each product, with no
    product's value written there."""
    out, count = [], [0]

    def name(prefix):
        count[0] += 1
        return f"%{prefix}{count[0]}"

    words = 8 * batches
    out += [
        "module {",
        "  func.func @syn() {",
        "    %c1 = arith.constant 1 : index",
        "    %c4 = arith.constant 4 : index",
        "    %c0 = arith.constant 0 : index",
        "    %c128 = arith.constant 128 : index",
        "    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) "
        "threads(%tx, %ty, %tz) in (%nx = %c128, %ny = %c1, %nz = %c1) {",
        "      %zero = arith.constant dense<0.000000e+00> : vector<1xf32>",
        "      %decay = arith.constant dense<5.000000e-01> : vector<1xf32>",
        "      %one = arith.constant 1 : i32",
        "      %da = arith.constant 4096 : i64",
        "      %db = arith.constant 49152 : i64",
    ]
    types = ", ".join(["vector<1xf32>"] * words)
    iters = ", ".join(f"%s{k} = %zero" for k in range(words))
    out.append(f"      %final:{words} = scf.for %i = %c0 to %c4 step %c1 iter_args({iters}) -> ({types}) {{")
    updated = []
    for batch in range(batches):
        total = None
        for _ in range(products):
            acc = ["%zero"] * 8
            if fenced:
                out.append('        llvm.inline_asm has_side_effects asm_dialect = att '
                           '"fence.proxy.async.shared::cta;\\0A", "" : () -> ()')
            out.append(
                '        llvm.inline_asm has_side_effects asm_dialect = att '
                '"wgmma.fence.sync.aligned;\\0A", "" : () -> ()'
            )
            for _ in range(4):
                result = name("r")
                out.append(
                    f"        {result} = llvm.inline_asm has_side_effects asm_dialect = att {ASM} "
                    f"{', '.join(acc)}, %da, %db, %one, %one, %one, %one, %one : {TYPES} -> {STRUCT}"
                )
                acc = []
                for k in range(8):
                    element = name("e")
                    out.append(f"        {element} = llvm.extractvalue {result}[{k}] : {STRUCT}")
                    acc.append(element)
            out.append(
                "        llvm.inline_asm has_side_effects asm_dialect = att "
                '"wgmma.commit_group.sync.aligned;\\0A", "" : () -> ()'
            )
            out.append(
                "        llvm.inline_asm has_side_effects asm_dialect = att "
                '"wgmma.wait_group.sync.aligned 0;\\0A", "" : () -> ()'
            )
            if total is None:
                total = acc
            else:
                added = []
                for left, right in zip(total, acc, strict=True):
                    value = name("t")
                    out.append(f"        {value} = arith.addf {left}, {right} : vector<1xf32>")
                    added.append(value)
                total = added
        for k, summed in enumerate(total):  # the batch's sum folded into its own carried tile
            scaled, value = name("m"), name("u")
            out.append(f"        {scaled} = arith.mulf %s{8 * batch + k}, %decay : vector<1xf32>")
            out.append(f"        {value} = arith.addf {scaled}, {summed} : vector<1xf32>")
            updated.append(value)
    out.append(f"        scf.yield {', '.join(updated)} : {types}")
    out.append("      }")
    holders = ",".join(f"${k}" for k in range(words))
    out.append(
        f'      llvm.inline_asm has_side_effects asm_dialect = att "// {{{holders}}}\\0A", '
        f'"{",".join(["f"] * words)}" {", ".join(f"%final#{k}" for k in range(words))} : ({types}) -> ()'
    )
    out += ["      gpu.terminator", "    }", "    return", "  }", "}"]
    return "\n".join(out) + "\n"


def ptx_text(batches: int, products: int, fenced: bool = False) -> str:
    """The same program as ptxas gets it: each product's zero tile, a fence, four `wgmma.mma_async`, commit
    and wait, its words added into the batch's sum right after, the sum folded into the batch's carried tile
    of 8 words, a barrier between batches."""
    lines, floats, ints = [], [1], [10]
    states = []
    for batch in range(batches):
        tile = []
        for k in range(8):
            lines.append(f"\tld.global.f32 \t%f{floats[0]}, [%rd4+{4 * k + 32 * batch}];")
            tile.append(floats[0])
            floats[0] += 1
        states.append(tile)
    decay = floats[0]
    floats[0] += 1
    lines.append(f"\tld.global.f32 \t%f{decay}, [%rd4+1024];")

    def descriptor(n, offset):
        return [
            f"\tadd.s64 \t%rd{n}, %rd5, {offset};",
            f"\tshr.u64 \t%rd{n}, %rd{n}, 4;",
            f"\tand.b64 \t%rd{n}, %rd{n}, 16383;",
            f"\tor.b64 \t%rd{n}, %rd{n}, -4611685468670525440;",
        ]

    for batch in range(batches):
        if batch:
            lines.append("\tbar.sync \t1, 128;")
        total = None
        for product in range(products):
            tile = [ints[0] + k for k in range(8)]
            ints[0] += 8
            lines += [f"\tmov.b32 \t%r{r}, 0;" for r in tile]
            if fenced:
                lines.append("\tfence.proxy.async.shared::cta;")
            lines.append("\twgmma.fence.sync.aligned;")
            out = ",".join(f"%r{r}" for r in tile)
            for step in range(4):
                a, b = 10 + 2 * step, 11 + 2 * step
                lines += descriptor(a, 4096 * (product % 8) + 256 * step) + descriptor(
                    b, 49152 + 2048 * (product % 8) + 32 * step
                )
                lines.append(
                    "\t{ .reg .pred p; setp.ne.b32 p, 1, 0; "
                    "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                    f"{{{out}}}, %rd{a}, %rd{b}, p, 1, 1, 1, 1; }}"
                )
            lines += ["\twgmma.commit_group.sync.aligned;", "\twgmma.wait_group.sync.aligned 0;"]
            if total is None:
                total = []
                for r in tile:
                    lines.append(f"\tmov.b32 \t%f{floats[0]}, %r{r};")
                    total.append(floats[0])
                    floats[0] += 1
            else:
                for t, r in zip(total, tile, strict=True):
                    lines += [
                        f"\tmov.b32 \t%f{floats[0]}, %r{r};",
                        f"\tadd.f32 \t%f{t}, %f{t}, %f{floats[0]};",
                    ]
                    floats[0] += 1
        for k in range(8):
            lines.append(
                f"\tfma.rn.f32 \t%f{states[batch][k]}, %f{states[batch][k]}, %f{decay}, %f{total[k]};"
            )
    for batch in range(batches):
        for k in range(8):
            lines.append(f"\tst.global.f32 \t[%rd2+{4 * k + 32 * batch}], %f{states[batch][k]};")
    header = (
        ".version 8.8\n.target sm_90a\n.address_size 64\n\n.visible .entry syn(\n\t.param .u64 syn_param_0,\n"
        "\t.param .u64 syn_param_1\n)\n.maxntid 128, 1, 1\n.minnctapersm 1\n{\n"
        f"\t.reg .pred \t%p<8>;\n\t.reg .b32 \t%r<{ints[0] + 8}>;\n\t.reg .f32 \t%f<{floats[0] + 8}>;\n"
        "\t.reg .b64 \t%rd<64>;\n\t.shared .align 128 .b8 smem[98304];\n"
        "\tld.param.u64 \t%rd1, [syn_param_0];\n\tcvta.to.global.u64 \t%rd2, %rd1;\n"
        "\tld.param.u64 \t%rd3, [syn_param_1];\n\tcvta.to.global.u64 \t%rd4, %rd3;\n"
        "\tmov.u32 \t%r1, %tid.x;\n\tmov.u32 \t%r2, smem;\n\tcvt.u64.u32 \t%rd5, %r2;\n"
    )
    return header + "\n".join(lines) + "\n\tret;\n}\n"


def _peak(batches: int, products: int, fenced: bool = False) -> int:
    with jax_mlir.make_ir_context() as context:
        kernel_liveness.mosaic_gpu_dialect.register_dialect(context)
        context.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(module_text(batches, products, fenced=fenced))
        kernel_liveness.passmanager.PassManager.parse(kernel_liveness.CANONICAL).run(module.operation)
        divergent = kernel_liveness.divergent_values(module)
        totals = []
        for func in module.body.operations:
            if not func.operation.regions:
                continue
            peak, run = kernel_liveness.Peak(), kernel_liveness._Pass(divergent, kernel_compile._compiler())
            for region in func.operation.regions:
                for block in region.blocks:
                    run.block(block, set(), peak, 0)
            totals += [launch.total for launch in (run.launches or [peak])]
        return max(totals)


@pytest.mark.parametrize("batches, products", SHAPES)
def test_ptxas_allocates_every_independent_products_accumulators_together(batches, products):
    """ptxas's registers a thread for the products compiled as the gate does: all of them, a barrier between
    the batches or not, the sums that follow each wait in the program order read below the last product
    issued; an in-place chain of the same products (one accumulator) takes a fraction."""
    used, spilled, took = kernel_compile._ptxas(ptx_text(batches, products), "sm90", cap=None)
    assert took and not spilled
    assert used >= 8 * (batches * products + batches), (used, "registers for", batches * products)


@pytest.mark.parametrize("batches, products", SHAPES)
def test_the_pass_counts_the_registers_ptxas_allocates_for_independent_products(batches, products):
    """The pass's peak for the loop counts every product's accumulators and each batch's carried tile, as
    ptxas allocates them, and no more than ptxas allocates: it holds every product's accumulators of the
    loop's body to its last product (`kernel_liveness._product`), where the program order would kill a
    batch's at its fold into its carried tile and count one batch's."""
    used, _, _ = kernel_compile._ptxas(ptx_text(batches, products), "sm90", cap=None)
    assert 8 * (batches * products + batches) <= _peak(batches, products) <= used


def test_a_fence_no_products_value_crossed_keeps_the_products_together():
    """The independent products with a fence of shared memory to the async proxy before each, and no product's
    value written to shared memory: ptxas still allocates every product's accumulators, and the pass, which
    closes a run at a fence only where a run's product was stored before it (`_define_all`), counts them
    all, no more than ptxas allocates."""
    used, spilled, took = kernel_compile._ptxas(ptx_text(2, 6, fenced=True), "sm90", cap=None)
    assert took and not spilled
    assert 8 * (2 * 6 + 2) <= _peak(2, 6, fenced=True) <= used


def test_the_pass_reads_no_more_than_ptxas_holds_an_attention_backward():
    """The H100's fused key and value backward of a causal attention 128 wide (the attention resource gate's
    case, tests/test_attention_mosaic.py), whose products read each other (S, then P, then the value and key
    products): the pass's registers a thread (its lowered model) no more than ptxas's least spill-free cap.
    Holding every product of its body to its last read 307 against ptxas's 247 (row 3154)."""
    import jax
    import jax.numpy as jnp
    import test_attention_mosaic as attention
    from fla_gdn_support import KernelCase

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import attention as template, attention_mosaic

    queries, width = 512, 128
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.bfloat16)
    residuals = attention._residuals(query.shape)

    def backward(query, key, value, dout, residuals):
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                          accurate=False)
        return grads.query, grads.key, grads.value

    accurate = False
    op, ba, config = attention._gated_backward("sm90", width, accurate, "none")
    with generation.assumed("sm90"):
        models = attention._backward_models(op, ba, config)
        shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)

        def case(models: dict):
            return KernelCase("backward", lambda: (backward, (query, key, key, query, shapes)), models)

        # The pass's count by launch (`lowered_models`) and each launch's least spill-free cap by ptxas
        # (`least`, `kernel_compile.register_need`).
        (record,) = kernel_compile.compile_mosaic_kernels([case(models)], "sm90", least=True)
    lowered = record.get("lowered_models", {})
    least = {kernel_compile.report.launch_name(kernel["kernel"], kernel["launch"]): kernel["least"]
             for kernel in record["kernels"]}
    checked = [(name, model["registers"], least.get(name)) for name, model in lowered.items()]
    assert checked and all(found is not None for _, _, found in checked), (checked, sorted(least))
    assert all(registers <= found for _, registers, found in checked), checked
