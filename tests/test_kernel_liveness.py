"""Register liveness across the control flow a Mosaic kernel emits. These small programs isolate the
scope and alias rules independently of ptxas's scheduling, before the resource gate compares kernels."""

import io

import pytest
from generation_support import COMPILES_ON_CPU
from jax._src.interpreters import mlir as jax_mlir
from jax.extend.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mosaic.python import mosaic_gpu as mosaic_gpu_dialect
from kernel_liveness import Compiler, divergent_values, kernel_peaks, lowered_registers


def serialized(body: str):
    """Serialize through Mosaic's actual versioned format, then read it through the liveness pass."""
    with jax_mlir.make_ir_context() as context:
        mosaic_gpu_dialect.register_dialect(context)
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(body)
        passmanager.PassManager.parse("builtin.module(mosaic_gpu-serde{serialize=true})").run(module.operation)
        bytecode = io.BytesIO()
        module.operation.write_bytecode(bytecode)
    return bytecode.getvalue()


def analyzed(body: str, compiler=None):
    return {name: peak for name, peak, _ in kernel_peaks(serialized(body), compiler)}["kernel"]


@pytest.mark.parametrize("width", [1, 2, 4])
def test_a_bitcast_reuses_the_words_it_reads(width):
    peak = analyzed(f"""
      module {{
        func.func @kernel(%x: vector<{2 * width}xbf16>) -> vector<{width}xi32> {{
          %y = vector.bitcast %x : vector<{2 * width}xbf16> to vector<{width}xi32>
          return %y : vector<{width}xi32>
        }}
      }}
    """)
    assert peak.total == width


def test_a_loop_carry_consumes_its_initial_value():
    peak = analyzed("""
      module {
        func.func private @effect(f32) -> f32
        func.func @kernel(%seed: f32) -> f32 {
          %lo = arith.constant 0 : index
          %hi = arith.constant 4 : index
          %step = arith.constant 1 : index
          %out = scf.for %i = %lo to %hi step %step iter_args(%carry = %seed) -> f32 {
            %next = func.call @effect(%carry) : (f32) -> f32
            scf.yield %next : f32
          }
          return %out : f32
        }
      }
    """)
    # A call may need both its input and output; the loop's initial value is neither a second carry
    # nor a value that remains live through every iteration.
    assert peak.total == 2


def test_a_void_barrier_assembly_does_not_allocate_data_registers():
    peak = analyzed("""
      module {
        func.func @kernel(%x: f32) -> f32 {
          "llvm.inline_asm"() {asm_string = "bar.sync 0;", constraints = "", has_side_effects} : () -> ()
          return %x : f32
        }
      }
    """)
    assert peak.total == 1


def test_an_assemblys_tied_result_reuses_its_input_with_a_memory_clobber():
    peak = analyzed("""
      module {
        func.func @kernel(%x: f32) -> f32 {
          %y = "llvm.inline_asm"(%x) {asm_string = "mov.f32 $0, $1;", constraints = "=f,0,~{memory}"}
            : (f32) -> f32
          return %y : f32
        }
      }
    """)
    assert peak.total == 1


def test_a_branch_does_not_hold_the_other_branchs_captures():
    peak = analyzed("""
      module {
        func.func private @effect(f32) -> f32
        func.func @kernel(%first: f32, %second: f32, %choose: i1) -> f32 {
          %out = scf.if %choose -> f32 {
            %x = func.call @effect(%first) : (f32) -> f32
            %y = func.call @effect(%x) : (f32) -> f32
            scf.yield %y : f32
          } else {
            %x = func.call @effect(%second) : (f32) -> f32
            %y = func.call @effect(%x) : (f32) -> f32
            scf.yield %y : f32
          }
          return %out : f32
        }
      }
    """)
    assert peak.total == 2


def test_a_uniform_address_still_occupies_the_loads_general_registers():
    peak = analyzed("""
      module {
        func.func @kernel(%address: !llvm.ptr) -> f32 {
          %value = "llvm.inline_asm"(%address) {asm_string = "ld.global.f32 $0, [$1];", constraints = "=f,l"}
            : (!llvm.ptr) -> f32
          return %value : f32
        }
      }
    """)
    assert peak.total >= 2


def test_packed_and_original_values_take_distinct_live_registers():
    peak = analyzed("""
      module {
        func.func @kernel(%x: vector<2xf32>) {
          %narrow = arith.truncf %x : vector<2xf32> to vector<2xbf16>
          %bits = vector.bitcast %narrow : vector<2xbf16> to vector<1xi32>
          %packed = vector.extract %bits[0] : i32 from vector<1xi32>
          %first = vector.extract %x[0] : f32 from vector<2xf32>
          %second = vector.extract %x[1] : f32 from vector<2xf32>
          llvm.inline_asm has_side_effects "", "r,f,f" %packed, %first, %second : (i32, f32, f32) -> ()
          return
        }
      }
    """)
    assert peak.total == 3


@pytest.mark.parametrize('variable_trip', [False, True])
def test_a_loop_propagates_its_carries_separately_and_keeps_trip_count_dependencies(variable_trip):
    upper = '%tid' if variable_trip else '%many'
    with jax_mlir.make_ir_context():
        module = ir.Module.parse(f"""
          module {{
            func.func @kernel() -> (f32, index) {{
              %zero = arith.constant 0 : index
              %one = arith.constant 1 : index
              %many = arith.constant 16 : index
              %tid = gpu.thread_id x
              %integer = arith.index_cast %tid : index to i32
              %seed = arith.sitofp %integer : i32 to f32
              %out:2 = scf.for %i = %zero to {upper} step %one
                  iter_args(%value = %seed, %counter = %zero) -> (f32, index) {{
                %next = arith.addf %value, %value : f32
                %count = arith.addi %counter, %one : index
                scf.yield %next, %count : f32, index
              }}
              return %out#0, %out#1 : f32, index
            }}
          }}
        """)
        varying = divergent_values(module)
        function = module.body.operations[0]
        loop = next(op for op in function.regions[0].blocks[0].operations if op.operation.name == 'scf.for')
        _, value, counter = loop.regions[0].blocks[0].arguments
        assert value in varying
        assert (counter in varying) == variable_trip
        assert (loop.results[1] in varying) == variable_trip


@pytest.mark.parametrize('mask', [-1, 65535])
def test_only_a_full_warp_broadcast_makes_a_varying_value_uniform(mask):
    with jax_mlir.make_ir_context():
        module = ir.Module.parse(f"""
          module {{
            func.func @kernel() -> i32 {{
              %mask = arith.constant {mask} : i32
              %lane = arith.constant 0 : i32
              %clamp = arith.constant 31 : i32
              %tid = gpu.thread_id x
              %source = arith.index_cast %tid : index to i32
              %shared = nvvm.shfl.sync idx %mask, %source, %lane, %clamp : i32 -> i32
              return %shared : i32
            }}
          }}
        """)
        varying = divergent_values(module)
        function = module.body.operations[0]
        returned = list(function.regions[0].blocks[0].operations)[-1].operation.operands[0]
        assert (returned in varying) == (mask != -1)


def test_the_compiler_input_accounts_for_a_transcendental_chain_kept_as_a_vector():
    body = """
      module {
        func.func @kernel(%x: vector<4xf32>) -> f32 {
          %exp = math.exp %x : vector<4xf32>
          %sum = vector.reduction <add>, %exp : vector<4xf32> into f32
          %old = vector.reduction <add>, %x : vector<4xf32> into f32
          %result = arith.addf %sum, %old : f32
          return %result : f32
        }
      }
    """
    vectorized = analyzed(body, Compiler((12, 9)))
    streamed = analyzed(body, Compiler((13, 4)))
    assert vectorized.total > streamed.total


def test_each_lowered_launch_has_its_own_budget_and_residency():
    from dew.nn.kernels.generation import gpu_target
    from dew.nn.kernels.templates import bundles, resources

    body = """
      module {
        func.func private @effect(f32) -> f32
        func.func @kernel(%x: f32) {
          %one = arith.constant 1 : index
          %small = arith.constant 128 : index
          %large = arith.constant 384 : index
          gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
            threads(%tx, %ty, %tz) in (%sx = %small, %sy = %one, %sz = %one) {
              %v = func.call @effect(%x) : (f32) -> f32
              gpu.terminator
            }
          gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
            threads(%tx, %ty, %tz) in (%sx = %large, %sy = %one, %sz = %one) {
              %v = func.call @effect(%x) : (f32) -> f32
              gpu.terminator
            }
          return
        }
      }
    """
    compiler = Compiler((13, 4))
    small, large = lowered_registers(serialized(body), compiler)
    target = gpu_target("sm90")
    fed = resources.register_model(bundles.named("wgmma.fed", "bfloat16"), 1, None)
    producer = resources.register_model(bundles.named("wgmma", "bfloat16"), 2, None)
    first = small.model(fed, target, shared=0, tmem=0)
    second = large.model(producer, target, shared=0, tmem=0)
    assert first.budget > second.budget
    assert first.most > second.most
    assert small.model(fed, target, shared=target.shared_limit, tmem=0).most == 1


def test_residency_charges_whole_register_allocation_units():
    from dew.nn.kernels.templates import resources

    # Two 384-thread blocks at 85 live registers would fit without allocation rounding. At the
    # hardware's 88-register allocation they need 67,584 registers, beyond the 64K file.
    model = resources.RegisterModel(85, 168, 3, 2, 8, 0)
    assert model.resident(85) == 1


def test_a_launch_past_an_allocation_limit_has_no_residency():
    from dew.nn.kernels.generation import gpu_target
    from dew.nn.kernels.templates import bundles, resources

    target = gpu_target("sm100")
    family = bundles.named("mma.sync", "bfloat16")
    assert resources.register_model(family, 1, 128, shared=target.shared_limit + 1, target=target).most == 0
    assert resources.register_model(family, 1, 128, target=target, tmem=target.tmem_columns + 1).most == 0


def _masked_loop(x, *, masked: bool, rows: int = 64, steps: int = 4, columns: int = 224):
    """Each row's maximum of `x` `[rows, steps * columns]` over a loop of `steps` tiles `columns` wide (not a
    power of two, so its columns' positions take a multiply-high remainder), one fp32 tile carried beside it,
    each tile masked by position (a column past its row reads -inf) where `masked`."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    layout = plgpu.Layout.MMA_ACC(jnp.bfloat16, m_warps=4)

    def body(x_ref, o_ref):
        def step(j, state):
            maximum, carried = state
            tile = plgpu.load(x_ref.at[:, pl.ds(j * columns, columns)], layout=layout, optimized=False)
            if masked:
                column = j * columns + lax.broadcasted_iota(jnp.int32, (rows, columns), 1)
                row = lax.broadcasted_iota(jnp.int32, (rows, columns), 0) + 3 * columns
                tile = jnp.where(column <= row, tile, -jnp.inf)
            return jnp.maximum(maximum, tile.max(axis=1)), carried + tile

        maximum = plgpu.layout_cast(jnp.full((rows,), -jnp.inf, jnp.float32), layout.reduce(1))
        carried = plgpu.layout_cast(jnp.zeros((rows, columns), jnp.float32), layout)
        maximum, carried = lax.fori_loop(0, steps, step, (maximum, carried))
        plgpu.store(o_ref, maximum + carried.sum(axis=1), optimized=False)

    return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((rows,), jnp.float32), grid=(1,),
                        grid_names=("sm",), num_threads=1, thread_name="wg",
                        compiler_params=plgpu.CompilerParams(approx_math=True))(x)


@COMPILES_ON_CPU
def test_a_hoisted_index_chain_longer_than_ptxas_recomputes_is_held_across_its_loop():
    """A loop whose tile is masked by its columns' positions and the same loop unmasked, compiled for an A100
    as the compile gate does: ptxas spills the masked one and not the other, so the pass puts the masked one
    above the 255 registers a thread holds and the unmasked one under them. The positions are integer chains
    over the thread id that LLVM hoists out of the loop; longer than ptxas's three recomputed instructions
    (`kernel_liveness.RECOMPUTED`), they are kept, which the pass counted free."""
    import functools

    import jax
    import jax.numpy as jnp
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    from dew.nn.kernels import generation

    spec = jax.ShapeDtypeStruct((64, 4 * 224), jnp.float32)
    found = {}
    for masked in (False, True):
        with generation.assumed("sm80"):
            record, = compile_mosaic_kernels(
                [KernelCase(f"masked-{masked}", lambda masked=masked: (
                    functools.partial(_masked_loop, masked=masked), (spec,)), {})], "sm80")
        kernel, = record["kernels"]
        peak, = (launch["registers"] for launches in record["lowered"].values() for launch in launches)
        found[masked] = (kernel["spilled"], peak)
    assert found[False][0] == 0 < found[True][0]
    assert found[False][1] <= 255 < found[True][1]


_TIED_SCALAR = """
  module {
    func.func @kernel(%x: f32) -> f32 {
      %zero = arith.constant 0.0 : f32
      %acc = "llvm.inline_asm"(%zero) {asm_string = "mov.f32 $0, $1;", constraints = "=f,0"} : (f32) -> f32
      %y = arith.addf %acc, %x : f32
      return %y : f32
    }
  }
"""

_TIED_STRUCT = """
  module {
    func.func @kernel(%x: f32) -> f32 {
      %zero = arith.constant 0.0 : f32
      %undefined = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
      %a0 = llvm.insertvalue %zero, %undefined[0] : !llvm.struct<(f32, f32, f32, f32)>
      %a1 = llvm.insertvalue %zero, %a0[1] : !llvm.struct<(f32, f32, f32, f32)>
      %a2 = llvm.insertvalue %zero, %a1[2] : !llvm.struct<(f32, f32, f32, f32)>
      %a3 = llvm.insertvalue %zero, %a2[3] : !llvm.struct<(f32, f32, f32, f32)>
      %c0 = llvm.extractvalue %a3[0] : !llvm.struct<(f32, f32, f32, f32)>
      %c1 = llvm.extractvalue %a3[1] : !llvm.struct<(f32, f32, f32, f32)>
      %c2 = llvm.extractvalue %a3[2] : !llvm.struct<(f32, f32, f32, f32)>
      %c3 = llvm.extractvalue %a3[3] : !llvm.struct<(f32, f32, f32, f32)>
      %acc = "llvm.inline_asm"(%c0, %c1, %c2, %c3) {asm_string = "mov.f32 $0, $4;",
          constraints = "=f,=f,=f,=f,0,1,2,3"} : (f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
      %r0 = llvm.extractvalue %acc[0] : !llvm.struct<(f32, f32, f32, f32)>
      %r1 = llvm.extractvalue %acc[1] : !llvm.struct<(f32, f32, f32, f32)>
      %r2 = llvm.extractvalue %acc[2] : !llvm.struct<(f32, f32, f32, f32)>
      %r3 = llvm.extractvalue %acc[3] : !llvm.struct<(f32, f32, f32, f32)>
      %s0 = arith.addf %r0, %x : f32
      %s1 = arith.addf %r1, %s0 : f32
      %s2 = arith.addf %r2, %s1 : f32
      %s3 = arith.addf %r3, %s2 : f32
      return %s3 : f32
    }
  }
"""


@pytest.mark.parametrize(("body", "words"), [(_TIED_SCALAR, 1), (_TIED_STRUCT, 4)], ids=["scalar", "struct"])
def test_an_accumulator_started_from_zeros_takes_its_registers(body, words):
    """An assembly whose outputs are tied to zeros (a warpgroup product into a fresh accumulator: `use_acc`
    false), read beside another live value: its result words and that value are live together. The pass gave
    the result its inputs' registers, which zeros have none of, so the sm90 dQ launch's two zero-started
    products, 128 words in flight, read as none while ptxas spilled them."""
    assert analyzed(body).total >= words + 1
