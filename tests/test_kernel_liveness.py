"""Register liveness across the control flow a Mosaic kernel emits. These small programs isolate the
scope and alias rules independently of ptxas's scheduling, before the resource gate compares kernels."""

import io

import pytest
from jax._src.interpreters import mlir as jax_mlir
from jax.extend.mlir import ir
from jaxlib.mlir import passmanager
from jaxlib.mosaic.python import mosaic_gpu as mosaic_gpu_dialect

from dew.nn.kernels.templates.liveness import Compiler, kernel_peaks


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
    vectorized = analyzed(body, Compiler((12, 9), relocatable=True))
    streamed = analyzed(body, Compiler((13, 4), relocatable=True))
    assert vectorized.total > streamed.total


def test_each_lowered_launch_has_its_own_budget_and_residency():
    from dew.nn.kernels.generation import gpu_target
    from dew.nn.kernels.templates import resources

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
    compiler = Compiler((13, 4), relocatable=True)
    small, large = resources.lowered_registers(serialized(body), compiler)
    target = gpu_target("sm100")
    first = small.model(target, compiler, shared=0, tmem=0)
    second = large.model(target, compiler, shared=0, tmem=0)
    assert first.budget > second.budget
    assert first.most > second.most
    assert small.model(target, compiler, shared=target.shared_limit, tmem=0).most == 1
    assert small.model(target, compiler, shared=0, tmem=target.tmem_columns).most == 1
