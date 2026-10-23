// CUTLASS's grouped GEMMs as exported XLA FFI handlers, which dew_cutlass registers through ctypes, so the
// library carries no Python ABI. Each handler dispatches on the compute capability of the device its stream
// runs on, to the kernel built for it (CMakeLists.txt's architectures), and answers an error naming any
// other. Problem shapes are set up on the device from the group sizes, in scratch from XLA's allocator, so a
// call never synchronizes with the host and stays jit- and graph-safe.
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// The CUDA architectures the library holds code for, CMake's list joined by commas ("80-real,90a-real,...").
extern "C" const char* dew_cutlass_architectures() { return DEW_CUTLASS_ARCHITECTURES; }

// x [M, K] and w [G, K, N] (or [G, N, K] under transpose_w), bf16 or f16, the rows of x grouped by
// group_sizes (s32 [G], counts); out [M, N] (bf16, f16 or f32), group g's rows of x times w[g], rows past
// the groups' sum zero. The forward pass, and its dx under transpose_w.
static ffi::Error GroupedGemm(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x,
                              ffi::AnyBuffer w, ffi::Buffer<ffi::S32> group_sizes,
                              ffi::Result<ffi::AnyBuffer> out, bool transpose_w) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented, "dew_cutlass_grouped_gemm: no kernel is built yet");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dew_cutlass_grouped_gemm, GroupedGemm,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<bool>("transpose_w"));

// x [M, K] and dy [M, N], bf16 or f16, grouped along M by group_sizes (s32 [G], counts); out [G, K, N]
// (f32 or bf16), group g's rows of x, transposed, times its rows of dy; a group of no rows zero. The weight
// gradient.
static ffi::Error GroupedGemmK(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x,
                               ffi::AnyBuffer dy, ffi::Buffer<ffi::S32> group_sizes,
                               ffi::Result<ffi::AnyBuffer> out) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented, "dew_cutlass_grouped_gemm_k: no kernel is built yet");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dew_cutlass_grouped_gemm_k, GroupedGemmK,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>());
