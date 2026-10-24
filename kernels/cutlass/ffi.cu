// CUTLASS's grouped GEMMs as exported XLA FFI handlers, which dew_cutlass registers through ctypes, so the
// library carries no Python ABI. Each handler dispatches on the compute capability of the device its stream
// runs on, to the kernel built for it (grouped_sm90.cu, ...), and answers an error naming any other. Problem
// shapes are set up on the device from the group sizes, in scratch from XLA's allocator, so a call never
// synchronizes with the host and stays jit- and graph-safe.
#include <cuda_runtime.h>

#include <string>

#include "grouped.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// The CUDA architectures the library holds grouped kernels for, CMake's list joined by commas ("90a-real").
extern "C" const char* dew_cutlass_architectures() { return DEW_CUTLASS_ARCHITECTURES; }

namespace dew_cutlass {
namespace {

// Rows a block of zero_unrouted_rows checks against the groups' sum.
constexpr int kZeroRows = 16;

__global__ void zero_rows(char* out, int64_t m, int64_t row_bytes, const int32_t* group_sizes, int groups) {
  __shared__ int64_t routed;
  if (threadIdx.x == 0) {
    int64_t sum = 0;
    for (int i = 0; i < groups; ++i) sum += group_sizes[i];
    routed = sum;
  }
  __syncthreads();
  const int64_t first = static_cast<int64_t>(blockIdx.x) * kZeroRows;
  for (int64_t row = first; row < first + kZeroRows && row < m; ++row) {
    if (row < routed) continue;
    int4* words = reinterpret_cast<int4*>(out + row * row_bytes);
    for (int64_t word = threadIdx.x; word < row_bytes / 16; word += blockDim.x) words[word] = int4{0, 0, 0, 0};
  }
}

__global__ void zero_groups(char* out, int64_t block_bytes, const int32_t* group_sizes) {
  if (group_sizes[blockIdx.x] != 0) return;
  int4* words = reinterpret_cast<int4*>(out + static_cast<int64_t>(blockIdx.x) * block_bytes);
  for (int64_t word = threadIdx.x; word < block_bytes / 16; word += blockDim.x) words[word] = int4{0, 0, 0, 0};
}

std::optional<Dtype> dtype_of(ffi::DataType type) {
  switch (type) {
    case ffi::DataType::BF16:
      return Dtype::kBF16;
    case ffi::DataType::F16:
      return Dtype::kF16;
    case ffi::DataType::F32:
      return Dtype::kF32;
    default:
      return std::nullopt;
  }
}

// The stream's device, its compute capability (10 * major + minor) and multiprocessors.
struct Device {
  int index, capability, multiprocessors;
};

std::optional<Device> device_of(cudaStream_t stream) {
  int index = 0;
  if (cudaGetDevice(&index) != cudaSuccess) return std::nullopt;
  int major = 0, minor = 0, multiprocessors = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, index) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, index) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, index) != cudaSuccess) {
    return std::nullopt;
  }
  return Device{index, 10 * major + minor, multiprocessors};
}

// Scratch aligned to 128 bytes. XLA's FFI refuses an alignment past 16 (ffi_api.cc's kMaxAlignment), so
// each allocation is 112 bytes longer, its start rounded up.
Allocate allocator(ffi::ScratchAllocator& scratch) {
  return [&scratch](std::size_t bytes) -> void* {
    std::optional<void*> memory = scratch.Allocate(bytes + 112, 16);
    if (!memory.has_value()) return nullptr;
    const auto address = reinterpret_cast<std::uintptr_t>(*memory);
    return reinterpret_cast<void*>((address + 127) / 128 * 128);
  };
}

ffi::Error failed(const char* target, const std::string& why) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, std::string(target) + ": " + why);
}

}  // namespace

void zero_unrouted_rows(void* out, int64_t m, int64_t row_bytes, const int32_t* group_sizes, int groups,
                        cudaStream_t stream) {
  if (m == 0) return;
  const int blocks = static_cast<int>((m + kZeroRows - 1) / kZeroRows);
  zero_rows<<<blocks, 128, 0, stream>>>(static_cast<char*>(out), m, row_bytes, group_sizes, groups);
}

void zero_empty_groups(void* out, int64_t block_bytes, const int32_t* group_sizes, int groups,
                       cudaStream_t stream) {
  if (groups == 0) return;
  zero_groups<<<groups, 256, 0, stream>>>(static_cast<char*>(out), block_bytes, group_sizes);
}

}  // namespace dew_cutlass

namespace {

using dew_cutlass::Device;

// x [M, K] and w [G, K, N] (or [G, N, K] under transpose_w), bf16 or f16, the rows of x grouped by
// group_sizes (s32 [G], counts); out [M, N] (bf16, f16 or f32), group g's rows of x times w[g], rows past
// the groups' sum zero. The forward pass, and its dx under transpose_w.
ffi::Error GroupedGemm(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x, ffi::AnyBuffer w,
                       ffi::Buffer<ffi::S32> group_sizes, ffi::Result<ffi::AnyBuffer> out, bool transpose_w) {
  constexpr const char* kTarget = "dew_cutlass_grouped_gemm";
  const auto operands = dew_cutlass::dtype_of(x.element_type());
  const auto result = dew_cutlass::dtype_of(out->element_type());
  if (!operands || x.element_type() != w.element_type() || *operands == dew_cutlass::Dtype::kF32 || !result) {
    return dew_cutlass::failed(kTarget, "x and w are one 16-bit dtype and out bf16, f16 or f32");
  }
  const auto x_dims = x.dimensions(), w_dims = w.dimensions(), out_dims = out->dimensions();
  if (x_dims.size() != 2 || w_dims.size() != 3 || out_dims.size() != 2 ||
      group_sizes.dimensions().size() != 1 || group_sizes.dimensions()[0] != w_dims[0]) {
    return dew_cutlass::failed(kTarget, "x is [M, K], w [G, K, N] or [G, N, K], group_sizes [G], out [M, N]");
  }
  const int64_t m = x_dims[0], k = x_dims[1];
  const int64_t n = transpose_w ? w_dims[1] : w_dims[2];
  if ((transpose_w ? w_dims[2] : w_dims[1]) != k || out_dims[0] != m || out_dims[1] != n) {
    return dew_cutlass::failed(kTarget, "the extents of x, w and out do not chain");
  }
  const std::optional<Device> device = dew_cutlass::device_of(stream);
  if (!device) return dew_cutlass::failed(kTarget, "the device's attributes could not be read");
  const dew_cutlass::Grouped call{x.untyped_data(), w.untyped_data(), group_sizes.typed_data(),
                                  out->untyped_data(), m, k, n, static_cast<int>(w_dims[0]), transpose_w,
                                  *operands, *result};
  const char* why = "no kernel is built for this device's compute capability";
  if (device->capability / 10 == 9) {
    why = dew_cutlass::sm90_grouped(call, stream, dew_cutlass::allocator(scratch), device->index,
                                    device->multiprocessors);
  }
  if (why != nullptr) return dew_cutlass::failed(kTarget, why);
  dew_cutlass::zero_unrouted_rows(call.out, m, n * static_cast<int64_t>(ffi::ByteWidth(out->element_type())),
                                  call.group_sizes, call.groups, stream);
  return cudaGetLastError() == cudaSuccess ? ffi::Error::Success()
                                           : dew_cutlass::failed(kTarget, "a kernel did not launch");
}

// x [M, K] and dy [M, N], bf16 or f16, grouped along M by group_sizes (s32 [G], counts); out [G, K, N]
// (f32 or bf16), group g's rows of x, transposed, times its rows of dy; a group of no rows zero. The weight
// gradient.
ffi::Error GroupedGemmK(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x,
                        ffi::AnyBuffer dy, ffi::Buffer<ffi::S32> group_sizes, ffi::Result<ffi::AnyBuffer> out) {
  constexpr const char* kTarget = "dew_cutlass_grouped_gemm_k";
  const auto operands = dew_cutlass::dtype_of(x.element_type());
  const auto result = dew_cutlass::dtype_of(out->element_type());
  if (!operands || x.element_type() != dy.element_type() || *operands == dew_cutlass::Dtype::kF32 || !result) {
    return dew_cutlass::failed(kTarget, "x and dy are one 16-bit dtype and out bf16, f16 or f32");
  }
  const auto x_dims = x.dimensions(), dy_dims = dy.dimensions(), out_dims = out->dimensions();
  if (x_dims.size() != 2 || dy_dims.size() != 2 || out_dims.size() != 3 ||
      group_sizes.dimensions().size() != 1 || group_sizes.dimensions()[0] != out_dims[0]) {
    return dew_cutlass::failed(kTarget, "x is [M, K], dy [M, N], group_sizes [G], out [G, K, N]");
  }
  const int64_t m = x_dims[0], k = x_dims[1], n = dy_dims[1];
  if (dy_dims[0] != m || out_dims[1] != k || out_dims[2] != n) {
    return dew_cutlass::failed(kTarget, "the extents of x, dy and out do not chain");
  }
  const std::optional<Device> device = dew_cutlass::device_of(stream);
  if (!device) return dew_cutlass::failed(kTarget, "the device's attributes could not be read");
  const dew_cutlass::GroupedK call{x.untyped_data(), dy.untyped_data(), group_sizes.typed_data(),
                                   out->untyped_data(), m, k, n, static_cast<int>(out_dims[0]), *operands,
                                   *result};
  const char* why = "no kernel is built for this device's compute capability";
  if (device->capability / 10 == 9) {
    why = dew_cutlass::sm90_grouped_k(call, stream, dew_cutlass::allocator(scratch), device->index,
                                      device->multiprocessors);
  }
  if (why != nullptr) return dew_cutlass::failed(kTarget, why);
  dew_cutlass::zero_empty_groups(call.out, k * n * static_cast<int64_t>(ffi::ByteWidth(out->element_type())),
                                 call.group_sizes, call.groups, stream);
  return cudaGetLastError() == cudaSuccess ? ffi::Error::Success()
                                           : dew_cutlass::failed(kTarget, "a kernel did not launch");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(dew_cutlass_grouped_gemm, GroupedGemm,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<bool>("transpose_w"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(dew_cutlass_grouped_gemm_k, GroupedGemmK,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>());
