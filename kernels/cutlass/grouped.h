// The grouped GEMMs' calls as the FFI handlers (ffi.cu) hand them to each architecture's kernels (one source
// file an architecture, each compiled for its own): plain pointers and extents, no XLA type, so a kernel file
// needs only CUTLASS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include <cuda_runtime.h>

namespace dew_cutlass {

enum class Dtype { kBF16, kF16, kF32 };

// Device memory for one call, from XLA's scratch allocator; nullptr where it has none left.
using Allocate = std::function<void*(std::size_t bytes)>;

// x [m, k] and w [g, k, n] (or [g, n, k] under transpose_w), the rows of x grouped by group_sizes (int32
// counts on the device); out [m, n], group i's rows of x times w[i], the rows past the groups' sum zero.
struct Grouped {
  const void* x;
  const void* w;
  const int32_t* group_sizes;
  void* out;
  int64_t m, k, n;
  int groups;
  bool transpose_w;
  Dtype operands, result;
};

// x [m, k] and dy [m, n], grouped along m by group_sizes; out [g, k, n], group i's rows of x, transposed,
// times its rows of dy, a group of no rows zero.
struct GroupedK {
  const void* x;
  const void* dy;
  const int32_t* group_sizes;
  void* out;
  int64_t m, k, n;
  int groups;
  Dtype operands, result;
};

// Each runs the call on `stream` and returns nullptr, or names why it cannot (an unsupported dtype or
// alignment, scratch exhausted, a CUTLASS status).
const char* sm90_grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                         int multiprocessors);
const char* sm90_grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                           int multiprocessors);

// The rows of out [m, n] (row_bytes each) from the groups' sum to m, written zero on `stream`, and, for the
// weight gradient, each empty group's [k, n] block of out (block_bytes each).
void zero_unrouted_rows(void* out, int64_t m, int64_t row_bytes, const int32_t* group_sizes, int groups,
                        cudaStream_t stream);
void zero_empty_groups(void* out, int64_t block_bytes, const int32_t* group_sizes, int groups,
                       cudaStream_t stream);

}  // namespace dew_cutlass
