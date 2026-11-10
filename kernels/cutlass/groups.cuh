// The group bookkeeping both architectures' grouped GEMMs share (grouped_sm80.cu, grouped_sm90.cu): the groups'
// first rows scanned on the device, and the groups' arrays carved from one scratch allocation. Neither holds an
// instruction family's behaviour; each file keeps its own kernel, its prepare kernels and its stride adapters.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dew_cutlass {

// The groups' first rows into `starts` (shared memory, `groups + 1` of them, the last the groups' sum), a
// serial scan by one thread: a call's groups are few (a model's experts), and the prepare kernel runs once a
// call.
__device__ inline void scanned(const int32_t* group_sizes, int groups, int64_t* starts) {
  if (threadIdx.x == 0) {
    int64_t start = 0;
    for (int i = 0; i < groups; ++i) {
      starts[i] = start;
      start += group_sizes[i];
    }
    starts[groups] = start;
  }
  __syncthreads();
}

// The bytes `carved` takes for `count` of T: whole 128-byte lines.
template <class T>
constexpr std::size_t carved_bytes(int count) {
  return (sizeof(T) * count + 127) / 128 * 128;
}

// An array of `count` T at `cursor`, which moves past it: each array 128-byte aligned, as CUTLASS's TMA
// descriptors and vectorized loads want.
template <class T>
T* carved(char*& cursor, int count) {
  T* array = reinterpret_cast<T*>(cursor);
  cursor += carved_bytes<T>(count);
  return array;
}

}  // namespace dew_cutlass
