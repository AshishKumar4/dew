// The group bookkeeping both architectures' grouped GEMMs share (grouped_sm80.cu, grouped_sm90.cu): the groups'
// first rows scanned on the device, each group's GEMM (its shape and operand offsets), and the groups' arrays
// laid out in one scratch allocation from their one field list. Neither holds an instruction family's
// behaviour; each file keeps its own kernel and its adapters of a group to its shapes and strides.
#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

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

// A grouped kernel's group arrays (each architecture's `Arrays`): a struct of array pointers whose
// `fields()` ties every member in order, its one field list, from which the scratch's bytes
// (`arrays_bytes`) and the arrays' places in it (`carved_arrays`) are both derived. Every member is an
// array pointer, so a member `fields()` leaves out is a compile error.
template <class Arrays>
constexpr std::size_t field_count() {
  return std::tuple_size_v<decltype(std::declval<Arrays&>().fields())>;
}

template <class Arrays>
std::size_t arrays_bytes(int count) {
  static_assert(sizeof(Arrays) == field_count<Arrays>() * sizeof(void*), "Arrays::fields() ties every member");
  Arrays probe{};
  return std::apply(
      [count](auto&... field) {
        return (carved_bytes<std::remove_pointer_t<std::remove_reference_t<decltype(field)>>>(count) + ...);
      },
      probe.fields());
}

template <class Arrays>
Arrays carved_arrays(char* cursor, int count) {
  Arrays arrays{};
  std::apply(
      [&cursor, count](auto&... field) {
        ((field = carved<std::remove_pointer_t<std::remove_reference_t<decltype(field)>>>(cursor, count)), ...);
      },
      arrays.fields());
  return arrays;
}

// A group's GEMM as both architectures' grouped kernels compute it, before each adapts it to its own shapes
// and strides: its (m, n, k) and its operands' and output's first element, by offset.
struct Group {
  int64_t m, n, k;
  int64_t a, b, d;
};

// The forward's (and dx's) group i: rows starts[i] to starts[i + 1] of x and out, times w[i] (k x n).
__device__ inline Group forward_group(int i, const int32_t* group_sizes, const int64_t* starts, int k, int n) {
  return {group_sizes[i], n, k, starts[i] * k, static_cast<int64_t>(i) * k * n, starts[i] * n};
}

// The weight gradient's group i: x's rows starts[i] to starts[i + 1], transposed (k x rows), times dy's
// (rows x n), into out[i] (k x n). An empty group has no tiles (no rows of A, its depth one so the
// problem stays well formed): zero_empty_groups writes its block.
__device__ inline Group gradient_group(int i, const int32_t* group_sizes, const int64_t* starts, int k, int n) {
  const int64_t rows = group_sizes[i];
  return {rows > 0 ? k : 0, n, rows > 0 ? rows : 1, starts[i] * k, starts[i] * n, static_cast<int64_t>(i) * k * n};
}

}  // namespace dew_cutlass
