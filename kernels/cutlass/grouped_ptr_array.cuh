// CUTLASS 3's pointer-array grouped GEMMs (Hopper's sm90a, Blackwell's sm100a): what each architecture's file
// (grouped_sm90.cu, grouped_sm100.cu) shares but its kernel's types. Each group's problem shape, pointers and
// strides are written by a prepare kernel from the group sizes on the device (CUTLASS's examples 57 and 75,
// their shapes device-side alone), so a call never reads the group sizes on the host. Each file includes it
// in its own compilation, for its own architecture.
#pragma once

#include "grouped.h"
#include "groups.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace dew_cutlass {
namespace ptr_array {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // (m, n, k) a group
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

// An architecture's kernel (`Built`: its collective `Mainloop` and `Epilogue`, its `Tile` and `Cluster`) as
// the calls below launch it: its GEMM, its operands' element and stride types, and the groups' arrays.
template <class Built>
struct Kernel : Built {
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, typename Built::Mainloop, typename Built::Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  using A = typename GemmKernel::ElementA;
  using B = typename GemmKernel::ElementB;
  using D = typename GemmKernel::ElementD;

  // Each group's shape, operands and strides, written by `prepare`.
  struct Arrays {
    typename ProblemShape::UnderlyingProblemShape* shapes;
    const A** a;
    const B** b;
    D** d;
    StrideA* stride_a;
    StrideB* stride_b;
    StrideC* stride_c;
    StrideD* stride_d;

    auto fields() { return std::tie(shapes, a, b, d, stride_a, stride_b, stride_c, stride_d); }
  };
};

// The forward's (and dx's) groups: group i is rows starts[i] to starts[i + 1] of x and out, times w[i].
template <class K>
__global__ void prepare_grouped(typename K::Arrays arrays, const typename K::A* x, const typename K::B* w,
                                typename K::D* out, const int32_t* group_sizes, int groups, int k, int n,
                                typename K::StrideA stride_a, typename K::StrideB stride_b,
                                typename K::StrideC stride_c, typename K::StrideD stride_d) {
  extern __shared__ int64_t starts[];
  scanned(group_sizes, groups, starts);
  for (int i = threadIdx.x; i < groups; i += blockDim.x) {
    const Group group = forward_group(i, group_sizes, starts, k, n);
    arrays.shapes[i] = {static_cast<int>(group.m), static_cast<int>(group.n), static_cast<int>(group.k)};
    arrays.a[i] = x + group.a;
    arrays.b[i] = w + group.b;
    arrays.d[i] = out + group.d;
    arrays.stride_a[i] = stride_a;
    arrays.stride_b[i] = stride_b;
    arrays.stride_c[i] = stride_c;
    arrays.stride_d[i] = stride_d;
  }
}

// The weight gradient's groups: group i is x's rows starts[i] to starts[i + 1], transposed (k x rows, A
// column-major in x's row stride), times dy's (rows x n), into out[i] (k x n). An empty group has no tiles
// (no rows of A): zero_empty_groups writes its block.
template <class K>
__global__ void prepare_grouped_k(typename K::Arrays arrays, const typename K::A* x, const typename K::B* dy,
                                  typename K::D* out, const int32_t* group_sizes, int groups, int k, int n,
                                  typename K::StrideA stride_a, typename K::StrideB stride_b,
                                  typename K::StrideC stride_c, typename K::StrideD stride_d) {
  extern __shared__ int64_t starts[];
  scanned(group_sizes, groups, starts);
  for (int i = threadIdx.x; i < groups; i += blockDim.x) {
    const Group group = gradient_group(i, group_sizes, starts, k, n);
    arrays.shapes[i] = {static_cast<int>(group.m), static_cast<int>(group.n), static_cast<int>(group.k)};
    arrays.a[i] = x + group.a;
    arrays.b[i] = dy + group.b;
    arrays.d[i] = out + group.d;
    arrays.stride_a[i] = stride_a;
    arrays.stride_b[i] = stride_b;
    arrays.stride_c[i] = stride_c;
    arrays.stride_d[i] = stride_d;
  }
}

// One call of kernel K: the group arrays carved from scratch, written by `prepare` on the stream, then the
// GEMM. `extents` are CUTLASS's (m, n, k) of the packed strides, which no group's rows change.
template <class K, class Prepare>
const char* run(Prepare prepare, const void* lhs, const void* rhs, void* out, const int32_t* group_sizes,
                int groups, int k, int n, Shape<int, int, int> extents, cudaStream_t stream,
                const Allocate& allocate, int device, int multiprocessors) {
  // No group, no GEMM: CUTLASS's group scheduler reads the first group's shape at its construction, which none
  // holds. The handlers' zeroing writes a forward's rows; a weight gradient of no group has no block.
  if (groups == 0) return nullptr;
  const int m_extent = get<0>(extents), n_extent = get<1>(extents), k_extent = get<2>(extents);
  const auto stride_a = cutlass::make_cute_packed_stride(typename K::StrideA{}, make_shape(m_extent, k_extent, 1));
  const auto stride_b = cutlass::make_cute_packed_stride(typename K::StrideB{}, make_shape(n_extent, k_extent, 1));
  const auto stride_c = cutlass::make_cute_packed_stride(typename K::StrideC{}, make_shape(m_extent, n_extent, 1));
  const auto stride_d = cutlass::make_cute_packed_stride(typename K::StrideD{}, make_shape(m_extent, n_extent, 1));
  char* cursor = static_cast<char*>(allocate(arrays_bytes<typename K::Arrays>(groups)));
  if (cursor == nullptr) return "the scratch allocator has no room for the groups' arrays";
  const auto arrays = carved_arrays<typename K::Arrays>(cursor, groups);
  prepare<<<1, 256, (groups + 1) * sizeof(int64_t), stream>>>(
      arrays, static_cast<const typename K::A*>(lhs), static_cast<const typename K::B*>(rhs),
      static_cast<typename K::D*>(out), group_sizes, groups, k, n, stride_a, stride_b, stride_c, stride_d);
  if (cudaGetLastError() != cudaSuccess) return "the prepare kernel did not launch";

  cutlass::KernelHardwareInfo hardware{device, multiprocessors};
  typename K::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, arrays.shapes, nullptr},
      {arrays.a, arrays.stride_a, arrays.b, arrays.stride_b},
      {{}, nullptr, arrays.stride_c, arrays.d, arrays.stride_d},
      hardware};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  typename K::Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return "CUTLASS cannot implement the call (its extents' alignment)";
  }
  void* workspace = nullptr;
  if (const std::size_t size = K::Gemm::get_workspace_size(arguments); size > 0) {
    workspace = allocate(size);
    if (workspace == nullptr) return "the scratch allocator has no room for CUTLASS's workspace";
  }
  if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess) {
    return "CUTLASS could not initialize the kernel";
  }
  if (gemm.run(stream) != cutlass::Status::kSuccess) return "the CUTLASS kernel did not launch";
  return nullptr;
}

template <template <class, class, class, class> class Built, class ElementAB, class ElementD>
const char* grouped_typed(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                          int multiprocessors) {
  const int k = static_cast<int>(call.k), n = static_cast<int>(call.n);
  const Shape<int, int, int> extents{static_cast<int>(call.m), n, k};
  // w[i] is CUTLASS's B, k x n: row-major as stored, column-major as its transpose [n, k] is.
  if (call.transpose_w) {
    using K = Kernel<Built<ElementAB, ElementD, RowMajor, ColumnMajor>>;
    return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, extents,
                  stream, allocate, device, multiprocessors);
  }
  using K = Kernel<Built<ElementAB, ElementD, RowMajor, RowMajor>>;
  return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, extents,
                stream, allocate, device, multiprocessors);
}

template <template <class, class, class, class> class Built, class ElementAB, class ElementD>
const char* grouped_k_typed(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                            int multiprocessors) {
  // CUTLASS's (m, n, k) is (k, n, rows): A is x's rows transposed, column-major with x's row stride k.
  using K = Kernel<Built<ElementAB, ElementD, ColumnMajor, RowMajor>>;
  const int k = static_cast<int>(call.k), n = static_cast<int>(call.n);
  return run<K>(prepare_grouped_k<K>, call.x, call.dy, call.out, call.group_sizes, call.groups, k, n,
                Shape<int, int, int>{k, n, static_cast<int>(call.m)}, stream, allocate, device, multiprocessors);
}

// A call of an architecture's kernels (`Built`), its operands and result checked: bf16 operands, bf16 or f32
// out, as each architecture's file names itself (`name`).
template <template <class, class, class, class> class Built>
const char* grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                    int multiprocessors) {
  if (call.operands != Dtype::kBF16) return "the kernel takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_typed<Built, cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate, device,
                                                                            multiprocessors);
    case Dtype::kF32:
      return grouped_typed<Built, cutlass::bfloat16_t, float>(call, stream, allocate, device, multiprocessors);
    default:
      return "the kernel writes bf16 or f32";
  }
}

template <template <class, class, class, class> class Built>
const char* grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                      int multiprocessors) {
  if (call.operands != Dtype::kBF16) return "the kernel takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_k_typed<Built, cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate, device,
                                                                              multiprocessors);
    case Dtype::kF32:
      return grouped_k_typed<Built, cutlass::bfloat16_t, float>(call, stream, allocate, device,
                                                               multiprocessors);
    default:
      return "the kernel writes bf16 or f32";
  }
}

// A result's kernel (every layout of it instantiates one tile, its stages and threads; its result's epilogue
// storage sets its stages) as its types state it (`Instantiated`): its tile, the stages CUTLASS's builder
// fitted beside its epilogue's shared memory, `Built`'s consumer threads (`kConsumers`), its cluster's rows
// and its instruction family.
template <template <class, class, class, class> class Built>
Instantiated instantiated(Dtype result, const char* instructions) {
  using BF16 = cutlass::bfloat16_t;
  const auto of = [instructions](auto kernel) -> Instantiated {
    using K = decltype(kernel);
    using Mainloop = typename K::Mainloop;
    return {size<0>(typename K::Tile{}), size<1>(typename K::Tile{}), size<2>(typename K::Tile{}),
            Mainloop::DispatchPolicy::Stages, K::kConsumers,
            size<0>(typename K::Cluster{}), instructions};
  };
  return result == Dtype::kF32 ? of(Kernel<Built<BF16, float, RowMajor, RowMajor>>{})
                               : of(Kernel<Built<BF16, BF16, RowMajor, RowMajor>>{});
}

}  // namespace ptr_array
}  // namespace dew_cutlass
