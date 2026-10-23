// Hopper's grouped GEMMs (sm90a): CUTLASS 3's pointer-array warp-specialized cooperative kernel, its TMA
// descriptors set per group on the device, its problem shapes, pointers and strides written by a prepare
// kernel from the group sizes on the device (CUTLASS's example 57, its shapes device-side alone), so a call
// never reads the group sizes on the host.
#include "grouped.h"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace dew_cutlass {
namespace {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // (m, n, k) a group

// One kernel: 16-bit operands of `LayoutA` and `LayoutB` (CUTLASS's: A is m x k, B is k x n), fp32 sums,
// row-major `ElementD` out, a 128 x 256 x 64 tile of a pair of multiprocessors (cooperative, as CUTLASS's
// grouped GEMMs on Hopper default to).
template <class ElementAB, class ElementD, class LayoutA, class LayoutB>
struct Kernel {
  using LayoutD = cutlass::layout::RowMajor;
  using Tile = Shape<_128, _256, _64>;
  using Cluster = Shape<_2, _1, _1>;
  static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, kAlignD, ElementD,
      LayoutD*, kAlignD, cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA*, kAlignAB, ElementAB, LayoutB*,
      kAlignAB, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
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
  };
};

// The groups' first rows, a serial scan in shared memory by one thread: a call's groups are few (a model's
// experts), and the prepare kernel runs once a call.
__device__ void scanned(const int32_t* group_sizes, int groups, int64_t* starts) {
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

// The forward's (and dx's) groups: group i is rows starts[i] to starts[i + 1] of x and out, times w[i].
template <class K>
__global__ void prepare_grouped(typename K::Arrays arrays, const typename K::A* x, const typename K::B* w,
                                typename K::D* out, const int32_t* group_sizes, int groups, int k, int n,
                                typename K::StrideA stride_a, typename K::StrideB stride_b,
                                typename K::StrideC stride_c, typename K::StrideD stride_d) {
  extern __shared__ int64_t starts[];
  scanned(group_sizes, groups, starts);
  for (int i = threadIdx.x; i < groups; i += blockDim.x) {
    const int rows = group_sizes[i];
    arrays.shapes[i] = {rows, n, k};
    arrays.a[i] = x + starts[i] * k;
    arrays.b[i] = w + static_cast<int64_t>(i) * k * n;
    arrays.d[i] = out + starts[i] * n;
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
    const int rows = group_sizes[i];
    arrays.shapes[i] = {rows > 0 ? k : 0, n, rows > 0 ? rows : 1};
    arrays.a[i] = x + starts[i] * k;
    arrays.b[i] = dy + starts[i] * n;
    arrays.d[i] = out + static_cast<int64_t>(i) * k * n;
    arrays.stride_a[i] = stride_a;
    arrays.stride_b[i] = stride_b;
    arrays.stride_c[i] = stride_c;
    arrays.stride_d[i] = stride_d;
  }
}

template <class T>
T* carved(char*& cursor, int count) {
  // Each array 128-byte aligned, as CUTLASS's TMA descriptors and vectorized loads want.
  T* array = reinterpret_cast<T*>(cursor);
  cursor += (sizeof(T) * count + 127) / 128 * 128;
  return array;
}

// One call of kernel K: the group arrays carved from scratch, written by `prepare` on the stream, then the
// GEMM. `extents` are CUTLASS's (m, n, k) of the packed strides, which no group's rows change.
template <class K, class Prepare>
const char* run(Prepare prepare, const void* lhs, const void* rhs, void* out, const int32_t* group_sizes,
                int groups, int k, int n, Shape<int, int, int> extents, cudaStream_t stream,
                const Allocate& allocate, int device, int multiprocessors) {
  const int m_extent = get<0>(extents), n_extent = get<1>(extents), k_extent = get<2>(extents);
  const auto stride_a = cutlass::make_cute_packed_stride(typename K::StrideA{}, make_shape(m_extent, k_extent, 1));
  const auto stride_b = cutlass::make_cute_packed_stride(typename K::StrideB{}, make_shape(n_extent, k_extent, 1));
  const auto stride_c = cutlass::make_cute_packed_stride(typename K::StrideC{}, make_shape(m_extent, n_extent, 1));
  const auto stride_d = cutlass::make_cute_packed_stride(typename K::StrideD{}, make_shape(m_extent, n_extent, 1));
  const std::size_t bytes =
      8 * 128 + groups * (sizeof(typename ProblemShape::UnderlyingProblemShape) + 3 * sizeof(void*) +
                          sizeof(typename K::StrideA) + sizeof(typename K::StrideB) +
                          sizeof(typename K::StrideC) + sizeof(typename K::StrideD));
  char* cursor = static_cast<char*>(allocate(bytes));
  if (cursor == nullptr) return "the scratch allocator has no room for the groups' arrays";
  typename K::Arrays arrays{carved<typename ProblemShape::UnderlyingProblemShape>(cursor, groups),
                            carved<const typename K::A*>(cursor, groups), carved<const typename K::B*>(cursor, groups),
                            carved<typename K::D*>(cursor, groups), carved<typename K::StrideA>(cursor, groups),
                            carved<typename K::StrideB>(cursor, groups), carved<typename K::StrideC>(cursor, groups),
                            carved<typename K::StrideD>(cursor, groups)};
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

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template <class ElementAB, class ElementD>
const char* grouped_typed(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                          int multiprocessors) {
  const int k = static_cast<int>(call.k), n = static_cast<int>(call.n);
  const Shape<int, int, int> extents{static_cast<int>(call.m), n, k};
  // w[i] is CUTLASS's B, k x n: row-major as stored, column-major as its transpose [n, k] is.
  if (call.transpose_w) {
    using K = Kernel<ElementAB, ElementD, RowMajor, ColumnMajor>;
    return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, extents,
                  stream, allocate, device, multiprocessors);
  }
  using K = Kernel<ElementAB, ElementD, RowMajor, RowMajor>;
  return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, extents,
                stream, allocate, device, multiprocessors);
}

template <class ElementAB, class ElementD>
const char* grouped_k_typed(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                            int multiprocessors) {
  // CUTLASS's (m, n, k) is (k, n, rows): A is x's rows transposed, column-major with x's row stride k.
  using K = Kernel<ElementAB, ElementD, ColumnMajor, RowMajor>;
  const int k = static_cast<int>(call.k), n = static_cast<int>(call.n);
  return run<K>(prepare_grouped_k<K>, call.x, call.dy, call.out, call.group_sizes, call.groups, k, n,
                Shape<int, int, int>{k, n, static_cast<int>(call.m)}, stream, allocate, device, multiprocessors);
}

}  // namespace

const char* sm90_grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                         int multiprocessors) {
  if (call.operands != Dtype::kBF16) return "sm90 takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_typed<cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate, device,
                                                                     multiprocessors);
    case Dtype::kF32:
      return grouped_typed<cutlass::bfloat16_t, float>(call, stream, allocate, device, multiprocessors);
    default:
      return "sm90 writes bf16 or f32";
  }
}

const char* sm90_grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                           int multiprocessors) {
  if (call.operands != Dtype::kBF16) return "sm90 takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_k_typed<cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate, device,
                                                                       multiprocessors);
    case Dtype::kF32:
      return grouped_k_typed<cutlass::bfloat16_t, float>(call, stream, allocate, device, multiprocessors);
    default:
      return "sm90 writes bf16 or f32";
  }
}

}  // namespace dew_cutlass
