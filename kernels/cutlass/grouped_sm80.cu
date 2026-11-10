// Ampere's grouped GEMMs (sm80, every compute capability 8.x): CUTLASS 2's grouped kernel (CUTLASS's example
// 24), cp.async into a multistage ring and mma.sync, its tiles scheduled on the device from the problem sizes
// a prepare kernel writes from the group sizes (GroupScheduleMode::kDeviceOnly), so a call never reads the
// group sizes on the host.
#include "grouped.h"
#include "groups.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace dew_cutlass {
namespace {

using cutlass::gemm::GemmCoord;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

// One kernel: 16-bit operands of `LayoutA` and `LayoutB` (CUTLASS's: A is m x k, B is k x n), fp32 sums,
// row-major `ElementD` out, a 128 x 128 x 32 tile of four warps of 64 x 64 each at four stages (64 KiB of
// shared memory, within every 8.x part's 99 KiB): cuBLAS's own Ampere bf16 tile
// (ampere_s16816gemm_bf16_128x128_ldg8, FS-DW-NCU2-a100).
template <class ElementAB, class ElementD, class LayoutA, class LayoutB>
struct Kernel {
  static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int kStages = 4;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementAB, LayoutA, cutlass::ComplexTransform::kNone, kAlignAB, ElementAB, LayoutB,
      cutlass::ComplexTransform::kNone, kAlignAB, ElementD, RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<ElementD, kAlignD, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kStages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using A = typename GemmKernel::ElementA;
  using B = typename GemmKernel::ElementB;
  using D = typename GemmKernel::ElementC;
  using Index = int64_t;  // the layouts' LongIndex

  // Each group's shape, operands and leading dimensions, written by `prepare`.
  struct Arrays {
    GemmCoord* shapes;
    A** a;
    B** b;
    D** d;
    Index* lda;
    Index* ldb;
    Index* ldd;

    auto fields() { return std::tie(shapes, a, b, d, lda, ldb, ldd); }
  };
};

// The forward's (and dx's) groups: group i is rows starts[i] to starts[i + 1] of x and out, times w[i]
// (k x n row-major, or under `transpose_w` its [n, k] transpose, column-major with leading dimension k).
template <class K>
__global__ void prepare_grouped(typename K::Arrays arrays, const typename K::A* x, const typename K::B* w,
                                typename K::D* out, const int32_t* group_sizes, int groups, int k, int n,
                                bool transpose_w) {
  extern __shared__ int64_t starts[];
  scanned(group_sizes, groups, starts);
  for (int i = threadIdx.x; i < groups; i += blockDim.x) {
    const Group group = forward_group(i, group_sizes, starts, k, n);
    arrays.shapes[i] = GemmCoord(group.m, group.n, group.k);
    arrays.a[i] = const_cast<typename K::A*>(x + group.a);
    arrays.b[i] = const_cast<typename K::B*>(w + group.b);
    arrays.d[i] = out + group.d;
    arrays.lda[i] = k;
    arrays.ldb[i] = transpose_w ? k : n;
    arrays.ldd[i] = n;
  }
}

// The weight gradient's groups: group i is x's rows starts[i] to starts[i + 1], transposed (k x rows, A
// column-major with x's row stride k), times dy's (rows x n), into out[i] (k x n). An empty group has no tiles
// (no rows of A): zero_empty_groups writes its block.
template <class K>
__global__ void prepare_grouped_k(typename K::Arrays arrays, const typename K::A* x, const typename K::B* dy,
                                  typename K::D* out, const int32_t* group_sizes, int groups, int k, int n,
                                  bool) {
  extern __shared__ int64_t starts[];
  scanned(group_sizes, groups, starts);
  for (int i = threadIdx.x; i < groups; i += blockDim.x) {
    const Group group = gradient_group(i, group_sizes, starts, k, n);
    arrays.shapes[i] = GemmCoord(group.m, group.n, group.k);
    arrays.a[i] = const_cast<typename K::A*>(x + group.a);
    arrays.b[i] = const_cast<typename K::B*>(dy + group.b);
    arrays.d[i] = out + group.d;
    arrays.lda[i] = k;
    arrays.ldb[i] = n;
    arrays.ldd[i] = n;
  }
}

// One call of kernel K: the group arrays carved from scratch, written by `prepare` on the stream, then the
// GEMM over as many programs as the device's multiprocessors hold at once, each walking the groups' tiles in
// turn (CUTLASS's device-side problem visitor).
template <class K, class Prepare>
const char* run(Prepare prepare, const void* lhs, const void* rhs, void* out, const int32_t* group_sizes,
                int groups, int k, int n, bool transpose_w, cudaStream_t stream, const Allocate& allocate,
                int multiprocessors) {
  char* cursor = static_cast<char*>(allocate(arrays_bytes<typename K::Arrays>(groups)));
  if (cursor == nullptr) return "the scratch allocator has no room for the groups' arrays";
  const auto arrays = carved_arrays<typename K::Arrays>(cursor, groups);
  prepare<<<1, 256, (groups + 1) * sizeof(int64_t), stream>>>(
      arrays, static_cast<const typename K::A*>(lhs), static_cast<const typename K::B*>(rhs),
      static_cast<typename K::D*>(out), group_sizes, groups, k, n, transpose_w);
  if (cudaGetLastError() != cudaSuccess) return "the prepare kernel did not launch";

  // Its shared memory past 48 KiB is granted here (`maximum_active_blocks` sets the kernel's attribute).
  const int resident = K::Gemm::maximum_active_blocks();
  if (resident <= 0) return "the kernel fits no multiprocessor";
  // beta is 0, so the epilogue reads no C; D stands in for it.
  typename K::Gemm::Arguments arguments(arrays.shapes, groups, resident * multiprocessors, {1.0f, 0.0f},
                                        arrays.a, arrays.b, arrays.d, arrays.d, arrays.lda, arrays.ldb,
                                        arrays.ldd, arrays.ldd);
  typename K::Gemm gemm;
  if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
    return "CUTLASS could not initialize the kernel";
  }
  if (gemm.run(stream) != cutlass::Status::kSuccess) return "the CUTLASS kernel did not launch";
  return nullptr;
}

template <class ElementAB, class ElementD>
const char* grouped_typed(const Grouped& call, cudaStream_t stream, const Allocate& allocate,
                          int multiprocessors) {
  const int k = static_cast<int>(call.k), n = static_cast<int>(call.n);
  // w[i] is CUTLASS's B, k x n: row-major as stored, column-major as its transpose [n, k] is.
  if (call.transpose_w) {
    using K = Kernel<ElementAB, ElementD, RowMajor, ColumnMajor>;
    return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, true,
                  stream, allocate, multiprocessors);
  }
  using K = Kernel<ElementAB, ElementD, RowMajor, RowMajor>;
  return run<K>(prepare_grouped<K>, call.x, call.w, call.out, call.group_sizes, call.groups, k, n, false,
                stream, allocate, multiprocessors);
}

template <class ElementAB, class ElementD>
const char* grouped_k_typed(const GroupedK& call, cudaStream_t stream, const Allocate& allocate,
                            int multiprocessors) {
  // CUTLASS's (m, n, k) is (k, n, rows): A is x's rows transposed, column-major with x's row stride k.
  using K = Kernel<ElementAB, ElementD, ColumnMajor, RowMajor>;
  return run<K>(prepare_grouped_k<K>, call.x, call.dy, call.out, call.group_sizes, call.groups,
                static_cast<int>(call.k), static_cast<int>(call.n), false, stream, allocate, multiprocessors);
}

// Its instantiation's geometry (`Instantiated`): the threadblock's MMA's tile and stages, the warps of its
// threadblock, one block a cluster.
template <class K>
Instantiated instantiated() {
  using Mma = typename K::GemmKernel::Mma;
  return {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK, Mma::kStages, K::GemmKernel::kThreadCount, 1,
          "mma.sync"};
}

}  // namespace

Instantiated sm80_kernel(Dtype result) {
  // Every layout of a result's kernel instantiates the same tile, stages and warps.
  using BF16 = cutlass::bfloat16_t;
  return result == Dtype::kF32 ? instantiated<Kernel<BF16, float, RowMajor, RowMajor>>()
                               : instantiated<Kernel<BF16, BF16, RowMajor, RowMajor>>();
}

const char* sm80_grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                         int multiprocessors) {
  (void)device;
  if (call.operands != Dtype::kBF16) return "sm80 takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_typed<cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate, multiprocessors);
    case Dtype::kF32:
      return grouped_typed<cutlass::bfloat16_t, float>(call, stream, allocate, multiprocessors);
    default:
      return "sm80 writes bf16 or f32";
  }
}

const char* sm80_grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                           int multiprocessors) {
  (void)device;
  if (call.operands != Dtype::kBF16) return "sm80 takes bf16 operands";
  switch (call.result) {
    case Dtype::kBF16:
      return grouped_k_typed<cutlass::bfloat16_t, cutlass::bfloat16_t>(call, stream, allocate,
                                                                       multiprocessors);
    case Dtype::kF32:
      return grouped_k_typed<cutlass::bfloat16_t, float>(call, stream, allocate, multiprocessors);
    default:
      return "sm80 writes bf16 or f32";
  }
}

}  // namespace dew_cutlass
