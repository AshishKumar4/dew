// Blackwell's grouped GEMMs (sm100a): CUTLASS 3's pointer-array warp-specialized tcgen05 kernel on a pair of
// multiprocessors (its 2SM MMA, CUTLASS's example 75), its TMA descriptors set per group on the device
// (grouped_ptr_array.cuh, which sm90's shares).
#include "grouped_ptr_array.cuh"

namespace dew_cutlass {
namespace {

using namespace cute;

// One kernel: 16-bit operands of `LayoutA` and `LayoutB` (CUTLASS's: A is m x k, B is k x n), fp32 sums in
// tensor memory, row-major `ElementD` out, a 256 x 256 x 64 MMA tile issued across a pair of multiprocessors
// (tcgen05's 2SM MMA, example 75's `MMA2SMConfig`, its cluster static).
template <class ElementAB, class ElementD, class LayoutA, class LayoutB>
struct Built {
  using LayoutD = cutlass::layout::RowMajor;
  using Tile = Shape<_256, _256, _64>;
  using Cluster = Shape<_2, _1, _1>;
  static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, kAlignD, ElementD,
      LayoutD*, kAlignD, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA*, kAlignAB, ElementAB, LayoutB*,
      kAlignAB, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>::CollectiveOp;

  // One thread issues its MMAs into tensor memory; its consumers are the epilogue's threads, which read the
  // sums out of it.
  static constexpr int kConsumers = static_cast<int>(Epilogue::ThreadCount);
};

}  // namespace

Instantiated sm100_kernel(Dtype result) { return ptr_array::instantiated<Built>(result, "tcgen05"); }

const char* sm100_grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                          int multiprocessors) {
  return ptr_array::grouped<Built>(call, stream, allocate, device, multiprocessors);
}

const char* sm100_grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                            int multiprocessors) {
  return ptr_array::grouped_k<Built>(call, stream, allocate, device, multiprocessors);
}

}  // namespace dew_cutlass
