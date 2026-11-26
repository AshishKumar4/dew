// Hopper's grouped GEMMs (sm90a): CUTLASS 3's pointer-array warp-specialized cooperative kernel, its TMA
// descriptors set per group on the device (grouped_ptr_array.cuh, which sm100's shares).
#include "grouped_ptr_array.cuh"

namespace dew_cutlass {
namespace {

using namespace cute;

// One kernel: 16-bit operands of `LayoutA` and `LayoutB` (CUTLASS's: A is m x k, B is k x n), fp32 sums,
// row-major `ElementD` out, a 128 x 256 x 64 tile of a pair of multiprocessors (cooperative, as CUTLASS's
// grouped GEMMs on Hopper default to).
template <class ElementAB, class ElementD, class LayoutA, class LayoutB>
struct Built {
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

  // The threads of its tiled MMA: two consumer warpgroups.
  static constexpr int kConsumers = static_cast<int>(size(typename Mainloop::TiledMma{}));
};

}  // namespace

Instantiated sm90_kernel(Dtype result) { return ptr_array::instantiated<Built>(result, "wgmma"); }

const char* sm90_grouped(const Grouped& call, cudaStream_t stream, const Allocate& allocate, int device,
                         int multiprocessors) {
  return ptr_array::grouped<Built>(call, stream, allocate, device, multiprocessors);
}

const char* sm90_grouped_k(const GroupedK& call, cudaStream_t stream, const Allocate& allocate, int device,
                           int multiprocessors) {
  return ptr_array::grouped_k<Built>(call, stream, allocate, device, multiprocessors);
}

}  // namespace dew_cutlass
