# libdew_cutlass.so: CUTLASS's grouped GEMMs (NVIDIA, BSD-3) as XLA FFI targets, with no Python ABI.
# UPSTREAM is a checkout of CUTLASS at CUTLASS_COMMIT (build.sh); XLA_FFI_INCLUDE is jaxlib's include
# directory, which holds XLA's FFI headers (Apache-2.0).
cmake_minimum_required(VERSION 3.24)
project(dew_cutlass LANGUAGES CXX CUDA)

set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/upstream" CACHE PATH "CUTLASS checkout")
set(XLA_FFI_INCLUDE "" CACHE PATH "jaxlib's include directory, holding xla/ffi/api/ffi.h")
# sm80 code runs on every compute capability 8.x (CUTLASS 2.x's grouped GEMM); sm90a is Hopper's
# warp-specialized TMA kernels, sm100a Blackwell's datacenter parts' tcgen05 ones. Each handler picks the
# device's at the call (ffi.cu).
set(CMAKE_CUDA_ARCHITECTURES 80-real 90a-real 100a-real)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit REQUIRED)

add_library(dew_cutlass SHARED ffi.cu)
target_include_directories(dew_cutlass PRIVATE "${XLA_FFI_INCLUDE}" "${UPSTREAM}/include"
    "${UPSTREAM}/tools/util/include")
# The library reports the architectures it holds code for (dew_cutlass.ARCHITECTURES), so Dew asks the
# installed wheel rather than knowing each release's list.
get_target_property(compiled dew_cutlass CUDA_ARCHITECTURES)
list(JOIN compiled "," ARCHITECTURES)
target_compile_definitions(dew_cutlass PRIVATE DEW_CUTLASS_ARCHITECTURES="${ARCHITECTURES}" NDEBUG)
target_compile_options(dew_cutlass PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 --expt-relaxed-constexpr --expt-extended-lambda>)
target_link_libraries(dew_cutlass PRIVATE CUDA::cudart)
# The CUDA runtime comes from the nvidia wheel jax's CUDA plugin installs beside this one: CUDA 12's
# nvidia/cuda_runtime/lib, CUDA 13's nvidia/cu13/lib.
set_target_properties(dew_cutlass PROPERTIES
    INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cu13/lib" BUILD_WITH_INSTALL_RPATH ON)
