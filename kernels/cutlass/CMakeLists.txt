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

# Each architecture's kernels compile for it alone (grouped_sm90.cu for sm90a), so no file holds code its
# instructions cannot run; ffi.cu's handlers and zeroing kernels compile for every architecture.
add_library(dew_cutlass_sm90 OBJECT grouped_sm90.cu)
set_target_properties(dew_cutlass_sm90 PROPERTIES CUDA_ARCHITECTURES 90a-real)
add_library(dew_cutlass SHARED ffi.cu $<TARGET_OBJECTS:dew_cutlass_sm90>)
foreach(target dew_cutlass dew_cutlass_sm90)
  target_include_directories(${target} PRIVATE "${XLA_FFI_INCLUDE}" "${UPSTREAM}/include"
      "${UPSTREAM}/tools/util/include")
  target_compile_definitions(${target} PRIVATE NDEBUG)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -O3 --expt-relaxed-constexpr --expt-extended-lambda>)
endforeach()
# The library reports the architectures it holds grouped kernels for (dew_cutlass.ARCHITECTURES): each
# kernel object's, not the handlers', which compile for every architecture and refuse a device none of
# the kernels runs on. Dew asks the installed wheel rather than knowing each release's list.
set(kernels)
foreach(object dew_cutlass_sm90)
  get_target_property(built ${object} CUDA_ARCHITECTURES)
  list(APPEND kernels ${built})
endforeach()
list(JOIN kernels "," ARCHITECTURES)
target_compile_definitions(dew_cutlass PRIVATE DEW_CUTLASS_ARCHITECTURES="${ARCHITECTURES}")
target_link_libraries(dew_cutlass PRIVATE CUDA::cudart)
# The CUDA runtime comes from the nvidia wheel jax's CUDA plugin installs beside this one: CUDA 12's
# nvidia/cuda_runtime/lib, CUDA 13's nvidia/cu13/lib.
set_target_properties(dew_cutlass PROPERTIES
    INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cu13/lib" BUILD_WITH_INSTALL_RPATH ON)
