# libdew_flash_attn.so: FlashAttention-2's CUDA kernels (Dao-AILab, through
# flash_attn_jax's XLA FFI port, both BSD-3) with no Python ABI. UPSTREAM is
# a checkout of flash_attn_jax at FLASH_ATTN_JAX_COMMIT (build.sh).
cmake_minimum_required(VERSION 3.24)
project(dew_flash_attn LANGUAGES CXX CUDA)

set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/upstream" CACHE PATH "flash_attn_jax checkout")
# sm80 code runs on every compute capability 8.x, and sm120 code on Blackwell's workstation and consumer
# parts (the RTX PRO 6000 and RTX 50 series), which run Ampere's MMA instructions but not sm80 code.
# FlashAttention-2 is not Hopper's or the datacenter Blackwell's kernel.
set(CMAKE_CUDA_ARCHITECTURES 80-real 120-real)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit REQUIRED)

file(GLOB KERNELS "${UPSTREAM}/csrc/flash_attn/src/flash_fwd_hdim*.cu"
                  "${UPSTREAM}/csrc/flash_attn/src/flash_bwd_hdim*.cu"
                  "${UPSTREAM}/csrc/flash_attn/src/flash_fwd_split_hdim*.cu")
add_library(dew_flash_attn SHARED ffi.cpp
            "${UPSTREAM}/csrc/flash_attn/mha_fwd.cpp"
            "${UPSTREAM}/csrc/flash_attn/mha_bwd.cpp"
            "${UPSTREAM}/csrc/flash_attn/flash_common.cpp"
            ${KERNELS})
target_include_directories(dew_flash_attn PRIVATE
    "${UPSTREAM}/csrc" "${UPSTREAM}/csrc/flash_attn" "${UPSTREAM}/csrc/flash_attn/src"
    "${UPSTREAM}/csrc/cutlass/include")
set(DEFINES FLASHATTENTION_DISABLE_DROPOUT=1 FLASHATTENTION_DISABLE_ALIBI=1)
target_compile_definitions(dew_flash_attn PRIVATE ${DEFINES})
target_compile_options(dew_flash_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
target_link_libraries(dew_flash_attn PRIVATE CUDA::cudart)
# The CUDA runtime comes from the nvidia wheel jax's CUDA plugin installs beside this one: CUDA 12's
# nvidia/cuda_runtime/lib, CUDA 13's nvidia/cu13/lib.
set_target_properties(dew_flash_attn PROPERTIES
    INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cu13/lib" BUILD_WITH_INSTALL_RPATH ON)
