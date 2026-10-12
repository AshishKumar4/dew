# libdew_flash_attn.so: FlashAttention-2's CUDA kernels (Dao-AILab, through
# flash_attn_jax's XLA FFI port, both BSD-3) with no Python ABI. UPSTREAM is
# a checkout of flash_attn_jax at FLASH_ATTN_JAX_COMMIT (build.sh).
# libdew_flash_attn3.so: FlashAttention-3's Hopper kernels (Dao-AILab's hopper/, through
# flash-attn3-jax's torch-free port, both BSD-3), the same way. UPSTREAM3 is a checkout of
# flash-attn3-jax at FLASH_ATTN3_JAX_COMMIT. It is a library of its own because its kernels need a
# newer CUTLASS and C++20, and the two ports define the same names.
cmake_minimum_required(VERSION 3.24)
project(dew_flash_attn LANGUAGES CXX CUDA)

set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/upstream" CACHE PATH "flash_attn_jax checkout")
set(UPSTREAM3 "${CMAKE_CURRENT_SOURCE_DIR}/upstream3" CACHE PATH "flash-attn3-jax checkout")
# sm80 code runs on every compute capability 8.x, and sm120 code on Blackwell's workstation and consumer
# parts (the RTX PRO 6000 and RTX 50 series), which run Ampere's MMA instructions but not sm80 code.
# Hopper runs FlashAttention-3 (libdew_flash_attn3.so).
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
# The library reports the architectures it holds code for (dew_flash_attn.ARCHITECTURES), so Dew asks
# the installed wheel rather than knowing each release's list.
list(JOIN CMAKE_CUDA_ARCHITECTURES "," ARCHITECTURES)
set(DEFINES FLASHATTENTION_DISABLE_DROPOUT=1 FLASHATTENTION_DISABLE_ALIBI=1
    DEW_FLASH_ARCHITECTURES="${ARCHITECTURES}")
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

# FlashAttention-3 for sm90a, the instantiations a training call reaches: bf16 and fp16, heads of
# 64, 128 and 256 (flash.cpp rounds 96 up to 128 and 192 up to 256 with those two disabled), with
# variable lengths but no split keys, paged or appended caches, softcap, sliding window, packed GQA
# or fp8.
set(FLASH3_DISABLED SPLIT PAGEDKV APPENDKV SOFTCAP LOCAL PACKGQA FP8 SM8x CLUSTER HDIM96 HDIM192
    HDIMDIFF64 HDIMDIFF192)
file(GLOB KERNELS3 "${UPSTREAM3}/csrc/flash_attn/src/instantiations/flash_fwd_hdim*_sm90.cu"
                   "${UPSTREAM3}/csrc/flash_attn/src/instantiations/flash_bwd_hdim*_sm90.cu")
list(FILTER KERNELS3 INCLUDE REGEX "/flash_(fwd|bwd)_hdim(64|128|256)_(bf16|fp16)_sm90\\.cu$")
add_library(dew_flash_attn3 SHARED ffi3.cpp
            "${UPSTREAM3}/csrc/flash_attn/flash.cpp"
            "${UPSTREAM3}/csrc/flash_attn/fill_kernel.cu"
            "${UPSTREAM3}/csrc/flash_attn/src/flash_prepare_scheduler.cu"
            ${KERNELS3})
# With split keys disabled, flash.cpp's run_mha_fwd_combine returns its refusal from a void function,
# which GCC rejects without -fpermissive. Nothing calls it then: the port's split count is always 1.
set_source_files_properties("${UPSTREAM3}/csrc/flash_attn/flash.cpp" PROPERTIES COMPILE_OPTIONS -fpermissive)
target_include_directories(dew_flash_attn3 PRIVATE
    "${UPSTREAM3}/csrc" "${UPSTREAM3}/csrc/flash_attn" "${UPSTREAM3}/csrc/flash_attn/src"
    "${UPSTREAM3}/csrc/cutlass/include")
set_target_properties(dew_flash_attn3 PROPERTIES CUDA_ARCHITECTURES 90a-real CXX_STANDARD 20 CUDA_STANDARD 20)
list(TRANSFORM FLASH3_DISABLED PREPEND FLASHATTENTION_DISABLE_ OUTPUT_VARIABLE DEFINES3)
target_compile_definitions(dew_flash_attn3 PRIVATE ${DEFINES3} FLASHATTENTION_DISABLE_DROPOUT=1
    CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED CUTLASS_ENABLE_GDC_FOR_SM90 CUTLASS_DEBUG_TRACE_LEVEL=0 NDEBUG
    DEW_FLASH_ARCHITECTURES="90a-real")
target_compile_options(dew_flash_attn3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
# Its calls to its own functions stay inside it, so loaded beside libdew_flash_attn.so, whose port
# exports the same names, neither binds the other's.
target_link_options(dew_flash_attn3 PRIVATE -Wl,-Bsymbolic)
target_link_libraries(dew_flash_attn3 PRIVATE CUDA::cudart)
set_target_properties(dew_flash_attn3 PROPERTIES
    INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cu13/lib" BUILD_WITH_INSTALL_RPATH ON)
