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
target_compile_definitions(dew_flash_attn PRIVATE FLASHATTENTION_DISABLE_DROPOUT=1 FLASHATTENTION_DISABLE_ALIBI=1)

# FlashAttention-3 for sm90a, the instantiations a training call reaches: bf16 and fp16, heads of
# 64, 128, 192 and 256 (flash.cpp rounds 96 up to 128 with it disabled) and sliding windows, with
# variable lengths but no split keys, paged or appended caches, softcap, packed GQA or fp8. A value
# narrower than the query comes padded (Dew's `widen_value_heads`): the port sets the value's and
# the output's strides from the query's width (flash.cpp:46-84), so its 192-by-128 kernels would
# read and write past them.
set(FLASH3_DISABLED SPLIT PAGEDKV APPENDKV SOFTCAP PACKGQA FP8 SM8x CLUSTER HDIM96 HDIMDIFF64 HDIMDIFF192)
file(GLOB KERNELS3 "${UPSTREAM3}/csrc/flash_attn/src/instantiations/flash_fwd_hdim*_sm90.cu"
                   "${UPSTREAM3}/csrc/flash_attn/src/instantiations/flash_bwd_hdim*_sm90.cu")
list(FILTER KERNELS3 INCLUDE REGEX "/flash_(fwd|bwd)_hdim(64|128|192|256)_(bf16|fp16)_sm90\\.cu$")
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
    CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED CUTLASS_ENABLE_GDC_FOR_SM90 CUTLASS_DEBUG_TRACE_LEVEL=0 NDEBUG)
# Its calls to its own functions stay inside it, so loaded beside libdew_flash_attn.so, whose port
# exports the same names, neither binds the other's.
target_link_options(dew_flash_attn3 PRIVATE -Wl,-Bsymbolic)

foreach(library dew_flash_attn dew_flash_attn3)
  # Each library reports the architectures it holds code for (dew_flash_attn.ARCHITECTURES), so Dew
  # asks the installed wheel rather than knowing each release's list.
  get_target_property(compiled ${library} CUDA_ARCHITECTURES)
  list(JOIN compiled "," ARCHITECTURES)
  target_compile_definitions(${library} PRIVATE DEW_FLASH_ARCHITECTURES="${ARCHITECTURES}")
  target_compile_options(${library} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
      -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
  target_link_libraries(${library} PRIVATE CUDA::cudart)
  # The CUDA runtime comes from the nvidia wheel jax's CUDA plugin installs beside this one: CUDA 12's
  # nvidia/cuda_runtime/lib, CUDA 13's nvidia/cu13/lib.
  set_target_properties(${library} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cu13/lib" BUILD_WITH_INSTALL_RPATH ON)
endforeach()
