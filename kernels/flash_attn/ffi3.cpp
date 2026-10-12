// FlashAttention-3's forward and backward for Hopper (Dao-AILab's hopper/ kernels, through
// flash-attn3-jax's torch-free port, both BSD-3) as exported XLA FFI handlers, which dew_flash_attn
// registers through ctypes, so the library carries no Python ABI. The bindings are flash-attn3-jax's
// (csrc/flash_attn/flash_api.cpp), argument for argument; its implementations are in flash.cpp.
#include "check.h"
#include "flash.h"

namespace ffi = xla::ffi;

ffi::Error mha_fwd_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q, ffi::AnyBuffer k,
    ffi::AnyBuffer v, ffi::AnyBuffer k_new_, ffi::AnyBuffer v_new_, ffi::AnyBuffer q_v_,
    ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer cu_seqlens_k_new_,
    ffi::AnyBuffer seqused_q_, ffi::AnyBuffer seqused_k_, ffi::AnyBuffer page_table_,
    ffi::AnyBuffer kv_batch_idx_, ffi::AnyBuffer leftpad_k_, ffi::AnyBuffer rotary_cos_,
    ffi::AnyBuffer rotary_sin_, ffi::AnyBuffer seqlens_rotary_, ffi::AnyBuffer q_descale_,
    ffi::AnyBuffer k_descale_, ffi::AnyBuffer v_descale_, ffi::AnyBuffer scheduler_metadata_,
    int64_t max_seqlen_q_val, bool max_seqlen_q_has_value, int64_t max_seqlen_k_val,
    bool max_seqlen_k_has_value, double softmax_scale_val, bool softmax_scale_has_value, bool pack_gqa_val,
    bool pack_gqa_has_value, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, double softcap, bool is_rotary_interleaved, int64_t num_splits,
    int64_t sm_margin, ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> softmax_lse,
    ffi::Result<ffi::AnyBuffer> out_accum, ffi::Result<ffi::AnyBuffer> softmax_lse_accum);

ffi::Error mha_bwd_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer dout, ffi::AnyBuffer q,
    ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer seqused_q_,
    ffi::AnyBuffer seqused_k_, int64_t max_seqlen_q_val, bool max_seqlen_q_has_value,
    int64_t max_seqlen_k_val, bool max_seqlen_k_has_value, double softmax_scale_val,
    bool softmax_scale_has_value, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    double softcap, bool deterministic, int64_t sm_margin, ffi::Result<ffi::AnyBuffer> dq,
    ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv, ffi::Result<ffi::AnyBuffer> softmax_d,
    ffi::Result<ffi::AnyBuffer> softmax_lse_log2);

// The CUDA architectures the library holds code for, CMake's list joined by commas ("90a-real").
extern "C" const char* dew_flash3_architectures() { return DEW_FLASH_ARCHITECTURES; }

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_fwd, mha_fwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // k_new
        .Arg<ffi::AnyBuffer>()  // v_new
        .Arg<ffi::AnyBuffer>()  // q_v
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_q
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k_new
        .Arg<ffi::AnyBuffer>()  // seqused_q
        .Arg<ffi::AnyBuffer>()  // seqused_k
        .Arg<ffi::AnyBuffer>()  // page_table
        .Arg<ffi::AnyBuffer>()  // kv_batch_idx
        .Arg<ffi::AnyBuffer>()  // leftpad_k
        .Arg<ffi::AnyBuffer>()  // rotary_cos
        .Arg<ffi::AnyBuffer>()  // rotary_sin
        .Arg<ffi::AnyBuffer>()  // seqlens_rotary
        .Arg<ffi::AnyBuffer>()  // q_descale
        .Arg<ffi::AnyBuffer>()  // k_descale
        .Arg<ffi::AnyBuffer>()  // v_descale
        .Arg<ffi::AnyBuffer>()  // scheduler_metadata
        .Attr<int64_t>("max_seqlen_q")
        .Attr<bool>("max_seqlen_q_has_value")
        .Attr<int64_t>("max_seqlen_k")
        .Attr<bool>("max_seqlen_k_has_value")
        .Attr<double>("softmax_scale_val")
        .Attr<bool>("softmax_scale_has_value")
        .Attr<bool>("pack_gqa_val")
        .Attr<bool>("pack_gqa_has_value")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<int64_t>("attention_chunk")
        .Attr<double>("softcap")
        .Attr<bool>("is_rotary_interleaved")
        .Attr<int64_t>("num_splits")
        .Attr<int64_t>("sm_margin")
        .Ret<ffi::AnyBuffer>()  // out
        .Ret<ffi::AnyBuffer>()  // softmax_lse
        .Ret<ffi::AnyBuffer>()  // out_accum
        .Ret<ffi::AnyBuffer>()  // softmax_lse_accum
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_bwd, mha_bwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>()  // dout
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // out
        .Arg<ffi::AnyBuffer>()  // softmax_lse
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_q
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k
        .Arg<ffi::AnyBuffer>()  // seqused_q
        .Arg<ffi::AnyBuffer>()  // seqused_k
        .Attr<int64_t>("max_seqlen_q_val")
        .Attr<bool>("max_seqlen_q_has_value")
        .Attr<int64_t>("max_seqlen_k_val")
        .Attr<bool>("max_seqlen_k_has_value")
        .Attr<double>("softmax_scale_val")
        .Attr<bool>("softmax_scale_has_value")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<double>("softcap")
        .Attr<bool>("deterministic")
        .Attr<int64_t>("sm_margin")
        .Ret<ffi::AnyBuffer>()  // dq
        .Ret<ffi::AnyBuffer>()  // dk
        .Ret<ffi::AnyBuffer>()  // dv
        .Ret<ffi::AnyBuffer>()  // softmax_d
        .Ret<ffi::AnyBuffer>()  // softmax_lse_log2
);
