// FlashAttention-3's forward and backward for Hopper (Dao-AILab's hopper/ kernels, through
// flash-attn3-jax's torch-free port, both BSD-3) as exported XLA FFI handlers, which dew_flash_attn
// registers through ctypes, so the library carries no Python ABI. The port's entry points
// (flash.cpp) take every optional input of FlashAttention-3's API as a buffer, an empty one where it
// is not given; these handlers take only what a call gives and pass the port empty buffers for the
// rest. A zero-size operand in the program, under a mesh, is a residual the partitioner splits, and
// XLA crashed compiling such a backward.
#include "check.h"
#include "flash.h"

namespace ffi = xla::ffi;

using Out = ffi::Result<ffi::AnyBuffer>;

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
    int64_t sm_margin, Out out, Out softmax_lse, Out out_accum, Out softmax_lse_accum);

ffi::Error mha_bwd_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer dout, ffi::AnyBuffer q,
    ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer seqused_q_,
    ffi::AnyBuffer seqused_k_, int64_t max_seqlen_q_val, bool max_seqlen_q_has_value,
    int64_t max_seqlen_k_val, bool max_seqlen_k_has_value, double softmax_scale_val,
    bool softmax_scale_has_value, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    double softcap, bool deterministic, int64_t sm_margin, Out dq, Out dk, Out dv, Out softmax_d,
    Out softmax_lse_log2);

namespace {

// The port reads an input as absent when it holds no element.
int64_t kNone = 0;

XLA_FFI_Buffer EmptyBuffer(XLA_FFI_DataType dtype) {
  return XLA_FFI_Buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, nullptr, 1, &kNone};
}

// A batch has no offsets: `starts` is null, and the lengths are the arrays' own.
ffi::Error Forward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q,
                   ffi::AnyBuffer k, ffi::AnyBuffer v, const ffi::AnyBuffer* starts, int64_t longest,
                   double scale, bool causal, Out out, Out lse, Out out_accum, Out lse_accum) {
  XLA_FFI_Buffer like_q = EmptyBuffer(static_cast<XLA_FFI_DataType>(q.element_type()));
  XLA_FFI_Buffer ints = EmptyBuffer(XLA_FFI_DataType_S32);
  XLA_FFI_Buffer floats = EmptyBuffer(XLA_FFI_DataType_F32);
  ffi::AnyBuffer none_q(&like_q), none_int(&ints), none_float(&floats);
  ffi::AnyBuffer offsets = starts ? *starts : none_int;
  return mha_fwd_impl(stream, std::move(scratch), device, q, k, v, none_q, none_q, none_q, offsets, offsets,
                      none_int, none_int, none_int, none_int, none_int, none_int, none_q, none_q, none_int,
                      none_float, none_float, none_float, none_int, longest, true, longest, true, scale, true,
                      false, false, causal, -1, -1, 0, 0.0, false, 1, 0, out, lse, out_accum, lse_accum);
}

ffi::Error Backward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer dout,
                    ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer lse,
                    const ffi::AnyBuffer* starts, int64_t longest, double scale, bool causal, Out dq, Out dk,
                    Out dv, Out softmax_d, Out lse_log2) {
  XLA_FFI_Buffer ints = EmptyBuffer(XLA_FFI_DataType_S32);
  ffi::AnyBuffer none_int(&ints);
  ffi::AnyBuffer offsets = starts ? *starts : none_int;
  return mha_bwd_impl(stream, std::move(scratch), device, dout, q, k, v, out, lse, offsets, offsets, none_int,
                      none_int, longest, true, longest, true, scale, true, causal, -1, -1, 0.0, false, 0, dq,
                      dk, dv, softmax_d, lse_log2);
}

ffi::Error DenseForward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q,
                        ffi::AnyBuffer k, ffi::AnyBuffer v, double scale, bool causal, Out out, Out lse,
                        Out out_accum, Out lse_accum) {
  // The port reads max_seqlen for packed calls only; a batch's lengths are its arrays'.
  return Forward(stream, std::move(scratch), device, q, k, v, nullptr, 0, scale, causal, out, lse, out_accum,
                 lse_accum);
}

ffi::Error PackedForward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q,
                         ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer starts, int64_t longest,
                         double scale, bool causal, Out out, Out lse, Out out_accum, Out lse_accum) {
  return Forward(stream, std::move(scratch), device, q, k, v, &starts, longest, scale, causal, out, lse,
                 out_accum, lse_accum);
}

ffi::Error DenseBackward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer dout,
                         ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer out,
                         ffi::AnyBuffer lse, double scale, bool causal, Out dq, Out dk, Out dv, Out softmax_d,
                         Out lse_log2) {
  return Backward(stream, std::move(scratch), device, dout, q, k, v, out, lse, nullptr, 0, scale, causal, dq,
                  dk, dv, softmax_d, lse_log2);
}

ffi::Error PackedBackward(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device,
                          ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                          ffi::AnyBuffer out, ffi::AnyBuffer lse, ffi::AnyBuffer starts, int64_t longest,
                          double scale, bool causal, Out dq, Out dk, Out dv, Out softmax_d, Out lse_log2) {
  return Backward(stream, std::move(scratch), device, dout, q, k, v, out, lse, &starts, longest, scale, causal,
                  dq, dk, dv, softmax_d, lse_log2);
}

}  // namespace

// The CUDA architectures the library holds code for, CMake's list joined by commas ("90a-real").
extern "C" const char* dew_flash3_architectures() { return DEW_FLASH_ARCHITECTURES; }

#define DEW_FLASH3_CONTEXT                          \
  ffi::Ffi::Bind()                                  \
      .Ctx<ffi::PlatformStream<cudaStream_t>>()     \
      .Ctx<ffi::ScratchAllocator>()                 \
      .Ctx<ffi::DeviceOrdinal>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_fwd, DenseForward,
    DEW_FLASH3_CONTEXT.Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<double>("softmax_scale").Attr<bool>("is_causal")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_varlen_fwd, PackedForward,
    DEW_FLASH3_CONTEXT.Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<int64_t>("max_seqlen").Attr<double>("softmax_scale").Attr<bool>("is_causal")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_bwd, DenseBackward,
    DEW_FLASH3_CONTEXT.Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<double>("softmax_scale").Attr<bool>("is_causal")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    dew_flash3_mha_varlen_bwd, PackedBackward,
    DEW_FLASH3_CONTEXT.Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<int64_t>("max_seqlen").Attr<double>("softmax_scale").Attr<bool>("is_causal")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());
