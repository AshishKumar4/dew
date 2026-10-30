"""The models' GPU flow (G14, the owner's rule: no unit tests on GPUs): each model that runs a GPU kernel
of its own, run through it in one process (`gpu_flow`), each cell its own file's property:

- A causal transformer's cached prompt reads its cache as the uncached forward does, through
  FlashAttention; a layer naming cuDNN takes a packed batch as its document mask, decodes plain steps
  and prefills a padded batch through it (test_causal_transformer).
- A float32 decoder under a run's accurate precision policy is as exact as transformers' float32 on a
  GPU, JAX's TF32 default cleared (test_hf_decoders).
- ModernBERT's states in bf16 on the kernels it selects, cuDNN on its global layers
  (test_hf_modernbert).
- A weight push whose replica refuses its join aborts its NCCL group (test_rollout_servers).
- A packed window decides each row as the row alone on the joint arm, its float64 truth at model size,
  and a bf16 joint head on flash reads each packed row alone (test_decision_packing).
- The paged BF16 decode kernel's value and derivatives are the gathered attention's, every width, window,
  softcap and sink, and a softcapped model decodes a paged cache on par with its compact one, on sm80 or
  later (test_kv_cache).
"""

import functools

import pytest
import test_causal_transformer as causal
import test_decision_packing as decision_packing
import test_hf_decoders as decoders
import test_hf_modernbert as modernbert
import test_kv_cache as kv_cache
import test_rollout_servers as rollout
from gpu_flow import Cell, run_cells

pytestmark = pytest.mark.gpu(reason="each model's own GPU kernels: flash, cuDNN, TF32 products, NCCL")


def test_the_models_run_their_gpu_kernels(rng, without_deterministic_ops):
    """Every model cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together. The cuDNN cells run without XLA's deterministic ops, which Dew refuses cuDNN attention
    under (`without_deterministic_ops`)."""
    run_cells([
        Cell("a causal transformer's cached prompt reads its cache as the uncached forward (flash)",
             lambda: causal.a_cached_prompt_reads_its_cache_as_the_uncached_forward(
                 without_deterministic_ops)),
        Cell("a layer naming cuDNN takes a packed batch as its document mask",
             lambda: causal.a_layer_naming_cudnn_takes_a_packed_batch_as_its_document_mask(
                 rng, without_deterministic_ops)),
        Cell("a plain decode step attends through cuDNN",
             lambda: causal.a_plain_decode_step_attends_through_cudnn_where_it_runs(
                 rng, without_deterministic_ops)),
        Cell("a padded prefill attends through cuDNN",
             lambda: causal.a_padded_prefill_attends_through_cudnn_where_it_runs(
                 rng, without_deterministic_ops)),
        Cell("a float32 decoder under the accurate policy is as exact as transformers",
             decoders.a_float32_decoder_under_the_accurate_policy_is_as_exact_as_transformers_on_a_gpu),
        Cell("ModernBERT's states in bf16 on its GPU kernels",
             lambda: modernbert.modernbert_states_in_bf16_on_the_gpu_kernels(without_deterministic_ops)),
        Cell("a packed window decides each row as the row alone: the joint arm",
             functools.partial(decision_packing.test_a_packed_window_decides_each_row_as_the_row_alone,
                               "joint")),
        Cell("a bf16 joint head on flash reads each packed row alone",
             functools.partial(decision_packing.a_bf16_joint_head_on_flash_reads_each_packed_row_alone,
                               without_deterministic_ops)),
        Cell("a push whose replica refuses its join aborts its NCCL group",
             rollout.a_push_whose_replica_refuses_its_join_aborts_its_group_rather_than_waiting),
        *(Cell(f"the paged decode kernel's value and derivatives are the gathered attention's: width {width}"
               f"{' window' if windowed else ''}{' softcap' if softcap else ''}{' sinks' if sunk else ''}",
               functools.partial(kv_cache.native_gpu_paged_value_and_derivatives_are_the_gathered_attentions,
                                 without_deterministic_ops, width, windowed, softcap, sunk), "sm80")
          for width in (64, 256) for windowed in (False, True) for softcap in (None, 0.5)
          for sunk in (False, True)),
        Cell("a softcapped model decodes a paged cache on par with its compact cache",
             functools.partial(kv_cache.a_softcapped_model_decodes_a_paged_cache_on_par_with_its_compact_cache,
                               without_deterministic_ops), "sm80"),
    ])
