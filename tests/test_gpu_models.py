"""The models' GPU flow (G14, the owner's rule: no unit tests on GPUs): each model that runs a GPU kernel
of its own, run through it in one process (`gpu_flow`), each cell its own file's property:

- A causal transformer's cached prompt reads its cache as the uncached forward does, through
  FlashAttention; a layer naming cuDNN takes a packed batch as its document mask, decodes plain steps
  and prefills a padded batch through it (test_causal_transformer).
- A float32 decoder multiplies float32 products at the GPU's own precision, TF32, as transformers does
  (test_hf_decoders).
- ModernBERT's states in bf16 on the kernels it selects, cuDNN on its global layers
  (test_hf_modernbert).
- A weight push whose replica refuses its join aborts its NCCL group (test_rollout_servers).
"""

import pytest
import test_causal_transformer as causal
import test_hf_decoders as decoders
import test_hf_modernbert as modernbert
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
        Cell("a float32 decoder multiplies float32 products (TF32) as transformers does",
             decoders.a_float32_decoder_multiplies_float32_products_on_a_gpu),
        Cell("ModernBERT's states in bf16 on its GPU kernels",
             lambda: modernbert.modernbert_states_in_bf16_on_the_gpu_kernels(without_deterministic_ops)),
        Cell("a push whose replica refuses its join aborts its NCCL group",
             rollout.a_push_whose_replica_refuses_its_join_aborts_its_group_rather_than_waiting),
    ])
