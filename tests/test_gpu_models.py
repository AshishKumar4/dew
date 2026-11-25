"""The models' GPU flow (G14, the owner's rule: no unit tests on GPUs): each model that runs a GPU kernel
of its own, run through it in one process (`gpu_flow`), each cell its own file's property:

- A bf16 causal transformer, in one flow, trains a packed batch through a layer naming cuDNN (its
  documents the kernel's mask, logits and every gradient), reads a cached prompt as the uncached forward
  whatever kernel the GPU takes, decodes plain steps through cuDNN and prefills a padded batch
  (test_causal_transformer).
- A float32 decoder under a run's accurate precision policy is as exact as transformers' float32 on a
  GPU, its logits and every parameter's gradient from one program, JAX's TF32 default cleared
  (test_hf_decoders).
- ModernBERT's states in bf16 on the kernels it selects, cuDNN on its global layers
  (test_hf_modernbert).
- A weight push whose replica refuses its join aborts its NCCL group (test_rollout_servers).
- A packed window decides each row as the row alone on the joint arm, its float64 truth at model size,
  and a bf16 joint head on flash reads each packed row alone (test_decision_packing).
- The paged BF16 decode kernel's value and derivatives are the gathered attention's, every width, window,
  softcap and sink, and a softcapped model decodes a paged cache on par with its compact one, one flow on
  sm80 or later (test_kv_cache).
- A VAE encoder and its VJP, and a strided convolution's every linearization, compiled for the GPU are on
  par with its host's (test_models: a device whose compiler is not the host's, which a CPU lane is not).
"""

import functools

import pytest
import test_causal_transformer as causal
import test_decision_packing as decision_packing
import test_hf_decoders as decoders
import test_hf_modernbert as modernbert
import test_kv_cache as kv_cache
import test_models as models
import test_rollout_servers as rollout
from gpu_flow import BF16_PRODUCTS, Cell, run_cells, when

from dew.nn.kernels.generation import bf16_dot_runs

pytestmark = pytest.mark.gpu(reason="each model's own GPU kernels: flash, cuDNN, TF32 products, NCCL")


def test_the_models_run_their_gpu_kernels(without_deterministic_ops):
    """The model cells above whose properties hold under the suite's XLA flags (`gpu_flow.run_cells`), each
    one's failure kept and all raised together, Dew's own kernels taking their default schedules
    (`without_deterministic_ops`): XLA parsed the process's flags at its first compile, so taking the flag
    out of the environment moves Dew's trace-time choice of a fused kernel's schedule, the joint head's flash
    among them, and none of XLA's own (row 5491). The packed decisions, the NCCL push and the paged decode;
    the host-precision cells are their own test, each its own task (row 12439: together, a prepare's
    compiles took 687 s of a train's 667 s bound)."""
    run_cells([
        Cell("a packed window decides each row as the row alone: the joint arm",
             functools.partial(decision_packing.packed_rows_decided_alone, "joint")),
        Cell("a bf16 joint head on flash reads each packed row alone",
             functools.partial(decision_packing.a_bf16_joint_head_on_flash_reads_each_packed_row_alone,
                               without_deterministic_ops)),
        Cell("a push whose replica refuses its join aborts its NCCL group",
             rollout.a_push_whose_replica_refuses_its_join_aborts_its_group_rather_than_waiting,
             at_run="a real NCCL group on the GPU, which a prepare's compile-only device opens none of, and "
                    "no check-1 site (row 11788)"),
        Cell("the paged decode kernel decodes on par with the gathered attention, a softcapped model too",
             when(bf16_dot_runs(), BF16_PRODUCTS,
                  kv_cache.the_paged_decode_kernel_decodes_on_par_with_the_gathered_attention)),
    ])


def test_the_models_keep_their_hosts_precision_on_the_gpu(without_deterministic_ops):
    """The model cells above whose compiled programs keep their host's precision on a GPU, a compiler not
    the host's, under the suite's XLA flags as the test above (`without_deterministic_ops`, row 5491): a VAE
    encoder and its VJP at two batches, and a strided convolution's every linearization (row 12439)."""
    run_cells([
        *(Cell(f"a VAE encoder keeps its host's precision: batch {batch}, {'vjp' if backward else 'encode'}",
               functools.partial(models.a_vae_encoder_keeps_the_host_reference_precision, batch, backward))
          for batch in (1, 4) for backward in (False, True)),
        *(Cell(f"a strided convolution's {transform} keeps its host's precision",
               functools.partial(models.strided_convolution_linearizations_keep_host_precision, transform))
          for transform in ("jvp", "transpose", "forward_over_reverse", "reverse_over_forward")),
    ])


@pytest.mark.users_process
def test_the_models_run_their_gpu_kernels_as_a_users_process_does():
    """The model cells above whose properties are a user's process's, in a process of their own under a
    user's XLA flags (`users_process`: no deterministic ops, JAX's own matmul precision), which XLA reads
    once, at the process's first compile (row 5491): the GPU's attention kernels at their default schedules,
    and the accurate precision policy clearing JAX's own TF32 default, each one's failure kept and all
    raised together."""
    run_cells([
        Cell("a bf16 decoder trains on a packed batch and serves its cache through the GPU's kernels",
             causal.a_bf16_decoder_trains_and_serves_through_its_gpu_kernels),
        *(Cell(f"a float32 {fixture}'s logits and gradients under the accurate policy are as exact as "
               "transformers",
               functools.partial(
                   decoders.a_float32_decoder_under_the_accurate_policy_is_as_exact_as_transformers_on_a_gpu,
                   fixture))
          for fixture in ("qwen3-tiny", "llama-tiny")),
        Cell("ModernBERT's states in bf16 on its GPU kernels",
             modernbert.modernbert_states_in_bf16_on_the_gpu_kernels),
    ])
