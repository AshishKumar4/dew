"""Two GPUs (G14, the owner's rule: no unit tests on GPUs; ledger 2221's two-GPU batch), run in one process
(`gpu_flow`), each cell its own file's property, each skipped where this process sees fewer than two:

- A launched process keeps every device inside a Slurm step; a rank that stalls between collectives ends
  the pool; a rank that holds nothing copies none of the tree to its host (test_distribution).
- cuDNN's fused attention runs inside either sequence-parallel exchange, full and causal
  (test_sequence_parallel), and sharded across processes compiles once (tests/sharded_attention_cells.py).
- An NCCL push delivers every tensor to each replica on another GPU, bf16 and FP8 (test_rollout_servers).
- Each process reads the facts of the GPU it sees (test_lowering).
- Flash keeps the kernel contract on a mesh, over data and tensor axes (test_kernels).
"""

import functools

import jax
import pytest
import sharded_attention_cells
import test_distribution as distribution
import test_kernels as kernels
import test_lowering as lowering
import test_rollout_servers as rollout
import test_sequence_parallel as sequence_parallel
from gpu_flow import Cell, cells_of, run_cells

pytestmark = [pytest.mark.gpu(reason="two CUDA devices: NCCL, a cluster's device pins, sharded cuDNN"),
              pytest.mark.mesh(devices=2)]


def on_two(check):
    """`check` where this process sees two GPUs, else skipped."""
    def run():
        if jax.default_backend() != "gpu" or jax.device_count() < 2:
            pytest.skip("needs two GPUs")
        check()
    return run


def test_two_gpus_run_their_collectives_and_shards(tmp_path_factory):
    """Every two-GPU cell above (`gpu_flow.run_cells`), each one's failure kept and all raised together."""
    run_cells([
        Cell("a launched process keeps every device inside a Slurm step",
             on_two(distribution.a_launched_process_keeps_every_device_inside_a_slurm_step)),
        Cell("a rank that stalls between collectives ends the pool",
             on_two(distribution.a_rank_that_stalls_between_collectives_ends_the_pool)),
        Cell("a rank that holds nothing copies none of the tree to its host",
             on_two(distribution.a_rank_that_holds_nothing_copies_none_of_the_tree_to_its_host)),
        *(Cell(f"an NCCL push delivers every tensor to each replica on another GPU: {name}",
               on_two(functools.partial(
                   rollout.an_nccl_push_delivers_every_tensor_to_each_replica_on_another_gpu, fixture)))
          for name, fixture in (("bfloat16", rollout.FIXTURE), ("fp8", rollout.FP8))),
        Cell("each process reads the facts of the GPU it sees",
             on_two(lambda: lowering.each_process_reads_the_facts_of_the_gpu_it_sees(
                 tmp_path_factory.mktemp("facts")))),
    ])


def test_two_gpus_shard_cudnn_attention(tmp_path_factory, without_deterministic_ops):
    """The two-GPU cuDNN cells, without XLA's deterministic ops, which Dew refuses cuDNN attention under
    (`without_deterministic_ops`)."""
    run_cells([
        *(Cell(f"cuDNN runs inside the {exchange} exchange: {'causal' if causal else 'full'}",
               on_two(functools.partial(sequence_parallel.cudnn_runs_inside_either_exchange, exchange, causal,
                                        without_deterministic_ops)))
          for exchange in sorted(sequence_parallel.EXCHANGES) for causal in (False, True)),
        *(Cell(cell.label, on_two(cell.run))
          for cell in cells_of(kernels.flash_keeps_the_kernel_contract_on_a_mesh,
                               without_deterministic_ops=without_deterministic_ops)),
        Cell("a sharded cuDNN attention compiles once across processes",
             on_two(lambda: sharded_attention_cells.a_sharded_cudnn_attention_compiles_once_across_processes(
                 tmp_path_factory.mktemp("sharded"), without_deterministic_ops))),
    ])
