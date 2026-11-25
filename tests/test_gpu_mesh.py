"""Two GPUs (G14, the owner's rule: no unit tests on GPUs; ledger 2221's two-GPU batch), run in one process
(`gpu_flow`), each cell its own file's property, each skipped where this process sees fewer than two:

- A launched process keeps every device inside a Slurm step; a rank that stalls between collectives ends
  the pool; a rank that holds nothing copies none of the tree to its host (test_distribution).
- An NCCL push delivers every tensor to each replica on another GPU, bf16 and FP8 (test_rollout_servers).
- Each process reads the facts of the GPU it sees (test_lowering).

Attention's two-GPU cells are tests/test_gpu_mesh_attention.py's.
"""

import functools

import pytest
import test_distribution as distribution
import test_lowering as lowering
import test_rollout_servers as rollout
from gpu_flow import Cell, isolated, on_two_gpus, run_cells

pytestmark = [pytest.mark.gpu(reason="two CUDA devices: NCCL, a cluster's device pins"),
              pytest.mark.mesh(devices=2)]


def test_two_gpus_run_their_collectives_and_shards(tmp_path_factory):
    """Every two-GPU cell above (`gpu_flow.run_cells`), each one's failure kept and all raised together."""
    run_cells([
        Cell("a launched process keeps every device inside a Slurm step",
             on_two_gpus(distribution.a_launched_process_keeps_every_device_inside_a_slurm_step)),
        Cell("a rank that stalls between collectives ends the pool",
             on_two_gpus(distribution.a_rank_that_stalls_between_collectives_ends_the_pool)),
        Cell("a rank that holds nothing copies none of the tree to its host",
             on_two_gpus(distribution.a_rank_that_holds_nothing_copies_none_of_the_tree_to_its_host)),
        *(Cell(f"an NCCL push delivers every tensor to each replica on another GPU: {name}",
               on_two_gpus(functools.partial(
                   rollout.an_nccl_push_delivers_every_tensor_to_each_replica_on_another_gpu, fixture)))
          for name, fixture in (("bfloat16", rollout.FIXTURE), ("fp8", rollout.FP8))),
        Cell("each process reads the facts of the GPU it sees",
             on_two_gpus(isolated(tmp_path_factory)(
                 lowering.each_process_reads_the_facts_of_the_gpu_it_sees, "facts"))),
    ])
