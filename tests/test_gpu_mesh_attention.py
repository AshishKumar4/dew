"""Attention on two GPUs (G14, the owner's rule: no unit tests on GPUs; ledger 2221's two-GPU batch), run in
one process (`gpu_flow`), each cell its own file's property, each skipped where this process sees fewer than
two:

- cuDNN's fused attention runs inside either sequence-parallel exchange, full and causal
  (test_sequence_parallel), and sharded across processes compiles once (tests/sharded_attention_cells.py).
- Flash keeps the kernel contract on a mesh, over data and tensor axes, in a process computing as a user's
  does (test_kernels; `users_process`).
"""

import functools

import pytest
import sharded_attention_cells
import test_kernels as kernels
import test_sequence_parallel as sequence_parallel
from gpu_flow import Cell, cells_of, isolated, on_two_gpus, run_cells

pytestmark = [pytest.mark.gpu(reason="two CUDA devices: sharded cuDNN and Flash attention"),
              pytest.mark.mesh(devices=2)]


def test_two_gpus_shard_cudnn_attention(tmp_path_factory):
    """The two-GPU cuDNN cells, under the lane's deterministic ops."""
    run_cells([
        *(Cell(f"cuDNN runs inside the {exchange} exchange: {'causal' if causal else 'full'}",
               on_two_gpus(functools.partial(sequence_parallel.cudnn_runs_inside_either_exchange, exchange,
                                             causal)))
          for exchange in sorted(sequence_parallel.EXCHANGES) for causal in (False, True)),
        Cell("a sharded cuDNN attention compiles once across processes",
             on_two_gpus(isolated(tmp_path_factory)(
                 sharded_attention_cells.a_sharded_cudnn_attention_compiles_once_across_processes,
                 "sharded"))),
    ])


@pytest.mark.users_process
def test_two_gpus_shard_flash_attention_as_a_users_process_runs_it():
    """FlashAttention's mesh cells in a process computing as a user's does (`users_process`: XLA's own
    schedules, no deterministic ops nor the suite's JAX_DEFAULT_MATMUL_PRECISION=highest), the claim each
    makes (row 5491: under the lane's precision they held only at the suite's IEEE precision)."""
    run_cells([Cell(cell.label, on_two_gpus(cell.run))
               for cell in cells_of(kernels.flash_keeps_the_kernel_contract_on_a_mesh)])
