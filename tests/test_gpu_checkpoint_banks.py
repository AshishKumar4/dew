"""A CPU-owned LoRA run's checkpoint on a GPU, where its host companion and its accelerator are two meshes:
the GPU tier's file, apart from tests/test_checkpoint_banks.py's CPU tests, which no GPU runs (the owner's
rule: no unit test on a GPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_checkpoint_banks as banks
from flax import linen as nn

from dew.checkpoints import Checkpoints
from dew.lora import LoRA
from dew.objectives.lm import LMObjective


@pytest.mark.gpu(reason="a CPU-owned run's host companion and its accelerator are two meshes only on a GPU")
def test_a_cpu_owned_lora_run_saves_across_its_two_meshes(tmp_path):
    """A LoRA run whose variables the CPU owns holds its moving leaves on the host companion and its frozen
    base on the accelerator: its save's refusal of a non-finite state reads both, one program a placement,
    and the run saves (train 904's review, F1: one program over both meshes was refused, so every save of
    such a run crashed on a GPU)."""
    base = banks.decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, nn.unbox(base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))), key=1)
    objective = LMObjective(adapter.model, 8, head_chunks=1, ema_decay=None,
                            variables=jax.tree.map(np.asarray, adapter.variables))
    banks.trained(tmp_path / "run", objective, banks.HOST)
    assert Checkpoints(str(tmp_path / "run")).latest == 1
