"""A step over several GPUs re-gathers no sharded weight to fit them: its plan fits it by Dew's own ladder,
XLA's rematerialization pass off (`dew.training.memory.COLLECTIVES`, row 1307)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import generation
from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.lm import LMObjective
from dew.telemetry.instrumentation import hlo_instructions
from dew.telemetry.trace import collective
from dew.training import MeshSpec, Trainer
from dew.training.ahead import compile_ahead

ROWS, LENGTH = 4, 2048
"""The step's rows and their tokens."""


def compiled(memory: int):
    """A four-layer decoder's step, its compute bf16, over four compile-only H100s split fsdp 2 by tensor 2,
    each stating `memory` bytes, compiled ahead as a run on them plans it, its narrow copies off: the step
    held as it compiled (`AheadOfTime`) and the executable."""
    model = CausalTransformer(vocab_size=32000, emb_features=1024, num_layers=4, num_heads=8, num_kv_heads=4,
                              head_dim=128, mlp_features=4096, max_seq_len=LENGTH, dtype=jnp.bfloat16)
    trainer = Trainer(LMObjective(model, seq_len=LENGTH), optax.adamw(1e-4), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, tensor=2), split=1, checkpoints=None, tracker=None)
    trainer.device_mesh = trainer.mesh.build(list(compile_only_gpus("sm90", 4, memory)))
    # No narrow copies, which a plan drops where its room is short: one program at any room.
    with generation.chosen(narrow_copies="off"):
        ahead = compile_ahead(trainer, {"text": np.zeros((ROWS, LENGTH + 1), np.int32)})
    return ahead, trainer.executable


def gathered(executable) -> list[int]:
    """The bytes of each all-gather `executable` runs, an async pair once (`hlo_instructions`)."""
    return sorted(sum(instruction.buffers) for instruction in hlo_instructions(executable).values()
                  if collective(instruction.op) == "all-gather" and not instruction.op.endswith("-start"))


def test_a_tensor_parallel_step_at_its_rooms_edge_gathers_what_it_gathers_with_room_to_spare():
    """A tensor-parallel step whose GPUs state 1.2 times what it needs at its plan's first rung keeps that
    rung, and gathers what the same step gathers on GPUs of 80 GiB, gather for gather: no weight gathered
    again to fit. XLA's rematerialization pass, against a limit of its own below the plan's, re-gathered two
    of its weights there (33.6 MB more a step; at Qwen3-8B's widths over four H100s at 48 GiB, 14
    all-gathers, 16.8 GB against 11.7, the plan's rung and temporaries alike either way)."""
    roomy, spared = compiled(80 * 2 ** 30)
    ahead, executable = compiled(int(1.2 * (roomy.temp_bytes + roomy.argument_bytes)))
    assert ahead.plan["remat"] is None, ahead.plan["remat"]
    assert gathered(executable) == gathered(spared)
