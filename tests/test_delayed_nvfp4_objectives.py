"""A delayed-scaling NVFP4 run's amax histories as the objectives that call a model more than once, or in
parts, record them (row 12721): each call's records where its histories lie, every pass's appended."""

import jax
import numpy as np
import optax
from delayed_nvfp4 import DELAYED, rolled_once

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import generation
from dew.training import Trainer


def test_a_decision_step_rolls_its_backbones_histories_where_they_lie():
    """A decision objective's step, its backbone and head applied apart (`DecisionModel.beside_head`), rolls
    each history its backbone holds under its own part (`nvfp4/backbone/...`): its records nest where its
    histories lie, every scored exit's beside the last (review of row 12721: they came back unnested, and
    the histories stayed at their start)."""
    from decision_support import decision_rows
    from test_decision_objective import EXAMPLES, LAYOUT, SPECIALS, TOKENIZER, tiny_backbone

    from dew.decision import DecisionObjective

    with generation.run_precision(DELAYED):
        objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
        state = Trainer(objective, optax.adamw(1e-3), key=0).fit(
            decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1)
    rolled_once(state)


def test_a_block_diffusion_step_rolls_the_largest_of_every_pass():
    """A block diffusion step runs its projections in three passes (the encoder's, the conditioning pass's
    and the denoiser's): its histories roll the largest any of them recorded, the step taken, not refused
    (review of row 12721: the objective dropped its records and the step was refused)."""
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    text = CausalTransformer(vocab_size=16, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                             max_seq_len=8, layer_scalar="trainable")
    objective = BlockDiffusionObjective(DiffusionGemma(text=text, canvas_length=4), prompt_length=4)
    rows = 8 * jax.device_count()
    batch = {"text": np.random.default_rng(12721).integers(1, 16, size=(rows, 8)).astype(np.int32)}
    with generation.run_precision(DELAYED):
        trainer = Trainer(objective, optax.adamw(1e-3), key=0, split=1)
        initial = trainer.initial_state()
        state, *_ = trainer.compile(initial, batch)(initial, batch)
    rolled_once(state)
