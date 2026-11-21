"""A looped decoder's training features composed (row 10770: a pass count drawn per microbatch beside an
implicit loop and narrow copies refused its first step): a pass count drawn or fixed, the loop implicit or
unrolled, the parameters read through narrow copies or not (forced on, as every GPU's run reads them), an
exit gate or none, each pair of them met at least once and all four together, each run planned and
stepped on the CPU."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_loop_sampled import CAP, TRAINED, token_batch
from test_pass_weights import phased_trainer

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.kernels.generation import chosen
from dew.training.compiler import DrawingStep
from dew.training.interval import Interval

COMPOSED = [
    # (drawn, implicit, copies, gated): every pair of the four met, and all four together.
    (False, False, False, False), (False, False, True, True), (False, True, False, True),
    (True, False, False, True), (True, True, True, False), (True, True, True, True),
]


@pytest.mark.parametrize(("drawn", "implicit", "copies", "gated"), COMPOSED,
                         ids=lambda value: "on" if value else "off")
def test_a_looped_decoders_features_composed_plan_and_step(drawn, implicit, copies, gated):
    """A bf16 decoder of three layers whose middle one loops `CAP` passes, its features composed: it plans
    and takes four steps, two microbatches an update, each loss finite; it holds narrow copies where they
    are on and none where off; and a drawn pass count steps more than one count."""
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla", dtype=jnp.bfloat16,
                              loop=Loop(CAP, layers=(1, 2), implicit=implicit, step_norm=not implicit,
                                        exit_gate=gated))
    batch = token_batch()
    with chosen(narrow_copies="on" if copies else "off"):
        trainer = phased_trainer(model, 0, None, sampled_passes=TRAINED if drawn else None, accumulation=2)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        interval = Interval(0.0, None, microbatches=0)
        counts = []
        for index in range(4):
            if isinstance(step, DrawingStep):
                step.drawn(interval, index)
                counts.extend(step.counts)
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, counts)
    assert isinstance(step, DrawingStep) == drawn and (not drawn or len(set(counts)) > 1), counts
    held = [] if state.compute is None else jax.tree.leaves(state.compute)
    assert bool(held) == copies, (drawn, implicit, copies, gated)
