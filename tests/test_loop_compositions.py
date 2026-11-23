"""A looped decoder's training features composed (row 10770: a pass count drawn per microbatch beside an
implicit loop and narrow copies refused its first step): a pass count drawn or fixed, the loop implicit or
unrolled, the parameters read through narrow copies or not (forced on, as every GPU's run reads them), an
exit gate or none, each pair of them met at least once and all four together, each run planned and
stepped on the CPU."""


import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_loop_sampled import CAP, TRAINED, token_batch
from test_pass_weights import phased_trainer

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.kernels.generation import chosen
from dew.training import TrainerAdvanced
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
    and takes four steps, two microbatches an update, committing two (`updates`), each loss finite; it holds
    narrow copies where they are on and none where off; and a drawn pass count steps more than one count,
    each microbatch's loss, with every feature on, that of the program a run fixed at its count compiles
    from the same state (its copies or its weights read whole the same values)."""
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla", dtype=jnp.bfloat16,
                              loop=Loop(CAP, layers=(1, 2), implicit=implicit, step_norm=not implicit,
                                        exit_gate=gated))
    batch = token_batch()
    with chosen(narrow_copies="on" if copies else "off"):
        trainer = phased_trainer(model, 0, None,
                                 advanced=TrainerAdvanced(sampled_passes=TRAINED if drawn else None),
                                 accumulation=2)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        interval = Interval(0.0, None, microbatches=0)
        counts, fixed = [], {}
        for index in range(4):
            if isinstance(step, DrawingStep):
                step.drawn(interval, index)
                counts.extend(step.counts)
            oracle = drawn and implicit and copies and gated
            # The state before the step donates it, read where the fixed-count program steps it too.
            given = None if not oracle else (jax.device_get(state),
                                             jax.tree.map(lambda leaf: leaf.sharding, state))
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, counts)
            if oracle:
                (count,) = step.counts
                if count not in fixed:
                    alone = phased_trainer(model.with_passes(count), 0, None, accumulation=2)
                    start, _, _ = alone.place()
                    fixed[count] = alone.compile(start, batch)
                own_loss = fixed[count](jax.device_put(*given), batch)[1]
                np.testing.assert_array_equal(np.asarray(loss), np.asarray(own_loss), err_msg=f"{index}")
    assert isinstance(step, DrawingStep) == drawn and (not drawn or len(set(counts)) > 1), counts
    assert int(state.step) == 4 and int(state.updates) == 2, (int(state.step), int(state.updates))
    held = [] if state.compute is None else jax.tree.leaves(state.compute)
    assert bool(held) == copies, (drawn, implicit, copies, gated)


def test_a_drawn_pass_count_read_in_parts_is_refused_where_its_objective_computes_its_batchs_statistics():
    """An objective computing statistics of a step's whole batch (`batch_statistics`) computes them at its
    first part, by that part's pass count, and reads them at every part: a drawn count read in two parts a
    step is refused, naming why, where read whole it plans and steps (row 10770's composition audit)."""
    import optax

    from dew.distributed.placement import MeshSpec
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, Trainer
    from dew.training.ahead import compile_ahead

    class Normalized(LMObjective):
        def batch_statistics(self, variables, part_at, split):
            return (jnp.float32(split),)

    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla", loop=Loop(CAP, layers=(1, 2)))
    batch = token_batch()

    def trainer_of(split: int, law=TRAINED):
        looped_for = model if law.cap == CAP else model.with_passes(law.cap)
        return Trainer(Normalized(looped_for, 16), optax.sgd(0.1), key=jax.random.key(0),
                       advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0), phase_passes=None,
                                                sampled_passes=law), accumulation=2, split=split)

    def stepped(trainer) -> float:
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        step.drawn(Interval(0.0, None, microbatches=0), 0)
        return float(step(state, batch)[1])

    with pytest.raises(ValueError):
        stepped(trainer_of(2))
    # A build ahead builds the step as the run does, and refuses alike.
    def built(law=TRAINED):
        trainer = trainer_of(2, law)
        trainer.device_mesh = MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2 ** 30))
        return compile_ahead(trainer, batch)

    with pytest.raises(ValueError):
        built()
    # The same build of a one-count law, which reads no other count's statistics, is made.
    assert built(dataclasses.replace(TRAINED, cap=1)) is not None
    assert np.isfinite(stepped(trainer_of(1)))
    # A law of one count's parts read the statistics of their own count.
    assert np.isfinite(stepped(trainer_of(2, dataclasses.replace(TRAINED, cap=1))))
