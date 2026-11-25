"""A looped decoder's training features composed (row 10770: a pass count drawn per microbatch beside an
implicit loop and narrow copies refused its first step): a pass count drawn or fixed, the loop implicit or
unrolled, the parameters read through narrow copies or not (forced on, as every GPU's run reads them), an
exit gate or none, each pair of them met at least once and all four together, each run planned and
stepped on the CPU."""


import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from test_loop_sampled import CAP, TRAINED, token_batch
from test_pass_weights import phased_trainer

from dew.batch import LOOP_PASSES
from dew.data.dataset import rows_of
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.kernels.generation import chosen
from dew.objectives.base import Ratio
from dew.training import TrainerAdvanced
from dew.training.compiler import DrawingStep
from dew.training.interval import Interval

COMPOSED = [
    # (drawn, implicit, copies, gated): every pair of the four met, and all four together.
    (False, False, False, False), (False, False, True, True), (False, True, False, True),
    (True, False, False, True), (True, True, True, False), (True, True, True, True),
]


@attending("xla")
@pytest.mark.parametrize(("drawn", "implicit", "copies", "gated"), COMPOSED,
                         ids=lambda value: "on" if value else "off")
def test_a_looped_decoders_features_composed_plan_and_step(drawn, implicit, copies, gated):
    """A bf16 decoder of three layers whose middle one loops `CAP` passes, its features composed: it plans
    and takes four steps, two microbatches an update, committing two (`updates`), each loss finite; it holds
    narrow copies where they are on and none where off; and a drawn pass count steps more than one count,
    each microbatch's loss, with every feature on, that of the program a run fixed at its count compiles
    from the same state (its copies or its weights read whole the same values)."""
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, dtype=jnp.bfloat16,
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


@attending("xla")
def test_a_drawn_pass_count_read_in_parts_computes_its_batchs_statistics_at_each_parts_count():
    """An objective computing statistics of a step's whole batch (`batch_statistics`) reads each part cut from
    the whole batch, which carries each part's drawn pass count (`LOOP_PASSES`): read in two parts a step
    at two counts, the statistic is every row's own count, summed, both at the run and compiled ahead
    (row 12867: refused as reading the first part's count at every part, row 10770's audit)."""
    import optax

    from dew.distributed.placement import MeshSpec
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, Trainer
    from dew.training.ahead import compile_ahead

    class Counted(LMObjective):
        """The LM objective whose loss is its step's whole batch's pass counts, summed (`batch_statistics`):
        a mean of that sum over its targets, so its window's loss reads it whatever the update pools."""

        def batch_statistics(self, variables, part_at, split):
            return (functools.reduce(jnp.add, [jnp.sum(part_at(jnp.int32(index))[0][LOOP_PASSES])
                                               for index in range(split)]).astype(jnp.float32),)

        def loss(self, variables, batch, step):
            stats, aux = super().loss(variables, batch, step)
            counted = (jnp.sum(batch[LOOP_PASSES]).astype(jnp.float32) if step.batch_statistics is None
                       else step.batch_statistics[0])
            return Ratio(counted * jax.lax.stop_gradient(stats.mass), jax.lax.stop_gradient(stats.mass)), aux

    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, loop=Loop(CAP, layers=(1, 2)))
    batch = token_batch()

    def trainer_of(split: int) -> Trainer:
        return Trainer(Counted(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                       advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0), phase_passes=None,
                                                sampled_passes=TRAINED), split=split)

    trainer = trainer_of(2)
    state, _, _ = trainer.place()
    step = trainer.compile(state, batch)
    assert isinstance(step, DrawingStep)
    seen = set()
    for index in range(4):
        step.drawn(Interval(0.0, None, microbatches=0), index)
        state, loss, *_ = step(state, batch)
        counts = step.counts
        seen.update(counts)
        # Each part holds every other row: half the batch's rows at each part's count.
        assert float(loss) == sum(counts) * rows_of(batch) // len(counts), (float(loss), counts)
    assert len(seen) > 1, seen
    # A build ahead builds the step as the run does.
    ahead = trainer_of(2)
    ahead.device_mesh = MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2 ** 30))
    assert compile_ahead(ahead, batch) is not None
