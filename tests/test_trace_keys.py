"""An objective states the value its loss traces by (`Objective.trace_key`): equal for two built alike, so
a refit or a resume in the same process reads the step its equal traced (row 3873), and apart for one
whose loss reads another setting, module or held tree, or a head at another policy."""

import jax
import pytest

from dew.diffusion.discrete import MDLM
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import stated_trace
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm.objective import LMObjective
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.preference import DPOObjective


def tiny(**fields) -> CausalTransformer:
    return CausalTransformer(**{"vocab_size": 8, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                "max_seq_len": 8, **fields})


BUILT = {
    "lm": (lambda **settings: LMObjective(tiny(), 8, **settings), {"z_loss": 1e-4}),
    "dpo": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"beta": 0.5}),
    "grpo": (lambda **settings: GRPOObjective(tiny(), 8, **settings), {"beta": 0.5}),
    "masked": (lambda **settings: MaskedDiffusionObjective(tiny(causal=False), MDLM(mask_id=7)(), 8,
                                                           **settings), {"steps": 3}),
}


@pytest.mark.parametrize("name", sorted(BUILT))
def test_two_objectives_built_alike_trace_by_one_key_and_another_setting_by_its_own(name):
    """Two objectives built from equal arguments state one key, built anew; one of another setting its loss
    reads, of another model, or holding a tree, states its own."""
    built, other = BUILT[name]
    key = stated_trace(built())
    assert key is not None and hash(key) == hash(stated_trace(built())) and key == stated_trace(built())
    assert stated_trace(built(**other)) != key
    held = built()
    held.variables = jax.eval_shape(held.init, jax.random.key(0))
    assert stated_trace(held) != key


def test_a_head_at_another_policy_and_a_subclass_trace_by_keys_of_their_own():
    """A head the fit ladder tiles (`tile_head`) traces by another key than the whole head; a subclass,
    whose loss may read what its parent's key does not state, states none of its own (`stated_trace`)."""
    whole, tiled = LMObjective(tiny(), 8), LMObjective(tiny(), 8, head_tile="tiled")
    assert stated_trace(whole) != stated_trace(tiled)

    class Counted(LMObjective):
        def loss(self, variables, batch, step):
            return super().loss(variables, batch, step)

    assert stated_trace(Counted(tiny(), 8)) is None
