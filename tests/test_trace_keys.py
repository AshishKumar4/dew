"""An objective states the value its loss traces by (`Objective.trace_key`): equal for two built alike, so
a refit or a resume in the same process reads the step its equal traced (row 3873), and apart for one
whose loss reads another setting, module or held tree, or a head at another policy."""

import jax
import jax.numpy as jnp
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
    "dpo pairs": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"pairs": 2}),
    "dpo segments": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"segments": 1}),
    "grpo": (lambda **settings: GRPOObjective(tiny(), 8, **settings), {"beta": 0.5}),
    "masked": (lambda **settings: MaskedDiffusionObjective(tiny(causal=False), MDLM(mask_id=7)(), 8,
                                                           **settings), {"steps": 3}),
    "masked chunks": (lambda **settings: MaskedDiffusionObjective(tiny(causal=False), MDLM(mask_id=7)(), 8,
                                                                  **settings), {"head_chunks": 2}),
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


def test_an_objective_holding_arrays_traces_by_their_shapes_whatever_their_values():
    """A held tree's arrays pass to the trace as arguments (`held_variables`), so two objectives holding
    trees of the same shapes and dtypes state one key whatever the arrays' values, one whose tree is stored
    in another dtype a key of its own, and each apart from the objective that draws its tree from a key
    (the train whose every objective holding device arrays failed at its key: `held_shapes` converted each
    array to a dtype)."""
    def holding(seed: int, dtype=None) -> LMObjective:
        held = LMObjective(tiny(), 8)
        tree = held.init(jax.random.key(seed))
        held.variables = tree if dtype is None else jax.tree.map(
            lambda leaf: leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, tree)
        return held

    key = stated_trace(holding(0))
    assert key is not None and key == stated_trace(holding(1)) and hash(key) == hash(stated_trace(holding(1)))
    assert stated_trace(holding(0, jnp.bfloat16)) != key
    assert stated_trace(LMObjective(tiny(), 8)) != key


def test_a_head_at_another_policy_and_a_subclass_trace_by_keys_of_their_own():
    """A head the fit ladder tiles (`tile_head`) traces by another key than the whole head; a subclass,
    whose loss may read what its parent's key does not state, states none of its own (`stated_trace`)."""
    whole, tiled = LMObjective(tiny(), 8), LMObjective(tiny(), 8, head_tile="tiled")
    assert stated_trace(whole) != stated_trace(tiled)

    class Counted(LMObjective):
        def loss(self, variables, batch, step):
            return super().loss(variables, batch, step)

    assert stated_trace(Counted(tiny(), 8)) is None


def test_an_ema_of_another_decay_traces_apart_and_one_given_as_a_function_keeps_no_key():
    """The EMA a step keeps is the step's: a decay of its own traces apart (`EMASpec.stated`), and an EMA
    whose decay is a schedule given as a function states no value, so its objective keeps no key."""
    import dataclasses

    import optax

    slow, fast = LMObjective(tiny(), 8, ema_decay=0.99), LMObjective(tiny(), 8, ema_decay=0.9)
    assert stated_trace(slow) != stated_trace(fast) != stated_trace(LMObjective(tiny(), 8))
    scheduled = LMObjective(tiny(), 8, ema_decay=0.99)
    decay = optax.linear_schedule(0.9, 0.99, 10)
    scheduled.ema = dataclasses.replace(scheduled.ema, decay=decay, stated=None)
    assert stated_trace(scheduled) is None


def test_a_settings_value_holds_its_mappings_in_order_and_none_for_a_function():
    """A step's settings are kept by value (`frozen_value`): an optimizer config's option mapping in any
    order alike, its options apart from another's, and a setting that is a function, none."""
    from dew.objectives.base import UNKEYED, frozen_value
    from dew.optim import OptimConfig

    one = OptimConfig(optimizer_opts={"b1": 0.8, "b2": 0.9})
    assert frozen_value(one) == frozen_value(OptimConfig(optimizer_opts={"b2": 0.9, "b1": 0.8}))
    assert frozen_value(one) != frozen_value(OptimConfig(optimizer_opts={"b1": 0.8, "b2": 0.95}))
    assert hash(frozen_value(one)) is not None
    assert frozen_value((one, lambda batch: batch)) is UNKEYED


@pytest.mark.mesh(devices=4)
def test_a_refit_on_a_sequence_split_mesh_reads_the_step_its_first_fit_traced():
    """A trainer whose mesh splits the sequence fits again in the same process tracing what a trainer on a
    mesh of no split axis traces again, its step and its plan's candidates read as its first fit traced
    them: its key holds the links' measured figures (`dew.nn.sharding.trace_context`) and not the links its
    last compile decided, and each fit plans from the split the trainer was given, both of which keyed every
    refit on a tensor- or sequence-split mesh apart from the fit before it (row 6022); the kept step's
    spreads are its trace's (`Candidate.spread`)."""
    import jax.monitoring
    import numpy as np

    from dew.data.dataset import Loader
    from dew.optim import OptimConfig
    from dew.training import MeshSpec, Trainer

    traced = []

    def heard(event: str, seconds: float, **_) -> None:
        if event == "/jax/core/compile/jaxpr_trace_duration":
            traced.append(seconds)

    rows = {"text": np.random.default_rng(0).integers(0, 8, size=(8, 17)).astype(np.int32)}

    def refit(mesh: MeshSpec) -> tuple[dict, dict, int]:
        """The links each of two fits decided, and the programs JAX traced again for the second."""
        # The optimizer stated as a config, which keys the step.
        optimizer = OptimConfig(optimizer="adam", learning_rate=1e-2)
        trainer = Trainer(LMObjective(tiny(max_seq_len=16), 16), optimizer, key=jax.random.key(0), mesh=mesh)
        fitted = []
        for _ in range(2):
            before = len(traced)
            data = Loader(train=lambda partition: iter([rows] * 4), val=None, records=None, batch=8)
            trainer.fit(data, steps=2)
            fitted.append((dict(trainer.links), len(traced) - before))
        return fitted[0][0], fitted[1][0], fitted[1][1]

    jax.monitoring.register_event_duration_secs_listener(heard)
    try:
        _, _, plain = refit(MeshSpec(fsdp=4))
        first, again, split = refit(MeshSpec(fsdp=2, sequence=2))
    finally:
        jax.monitoring.unregister_event_duration_listener(heard)
    assert split == plain
    assert {axis: link.spread for axis, link in again.items()} == {
        axis: link.spread for axis, link in first.items()}
