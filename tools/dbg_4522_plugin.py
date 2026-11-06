"""Diagnostic test hook for the CPU slot-scoring experiment."""
import os


def pytest_configure():
    import jax
    import jax.numpy as jnp
    import reference_error
    from dew.objectives.lm.adaptation import _Chain

    reference_error.WITHOUT_YNN = {}
    if os.environ.get("DEW_DBG_TOKEN") != "1":
        return
    original = _Chain.scored

    def scored(self, versions, chosen, states, targets, weights):
        def score(slot):
            at, state, target, weight = slot
            return original(self, versions, [which[:, None] for which in at], state[:, None],
                            target[:, None], weight[:, None])
        slots = (tuple(which.swapaxes(0, 1) for which in chosen), states.swapaxes(0, 1),
                 targets.swapaxes(0, 1), weights.swapaxes(0, 1))
        result = jax.lax.map(score, slots)
        return jax.tree.map(lambda value: jnp.squeeze(value, axis=2).swapaxes(0, 1), result)

    _Chain.scored = scored
