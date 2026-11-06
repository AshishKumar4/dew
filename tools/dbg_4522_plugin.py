"""Diagnostic test hook for the CPU slot-scoring experiment."""
import os
import contextlib


def pytest_configure():
    import jax
    import jax.numpy as jnp
    import reference_error
    from dew.objectives.lm.adaptation import _Chain

    reference_error.WITHOUT_YNN = {}
    if os.environ.get("DEW_DBG_USER_DEFAULT") == "1":
        os.environ["DEW_WIDE_CPU_ARITHMETIC"] = "0"
        jax.config.update("jax_default_matmul_precision", None)
        reference_error.summed_as_run = contextlib.nullcontext
        print("USER_SCOPE", "precision", jax.config.jax_default_matmul_precision,
              "wide_cpu", os.environ["DEW_WIDE_CPU_ARITHMETIC"], flush=True)
    if os.environ.get("DEW_DBG_POLICY_HIGHEST") == "1":
        import arm_arithmetic
        import dew.nn.precision
        arm_arithmetic.FP32_SUMS = "highest"
        dew.nn.precision.FP32_SUMS = "highest"
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
