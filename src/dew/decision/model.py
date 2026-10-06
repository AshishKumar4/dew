"""A backbone and a decision head, applied as two modules over one variables tree."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dew.decision.head import Head
from dew.decision.layout import DecisionInputs
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Variables, part, thaw


@dataclass(frozen=True)
class DecisionModel:
    """Any backbone with `hidden_states`, read by a decision head.

    The variables hold the two under `backbone` and `head` in every collection
    (`dew.objectives.base.joined`). Each is applied as its own module over its
    part, so a backbone that `Pretrained.adapt` gave LoRA factors runs with its
    own adapter, with its base weights under `frozen`.

    In the parallel layout an option attends to no other option's tokens, and the
    backbone's attention windows measure distance in positions
    (`attention_pairwise_mask`, `attention_key_positions`).
    """

    backbone: CausalTransformer
    head: Head

    def states(self, variables: Variables, inputs: DecisionInputs, *, train: bool = False,
               rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the backbone's `[B, L, features]` final states."""
        backbone = thaw(part(variables, "backbone"))
        method = type(self.backbone).hidden_states
        if inputs.positions is None or inputs.slots is None:
            return _array(self.backbone.apply(backbone, inputs.tokens, train=train,
                                              attention_mask=inputs.valid, method=method, rngs=rngs))
        slots = inputs.slots
        other = (slots[:, :, None] > 0) & (slots[:, None, :] > 0) & (slots[:, :, None] != slots[:, None, :])
        visible = inputs.valid[:, None, :] & ~other
        return _array(self.backbone.apply(backbone, inputs.tokens, train=train, positions=inputs.positions,
                                          attention_pairwise_mask=visible,
                                          attention_key_positions=inputs.positions, method=method, rngs=rngs))

    def table(self, variables: Variables) -> jax.Array:
        """Return the backbone's output table as `[V, D]` rows, which a head may read."""
        matrix, rows = self.backbone.head_table(thaw(part(variables, "backbone"))["params"])
        return matrix if rows else matrix.T

    def logits(self, variables: Variables, inputs: DecisionInputs, *, train: bool = False,
               rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the `[B, Q, K]` fp32 option logits, with -inf past each question's options."""
        states = self.states(variables, inputs, train=train, rngs=rngs)
        table = self.table(variables) if self.head.reads_table else None
        logits = _array(self.head.apply(thaw(part(variables, "head")), states, inputs, table, train=train,
                                        rngs=rngs))
        return jnp.where(inputs.options, logits, -jnp.inf)


def _array(applied: object) -> jax.Array:
    """A module's output when the call names no mutable collection."""
    if not isinstance(applied, jax.Array):
        raise TypeError(f"expected an array from the module, got {type(applied).__name__}")
    return applied
