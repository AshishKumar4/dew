"""A backbone and a decision head, applied as two modules over one variables tree."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from dew.decision.head import Head
from dew.decision.layout import DecisionInputs
from dew.nn.protocols import AffineHead, HiddenStates, OutputTable
from dew.objectives.base import Variables, part, thaw


@runtime_checkable
class Ordered(Protocol):
    """A model that says whether it reads its tokens in order, each after the
    ones before it, which decides where a layout puts its markers."""

    @property
    def causal(self) -> bool: ...


@dataclass(frozen=True)
class DecisionModel:
    """Any backbone with final states (`HiddenStates`), read by a decision head.

    The variables hold the two under `backbone` and `head` in every collection
    (`dew.objectives.base.joined`). Each is applied as its own module over its
    part, so a backbone that `Pretrained.adapt` gave LoRA factors runs with its
    own adapter, with its base weights under `frozen`.

    In the parallel layout an option attends to no other option's tokens, and the
    backbone's attention windows measure distance in positions
    (`attention_pairwise_mask`, `attention_key_positions`).
    """

    backbone: nn.Module
    head: Head

    def __post_init__(self):
        if not isinstance(self.backbone, HiddenStates):
            raise TypeError(f"a decision head reads its backbone's final states (`HiddenStates`), "
                            f"which a {type(self.backbone).__name__} does not give")

    def states(self, variables: Variables, inputs: DecisionInputs, *, train: bool = False,
               rngs: dict[str, jax.Array] | None = None) -> jax.Array:
        """Return the backbone's `[B, L, features]` final states."""
        backbone = thaw(part(variables, "backbone"))
        method = "hidden_states"
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
        """Return the backbone's output table as `[V, D]` rows, which a head may read (`AffineHead`)."""
        if not isinstance(self.backbone, AffineHead):
            raise TypeError(f"the head reads its backbone's output table (`AffineHead`), which a "
                            f"{type(self.backbone).__name__} does not give")
        table = self.backbone.apply(thaw(part(variables, "backbone")), method="output_table")
        if not isinstance(table, OutputTable):
            raise ValueError("the head reads its backbone's output table, and no matrix alone gives "
                             "this backbone's logits")
        return table.matrix if table.vocab_major else table.matrix.T

    @staticmethod
    def head_size(backbone: nn.Module) -> tuple[int, Dtype | None]:
        """Return the width of `backbone`'s final states, and the dtype a head over them computes in.

        The backbone is traced, not run, so this costs no compute. A backbone that
        computes in reduced precision has its head compute in it too; over
        full-precision states a head computes at its parameters' precision.
        """
        tokens = jnp.zeros((1, 2), jnp.int32)

        def states(key: jax.Array) -> jax.Array:
            return backbone.init_with_output(key, tokens, method="hidden_states")[0]

        shape = jax.eval_shape(states, jax.random.key(0))
        reduced = jnp.finfo(shape.dtype).bits < 32
        return shape.shape[-1], shape.dtype if reduced else None

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
