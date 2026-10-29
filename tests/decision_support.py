"""Decision rows as a run reads them, for the tests of decision objectives and their packing."""

from __future__ import annotations

import jax


def decision_rows(examples, batch: int | None, validation=None, *, objective=None, tokens: int | None = None):
    """`examples` as a run's decision batches of `batch` rows (`dew.decision.decisions`), or of `tokens`
    tokens, read in memory, with `validation`'s examples as its held-out split, compiled with
    `objective`'s stages (`dew.data.compile.compile`)."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.decision import decisions

    def chain(rows):
        parsed = Memory([{"example": example} for example in rows]).pipe(decisions)
        return parsed.batch(tokens=tokens) if tokens is not None else parsed.batch(batch)

    held = None if validation is None else chain(validation)
    return compile(chain(examples), validation=held, read="memory", objective=objective)


ALGEBRAIC_ZEROS = ("to_k",)
"""The joint heads' modules whose bias gradient is zero by algebra, whatever reads the logits: a key's bias
shifts every key's logit of a query alike, which the attention's softmax does not move with. Their float64
values are residue, so a check-1 truth holds them exactly zero."""

LOSS_ZEROS = ("scorer_out",)
"""The modules whose bias gradient is zero by algebra only under a loss over each question's softmax: the
scorer's shared bias shifts every option's logit alike, which the loss does not move with, where the
logits' own gradient under another cotangent it does."""


def algebraically_zero(name: str, *, loss: bool = False) -> bool:
    """Whether the gradient leaf `name` (`jax.tree_util.keystr`) is an `ALGEBRAIC_ZEROS` module's bias, or,
    under the options' loss (`loss`), a `LOSS_ZEROS` one's."""
    modules = (*ALGEBRAIC_ZEROS, *(LOSS_ZEROS if loss else ()))
    return name.endswith("['bias']") and any(f"['{module}']" in name for module in modules)


def zeroed(tree, *, loss: bool = False):
    """`tree` with the algebraic zeros' bias gradients exactly zero (`algebraically_zero`)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf * 0 if algebraically_zero(jax.tree_util.keystr(path), loss=loss) else leaf,
        tree)
