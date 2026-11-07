"""DeepSeek sparse attention's reference selections: whether one a tie decides.

A lightning indexer scores a key `sum_h w_h relu(q_h . k)`, exactly 0 where every head's relu zeroes it,
and selects each query's `index_topk` best keys. Where the cut falls among equal scores, torch's `topk`
promises no tie order and Dew chooses the lower key index (`dew.nn.sparse_selection.top_k_selection`), so
the two select different keys and neither is the reference's function: a check over such a prompt
compares two tie rules, not two implementations.
"""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def indexer_rankings(model: torch.nn.Module, indexer: type[torch.nn.Module]):
    """The scores each `indexer` module of `model` ranks while the block runs, with the count it keeps:
    every `Tensor.topk` taken inside one, as `(scores, k)`."""
    taken: list[tuple[np.ndarray, int]] = []
    inside: list[torch.nn.Module] = []
    original = torch.Tensor.topk

    def topk(scores, k, *args, **kwargs):
        if inside:
            taken.append((scores.detach().float().numpy(), int(k)))
        return original(scores, k, *args, **kwargs)

    def entered(module, inputs) -> None:
        inside.append(module)

    def left(module, inputs, output) -> None:
        # A forward hook that returns a value replaces the module's output.
        inside.pop()

    hooks = [handle for module in model.modules() if isinstance(module, indexer)
             for handle in (module.register_forward_pre_hook(entered), module.register_forward_hook(left))]
    torch.Tensor.topk = topk
    try:
        yield taken
    finally:
        torch.Tensor.topk = original
        for handle in hooks:
            handle.remove()


def tied_cuts(taken: list[tuple[np.ndarray, int]]) -> int:
    """How many queries' selections a tie decides: the `k`-th best of their allowed (finite) scores equal
    to the next one."""
    tied = 0
    for scores, k in taken:
        ranked = -np.sort(-scores, axis=-1)
        if ranked.shape[-1] <= k:
            continue
        kth, after = ranked[..., k - 1], ranked[..., k]
        tied += int(np.sum(np.isfinite(after) & (kth == after)))
    return tied
