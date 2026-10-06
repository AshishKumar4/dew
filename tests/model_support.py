"""The tiny decoder and the parameter-tree view several model tests share."""

import jax

TINY_DECODER = {"emb_features": 32, "num_layers": 2, "num_heads": 4, "mlp_features": 64, "max_seq_len": 16}
"""A causal transformer 32 wide, two layers of four heads; each test file adds
its own vocabulary and any field its claim turns on."""


def flat_tree(tree) -> dict:
    """`tree`'s leaves by their dictionary keys joined with dots, the way a
    checkpoint names them: 'params.layers_0.mlp.gate.kernel'."""
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}
