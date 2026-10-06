"""The tiny decoder several model tests share.

`TINY_DECODER` is the geometry alone: each test file names its own vocabulary
and adds any field its claim turns on, such as grouped key/value heads.
"""

TINY_DECODER = {"emb_features": 32, "num_layers": 2, "num_heads": 4, "mlp_features": 64,
                "max_seq_len": 16}
"""A causal transformer 32 wide, two layers of four heads, without its vocabulary."""

