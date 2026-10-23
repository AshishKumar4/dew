"""The gated delta net's real-shape cases, by the model whose layer they are: the one table the compile tests
(fla_gdn_support) and dew-ops' kernel lab, which loads this file from a job's checkout by path, read. Plain
Python, no import: a case is its operands' dimensions and the chunk the rule runs at."""

CASES = {
    "0.8b": {"batch": 8, "length": 3072, "key_heads": 16, "value_heads": 16,
             "width": 128, "columns": 128, "chunk": 64},
    "9b": {"batch": 4, "length": 3072, "key_heads": 16, "value_heads": 32,
           "width": 128, "columns": 128, "chunk": 64},
    "27b": {"batch": 1, "length": 4096, "key_heads": 16, "value_heads": 48,
            "width": 128, "columns": 128, "chunk": 64},
    "wide keys": {"batch": 1, "length": 512, "key_heads": 2, "value_heads": 4,
                  "width": 256, "columns": 128, "chunk": 64},
    "tiny": {"batch": 1, "length": 128, "key_heads": 1, "value_heads": 2,
             "width": 64, "columns": 64, "chunk": 64},
}
"""Qwen3.5-0.8B's, 9B's and 27B's gated delta nets at their training rows and tokens, whose heads are
`fla_gdn.WIDEST` wide and 27B's three value heads a key head (no power of two); the widest keys the port takes
in 16 bits (`fla_gdn.WIDEST_KEYS`); and a CPU test's."""
