"""Measurement script (never lands): the seeds `reference_error.seeds_for` sizes for the stability family at
index argv[1] of the suite's families in order, from FEWEST_SEEDS, as one JSON line."""
import json
import sys

import jax
import test_attention_stability as suite
from reference_error import seeds_for, with_gradients

name = list(suite.FAMILIES)[int(sys.argv[1])]
family = suite.FAMILIES[name]
draw = suite._cast(family, family.draw)
count = seeds_for(jax.jit(with_gradients(suite._reference(family, anchored=False))), draw, name,
                  truth_fn=with_gradients(suite._reference(family, anchored=True)))
print(json.dumps({"family": name, "declared": family.seeds, "sized": count}))
