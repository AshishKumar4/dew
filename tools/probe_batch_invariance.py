"""Score a fixed decision row before and after right-padding it."""

import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.decision import Choice, Decide, DecisionInputs


def main():
    fixtures = Path.cwd() / "tests" / "fixtures"
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)
        backbone = fixtures / "hf" / "qwen38-dense-tiny"
        joint = fixtures / "joint_checkpoint" / "tiny"
        for item in [*backbone.iterdir(), joint / "joint_head_config.json",
                     joint / "joint_head.safetensors"]:
            (source / item.name).symlink_to(item.resolve())
        task = Decide.from_pretrained(source, attention_impl="xla", dtype="bfloat16")
        short = ("refund", {"q": Choice("team", ["pay", "tech", "sales"])})
        long = ("invoice " * 20, {"q": Choice("which department", ["billing", "technical", "sales",
                                                                 "support", "account", "other"])})
        rows = [task.layout.rows(task.tokenizer, task.specials, state, questions)[0]
                for state, questions in (short, long)]
        single = DecisionInputs.collate(rows[:1], task.specials.pad)
        mixed = DecisionInputs.collate(rows, task.specials.pad)
        padded = jax.tree.map(lambda leaf: leaf[:1], mixed)
        forward = jax.jit(task.model.logits)
        found = {}
        passes = [("alone", single), ("padded", padded), ("mixed", mixed)]
        if "--one-pad" in sys.argv:
            passes.append(("one-pad", replace(single, tokens=jnp.pad(single.tokens, ((0, 0), (0, 1))),
                                               segments=jnp.pad(single.segments, ((0, 0), (0, 1))))))
        for name, inputs in passes:
            logits = np.asarray(forward(task.variables, inputs))[0, 0, :3]
            p = task.calibration.probabilities(short[1]["q"], logits)
            found[name] = {"logits": logits.tolist(), "probabilities": p.tolist()}
        found["lengths"] = [len(row.tokens) for row in rows]
        found["token_digests"] = [hashlib.sha256(np.asarray(row.tokens, np.int32).tobytes()).hexdigest()
                                  for row in rows]
        print("BATCH_INVARIANCE", json.dumps(found), flush=True)


if __name__ == "__main__":
    main()
