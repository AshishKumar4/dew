"""The training step of a decision run whose rows ask a request's questions over one copy of its state
(`QuestionLayout.shared_state`) against the same run with a row a question, on the same requests: ms a step,
the questions a step trains, ms a question and the tokens each step reads, both arms in one process.

    python state_once_step.py --out DIR [--width 1024 --layers 12 --state 1800 --questions 8 --requests 4]
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np
import optax

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Example, decisions
from dew.decision.head import LabelReadout
from dew.decision.layout import AnswerCue, Instructions, Labels, OptionLines, Specials, State, StateFirstLayout, Text
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.training import Trainer

SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
WORDS = ["card", "loan", "charge", "refund", "account", "transfer", "branch", "payment", "statement", "limit"]


def requests(count: int, state: int, questions: int, seed: int) -> list[Example]:
    """`count` requests, each a state of about `state` bytes of words and `questions` questions of four options,
    each answered."""
    rng = np.random.default_rng(seed)
    asked = {f"q{index}": Choice(f"Question {index}: which one fits the case best?",
                                 [f"option {index}.{option}" for option in range(4)]) for index in range(questions)}
    examples = []
    for _ in range(count):
        text = " ".join(rng.choice(WORDS, size=state // 6))
        examples.append(Example(text, asked, {name: question.options[int(rng.integers(4))]
                                              for name, question in asked.items()}))
    return examples


def arm(shared: bool, args, examples) -> dict:
    """One arm's step: its objective over a fresh backbone, its first batch of `windows`, compiled once and timed
    over `--steps` steps after `--warmup`."""
    labels = Labels(names=("A", "B", "C", "D"), variants=("{}",), cue="\nAnswer:")
    layout = StateFirstLayout(
        labels=labels, max_len=args.max_len,
        segments=(Text("STATE:\n"), State(), Instructions({"choice": "\n{}"}),
                  OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="", noul_first="false"),
                  AnswerCue(template="\nAnswer:", many="\nAnswer:")))
    if shared:
        layout = dataclasses.replace(layout, shared_state=args.questions)
    backbone = CausalTransformer(vocab_size=256, emb_features=args.width, num_layers=args.layers,
                                 num_heads=args.width // 64, mlp_features=4 * args.width, max_seq_len=args.max_len,
                                 dtype=args.dtype)
    width, dtype = DecisionModel.head_size(backbone)
    objective = DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=SPECIALS, layout=layout,
                                  head=LabelReadout(features=width, label_ids=(), dtype=dtype, count=4),
                                  segments=args.segments)
    windows = args.shared_windows if shared else args.own_windows
    chain = Memory([{"example": example} for example in examples]).pipe(decisions).batch(windows)
    batch = next(iter(compile(chain, read="memory", objective=objective.data()).train(DataPartition())))
    trainer = Trainer(objective, optax.adamw(1e-5), key=jax.random.key(0))
    state = trainer.initial_state()
    started = time.perf_counter()
    step = trainer.compile(state, batch)
    compiled = time.perf_counter() - started
    for _ in range(args.warmup):
        state, *_ = step(state, batch)
    jax.block_until_ready(state)
    times = []
    for _ in range(args.steps):
        started = time.perf_counter()
        state, *_ = step(state, batch)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - started)
    questions = int(np.sum(np.any(np.asarray(batch["options"]), axis=-1)))
    segments = np.asarray(batch["segments"])
    ms = 1e3 * float(np.median(times))
    return {"arm": "shared" if shared else "own", "windows": int(segments.shape[0]),
            "window_tokens": int(segments.shape[1]), "real_tokens": int(np.count_nonzero(segments)),
            "questions": questions, "compile_s": round(compiled, 2), "ms_step_p50": round(ms, 3),
            "ms_step_p10": round(1e3 * float(np.percentile(times, 10)), 3),
            "ms_step_p90": round(1e3 * float(np.percentile(times, 90)), 3),
            "ms_question": round(ms / questions, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--state", type=int, default=1800, help="the state's bytes, a byte a token")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--requests", type=int, default=4, help="requests a step trains")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--segments", type=int, default=2, help="rows a window packs")
    parser.add_argument("--own-windows", type=int, default=16)
    parser.add_argument("--shared-windows", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    examples = requests(args.requests * 4, args.state, args.questions, seed=0)
    rows = [arm(shared, args, examples) for shared in (False, True)]
    report = {"device": jax.devices()[0].device_kind, "args": vars(args) | {"out": str(args.out)}, "arms": rows,
              "ms_question_ratio": round(rows[0]["ms_question"] / rows[1]["ms_question"], 3)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "state_once.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
