"""Fine-tune a decision model on BANKING77, then answer and score it.

The script scores Laya's released checkpoint on BANKING77's 77 intents
zero-shot, fine-tunes it on the training split, fits its temperatures on
held-out rows, and scores the test split again. `--backbone qwen` instead
trains a LoRA adapter on Qwen3-0.6B under a fresh decision head.

    python examples/banking77_decisions.py --backbone laya --epochs 3 --run runs/laya-banking77
    python examples/banking77_decisions.py --backbone qwen --epochs 3 --learning-rate 1e-4 \\
        --run runs/qwen-banking77

Each score is one JSON line, on stdout and appended to `--out`: accuracy,
expected calibration error, the area under the risk-coverage curve and the
log loss. The run checkpoints into `--run` every ten minutes and resumes
from it when started again.
"""

import datetime
import json
from dataclasses import dataclass, replace
from typing import Literal

import tyro

from dew import Trainer
from dew.checkpoints import Checkpoints
from dew.config import OptimConfig, TrainerConfig
from dew.decision import (
    AURC,
    ECE,
    Accuracy,
    Brier,
    Decide,
    DecisionObjective,
    Example,
    LogLoss,
    StateFirstLayout,
    decisions,
)
from dew.interop import Pretrained
from dew.lora import LoRA

LAYA = ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148")
QWEN = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
BANKING77 = ("mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2")


@dataclass(frozen=True)
class Options:
    backbone: Literal["laya", "qwen"] = "laya"
    epochs: int = 3
    batch: int = 32
    learning_rate: float = 2.5e-5
    """Laya's encoder rate; an adapter on Qwen trains at 1e-4."""
    eval_every: int = 100
    run: str = "runs/banking77"
    out: str = "banking77.jsonl"


def report(out: str, line: dict) -> None:
    print(json.dumps(line), flush=True)
    with open(out, "a") as file:
        file.write(json.dumps(line) + "\n")


def main(options: Options) -> None:
    from dew.data import Hub, Memory
    from dew.data.compile import iterate

    def examples(rows, **settings) -> list[Example]:
        """The table's rows as one intent question each, read whole; each Hub split is downloaded
        and prepared the first time it is read."""
        asked = rows.pipe(decisions, label="label_text", question="intent",
                          instructions="Which banking request is this?", **settings)
        return [Example.of(row["example"]) for row in iterate(asked, read="memory", provision=True)]

    def table(rows: list[Example]):
        return Memory([{"example": row} for row in rows]).pipe(decisions).read("memory")

    # A tenth of the training split, by the hash of each row, is held out, at most 400 rows of it.
    training, held = Hub(BANKING77[0], revision=BANKING77[1]).holdout(0.1)
    train, held_out = examples(training), examples(held)[:400]
    # The test split asks the same 77 options, in the training split's order.
    intents = train[0].questions["intent"].options
    test = examples(Hub(BANKING77[0], revision=BANKING77[1], split="test"), options=tuple(intents))

    if options.backbone == "laya":
        laya = Decide.from_pretrained(LAYA[0], revision=LAYA[1], dtype="bfloat16")
        report(options.out, {"backbone": "laya", "stage": "released", "test": laya.score(test)})
        # 77 options at Laya's 192-token budget keep three or four tokens each.
        laya = replace(laya, layout=replace(laya.layout, max_len=768, head_max_len=640))
        report(options.out, {"backbone": "laya", "stage": "zero-shot, 640-token head",
                             "test": laya.score(test)})
        objective = DecisionObjective(laya, loss=LogLoss() + 0.5 * Brier())
    else:
        adapter = LoRA(rank=16, modules=("q_proj", "v_proj"))
        # The adapter keeps Qwen's weights frozen, so they load where the trainer's start keeps them.
        placement = TrainerConfig().source_placement(frozen=True)
        qwen = Pretrained.load(QWEN[0], revision=QWEN[1], **placement).adapt(adapter, key=0)
        layout = StateFirstLayout(max_len=512, head_max_len=448, option_tokens=12)
        objective = DecisionObjective(qwen, loss=LogLoss() + 0.5 * Brier(), layout=layout)

    checkpoints = Checkpoints(options.run, keep=1)
    state = Trainer(objective, OptimConfig(learning_rate=options.learning_rate), key=0,
                    checkpoints=checkpoints).fit(
        table(train).shuffle().repeat().batch(options.batch), epochs=options.epochs,
        validation=table(held_out[:options.batch * 8]).batch(options.batch), eval_every=options.eval_every,
        checkpoint_every=datetime.timedelta(minutes=10), metrics=[Accuracy(), ECE(), AURC(), LogLoss()])
    checkpoints.wait()

    decide = objective.pipeline(state)
    report(options.out, {"backbone": options.backbone, "stage": "fine-tuned", "test": decide.score(test)})
    calibrated = decide.calibrated(held_out)
    calibrated.save(options.run)
    report(options.out, {"backbone": options.backbone, "stage": "fine-tuned, calibrated",
                         "temperatures": dict(calibrated.calibration.temperatures.types),
                         "test": calibrated.score(test)})
    report(options.out, calibrated.systemone({
        "state": "I was charged twice for the same card payment yesterday",
        "questions": {"intent": {"type": "choice", "instructions": "Which banking request is this?",
                                 "criteria": ["card_payment_wrong_exchange_rate", "transaction_charged_twice",
                                              "card_arrival"]}}}))


if __name__ == "__main__":
    main(tyro.cli(Options))
