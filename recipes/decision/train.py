"""Fine-tune a decision model on a table of labelled text (`dew.decision.config.DecisionRunConfig`).

    python recipes/decision/train.py --trainer.epochs 3
    python recipes/decision/train.py --pretrained Qwen/Qwen3-0.6B --lora.rank 16 \\
        --lora.modules q_proj v_proj --optim.learning-rate 1e-4 --trainer.steps 2000

The default table is BANKING77's training split: each row's `text` is the state and its
`label_text` the answer to one `intent` question over its 77 intents. A tenth of the rows, by
the hash of each, are held out; validation scores them, and after training they fit the
temperatures saved into the run. A table of your own is a run file whose `run` names its chains,
`dew train tickets.py` (docs/guides/decision-models.md).

The loss is the log loss unless the objective names another rule, or a
weighted sum of rules (`--objective.loss '{"class": "dew.decision.scoring:Combined",
"fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}'`).
"""

from dew.data import Hub
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig

TRAIN, HELD = Hub("mteb/banking77", revision="18072d2685ea682290f7b8924d94c62acc19c0b2").holdout(0.1)


def asked(rows):
    """`rows` as decisions, read whole so the options are the whole table's labels."""
    return rows.pipe(decisions, label="label_text", question="intent",
                     instructions="Which banking request is this?").read("memory")


if __name__ == "__main__":
    DecisionRunConfig.cli(default=DecisionRunConfig(data=asked(TRAIN).shuffle().repeat().batch(32),
                                                    validation=asked(HELD).batch(32))).run()
