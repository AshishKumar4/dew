"""Fine-tune a decision model on a table of labelled text (`dew.decision.config.DecisionRunConfig`).

    python recipes/decision/train.py --trainer.epochs 3
    python recipes/decision/train.py --pretrained Qwen/Qwen3-0.6B --lora.rank 16 \\
        --lora.modules q_proj v_proj --optim.lr 1e-4 --trainer.steps 2000

The default table is BANKING77's training split: each row's `text` is the state and its
`label_text` the answer to one `intent` question over its 77 intents. A tenth of the rows, by
the hash of each, are held out; validation scores them, and after training they fit the
temperatures saved into the run. A table of your own is a run file whose `run` names its chains,
`dew train tickets.py` (docs/guides/decision-models.md).

The loss is the log loss unless the run's objective binds another rule, or a weighted sum of rules,
`ObjectiveConfig(functools.partial(DecisionObjective, loss=...))` with a `dew.decision.scoring.Combined`
of log loss and a half of Brier, in `default`.
"""

from dew.data import Hub
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig


def banking77():
    """BANKING77's training split at its pinned commit, a tenth of its rows held out."""
    return Hub("mteb/banking77", revision="18072d2685ea682290f7b8924d94c62acc19c0b2").holdout(0.1)


def asked(rows):
    """`rows` as decisions, read whole so the options are the whole table's labels."""
    return rows.pipe(decisions, label="label_text", question="intent",
                     instructions="Which banking request is this?").read("memory")


def default(train, held) -> DecisionRunConfig:
    """The recipe's run over a table's `train` and `held` rows, each a `text` and its `label_text`."""
    return DecisionRunConfig(data=asked(train).shuffle().repeat().batch(32), validation=asked(held).batch(32))


def main(argv: list[str] | None = None, table=None):
    """Run the recipe on `argv`, the process's own command line where None, over `table`'s training and
    held-out rows, BANKING77's where None, and return the run's last state."""
    return DecisionRunConfig.cli(argv, default=default(*(banking77() if table is None else table))).run()


if __name__ == "__main__":
    main()
