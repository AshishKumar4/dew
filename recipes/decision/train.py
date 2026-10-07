"""Fine-tune a decision model on a table of labelled text, as `laya-train --data tickets.csv` does.

    python recipes/decision/train.py --data.path tickets.csv --data.text text --data.label label \\
        --trainer.batch-size 32 --trainer.epochs 4 --optim.learning-rate 2.5e-5

By default the run starts from Laya's released English checkpoint, encoder
and head. `--pretrained` names another: a Laya-style repository (one with
an `rl_agent_config.json`), or any Hugging Face model, whose head is then
drawn fresh, with `--lora.rank 16 --lora.modules q_proj v_proj` to train an
adapter in place of the whole model:

    python recipes/decision/train.py --pretrained Qwen/Qwen3-0.6B --lora.rank 16 \\
        --lora.modules q_proj v_proj --data.path PolyAI/banking77 --data.validation-split test \\
        --data.question intent --trainer.batch-size 32 --trainer.steps 2000

The loss is the log loss unless the objective names another rule, or a
weighted sum of rules (`--objective.loss '{"class": "dew.decision.scoring:Combined",
"fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}'`).
A tenth of the rows, at most 400,
are held out (`--data.held-out`); validation scores them, and after
training they fit the temperatures the task divides its logits by, saved
into the run, so `Decide.from_run` and `dew.pipeline` answer calibrated.
A question type's temperature needs ten held-out answers and a bucket's
2000, Laya's floors (`dew.decision.config.DecisionRunConfig`).
"""

from dew.decision.config import DecisionRunConfig

if __name__ == "__main__":
    DecisionRunConfig.cli().run()
