"""Fine-tune a decision model on a table of labelled text, as `laya-train --data tickets.csv` does
(`dew.decision.config.DecisionRunConfig`).

    python recipes/decision/train.py --data.path tickets.csv --data.text text --data.label label \\
        --trainer.batch-size 32 --trainer.epochs 4 --optim.learning-rate 2.5e-5
    python recipes/decision/train.py --pretrained Qwen/Qwen3-0.6B --lora.rank 16 \\
        --lora.modules q_proj v_proj --data.path PolyAI/banking77 --data.validation-split test \\
        --data.question intent --trainer.batch-size 32 --trainer.steps 2000

The loss is the log loss unless the objective names another rule, or a
weighted sum of rules (`--objective.loss '{"class": "dew.decision.scoring:Combined",
"fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}'`).
"""

from dew.decision.config import DecisionRunConfig

if __name__ == "__main__":
    DecisionRunConfig.cli().run()
