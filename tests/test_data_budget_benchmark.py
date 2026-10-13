"""The GPU plan shares Counted's recurrent real-token convention."""

import json
from pathlib import Path

from tools.benchmark_decision_budgets import model_flops


def test_real_qwen_budget_work_matches_the_approved_counted_reference():
    path = Path(__file__).parent / "fixtures/hf/qwen35-0.8b/config.json"
    config = json.loads(path.read_text())
    assert model_flops(config, 8 * 3072, 8 * 3072**2) == 81_095_156_563_968


def test_real_lengths_not_the_padded_batch_determine_model_work():
    path = Path(__file__).parent / "fixtures/hf/qwen35-0.8b/config.json"
    config = json.loads(path.read_text())
    split = model_flops(config, 512, 512**2) + model_flops(config, 1024, 1024**2)
    assert model_flops(config, 1536, 512**2 + 1024**2) == split
    assert split < model_flops(config, 2 * 3072, 2 * 3072**2)
