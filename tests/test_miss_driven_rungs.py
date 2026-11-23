"""A fit search reads its first miss's live set (`jax.stages.Compiled.peak_live`) to choose what it asks
next (row 12463): the joined Newton-Schulz stacks parted only where their copies could cover what the step
was short of, and the remat rung whose dropped values cover it asked first, the lighter ones after. A 4B
decision step on 2 x H100 asked the stacks' release for 0.1 GB of a 9.9 GB shortfall, 290 s a compile,
before climbing one rung at a time. Compiled for compile-only H100s on the CPU, never run."""

import logging

import jax
import numpy as np
import pytest

from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.lm import LMObjective
from dew.optim import Muon, OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead

SEQ = 512


def _hybrid():
    """A gated delta net's layer beside attention, its MLP wide, in bf16: what its remat rungs keep differs
    by name (`REMAT_POLICIES`)."""
    return models.build(
        "causal_transformer", vocab_size=1024, emb_features=512, num_layers=2, num_heads=8, num_kv_heads=2,
        head_dim=64, mlp_features=2048, max_seq_len=SEQ, dtype="bfloat16", attention_impl="xla",
        layer_types=("linear_attention", "attention"),
        kinds={"attention": {}, "linear_attention": {"mixer": {"class": "gated_delta_net", "fields": {
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
            "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4}}}})


@pytest.mark.parametrize("limit", [2 ** 27], ids=["starved"])
def test_a_search_asks_first_the_rung_its_miss_reads_as_dropping_what_it_was_short_of(caplog, limit):
    """A step that misses its devices at every split asks no release of its joined Newton-Schulz stacks,
    whose copies hold less than it was short of, and climbs to the rung its miss reads as dropping enough
    before any lighter one: the first rung it recomputes more at is the rung it fits at, or, where XLA
    schedules that rung's peak otherwise, the search says what moved and climbs on (row 12463)."""
    mesh = MeshSpec().build(compile_only_gpus("sm90", 1, limit))
    optimizer = OptimConfig(optimizer=Muon(), learning_rate=1e-3).build(100)
    trainer = Trainer(LMObjective(_hybrid(), SEQ), optimizer, key=jax.random.key(0), mesh=mesh,
                      layout=Layout(min_shard=256))
    with caplog.at_level(logging.INFO, "dew.training.planning"):
        ahead = compile_ahead(trainer, {"text": np.zeros((16, SEQ + 1), np.int32)})
    said = [record.getMessage() for record in caplog.records if record.name == "dew.training.planning"]
    assert not [line for line in said if "Newton-Schulz stacks joined; compiling it again" in line], said
    climbed = [line for line in said if "recomputing more at" in line]
    read = [line for line in said if "the step's miss reads first" in line]
    assert read and climbed, said
    moved = [line for line in said if "XLA scheduled its peak otherwise" in line]
    first = climbed[0]
    assert moved or ahead.plan["remat"] in first, (ahead.plan["remat"], said)
