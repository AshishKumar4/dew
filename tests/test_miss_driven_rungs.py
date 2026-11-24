"""A fit search reads its first miss's live set (`jax.stages.Compiled.peak_live`) to choose what it asks
next (row 12463): the joined Newton-Schulz stacks parted only where their copies could cover what the step
was short of, and the remat rung whose dropped values cover it asked first, the lighter ones after. A 4B
decision step on 2 x H100 asked the stacks' release for 0.1 GB of a 9.9 GB shortfall, 290 s a compile,
before climbing one rung at a time. Compiled for compile-only H100s on the CPU, never run."""

import logging

import jax
import numpy as np

from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.lm import LMObjective
from dew.optim import Adam, Muon, OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead
from dew.training.planning import PlanRefused


def _ahead(model, optimizer, seq: int, rows: int, limit: int, caplog):
    """`model`'s step compiled ahead for one compile-only H100 of `limit` bytes over `rows` rows of `seq`
    tokens, with what the search said; None for the plan where it fit none."""
    mesh = MeshSpec().build(compile_only_gpus("sm90", 1, limit))
    built = OptimConfig(optimizer=optimizer, learning_rate=1e-3).build(100)
    trainer = Trainer(LMObjective(model, seq), built, key=jax.random.key(0), mesh=mesh,
                      layout=Layout(min_shard=256))
    caplog.clear()
    with caplog.at_level(logging.INFO, "dew.training.planning"):
        try:
            ahead = compile_ahead(trainer, {"text": np.zeros((rows, seq + 1), np.int32)})
        except PlanRefused:
            ahead = None
    return ahead, [record.getMessage() for record in caplog.records if record.name == "dew.training.planning"]


def test_a_miss_whose_joined_stacks_hold_less_than_it_was_short_of_parts_none_of_them(caplog):
    """A step that misses its devices, whose joined Newton-Schulz stacks hold less at its peak than it was
    short of, asks no compile parting them, which frees at most their copies, and climbs on."""
    model = models.build("causal_transformer", vocab_size=1024, emb_features=512, num_layers=2, num_heads=8,
                         num_kv_heads=2, head_dim=64, mlp_features=2048, max_seq_len=512, dtype="bfloat16",
                         attention_impl="xla")
    _, said = _ahead(model, Muon(), 512, 16, 2 ** 27, caplog)
    assert [line for line in said if "Newton-Schulz stacks hold" in line], said
    assert not [line for line in said if "Newton-Schulz stacks joined; compiling it again" in line], said


def test_a_search_asks_first_the_rung_its_miss_reads_as_dropping_what_it_was_short_of(caplog):
    """A step of one row, which no split shrinks, whose MLPs' forward values hold most of its temporaries,
    that misses its devices at no remat by a quarter of them asks first the rung its miss reads as dropping
    that many bytes of its forward's values (`RecomputeStatement`), before any lighter one: the first rung
    it compiles again at is that rung, at which it fits, or, where XLA schedules that rung's peak
    otherwise, says what moved and climbs on."""
    model = models.build("causal_transformer", vocab_size=256, emb_features=256, num_layers=4, num_heads=4,
                         num_kv_heads=2, head_dim=64, mlp_features=8192, max_seq_len=1024, dtype="bfloat16",
                         attention_impl="xla")
    whole, _ = _ahead(model, Adam(), 1024, 1, 2 ** 34, caplog)
    assert whole is not None and whole.plan["remat"] is None, whole and whole.plan
    beside = whole.argument_bytes + whole.output_bytes - whole.alias_bytes
    ahead, said = _ahead(model, Adam(), 1024, 1, beside + 3 * whole.temp_bytes // 4, caplog)
    read = [line for line in said if "the step's miss reads first" in line]
    climbed = [line for line in said if "recomputing more at" in line]
    assert read and climbed, said
    moved = [line for line in said if "XLA scheduled its peak otherwise" in line]
    assert moved or (ahead is not None and f'"remat": "{ahead.plan["remat"]}"' in climbed[0]), (said, ahead)
    assert ahead is None or ahead.plan["remat"] is not None, ahead.plan
