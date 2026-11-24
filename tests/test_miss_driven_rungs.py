"""A fit search reads its first miss's live set (`jax.stages.Compiled.peak_live`) to choose what it asks
next (row 12463): the joined Newton-Schulz stacks parted first only where their copies could cover what the
step was short of, else after the remat rungs, and the remat rung whose dropped values cover it asked first.
A 4B decision step on 2 x H100 asked the stacks' release for 0.1 GB of a 9.9 GB shortfall, 290 s a compile,
before climbing one rung at a time. Compiled for compile-only H100s on the CPU, never run; each search read
by the probes its plan or refusal records."""

import jax
import numpy as np
import pytest

from dew.lora import LoRA
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.kernels.generation import compile_only_gpus
from dew.objectives.lm import LMObjective
from dew.optim import Adam, Muon, OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import AheadOfTime, compile_ahead
from dew.training.memory import least_limit
from dew.training.planning import PlanRefused


def _ahead(model, optimizer, seq: int, rows: int, limit: int, variables=None
           ) -> tuple[AheadOfTime | None, list[dict]]:
    """`model`'s step compiled ahead for one compile-only H100 of `limit` bytes over `rows` rows of `seq`
    tokens, its `variables` where given, None where it fit none, and each probe its search compiled, in
    order."""
    mesh = MeshSpec().build(compile_only_gpus("sm90", 1, limit))
    built = OptimConfig(optimizer=optimizer, learning_rate=1e-3).build(100)
    objective = LMObjective(model, seq) if variables is None else LMObjective(model, seq, variables=variables)
    trainer = Trainer(objective, built, key=jax.random.key(0), mesh=mesh, layout=Layout(min_shard=256))
    try:
        ahead = compile_ahead(trainer, {"text": np.zeros((rows, seq + 1), np.int32)})
    except PlanRefused as refused:
        return None, [probe.record() for probe in refused.probes]
    return ahead, list(ahead.plan["probes"])


def _needed(ahead: AheadOfTime) -> int:
    """The least an H100 holds for `ahead`'s step to be admitted (`least_limit`)."""
    gpu = compile_only_gpus("sm90", 1, 2 ** 34)[0]
    beside = ahead.output_bytes - ahead.alias_bytes
    return least_limit(gpu, [(ahead.argument_bytes, beside, ahead.temp_bytes)])


def test_a_miss_whose_joined_stacks_hold_less_than_it_was_short_of_parts_them_after_the_remat_rungs():
    """A step that misses its devices, whose joined Newton-Schulz stacks hold less at its peak than it was
    short of, asks no compile parting them (which frees at most their copies) before the remat rungs: it
    climbs them first, and, where none fits, parts the stacks after them before it refuses the step."""
    model = models.build("causal_transformer", vocab_size=1024, emb_features=512, num_layers=2, num_heads=8,
                         num_kv_heads=2, head_dim=64, mlp_features=2048, max_seq_len=512, dtype="bfloat16",
                         attention_impl="xla")
    ahead, probes = _ahead(model, Muon(), 512, 16, 2 ** 27)
    climbed = next(at for at, probe in enumerate(probes) if probe["remat"] is not None)
    assert all(probe["ns_group"] is None for probe in probes[:climbed]), probes
    assert ahead is not None or any(probe["ns_group"] == 1 for probe in probes), probes


@pytest.mark.parametrize("stack", ["plain", "scanned", "rows"])
def test_a_search_asks_first_the_rung_its_miss_reads_as_dropping_what_it_was_short_of(stack):
    """A step of one row, which no split shrinks, whose MLPs' forward values hold most of its temporaries,
    that misses its devices at no remat on an H100 between what 'save_dot_except_mlp' needs and what
    'minimal' needs, asks 'save_dot_except_mlp' next, the rung its miss reads as dropping what it was short
    of (`RecomputeStatement`), passing over 'minimal', which drops too little, and is admitted there: two
    compiles, where the climb a rung at a time compiled 'minimal' between them; its stack a layer at a time,
    scanned, or scanned reading a frozen base's rows where they lie under a LoRA adapter
    (`decoder_stack._rows_recomputed`), whose body's values the rung's policy keeps or drops as an unscanned
    block's (`dew.nn.protocols.RUNG_SCOPE`)."""
    model = models.build("causal_transformer", vocab_size=256, emb_features=256, num_layers=4, num_heads=4,
                         num_kv_heads=2, head_dim=64, mlp_features=8192, max_seq_len=1024, dtype="bfloat16",
                         attention_impl="xla", scan_layers=stack != "plain")

    def ahead_at(remat: str | None, limit: int) -> tuple[AheadOfTime | None, list[dict]]:
        at = model if remat is None else model.clone(remat=REMAT_POLICIES[remat])
        if stack != "rows":
            return _ahead(at, Adam(), 1024, 1, limit)
        base = at.init(jax.random.key(0), np.zeros((1, 1024), np.int32))
        adapter = LoRA(rank=2, modules=("q_proj",)).apply(at, base, key=1)
        return _ahead(adapter.model, Adam(), 1024, 1, limit, adapter.variables)

    rung = {name: ahead_at(name, 2 ** 34)[0] for name in ("minimal", "save_dot_except_mlp")}
    assert all(ahead is not None for ahead in rung.values()), rung
    lighter, dropping = (_needed(ahead) for ahead in rung.values() if ahead is not None)
    # Its devices between the two rungs' needs: 'minimal' misses there, 'save_dot_except_mlp' fits.
    assert dropping < lighter, (dropping, lighter)
    ahead, probes = ahead_at(None, (dropping + lighter) // 2)
    assert ahead is not None and ahead.plan["remat"] == "save_dot_except_mlp", probes
    asked = [probe["remat"] for probe in probes]
    assert asked == [None, "save_dot_except_mlp"], probes
    assert [probe["fits"] for probe in probes] == [False, True], probes
