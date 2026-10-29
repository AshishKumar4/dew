"""A loaded base's freshly drawn modules learn at their own rate (row 2391).

The toy is a gate diagnosis: a row of noise tokens ends in a query marker, and its next token is the
first token of a context only cross-attention reads. A four-layer decoder, its weights a loaded base,
frozen, gains cross-attention blocks before layers 1 and 3, which train under AdamW at a fine-tuning
rate.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.batch import freeze, merge, select, thaw
from dew.config import fresh_parameters
from dew.data.dataset import Loader
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.cross_attention import CrossAttention, crossing
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS
from dew.objectives.base import Aux, Objective
from dew.optim import FRESH_LEARNING_RATE_MULTIPLIER, OptimConfig, ParamGroup
from dew.training import Trainer

VOCAB, LENGTH, CONTEXT_TOKENS, BATCH, QUERY = 64, 8, 4, 32, 1
RATE = 1e-5
"""A pretrained base's fine-tuning rate, at which a fresh block, ungrouped, learns nothing."""


def rows(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Rows of noise tokens ending in the query marker, each with a context whose first token is the
    row's answer."""
    ids = rng.integers(2, VOCAB, (BATCH, LENGTH)).astype(np.int32)
    ids[:, -1] = QUERY
    context = rng.integers(2, VOCAB, (BATCH, CONTEXT_TOKENS)).astype(np.int32)
    return {"ids": ids, "context": context}


class Answered(Objective):
    """The cross-entropy of a row's last logits against its context's first token, the context read
    through the model's own token table, or none where `read` is False, starting from `variables`."""

    ema = None

    def __init__(self, model: CausalTransformer, variables):
        self.model, self.variables = model, variables

    def loss(self, variables, batch, step, read: bool = True):
        embedded = self.model.apply(variables, batch["context"], method="token_embeddings")
        lengths = jnp.full((batch["ids"].shape[0],), CONTEXT_TOKENS if read else 0, jnp.int32)
        logits = self.model.apply(variables, batch["ids"],
                                  conditioning={CONTEXT: embedded, CONTEXT_LENGTHS: lengths})[:, -1]
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["context"][:, 0])
        return losses.mean(), Aux({})


def loaded(seed: int = 0) -> Answered:
    """The toy's objective over a base drawn from `seed` as a checkpoint would hold it, frozen, and
    fresh cross-attention blocks before layers 1 and 3, as `Pretrained.with_cross_attention` adds them."""
    base = CausalTransformer(vocab_size=VOCAB, emb_features=64, num_layers=4, num_heads=4, num_kv_heads=2,
                             mlp_features=128, max_seq_len=16)
    tokens = jnp.zeros((1, LENGTH), jnp.int32)
    model = base.clone(cross_attention=CrossAttention((1, 3), CONTEXT_TOKENS))
    blocks = select(thaw(model.init(jax.random.key(seed + 1), tokens)),
                    lambda path: path[0] == "params" and crossing(path))
    return Answered(model, freeze(merge(base.init(jax.random.key(seed), tokens), blocks), crossing))


def test_a_loaded_bases_fresh_blocks_learn_at_the_fresh_rate_where_no_group_is_named():
    """At the base's fine-tuning rate, the default puts the blocks in a group of their own at
    `FRESH_LEARNING_RATE_MULTIPLIER` times it (`OptimConfig.with_fresh`), and the blocks learn to read
    the context: the held loss falls well below the no-context control's, the base's own guess (2.7
    against about 4.5 measured on this toy; ungrouped, the blocks learn nothing)."""
    objective = loaded()
    fresh = fresh_parameters(objective)
    assert fresh and all("cross_attention" in name for name in fresh)
    optimizer = OptimConfig(optimizer="adamw", learning_rate=RATE, weight_decay=0.0).with_fresh(fresh)
    assert optimizer.param_groups[0].learning_rate_multiplier == FRESH_LEARNING_RATE_MULTIPLIER
    rng, held = np.random.default_rng(0), rows(np.random.default_rng(10_000))

    def stream():
        while True:
            yield rows(rng)

    data = Loader(train=lambda partition: stream(), val=None, records=None, batch=BATCH)
    state = Trainer(objective, optimizer, key=jax.random.key(0)).fit(data, steps=1000, log_every=1000)
    control = float(objective.loss(objective.variables, held, None, read=False)[0])
    answered = float(objective.loss(state.variables, held, None)[0])
    assert control > 4.0 and answered < control - 1.0, (answered, control)


def test_named_groups_and_a_base_that_holds_every_parameter_keep_their_rates():
    """The default is a default: a config that names groups keeps them, and a run whose every
    parameter its base holds (an adapter's factors among them, which `Pretrained.adapt` draws into the
    base's tree) has nothing fresh to group."""
    named = OptimConfig(learning_rate=RATE, param_groups=(ParamGroup("all", ("*",)),))
    assert named.with_fresh(("layers_1/cross_attention/q/kernel",)) is named
    objective = loaded()
    whole = Answered(objective.model, thaw(objective.variables))
    assert [name for name in fresh_parameters(whole) if "cross_attention" not in name] == []
    plain = OptimConfig(learning_rate=RATE)
    assert plain.with_fresh(()) is plain


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_the_loaded_group_steps_as_an_ungrouped_optimizer_does(optimizer):
    """A loaded parameter steps under the default's groups exactly as it does with none, its decay
    sparing the biases and norm scales as before (`ParamGroup.spared`); a fresh one steps at the
    multiplier's rate."""
    params = {"layers_0": {"dense": {"kernel": jnp.full((3, 2), 0.5), "bias": jnp.full((2,), 0.5)}},
              "cross_attention": {"q": {"kernel": jnp.full((2, 2), 0.5)}}}
    gradient = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.1), params)
    plain = OptimConfig(optimizer=optimizer, learning_rate=RATE, weight_decay=0.1)
    grouped = plain.with_fresh(("cross_attention/q/kernel",))

    def step(config):
        solver = config.build(4)
        return solver.update(gradient, solver.init(params), params)[0]

    before, after = step(plain), step(grouped)
    for name in ("kernel", "bias"):
        np.testing.assert_array_equal(after["layers_0"]["dense"][name], before["layers_0"]["dense"][name])
    np.testing.assert_allclose(after["cross_attention"]["q"]["kernel"],
                               FRESH_LEARNING_RATE_MULTIPLIER * before["cross_attention"]["q"]["kernel"],
                               rtol=1e-6)


@pytest.mark.mesh(devices=8)
def test_a_decision_run_records_its_context_blocks_fresh_group_and_keeps_a_named_one(tmp_path):
    """A decision run whose context layers draw cross-attention blocks over its loaded backbone records
    the default's groups in its run.json, the blocks' and its fresh head's parameters fresh and an
    adapter's factors among the loaded ones, which a resume of the record trains in; a run that names
    groups records its own (row 2391)."""
    from test_decision_recipe import QWEN
    from test_run_options import decision, trained

    from dew.config import RunConfig
    from dew.lora import LoRA

    run = dataclasses.replace(decision(tmp_path, drawn=False), pretrained=str(QWEN), context_layers=(0,),
                              context_length=16, lora=LoRA(rank=2, modules=("q_proj",)))
    trained(run, tmp_path, "adapted")
    fresh, loaded_group = RunConfig.load(str(tmp_path / "runs" / "adapted")).optim.param_groups
    assert (fresh.name, loaded_group.name, loaded_group.patterns) == ("fresh", "loaded", ("*",))
    assert fresh.learning_rate_multiplier == FRESH_LEARNING_RATE_MULTIPLIER
    # The blocks and the head the objective draws are fresh; the adapter's factors are the base's.
    assert any("cross_attention" in name for name in fresh.patterns)
    assert all("cross_attention" in name or name.startswith("head/") for name in fresh.patterns)
    assert not any("lora" in name for name in fresh.patterns)
    named = (ParamGroup("all", ("*",)),)
    trained(dataclasses.replace(run, optim=dataclasses.replace(run.optim, param_groups=named)), tmp_path,
            "named")
    assert RunConfig.load(str(tmp_path / "runs" / "named")).optim.param_groups == named
