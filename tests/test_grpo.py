"""GRPO: the clipped-surrogate-plus-KL composition against verl.

The loss must match verl 0.9's PPO path with the GRPO settings
(`compute_policy_loss_vanilla` with `clip_ratio` 0.2 both sides,
`clip_ratio_c` 3.0 and `token-mean`, plus `beta` times the `token-mean` k3
from `kl_penalty_forward`). The reference is tests/fixtures/rl/grpo.npz,
recorded by tools/parity_grpo.py from torch over the seeds check 1 draws,
each a rollout: old, current and reference log-probabilities, both-signed
advantages, and a mask with short tails, with entries past every clip point.
`GRPOObjective` reads the same terms out of the rolled-out batch.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from one_draw import assert_as_exact_as_the_reference
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, with_gradients
from rl_support import Published, ScriptedSource, TinyHead, TinyLogits, clipped_surrogate, token_mean
from steady_state import steady_state

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY, _utf8
from dew.objectives.base import Step
from dew.objectives.rl import GRPOObjective, RolloutScheduler
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
)
from dew.rl import k3_kl, token_log_ratio
from dew.sampling import Sampling

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "grpo.npz"
VOCAB = 8
PROMPT_WIDTH = 5
RESPONSE_WIDTH = 3
ROWS = 2


@pytest.fixture(scope="module")
def reference():
    return recording(FIXTURE)


def test_grpo_loss_and_gradient_match_verl(reference):
    """Dew's composition against verl 0.9.0's own `compute_policy_loss_vanilla`
    and `kl_penalty` (tools/parity_grpo.py), the loss and its gradient in the
    current log-probabilities on par with verl's fp32 over the recorded
    rollouts (check 1), its float64 run the truth."""
    beta = float(reference["beta"])
    seeds = declared_seeds(FIXTURE)["grpo"]
    recorded_draw, recorded, truth = indexed_recording(reference, "grpo", seeds)

    @jax.jit
    def dew(index, rollout, cotangent):
        def loss(current):
            ratio = token_log_ratio(current, rollout["old_log_probs"])
            pg, _ = clipped_surrogate(ratio, rollout["advantages"], rollout["response_mask"])
            kl = token_mean(k3_kl(current, rollout["ref_log_probs"]), rollout["response_mask"])
            return pg + beta * kl

        value, gradient = with_gradients(loss)(rollout["current_log_probs"], cotangent)
        return {"loss": value, "grad": gradient}

    # The recording holds the loss's gradient, a scalar loss's under the cotangent 1, which each draw
    # carries for Dew's arm.
    assert_as_exact(dew, lambda index, rollout, cotangent: recorded(index, rollout),
                    lambda seed: (*recorded_draw(seed), np.float32(1)), "GRPO loss and gradient", seeds=seeds,
                    truth_fn=lambda index, rollout, cotangent: truth(index, rollout))


# --- the objective -------------------------------------------------------------

def rollout_batch(seed=0):
    """Two packed rows, one chain each, the last RESPONSE_WIDTH ids sampled,
    with a short second response."""
    rng = np.random.RandomState(seed)
    width = PROMPT_WIDTH + RESPONSE_WIDTH
    ids = rng.randint(0, VOCAB, (ROWS, width)).astype(np.int32)
    mask = np.zeros((ROWS, width), np.float32)
    mask[:, PROMPT_WIDTH:] = 1
    mask[1, -1] = 0
    terms = np.where(mask != 0, rng.normal(-0.5, 0.5, (ROWS, width)), 0).astype(np.float32)
    advantages = np.where(mask != 0, rng.normal(0, 1, (ROWS, width)), 0).astype(np.float32)
    return {
        IDS_KEY: jnp.asarray(ids),
        SEGMENT_IDS_KEY: jnp.ones((ROWS, width), jnp.int32),
        POSITIONS_KEY: jnp.tile(jnp.arange(width, dtype=jnp.int32), (ROWS, 1)),
        OLD_LOG_PROBS_KEY: jnp.asarray(terms),
        BEHAVIOR_LOG_PROBS_KEY: jnp.asarray(terms),
        ADVANTAGES_KEY: jnp.asarray(advantages),
        RESPONSE_MASK_KEY: jnp.asarray(mask),
    }


def test_a_prompt_dataset_trains_as_the_prompt_config_does():
    """`Trainer.fit` compiles a Dataset with the objective's own stages.
    GRPO adds none, so one rollout step over `prompts.encode` rows moves the
    policy exactly as one over the same prompts through the prompts preset."""
    import optax
    from bank_support import identical

    from dew.data import Loading, Memory, presets, prompts
    from dew.data.compile import compile
    from dew.data.text import tokenizer_for
    from dew.training import Layout, Trainer

    tokenizer = str(Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat")
    vocab = tokenizer_for(tokenizer).vocab_size
    rows = [{"prompt": "hi", "data_source": "rule", "ground_truth": "a"}] * 8

    def trained(data):
        objective = GRPOObjective(TinyHead(vocab_size=vocab), 8 + 4 - 1)
        rollout = RolloutScheduler.sampled(objective, lambda text, record: float(len(text)), groups=2,
                                           max_new_tokens=4, sampling=Sampling(temperature=1.0))
        trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0), rollout=rollout)
        return trainer.fit(data, steps=1).variables

    identical(trained(Memory(rows).filter_map(prompts.encode(tokenizer, max_prompt_len=8)).batch(8)),
              trained(compile(Memory(rows).pipe(presets.prompts, tokenizer, max_prompt_len=8, batch=8),
                              read="memory", loading=Loading(workers=0))))


def test_the_loss_reads_the_rolled_out_batch():
    """The objective's loss is the composition over the batch's own terms:
    current log-probabilities scored along each chain, ratio against the
    stored old ones, surrogate and KL masked alike."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB),
                              PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    params = objective.init(jax.random.key(0))
    frozen = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf)), params)
    batch = rollout_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=frozen)

    loss, aux = objective.scalar_loss(params, batch, step)

    ids = np.asarray(batch[IDS_KEY])
    start = PROMPT_WIDTH - 1
    policy = np.asarray(objective.per_token_log_probs(params, ids))[:, start:]
    ref = np.asarray(objective.per_token_log_probs(frozen, ids))[:, start:]
    old = np.asarray(batch[OLD_LOG_PROBS_KEY])[:, PROMPT_WIDTH:]
    advantages = np.asarray(batch[ADVANTAGES_KEY])[:, PROMPT_WIDTH:]
    mask = np.asarray(batch[RESPONSE_MASK_KEY])[:, PROMPT_WIDTH:]
    ratio = token_log_ratio(jnp.asarray(policy), jnp.asarray(old))
    pg, _ = clipped_surrogate(ratio, jnp.asarray(advantages), jnp.asarray(mask))
    kl = token_mean(k3_kl(jnp.asarray(policy), jnp.asarray(ref)), jnp.asarray(mask))
    assert float(loss) == pytest.approx(float(pg + 0.01 * kl), rel=1e-5)
    assert {"pg", "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower", "kl"} <= set(aux.metrics)


@pytest.mark.parametrize("policy", [TinyHead, TinyLogits])
def test_evaluation_scores_prompt_perplexity(policy):
    """Validation never samples: each prompt's shifted cross entropy, the
    real suffix weighted off `prompt_length`, pads counting nothing, compiled
    once for the objective's modules; a policy that gives its logits only
    from its tokens is scored as the training loss scores it."""
    objective = GRPOObjective(policy(vocab_size=VOCAB), 4, beta=0.0)
    params = objective.init(jax.random.key(0))
    prompts = np.array([[0, 0, 1, 2, 3], [4, 5, 6, 7, 1]], np.int32)
    batch = {PROMPT_KEY: jnp.asarray(prompts),
             LENGTH_KEY: jnp.asarray([5, 3], np.int32)}
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)

    scores = objective.evaluate(params, batch, step)
    # A second held pass runs the program the first compiled.
    with steady_state():
        objective.evaluate(params, batch, step)

    expected = np.asarray(objective.per_token_log_probs(params, prompts))
    weights = np.asarray(scores.weights)
    np.testing.assert_allclose(np.asarray(scores.losses) * weights, -expected * weights, rtol=1e-5)
    np.testing.assert_array_equal(weights, [[1, 1, 1, 1], [0, 0, 1, 1]])

    with pytest.raises(ValueError):
        objective.evaluate(params, {PROMPT_KEY: jnp.ones((1, 1), jnp.int32),
                                    LENGTH_KEY: jnp.ones(1, jnp.int32)}, step)


def test_a_rollout_after_the_first_reuses_its_programs_and_reads_only_what_it_scores():
    """Sampling a group of the same prompt width again runs the programs the
    first rollout compiled, and the host reads the drawn rows and the group
    advantages by name and nothing else (`steady_state`). The prompts reach
    the device as each batch arrives, so only reads are held."""
    objective = GRPOObjective(TinyHead(vocab_size=VOCAB), PROMPT_WIDTH + RESPONSE_WIDTH - 1, beta=0.01)
    state = SimpleNamespace(variables=objective.init(jax.random.key(0)), step=0, updates=0)
    rollout = RolloutScheduler.sampled(objective, lambda *args: float(len(args[0]) % 3), groups=2,
                                       max_new_tokens=RESPONSE_WIDTH, sampling=Sampling(temperature=1.0))
    info = len("other")
    def pad(text):
        return _utf8(text, info)

    def batch(offset):
        rows = np.arange(PROMPT_WIDTH, dtype=np.int32)[None] + np.arange(2)[:, None] + offset
        prompts = rows % (VOCAB - 1) + 1
        return {PROMPT_KEY: prompts, LENGTH_KEY: np.full(2, PROMPT_WIDTH, np.int32),
                SOURCE_KEY: np.stack([pad("rule"), pad("other")]), TRUTH_KEY: np.stack([pad("1"), pad("2")]),
                INFO_KEY: np.stack([pad(""), pad("")])}

    keys = [jax.random.key(index) for index in range(3)]
    rollout(state, batch(0), keys[0])
    with steady_state(allow=("host_to_device",)):
        rolled = [rollout(state, batch(offset), key) for offset, key in zip((1, 2), keys[1:], strict=True)]
    assert all(rows[IDS_KEY].shape == (4, PROMPT_WIDTH + RESPONSE_WIDTH) for rows in rolled)


def test_a_small_log_ratios_k3_estimate_keeps_its_leading_term():
    """The k3 estimate `exp(d) - d - 1` of a log ratio `d` near zero is `d**2 / 2` to leading order. In
    float32 `exp(d)` rounds to a spacing of `2**-24` about 1, which `- 1` leaves standing: at `d` of 1e-6
    the estimate is off by most of itself. Through `expm1` (`dew.rl.surrogate.sampled_kl`) the estimate
    is on par with NumPy's float32 `expm1(d) - d` from float64's (check 1)."""
    from dew.rl.surrogate import sampled_kl

    log_ratio = np.asarray([1e-6, -1e-6, 3e-5, -2e-4, 0.1, -0.5], np.float32)
    estimate = np.asarray(k3_kl(jnp.zeros_like(log_ratio), jnp.asarray(log_ratio)))
    wide = log_ratio.astype(np.float64)
    assert_as_exact_as_the_reference(estimate, np.expm1(log_ratio) - log_ratio, np.expm1(wide) - wide,
                                     "k3 at small log ratios")
    np.testing.assert_array_equal(np.asarray(sampled_kl(jnp.asarray(log_ratio))), estimate)


def test_a_synchronous_collection_refuses_a_ratio_read_at_another_temperature():
    """A synchronous collection's proximal likelihoods are the sampler's raw ones, the policy at
    temperature 1 (`Synchronous.rescore`): a GRPO objective reading its ratio at another
    `sampling_temperature` is refused, where its unchanged weights would not give a ratio of 1; one at
    1.0 is taken, whatever temperature and truncation the sampler draws at."""
    def collected(objective):
        return RolloutScheduler.sampled(objective, lambda *args: 1.0, groups=2, max_new_tokens=RESPONSE_WIDTH,
                                        sampling=Sampling(temperature=1.3, top_k=5))

    with pytest.raises(ValueError):
        collected(GRPOObjective(TinyHead(vocab_size=VOCAB), PROMPT_WIDTH + RESPONSE_WIDTH - 1,
                                sampling_temperature=0.7))
    collected(GRPOObjective(TinyHead(vocab_size=VOCAB), PROMPT_WIDTH + RESPONSE_WIDTH - 1))


def annealing(**fields) -> GRPOObjective:
    """GRPO over a DEX decoder whose anneal runs ten updates, the off-policy correction an asynchronous
    collection takes on, and `fields`."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.dex import DifferentialExtension

    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=12, dtype="float32", attention_impl="xla",
                              dex=DifferentialExtension(((0, (0, 1)),), anneal_steps=10))
    return GRPOObjective(model, seq_len=7, behavior_importance=2.0, **fields)


def asynchronously(objective: GRPOObjective, seen, steps: int = 6, *, rewarded: bool = True,
                   estimator: str = "group") -> None:
    """`steps` of `objective` at a rate of nothing, its rows collected asynchronously from a scripted
    source (`Asynchronous`, rescored under the trainer's weights, a row a task, as many as a host's
    devices hold), each step's packed rows and state handed to `seen`; the reward's reference penalty
    paid against `objective`'s starting policy where it names one (`reward_kl`); each sample rewarded
    by its index, or all alike where not `rewarded`, under `estimator`."""
    import itertools

    import optax

    from dew.data.dataset import Loader
    from dew.objectives.rl.scheduler import Asynchronous
    from dew.objectives.rl.sessions import Call, Session, Status
    from dew.training import Trainer

    def drawn(task, sample, version):
        call = Call((1, 2), (3 + sample, 4), (-0.5, -0.25), "stop", version)
        return Session(str(task.id), "", sample, 0, (call,), Status.COMPLETED,
                       float(sample) if rewarded else 1.0)

    tasks = np.arange(1, 9, dtype=np.int32)
    scheduler = RolloutScheduler(Asynchronous(ScriptedSource(drawn), Published(), objective, width=8,
                                              rows=len(tasks), max_lag=1, ahead=0), groups=2,
                                 estimator=estimator,
                                 reference=objective if objective.reward_kl > 0 else None)

    def rollout(state, batch, key):
        packed = scheduler(state, batch, key)
        seen(state, packed)
        return packed

    data = scheduler.tasks(Loader(train=lambda partition: itertools.repeat({"task_id": tasks}), val=None,
                                  records=len(tasks), batch=len(tasks)))
    Trainer(objective, optax.sgd(0.0), key=0, rollout=rollout).fit(data, steps=steps, log_every=100,
                                                                   checkpoint_every=None)


def test_an_asynchronous_rescore_reads_the_policy_at_the_progress_its_loss_reads():
    """An asynchronous collection's proximal policy, rescored under the trainer's weights
    (`Asynchronous.rescore`), is the policy the step's loss reads, at the updates the run has committed
    (`Progress(state.updates)`; ledger 2635): over a DEX decoder half way through its anneal its
    likelihoods are those its loss scores at those updates, not the run's end's, so a ratio over
    unchanged weights is 1."""
    from dew.nn.progress import END, Progress
    from dew.objectives.rl.scheduler import scored_rows

    objective = annealing()
    seen = []

    def scores(state, packed):
        # Each step's rescored rows beside the same rows scored at the step's updates and at the run's end.
        seen.append((int(state.updates), np.asarray(packed[OLD_LOG_PROBS_KEY]),
                     scored_rows(objective, state.variables, packed, Progress(state.updates)),
                     scored_rows(objective, state.variables, packed, END)))

    asynchronously(objective, scores)
    updates, rescored, stepped, ended = seen[-1]
    assert updates == 5
    np.testing.assert_array_equal(rescored, stepped)
    assert not np.array_equal(rescored, ended)


def test_a_reward_penalty_over_the_starting_weights_is_nil_half_way_through_an_anneal():
    """The reward's reference penalty (`GRPOObjective.reward_kl`) scores the reference at the progress the
    proximal column is read at (ledger 2666): half way through a DEX decoder's anneal, rows rescored at
    the updates the run committed pay no penalty while the weights are the reference's, so where every
    sample earns one reward its leave-one-out advantages (`rloo`, which no deviation rescales) are
    nothing, where a reference read at the run's end would charge each sample a penalty of its own."""
    advantages = {}

    def kept(state, packed):
        advantages[int(state.updates)] = np.asarray(packed[ADVANTAGES_KEY])

    asynchronously(annealing(reward_kl=0.5), kept, rewarded=False, estimator="rloo")
    np.testing.assert_array_equal(advantages[5], 0)


def test_a_run_names_a_method_that_builds_its_objective_and_its_rollout(tmp_path):
    """A GRPO run whose rollout names a method (`--rollout.method rloo`) builds the method's objective and
    scheduler from it: RLOO's reference penalty in the reward, weighted by the run's
    `--objective.reward-kl`, keeps the frozen starting policy the step's rewards are penalized against.
    The run records the method and loads with it, and a setting the method fixes, stated on the command
    line (`--objective.beta` under DAPO, which penalizes no reference), is refused."""
    import rl_support

    from dew.config import ObjectiveConfig
    from dew.config.lm import LMRunConfig, PromptRollout
    from dew.data import Memory, presets
    from dew.registry import ModelConfig

    tokenizer = str(Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat")
    rows = [{"prompt": "hi there", "data_source": "rule", "ground_truth": "a"}] * 8
    rollout = PromptRollout(reward=f"{rl_support.__name__}:completion_length", groups=2, max_new_tokens=4)
    model = ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1, "num_heads": 2,
                                               "dtype": "float32"})
    default = LMRunConfig(data=Memory(rows).pipe(presets.prompts, tokenizer, max_prompt_len=8, batch=8),
                          tokenizer=tokenizer, objective=ObjectiveConfig("grpo"), rollout=rollout,
                          model=model, max_new_tokens=0)
    flags = ["--trainer.steps", "1", "--trainer.name", "rloo", "--trainer.checkpoint-dir", str(tmp_path),
             "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False"]
    config = LMRunConfig.cli([*flags, "--rollout.method", "rloo", "--objective.reward-kl", "0.05"],
                             default=default)
    state = config.run()
    assert int(state.step) == 1 and state.ema is not None
    assert LMRunConfig.load(str(tmp_path / "rloo")).rollout.method == "rloo"
    refused = LMRunConfig.cli([*flags, "--trainer.name", "dapo", "--rollout.method", "dapo",
                               "--objective.beta", "0.1"], default=default)
    with pytest.raises(ValueError):
        refused.run()


def test_a_recorded_grpo_run_trains_on_the_rollout_it_records(tmp_path):
    """An LM run that names GRPO records its rollout as a field (`LMRunConfig.rollout`, `--rollout.reward`):
    over a prompts chain whose prompts fill its width it trains on the completions its policy draws,
    scored by the reward its import path names, its context a full-width prompt and its whole
    completion budget, and the run it records loads with that rollout. GRPO with no rollout, and a
    rollout under another objective, are refused."""
    import dataclasses

    import rl_support

    from dew.config import ObjectiveConfig
    from dew.config.lm import LMRunConfig, PromptRollout
    from dew.data import Memory, presets
    from dew.registry import ModelConfig

    tokenizer = str(Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat")
    # Longer than the width: each prompt keeps its last 8 ids (the preset's `long='truncate'`).
    rows = [{"prompt": "hi there " * 8, "data_source": "rule", "ground_truth": "a"}] * 8
    rollout = PromptRollout(reward=f"{rl_support.__name__}:completion_length", groups=2, max_new_tokens=4)
    model = ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1, "num_heads": 2,
                                               "dtype": "float32"})
    default = LMRunConfig(data=Memory(rows).pipe(presets.prompts, tokenizer, max_prompt_len=8, batch=8),
                          tokenizer=tokenizer, objective=ObjectiveConfig("grpo"), rollout=rollout,
                          model=model, max_new_tokens=0)
    config = LMRunConfig.cli(["--trainer.steps", "1", "--trainer.name", "grpo", "--trainer.checkpoint-dir",
                              str(tmp_path), "--trainer.compilation-cache-dir", "None",
                              "--trainer.multi-host", "False", "--trainer.log-every", "1"], default=default)
    assert config.tokens().seq_len == 8 + 4 - 1
    state = config.run()
    assert int(state.step) == 1
    assert LMRunConfig.load(str(tmp_path / "grpo")).rollout == rollout
    with pytest.raises(ValueError):
        dataclasses.replace(config, rollout=None)
    with pytest.raises(ValueError):
        dataclasses.replace(config, objective=ObjectiveConfig("lm"))
