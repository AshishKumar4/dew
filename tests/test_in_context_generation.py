"""Generation by a language model whose continuum level steps in context (row 9066): its decode steps the
level as its training forward steps it (`dew.objectives.lm.adaptation.InContextDecoder`), a row's next
token read through the level its training forward gives that slot, and a saved run's task steps it alike.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import dataset
from block_support import model as kinded, wrapped
from reference_error import assert_as_exact, dew_path
from test_chunk_adaptation import LENGTH, LEVEL, VOCAB, _replayed, decoder, drawn_variables, variable_shapes

from dew.batch import BLOCKS_FIELD, COORDINATES_FIELD, token_coordinates
from dew.data.chat import Role
from dew.inference.tasks import TextGeneration
from dew.interop import Pretrained
from dew.lora import LoRA
from dew.nn.blocks import token_table
from dew.nn.inputs import ModelInputs
from dew.nn.mixers import AttentionMixer
from dew.nn.projections import projection_groups
from dew.nn.protocols import (
    BlockReading,
    Bounded,
    CacheCapacity,
    Generating,
    Layered,
    ReplacesTokens,
    Selective,
    Serving,
)
from dew.nn.visibility import served_blocks
from dew.objectives.lm import LMObjective, Samples
from dew.objectives.lm.adaptation import ChunkAdaptation, InContextDecoder
from dew.objectives.lm.objective import prefixed_fields
from dew.sampling import Sampling
from dew.sampling.strategies import Beam, Speculative
from dew.sampling.text import decode_ops, generate, prefill_state
from dew.training import Checkpoints, Trainer

ROWS = 2
PROMPT = 5
PAD = 7
"""The padding id the padded case's objective names, which its draws hold among their real tokens."""
PREFIX = 3
MASK = np.ones((ROWS, LENGTH), bool)
MASK[0, 3] = False
MASK[1, :2] = False
"""Every draw's real slots: a hole inside the first row's prompt, which no target crosses, and the second
row's first two slots padding, as a padded prompt's are."""

DECODE_SEEDS = {"one level": 8, "a chain": 8, "padding ids": 8, "an adapter": 8,
                "a prefix under a wrapper": 32}
"""Check 1's declared seeds for each decode against its objective's training forward, sized paired by
`seeds_for` on armada (cf76ed921)."""


def plain(dtype):
    """The test decoder (`test_chunk_adaptation.decoder`), one level on its final states."""
    return decoder(LEVEL, dtype=dtype)


def chained(dtype):
    """The test decoder with a chain of two levels."""
    return decoder((LEVEL, LEVEL), dtype=dtype)


def adapted(dtype):
    """The test decoder under a LoRA adapter on its queries, which its trunk computes under the adapter's
    interceptor (`dew.lora`), its factors drawn with every other leaf."""
    base = plain(jnp.float32)
    model = LoRA(rank=2, modules=("q_proj",)).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, LENGTH), jnp.int32)), key=0).model
    return model.clone(dtype=dtype)


def prefixed(dtype):
    """A decoder whose attention reads a prefix run's blocks whole, its level on its final states, under a
    multimodal wrapper, which holds it as a field under its own name (`block_support.wrapped`)."""
    text = kinded(full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)})
    return wrapped(text.clone(continuum=LEVEL, dtype=dtype))


CASES = {"one level": (plain, 3, None, None), "a chain": (chained, (2, 4), None, None),
         "padding ids": (plain, 3, PAD, None), "an adapter": (adapted, 3, None, None),
         "a prefix under a wrapper": (prefixed, 3, None, PREFIX)}
"""Each decode case: its model at a dtype, its adaptation's window, its objective's padding id, and its
prefix run."""


def objective_of(case: str, dtype=jnp.float32) -> LMObjective:
    """The case's objective over its model at `dtype`, its level stepped at rate 0.1."""
    build, window, pad_id, prefix = CASES[case]
    return LMObjective(build(dtype), LENGTH - 1, adaptation=ChunkAdaptation(window=window, rate=0.1),
                       pad_id=pad_id, prefix=prefix)


def drawn(seed: int, case: str):
    """A seed's variables (`drawn_variables`) and token rows, real where `MASK` holds; the padded case's
    rows hold its padding id among their real tokens, in the prompt and past it."""
    model = objective_of(case).model
    rng = np.random.default_rng([9066, seed])
    tokens = rng.integers(0, VOCAB, (ROWS, LENGTH)).astype(np.int32)
    if CASES[case][2] is not None:
        tokens[:, [2, 8]] = PAD
    return drawn_variables(seed, model, scale=0.5), tokens, MASK.copy()


def trained(objective: LMObjective):
    """The objective's training forward's logits at each slot but the last (`token_scores`), the last
    slot's token the one before it's target, a prefix run's blocks laid as its loss lays them
    (`prefixed_fields`)."""
    def run(variables, tokens, mask):
        laid = {COORDINATES_FIELD: token_coordinates(tokens.shape, valid=mask)}
        rows = ModelInputs(jnp.asarray(tokens), prefixed_fields(laid, tokens.shape, objective.prefix, None))
        return objective.token_scores(variables, rows, logits=True).logits
    return run


def decoder_of(objective: LMObjective) -> InContextDecoder:
    """The decoder the objective's policy draws with (`LMObjective.policy`)."""
    return InContextDecoder(objective.model, objective.adaptation, pad_id=objective.pad_id)


def decoded(objective: LMObjective):
    """The objective's decoder's logits at the slots its training forward scores: the prompt's slots
    prefilled, a prefix run's blocks served over it as a generation serves them, on the host
    (`served_blocks` of `MASK`'s), each later token decoded alone, the cache carried between the calls."""
    decoding = decoder_of(objective)
    served = ({} if objective.prefix is None
              else {BLOCKS_FIELD: served_blocks(MASK[:, :PROMPT], objective.prefix)})

    def run(variables, tokens, mask):
        cache = decoding.apply(variables, tokens.shape[0], method="init_cache", mutable=["cache"])[1]["cache"]
        found, updated = decoding.apply({**variables, "cache": cache}, tokens[:, :PROMPT], decode=True,
                                        attention_mask=mask[:, :PROMPT], mutable=["cache"], **served)
        steps = [found]
        for at in range(PROMPT, tokens.shape[1] - 1):
            step, updated = decoding.apply({**variables, "cache": updated["cache"]}, tokens[:, at:at + 1],
                                           decode=True, mutable=["cache"])
            steps.append(step)
        return jnp.concatenate(steps, axis=1)
    return run


def on_real(run):
    """`run`'s logits at its rows' real input slots, nothing at the rest: a slot the mask leaves out holds
    no token, which neither a decode nor the training forward gives logits that anything reads."""
    def real(variables, tokens, mask):
        return jnp.where(jnp.asarray(mask)[:, :-1, None], run(variables, tokens, mask), 0.0)
    return real


@pytest.mark.parametrize("case", list(CASES))
def test_a_decode_reads_the_logits_its_training_forward_gives(case):
    """The adapting decoder's logits at every real slot of a padded prompt, one with a hole, and the tokens
    decoded after it, two of its windows closing past the prompt, finite at every slot, on par with its
    objective's training forward's over the same tokens (check 1, the forward Dew's own path: `dew_path`), the
    truth the forward in float64: the decode steps each row's level where its training steps it, by the
    targets its loss counts, and reads each slot through the version its training reads it at (row 9066). One
    level, a chain of two, the objective's padding id among the tokens, the trunk under a LoRA adapter, and a
    prefix run's blocks read by a decoder a multimodal wrapper holds."""
    objective, wide = objective_of(case), objective_of(case, jnp.float64)
    assert_as_exact(jax.jit(on_real(decoded(objective))), dew_path(jax.jit(on_real(trained(objective)))),
                    lambda seed: drawn(seed, case), f"a decode stepping {case}", seeds=DECODE_SEEDS[case],
                    truth_fn=on_real(trained(wide)))
    assert np.all(np.isfinite(np.asarray(jax.jit(decoded(objective))(*drawn(0, case)))))


PHI3 = Path(__file__).parent / "fixtures" / "hf" / "phi3-tiny"

CROSSING_SEEDS = 8
STEPS = 7
"""The crossing's decode steps past its four prompt ids: states of four to ten ids, the switch past eight."""
"""Check 1's declared seeds for a decode across LongRoPE's switch against the training forward, sized
paired by `seeds_for` on armada (cf76ed921)."""


def test_a_decode_across_longrope_rebuilds_its_levels_with_its_cache():
    """Phi-3's LongRoPE switches its factors past its original context, where a generation rebuilds a
    crossing row's cache from its history (`cache_rebuild_position`): four ids prefilled and five more
    fed one at a time through the adapting decoder's cache, its prompt prefilled for its last slot's logits
    alone (`Selective`), each step's logits on par with the training forward's at the ids so far (check 1,
    the truth in float64): the steps before the switch at eight ids and the two past it, where its levels
    rebuilt with the trunk's cache read the factors the whole history reads (row 9066). A prompt's blocks,
    which a rebuild's replay of its ids would not carry, are refused there."""
    loaded = Pretrained.load(PHI3, dtype="float32", attention_impl="reference", max_seq_len=48)
    adaptation = ChunkAdaptation(window=3, rate=0.1)
    model = loaded.model.clone(continuum=LEVEL)
    objective = LMObjective(model, 9, adaptation=adaptation)
    wide = LMObjective(model.clone(dtype=jnp.float64), 9, adaptation=adaptation)
    decoding = InContextDecoder(model, adaptation)
    assert decoding.cache_rebuild_position == model.cache_rebuild_position is not None

    def walk(variables, ids):
        ops = decode_ops(decoding, variables, 0, 0)
        state, _ = prefill_state(decoding, variables, ModelInputs(ids[:, :4]), ops)
        steps = {}
        for step in range(STEPS):
            steps[f"step {step}"] = state.logits
            state = ops.advance(state, ids[:, 4 + step], jnp.ones(ROWS, bool))
        return steps

    def forward(objective):
        def run(variables, ids):
            return {f"step {step}": objective.token_scores(variables, ids[:, :5 + step],
                                                           logits=True).logits[:, -1]
                    for step in range(STEPS)}
        return run

    def draw(seed):
        rng = np.random.default_rng([9066, 3, seed])
        ids = rng.integers(0, model.vocab_size, (ROWS, 5 + STEPS)).astype(np.int32)
        return drawn_variables(seed, model, scale=0.5), ids

    assert model.cache_rebuild_position == 8
    assert_as_exact(walk, dew_path(forward(objective)), draw, "a decode across LongRoPE's switch",
                    seeds=CROSSING_SEEDS, truth_fn=forward(wide))
    variables, ids = draw(0)
    blocked = ModelInputs(jnp.asarray(ids[:, :4]), {BLOCKS_FIELD: jnp.ones((ROWS, 4), jnp.int32)})
    with pytest.raises(ValueError):
        prefill_state(decoding, variables, blocked, decode_ops(decoding, variables, 0, 0))


def test_an_adapting_run_draws_what_its_saved_task_draws(tmp_path):
    """A run that carries its chain's levels, its windows and rates a level each and a padding id among its
    targets: its policy (`LMObjective.policy`) and the task its saved run loads (`TextGeneration.from_run`)
    draw the same tokens at the same likelihoods, bit for bit, the restored task stepping the levels the
    run carried to it as the live policy does (row 9066)."""
    adaptation = ChunkAdaptation(window=(2, 4), rate=(0.1, 0.05), carry="run")
    objective = LMObjective(chained(jnp.float32), LENGTH, adaptation=adaptation, pad_id=0)
    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "run")))
    trained_run = trainer.fit(dataset(_replayed), steps=2, log_every=1, checkpoint_every=2)
    live = objective.policy(trained_run.variables, sampling=Sampling(temperature=0.7))
    saved = TextGeneration.from_run(str(tmp_path / "run"))
    assert isinstance(saved.model, InContextDecoder)
    assert (saved.model.adaptation, saved.model.pad_id) == (adaptation, 0)
    prompt = np.random.default_rng(3).integers(0, VOCAB, (ROWS, PROMPT)).tolist()
    ran, again = live(prompt, 7, key=0).host(), saved(prompt, 7, key=0, sampling=live.sampling).host()
    for field in ("tokens", "lengths", "raw_log_probs", "behavior_log_probs"):
        np.testing.assert_array_equal(np.asarray(getattr(ran, field)), np.asarray(getattr(again, field)))


def test_a_server_draws_what_its_adapting_task_draws():
    """A server of an adapting decoder's task (`Server.from_task`), two slots for three requests so a
    slot a finished request held seats another, draws each request's tokens as the task alone does: a
    seated row's levels start from the run's, its chain the cache's own row (row 9066)."""
    from dew.inference.serving import Server

    model = InContextDecoder(plain(jnp.float32), ChunkAdaptation(window=3, rate=0.1))
    task = TextGeneration(model, drawn_variables(0), sampling=Sampling(temperature=0, eos_token_ids=None))
    rng = np.random.default_rng(5)
    prompts = tuple(rng.integers(0, VOCAB, length).astype(np.int32) for length in (4, 2, 6))
    server = Server.from_task(task, slots=2, capacity=32, admission=2)
    assert server.mixed_refusal == model.mixed_admission_refusal()
    tickets = [server.submit(prompt, length, key=index)
               for index, (prompt, length) in enumerate(zip(prompts, (5, 3, 7), strict=True))]
    server.run()
    for index, (ticket, prompt, length) in enumerate(zip(tickets, prompts, (5, 3, 7), strict=True)):
        served, alone = ticket.result().host(), task(prompt[None], length, key=index).host()
        np.testing.assert_array_equal(served.tokens, alone.tokens)
        np.testing.assert_array_equal(served.lengths, alone.lengths)


def test_the_decoder_answers_its_models_generation_hooks():
    """The adapting decoder answers the hooks a generation reads as its model does (`dew.nn.protocols`):
    its bounds, its cache's layout and capacity, its LongRoPE rebuild, the projections it packs by its own
    rules (an adapter's none), its blocks, the ids a multimodal model replaces before its table, its run's
    progress and its own refusal; a mixed call's rows and a shared prefix it refuses."""
    model = plain(jnp.float32)
    decoding = InContextDecoder(model, ChunkAdaptation(window=3))
    for protocol in (Bounded, CacheCapacity, Layered, Serving, Generating, BlockReading, Selective,
                     ReplacesTokens):
        assert isinstance(decoding, protocol), protocol.__name__
    answers = (decoding.max_seq_len, decoding.vocab_size, decoding.kv_cache, decoding.cache_rebuild_position)
    assert answers == (model.max_seq_len, model.vocab_size, model.kv_cache, model.cache_rebuild_position)
    adaptation = decoding.adaptation
    assert decoding.with_cache_capacity(8) == InContextDecoder(model.with_cache_capacity(8), adaptation)
    paged = dataclasses.replace(model.kv_cache, page_size=4, pages=8)
    assert decoding.with_kv_cache(paged) == InContextDecoder(model.with_kv_cache(paged), adaptation)
    assert decoding.reads_progress() == model.reads_progress()
    assert decoding.generation_refusal() == model.generation_refusal()
    assert decoding.block_chunks() == model.block_chunks()
    shapes = variable_shapes(model)
    assert decoding.inference_projection_groups(shapes) == projection_groups(model, shapes)
    lora = adapted(jnp.float32)
    assert InContextDecoder(lora, adaptation).inference_projection_groups(variable_shapes(lora)) == ()
    multimodal = prefixed(jnp.float32)
    wrapping = InContextDecoder(multimodal, adaptation)
    replacing = (wrapping.replaced_ids, wrapping.replaces_media)
    assert replacing == (multimodal.replaced_ids, multimodal.replaces_media)
    assert token_table(wrapping) == token_table(multimodal)
    assert decoding.mixed_admission_refusal() is not None and decoding.split_prefill_refusal() is not None


def test_what_an_adapting_decoder_cannot_draw_is_refused():
    """A strategy that reads tokens it then rejects or reorders (a speculative block's, a beam's) draws
    past what an adapting decoder's cache rewinds, through a task or the sampler alike; a damped or
    stacked adaptation decodes none, a packed request steps no row's level by one document, and an
    objective with either, or whose loss scores one role's targets, steps its level where no generation
    can, so it records no task: each refused, naming why (ValueError)."""
    model = InContextDecoder(plain(jnp.float32), ChunkAdaptation(window=3))
    variables = drawn_variables(0)
    prompt = jnp.zeros((ROWS, PROMPT), jnp.int32)
    for strategy in (Beam(width=2), Speculative()):
        with pytest.raises(ValueError):
            TextGeneration(model, variables, strategy=strategy)
        with pytest.raises(ValueError):
            generate(model, variables, ModelInputs(prompt), 4, key=jax.random.key(0), strategy=strategy)
    for adaptation in (ChunkAdaptation(damped=True), ChunkAdaptation(layers=(0,))):
        with pytest.raises(ValueError):
            InContextDecoder(plain(jnp.float32), adaptation)
    cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
    with pytest.raises(ValueError):
        model.apply({**variables, "cache": cache}, prompt, decode=True, mutable=["cache"],
                    segment_ids=jnp.asarray([[1, 1, 2, 2, 2]] * ROWS))
    for undrawn in (LMObjective(plain(jnp.float32), LENGTH, adaptation=ChunkAdaptation(damped=True)),
                    LMObjective(plain(jnp.float32), LENGTH, adaptation=ChunkAdaptation(layers=(0,))),
                    LMObjective(plain(jnp.float32), LENGTH, adaptation=ChunkAdaptation(window=3),
                                loss_role=Role.ASSISTANT)):
        assert undrawn.task_settings() is None
        with pytest.raises(ValueError):
            undrawn.policy(variables)
    with pytest.raises(ValueError):
        LMObjective(plain(jnp.float32), LENGTH, adaptation=ChunkAdaptation(window=3),
                    loss_role=Role.ASSISTANT, samples=Samples([1, 2], 2))
