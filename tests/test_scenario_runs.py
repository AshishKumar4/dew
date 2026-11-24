"""Every scenario's covering set over its settings axes (docs/design/testing.md, settings and devices): each
cell one short run, a user's `dew train` or the scenario's own API in a fresh process that sees the cell's
devices, held to the contracts its flow touches (tests/scenario_contract.py) on the run it made. The
contracts a run of their own shows are held once a flow of a scenario, on its first cell (`Scenario.firsts`):
its run resumed and its sampling rerun; and its first step under other rows, once a scenario (`first`)."""

import dataclasses
import functools
import itertools
import json
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scenario_called
from flax import linen as nn
from interop_support import extract_fixture
from placement_parity import layouts_agree
from run_kinds import blockwise, cli_settings, lm, saved_record, scalars, settled
from scenario_called import CODECS, DIFFUSION_RL, GGUF, STEPS, quantized_loaded
from scenario_contract import (
    Called,
    Phase,
    Recorded,
    assert_resumes_onto_its_uninterrupted_run,
    assert_samples_from_its_cached_programs,
    assert_steps_finite,
    covering,
    first_steps,
)
from selection_margins import MARGIN, NARROW_MARGIN, clearing, top_margin
from test_mixers import KIND_FIELDS

import dew
from dew.checkpoints import Checkpoints
from dew.config import ObjectiveConfig, RunConfig
from dew.config.lm import LMRunConfig
from dew.data import Memory, presets
from dew.inference import TextGeneration
from dew.inference.serving import PagedKV, Server
from dew.nn.kernels.generation import run_precision
from dew.nn.mixer_base import MixerBase
from dew.nn.precision import numerics_of
from dew.quantization import QuantizedDtype
from dew.registry import mixers, to_record
from dew.sampling import Sampling, Speculative

MODULE = "test_scenario_runs"
"""This module as a fresh process imports it, to call a sampling made through Dew's API."""


@dataclass(frozen=True)
class Scenario:
    """A scenario's settings axes (the values each takes on the CPU tier) and its cell: one short run at the
    cell's settings under `root`, held to its claims. Its cells are a covering set of the combinations of
    its axes' values its flows admit (`admits`, every one where None; `cells`). Where it trains, its
    training run at a cell (`trained`), which the resume claim pauses and resumes; where it samples, this
    module's function that samples a cell (`sampled`), called on the cell and the directory it samples from:
    the cell's trained run, or the cell's root, where it samples what it writes there itself
    (`sampled_from`). Where an axis's values are flows of their own (an RL method, a diffusion LM's
    objective, a way of serving: programs and state none of the others runs), each value's first cell holds
    the claims a run of its own shows (`apart`, `firsts`). Where its runs read the LM data path, the axes
    over whose every value its first cell holds the rows claims (`laid`, none for its first cell alone;
    `LAID`)."""

    axes: Mapping[str, tuple]
    run: Callable[[Mapping, Path], None]
    trained: Callable[[Mapping, Path], Recorded | Called] | None = None
    sampled: str | None = None
    sampled_from: str = "run"
    laid: tuple[str, ...] | None = None
    admits: Callable[[dict], bool] | None = None
    apart: tuple[str, ...] = ()

    def cells(self) -> list[dict]:
        """The scenario's cells: a covering set of what it admits (`scenario_contract.covering`)."""
        return covering(self.axes, self.admits)

    def firsts(self) -> list[dict]:
        """The cells where the scenario holds the claims a run of its own shows: its first cell at each value
        of its axes `apart`, or its first cell alone where it names none."""
        held: dict[tuple, dict] = {}
        for cell in self.cells():
            held.setdefault(tuple(cell[axis] for axis in self.apart), cell)
        return list(held.values())


def settings(cell: Mapping, root: Path, more: Mapping[str, str] | None = None) -> dict[str, str]:
    """A cell's `dew train` settings (`run_kinds.cli_settings`): a short run logging every step, at the cell's
    precision over its devices' fsdp split, small leaves sharded, and `more`."""
    return cli_settings(root, "cell", {"trainer.steps": str(STEPS), "trainer.log_every": "1",
                                       "trainer.precision": cell.get("precision", "32-true"),
                                       "trainer.mesh.fsdp": str(cell.get("devices", 1)),
                                       "trainer.layout.min_shard": "64", **(more or {})})


EVALUATED = {"trainer.eval_every": "1"}
"""A run that holds validation data evaluates it every step: a resumed run then evaluates before the last
step it logs, whose persistent-cache counts show its evaluation's programs too."""


def called(function: str) -> Callable[[Mapping, Path], Called]:
    """A scenario's run made through Dew's API by `scenario_called`'s `function` in a fresh process that
    sees the cell's devices (`Called`)."""
    return lambda cell, root: Called(scenario_called.__name__, function, cell, root, cell.get("devices", 1))


# --- dense language-model pretraining ----------------------------------------------------------------------

def lm_run(cell: Mapping, root: Path) -> LMRunConfig:
    """The shared LM run over a token store's documents (`run_kinds.lm`), laid as the cell lays its rows
    (padded, one document a row, where it names none) and read as the cell reads (memory where it names
    none), its model's dtype left for the run's precision to derive."""
    return lm(root, drawn=False, rows=cell.get("rows", "padded"), reading=cell.get("read", "memory"),
              dtype=None)


def assert_generates_as_its_checkpoint(run: Path) -> None:
    """Export then reload is identical: the run's task (`dew.pipeline`) generates, greedily, what a
    generation over its model and its final checkpoint's tree generates."""
    task = dew.pipeline(str(run))
    restored = TextGeneration(task.model, Checkpoints(str(run)).variables())
    greedy = Sampling(temperature=0)
    np.testing.assert_array_equal(task("the ", key=1, sampling=greedy).host().tokens,
                                  restored([list(b"the ")], 4, key=1, sampling=greedy).host().tokens)


def generated(cell: Mapping, run: str) -> np.ndarray:
    """The run's task's greedy generation of a prompt (`dew.pipeline`)."""
    return np.asarray(dew.pipeline(run)("the ", key=1, sampling=Sampling(temperature=0)).host().tokens)


def dense_lm_trained(cell: Mapping, root: Path) -> Recorded:
    """A decoder built and trained over a token store's documents packed or padded, read as the cell reads,
    at the cell's memory rung."""
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4",
                                          "model.remat": str(cell["memory"])}),
                    cell["devices"])


def dense_lm(cell: Mapping, root: Path) -> None:
    """Build a decoder, train it and reload its run: every step's gradient finite (`assert_steps_finite`), and
    the reloaded task generating what its final checkpoint's model generates."""
    run, _ = dense_lm_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "dense LM")
    assert_generates_as_its_checkpoint(run)


# --- a served decoder's continuation, held to its float64 forward -------------------------------------------

def float64_twin(served: TextGeneration, canonical: nn.Module) -> tuple[nn.Module, dict]:
    """The truth of a served task's logits: its canonical float model (`canonical`, the model the task serves
    before any quantization) in float64 on the reference kernel, and the served weights in float64, each
    quantized one its integers times its scales, their exact function (`test_quantization`'s split quantized
    tree)."""
    from dew.quantization import Annotated

    def value(node):
        if isinstance(node, Annotated):
            import qwix

            return np.asarray(qwix.dequantize(node.array.astype(jnp.float64)), np.float64)
        held = np.asarray(node)
        return held.astype(np.float64) if np.issubdtype(held.dtype, np.floating) else held

    with jax.enable_x64(new_val=True):
        weights = jax.tree.map(value, dict(served.variables),
                               is_leaf=lambda node: isinstance(node, Annotated))
    return canonical.clone(dtype=jnp.float64), weights


Draws = list[tuple[np.ndarray, int, np.ndarray]]
"""Served rows, each its tokens (its prompt's, then its new ones), its new tokens' count and their
log-probabilities."""


def draws_of(generation) -> Draws:
    """A generation's rows (`Generation.host`) as `Draws`."""
    held = generation.host()
    return [(np.asarray(held.tokens)[row], int(held.lengths[row]), np.asarray(held.raw_log_probs)[row])
            for row in range(len(held.lengths))]


def called_on(task: TextGeneration) -> Callable[[list[list[int]], int], Draws]:
    """Prompts served by one greedy call of `task` over them all: `serve(prompts, steps)`."""
    return lambda prompts, steps: draws_of(task(prompts, steps, key=0, sampling=Sampling(temperature=0)))


def assert_served_as_its_forward(served: TextGeneration, serve: Callable[[list[list[int]], int], Draws],
                                 canonical: nn.Module, prompts: int, compute, steps: int = 4) -> None:
    """A served task's greedy continuation is its model's (testing.md: an argmax held at a stated margin): on
    `prompts` prompts of two tokens drawn so that every step of each one's float64 greedy walk
    (`model_support.greedy_walk` over `float64_twin`) clears the top-2 margin the served `compute` dtype's
    rounding moves a logit by (`selection_margins`: MARGIN, or NARROW_MARGIN in bfloat16), the tokens `serve`
    serves them (`Draws`) are the walk's and every generated token's log-probability is finite."""
    from model_support import greedy_walk

    wide, weights = float64_twin(served, canonical)
    least = NARROW_MARGIN if jnp.dtype(compute) == jnp.bfloat16 else MARGIN

    def walked(prompt: np.ndarray) -> tuple[np.ndarray, float]:
        margins = []

        def margined(sequence, logits):
            margins.append(float(top_margin(logits, 1).min()))
            return logits

        with jax.enable_x64(new_val=True):
            sequence = greedy_walk(wide, weights, prompt[None], steps, margined)
        return sequence[0], min(margins)

    def drawn(seed: int) -> np.ndarray:
        return np.random.default_rng([0x5E7, seed]).integers(1, canonical.vocab_size, (prompts, 2)).astype(
            np.int32)

    rows = clearing(drawn, lambda rows: all(walked(row)[1] > least for row in rows))(0)
    for row, (tokens, length, log_probs) in zip(rows, serve([list(row) for row in rows], steps), strict=True):
        assert length > 0
        np.testing.assert_array_equal(tokens[:len(row) + length], walked(row)[0][:len(row) + length])
        assert np.isfinite(log_probs[:length]).all()


# --- pretrained fine-tuning ---------------------------------------------------------------------------------

def pretrained_trained(cell: Mapping, root: Path) -> Recorded:
    """A published decoder (a local HF-layout export; its checkpoint decides its geometry and its remat)
    fine-tuned in its full weights or a LoRA over them, quantization-aware where the cell trains so."""
    from test_lm_recipe import export_tiny_decoder

    from dew.lora import LoRA
    from dew.quantization import Quantization

    shared = lm_run(cell, root)
    config = LMRunConfig(data=shared.data, validation=shared.validation,
                         pretrained=str(export_tiny_decoder(root / "checkpoint")),
                         lora=LoRA(rank=4, modules=("q_proj",)) if cell["adapter"] == "lora" else None)
    if cell["quantization"] == "qat":
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer,
                                                                        quantization=Quantization()))
    return Recorded(saved_record(config, root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4"}), cell["devices"])


def pretrained(cell: Mapping, root: Path) -> None:
    """Load a published decoder, fine-tune it and reload the run: every step's gradient finite, and its task
    generating what its checkpoint generates; served quantized after training (PTQ, weight-only) where the
    cell serves so, its continuation its float64 forward's (`assert_served_as_its_forward`)."""
    from dew.quantization import Quantization

    run, _ = pretrained_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "pretrained fine-tuning")
    assert_generates_as_its_checkpoint(run)
    if cell["quantization"] == "ptq":
        task = dew.pipeline(str(run))
        served = task.quantized(Quantization(weight_only=True))
        numerics = numerics_of(cell["precision"])
        with run_precision(numerics):
            assert_served_as_its_forward(served, called_on(served), task.model, 1, numerics.dtypes.compute)


# --- diffusion language models -----------------------------------------------------------------------------

MASK = 255
"""The masked diffusion LM's mask id: the byte vocabulary's last."""
PROMPT = [5, 6, 7]


def masked_trained(cell: Mapping, root: Path) -> Recorded:
    """A masked diffusion LM over the shared LM data path, its documents packed or padded, a bidirectional
    decoder masking at the vocabulary's last id."""
    config = dataclasses.replace(lm_run(cell, root),
                                 objective=ObjectiveConfig("masked_diffusion", {"ema_decay": None}))
    return Recorded(saved_record(config, root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "0", "model.causal": "False",
                                          "model.mask_token_id": str(MASK)}), cell["devices"])


def block_trained(cell: Mapping, root: Path) -> Recorded:
    """A block-diffusion LM: DiffusionGemma's release fine-tuned on the shared LM data path's windows of a
    token store in the release's own tokenizer (`run_kinds.blockwise`), five batches of its rows of 12 ids,
    more than a run's steps read, its model's dtype left for the run's precision to derive."""
    config = blockwise(root, drawn=False, tokens=5 * 8 * 12)
    fields = {name: value for name, value in config.model.fields.items() if name != "dtype"}
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, fields=fields))
    return Recorded(saved_record(config, root, "cell"), settings(cell, root), cell["devices"])


DIFFUSION_LMS = {"masked": masked_trained, "block": block_trained}
"""Each diffusion LM's training run at a cell, by its objective."""


def diffusion_lm_trained(cell: Mapping, root: Path) -> Recorded:
    """The cell's diffusion LM trained (`DIFFUSION_LMS`)."""
    return DIFFUSION_LMS[cell["objective"]](cell, root)


def windowed(cell: Mapping) -> bool:
    """Whether a diffusion LM's cell is one its objective admits: a block-diffusion LM trains on the stream's
    windows, a span of it a row (its run refuses packed documents), and a masked one on documents packed or
    padded."""
    return (cell["objective"] == "block") == (cell["rows"] == "windows")


def diffusion_sampled(cell: Mapping, run: str) -> np.ndarray:
    """The run's sampler's continuation of a prompt (`dew.pipeline`): a masked LM's in two denoising steps,
    a block LM's canvases as its release's process refines them."""
    steps = {"steps": 2} if cell["objective"] == "masked" else {}
    return np.asarray(dew.pipeline(run)([PROMPT], 4, key=0, **steps).host().tokens)


def diffusion_lm(cell: Mapping, root: Path) -> None:
    """Train a masked or block diffusion LM and reload its run to a sampler: every step's gradient finite,
    and its sample keeping its prompt; a masked LM's leaving no position masked, and a block LM's what a
    generation over its final checkpoint's tree samples (export then reload is identical)."""
    run, _ = diffusion_lm_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"{cell['objective']} diffusion LM")
    tokens = diffusion_sampled(cell, str(run))
    assert list(tokens[0, :len(PROMPT)]) == PROMPT
    if cell["objective"] == "masked":
        assert (tokens[0, len(PROMPT):len(PROMPT) + 4] != MASK).all()
        return
    restored = dew.pipeline(str(run)).bind(Checkpoints(str(run)).variables())
    np.testing.assert_array_equal(tokens, np.asarray(restored([PROMPT], 4, key=0).host().tokens))


# --- hybrid and state-space models --------------------------------------------------------------------------

LENGTHS = (16, 32)
"""The two lengths a tiny decoder of each kind is built for, whose caches `recurrent` compares."""


def cache_shapes(kind: str, length: int, *, initialized: bool = False):
    """The leaf shapes of the cache a tiny decoder of mixer `kind` (`test_mixers.tiny`, at the kind's fields
    there, `KIND_FIELDS`) built for `length` states for a row (`cache_spec`, each holder's own statement of
    it): read with the parameters its statement makes alone (`init_with_output`, no forward: half a second
    for every kind at both lengths), or, `initialized`, over the decoder's whole variables as an
    initialization makes them (`jax.eval_shape` of its init, each kind's forward traced: 13 s)."""
    from test_mixers import tiny

    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=length, **overrides)

    def stated(module, rows):
        return module.cache_spec(rows)

    if initialized:
        variables = nn.unbox(jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
        spec = model.apply(variables, 1, method=stated)
    else:
        spec, _ = model.init_with_output(jax.random.key(0), 1, method=stated)
    return jax.tree.map(lambda leaf: tuple(leaf.shape), spec)


def recurrent_of(shapes) -> tuple[str, ...]:
    """The registry's kinds (`mixers.paths`) whose cache `shapes(kind, length)` states (`cache_shapes`) has
    the same leaf shapes built for either of `LENGTHS`: a state of its own size, where keys and values grow
    with the sequence; a kind that states no cache (a token-wise MLP's) holds no state to decode from."""
    stated = {kind: [shapes(kind, length) for length in LENGTHS] for kind in sorted(mixers.paths)}
    return tuple(kind for kind, (short, long) in stated.items() if jax.tree.leaves(short) and short == long)


@functools.cache
def recurrent() -> tuple[str, ...]:
    """The registry's recurrent mixer kinds (`recurrent_of`), each read by its cache statement alone
    (`cache_shapes`), as every task collecting this file reads them: over its initialized decoder's
    variables they took 13 s of each task's start (main's train 1089: this file's 112 tasks half their
    start; the owner's audit, 2026-11-20), which one test holds them to
    (`test_the_recurrent_kinds_are_those_an_initialized_decoder_states`)."""
    return recurrent_of(cache_shapes)


def test_the_recurrent_kinds_are_those_an_initialized_decoder_states():
    """The kinds each tiny decoder's cache statement read alone names recurrent (`recurrent`, the hybrid
    scenario's axis) are those its statement over its initialized decoder's variables names, as a decode
    reads them: a kind whose statement reads its parameters (a memory's momenta) is named alike."""
    assert recurrent() == recurrent_of(functools.partial(cache_shapes, initialized=True))


def mixer_settings(kind: str) -> dict[str, str]:
    """The `dew train` settings of a decoder of mixer `kind`: the kind's record (`to_record`) at its tiny
    fields (`test_mixers.KIND_FIELDS`), and the model's fields the kind needs beside it."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    return {"model.mixer": json.dumps(to_record(mixers.build(kind, **fields), MixerBase | None)),
            **{f"model.{name}": json.dumps(value) for name, value in overrides.items()}}


def hybrid_trained(cell: Mapping, root: Path) -> Recorded:
    """A decoder of the cell's recurrent mixer trained through the shared components."""
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4",
                                          **mixer_settings(cell["mixer"])}),
                    cell["devices"])


def hybrid(cell: Mapping, root: Path) -> None:
    """Train a decoder of the cell's recurrent mixer and decode it from its run: every step's gradient finite,
    and its task generating what its checkpoint generates."""
    run, _ = hybrid_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"a {cell['mixer']} decoder")
    assert_generates_as_its_checkpoint(run)


# --- distributed training -----------------------------------------------------------------------------------

def distributed_trained(cell: Mapping, root: Path) -> Recorded:
    """A dense decoder or a mixture of experts trained over a mesh of the cell's split, on two devices."""
    moe = {"model.mixture": json.dumps({"experts": 4, "top_k": 2})} if cell["model"] == "moe" else {}
    mesh = {"trainer.mesh.fsdp": "2"} if cell["mesh"] == "fsdp" else {"trainer.mesh.fsdp": "1",
                                                                      "trainer.mesh.expert": "2"}
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4", **moe, **mesh}), 2)


def distributed(cell: Mapping, root: Path) -> None:
    """Train a dense decoder or a mixture of experts over a mesh, its experts sharded where the cell shards
    them: every step's gradient finite and its run reloading."""
    run, _ = distributed_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"a {cell['model']} decoder over {cell['mesh']}")
    assert_generates_as_its_checkpoint(run)


# --- diffusion training -------------------------------------------------------------------------------------

def imaged(cell: Mapping, run: str) -> np.ndarray:
    """The run's pipeline's sample (`dew.pipeline`), two steps at one key."""
    return np.asarray(dew.pipeline(run)("", steps=2, key=0).host().images)


def diffusion(cell: Mapping, root: Path) -> None:
    """Train an image model, checkpointing it, and reload its run: every step's gradient finite, and the
    reloaded pipeline sampling what a pipeline over its final checkpoint's tree samples."""
    run, _ = called("diffusion_trained")(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "diffusion training")
    task = dew.pipeline(str(run))
    restored = task.bind(Checkpoints(str(run)).variables())
    np.testing.assert_array_equal(np.asarray(task("", steps=2, key=0).host().images),
                                  np.asarray(restored("", steps=2, key=0).host().images))


# --- representation learning --------------------------------------------------------------------------------

def jepa(cell: Mapping, root: Path) -> None:
    """Representation learning: a JEPA encoder trained and its representations read, in a process of the
    cell's devices (`jepa_trained`)."""
    called("jepa_trained")(cell, root).trained()


# --- online post-training -----------------------------------------------------------------------------------

LANGUAGE_RL = ("grpo", "rloo")
"""The rollout methods an online post-training cell trains a language policy by (`PromptRollout.method`); a
diffusion policy's are `scenario_called.DIFFUSION_RL`'s."""


def online_rl_trained(cell: Mapping, root: Path) -> Recorded | Called:
    """The cell's RL run. A language policy's: the recorded GRPO run (`rl_support.recorded_grpo_config`), its
    rollout's method the cell's (GRPO or RLOO), collecting rollouts from its policy and scoring them by the
    reward its import path names, its model's dtype left for the run's precision to derive. A diffusion
    policy's: a tiny DiT's flow RL run through Dew's API, in a fresh process
    (`scenario_called.diffusion_rl_trained`)."""
    import rl_support

    if cell["method"] in DIFFUSION_RL:
        return called("diffusion_rl_trained")(cell, root)
    config = rl_support.recorded_grpo_config(root, "--rollout.method", cell["method"])
    fields = {name: value for name, value in config.model.fields.items() if name != "dtype"}
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, fields=fields))
    return Recorded(saved_record(config, root, "cell"), settings(cell, root), cell["devices"],
                    trust=(rl_support.__name__,))


def online_rl(cell: Mapping, root: Path) -> None:
    """Language or diffusion RL: collect rollouts, score them, update the policy, as a user runs it (`dew
    train`, or Dew's API): every step's gradient finite. The run's scalars show no estimator, so which method
    it ran is not read here (the methods' own tests hold each)."""
    run, _ = online_rl_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"online {cell['method']}")


# --- generation and serving ---------------------------------------------------------------------------------

PROMPTS = (1, 3)
"""A served batch's prompts: one alone, or three together."""


SERVERS = {"paged": {"cache": PagedKV(page_size=16, pages=12)}, "speculative": {}}
"""Each way a cell serves its prompts through a server (`Server.from_task`), each prompt a request: over a
paged cache, pages of 16 tokens pooled over its rows; or drafting speculative blocks (its task's
`Speculative` strategy) over the dense cache a server speculates over."""
SERVING = ("batch", *SERVERS)
"""Each way a cell serves its prompts: one call of its task over them all (a batch), or a server's way
(`SERVERS`)."""


def serving_task(cell: Mapping) -> tuple[TextGeneration, nn.Module]:
    """The cell's served task and the canonical float model it serves: a decoder computing in the cell's
    dtype (`model_support.decoder`, its weights drawn), with a prediction depth its task drafts speculative
    blocks of three by where the cell speculates, quantized after training (PTQ, weight-only) where the cell
    serves so, traced at the task's default example, a row as long as a prediction depth reads (row 9669)."""
    from model_support import decoder

    from dew.quantization import Quantization

    speculative = cell["serving"] == "speculative"
    model = decoder().clone(max_seq_len=128, num_nextn_predict_layers=int(speculative))
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    canonical = model.clone(dtype=numerics_of(cell["precision"]).dtypes.compute)
    drafting = Speculative(block=3) if speculative else None
    task = TextGeneration(canonical, variables, sampling=Sampling(temperature=0, strategy=drafting))
    if cell["quantization"] is not None:
        task = task.quantized(Quantization(dtype=cell["quantization"], weight_only=True))
    return task, canonical


def served_through(cell: Mapping, task: TextGeneration) -> Callable[[list[list[int]], int], Draws]:
    """Prompts served the cell's way (`SERVING`) over `task`, greedily: `serve(prompts, steps)`."""
    if cell["serving"] == "batch":
        return called_on(task)

    def serve(prompts: list[list[int]], steps: int) -> Draws:
        server = Server.from_task(task, slots=len(prompts), capacity=64, **SERVERS[cell["serving"]])
        return [draws_of(generation)[0] for generation in server(prompts, steps, key=0)]
    return serve


def served(cell: Mapping, root: str) -> np.ndarray:
    """The cell's served greedy continuation of its batch's prompts, served its way (`served_through`): each
    row's prompt and new tokens, one row after another."""
    task, _ = serving_task(cell)
    prompts = [[3, 4], [5, 6], [8, 9]][:cell["prompts"]]
    with run_precision(numerics_of(cell["precision"])):
        draws = served_through(cell, task)(prompts, 4)
    return np.concatenate([tokens[:len(prompt) + length]
                           for prompt, (tokens, length, _) in zip(prompts, draws, strict=True)])


def serving(cell: Mapping, root: Path) -> None:
    """Generation and serving: a decoder computing in the cell's dtype, quantized after training as the cell
    serves, over one prompt or a batch of three, served in one call, by a server over a paged cache or by a
    server drafting speculative blocks: its greedy continuation its float64 forward's
    (`assert_served_as_its_forward`)."""
    task, canonical = serving_task(cell)
    with run_precision(numerics_of(cell["precision"])):
        assert_served_as_its_forward(task, served_through(cell, task), canonical, cell["prompts"],
                                     canonical.dtype)


# --- quantized interchange ----------------------------------------------------------------------------------

def quantized_written(cell: Mapping, root: Path) -> None:
    """The cell's quantized checkpoint where `scenario_called.quantized_loaded` reads it with `root`: a llama
    written as a GGUF file of the cell's encoding under `root`, where its format is one (`GGUF`); a codec
    checkpoint is read where its author wrote it."""
    import gguf
    from test_gguf import hf_tensors, write_gguf

    if cell["format"] in GGUF:
        rng = np.random.default_rng(0)
        write_gguf(root / "model.gguf", "llama", getattr(gguf.GGMLQuantizationType, cell["format"]),
                   hf_tensors("llama", rng), rng)


FORMATS = (*GGUF, "modelopt/tiny", "storage/nvfp4")
"""A quantized interchange cell's checkpoint formats: a GGUF file of each of its encodings, and ModelOpt's and
compressed-tensors' NVFP4 checkpoints as their authors wrote them (`scenario_called.CODECS`), ModelOpt's
quantizing its activations too and compressed-tensors' its weights alone."""
ACTIVATIONS_QUANTIZED = ("modelopt/tiny",)
"""The formats whose forward quantizes its activations as well as its weights."""


def assert_served_alike(task: TextGeneration, vocab: int) -> None:
    """A checkpoint that quantizes its activations served greedily (row 9704): three prompts of one width
    served by one call and again, and by a server over a paged cache. The repeated call draws what the
    first drew, bit for bit. Each server row draws the call's tokens, and every log-probability past the
    prompt's step bit for bit: each decode step reduces in one order whichever cache it reads, where the
    prompt's step heads a batch of another width. Every token is in the vocabulary, every row draws each of
    its steps, and every log-probability is finite. No forward is the truth of its continuation at a margin:
    an activation at its format's grid midpoint quantizes either way under another path's rounding
    (docs/design/testing.md)."""
    prompts, steps = [[3, 4, 5], [6, 7, 8], [9, 10, 11]], 4
    once = called_on(task)(prompts, steps)
    again = called_on(task)(prompts, steps)
    paged = served_through({"serving": "paged"}, task)(prompts, steps)
    for (tokens, length, log_probs), (tokens_again, length_again, log_probs_again), (
            served_tokens, served_length, served_log_probs) in zip(once, again, paged, strict=True):
        assert (length, length_again, served_length) == (steps, steps, steps)
        drawn = tokens[:len(prompts[0]) + steps]
        assert drawn.tobytes() == tokens_again[:len(drawn)].tobytes() == served_tokens[:len(drawn)].tobytes()
        assert log_probs[:steps].tobytes() == log_probs_again[:steps].tobytes()
        assert log_probs[1:steps].tobytes() == served_log_probs[1:steps].tobytes()
        assert ((drawn >= 0) & (drawn < vocab)).all()
        assert np.isfinite(log_probs[:steps]).all() and np.isfinite(served_log_probs[:steps]).all()


def quantized_source(cell: Mapping, root: Path):
    """The cell's quantized checkpoint (`quantized_written`) loaded (`scenario_called.quantized_loaded`)."""
    quantized_written(cell, root)
    return quantized_loaded(cell["format"], root)


def quantized_run(cell: Mapping, root: Path) -> Called:
    """The cell's quantized checkpoint (`quantized_written`) and its training run under `root`
    (`scenario_called.quantized_trained`), in a fresh process."""
    quantized_written(cell, root)
    return called("quantized_trained")(cell, root)


def quantized_served(cell: Mapping, root: str) -> np.ndarray:
    """The cell's quantized checkpoint served (`quantized_source`): its greedy continuation of a prompt."""
    source = quantized_source(cell, Path(root))
    task = TextGeneration(source.model, source.variables, sampling=Sampling(temperature=0))
    return np.asarray(task([[3, 4, 5]], 4, key=0).host().tokens)


def quantized_interchange(cell: Mapping, root: Path) -> None:
    """Quantized interchange: a quantized checkpoint of the cell's format (`FORMATS`) loaded, then trained on
    from its decoded weights (`quantized_trained`) or served: a weight-only format's greedy continuation its
    float64 forward's, one that quantizes its activations served alike by a call and a server
    (`assert_served_alike`)."""
    if cell["use"] == "train":
        quantized_run(cell, root).trained()
        return
    source = quantized_source(cell, root)
    task = TextGeneration(source.model, source.variables, sampling=Sampling(temperature=0))
    if cell["format"] in ACTIVATIONS_QUANTIZED:
        vocab = json.loads((CODECS / cell["format"] / "config.json").read_text())["vocab_size"]
        assert_served_alike(task, vocab)
        return
    assert_served_as_its_forward(task, called_on(task), source.model, 1, source.model.dtype)


# --- published diffusion pipelines --------------------------------------------------------------------------

def published_task(cell: Mapping, root: Path):
    """The cell's released pipeline, extracted under `root` and loaded at the cell's dtype."""

    from test_pipeline_streaming import PIPELINES

    from dew.interop.pretrained import Pretrained

    archive, inside = PIPELINES[cell["family"]]
    extract_fixture(Path(__file__).resolve().parents[1] / f"tests/fixtures/{archive}.tar.xz", root / archive)
    loaded = Pretrained.load(str(root / archive / inside), dtype=cell["dtype"])
    return loaded.text_to_image()


def pipeline_sampled(cell: Mapping, root: str) -> np.ndarray:
    """The cell's released pipeline's sample of a prompt, two steps at one key."""
    return np.asarray(published_task(cell, Path(root))(["a red bird"], steps=2, key=3).host().images)


def published_pipeline(cell: Mapping, root: Path) -> None:
    """Published diffusion pipelines: a tiny released pipeline loaded and sampled end to end at the cell's
    family, its images finite and the same each call at one key."""
    task = published_task(cell, root)
    first, second = (np.asarray(task(["a red bird"], steps=2, key=3).host().images, np.float32)
                     for _ in range(2))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def _pipelines() -> tuple[str, ...]:
    from test_pipeline_streaming import PIPELINES

    return tuple(PIPELINES)


PRECISIONS = ("32-true", "bf16-mixed")
"""The precisions a scenario's CPU cells train at (fp8 needs widths a multiple of 128)."""

SCENARIOS: Mapping[str, Scenario] = {
    "Dense language-model pretraining": Scenario(
        {"precision": PRECISIONS, "rows": ("padded", "packed"), "memory": (None, "full"), "devices": (1, 2),
         "read": ("memory", "indexed", "streamed")}, dense_lm, dense_lm_trained, "generated", laid=()),
    "Pretrained fine-tuning": Scenario(
        {"precision": PRECISIONS, "adapter": ("full", "lora"), "quantization": (None, "qat", "ptq"),
         "devices": (1, 2)}, pretrained, pretrained_trained, "generated", laid=()),
    "Diffusion language models": Scenario(
        {"objective": tuple(DIFFUSION_LMS), "precision": PRECISIONS, "rows": ("padded", "packed", "windows"),
         "devices": (1, 2)}, diffusion_lm, diffusion_lm_trained, "diffusion_sampled", admits=windowed,
        apart=("objective",)),
    "Hybrid and state-space models": Scenario(
        {"mixer": recurrent(), "precision": PRECISIONS, "rows": ("padded", "packed"), "devices": (1, 2)},
        hybrid, hybrid_trained, "generated", laid=("mixer",)),
    "Distributed training": Scenario(
        {"model": ("dense", "moe"), "mesh": ("fsdp", "expert"), "precision": PRECISIONS}, distributed,
        distributed_trained, "generated", laid=()),
    "Diffusion training": Scenario(
        {"precision": PRECISIONS, "quantization": (None, "qat"), "memory": (False, "full"),
         "devices": (1, 2)}, diffusion, called("diffusion_trained"), "imaged"),
    "Representation learning": Scenario(
        {"precision": PRECISIONS, "devices": (1, 2), "read": ("memory", "indexed", "streamed")}, jepa,
        called("jepa_trained")),
    "Online post-training": Scenario(
        {"method": (*LANGUAGE_RL, *DIFFUSION_RL), "precision": PRECISIONS, "devices": (1, 2)}, online_rl,
        online_rl_trained, apart=("method",)),
    "Generation and serving": Scenario(
        {"serving": SERVING, "precision": PRECISIONS,
         "quantization": (None, *typing.get_args(QuantizedDtype)), "prompts": PROMPTS}, serving,
        sampled="served", sampled_from="root", apart=("serving",)),
    "Quantized interchange": Scenario(
        {"format": FORMATS, "use": ("train", "serve")}, quantized_interchange, quantized_run,
        "quantized_served", "root"),
    "Published diffusion pipelines": Scenario(
        {"family": _pipelines(), "dtype": ("float32", "bfloat16")}, published_pipeline,
        sampled="pipeline_sampled", sampled_from="root"),
}
"""Every scenario of docs/design/testing.md's table this file runs, by its name there."""


def _named(cell: Mapping) -> str:
    return ",".join(f"{axis}={value}" for axis, value in cell.items())


CELLS = [pytest.param(name, cell, id=f"{name}-{_named(cell)}")
         for name, scenario in SCENARIOS.items() for cell in scenario.cells()]


@pytest.mark.parametrize(("name", "cell"), CELLS)
def test_a_scenarios_cell_holds_the_contracts_its_flow_touches(name, cell, tmp_path):
    """One cell of a scenario's covering set: its run at the cell's settings, held to its claims."""
    SCENARIOS[name].run(cell, tmp_path)


def first(name: str) -> dict:
    """A scenario's first cell, where it holds its rows claims (`LAID`)."""
    return SCENARIOS[name].cells()[0]


def first_cells(held: Callable[[Scenario], object]) -> list:
    """The first cells of every scenario `held` holds (`Scenario.firsts`), as a test's parameters: the
    scenario's name and the cell, named by the scenario and the values of its axes apart."""
    def named(name: str, cell: dict) -> str:
        apart = {axis: cell[axis] for axis in SCENARIOS[name].apart}
        return f"{name}-{_named(apart)}" if apart else name

    return [pytest.param(name, cell, id=named(name, cell))
            for name, scenario in SCENARIOS.items() if held(scenario) for cell in scenario.firsts()]


@pytest.mark.parametrize(("name", "cell"), first_cells(lambda scenario: scenario.trained))
def test_a_scenarios_run_resumed_ends_on_its_uninterrupted_run_from_its_compiled_programs(name, cell,
                                                                                          tmp_path):
    """Resume equals the uninterrupted run, and a resumed run reuses its compiled programs (testing.md's
    contracts), once a flow of a scenario on its first cell (`assert_resumes_onto_its_uninterrupted_run`); and
    where the scenario samples its trained run, a repeated run reuses its compiled programs, sampling
    (`samples_from_its_cached_programs`), over that uninterrupted run, where a test of its own trained the
    run again first (row 10137's cut)."""
    scenario = SCENARIOS[name]
    whole = assert_resumes_onto_its_uninterrupted_run(scenario.trained(cell, tmp_path), tmp_path)
    if scenario.sampled is not None and scenario.sampled_from == "run":
        samples_from_its_cached_programs(name, cell, whole, tmp_path)


def samples_from_its_cached_programs(name: str, cell: Mapping, source: Path, root: Path) -> None:
    """A repeated run reuses its compiled programs, sampling (testing.md's contracts), once a flow of a
    scenario on its first cell: its sampling of `source`, the cell's trained run or its root, where it samples
    what it writes there itself, twice in fresh processes over one persistent cache under `root`
    (`assert_samples_from_its_cached_programs`)."""
    assert_samples_from_its_cached_programs(MODULE, SCENARIOS[name].sampled, cell, str(source), root=root)


@pytest.mark.parametrize(("name", "cell"), first_cells(
    lambda scenario: scenario.sampled and (scenario.sampled_from == "root" or not scenario.trained)))
def test_a_scenarios_sampling_reruns_from_its_compiled_programs(name, cell, tmp_path):
    """A repeated run reuses its compiled programs, sampling, of what a scenario's first cell writes under its
    root (`samples_from_its_cached_programs`); a scenario that samples its trained run holds it beside its
    resume (`test_a_scenarios_run_resumed_ends_on_its_uninterrupted_run_from_its_compiled_programs`)."""
    samples_from_its_cached_programs(name, cell, tmp_path, tmp_path)


# --- a first step however its rows are laid or split ------------------------------------------------------

LENGTH, DOCUMENTS = 11, 6
"""A drawn document's tokens and a draw's documents: three documents fill a row of 32 inputs and their
targets whole, so packed rows hold no padding, and a row of one document's own length none either."""

ROWS = {"packed": ({"pack": "first-fit", "bins": 2}, 3 * LENGTH - 1, DOCUMENTS // 3),
        "separate": ({"pack": "first-fit", "segments": 1}, 3 * LENGTH - 1, DOCUMENTS),
        "unpadded": ({"pack": "first-fit", "segments": 1}, LENGTH - 1, DOCUMENTS)}
"""Each way the LM data path (`presets.lm`) lays a draw's documents: its pack, its rows' inputs and the rows
it makes of them. Packed beside each other; one a row, right-padded; one a row of its own length."""

LAID = {"packed": ("packed rows equal their documents run separately", ("packed", 1, 1),
                   ("separate", 1, 1)),
        "padded": ("right-padded rows equal unpadded ones on the real tokens", ("separate", 1, 1),
                   ("unpadded", 1, 1)),
        "split": ("rows split over two devices equal one device's read in two microbatches", ("packed", 2, 1),
                  ("packed", 1, 2))}
"""Each claim of a scenario's rows (testing.md's contracts): its statement, the arm it holds and its
reference, each the way its rows are laid, the devices they are split over and the microbatches a device
reads them in. The suite's CPU sums a whole batch's gradient reduction in float64 and rounds it once
(`WIDE_CPU_ARITHMETIC`'s widened reductions, which the gate keeps, `arm_arithmetic.summed_as_run`), where any
split of it, over devices or microbatches, rounds each part and their sum: no split holds a leaf whose error
is that one rounding (Mamba's A_log, three roundings to one, row 6459), so the rows over two devices are held
to one device reading them as two microbatches, the same split and the same roundings."""


def laid(rows: str, seed) -> dict:
    """Draw `seed`'s documents (`DOCUMENTS` of `LENGTH` token ids each) as the LM data path lays them `rows`'s
    way, every row in one batch."""
    from test_presets import first_batch

    pack, inputs, count = ROWS[rows]
    rng = np.random.default_rng([0x1A1D, int(seed)])
    documents = Memory([{"text": rng.integers(1, 250, LENGTH).astype(np.int32)} for _ in range(DOCUMENTS)])
    return first_batch(documents.pipe(presets.lm, None, seq_len=inputs, batch=count, **pack))


LAID_CELLS = [pytest.param(name, {**first(name), **dict(zip(scenario.laid, values, strict=True))}, site,
                           id=site)
              for name, scenario in SCENARIOS.items() if scenario.laid is not None
              for values in itertools.product(*(scenario.axes[axis] for axis in scenario.laid))
              for site in ["-".join([name.split()[0].lower(), *map(str, values)])]]
"""The scenarios whose runs read the LM data path, by their first cells (`Scenario.laid`): a hybrid's at
each recurrent mixer, each of which resets its state at a document's start in its own arithmetic, a
distributed run's its dense decoder's (a mixture's routing is a selection, which the rows claims' draws do not
hold clear of its boundaries, `selection_margins`; the parallelism matrix holds its layouts). The masked
diffusion LM draws one noise time a row, every document packed in it sharing it, where each run alone draws
its own, and a mask over a row's every position, a row right-padded drawing other masks for its real tokens
than its own length's: its rows are held over devices alone, by the parallelism matrix's sequence and fsdp
cells (tests/test_parallelism_matrix.py, dense_mdlm)."""

LAID_SEEDS = {"dense": {"packed": 8, "padded": 128, "split": 256},
              "pretrained": {"packed": 128, "padded": 128, "split": 128},
              "distributed": {"packed": 8, "padded": 128, "split": 256},
              "hybrid-gated_delta_net": {"packed": 128, "padded": 128, "split": 128},
              "hybrid-kimi_delta_attention": {"packed": 128, "padded": 128, "split": 256},
              "hybrid-lightning_attention": {"packed": 32, "padded": 128, "split": 256},
              "hybrid-mamba": {"packed": 8, "padded": 64, "split": 256},
              "hybrid-mamba2": {"packed": 16, "padded": 256, "split": 256},
              "hybrid-mamba3": {"packed": 256, "padded": 256, "split": 256},
              "hybrid-memory": {"packed": 256, "padded": 256, "split": 256},
              "hybrid-short_conv": {"packed": 8, "padded": 64, "split": 128}}
"""Each site's declared check-1 seeds, by its cell and claim: `seeds_for` paired, on the CPU lane at
5b91fd892, pretrained's and distributed's at 88edfcc4a (armada, 2026-11-15)."""


CHECK1_SHARDS = {
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-kimi_delta_attention-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba2-padded]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba2-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-memory-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba3-split]":
        {"seeds": 256, "per_task": 64},
}
"""The rows sites whose check-1 seeds run as map tasks (`check1_shards`), by node id, a literal the CI plan
reads as it reads a mark's (tools/armada/ci.py): each takes 62 to 118 s whole on the CPU lane (main's CI
durations) and 122 to 144 s under a loaded train's (queue train 922, row 6885), past the group target, its
setup some 15 to 20 s of it, so 64 seeds a task keeps each well inside it."""


def laid_sites(test: str) -> list:
    """Every rows site (`LAID_CELLS`, each claim of `LAID`), as `test`'s parameters: each marked to run its
    seeds as map tasks where `CHECK1_SHARDS` names its node, its declared count `LAID_SEEDS`'."""
    def param(cell, claim: str):
        name, values, site = cell.values
        held = CHECK1_SHARDS.get(f"{test}[{site}-{claim}]")
        assert held is None or held["seeds"] == LAID_SEEDS[site][claim], (site, claim, held)
        return pytest.param(name, values, site, claim, id=f"{site}-{claim}",
                            marks=[pytest.mark.check1_shards(**held)] if held else [])

    return [param(cell, claim) for cell in LAID_CELLS for claim in LAID]


@pytest.mark.parametrize(("name", "cell", "site", "claim"),
                         laid_sites("test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid"))
def test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid(name, cell, site, claim, tmp_path):
    """Packed rows equal their documents run separately, right-padded rows equal unpadded ones on the real
    tokens, and rows split over devices equal one device's read in as many microbatches (testing.md's
    contracts, `LAID`): the scenario's first step, its loss and the gradient it hands the optimizer, as its
    own run builds its objective (`RunConfig.prepare`, at the cell's settings, its model in float64 for the
    truth) over the LM data path's batches of drawn documents (`laid`), on par by check 1
    (`placement_parity.layouts_agree`) over the site's declared seeds (`LAID_SEEDS`)."""
    recorded = SCENARIOS[name].trained(cell, tmp_path)
    run = RunConfig.load(str(recorded.record))
    given = settled({**recorded.settings, **Phase().settings()})

    def objective_of(wide: bool):
        return run.override({**given, "model.dtype": "float64" if wide else "float32"}).prepare().objective

    stated, arm, reference = LAID[claim]
    layouts_agree(first_steps(objective_of, laid, 64, reference), lambda seed: (np.int32(seed),),
                  f"{name}: {stated}", arm, reference, seeds=LAID_SEEDS[site][claim])
