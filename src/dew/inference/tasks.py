"""Tasks that hold a model, its weights and its host processing for repeated generation.

A task holds what a generation needs besides the request itself: the native
model, a frozen variables mapping and, when the source ships one, the host
processor that turns text and media into `ModelInputs`. A call takes the same
typed controls training uses (`Sampling`, `BlockProcess`) and returns the
typed records the sampling kernels produce. `bind` returns the same task over
other weights, so a training loop can draw from a policy snapshot without
holding the trainer's mutable mapping. The task uses the array buffers of the
variables it is given without copying them, so do not mutate, donate or
delete those buffers while the task is using them.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze
from jax.typing import ArrayLike, DTypeLike

from dew.cache import persist_compilations
from dew.coordination import agreed
from dew.diffusion.block import BlockProcess, CanvasGeneration, refuse_non_denoiser
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask, refuse_causal
from dew.nn.inputs import Media, ModelInputs, mesh_of, request_key
from dew.nn.protocols import CacheCapacity, TokenModel
from dew.objectives.base import Variables, thaw
from dew.records import integer, json_value, record as named_fields, text as named
from dew.sampling import decoding, vocabulary
from dew.sampling.decoding import LogitsTransform, Stopping
from dew.sampling.strategies import Strategy
from dew.sampling.text import Bounded, Criteria, Generation, Sampling, Transforms, generate, with_ids_of
from dew.telemetry.profile import region

if TYPE_CHECKING:
    from dew.config import ModelConfig
    from dew.training.distributed import Layout, MeshSpec
    from dew.training.quantization import Quantization

Rows = ModelInputs | ArrayLike | Sequence[Sequence[int]]
Request = str | Sequence[str] | Rows

SHAPE_BUCKETS = tuple(1 << exponent for exponent in range(21))
"""The shapes a text request is rounded up to: powers of two.

Every distinct prompt width, budget and continuation count costs its own XLA
compile of several seconds. To share compiles, a call pads the prompt on the
left, behind the attention mask, to the smallest bucket that holds it and is
at least 64. It rounds the budget up to the next power of two and trims the
extra decode steps off the result. The call's cache is the padded prompt plus
the rounded budget, rounded up again, and the call uses it in place of the
model's whole `max_seq_len`. The number of rows is not bucketed. A request
whose buckets would exceed `max_seq_len` keeps its own shapes, so the model
refuses exactly the requests it would refuse without buckets. A request with
media or multi-axis positions also keeps its own shapes.
"""


class Processor(Protocol):
    """Interface for a loaded source's host processor, which prepares requests and decodes tokens.

    Calling it turns text, with optional images, into `ModelInputs`.
    `decode` turns token rows into one string per row, and `bos_id` is the
    id a sequence starts with, or None.
    """

    def __call__(self, text: str | Sequence[str], *, images: Media | None = None) -> ModelInputs: ...

    def decode(self, tokens: ArrayLike) -> list[str]: ...

    @property
    def bos_id(self) -> int | None: ...


def prepared_inputs(processor: Processor | None, request: Request, *, images: Media | None) -> ModelInputs:
    """Turn a request into `ModelInputs`, tokenizing text through `processor`.

    Token rows pass through with their ids and row order unchanged; only text
    reaches the processor.
    """
    if isinstance(request, str):
        text = [request]
    elif isinstance(request, Sequence) and request and all(isinstance(entry, str) for entry in request):
        text = [entry for entry in request if isinstance(entry, str)]
    else:
        text = None
    if text is not None:
        if processor is None:
            raise ValueError("text requests need a processor: pass processor= where the task is built, "
                             "as objective.pipeline(state, processor=source.processor) does, or request "
                             "ModelInputs or token rows")
        return processor(text, images=images)
    if images is not None:
        raise ValueError(
            "images travel with text through the processor; prepared rows carry them in ModelInputs"
        )
    if isinstance(request, ModelInputs):
        return ModelInputs.from_value(request)
    if isinstance(request, Sequence):
        for row in request:
            if isinstance(row, str):
                raise ValueError("a request cannot mix text prompts and numeric token rows")
            if isinstance(row, np.ndarray) and not np.issubdtype(row.dtype, np.integer):
                raise ValueError("token rows must contain integers")
            if isinstance(row, Sequence) and any(
                    isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer))
                    for token in row):
                raise ValueError("token rows must contain integers, not coerced token IDs")
        return ModelInputs.from_value(np.asarray(request))
    # A resident array stays where it is; a global one cannot be fetched.
    return ModelInputs.from_value(request)


def _task_inputs(processor: Processor | None, request: Request, *, images: Media | None,
                 collective: bool, max_new_tokens: int | None, default_tokens: int | None,
                 max_length: int | None, key: int | jax.Array | None) -> tuple[ModelInputs, int, jax.Array]:
    def prepared() -> tuple[ModelInputs, int, jax.Array]:
        """Tokenize the request, size its budget and draw its key."""
        random_key = request_key(key)
        inputs = prepared_inputs(processor, request, images=images)
        return inputs, _budget(max_new_tokens, default_tokens, max_length,
                               inputs.tokens.shape[1]), random_key

    return agreed("inference task input preparation", prepared) if collective else prepared()


def decoded_rows(
    processor: Processor | None, tokens: ArrayLike, lengths: ArrayLike, width: int
) -> tuple[str, ...]:
    if processor is None:
        return ()
    rows, counts = np.asarray(tokens), np.asarray(lengths)
    return tuple(processor.decode(rows[row:row + 1, width:width + int(counts[row])])[0]
                 for row in range(rows.shape[0]))


def _budget(requested: int | None, default: int | None, max_length: int | None, prompt_width: int) -> int:
    if requested is not None:
        return requested
    if default is not None:
        return default
    if max_length is not None:
        if type(max_length) is not int or max_length < prompt_width:
            raise ValueError("source max_length must be an integer at least as large as the prompt width")
        return max_length - prompt_width
    raise ValueError("max_new_tokens is required; the source declares no default budget")


def shape_bucket(value: int, smallest: int) -> int:
    """Return the smallest shape bucket that holds `value`, never below `smallest`."""
    for bucket in SHAPE_BUCKETS:
        if bucket >= value and bucket >= smallest:
            return bucket
    return value


def cache_ceiling(model: nn.Module) -> int | None:
    """Return the largest cache the model admits, or None where it declares none.

    A `Bounded` decoder declares `max_seq_len`, and a wrapper answers for
    the decoder it holds; `dew.sampling.text._validated` reads the same
    field to refuse a request too large for it.
    """
    declared = model.max_seq_len if isinstance(model, Bounded) else None
    return declared if type(declared) is int else None


def _bucketed(inputs: ModelInputs, budget: int, ceiling: int | None
              ) -> tuple[ModelInputs, int, int | None]:
    """Return the request at bucket shapes: padded inputs, scan trips and cache capacity.

    A capacity of None leaves the request its own shapes and the model its
    own cache. That is what a request too large for the ceiling gets, so it
    is refused where it is refused today, and what one carrying media or
    logical positions gets: a filler slot holds no media feature and no
    multi-axis coordinate. One-axis positions, which every source
    processor's text rows carry, pad like validity: the filler is invalid,
    so its position is never read, and the real tokens keep theirs.
    """
    if ceiling is None or budget < 1:
        return inputs, budget, None
    positions = inputs.token_fields.get("positions")
    if (set(inputs.token_fields) - {"attention_mask", "positions"} or inputs.conditioning
            or (positions is not None and positions.shape != inputs.tokens.shape)):
        return inputs, budget, None
    width = shape_bucket(inputs.tokens.shape[1], 64)
    trips = shape_bucket(budget, 1)
    capacity = shape_bucket(width + trips, 64)
    if capacity > ceiling:
        return inputs, budget, None
    return _padded(inputs, width), trips, capacity


def _padded(inputs: ModelInputs, width: int) -> ModelInputs:
    """Return `inputs` left-padded to `width` slots, with the filler marked invalid."""
    extra = width - inputs.tokens.shape[1]
    if extra < 1:
        return inputs
    valid = inputs.token_fields.get("attention_mask")
    if valid is None:
        valid = jnp.ones(inputs.tokens.shape, bool)
    left = ((0, 0), (extra, 0))
    fields = {name: jnp.pad(value, left) for name, value in inputs.token_fields.items()}
    return replace(inputs, tokens=jnp.pad(inputs.tokens, left),
                   token_fields={**fields, "attention_mask": jnp.pad(valid, left)})


@functools.cache
def cache_sized(model: nn.Module, capacity: int | None) -> nn.Module:
    """Return `model` with a decode cache of `capacity` slots, one model per capacity.

    A `CacheCapacity` model gives the same model at another cache size, a
    wrapper sizing the decoder it holds; any other model keeps its own cache.
    The models are kept because the model is a static argument of the
    compiled generation: one object per capacity is one compile per
    capacity rather than one per call.
    """
    if capacity is None or not isinstance(model, CacheCapacity):
        return model
    return model.with_cache_capacity(capacity)


def _requested(generated: Generation, budget: int, padding: int) -> Generation:
    """Return `generated` cut back to the shapes the caller asked for.

    A bucket pads the prompt on the left and scans past the budget, so the
    filler comes off the front of the rows and the extra trips off the back.
    A row the scan stopped in one of those trips did not stop inside the
    budget: it reaches the caller at the budget's length, unterminated,
    which is what the unbucketed scan reports for it.
    """
    trips = generated.behavior_log_probs.shape[1]
    if not padding and trips == budget:
        return generated
    lengths = generated.lengths
    return replace(generated,
                   tokens=generated.tokens[:, padding:generated.tokens.shape[1] - trips + budget],
                   lengths=jnp.minimum(lengths, budget),
                   terminated=generated.terminated & (lengths <= budget),
                   behavior_log_probs=generated.behavior_log_probs[:, :budget],
                   raw_log_probs=generated.raw_log_probs[:, :budget])


def _pulled(repo_id: str, revision: str | None) -> str:
    """Download a run directory published to the Hub and return its local path."""
    import os

    from dew.interop.hub import pull_from_hub
    return os.fspath(pull_from_hub(repo_id, revision=revision))


class SavedRun(NamedTuple):
    """One checkpoint of a run, pinned: its inference declaration, not the
    training configuration, and the exact step it is, which a loader reads
    the weights at too, so a step saved meanwhile cannot pair one step's
    declaration with another's weights."""

    record: Mapping[str, object]
    step: int


def run_record(directory: str, step: int | str | None = None, trust: Sequence[str] = ()) -> SavedRun:
    """The selected checkpoint of the run in `directory`, pinned (`SavedRun`).
    A task loaded from it compiles into the persistent cache
    (`persist_compilations`). `trust` names the packages outside Dew whose
    modules the record may import (`dew.registry.imported`)."""
    from dew.checkpoints import Checkpoints
    from dew.registry import import_trusted
    persist_compilations()
    checkpoints = Checkpoints(directory)
    step = checkpoints.pinned(step)
    record = checkpoints.artifact(step)
    import_trusted(record, trust)
    if record is None:
        raise ValueError("this checkpoint's objective declares no inference record; declare "
                         "Objective.inference_record, or call objective.pipeline(state)")
    record = named_fields(record, 'checkpoint artifact')
    if 'unrecorded' in record:
        raise ValueError(f"this run's checkpoints describe no model to load: {record['unrecorded']}")
    return SavedRun(record, step)


def saved_model(record: Mapping[str, object], dtype: DTypeLike | None) -> ModelConfig:
    """Read the run's model record, with `dtype` overriding the computation it saved."""
    from dew.config import ModelConfig
    from dew.registry import dtype_name, resolve_dtype

    config = ModelConfig.from_dict(named_fields(record["model"], "model"))
    return config.with_dtype(dtype_name(resolve_dtype(dtype)))


def recorded_tokenizer(processor: Processor | None) -> str | None:
    """The tokenizer name a run's record keeps for `processor`, which
    `_saved_processor` rebuilds through `tokenizer_for`: a run tokenizer's
    own name, byte or Hugging Face. Any other processor records none, and the
    run loads as weights that take ids."""
    from dew.data.text import ByteTokenizer, HFTokenizer
    from dew.inference.pipeline import RunProcessor

    if isinstance(processor, RunProcessor) and isinstance(processor.tokenizer, ByteTokenizer | HFTokenizer):
        return processor.tokenizer.name
    return None


def _saved_processor(record: Mapping[str, object]) -> Processor | None:
    """Build the run's tokenizer into a task's host processor."""
    from dew.data.text import tokenizer_for
    from dew.inference.pipeline import RunProcessor

    tokenizer = record.get('tokenizer')
    return None if tokenizer is None else RunProcessor(tokenizer_for(named(tokenizer, "tokenizer")))


def _saved_budget(record: Mapping[str, object]) -> int | None:
    """Return how many tokens the run's own previews drew, where it drew any."""
    budget = record.get("max_new_tokens")
    if budget is None:
        return None
    if type(budget) is not int or budget < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    return budget


def _saved_run(directory: str, dtype: DTypeLike | None, step: int | str | None, trust: Sequence[str]
               ) -> tuple[Mapping[str, object], int, ModelConfig, Processor | None]:
    """Read a run's record, the exact step it is, its model config at `dtype`, and its host processor."""
    record, step = run_record(directory, step, trust)
    return record, step, saved_model(record, dtype), _saved_processor(record)


def _freeze_variables(task: TextGeneration | BlockGeneration | MaskedGeneration,
                      variables: Variables) -> None:
    """Freeze `variables` onto `task`, whose own `__post_init__` cannot assign.

    A tree split for training, an adapter's or `dew.objectives.base.freeze`'s,
    is read whole. A frozen dataclass refuses attribute assignment, so the
    field is written through `object.__setattr__`.
    """
    object.__setattr__(task, "variables", freeze(dict(thaw(variables))))


def _canvas_text(processor: Processor | None, generation: CanvasGeneration,
                 trace: str) -> tuple[str, ...]:
    """Decode a canvas generation's rows past their prompt, traced under `trace`.

    A generation with no prompt width names no boundary to decode from, so it
    is refused rather than decoded from the start of the canvas.
    """
    with region(trace):
        if generation.prompt_width is None:
            raise ValueError("this generation has no prompt width")
        rows = generation.host()
        return decoded_rows(processor, rows.tokens, rows.lengths, generation.prompt_width)


def _saved_sampling(record: Mapping[str, object], budget: int | None) -> Sampling:
    """Return the policy the run drew its previews under, or the basic one."""
    controls = record.get("sampling")
    if controls is None:
        if budget:
            raise ValueError("run.json lacks the sampling policy for its text previews")
        return Sampling()
    if not isinstance(controls, dict):
        raise ValueError("the run's sampling policy must be a Sampling record")
    return Sampling(**controls)


@dataclass(frozen=True)
class TextGeneration:
    """Generates next tokens from a decoder, its weights and its processor.

    A call runs the shared cached prefill and decode, and returns the
    `Generation` a training rollout consumes, with the log-probability of
    every drawn action under the policy that drew it and under the raw model.
    When a call passes none, it uses the task's `sampling` policy, its
    `max_new_tokens` budget and its `n` continuations per prompt; a loaded
    source sets all three from its generation config. The `n` continuations
    of a prompt come back as `n` consecutive rows, in prompt order. The
    weights stay where they were placed. On a mesh, the rows are split over
    its batch axes and the results keep that sharding.

    `logits` is the whole logits transform chain, and None means the chain
    `sampling` compiles to. `stopping` holds the criteria that run beside
    the policy's EOS and stop-string criteria, and `strategy` is the device
    loop. A call that passes any of the three replaces the task's value
    whole, so to add a transform you pass
    `logits=task.sampling.transforms() + (mine,)`. A call that passes
    `sampling=` also drops the task's own `logits` chain, because that chain
    was built from the policy the call replaces. Where a call's policy
    leaves the EOS and pad ids None, it takes the task's. So to change one
    control, pass `sampling=replace(task.sampling, repetition_penalty=1.1)`
    or a fresh `Sampling(...)`, and either one still stops where the task
    does. `sampling.stop` strings compile against `processor` once for the
    task's policy, and again on each call whose policy has other stop strings.

    A call pads the request to `SHAPE_BUCKETS` shapes, with a cache sized to
    the buckets, and returns results at the shapes the request asked for. So
    two requests of nearby lengths share one compiled executable, and neither
    pays for the model's whole context.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None = None
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1
    logits: tuple[LogitsTransform, ...] | None = None
    stopping: tuple[Stopping, ...] = ()
    strategy: Strategy | None = None
    _stops: tuple[Stopping, ...] = dataclasses.field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _freeze_variables(self, self.variables)
        object.__setattr__(self, "_stops", self._stop_criteria(self.sampling.stop))

    def _stop_criteria(self, strings: tuple[str, ...]) -> tuple[Stopping, ...]:
        """Compile stop strings against the processor's vocabulary, sized for the model's head."""
        if not strings:
            return ()
        if not isinstance(
                self.processor, (vocabulary.Referencing, vocabulary.Tokenizing, vocabulary.Vocabulary)):
            raise ValueError("stop strings compile against the task's processor, and this task has none "
                             "that lists its vocabulary; pass processor=")
        return (decoding.stop_strings(self.processor, strings,
                                      self.model.vocab_size if isinstance(self.model, Bounded) else None),)

    def _controls(self, sampling: Sampling | None, logits: Transforms | None, stopping: Criteria | None
                  ) -> tuple[Sampling, Transforms | None, tuple[Stopping, ...]]:
        """The policy, chain and criteria a call runs, the stop strings compiled
        into criteria and the EOS and pad ids filled from the task's policy."""
        policy = self.sampling if sampling is None else with_ids_of(sampling, self.sampling)
        stops = self._stops if policy.stop == self.sampling.stop else self._stop_criteria(policy.stop)
        chain = (self.logits if sampling is None else None) if logits is None else logits
        criteria = decoding.components(self.stopping if stopping is None else stopping) + stops
        return replace(policy, stop=(), pad_id=policy.pad), chain, criteria

    def bind(self, variables: Variables) -> TextGeneration:
        """Return the same task over other weights, such as a policy snapshot."""
        return replace(self, variables=variables)

    def quantized(self, spec: Quantization, example: Rows = ((0,),)) -> TextGeneration:
        """Return the task with the weights matched by `spec` stored as int8 or fp8 through Qwix.

        Requires `dewml[quantization]`. `example` is one prepared model
        input, token rows or `ModelInputs`, for Qwix's abstract trace; for a
        multimodal model it must include the media fields too. The processor
        and decoding controls stay unchanged. Weights held as host NumPy
        arrays are quantized one kernel at a time on the default device and
        then return to host storage. If one kernel will not fit on that
        device, load the weights onto the task's mesh before quantizing them.
        """
        from dew.training.quantization import quantize_for_serving

        inputs = prepared_inputs(None, example, images=None)
        model, variables = quantize_for_serving(self.model, self.variables, spec,
                                                inputs.tokens, **inputs.kwargs())
        return replace(self, model=model, variables=variables)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> TextGeneration:
        """Load the causal language-model run in `directory` as a task.

        The task rebuilds the model that the run's `run.json` records, the
        way the recipe built it, over the weights of the latest checkpoint
        (or of `step`), and decodes text through the run's own tokenizer.
        The run's preview budget and sampling policy become the task's
        defaults, and a PPO run loads only its policy weights.

        `ema` selects the averaged weights; None reads them when the run
        kept them. When the run's objective keeps its average as a reference
        policy, the task always loads the trained weights. With `mesh`, the
        weights are restored directly onto that mesh under `layout`, the way
        the trainer places them. `dtype` overrides the computation dtype and
        `param_dtype` the parameter storage dtype; None keeps what the
        checkpoint stored. `trust` names the packages outside Dew the run's
        record may import, as `trust_remote_code` does in transformers.
        """
        from dew.registry import objectives

        record, step, model_config, processor = _saved_run(directory, dtype, step, trust)
        budget = _saved_budget(record)
        variables = objectives[named(record["objective"], "objective")]._saved_variables(
            directory, step=step, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        model = model_config.build()
        return cls(model, variables, processor, sampling=_saved_sampling(record, budget),
                   max_new_tokens=budget if budget else None)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        ema: bool | None = None, step: int | str | None = None,
                        mesh: MeshSpec | None = None, layout: Layout | None = None,
                        dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None, trust: Sequence[str] = ()) -> TextGeneration:
        """Load a run directory published to the Hugging Face Hub.

        You publish one by uploading the run directory itself with
        `HfApi().upload_folder`. `revision` pins a Hub revision, and the
        other arguments work as in `from_run`.
        """
        return cls.from_run(_pulled(repo_id, revision), ema=ema, step=step, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype, trust=trust)


    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 sampling: Sampling | None = None, images: Media | None = None,
                 logits: Transforms | None = None, stopping: Criteria | None = None,
                 strategy: Strategy | None = None) -> Generation:
        with region("inference.text"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images,
                                          collective=mesh_of(self.variables) is not None,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key)
            shaped, trips, capacity = _bucketed(inputs, budget, cache_ceiling(self.model))
            policy, chain, criteria = self._controls(sampling, logits, stopping)
            generated = generate(cache_sized(self.model, capacity), self.variables, shaped, trips,
                                 key=random_key, sampling=policy, n=self.n if n is None else n, logits=chain,
                                 stopping=criteria, strategy=self.strategy if strategy is None else strategy)
            decoder = None if self.processor is None else functools.partial(decoded_rows, self.processor)
            padding = shaped.tokens.shape[1] - inputs.tokens.shape[1]
            return replace(_requested(generated, budget, padding), decoder=decoder)

    def decode(self, generation: Generation) -> tuple[str, ...]:
        """Return each row's valid continuation as text, empty without a processor."""
        with region("inference.text.decode"):
            rows = generation.host()
            return decoded_rows(self.processor, rows.tokens, rows.lengths, generation.prompt_width)


@dataclass(frozen=True)
class BlockGeneration:
    """Generates block-diffusion canvases from a block denoiser and its weights.

    A call runs prefill, refinement and the commits of clean tokens as one
    device computation. It returns a `CanvasGeneration`, which has no
    autoregressive likelihoods. When a call passes none, it uses the task's
    `process` (the published sampler configuration), its `max_new_tokens`
    budget and its `n` continuations per prompt. The `n` continuations of a
    prompt come back as `n` consecutive rows, in prompt order. The model
    is a `dew.nn.protocols.BlockDenoiser`, as DiffusionGemma is.
    """

    model: nn.Module
    variables: Variables
    process: BlockProcess
    processor: Processor | None = None
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    def __post_init__(self) -> None:
        _freeze_variables(self, self.variables)

    def bind(self, variables: Variables) -> BlockGeneration:
        """Return the same task over other weights."""
        return replace(self, variables=variables)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> BlockGeneration:
        """Load the block-diffusion run in `directory` as a task.

        The task rebuilds the block denoiser that the run's `run.json`
        records, over the weights of the latest checkpoint (or of `step`),
        and samples with the run's saved `BlockProcess` over the canvas the
        model declares. The run's preview budget becomes the task's
        `max_new_tokens`. A run whose model is not a `BlockDenoiser` raises
        `TypeError` naming the operations it lacks.

        The arguments work as in `TextGeneration.from_run`: `ema` selects
        the averaged weights, `mesh` and `layout` place them, and the two
        dtypes override computation and storage.
        """
        from dew.diffusion.block import BlockProcess
        from dew.registry import from_record, objectives
        record, step, model_config, processor = _saved_run(directory, dtype, step, trust)
        model = model_config.build()
        refuse_non_denoiser(model)
        variables = objectives[named(record["objective"], "objective")]._saved_variables(
            directory, step=step, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        process = from_record(BlockProcess, json_value(record['process'], 'process'), dtypes=False)
        return cls(model, variables, process, processor,
                   max_new_tokens=_saved_budget(record) or None)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        ema: bool | None = None, step: int | str | None = None,
                        mesh: MeshSpec | None = None, layout: Layout | None = None,
                        dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None, trust: Sequence[str] = ()) -> BlockGeneration:
        """Load a run directory published to the Hugging Face Hub.

        You publish one by uploading the run directory itself with
        `HfApi().upload_folder`. `revision` pins a Hub revision, and the
        other arguments work as in `from_run`.
        """
        return cls.from_run(_pulled(repo_id, revision), ema=ema, step=step, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype, trust=trust)


    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 process: BlockProcess | None = None, images: Media | None = None) -> CanvasGeneration:
        with region("inference.block"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key)
            generated = (self.process if process is None else process).generate(
                self.model, self.variables, inputs, budget, key=random_key,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            decoder = None if self.processor is None else functools.partial(decoded_rows, self.processor)
            return replace(generated, decoder=decoder)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Return each row's valid continuation as text, empty without a processor."""
        return _canvas_text(self.processor, generation, "inference.block.decode")


@dataclass(frozen=True)
class MaskedGeneration:
    """Samples a whole response with native MDLM, holding the prompt fixed.

    It does not reproduce the source-specific remasking recipes of LLaDA or
    Dream. EOS trims the finished response after the bidirectional
    denoising, and does not end that denoising early. Results hold
    refinement counts and no autoregressive action likelihoods. When a call
    passes none, it uses the task's `steps`, its `max_new_tokens` response
    length and its `n` continuations per prompt.
    """

    model: nn.Module
    variables: Variables
    process: DiscreteProcess
    processor: Processor | None = None
    solver: Unmask = dataclasses.field(default_factory=Unmask)
    steps: int = MDLM_STEPS
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    def __post_init__(self) -> None:
        _freeze_variables(self, self.variables)

    def bind(self, variables: Variables) -> MaskedGeneration:
        """Return the same native MDLM task over another weight snapshot."""
        return replace(self, variables=variables)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> MaskedGeneration:
        """Load the masked-diffusion run in `directory` as a task.

        The task rebuilds the bidirectional model that the run's `run.json`
        records, over the weights of the latest checkpoint (or of `step`),
        and refines responses with MDLM over the run's own mask token, using
        the solver and step count the run saved. The model must score the
        whole row with every position reading the others (`Logits`, and
        `TokenModel` with `causal` False) and name a mask token (`TokenModel.mask_token_id`)
        that matches the saved process, or loading raises `ValueError`.

        The arguments work as in `TextGeneration.from_run`, and the run's
        preview budget becomes the response length a call omits.
        """
        from dew.diffusion.discrete import DiscreteProcess
        from dew.registry import from_record, objectives, solvers

        record, step, model_config, processor = _saved_run(directory, dtype, step, trust)
        budget = _saved_budget(record)
        model = model_config.build()
        refuse_causal(model)
        mask_id = model.mask_token_id if isinstance(model, TokenModel) else None
        if type(mask_id) is not int:
            raise ValueError(f"a saved masked run requires a model that names its mask token "
                             f"(TokenModel.mask_token_id), and this {type(model).__name__} names none")
        variables = objectives[named(record["objective"], "objective")]._saved_variables(
            directory, step=step, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        process = from_record(DiscreteProcess, json_value(record['process'], 'process'), dtypes=False)
        if process.mask_id != mask_id:
            raise ValueError("model and process mask token disagree")
        solver = named_fields(record['solver'], 'solver')
        unmask = solvers.build(named(solver['class'], 'solver'), named_fields(solver['fields'], 'fields'))
        if not isinstance(unmask, Unmask):
            raise ValueError("a saved masked run requires an Unmask solver")
        return cls(model, variables, process, processor, solver=unmask,
                   steps=integer(record['sampling_steps'], 'sampling_steps'),
                   pad_token_id=integer(record.get("pad_token_id", 0), "pad_token_id"),
                   max_new_tokens=budget or None)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        ema: bool | None = None, step: int | str | None = None,
                        mesh: MeshSpec | None = None, layout: Layout | None = None,
                        dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None, trust: Sequence[str] = ()) -> MaskedGeneration:
        """Load a run directory published to the Hugging Face Hub.

        You publish one by uploading the run directory itself with
        `HfApi().upload_folder`. `revision` pins a Hub revision, and the
        other arguments work as in `from_run`.
        """
        return cls.from_run(_pulled(repo_id, revision), ema=ema, step=step, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype, trust=trust)


    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 steps: int | None = None, images: Media | None = None) -> CanvasGeneration:
        with region("inference.masked"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                max_length=self.max_length, key=key)
            generated = self.process.generate(self.model, self.variables, inputs, budget, key=random_key,
                solver=self.solver, steps=self.steps if steps is None else steps,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            decoder = None if self.processor is None else functools.partial(decoded_rows, self.processor)
            return replace(generated, decoder=decoder)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Return each row's valid response as text, empty without a processor."""
        return _canvas_text(self.processor, generation, "inference.masked.decode")


__all__ = ["SHAPE_BUCKETS", "BlockGeneration", "MaskedGeneration", "Processor", "TextGeneration"]
