"""Run a saved run as an lm-evaluation-harness model.

`DewLM` puts a `TextGeneration` behind the three calls lm-eval-harness's `LM`
interface asks for, so any task suite runs against a run directory. The
trainer's own perplexity says how well a run predicts its training data and
nothing about what it can do, which is the other question a suite answers.

`lm_eval` is an optional extra (`pip install dew-ml[eval-harness]`), so this
module is the only one that imports it and `dew.eval` does not import this
module: a caller who never asks for a harness never needs it installed.
Importing this module registers the adapter under `dew`, which is what the
harness's registry reads, and lm_eval 0.4 has no plugin discovery of its
own, so the import has to happen in the process that runs the command:

    python -m dew.eval --model dew --model_args run=runs/shakespeare \\
        --tasks hellaswag --limit 4

is `lm_eval`'s own command line with this module imported first. In a
program that already imported it, plain `lm_eval --model dew` finds it too.

Scoring is the model's own forward under `jax.jit`, log-softmax over its
logits, read at the targets each row's continuation names. That is the
computation a prefill already runs; what this adds is the alignment, which
is where a harness adapter goes wrong: slot `i` of the logits predicts token
`i + 1` of the row, so a continuation of `n` tokens is read at the `n` slots
ending one before the row's last token.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from dew.inference.tasks import TextGeneration, _ceiling
from dew.sampling.text import Sampling

DEFAULT_CONTEXT = 2048
"""The scoring window for a model that declares no `max_seq_len`."""


@functools.partial(jax.jit, static_argnums=(0,))
def _scored(model, variables, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the per-target log-probability and whether the target was the argmax.

    `rows` is `[B, T]` of token ids; both results are `[B, T - 1]`, entry `i`
    belonging to `rows[:, i + 1]`. Rows are right-padded by the caller, which
    a causal model cannot read backwards, so a short row's own slots hold
    what they would hold alone.
    """
    logits = model.apply(variables, rows[:, :-1]).astype(jnp.float32)
    targets = rows[:, 1:, None]
    picked = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets, axis=-1)
    return picked[..., 0], jnp.argmax(logits, axis=-1) == rows[:, 1:]


def _batches(count: int, size: int) -> list[range]:
    """Yield `count` indices in runs of at most `size`, in order."""
    if type(size) is not int or size < 1:
        raise ValueError(f"batch_size is a positive integer, got {size!r}")
    return [range(start, min(start + size, count)) for start in range(0, count, size)]


def _padded(rows: Sequence[Sequence[int]]) -> np.ndarray:
    """Return one batch's token rows, right-padded to the longest with zeros."""
    width = max(len(row) for row in rows)
    return np.asarray([[*row, *([0] * (width - len(row)))] for row in rows], np.int32)


def _windows(tokens: Sequence[int], width: int) -> list[list[int]]:
    """`tokens` cut into consecutive rows of at most `width` ids.

    A row of one id scores nothing, because its only token has no context,
    so a tail of one id joins the row before it instead of standing alone.
    """
    if width < 2:
        raise ValueError(f"a scoring window holds at least two ids, and this model declares {width}")
    rows = [list(tokens[start:start + width]) for start in range(0, len(tokens), width)]
    if len(rows) > 1 and len(rows[-1]) < 2:
        rows[-2] = rows[-2] + rows.pop()
    return [row for row in rows if len(row) > 1]


@register_model("dew")
class DewLM(LM):
    """Puts a `TextGeneration` behind lm-eval-harness's `LM` interface.

    `task` is the run's own generation task, with its model, its weights and
    its processor; `batch_size` is how many rows one scoring call runs at
    once. Requests keep their order, which is what the harness pairs its
    documents back up by.

    Likelihoods are exact: the model's log-softmax at the continuation's own
    targets, summed. Generation is greedy unless a request's `gen_kwargs`
    ask for a temperature, which is the harness's own default and what a
    suite's reported numbers assume.
    """

    def __init__(self, task: TextGeneration, *, batch_size: int = 1) -> None:
        super().__init__()
        if not isinstance(task, TextGeneration):
            raise TypeError(
                f"the harness scores next-token likelihoods, which is what a "
                f"TextGeneration holds; {type(task).__name__} is a different task")
        if task.processor is None:
            raise ValueError(
                "the harness hands over text, so the task needs the processor that "
                "turns it into tokens; load the run through dew.pipeline")
        _batches(0, batch_size)
        self.task = task
        self.batch_size = batch_size

    @classmethod
    def from_run(cls, run: str, *, batch_size: int = 1, ema: bool = True,
                 step: int | None = None, dtype: str | None = None) -> DewLM:
        """Load the run in `run` as a harness model, the way `dew.pipeline` builds it."""
        return cls(TextGeneration.from_run(run, ema=ema, step=step, dtype=dtype),
                   batch_size=batch_size)

    @classmethod
    def create_from_arg_string(cls, arg_string: str,
                               additional_config: dict | None = None) -> DewLM:
        """Build the model from `--model_args run=<directory>,batch_size=4`."""
        return cls._from_arguments(utils.simple_parse_args_string(arg_string),
                                   additional_config)

    @classmethod
    def create_from_arg_obj(cls, arg_dict: dict,
                            additional_config: dict | None = None) -> DewLM:
        """Build the model from arguments already parsed, the route the CLI takes."""
        return cls._from_arguments(dict(arg_dict), additional_config)

    @classmethod
    def _from_arguments(cls, arguments: dict, additional: dict | None) -> DewLM:
        """Load the run one `--model_args` record names.

        The harness passes `batch_size` in both halves when the command line
        carries it in both, so the explicit configuration wins and the pair
        is read once rather than reaching `__init__` twice.
        """
        arguments.update({name: value for name, value in (additional or {}).items()
                          if value is not None})
        # The harness hands every model the same placement argument. A run's
        # weights are placed on a JAX mesh when the task is built, so there
        # is no device for a caller to pick here and the name is dropped
        # rather than refused; anything else it passes reaches `from_run`,
        # which names what it does not take.
        arguments.pop("device", None)
        run = arguments.pop("run", None)
        if run is None:
            raise ValueError(
                "--model dew evaluates a run directory; name it with "
                "--model_args run=<directory>")
        batch = arguments.pop("batch_size", 1)
        return cls.from_run(str(run), batch_size=int(batch), **dict(arguments))

    @property
    def eot_token_id(self) -> int:
        """Return the id a row with no context is conditioned on: the policy's EOS.

        `Sampling` normalises its own field to a tuple, and declares the
        form a caller may write, so both spellings are read here.
        """
        stops = self.task.sampling.eos_id
        if stops is None:
            return 0
        return stops if isinstance(stops, int) else (stops[0] if stops else 0)

    @property
    def max_length(self) -> int:
        """Return how many ids one scoring row may hold, as the model declares it.

        `_ceiling` is the same read a call already makes to size its cache,
        so a harness row and a generated row are bounded by the same field.
        """
        declared = _ceiling(self.task.model)
        return DEFAULT_CONTEXT if declared is None else declared

    def tok_encode(self, text: str) -> list[int]:
        """Encode `text` with the run's own tokenizer, one row of ids."""
        return [int(token) for token in np.asarray(self._processor([text]).tokens)[0]]

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self._processor.decode(np.asarray([list(tokens)], np.int32))[0]

    @property
    def _processor(self):
        processor = self.task.processor
        if processor is None:
            raise ValueError("this task lost its processor; a harness model needs one")
        return processor

    def _rows(self, rows: Sequence[Sequence[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return every row's per-target log-probabilities and argmax agreement."""
        scored: list[tuple[np.ndarray, np.ndarray]] = []
        for batch in _batches(len(rows), self.batch_size):
            tokens = _padded([rows[index] for index in batch])
            probabilities, greedy = _scored(self.task.model, self.task.variables,
                                            jnp.asarray(tokens))
            values, matched = np.asarray(probabilities), np.asarray(greedy)
            for offset, index in enumerate(batch):
                width = len(rows[index]) - 1
                scored.append((values[offset, :width], matched[offset, :width]))
        return scored

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Return each `(context, continuation)`'s summed log-probability, and whether
        greedy decoding of the context would have produced it.

        A request with no context is conditioned on `eot_token_id`, which is
        what the harness's own models condition such a request on.
        """
        rows: list[list[int]] = []
        widths: list[int] = []
        for request in requests:
            arguments = list(request.args)
            context, continuation = str(arguments[0]), str(arguments[1])
            prefix = self.tok_encode(context) if context else [self.eot_token_id]
            whole = (self.tok_encode(context + continuation) if context
                     else [*prefix, *self.tok_encode(continuation)])
            if len(whole) <= len(prefix):
                raise ValueError(
                    f"{continuation!r} adds no token to its context, so there is nothing "
                    f"to score; the tokenizer read the pair as {len(whole)} ids")
            rows.append(whole[-self.max_length:])
            widths.append(min(len(whole) - len(prefix), self.max_length - 1))
        return [(float(values[-width:].sum()), bool(matched[-width:].all()))
                for (values, matched), width in zip(self._rows(rows), widths, strict=True)]

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Return each string's own log-probability, every token after the first scored.

        A string longer than the model's context is scored in consecutive
        windows, each conditioned on what it holds, which is the harness's
        non-overlapping rolling window.
        """
        windows: list[list[int]] = []
        counts: list[int] = []
        for request in requests:
            text = str(next(iter(request.args)))
            parts = _windows(self.tok_encode(text), self.max_length)
            windows.extend(parts)
            counts.append(len(parts))
        scored = self._rows(windows)
        answers, start = [], 0
        for count in counts:
            answers.append(sum(float(values.sum()) for values, _ in scored[start:start + count]))
            start += count
        return answers

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Continue each context until one of its stop strings or its budget.

        The stop strings cut the decoded text, so a sequence that spans two
        tokens ends the answer the way the harness expects it to.
        """
        answers = []
        for request in requests:
            arguments = list(request.args)
            context, controls = str(arguments[0]), dict(arguments[1])
            until = controls.get("until") or []
            stops = [until] if isinstance(until, str) else [str(entry) for entry in until]
            sampling = Sampling(temperature=float(controls.get("temperature", 0.0)),
                                eos_id=self.task.sampling.eos_id)
            drawn = self.task(context, int(controls.get("max_gen_toks", 256)),
                              seed=int(controls.get("seed", 0)), sampling=sampling)
            text = self.task.decode(drawn)[0]
            cuts = [text.index(stop) for stop in stops if stop and stop in text]
            answers.append(text[:min(cuts)] if cuts else text)
        return answers
