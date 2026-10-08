"""Run a saved run as an lm-evaluation-harness model.

`DewLM` wraps a `TextGeneration` in lm-eval-harness's `TemplateLM`
interface, so any task suite can run against a run directory.

`lm_eval` is an optional extra (`pip install dewml[eval-harness]`). Only this
module imports it, and `dew.eval` does not import this module. Importing this
module registers the adapter under the name `dew`. lm_eval 0.4 has no plugin
discovery, so the import has to happen in the process that runs the command.
`python -m dew.eval` makes that import and then runs `lm_eval`'s own command
line:

    python -m dew.eval --model dew --model_args run=runs/shakespeare \\
        --tasks hellaswag --limit 4

lm-eval's own code decides which tokens are scored (`TemplateLM.loglikelihood`,
`get_rolling_token_windows`). This module adds `_loglikelihood_tokens`, which
builds the row `HFLM` builds from each `(context, continuation)` pair and
scores it with the model's forward under `jax.jit`. Slot `i` of the logits
predicts token `i + 1`, so a continuation of `n` tokens is read at the `n`
slots that end one before the row's last token.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text

from dew.inference.tasks import TextGeneration, cache_ceiling
from dew.nn.inputs import pad_token_rows
from dew.objectives.base import token_log_probs
from dew.sampling.text import Sampling

DEFAULT_CONTEXT = 2048
"""The scoring window for a model that declares no `max_seq_len`."""

DEFAULT_BUDGET = 256
"""The generation budget of a request that names none, `HFLM.max_gen_toks`."""

TokenRequest = tuple[tuple[str, str] | None, list[int], list[int]]
"""One `_loglikelihood_tokens` request: the strings, context ids, continuation ids."""


@functools.partial(jax.jit, static_argnums=(0,))
def _scored(model, variables, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the per-target log-probability and whether the target was the argmax.

    `rows` is `[B, T]` of token ids; both results are `[B, T - 1]`, entry `i`
    belonging to `rows[:, i + 1]`. Rows are right-padded by the caller, which
    a causal model cannot read backwards, so a short row's own slots hold
    what they would hold alone.
    """
    logits = model.apply(variables, rows[:, :-1])
    return token_log_probs(logits, rows[:, 1:]), jnp.argmax(logits, axis=-1) == rows[:, 1:]


def _batches(count: int, size: int) -> list[range]:
    """Yield `count` indices in runs of at most `size`, in order."""
    return [range(start, min(start + size, count)) for start in range(0, count, size)]


@register_model("dew")
class DewLM(TemplateLM):
    """Runs a `TextGeneration` as an lm-eval-harness model, through the `TemplateLM` interface.

    `task` is the run's own generation task, with its model, its weights and
    its processor. A task without a processor raises ValueError, because the
    harness passes text. `batch_size` is how many rows one scoring call runs
    at once. Answers come back in request order, because the harness matches
    them to its documents by order.

    Likelihoods are exact: each is the sum of the model's log-softmax at the
    continuation's own targets. Generation is greedy unless a request's
    `gen_kwargs` ask for sampling with a temperature, read the way `HFLM`
    reads them. Greedy is the harness's own default, and a suite's reported
    numbers assume it.
    """

    def __init__(self, task: TextGeneration, *, batch_size: int = 1) -> None:
        super().__init__()
        processor = task.processor
        if processor is None:
            raise ValueError(
                "the harness hands over text, so the task needs the processor that "
                "turns it into tokens; load the run through dew.pipeline")
        # A negative size would score nothing and return no answers.
        if batch_size < 1:
            raise ValueError(f"batch_size is a positive integer, got {batch_size!r}")
        self.task = task
        self._processor = processor
        self.batch_size = batch_size

    @classmethod
    def from_run(cls, run: str, *, batch_size: int = 1, ema: bool | None = None,
                 step: int | str | None = None, dtype: str | None = None) -> DewLM:
        """Load the run directory `run` as a harness model, the way `dew.pipeline` builds it.

        With `ema=None`, it uses the run's averaged weights when the run kept
        them, and its live weights otherwise.
        """
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
        """Build the model from arguments that are already parsed, as the CLI does."""
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
        # The harness hands every model a device; a run's weights are placed
        # on a JAX mesh when the task is built, so it is dropped, not refused.
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
        """The id a row with no context is conditioned on: the sampling policy's first EOS, or 0."""
        stops = self.task.sampling.stops
        return stops[0] if stops else 0

    @property
    def prefix_token_id(self) -> int:
        """The id a first token is conditioned on: the tokenizer's BOS, or `eot_token_id` without one.

        This matches `HFLM.prefix_token_id`. A vocabulary that starts every
        sequence with BOS scores its first token after BOS, and conditioning
        it on EOS instead would change every rolling and empty-context score.
        """
        bos = self._processor.bos_id
        return self.eot_token_id if bos is None else bos

    @property
    def max_length(self) -> int:
        """The number of ids one scoring row may hold, as the model declares it (2048 if it does not).

        Generation sizes its cache from the same declared field, so a
        harness row and a generated row have the same bound.
        """
        # cache_ceiling is the read a generation call makes to size its cache.
        declared = cache_ceiling(self.task.model)
        return DEFAULT_CONTEXT if declared is None else declared

    def tok_encode(self, string: str, add_special_tokens: bool | None = None,
                   **kwargs: int | None) -> list[int]:
        """Encode `string` with the run's own tokenizer into one row of ids.

        The run's processor adds special tokens the way it did in training,
        so `add_special_tokens` and the harness's other integer options
        (`left_truncate_len`) are accepted but ignored.
        """
        del add_special_tokens, kwargs
        return [int(token) for token in np.asarray(self._processor([string]).tokens)[0]]

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self._processor.decode(np.asarray([list(tokens)], np.int32))[0]

    def _loglikelihood_tokens(self, requests: Sequence[TokenRequest],
                              disable_tqdm: bool = False,
                              **kwargs: int | None) -> list[tuple[float, bool]]:
        """Return each pair's summed continuation log-probability, and whether
        greedy decoding of the context would have produced the continuation.

        Each pair is `HFLM`'s row: context and continuation joined, the last
        `max_length + 1` ids kept, so the forward reads `max_length` ids and
        the continuation's targets are the row's last ones. As `HFLM` does,
        a continuation longer than `max_length` is refused, since no row
        could hold it with any of its context, and a continuation with no
        ids of its own is refused: scored, it would be probability 1 and
        greedy, and win every multiple-choice comparison it is in.
        """
        del disable_tqdm, kwargs
        rows: list[list[int]] = []
        for strings, context, continuation in requests:
            named = repr(strings[1]) if strings is not None else "a continuation"
            if not continuation:
                raise ValueError(
                    f"{named} adds no token to its context, so there is nothing to score; "
                    f"the tokenizer read the pair as {len(context)} ids")
            if len(continuation) > self.max_length:
                raise ValueError(
                    f"{named} is {len(continuation)} ids, longer than this model's "
                    f"{self.max_length}-id window; lm-eval's HFLM refuses it too")
            rows.append([*context, *continuation][-(self.max_length + 1):])
        answers: list[tuple[float, bool]] = []
        for batch in _batches(len(rows), self.batch_size):
            tokens, _ = pad_token_rows([rows[index] for index in batch], padding_side="right")
            probabilities, argmax = _scored(self.task.model, self.task.variables,
                                            jnp.asarray(tokens))
            values, matched = np.asarray(probabilities), np.asarray(argmax)
            for offset, index in enumerate(batch):
                end = len(rows[index]) - 1
                start = end - len(requests[index][2])
                answers.append((float(values[offset, start:end].sum()),
                                bool(matched[offset, start:end].all())))
        return answers

    def loglikelihood_rolling(self, requests: list[Instance],
                              disable_tqdm: bool = False) -> list[float]:
        """Return each string's log-probability, with every token scored once.

        lm-eval's helpers cut the string into windows. The first window is
        conditioned on the prefix token and each later one on the
        `max_length` ids before it, and no two windows score the same token.
        """
        windows: list[TokenRequest] = []
        counts: list[int] = []
        for request in requests:
            text = str(next(iter(request.args)))
            parts = [utils.make_disjoint_window(pair) for pair in utils.get_rolling_token_windows(
                token_list=self.tok_encode(text), prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)]
            windows.extend((None, context, scored) for context, scored in parts)
            counts.append(len(parts))
        scored = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        answers, start = [], 0
        for count in counts:
            answers.append(sum(value for value, _ in scored[start:start + count]))
            start += count
        return answers

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        """Continue each context until one of its stop strings or its token budget, and return the texts.

        Each request is read the way `HFLM.generate_until` reads it, through
        lm-eval's own helpers. `normalize_gen_kwargs` accepts the budget under
        any of its names and makes `do_sample=False` greedy whatever the
        temperature. The EOS token's text is added to the stop strings, and a
        context longer than the window minus the budget keeps its last ids.
        The stop strings cut the decoded text, so a stop sequence that spans
        two tokens ends the answer where the harness expects. A budget that
        fills the whole window raises ValueError, as it does in `HFLM`.
        """
        del disable_tqdm
        eos = None if self.task.sampling.eos_id in (None, ()) else self.tok_decode([self.eot_token_id])
        answers = []
        for request in requests:
            arguments = list(request.args)
            context, raw = str(arguments[0]), dict(arguments[1])
            # normalize_gen_kwargs sets the budget and the stop strings; the
            # type leaves every key optional.
            controls = normalize_gen_kwargs(raw, DEFAULT_BUDGET)
            budget = controls.get("max_gen_toks", DEFAULT_BUDGET)
            if budget >= self.max_length:
                raise ValueError(
                    f"a budget of {budget} ids leaves no room for a context in this model's "
                    f"{self.max_length}-id window; lm-eval's HFLM refuses it too")
            ids = self.tok_encode(context)[-(self.max_length - budget):]
            sampling = Sampling(temperature=float(controls.get("temperature", 0.0)),
                                eos_id=self.task.sampling.eos_id)
            drawn = self.task([ids], budget, key=int(raw.get("seed", 0)), sampling=sampling)
            stops = handle_stop_sequences(controls.get("until"), eos=eos)
            answers.append(postprocess_generated_text(self.task.decode(drawn)[0], stops, None))
        return answers


__all__ = ["DewLM"]
