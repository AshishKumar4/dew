"""Run a saved run as an lm-evaluation-harness model.

`DewLM` puts a `TextGeneration` behind lm-eval-harness's `TemplateLM`, so
any task suite runs against a run directory. The trainer's own perplexity
says how well a run predicts its training data and nothing about what it
can do, which is the other question a suite answers.

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

Everything that decides which tokens are scored is lm-eval's own code:
`TemplateLM.loglikelihood` splits each pair (moving a context's trailing
whitespace into the continuation, conditioning an empty context on the
prefix token), and `get_rolling_token_windows` with `make_disjoint_window`
cuts a long string so every token is scored exactly once. What this module
adds is `_loglikelihood_tokens`, the row `HFLM` builds from each
`(context, continuation)` pair, scored by the model's own forward under
`jax.jit`: slot `i` of the logits predicts token `i + 1` of the row, so a
continuation of `n` tokens is read at the `n` slots ending one before the
row's last token.
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

from dew.inference.tasks import TextGeneration, _ceiling
from dew.objectives.likelihood import token_log_probs
from dew.sampling.text import Sampling

DEFAULT_CONTEXT = 2048
"""The scoring window for a model that declares no `max_seq_len`."""

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
    if type(size) is not int or size < 1:
        raise ValueError(f"batch_size is a positive integer, got {size!r}")
    return [range(start, min(start + size, count)) for start in range(0, count, size)]


def _padded(rows: Sequence[Sequence[int]]) -> np.ndarray:
    """Return one batch's token rows, right-padded to the longest with zeros."""
    width = max(len(row) for row in rows)
    return np.asarray([[*row, *([0] * (width - len(row)))] for row in rows], np.int32)


@register_model("dew")
class DewLM(TemplateLM):
    """Puts a `TextGeneration` behind lm-eval-harness's `TemplateLM` interface.

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
    def prefix_token_id(self) -> int:
        """Return the id a first token is conditioned on: the tokenizer's BOS, else EOS.

        This is `HFLM.prefix_token_id`. A vocabulary that starts every
        sequence with BOS scores its first token after BOS, and conditioning
        it on EOS instead would move every rolling and empty-context score.
        """
        bos = self._processor.bos_id
        return self.eot_token_id if bos is None else bos

    @property
    def max_length(self) -> int:
        """Return how many ids one scoring row may hold, as the model declares it.

        `_ceiling` is the same read a call already makes to size its cache,
        so a harness row and a generated row are bounded by the same field.
        """
        declared = _ceiling(self.task.model)
        return DEFAULT_CONTEXT if declared is None else declared

    def tok_encode(self, string: str, add_special_tokens: bool | None = None,
                   **kwargs: int | None) -> list[int]:
        """Encode `string` with the run's own tokenizer, one row of ids.

        The run's processor decides special tokens the way it did in
        training, so `add_special_tokens` and the harness's other integer
        options (`left_truncate_len`) are accepted and not read.
        """
        del add_special_tokens, kwargs
        return [int(token) for token in np.asarray(self._processor([string]).tokens)[0]]

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self._processor.decode(np.asarray([list(tokens)], np.int32))[0]

    @property
    def _processor(self):
        processor = self.task.processor
        if processor is None:
            raise ValueError("this task lost its processor; a harness model needs one")
        return processor

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
            tokens = _padded([rows[index] for index in batch])
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
        """Return each string's own log-probability, every token scored once.

        The windows are lm-eval's: the first conditioned on the prefix
        token, each later one on the `max_length` ids before it, none
        overlapping in what it scores.
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
